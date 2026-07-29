"""Sliding window detector for clip-level classifiers.

Wraps any clip-level classifier in a sliding window to produce frame-level
predictions, satisfying the Detector protocol for the evaluation pipeline.

Factory functions (create_perch2_detector, create_audioprotopnet_detector, etc.)
live here alongside the class, following the same pattern as frame_detector.py.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import torch
import yaml
from alp_data.discover import GBIFConverter
from pydantic import BaseModel

from esp_research.adapters import HttpClient
from esp_research.logging import logger
from esp_research.protocols.classifier import MultiLabelClassifierOutput
from esp_research.protocols.detector import DetectorConfig, DetectorOutput
from sound_event_detection.utils.io_utils import open_anypath


def _classifier_identity(server_config: dict | None) -> dict:
    """Extract a compact identity of a wrapped classifier from its ``GET /`` payload.

    Pulls the identity-bearing keys out of the backend classifier server's
    ``GET /`` metadata (captured once at construction), avoiding the bulky
    `labels` list. `weights_sha256` and `git_commit` are ``None`` when the
    backend server does not expose them (e.g. a stock classifier server that
    only reports ``labels`` / ``sample_rate``); this is the "fill in nan"
    fallback expressed JSON-safely, since a non-finite float cannot round-trip
    through the server's ``GET /`` response.

    Parameters
    ----------
    server_config : dict or None
        The wrapped classifier's ``GET /`` payload, or ``None`` if unavailable.

    Returns
    -------
    dict
        ``{type, weights_sha256, git_commit, sample_rate, n_labels}`` — each
        value ``None`` when the source payload does not carry it.
    """
    source = server_config or {}
    return {
        "type": source.get("type"),
        "weights_sha256": source.get("weights_sha256"),
        "git_commit": source.get("git_commit"),
        "sample_rate": source.get("sample_rate"),
        "n_labels": len(source["labels"]) if isinstance(source.get("labels"), list) else None,
    }


class SlidingWindowDetector:
    """Detector that slides a clip-level classifier over audio.

    Produces one prediction per hop, averages overlapping windows, and
    optionally remaps labels (e.g. eBird codes → GBIF scientific names).

    Satisfies the Detector protocol (labels, sample_rate, frame_rate,
    run). Also exposes a `server_config` identity dict (detector type, sliding
    window geometry, and the wrapped classifier's captured identity) that the
    serving layer surfaces via ``GET /``.
    """

    config_class = DetectorConfig

    def __init__(
        self,
        classify_fn: Callable[[torch.Tensor], torch.Tensor],
        classifier_labels: list[str],
        sample_rate: int,
        window_size: float,
        hop_size: float,
        labels: None | list[str],
        activation: str = "sigmoid",
        analysis_window: float | None = None,
        detector_type: str | None = None,
        classifier_server_config: dict | None = None,
    ) -> None:
        """Initialize SlidingWindowDetector.

        Parameters
        ----------
        classify_fn : Callable[[torch.Tensor], torch.Tensor]
            Function mapping audio tensor [B, samples] to logits [B, K].
        classifier_labels : list[str]
            Label names for each of the K logit outputs.
        labels : None | list[str]
            Final output label names (after remapping). If no remapper,
            this should equal classifier_labels or a subset.
        sample_rate : int
            Expected audio sample rate in Hz.
        window_size : float
            Classifier window duration in seconds (clips are padded to this).
        hop_size : float
            Hop duration in seconds. Must be <= analysis_window (or window_size
            if analysis_window is not set).
        activation : str
            Activation to apply to logits. "softmax" or "sigmoid".
        analysis_window : float | None
            Duration of audio to extract per hop in seconds. If smaller than
            window_size, each clip is left-padded with zeros to window_size
            before classification. Defaults to window_size (no padding).
        detector_type : str | None
            Detector-type tag recorded under ``server_config["type"]`` (e.g.
            ``"perch2"``). Set by the factory that built this detector; ``None``
            when unknown.
        classifier_server_config : dict | None
            The wrapped classifier server's ``GET /`` payload, captured at
            construction. Its identity-bearing keys are recorded under
            ``server_config["classifier"]`` (see `_classifier_identity`);
            ``None`` when unavailable.

        Raises
        ------
        ValueError
            If hop_size > effective analysis window, analysis_window > window_size,
            or if activation is not "softmax" or "sigmoid".
        """
        if analysis_window is None:
            analysis_window = window_size
        if analysis_window > window_size:
            raise ValueError(f"analysis_window ({analysis_window}) must be <= window_size ({window_size})")
        if hop_size > analysis_window:
            raise ValueError(f"hop_size ({hop_size}) must be <= analysis_window ({analysis_window})")
        if activation not in ("softmax", "sigmoid"):
            raise ValueError(f"activation must be 'softmax' or 'sigmoid', got '{activation}'")

        self._classify_fn = classify_fn
        self._classifier_labels = classifier_labels
        self._activation = activation

        self.labels = classifier_labels if labels is None else labels
        self.sample_rate = sample_rate
        self.frame_rate = 1.0 / hop_size
        self.window_size = window_size
        self.hop_size = hop_size
        self.analysis_window = analysis_window

        # Build a mapping from each output label to the classifier column(s)
        # that produce it. Multiple classifier columns can map to one output
        # label when the classifier has synonyms (e.g. taxonomic renames that
        # resolve to the same GBIF name). Labels not found in the classifier
        # get an empty list and will default to zero probability.
        classifier_label_to_indices: dict[str, list[int]] = {}
        for idx, label in enumerate(self._classifier_labels):
            classifier_label_to_indices.setdefault(label, []).append(idx)
        self._label_groups = [classifier_label_to_indices.get(label, []) for label in self.labels]

        self.server_config: dict = {
            "type": detector_type,
            "window_size": window_size,
            "hop_size": hop_size,
            "analysis_window": analysis_window,
            "classifier": _classifier_identity(classifier_server_config),
        }

    @property
    def window_duration(self) -> float:
        """Classifier input window duration in seconds.

        Returns
        -------
        float
            The classifier window size in seconds, which is the clip length the
            detector pads/crops to. Reported by the serving layer's ``GET /`` so
            clients (and the clip evaluation) know the input window length.
        """
        return self.window_size

    def run(
        self,
        audio: np.ndarray,
        batch_size: int = 32,
        overlap: float | None = None,
    ) -> DetectorOutput:
        """Run sliding window inference on a batch of audio files.

        All recordings in the batch must share the same number of samples.

        Parameters
        ----------
        audio : np.ndarray
            Batched waveform of shape ``(batch, n_samples)`` at self.sample_rate.
        batch_size : int
            Number of windows to classify at once.
        overlap : float | None
            Accepted for interface parity with frame detectors but ignored: a
            sliding-window detector's window overlap is fixed by `hop_size` and
            `window_size` at construction. A non-``None`` value is logged and
            discarded so a shared eval config carrying ``inference.overlap`` (for
            frame detectors) can also drive sliding-window baselines.

        Returns
        -------
        DetectorOutput
            Frame-level probabilities at self.frame_rate Hz, with `predictions` of
            shape ``(batch, n_frames, classes)``.

        Raises
        ------
        ValueError
            If audio is not 2-D.
        """
        if audio.ndim != 2:
            raise ValueError(f"Expected 2D audio array [batch, samples], got shape {audio.shape}")
        if overlap is not None:
            logger.debug(
                "SlidingWindowDetector ignores overlap=%s; window overlap is fixed by hop_size/window_size.",
                overlap,
            )

        batch = audio.shape[0]
        analysis_samples = int(self.analysis_window * self.sample_rate)
        classifier_samples = int(self.window_size * self.sample_rate)
        hop_samples = int(self.hop_size * self.sample_rate)
        pad_samples = classifier_samples - analysis_samples  # left-pad amount

        # Compute number of windows based on analysis_window
        audio_len = audio.shape[1]
        if audio_len <= analysis_samples:
            n_windows = 1
        else:
            n_windows = 1 + int(np.ceil((audio_len - analysis_samples) / hop_samples))

        # Pad audio so last analysis window fits
        required_len = (n_windows - 1) * hop_samples + analysis_samples
        if required_len > audio_len:
            audio = np.pad(audio, ((0, 0), (0, required_len - audio_len)), mode="constant")

        # Extract analysis-sized windows: (batch, n_windows, analysis_samples)
        windows = np.lib.stride_tricks.sliding_window_view(audio, analysis_samples, axis=1)[:, ::hop_samples]
        windows = windows[:, :n_windows]

        # Flatten the batch and window axes so a single forward pass spans recordings.
        flat_windows = windows.reshape(batch * n_windows, analysis_samples)

        # Classify in batches, collecting raw logits
        all_logits = []
        for i in range(0, flat_windows.shape[0], batch_size):
            batch_windows = np.array(flat_windows[i : i + batch_size])  # copy: sliding_window_view is read-only
            # Left-pad to classifier window size if analysis_window < window_size
            if pad_samples > 0:
                batch_windows = np.pad(batch_windows, ((0, 0), (pad_samples, 0)), mode="constant")
            forward_batch = torch.from_numpy(batch_windows).float()
            logits = self._classify_fn(forward_batch)
            all_logits.append(logits.cpu().numpy())

        all_logits = np.concatenate(all_logits, axis=0).reshape(batch, n_windows, -1)  # (batch, n_windows, K)

        # Average overlapping windows in logit space, then activate once.
        # For softmax this is mathematically correct (vs averaging post-softmax probabilities).
        averaged_logits = self._average_overlapping(all_logits)  # (batch, n_frames, K)
        averaged = self._apply_activation(torch.from_numpy(averaged_logits))  # (batch, n_frames, K)

        # Remap classifier columns to output labels. Each output label maps
        # to zero or more classifier columns (see self._label_groups):
        #   - 0 columns: label missing from classifier → zeros (no prediction)
        #   - 1 column:  direct copy
        #   - N columns: take max (e.g. taxonomic synonyms → pick strongest)
        n_frames = averaged.shape[1]
        remapped = np.zeros((batch, n_frames, len(self.labels)), dtype=averaged.dtype)
        for tgt_idx, srcs in enumerate(self._label_groups):
            if len(srcs) == 1:
                remapped[:, :, tgt_idx] = averaged[:, :, srcs[0]]
            elif len(srcs) > 1:
                remapped[:, :, tgt_idx] = np.max(averaged[:, :, srcs], axis=2)

        return DetectorOutput(predictions=remapped, frame_rate=self.frame_rate, class_names=self.labels)

    def run_as_classifier(
        self,
        audio: np.ndarray,
        batch_size: int = 32,
        overlap: float | None = None,
    ) -> MultiLabelClassifierOutput:
        """Run sliding window inference and pool to clip-level scores.

        Runs frame-level inference via `run` and pools the frame predictions to
        clip level by taking the maximum probability over the time dimension.

        Parameters
        ----------
        audio : np.ndarray
            Batched waveform of shape ``(batch, n_samples)`` at self.sample_rate.
        batch_size : int
            Number of windows to classify at once.
        overlap : float | None
            Accepted for interface parity with frame detectors but ignored
            (see `run`).

        Returns
        -------
        MultiLabelClassifierOutput
            Clip-level predictions with `predictions` of shape ``(batch, classes)``.
        """
        output = self.run(audio, batch_size=batch_size, overlap=overlap)
        clip_probs = output.predictions.max(axis=1)  # (batch, classes)
        return MultiLabelClassifierOutput(predictions=clip_probs, class_names=self.labels)

    def _apply_activation(self, logits: torch.Tensor) -> np.ndarray:
        """Apply activation function to logits and return numpy array.

        Parameters
        ----------
        logits : torch.Tensor
            Raw logits, shape [B, K].

        Returns
        -------
        np.ndarray
            Probabilities, shape [B, K].
        """
        with torch.no_grad():
            if self._activation == "softmax":
                probs = torch.softmax(logits, dim=-1)
            else:
                probs = torch.sigmoid(logits)
        return probs.cpu().numpy()

    def _average_overlapping(self, logits: np.ndarray) -> np.ndarray:
        """Average overlapping window logits across a batch of recordings.

        Each window covers ``frames_per_window`` output frames starting at its
        window index. Overlapping contributions are averaged. The frame coverage
        is identical across the batch axis.

        Parameters
        ----------
        logits : np.ndarray
            Per-window logits, shape (batch, n_windows, K).

        Returns
        -------
        np.ndarray
            Averaged logits, shape (batch, n_frames, K).
        """
        frames_per_window = int(self.analysis_window / self.hop_size)

        if frames_per_window <= 1:
            # No overlap — each window maps to exactly one frame
            return logits

        batch, n_windows, n_classes = logits.shape
        n_frames = n_windows  # one output frame per hop position
        frame_sums = np.zeros((batch, n_frames, n_classes), dtype=np.float64)
        frame_counts = np.zeros(n_frames, dtype=np.int32)

        for window_idx in range(n_windows):
            end_frame = min(window_idx + frames_per_window, n_frames)
            frame_sums[:, window_idx:end_frame] += logits[:, window_idx][:, np.newaxis, :]
            frame_counts[window_idx:end_frame] += 1

        return (frame_sums / frame_counts[np.newaxis, :, np.newaxis]).astype(np.float32)

    @classmethod
    def from_checkpoint_dir(cls, checkpoint_dir: Path, config: BaseModel | Path | str) -> Self:
        """Not implemented — SlidingWindowDetector loads pretrained weights via factory functions.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "SlidingWindowDetector does not support from_checkpoint_dir. "
            "Use the factory functions (e.g. create_perch2_detector) instead."
        )

    @classmethod
    def from_config(cls, config: DetectorConfig | Path | str) -> Self:
        """Build a SlidingWindowDetector from a config alone, without a checkpoint.

        Resolves the config to a dict and dispatches to
        `create_sliding_window_detector_from_config`, which selects the
        appropriate HTTP-server-backed factory based on ``config["type"]``.
        Labels are taken from the config's ``labels`` field when present
        (``None`` falls back to the classifier's own labels).

        Parameters
        ----------
        config : DetectorConfig | Path | str
            Either a `DetectorConfig` (or subclass) whose ``model_dump()``
            yields the model config dict, or a path to a config file.

        Returns
        -------
        SlidingWindowDetector
            Ready for ``run()``.
        """
        if isinstance(config, DetectorConfig):
            cfg: dict = config.model_dump()
        else:
            with open_anypath(str(config)) as f:
                cfg = yaml.safe_load(f)

        return create_sliding_window_detector_from_config(cfg, labels=cfg.get("labels"))


# ============= Factory Functions =============

_SAMPLE_RATE = 32000
_WINDOW_SIZE = 5.0

EBIRD_TO_SPECIES_MANUAL = {
    "dusscr1": "Megapodius freycinet",
    "chnfra2": "Pternistis castaneicollis",
    "monnig1": "Caprimulgus poliocephalus",
    "rubrat1": "Ocreatus addae",
    "augbuz2": "Buteo augur",
    "madsco1": "Otus rutilus",
    "fraeao1": "Ketupa poensis",
    "varkin1": "Ceyx lepidus",
    "capbat10": "Batis capensis",
    "whbshb1": "Pteruthius aeralatus",
    "malbrw1": "Nesillas typica",
    "sphlar12": "Chersomanes albofasciata",
    "klblar6": "Certhilauda subcoronata",
    "faclar8": "Calendulauda africanoides",
    "dunlar5": "Calendulauda erythrochlamys",
    "obfrob1": "Stiphrornis erythrothorax",
    "shtaka2": "Sheppardia poensis",
    "bocaka11": "Sheppardia bocagei",
    "sinbus6": "Mirafra javanica",
    "scbcup3": "Pnoepyga albiventer",
    "radacc2": "Prunella ocularis",
    "origre6": "Chloris sinica",
}

SCI_NAME_CORRECTION_MANUAL = {
    "Eupodotis rueppelii": "Eupodotis rueppellii",
    "Eudynamys melanorhynchus": "Eudynamys melanorhyncha",
    "Chrysococcyx meyerii": "Chrysococcyx meyeri",
    "Aramides cajaneus": "Aramides cajanea",
    "Laterallus spilonota": "Laterallus spilonotus",
    "Amaurornis moluccana": "Amaurornis olivacea",
    "Ictinaetus malaiensis": "Ictinaetus malayensis",
    "Indicator conirostris/minor": "Indicator conirostris",
    "Amazona mercenarius": "Amazona mercenaria",
    "Orthopsittaca manilatus": "Orthopsittaca manilata",
    "Neophema bourkii": "Neopsephotus bourkii",
    "Vini solitarius": "Phigys solitarius",
    "Glossoptila goldiei": "Glossoptilus goldiei",
    "Saudareos ornatus": "Saudareos ornata",
    "Serpophaga subcristata/munda": "Serpophaga subcristata",
    "Conopias parvus": "Conopias albovittatus",
    "Hylacola pyrrhopygia": "Hylacola pyrrhopygius",
    "Hylacola cauta": "Hylacola cautus",
    "Tchagra minutus": "Tchagra minuta",
    "Artamus leucorynchus": "Artamus leucorhynchos",
    "Dicrurus atactus": "Dicrurus modestus",
    "Dicrurus divaricatus": "Dicrurus adsimilis",
    "Certhilauda curvirostris/brevirostris": "Certhilauda curvirostris",
    "Galerida cristata/macrorhyncha": "Galerida cristata",
    "Rubigula squamata": "Rubigula squamatus",
    "Salpornis spilonota": "Salpornis spilonotus",
    "Saroglossa spilopterus": "Saroglossa spiloptera",
    "Cincloramphus mariei": "Cincloramphus mariae",
    "Sylvia nigricapillus": "Sylvia nigricapilla",
    "Phylloscopus sibilatrix": "Phylloscopus sibillatrix",
    "Phylloscopus affinis/occisinensis": "Phylloscopus affinis",
    "Notopholia corusca": "Notopholia corrusca",
    "Buphagus erythrorynchus": "Buphagus erythrorhynchus",
    "Neocossyphus finschi": "Neocossyphus finschii",
    "Calamornis heudei": "Paradoxornis heudei",
    "Cholornis paradoxus": "Cholornis paradoxa",
    "Phyllergates cucullatus": "Phyllergates cuculatus",
    "Monticola cinclorhyncha": "Monticola cinclorhynchus",
    "Ramphocelus bresilius": "Ramphocelus bresilia",
    "Sicalis uropygialis": "Sicalis uropigyalis",
    "Corcorax melanorhamphos": "Corcorax melanoramphos",
    "Chelidorhynx hypoxanthus": "Chelidorhynx hypoxantha",
    "Acrochordopus burmeisteri": "Phyllomyias burmeisteri",
    "Acrochordopus zeledoni": "Phyllomyias zeledoni",
    "Aerospiza castanilius": "Accipiter castanilius",
    "Aerospiza tachiro": "Accipiter tachiro",
    "Amirafra angolensis": "Mirafra angolensis",
    "Amirafra collaris": "Mirafra collaris",
    "Amirafra rufocinnamomea": "Mirafra rufocinnamomea",
    "Anarhynchus alticola": "Charadrius alticola",
    "Anarhynchus atrifrons": "Charadrius atrifrons",
    "Anarhynchus bicinctus": "Charadrius bicinctus",
    "Anarhynchus collaris": "Charadrius collaris",
    "Anarhynchus dealbatus": "Charadrius dealbatus",
    "Anarhynchus falklandicus": "Charadrius falklandicus",
    "Anarhynchus javanicus": "Charadrius javanicus",
    "Anarhynchus marginatus": "Charadrius marginatus",
    "Anarhynchus mongolus": "Charadrius mongolus",
    "Anarhynchus montanus": "Charadrius montanus",
    "Anarhynchus nivosus": "Charadrius alexandrinus",
    "Anarhynchus obscurus": "Charadrius obscurus",
    "Anarhynchus pallidus": "Charadrius pallidus",
    "Anarhynchus pecuarius": "Charadrius pecuarius",
    "Anarhynchus peronii": "Charadrius peronii",
    "Anarhynchus ruficapillus": "Charadrius ruficapillus",
    "Anarhynchus sanctaehelenae": "Charadrius sanctaehelenae",
    "Anarhynchus thoracicus": "Charadrius thoracicus",
    "Anarhynchus wilsonia": "Charadrius wilsonia",
    "Antiurus maculicaudus": "Hydropsalis maculicaudus",
    "Apteryx maxima": "Apteryx haastii",
    "Ardea coromanda": "Bubulcus coromandus",
    "Artomyias fuliginosa": "Bradornis fuliginosus",
    "Artomyias ussheri": "Muscicapa ussheri",
    "Astur bicolor": "Accipiter bicolor",
    "Astur chilensis": "Accipiter chilensis",
    "Astur cooperii": "Accipiter cooperii",
    "Astur gentilis": "Accipiter gentilis",
    "Astur gundlachi": "Accipiter gundlachi",
    "Astur henstii": "Accipiter henstii",
    "Astur melanoleucus": "Accipiter melanoleucus",
    "Astur meyerianus": "Accipiter meyerianus",
    "Botaurus cinnamomeus": "Ixobrychus cinnamomeus",
    "Botaurus dubius": "Ixobrychus dubius",
    "Botaurus eurhythmus": "Ixobrychus eurhythmus",
    "Botaurus exilis": "Ixobrychus exilis",
    "Botaurus flavicollis": "Dupetor flavicollis",
    "Botaurus involucris": "Ixobrychus involucris",
    "Botaurus minutus": "Ixobrychus minutus",
    "Botaurus sinensis": "Ixobrychus sinensis",
    "Botaurus sturmii": "Ixobrychus sturmii",
    "Buphagus erythroryncha": "Buphagus erythrorhynchus",
    "Centropus burchellii": "Centropus superciliosus",
    "Chalcopsitta fuscata": "Pseudeos fuscata",
    "Chiroxiphia bokermanni": "Antilophia bokermanni",
    "Chiroxiphia galeata": "Antilophia galeata",
    "Chrysuronia boucardi": "Amazilia boucardi",
    "Corypha africana": "Mirafra africana",
    "Corypha apiata": "Mirafra apiata",
    "Corypha fasciolata": "Mirafra fasciolata",
    "Corypha hypermetra": "Mirafra hypermetra",
    "Corypha somalica": "Mirafra somalica",
    "Cyclopsitta desmarestii": "Psittaculirostris desmarestii",
    "Cyclopsitta edwardsii": "Psittaculirostris edwardsii",
    "Cyclopsitta salvadorii": "Psittaculirostris salvadorii",
    "Daptrius albogularis": "Phalcoboenus albogularis",
    "Daptrius australis": "Phalcoboenus australis",
    "Daptrius carunculatus": "Phalcoboenus carunculatus",
    "Daptrius chimachima": "Milvago chimachima",
    "Daptrius chimango": "Milvago chimango",
    "Daptrius megalopterus": "Phalcoboenus megalopterus",
    "Driophlox atrimaxillaris": "Habia atrimaxillaris",
    "Driophlox cristata": "Habia cristata",
    "Driophlox fuscicauda": "Habia fuscicauda",
    "Driophlox gutturalis": "Habia gutturalis",
    "Emblema ruficauda": "Neochmia ruficauda",
    "Eopsaltria capito": "Tregellasia capito",
    "Eopsaltria leucops": "Tregellasia leucops",
    "Eopsaltria placens": "Poecilodryas placens",
    "Gyps rueppelli": "Gyps rueppellii",
    "Hesperoburhinus bistriatus": "Burhinus bistriatus vocifer",
    "Hesperoburhinus superciliaris": "Burhinus superciliaris",
    "Ixos leucogrammicus": "Pycnonotus leucogrammicus",
    "Leucophantes brachyurus": "Poecilodryas brachyura",
    "Lophorina latipennis": "Lophorina superba",
    "Lophospiza griseiceps": "Accipiter griseiceps",
    "Lophospiza trivirgata": "Accipiter trivirgatus",
    "Melanocharis piperata": "Rhamphocharis crassirostris",
    "Melanodryas bimaculata": "Peneothello bimaculata",
    "Melanodryas cryptoleuca": "Peneothello cryptoleuca",
    "Melanodryas cyanus": "Peneothello cyanus",
    "Melanodryas pulverulenta": "Peneoenanthe pulverulenta",
    "Melanodryas sigillata": "Peneothello sigillata",
    "Meliphaga chrysogenys": "Oreornis chrysogenys",
    "Meliphaga imitatrix": "Microptilotis imitatrix",
    "Microtarsus eutilotus": "Pycnonotus eutilotus",
    "Microtarsus fuscoflavescens": "Pycnonotus fuscoflavescens",
    "Microtarsus melanocephalos": "Pycnonotus atriceps",
    "Microtarsus priocephalus": "Pycnonotus priocephalus",
    "Microtarsus urostictus": "Pycnonotus urostictus",
    "Myopornis boehmi": "Muscicapa boehmi",
    "Nannopsittacus gulielmitertii": "Cyclopsitta gulielmitertii",
    "Neophilydor erythrocercum": "Philydor erythrocercum",
    "Neophilydor fuscipenne": "Philydor fuscipenne",
    "Oenanthe heuglinii": "Oenanthe heuglini",
    "Pachyglossa agilis": "Dicaeum agile",
    "Pachyglossa chrysorrhea": "Dicaeum chrysorrheum",
    "Pachyglossa everetti": "Dicaeum everetti",
    "Pachyglossa melanozantha": "Dicaeum melanozanthum",
    "Pachyglossa olivacea": "Prionochilus olivaceus",
    "Pachyglossa propria": "Dicaeum proprium",
    "Pachyglossa vincens": "Dicaeum vincens",
    "Plocealauda affinis": "Mirafra affinis",
    "Plocealauda assamica": "Mirafra assamica",
    "Plocealauda erythrocephala": "Mirafra erythrocephala",
    "Plocealauda erythroptera": "Mirafra erythroptera",
    "Plocealauda microptera": "Mirafra microptera",
    "Psitteuteles porphyrocephalus": "Glossopsitta porphyrocephala",
    "Psitteuteles pusillus": "Glossopsitta pusilla",
    "Quechuavis decussata": "Systellura decussata",
    "Rufirallus fasciatus": "Anurolimnas fasciatus",
    "Rufirallus leucopyrrhus": "Laterallus leucopyrrhus",
    "Rufirallus schomburgkii": "Micropygia schomburgkii",
    "Rufirallus xenopterus": "Laterallus xenopterus",
    "Strigops habroptilus": "Strigops habroptila",
    "Tachyspiza albogularis": "Accipiter albogularis",
    "Tachyspiza badia": "Accipiter badius",
    "Tachyspiza brevipes": "Accipiter brevipes",
    "Tachyspiza cirrocephala": "Accipiter cirrocephalus",
    "Tachyspiza erythrauchen": "Accipiter erythrauchen",
    "Tachyspiza erythropus": "Accipiter erythropus",
    "Tachyspiza fasciata": "Accipiter fasciatus",
    "Tachyspiza francesiae": "Accipiter francesiae",
    "Tachyspiza gularis": "Accipiter gularis",
    "Tachyspiza henicogramma": "Accipiter henicogrammus",
    "Tachyspiza hiogaster": "Accipiter hiogaster",
    "Tachyspiza melanochlamys": "Accipiter melanochlamys",
    "Tachyspiza minulla": "Accipiter minullus",
    "Tachyspiza nanus": "Accipiter nanus",
    "Tachyspiza novaehollandiae": "Accipiter novaehollandiae",
    "Tachyspiza poliocephala": "Accipiter poliocephalus",
    "Tachyspiza rhodogaster": "Accipiter rhodogaster",
    "Tachyspiza rufitorques": "Accipiter rufitorques",
    "Tachyspiza soloensis": "Accipiter soloensis",
    "Tachyspiza trinotata": "Accipiter trinotatus",
    "Tachyspiza virgata": "Accipiter virgatus",
    "Thinornis dubius": "Charadrius dubius",
    "Thinornis forbesi": "Charadrius forbesi",
    "Thinornis melanops": "Elseyornis melanops",
    "Thinornis placidus": "Charadrius placidus",
    "Thinornis tricollaris": "Charadrius tricollaris",
    "Trichoglossus borneus": "Eos bornea",
    "Trichoglossus concinnus": "Glossopsitta concinna",
    "Trichoglossus cyanogenius": "Eos cyanogenia",
    "Trichoglossus reticulatus": "Eos reticulata",
    "Trichoglossus squamatus": "Eos squamata",
    "Turdoides rufocinctus": "Kupeornis rufocinctus",
    "Tyranniscus cinereiceps": "Phyllomyias cinereiceps",
    "Tyranniscus nigrocapillus": "Phyllomyias nigrocapillus",
    "Tyranniscus uropygialis": "Phyllomyias uropygialis",
    "Vini margarethae": "Charmosyna margarethae",
    "Agalychnis taylori": "Agalychnis callidryas",
    "Anstisia alba": "Geocrinia alba",
    "Anstisia lutea": "Geocrinia lutea",
    "Anstisia rosea": "Geocrinia rosea",
    "Anstisia vitellina": "Geocrinia vitellina",
    "Aquarana catesbeiana": "Lithobates catesbeianus",
    "Aquarana clamitans": "Lithobates clamitans",
    "Aquarana grylio": "Lithobates grylio",
    "Aquarana septentrionalis": "Lithobates septentrionalis",
    "Arphia pseudonietana": "Arphia pseudo-nietana",
    "Boreorana sylvatica": "Lithobates sylvaticus",
    "Boulenophrys jinggangensis": "Megophrys jinggangensis",
    "Bufo praetextatus": "Bufo japonicus",
    "Cecropis rufula": "Cecropis daurica",
    "Cephalophorus harveyi": "Cephalophus harveyi",
    "Cinnyris frenatus": "Cinnyris jugularis",
    "Cinnyris infrenatus": "Cinnyris jugularis",
    "Cinnyris ornatus": "Cinnyris jugularis",
    "Circaetus spectabilis": "Dryotriorchis spectabilis",
    "Clemacantha goliath": "Eurycnema goliath",
    "Corvus philippinus": "Corvus macrorhynchos",
    "Corypha athi": "Mirafra africana",
    "Duellmanohyla legleri": "Ptychohyla legleri",
    "Duellmanohyla salvadorensis": "Ptychohyla salvadorensis",
    "Elachistocleis ovalis-complex": "Elachistocleis ovalis",
    "Emblema modesta": "Neochmia modesta",
    "Erethizon dorsatum": "Erethizon dorsatus",
    "Erythrogenys imberbis": "Megapomatorhinus erythrogenys",
    "Firouzophrynus stomaticus": "Bufo stomaticus",
    "Gastrotheca coeruleomaculata": "Gastrotheca coeruleomaculatus",
    "Hyalinobatrachium viridissimum": "Hyalinobatrachium fleischmanni",
    "Hyla flaviventris": "Dryophytes flaviventris",
    "Hyperolius hypsiphonus": "Alexteroon hypsiphonus",
    "Tibicinoides boweni": "Okanagana boweni",
    "Tibicinoides catalina": "Okanagana catalina",
    "Tibicinoides pallidula": "Okanagana pallidula",
    "Tibicinoides rubrovenosa": "Okanagana rubrovenosa",
    "Tibicinoides striatipes": "Okanagana striatipes",
    "Tibicinoides uncinata": "Okanagana uncinata",
    "Tibicinoides utahensis": "Okanagana utahensis",
    "Tibicinoides vanduzeei": "Okanagana vanduzeei",
    "Larus mongolicus": "Larus vegae",
    "Laterallus spilopterus": "Laterallus spiloptera",
    "Lupulella adusta": "Canis adustus",
    "Lupulella adustus": "Canis adustus",
    "Lupulella mesomelas": "Canis mesomelas",
    "Lycalopex grisea": "Pseudalopex griseus",
    "Lycalopex gymnocerca": "Lycalopex gymnocercus",
    "Magicicada cassinii": "Magicicada cassini",
    "Melanophryniscus formosus": "Melanophryniscus stelzneri",
    "Melogale subaurantiaca": "Melogale moschata",
    "Mertensophryne lughensis": "Poyntonophrynus lughensis",
    "Micropterus nigricans": "Micropterus floridanus",
    "Neogale vison": "Neovison vison",
    "Ochotona pallasii": "Ochotona pallasi",
    "Ololygon arduoa": "Ololygon arduous",
    "Otospermophilus douglasii": "Spermophilus beecheyi",
    "Pelobatrachus kobayashii": "Megophrys kobayashii",
    "Pelophylax 'esculentus'": "Pelophylax esculentus",
    "Pelophylax 'grafi'": "Pelophylax perezi",
    "Petaurista grandis": "Petaurista petaurista",
    "Philoria sphagnicola": "Philoria sphagnicolus",
    "Platyplectrum fletcheri": "Lechriodus fletcheri",
    "Pogoniulus uropygialis": "Pogoniulus pusillus",
    "Pogonotriccus difficilis": "Phylloscartes difficilis",
    "Pogonotriccus paulista": "Phylloscartes paulista",
    "Pycnogaster cucullata": "Pycnogaster cucullatus",
    "Ranoidea lesueuri": "Ranoidea lesueurii",
    "Romalea eques": "Taeniopoda eques",
    "Serranobatrachus sanctaemartae": "Eleutherodactylus sanctaemartae",
    "Stiphrornis mabirae": "Stiphrornis xanthogaster",
    "Tachiramantis cuentasi": "Eleutherodactylus cuentasi",
    "Tachiramantis tayrona": "Eleutherodactylus tayrona",
    "Tachyspiza haplochroa": "Accipiter haplochrous",
    "Tarsiger formosanus": "Tarsiger indicus",
    "Trachycephalus vermiculatus-complex": "Trachycephalus vermiculatus",
    "Troglodytes mesoleucus": "Troglodytes aedon",
    "Xenops mexicanus": "Xenops genibarbis",
    "Zoraena maculata": "Cordulegaster maculata",
    "Cossypha ansorgei": "Xenocopsychus ansorgei",
    "Curruca althaea": "Curruca curruca",
    "Melaenornis infuscatus": "Bradornis infuscatus",
    "Melaenornis mariquensis": "Bradornis mariquensis",
    "Melaenornis microrhynchus": "Bradornis microrhynchus",
    "Mops plicatus": "Chaerephon plicatus",
    "Ommatophoca rossi": "Ommatophoca rossii",
    "Percnostola fortis": "Hafferia fortis",
    "Percnostola goeldii": "Akletos goeldii",
    "Percnostola immaculata": "Hafferia immaculata",
    "Percnostola melanoceps": "Akletos melanoceps",
    "Percnostola zeledoni": "Hafferia zeledoni",
    "Rubigula melanicterus": "Rubigula melanictera",
    "Spermestes nigriceps": "Spermestes bicolor",
    "Spharagemon marmorata": "Spharagemon marmoratum",
}

GBIF_MISSING_SPECIES = [
    "Sporophila [undescribed",
    "Myiornis [undescribed",
    "Phacellodomus [undescribed",
    "Anasaitis canosus",
    "Angusta fangtingyui",
    "Atrapsalta audax",
    "Brachycephalus rotenbergae",
    "Cacosternum cederbergense",
    "Camponotus confusus",
    "Chinavia pensylvanica",
    "Chlorocanta viridis",
    "Chrysochraon beybienkoi",
    "Conepatus amazonicus",
    "Eleutherodactylus jamesdixoni",
    "Euspinolia militaris",
    "Geocrinia sparsiflora",
    "Hewlettia nigriviridis",
    "Hylarana sundabarat",
    "Imatismus villosus",
    "Indopurana cheeveeda",
    "Limnodynastes grayi",
    "Limnodynastes superciliaris",
    "Litoria amnicola",
    "Litoria calliscelis",
    "Litoria ridibunda",
    "Litoria sibilus",
    "Macrosemia fengi",
    "Mariazofia gibba",
    "Melanophryniscus diabolicus",
    "Metapurana nebulilinea",
    "Myopsalta bisonabilis",
    "Myopsalta wollomombi",
    "Nesosydne argyroxiphium",
    "Nyctimystes multicolor",
    "Odontophrynus asper",
    "Oliarus lorettae",
    "Opodiphthera eucalypti",
    "Orientopsaltria musicus",
    "Paropsis atomaria",
    "Pericallea ewartioides",
    "Planopleura kaempferi",
    "Planopleura takasagona",
    "Purapurana carmente",
    "Rhinella bella",
    "Satizabalus sodalis",
    "Tanychlamys indica",
    "Tettigetta shansiensis",
    "Tomopterna adiastola",
    "Uromenus bonneti",
    "Vietanna orientalis",
    "Yoyetta darug",
    "Yoyetta fumea",
    "Yoyetta psammitica",
    "A capella",
    "Accelerating, revving,",
    "Acoustic guitar",
    "Air brake",
    "Air conditioning",
    "Air horn,",
    "Aircraft engine",
    "Alarm clock",
    "Ambient music",
    "Ambulance (siren)",
    "Angry music",
    "Artillery fire",
    "Baby cry,",
    "Baby laughter",
    "Background music",
    "Basketball bounce",
    "Bass drum",
    "Bass guitar",
    "Bathtub (filling",
    "Battle cry",
    "Bee, wasp,",
    "Beep, bleep",
    "Belly laugh",
    "Bicycle bell",
    "Bird flight,",
    "Bird vocalization,",
    "Boat, Water",
    "Bowed string",
    "Brass instrument",
    "Burping, eructation",
    "Burst, pop",
    "Busy signal",
    "Canidae, dogs,",
    "Cap gun",
    "Car alarm",
    "Car passing",
    "Carnatic music",
    "Cash register",
    "Cattle, bovinae",
    "Change ringing",
    "Chewing, mastication",
    "Chicken, rooster",
    "Child singing",
    "Child speech,",
    "Children playing",
    "Children shouting",
    "Chink, clink",
    "Chirp tone",
    "Chirp, tweet",
    "Chopping (food)",
    "Chorus effect",
    "Christian music",
    "Christmas music",
    "Chuckle, chortle",
    "Church bell",
    "Civil defense",
    "Classical music",
    "Cnemotriccus sp.nov.",
    "Coin (dropping)",
    "Computer keyboard",
    "Crowing, cock-a-doodle-doo",
    "Crumpling, crinkling",
    "Crying, sobbing",
    "Cupboard open",
    "Cutlery, silverware",
    "Dance music",
    "Dental drill,",
    "Dial tone",
    "Dishes, pots,",
    "Domestic animals,",
    "Double bass",
    "Drawer open",
    "Drum and",
    "Drum kit",
    "Drum machine",
    "Drum roll",
    "Effects unit",
    "Electric guitar",
    "Electric piano",
    "Electric shaver,",
    "Electric toothbrush",
    "Electronic dance",
    "Electronic music",
    "Electronic organ",
    "Electronic tuner",
    "Emergency vehicle",
    "Engine knocking",
    "Engine starting",
    "Environmental noise",
    "Exciting music",
    "Female singing",
    "Female speech,",
    "Field recording",
    "Filing (rasp)",
    "Fill (with",
    "Finger snapping",
    "Fire alarm",
    "Fire engine,",
    "Fixed-wing aircraft,",
    "Fly, housefly",
    "Folk music",
    "French horn",
    "Frying (food)",
    "Funny music",
    "Gospel music",
    "Grus carunculata",
    "Gunshot, gunfire",
    "Hair dryer",
    "Hammond organ",
    "Happy music",
    "Heart murmur",
    "Heart sounds,",
    "Heavy engine",
    "Heavy metal",
    "Hip hop",
    "House music",
    "Hubbub, speech",
    "Ice cream",
    "Icteria ×",
    "Independent music",
    "Inside, large",
    "Inside, public",
    "Inside, small",
    "Jet engine",
    "Jingle (music)",
    "Jingle bell",
    "Jingle, tinkle",
    "Keyboard (musical)",
    "Keys jangling",
    "Lawn mower",
    "Light engine",
    "Livestock, farm",
    "Machine gun",
    "Mains hum",
    "Male singing",
    "Male speech,",
    "Mallet percussion",
    "Marimba, xylophone",
    "Mechanical fan",
    "Medium engine",
    "Microwave oven",
    "Middle Eastern",
    "Motor vehicle",
    "Motorboat, speedboat",
    "Music for",
    "Music of",
    "Music of",
    "Music of",
    "Music of",
    "Musical instrument",
    "Narration, monologue",
    "Neigh, whinny",
    "New-age music",
    "Outside, rural",
    "Outside, urban",
    "Pigeon, dove",
    "Pink noise",
    "Plucked string",
    "Police car",
    "Pop music",
    "Power tool",
    "Power windows,",
    "Progressive rock",
    "Propeller, airscrew",
    "Psychedelic rock",
    "Pump (liquid)",
    "Punk rock",
    "Race car,",
    "Rail transport",
    "Railroad car,",
    "Rain on",
    "Ratchet, pawl",
    "Rattle (instrument)",
    "Reversing beeps",
    "Rhythm and",
    "Roaring cats",
    "Rock and",
    "Rock music",
    "Rodents, rats,",
    "Romerus romeri",
    "Rowboat, canoe,",
    "Rustling leaves",
    "Sad music",
    "Sailboat, sailing",
    "Salsa music",
    "Scary music",
    "Scratching (performance",
    "Sewing machine",
    "Shuffling cards",
    "Sine wave",
    "Singing bowl",
    "Single-lens reflex",
    "Sink (filling",
    "Slap, smack",
    "Sliding door",
    "Smash, crash",
    "Smoke detector,",
    "Snare drum",
    "Soul music",
    "Sound effect",
    "Soundtrack music",
    "Speech synthesizer",
    "Splash, splatter",
    "Steam whistle",
    "Steel guitar,",
    "Stomach rumble",
    "String section",
    "Subway, metro,",
    "Swing music",
    "Synthetic singing",
    "Tapping (guitar",
    "Telephone bell",
    "Telephone dialing,",
    "Tender music",
    "Theme music",
    "Throat clearing",
    "Thump, thud",
    "Tire squeal",
    "Toilet flush",
    "Traditional music",
    "Traffic noise,",
    "Train horn",
    "Train wheels",
    "Train whistle",
    "Trance music",
    "Trickle, dribble",
    "Tubular bells",
    "Tuning fork",
    "Vacuum cleaner",
    "Vehicle horn,",
    "Video game",
    "Violin, fiddle",
    "Vocal music",
    "Wail, moan",
    "Walk, footsteps",
    "Water tap,",
    "Waves, surf",
    "Wedding music",
    "Whack, thwack",
    "Whale vocalization",
    "Whimper (dog)",
    "White noise",
    "Whoosh, swoosh,",
    "Wild animals",
    "Wind chime",
    "Wind instrument,",
    "Wind noise",
    "Wood block",
    "Zipper (clothing)",
    "Heliobletus sp.nov.lontras",
    "Antillicharis oriobates",
    "Microeca tax.nov.bismarck",
    "Myiornis sp.nov.maranhao_piaui",
    "Phacellodomus tax.nov.",
]


def _read_addr(addr_file: str) -> str:
    """Read host:port from a server.addr file written by a Slurm job.

    Returns
    -------
    str
        The ``host:port`` string.
    """
    from pathlib import Path

    with open(Path(addr_file).expanduser()) as f:
        return f.read().strip()


def _make_http_classify_fn(client: HttpClient) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a classify_fn that POSTs a batch of audio windows to a /logits HTTP server.

    Audio is base64-encoded and sent as JSON alongside ``num_windows`` (the batch
    size), which lets a server reshape the flat float32 buffer into per-window
    clips without assuming a fixed window length. Retry logic and connection
    pooling are handled by ``client``. Servers that fix the window length (e.g.
    Perch2, AudioProtoPNet) ignore ``num_windows`` and infer the batch from the
    payload byte length.

    Parameters
    ----------
    client : HttpClient
        Configured HTTP client pointing at the ``/logits`` route.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        ``(B, samples) → (B, n_classes)`` logits.
    """

    def classify_fn(audio: torch.Tensor) -> torch.Tensor:
        audio_np = audio.detach().cpu().numpy().astype(np.float32)
        response = client({"audio": audio_np.tobytes(), "num_windows": audio_np.shape[0]})
        return torch.tensor(response["logits"])  # (B, n_classes)

    return classify_fn


def create_perch2_detector(
    labels: list[str] | None,
    addr_file: str = "~/perch2-server/server.addr",
    hop_size: float = 5.0,
    window_size: float = 5.0,
    analysis_window: float | None = None,
) -> SlidingWindowDetector:
    """Create a SlidingWindowDetector backed by the Perch 2 HTTP server.

    Reads the server address from ``addr_file``, fetches the label list from
    ``/labels``, converts to GBIF, and wires up an HTTP classify_fn that POSTs
    WAV bytes to ``/logits``.

    Parameters
    ----------
    labels : list[str] | None
        GBIF scientific names expected by the evaluation pipeline.
        If None, defaults to classifier outputs.
    addr_file : str
        Path to the ``server.addr`` file written by the Slurm job.
    hop_size : float
        Hop duration in seconds. Default 5.0 (no overlap).
    window_size : float
        Window duration in seconds. Default 5.0.

    Raises
    -------
    ValueError
        If unable to convert all names to gbif

    Returns
    -------
    SlidingWindowDetector
        Ready for ``run()``.
    """
    addr = _read_addr(addr_file)
    client = HttpClient(f"http://{addr}", route="logits", audio_key="audio", timeout=60.0)
    classifier_meta = client.describe()
    classifier_labels: list[str] = classifier_meta["labels"]

    # convert sciname -> gbif
    sci_name_to_gbif = GBIFConverter()
    classifier_labels_gbif = []

    unconverted_scientific_names = []

    print("Computing Clements->GBIF conversion")
    for raw_label in classifier_labels:
        sci_name = " ".join(raw_label.split(" ")[:2])

        if len(sci_name.split(" ")) == 1:
            # shortcut for audioset classes
            classifier_labels_gbif.append(sci_name)
            continue

        if sci_name in GBIF_MISSING_SPECIES:
            classifier_labels_gbif.append(sci_name)
            continue

        sci_name = SCI_NAME_CORRECTION_MANUAL.get(sci_name, sci_name)

        info, ok = sci_name_to_gbif(sci_name)
        if not ok:
            unconverted_scientific_names.append(sci_name)
            continue

        classifier_labels_gbif.append(info["canonicalName"])

    if unconverted_scientific_names:
        print("Unconverted scientific names:")
        print(unconverted_scientific_names)
        raise ValueError("Unable to convert all ebird codes to GBIF")

    classify_fn = _make_http_classify_fn(client)

    return SlidingWindowDetector(
        classify_fn=classify_fn,
        classifier_labels=classifier_labels_gbif,
        sample_rate=_SAMPLE_RATE,
        window_size=window_size,
        hop_size=hop_size,
        activation="softmax",
        labels=labels,
        analysis_window=analysis_window,
        detector_type="perch2",
        classifier_server_config=classifier_meta,
    )


def create_audioprotopnet_detector(
    labels: None | list[str],
    addr_file: str = "~/audioprotopnet-server/server.addr",
    hop_size: float = 5.0,
    window_size: float = 5.0,
    analysis_window: float | None = None,
) -> SlidingWindowDetector:
    """Create a SlidingWindowDetector backed by the AudioProtoPNet HTTP server.

    Reads the server address from ``addr_file``, fetches the label list from
    ``/labels`` (eBird codes), converts to GBIF, builds an EBirdToGBIFConverter
    to remap to ``labels``, and wires up an HTTP classify_fn.

    Parameters
    ----------
    labels : None | list[str]
        GBIF scientific names expected by the evaluation pipeline.
        If None, defaults to classifier outputs
    addr_file : str
        Path to the ``server.addr`` file written by the Slurm job.
    hop_size : float
        Hop duration in seconds. Default 5.0 (no overlap).
    window_size : float
        Window duration in seconds. Default 5.0.

    Returns
    -------
    SlidingWindowDetector
        Ready for ``run()``.

    Raises
    -------
    ValueError
        If unable to convert all names to gbif
    """
    addr = _read_addr(addr_file)
    client = HttpClient(f"http://{addr}", route="logits", audio_key="audio", timeout=60.0)
    classifier_meta = client.describe()
    classifier_labels: list[str] = classifier_meta["labels"]

    # convert ebird2021 -> sciname -> gbif
    ebird_taxonomy_path = "gs://esp-ml-datasets/wabad/v0.1.0/eBird_Taxonomy_v2021.csv"
    clements_ontology = pd.read_csv(ebird_taxonomy_path)
    clements_ontology = clements_ontology[~pd.isna(clements_ontology["SCI_NAME"])]

    species_code_to_sci_name = clements_ontology.set_index("SPECIES_CODE")["SCI_NAME"].to_dict()
    sci_name_to_gbif = GBIFConverter()

    classifier_labels_gbif = []

    unconverted_species_codes = []
    unconverted_scientific_names = []

    print("Computing ebird->GBIF conversion")
    for ebird_label in classifier_labels:
        if ebird_label in EBIRD_TO_SPECIES_MANUAL:
            raw_sci_name = EBIRD_TO_SPECIES_MANUAL[ebird_label]
        else:
            raw_sci_name = species_code_to_sci_name.get(ebird_label, None)

        if raw_sci_name is None:
            unconverted_species_codes.append(ebird_label)
            continue

        sci_name = " ".join(raw_sci_name.split(" ")[:2])

        if sci_name in GBIF_MISSING_SPECIES:
            classifier_labels_gbif.append(sci_name)
            continue

        sci_name = SCI_NAME_CORRECTION_MANUAL.get(sci_name, sci_name)

        info, ok = sci_name_to_gbif(sci_name)
        if not ok:
            unconverted_scientific_names.append(sci_name)
            continue

        classifier_labels_gbif.append(info["canonicalName"])

    if unconverted_species_codes or unconverted_scientific_names:
        print("Unconverted species codes:")
        print(unconverted_species_codes)
        print("Unconverted scientific names:")
        print(unconverted_scientific_names)
        raise ValueError("Unable to convert all ebird codes to GBIF")

    classify_fn = _make_http_classify_fn(client)
    return SlidingWindowDetector(
        classify_fn=classify_fn,
        classifier_labels=classifier_labels_gbif,
        sample_rate=_SAMPLE_RATE,
        window_size=window_size,
        hop_size=hop_size,
        activation="sigmoid",
        labels=labels,
        analysis_window=analysis_window,
        detector_type="audioprotopnet",
        classifier_server_config=classifier_meta,
    )


def _convert_beats_sl_all_labels_to_gbif(classifier_labels: list[str]) -> list[str]:
    """Convert a list of scientific names to GBIF canonical names.

    Uses the same logic as the Perch2 factory: truncate to genus+species,
    pass through manual corrections, then look up via GBIFConverter.
    Single-word labels (e.g. AudioSet classes) and known-missing species
    are passed through unchanged.

    Returns
    -------
    list[str]
        GBIF canonical names, in the same order as the input.

    Raises
    ------
    ValueError
        If any two-word scientific name cannot be resolved to GBIF.
    """
    sci_name_to_gbif = GBIFConverter()
    classifier_labels_gbif = []
    unconverted_scientific_names = []

    print("Computing label->GBIF conversion")
    for raw_label in classifier_labels:
        sci_name = " ".join(raw_label.split(" ")[:2])

        if len(sci_name.split(" ")) == 1:
            # shortcut for audioset classes
            classifier_labels_gbif.append(sci_name)
            continue

        if sci_name in GBIF_MISSING_SPECIES:
            classifier_labels_gbif.append(sci_name)
            continue

        sci_name = SCI_NAME_CORRECTION_MANUAL.get(sci_name, sci_name)

        info, ok = sci_name_to_gbif(sci_name)
        if not ok:
            unconverted_scientific_names.append(sci_name)
            continue
        classifier_labels_gbif.append(info["canonicalName"])

    if unconverted_scientific_names:
        print("Unconverted scientific names:")
        print(unconverted_scientific_names)
        raise ValueError("Unable to convert all names to GBIF")

    return classifier_labels_gbif


_BEATS_SL_ALL_ADDR_FILE = "~/esp-research/projects/sound-event-detection/.server_addrs/beats_sl_all.addr"


def create_beats_sl_all_detector(
    labels: None | list[str],
    addr_file: str = _BEATS_SL_ALL_ADDR_FILE,
    hop_size: float = 10.0,
    window_size: float = 10.0,
    analysis_window: float | None = None,
) -> SlidingWindowDetector:
    """Create a SlidingWindowDetector backed by the BEATs-SL-All HTTP server.

    Reads the server address from ``addr_file``, fetches the raw classifier
    label list from ``/`` (scientific names + AudioSet classes), converts them
    to GBIF, and wires up an HTTP classify_fn that POSTs audio windows to
    ``/logits``. The ``esp_aves2_sl_beats_all`` model itself is loaded by the
    server (see `sound_event_detection.serving.sl_beats_all_server`).

    The model expects 16 kHz audio.  Evaluation configs must use datasets
    loaded at 16 kHz (e.g. ``wabad_16k/`` configs).

    Parameters
    ----------
    labels : None | list[str]
        GBIF scientific names expected by the evaluation pipeline.
        If None, defaults to the classifier's own (GBIF-converted) labels.
    addr_file : str
        Path to the ``server.addr`` file written by the Slurm job.
    hop_size : float
        Hop duration in seconds. Default 10.0 (no overlap, matches training window).
    window_size : float
        Window duration in seconds. Default 10.0 (matches training window).
    analysis_window : float | None
        If set, extract a shorter window per hop. Default None.

    Returns
    -------
    SlidingWindowDetector
        Ready for ``run()``.
    """
    addr = _read_addr(addr_file)
    client = HttpClient(f"http://{addr}", route="logits", audio_key="audio", timeout=60.0)
    classifier_meta = client.describe()
    classifier_labels: list[str] = classifier_meta["labels"]

    # GBIF conversion can raise ValueError if a classifier label can't be resolved.
    classifier_labels_gbif = _convert_beats_sl_all_labels_to_gbif(classifier_labels)

    classify_fn = _make_http_classify_fn(client)

    return SlidingWindowDetector(
        classify_fn=classify_fn,
        classifier_labels=classifier_labels_gbif,
        sample_rate=16000,
        window_size=window_size,
        hop_size=hop_size,
        activation="softmax",
        labels=labels,
        analysis_window=analysis_window,
        detector_type="beats_sl_all",
        classifier_server_config=classifier_meta,
    )


_SLIDING_WINDOW_FACTORIES = {
    "perch2": create_perch2_detector,
    "audioprotopnet": create_audioprotopnet_detector,
    "beats_sl_all": create_beats_sl_all_detector,
}

# Public, read-only view of the supported sliding-window detector types, for
# callers that need to test membership against the registry.
SLIDING_WINDOW_DETECTOR_TYPES = frozenset(_SLIDING_WINDOW_FACTORIES)


def create_sliding_window_detector_from_config(
    model_config: dict,
    labels: list[str] | None = None,
) -> SlidingWindowDetector:
    """Create a `SlidingWindowDetector` from a model config dict.

    Dispatches to the appropriate factory based on ``model_config["type"]``.
    Every supported type is backed by an HTTP server, so no compute device is
    needed here (the server holds the model).

    Parameters
    ----------
    model_config : dict
        Model config dict with keys:

        - ``type``: one of ``"perch2"``, ``"audioprotopnet"``, ``"beats_sl_all"``
        - ``addr_file``: server address file
        - ``hop_size``, ``window_size``, ``analysis_window``: sliding window params

        Keys other than ``type`` may be omitted, in which case the per-type
        factory's own defaults apply.
    labels : list[str] | None
        Output label list passed through to the underlying factory.

    Returns
    -------
    SlidingWindowDetector
        Ready for ``run()``.

    Raises
    ------
    ValueError
        If ``model_config["type"]`` is not a recognised sliding window detector type.
    """
    model_type = model_config["type"]
    if model_type not in _SLIDING_WINDOW_FACTORIES:
        raise ValueError(
            f"Unknown sliding window detector type: {model_type!r}. "
            f"Expected one of: {sorted(_SLIDING_WINDOW_FACTORIES)}."
        )
    kwargs = {
        key: model_config[key]
        for key in ("addr_file", "hop_size", "window_size", "analysis_window")
        if key in model_config
    }
    return _SLIDING_WINDOW_FACTORIES[model_type](labels=labels, **kwargs)
