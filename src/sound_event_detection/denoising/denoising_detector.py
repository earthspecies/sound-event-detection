"""Focal-species denoising detector bundling whole-file separation with detection.

`DenoisingDetector` is a *model that wraps clients*, like `SlidingWindowDetector`
wraps a classifier client: it owns no weights, but bundles a detector client (a
`DetectorClient`, e.g. `ServedDetectorClient`) and a source-separation client
(a `SourceSeparatorClient`, e.g. `BirdMixItClient`) and drives them together.
It is served by `sound_event_detection.serving.serve_denoising_detector` (launched with
``sed.denoising_app``) and reached by evaluation and large-scale inference
through a `ServedDenoisingDetectorClient`. Per recording it separates the whole
file into **stitched, whole-file stems**,
resamples each stem up to the detector's rate, and runs the detector once over
all stems. The detector does its own windowing/overlap, so there are no
chunk boundaries and no 5 s separation seams.

The irreducible product is a `StemDetections`: the whole-file stems plus their
per-stem framewise predictions. `combined` (per-frame per-class max over the
stems) and `denoise` (focal-gated stem sum) are **pure derivations** over it —
`DenoisingDetector.run` returns the former (the run-surface a `DetectorClient`
caller consumes); `separate_and_detect` hands back the `StemDetections` so
callers can derive the focal-gated denoised waveform too.

Audio is exchanged at the *separator's* sample rate (the honest input rate, and
the rate the denoised waveform is returned at). Stems are resampled up to the
detector's sample rate only for detection.
"""

import time
from collections.abc import Iterator, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Literal

import librosa
import numpy as np
from pydantic import BaseModel

from esp_research.adapters.client_config import HttpClientConfig
from esp_research.protocols.classifier import MultiLabelClassifierOutput
from esp_research.protocols.detector import DetectorOutput
from sound_event_detection.adapters.dispatch import DetectorClient
from sound_event_detection.denoising.source_separator import SourceSeparatorClient

#: Resampling method for the stem -> detector-rate resample (see
#: `DenoisingDetector._resample`).
ResamplingMethod = Literal["librosa_kaiser_best", "torchaudio_kaiser_best", "torchaudio_kaiser_fast"]

#: torchaudio `functional.resample` kwargs that mimic librosa's kaiser resamplers,
#: from the torchaudio resampling tutorial
#: (https://docs.pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html).
_TORCHAUDIO_KAISER_PARAMS: dict[str, dict[str, object]] = {
    "torchaudio_kaiser_best": {
        "lowpass_filter_width": 64,
        "rolloff": 0.9475937167399596,
        "resampling_method": "sinc_interp_kaiser",
        "beta": 14.769656459379492,
    },
    "torchaudio_kaiser_fast": {
        "lowpass_filter_width": 16,
        "rolloff": 0.85,
        "resampling_method": "sinc_interp_kaiser",
        "beta": 8.555504641634386,
    },
}

#: Default windows-per-forward-pass for the detector run over the stems, passed
#: to `detector.run`. Peak detector GPU memory scales with it (the served
#: AudioProtoPNet's cosine activation dominates), so the default is deliberately
#: small to fit on a shared card; raise it per call on a dedicated GPU.
DEFAULT_DETECTOR_BATCH_SIZE = 8

#: The denoising detector's model-type discriminator. Minted into
#: `DenoisingDetector.server_config` (and so the denoising server's ``GET /``
#: payload), matched by `detector_client_from_config` to pick the client
#: class, re-checked by `ServedDenoisingDetectorClient.from_config`, and
#: required as the ``type`` of the server's model-config YAML — one constant
#: so producer and consumers cannot drift apart.
DENOISING_DETECTOR_TYPE = "denoising_detector"


@dataclass(frozen=True)
class StemDetections:
    """The irreducible output of separate -> stitch -> detect for one recording.

    Everything else the pipeline produces is a view over these two arrays:
    `combined` (the `Detector` track) and `denoise` (the focal-isolated
    waveform) are pure derivations, not stored state.

    Attributes
    ----------
    stems : np.ndarray
        Whole-file separated stems at `sample_rate`, shape
        ``(n_stems, samples)``. Coherent across the whole file — the separator
        server has already resolved the cross-block stem permutation, so a stem
        index is a stable source for the whole recording.
    stem_preds : np.ndarray
        Per-stem framewise probabilities, shape ``(n_stems, frames, classes)``,
        at `frame_rate` — the detector run over the (resampled) stems.
    frame_rate : float
        Detector output frame_rate in Hz.
    labels : list[str]
        Detector output class labels; length equals ``stem_preds.shape[2]``.
    sample_rate : int
        Sample rate of `stems` in Hz (the separator's rate), also the rate of
        the denoised waveform.
    """

    stems: np.ndarray
    stem_preds: np.ndarray
    frame_rate: float
    labels: list[str]
    sample_rate: int

    def combined(self) -> DetectorOutput:
        """Combine the per-stem predictions into a framewise `DetectorOutput`.

        Returns
        -------
        DetectorOutput
            Per-frame per-class max over the stems, shape ``(1, frames,
            classes)`` — the detection track `DenoisingDetector.run` returns for
            this recording.
        """
        combined = self.stem_preds.max(axis=0)  # (frames, classes)
        return DetectorOutput(
            predictions=np.ascontiguousarray(combined[np.newaxis], dtype=np.float32),
            frame_rate=self.frame_rate,
            class_names=list(self.labels),
        )

    def denoise(self, focal_idxs: Sequence[int], threshold: float) -> np.ndarray:
        """Gate each stem by a set of focal tracks and sum into a denoised waveform.

        Each stem is kept on frames where *its own* probability for **any** of
        the focal classes in `focal_idxs` meets `threshold`, and zeroed
        elsewhere; the gated stems are summed. Because the stems are whole-file
        and coherent, the result has no chunk-boundary seams. Passing a
        single-element sequence isolates one species; passing several isolates
        their union (e.g. every species that might occur in the recording).

        Parameters
        ----------
        focal_idxs : Sequence[int]
            Indices of the focal classes within `labels`. A stem is gated in on
            a frame when its probability for any of them meets `threshold` (a
            union over the classes). Must be non-empty.
        threshold : float
            Focal probability threshold for the gate.

        Returns
        -------
        np.ndarray
            The denoised waveform of shape ``(samples,)`` at `sample_rate`,
            dtype float32.
        """
        focal_probs = self.stem_preds[:, :, focal_idxs].max(axis=-1)  # (n_stems, frames)
        gain = self._gain(focal_probs, threshold)  # (n_stems, frames)

        n_samples = self.stems.shape[1]
        frames = self.stem_preds.shape[1]
        frame_of_sample = np.minimum(
            (np.arange(n_samples) * self.frame_rate / self.sample_rate).astype(int),
            frames - 1,
        )
        gain_samples = gain[:, frame_of_sample]  # (n_stems, n_samples)
        waveform = (self.stems * gain_samples).sum(axis=0)  # (n_samples,)
        return np.ascontiguousarray(waveform, dtype=np.float32)

    def quality(self, focal_idxs: Sequence[int], threshold: float) -> tuple[np.ndarray, np.ndarray]:
        """Per-frame denoising-quality signals for the focal species.

        Both derive from the same gate `denoise` uses: a stem contributes to a
        frame when its probability for any class in `focal_idxs` meets
        `threshold`. These are meant to be pooled per recording downstream (e.g.
        mean detection probability, max stem count) for quality filtering.

        Parameters
        ----------
        focal_idxs : Sequence[int]
            Indices of the focal classes within `labels` (a union gate, matching
            `denoise`). Must be non-empty.
        threshold : float
            Focal probability threshold for the gate (the one `denoise` uses).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(detection_prob, n_gated_stems)``, both shape ``(frames,)``, dtype
            float32. Each stem's per-frame probability is its max over
            `focal_idxs`; `detection_prob` is the minimum of that across the
            gated-in stems per frame (the weakest contributor kept), or ``nan``
            on frames where no stem is gated in. `n_gated_stems` is the per-frame
            count of gated-in stems.
        """
        focal_probs = self.stem_preds[:, :, focal_idxs].max(axis=-1)  # (n_stems, frames)
        gated = focal_probs >= threshold  # (n_stems, frames)
        n_gated = gated.sum(axis=0).astype(np.float32)  # (frames,)
        # Min focal prob over gated-in stems: +inf where gated out so it never
        # wins the min, then nan on frames with no gated-in stem.
        masked = np.where(gated, focal_probs, np.inf)
        detection_prob = np.where(n_gated > 0, masked.min(axis=0), np.nan).astype(np.float32)
        return detection_prob, n_gated

    def stem_pairs(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(audio, preds)`` for each stem.

        Yields
        ------
        tuple[np.ndarray, np.ndarray]
            Per stem, its whole-file audio ``(samples,)`` at `sample_rate` and
            its framewise predictions ``(frames, classes)``.
        """
        for i in range(self.stems.shape[0]):
            yield self.stems[i], self.stem_preds[i]

    @staticmethod
    def _gain(focal_probs: np.ndarray, threshold: float) -> np.ndarray:
        """Return the per-frame gain applied to each stem.

        A hard gate: ``1.0`` where the focal probability meets `threshold`, else
        ``0.0``. Isolated here so a soft gain (a function of the probability)
        could replace it without touching `denoise`.

        Parameters
        ----------
        focal_probs : np.ndarray
            Focal-class probabilities, any shape.
        threshold : float
            Detection threshold in [0, 1].

        Returns
        -------
        np.ndarray
            Gain in [0, 1], same shape as `focal_probs`, dtype float32.
        """
        return (focal_probs >= threshold).astype(np.float32)


class DenoisingDetectorConfig(BaseModel):
    """Model config for building a `DenoisingDetector` to serve.

    Mirrors the ``type: denoising_detector`` model-config YAML consumed by
    `sound_event_detection.serving.serve_denoising_detector` (via ``SED_MODEL_CONFIG``);
    `DenoisingDetector.from_config` consumes it to bundle the two wrapped
    clients. It is not a `DetectorConfig` — a denoising detector owns no
    weights; its config says where its wrapped clients' servers live.

    Attributes
    ----------
    detector : HttpClientConfig
        Http-client config for the wrapped detector server. Its ``url`` (base
        URL of a running detector server) is required; ``timeout`` (seconds),
        ``retries``, and ``auth`` are optional. A plain mapping is coerced to
        an `HttpClientConfig`, so unknown keys are rejected at construction.
    separator : dict
        Http-client config for the wrapped source-separation server. Must hold
        ``url`` (base URL of the running BirdMixIt server); may hold
        ``timeout`` (seconds, default ``300.0``) and ``binary`` (raw-binary
        wire, default ``True``). ``retries``/``auth`` are not supported.
    threshold : float
        Default focal probability threshold for the denoised waveform,
        reported to clients via ``GET /`` and read by callers such as
        large-scale inference.
    resampling_method : ResamplingMethod
        Method for the stem -> detector-rate resample (see
        `DenoisingDetector._resample`).
    """

    detector: HttpClientConfig
    separator: dict
    threshold: float = 0.5
    resampling_method: ResamplingMethod = "librosa_kaiser_best"


class DenoisingDetector:
    """Detect and denoise a focal species by bundling separation with detection.

    A model that wraps clients: bundles an injected detector client
    (`DetectorClient`) and separator client (`SourceSeparatorClient`), the way
    `SlidingWindowDetector` wraps a classifier client. It owns no weights —
    its model config names the servers behind the two wrapped clients — and is
    itself served by `sound_event_detection.serving.serve_denoising_detector`. Reads its
    rates and labels from the two clients, so it stays correct if either is
    swapped. Exposes the standard detector serving surface — `run` (framewise
    max over the separated stems) and `run_as_classifier` (clip-level max over
    the stems' scores) — so the denoising server answers the same contract as
    any detector server; focal-species denoising is an add-on.
    `separate_and_detect` exposes the `StemDetections` core for callers (e.g.
    large-scale inference, visualization) that want the raw stems, the
    per-stem predictions, and the focal-gated denoised waveform; the server
    exposes it as ``POST /separate_and_detect``.

    Attributes
    ----------
    sample_rate : int
        Input audio sample rate in Hz — the separator's rate. The denoised
        waveform is also returned at this rate.
    frame_rate : float
        Detector output frame_rate in Hz.
    labels : list[str]
        Detector output class labels.
    window_duration : float
        The wrapped detector's input window duration in seconds.
    server_config : dict
        Composed identity of the detector and separator behind this model,
        merged into the denoising server's ``GET /`` payload (its ``type``
        key is what `detector_client_from_config` auto-detects on).
    threshold : float
        Default focal probability threshold for the denoised waveform, read by
        callers such as large-scale inference.
    resampling_method : ResamplingMethod
        Method used to resample stems from the separator rate up to the
        detector rate (see `_resample`).
    """

    def __init__(
        self,
        detector: DetectorClient,
        separator: SourceSeparatorClient,
        threshold: float = 0.5,
        resampling_method: ResamplingMethod = "librosa_kaiser_best",
    ) -> None:
        """Bundle a detector client and a separator client into a denoising detector.

        Parameters
        ----------
        detector : DetectorClient
            Detector client (e.g. `ServedDetectorClient`) providing the full
            `DetectorClient` surface; its `run` / `run_as_classifier` window
            whole recordings internally.
        separator : SourceSeparatorClient
            Source-separation client exposing `sample_rate` and `separate_file`.
        threshold : float
            Default focal probability threshold for the denoised waveform.
        resampling_method : ResamplingMethod
            Method for the stem -> detector-rate resample. ``"librosa_kaiser_best"``
            (default) matches the rest of the org's audio pipeline; the
            ``"torchaudio_kaiser_*"`` options mimic librosa's kaiser resamplers
            but are much faster (a cheap win, since separation already dominates
            any resampling-quality difference).

        Raises
        ------
        ValueError
            If `resampling_method` is not a recognized method.
        """
        if resampling_method != "librosa_kaiser_best" and resampling_method not in _TORCHAUDIO_KAISER_PARAMS:
            valid = ["librosa_kaiser_best", *_TORCHAUDIO_KAISER_PARAMS]
            raise ValueError(f"Unknown resampling_method {resampling_method!r}; expected one of {valid}.")
        self._detector = detector
        self._separator = separator
        self.threshold = threshold
        self.resampling_method: ResamplingMethod = resampling_method
        self.sample_rate: int = separator.sample_rate
        self.frame_rate: float = detector.frame_rate
        self.labels: list[str] = list(detector.labels)
        self.window_duration: float = detector.window_duration
        self.server_config: dict = {
            "type": DENOISING_DETECTOR_TYPE,
            "detector": detector.server_config,
            "separator": {
                "sample_rate": separator.sample_rate,
                "n_stems": separator.n_stems,
                # None ("nan" over JSON) when the separator server exposes no
                # weight hash; a real digest flows through if it ever does.
                "weights_sha256": getattr(separator, "server_config", {}).get("weights_sha256"),
            },
            "threshold": threshold,
            "resampling_method": resampling_method,
        }

    @classmethod
    def from_config(cls, config: DenoisingDetectorConfig, labels: list[str] | None = None) -> "DenoisingDetector":
        """Build a `DenoisingDetector` by connecting its two wrapped clients.

        Connects the detector client through the shared
        `detector_client_from_config` dispatcher (so `config.detector` reaches
        its server exactly like every other http-client config) and a
        `BirdMixItClient` to the separator server named by `config.separator`,
        then wraps them. Mirrors the `from_config` pattern used by the
        detector models, even though a denoising detector loads no checkpoint.
        Both backend servers must be up: each client fetches its metadata at
        construction. If the separator construction fails, the already
        connected detector client is closed before the error propagates.

        Parameters
        ----------
        config : DenoisingDetectorConfig
            Denoising model config (``detector`` + ``separator`` http-client
            sub-configs, optional ``threshold`` / ``resampling_method``).
        labels : list[str] or None
            Optional label list checked against the inner detector server; the
            `DenoisingDetector`'s labels come from that server. When it does
            not match, `ValueError` is raised.

        Returns
        -------
        DenoisingDetector
            The bundled model, ready for `run` / `separate_and_detect`.

        Raises
        ------
        ValueError
            If `config.separator` carries unknown keys or lacks ``url``, or if
            `labels` is provided and differs from the detector server's labels.
            Invalid detector configs (unknown keys, missing ``url``) are
            rejected earlier, when `DenoisingDetectorConfig` is constructed.
        """
        from sound_event_detection.adapters.dispatch import detector_client_from_config
        from sound_event_detection.denoising.birdmixit_client import BirdMixItClient

        unknown = sorted(set(config.separator) - {"url", "timeout", "binary"})
        if unknown:
            raise ValueError(
                f"Unknown separator config key(s) {unknown}; expected only 'url' plus optional 'timeout' and 'binary'."
            )
        if "url" not in config.separator:
            raise ValueError("Separator config must contain 'url' (base URL of the running BirdMixIt server).")

        detector = detector_client_from_config(config.detector, labels=labels)
        try:
            separator = BirdMixItClient(
                url=config.separator["url"],
                timeout=config.separator.get("timeout", 300.0),
                binary=config.separator.get("binary", True),
            )
        except Exception:
            detector.close()
            raise
        return cls(
            detector=detector,
            separator=separator,
            threshold=config.threshold,
            resampling_method=config.resampling_method,
        )

    def separate_and_detect(
        self,
        audio: np.ndarray,
        batch_size: int = DEFAULT_DETECTOR_BATCH_SIZE,
        timings: MutableMapping[str, float] | None = None,
        **detector_kwargs: object,
    ) -> StemDetections:
        """Separate a whole recording, resample the stems, and detect over them.

        Parameters
        ----------
        audio : np.ndarray
            Mono waveform of shape ``(samples,)`` at `self.sample_rate`.
        batch_size : int
            Windows per detector forward pass, forwarded to `detector.run`. See
            `DEFAULT_DETECTOR_BATCH_SIZE`.
        timings : MutableMapping[str, float] or None
            Optional out-parameter for stage-level timing. When given, the wall
            time (`time.perf_counter` seconds) of each stage is recorded under
            keys ``"separate"`` (the separator call), ``"resample"`` (the
            stem -> detector-rate resample), and ``"detect"`` (the detector run).
            Recording is additive (each stage's time is added to any existing
            value), so a single mapping can accumulate across recordings. When
            ``None`` (default) no timing is taken and overhead is negligible.
        **detector_kwargs : object
            Extra keyword arguments forwarded verbatim to the detector client's
            `run` (e.g. ``overlap``), matching the `DetectorClient` run-surface.

        Returns
        -------
        StemDetections
            The whole-file stems and their per-stem framewise predictions.

        Raises
        ------
        ValueError
            If `audio` is not a 1-D array.
        """
        if audio.ndim != 1:
            raise ValueError(f"Expected 1D audio array [samples], got shape {audio.shape}")

        sep_sr = self._separator.sample_rate
        det_sr = self._detector.sample_rate

        start = time.perf_counter()
        stems = self._separator.separate_file(np.ascontiguousarray(audio, dtype=np.float32))  # (n_stems, samples)
        stems = np.ascontiguousarray(stems, dtype=np.float32)
        separate_done = time.perf_counter()
        stems_det = self._resample(stems, sep_sr, det_sr)  # (n_stems, det_samples)
        resample_done = time.perf_counter()
        output = self._detector.run(stems_det, batch_size=batch_size, **detector_kwargs)  # windows per forward pass
        detect_done = time.perf_counter()

        if timings is not None:
            timings["separate"] = timings.get("separate", 0.0) + (separate_done - start)
            timings["resample"] = timings.get("resample", 0.0) + (resample_done - separate_done)
            timings["detect"] = timings.get("detect", 0.0) + (detect_done - resample_done)

        return StemDetections(
            stems=stems,
            stem_preds=np.ascontiguousarray(output.predictions, dtype=np.float32),
            frame_rate=output.frame_rate,
            labels=self.labels,
            sample_rate=sep_sr,
        )

    def run(
        self, audio: np.ndarray, batch_size: int = DEFAULT_DETECTOR_BATCH_SIZE, **detector_kwargs: object
    ) -> DetectorOutput:
        """Run detection, combining per-stem predictions by per-frame max.

        This is the `DetectorClient` run-surface: eval and large-scale inference
        call it exactly as they would any detector client.

        Parameters
        ----------
        audio : np.ndarray
            Batched waveform of shape ``(batch, samples)`` at `self.sample_rate`.
            Rows are equal length, so their framewise outputs stack cleanly.
        batch_size : int
            Windows per detector forward pass, forwarded to `detector.run`. See
            `DEFAULT_DETECTOR_BATCH_SIZE`.
        **detector_kwargs : object
            Extra keyword arguments forwarded verbatim to the detector client's
            `run` (e.g. ``overlap``), so the denoising client accepts the same
            run knobs the eval harness passes to any served detector.

        Returns
        -------
        DetectorOutput
            Frame-level predictions of shape ``(batch, time, classes)``, the
            per-frame per-class max over the separated stems. Raises `ValueError`
            (via `_each`) if `audio` is not a 2-D array.
        """
        cores = self._each(audio, batch_size, **detector_kwargs)
        predictions = np.concatenate([core.combined().predictions for core in cores], axis=0)
        return DetectorOutput(
            predictions=np.ascontiguousarray(predictions, dtype=np.float32),
            frame_rate=self.frame_rate,
            class_names=list(self.labels),
        )

    def run_as_classifier(
        self, audio: np.ndarray, batch_size: int = DEFAULT_DETECTOR_BATCH_SIZE, **detector_kwargs: object
    ) -> MultiLabelClassifierOutput:
        """Run clip-level classification, combining per-stem scores by max.

        Each recording is separated into stems (one separator call per
        recording — its API is single-recording) and all stems are stacked
        into ``(batch * n_stems, samples)`` before the wrapped detector
        client's own `run_as_classifier` scores them (clip pooling is
        model-defined, e.g. tempered pooling for a `FrameDetector`), so a
        recording's score for a class is the max of its stems' scores.
        Stacking keeps the detector's forward passes full instead of sending
        one underfilled request per recording; the stacked rows are chunked
        `batch_size` at a time so a large batch never inflates a single
        detector request.

        Parameters
        ----------
        audio : np.ndarray
            Batched waveform of shape ``(batch, samples)`` at `self.sample_rate`.
        batch_size : int
            Windows per detector forward pass, forwarded to
            `detector.run_as_classifier` (and used as the stacked-stem rows
            per detector request). See `DEFAULT_DETECTOR_BATCH_SIZE`.
        **detector_kwargs : object
            Extra keyword arguments forwarded verbatim to the detector client's
            `run_as_classifier` (e.g. ``overlap``).

        Returns
        -------
        MultiLabelClassifierOutput
            Clip-level predictions of shape ``(batch, classes)``.

        Raises
        ------
        ValueError
            If `audio` is not a 2-D array.
        """
        if audio.ndim != 2:
            raise ValueError(f"Expected 2D audio array [batch, samples], got shape {audio.shape}")

        stems_per_row = []
        for row in audio:
            stems = self._separator.separate_file(np.ascontiguousarray(row, dtype=np.float32))  # (n_stems, samples)
            stems_per_row.append(self._resample(stems, self._separator.sample_rate, self._detector.sample_rate))

        stacked = np.concatenate(stems_per_row, axis=0)  # (batch * n_stems, det_samples)
        rows_per_call = max(int(batch_size), 1)
        stem_scores = np.concatenate(
            [
                self._detector.run_as_classifier(
                    stacked[start : start + rows_per_call], batch_size=batch_size, **detector_kwargs
                ).predictions
                for start in range(0, stacked.shape[0], rows_per_call)
            ],
            axis=0,
        )  # (batch * n_stems, classes)
        predictions = stem_scores.reshape(audio.shape[0], -1, stem_scores.shape[-1]).max(axis=1)
        return MultiLabelClassifierOutput(
            predictions=np.ascontiguousarray(predictions, dtype=np.float32),
            class_names=list(self.labels),
        )

    def describe_summary(self) -> dict:
        """Return a compact, serialisable summary of the composite's metadata.

        Returns
        -------
        dict
            ``{"n_labels", "sample_rate", "frame_rate", "window_duration",
            "n_stems"}`` — recorded in the results file to identify which model
            produced them.
        """
        return {
            "n_labels": len(self.labels),
            "sample_rate": self.sample_rate,
            "frame_rate": self.frame_rate,
            "window_duration": self.window_duration,
            "n_stems": self._separator.n_stems,
        }

    def close(self) -> None:
        """Close the wrapped detector client and, when it supports it, the separator client."""
        self._detector.close()
        close = getattr(self._separator, "close", None)
        if close is not None:
            close()

    def _each(
        self, audio: np.ndarray, batch_size: int = DEFAULT_DETECTOR_BATCH_SIZE, **detector_kwargs: object
    ) -> list[StemDetections]:
        """Run `separate_and_detect` on each row of a batched waveform.

        Parameters
        ----------
        audio : np.ndarray
            Batched waveform of shape ``(batch, samples)`` at `self.sample_rate`.
        batch_size : int
            Windows per detector forward pass, forwarded to each
            `separate_and_detect` call. See `DEFAULT_DETECTOR_BATCH_SIZE`.
        **detector_kwargs : object
            Extra keyword arguments forwarded verbatim to each
            `separate_and_detect` call (and on to the detector client's `run`).

        Returns
        -------
        list[StemDetections]
            One `StemDetections` per row, in order.

        Raises
        ------
        ValueError
            If `audio` is not a 2-D array.
        """
        if audio.ndim != 2:
            raise ValueError(f"Expected 2D audio array [batch, samples], got shape {audio.shape}")
        return [self.separate_and_detect(row, batch_size=batch_size, **detector_kwargs) for row in audio]

    def _resample(self, x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample along the last axis from `orig_sr` to `target_sr`.

        Parameters
        ----------
        x : np.ndarray
            Audio with samples along the last axis.
        orig_sr : int
            Source sample rate in Hz.
        target_sr : int
            Destination sample rate in Hz.

        Returns
        -------
        np.ndarray
            Resampled audio, dtype float32 (a copy of `x` when the rates match).

        Notes
        -----
        The resampler is selected by `resampling_method`. ``"librosa_kaiser_best"``
        uses librosa (matching the org's audio pipeline); the
        ``"torchaudio_kaiser_*"`` options use `torchaudio.functional.resample`
        with kwargs that mimic librosa's kaiser resamplers, imported lazily so
        the default path never requires torchaudio.
        """
        x = np.ascontiguousarray(x, dtype=np.float32)
        if orig_sr == target_sr:
            return x
        if self.resampling_method == "librosa_kaiser_best":
            resampled = librosa.resample(x, orig_sr=orig_sr, target_sr=target_sr, axis=-1, res_type="kaiser_best")
        else:
            import torch
            import torchaudio.functional as taf

            params = _TORCHAUDIO_KAISER_PARAMS[self.resampling_method]
            resampled = taf.resample(torch.from_numpy(x), orig_sr, target_sr, **params).numpy()
        return np.ascontiguousarray(resampled, dtype=np.float32)
