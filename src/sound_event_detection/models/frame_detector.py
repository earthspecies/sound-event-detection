"""Frame-based sound event detection model.

Combines an audio encoder with a linear classification head for frame-level
sound event detection. Works with any encoder implementing the `AudioEncoder`
protocol from `esp_research`.
"""

import math
from pathlib import Path
from typing import Self

import numpy as np
import torch
import torch.nn as nn
import yaml
from alp_data.io import anypath, exists, filesystem_from_path
from avex import load_model
from pydantic import ConfigDict

from esp_research.mixins import HfHubMixin
from esp_research.protocols.classifier import MultiLabelClassifierOutput
from esp_research.protocols.detector import DetectorConfig, DetectorOutput
from esp_research.protocols.encoder import AudioEncoder
from esp_research.types import AnyPathOrStr
from sound_event_detection.models.encoders import BEATSEncoder, CNNEncoder, DualStreamEncoder
from sound_event_detection.utils.io_utils import load_labels, load_state_dict_verbose, open_anypath
from sound_event_detection.utils.pooling import tempered_pooling


class FrameDetectorConfig(DetectorConfig):
    """Full training config for a `FrameDetector`, used when loading from a checkpoint.

    Attributes
    ----------
    target_duration : float
        Clip duration in seconds fed to the encoder.
    encoder : dict
        Encoder configuration (passed through to `create_detector_from_config`).
    classifier : dict
        Classifier head configuration.
    pooling_temperature : float
        Temperature for the tempered pooling used by `run_as_classifier`
        (1.0 = linear softmax pooling, higher approaches max pooling).
    """

    model_config = ConfigDict(extra="allow")

    target_duration: float
    encoder: dict
    classifier: dict = {}
    pooling_temperature: float = 1.0


class FrameDetector(nn.Module, HfHubMixin):
    """Frame-level sound event detection model.

    Combines an audio encoder with a linear classifier head. Works with any
    encoder that implements the `AudioEncoder` protocol from `esp_research`
    (``output_dim``, ``output_frame_rate``, ``sample_rate``, ``window_duration``
    properties, and ``encode()`` method).

    Mixes in `HfHubMixin` to gain `push_to_hf_hub` and `from_hf_hub`, which round-trip
    through `save_to_checkpoint_dir` / `from_checkpoint_dir`.

    Attributes
    ----------
    encoder : AudioEncoder
        Audio encoder that produces frame-level embeddings.
    classifier : nn.Module
        Linear layer mapping embeddings to class logits.
    labels : list[str]
        List of class labels corresponding to output indices.
    """

    config_class = FrameDetectorConfig

    def __init__(
        self,
        encoder: AudioEncoder,
        labels: list[str],
        head: str = "linear",
        hidden_dim: int | None = None,
        pooling_temperature: float = 1.0,
    ) -> None:
        """Initialize frame-based SED model.

        Parameters
        ----------
        encoder : AudioEncoder
            Audio encoder implementing the `AudioEncoder` protocol. Must have
            ``output_dim``, ``output_frame_rate``, ``sample_rate``, ``window_duration``
            properties and an ``encode()`` method.
        labels : list[str]
            List of class labels for classification outputs.
        head : str
            Classifier head type. One of ``"linear"`` or ``"mlp"``.
        hidden_dim : int | None
            Hidden dimension for MLP head. Defaults to ``encoder.output_dim``.
            Ignored when ``head="linear"``.
        pooling_temperature : float
            Temperature for the tempered pooling used by `run_as_classifier`
            (1.0 = linear softmax pooling, higher approaches max pooling).

        Raises
        ------
        ValueError
            If `head` is not ``"linear"`` or ``"mlp"``.
        """
        super().__init__()

        self.encoder = encoder
        self.labels = labels
        self.pooling_temperature = pooling_temperature

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        if head == "linear":
            self.classifier: nn.Module = nn.Linear(encoder.output_dim, len(labels))
            self.classifier.bias.data.fill_(bias_value)  # type: ignore[union-attr]
        elif head == "mlp":
            hdim = hidden_dim if hidden_dim is not None else encoder.output_dim
            self.classifier = nn.Sequential(
                nn.Linear(encoder.output_dim, hdim),
                nn.ReLU(),
                nn.Linear(hdim, len(labels)),
            )
            self.classifier[-1].bias.data.fill_(bias_value)
        else:
            raise ValueError(f"Unknown head type: {head!r}. Expected 'linear' or 'mlp'.")

    @property
    def frame_rate(self) -> float:
        """Output frame_rate in Hz."""
        return self.encoder.output_frame_rate

    @property
    def sample_rate(self) -> int:
        """Expected audio sample rate in Hz."""
        return self.encoder.sample_rate

    @property
    def window_duration(self) -> float:
        """Expected input window duration in seconds."""
        return self.encoder.window_duration

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and classifier.

        Parameters
        ----------
        x : torch.Tensor
            Raw audio waveform of shape ``(batch, samples)``.

        Returns
        -------
        torch.Tensor
            Frame-level logits of shape ``(batch, time_frames, num_classes)``.
        """
        mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
        encoder_out = self.encoder.encode(x, mask)
        return self.classifier(encoder_out.embeddings)

    def save_to_checkpoint_dir(self, checkpoint_dir: AnyPathOrStr) -> None:
        """Save model weights and labels to a checkpoint subdirectory.

        Writes the files that `from_checkpoint_dir` reads back: ``best_model.pt``
        (the full model `state_dict`) and ``labels.txt`` (one label per line). This
        implements the `CheckpointSaveable` protocol, which is what
        `push_to_hf_hub` (from `HfHubMixin`) uses to populate the ``checkpoint/``
        subdirectory of a HuggingFace Hub repository.

        Parameters
        ----------
        checkpoint_dir : AnyPathOrStr
            Local path or cloud URI (``gs://...``, ``r2://...``) to this object's
            checkpoint subdirectory. Created if it does not exist.
        """
        checkpoint_dir = anypath(checkpoint_dir)
        if isinstance(checkpoint_dir, Path):
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        fs = filesystem_from_path(checkpoint_dir)

        with fs.open(str(checkpoint_dir / "best_model.pt"), "wb") as f:
            torch.save(self.state_dict(), f)

        with fs.open(str(checkpoint_dir / "labels.txt"), "w") as f:
            f.write("\n".join(self.labels) + "\n")

    @classmethod
    def from_checkpoint_dir(cls, checkpoint_dir: Path | str, config: DetectorConfig | Path | str) -> Self:
        """Load a trained FrameDetector from a local or cloud results folder.

        The results folder must contain a ``config.yaml`` (training config),
        a model checkpoint (``best_model.pt`` or ``checkpoints/best_model.pt``),
        and a ``labels.txt``.

        Parameters
        ----------
        checkpoint_dir : Path | str
            Local path or cloud URI (``gs://...``, ``r2://...``) to the
            results folder produced by training.
        config : DetectorConfig | Path | str
            Either a `DetectorConfig` (or subclass) whose ``model_dump()``
            yields the training config dict, or a path to a ``config.yaml``
            file.

        Returns
        -------
        FrameDetector
            Model loaded from checkpoint, in eval mode.

        Raises
        ------
        FileNotFoundError
            If the checkpoint file or ``labels.txt`` cannot be found.
        """
        base = anypath(str(checkpoint_dir))

        if isinstance(config, DetectorConfig):
            cfg: dict = config.model_dump()
        else:
            with open_anypath(str(config)) as f:
                cfg = yaml.safe_load(f)

        checkpoint_path = base / "best_model.pt"
        if not exists(checkpoint_path):
            checkpoint_path = base / "checkpoints" / "best_model.pt"
        if not exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found in {checkpoint_dir}")

        labels_path = base / "labels.txt"
        if not exists(labels_path):
            raise FileNotFoundError(f"labels.txt not found in {checkpoint_dir}")
        labels = load_labels(str(labels_path))

        model = create_detector_from_config(cfg, labels=labels)
        model.pooling_temperature = float(cfg.get("pooling_temperature", 1.0))
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        load_state_dict_verbose(model, state_dict)
        return model.eval()  # type: ignore[return-value]

    @classmethod
    def from_config(cls, config: DetectorConfig | Path | str) -> Self:
        """Build a FrameDetector from a config alone, without a checkpoint.

        Unlike `from_checkpoint_dir`, this does not load trained weights or a
        ``labels.txt``: the encoder is built from the config (typically with
        pretrained encoder weights referenced by the config) and the classifier
        head is freshly initialized. Labels are taken from the config's
        ``labels`` field.

        Parameters
        ----------
        config : DetectorConfig | Path | str
            Either a `DetectorConfig` (or subclass) whose ``model_dump()``
            yields the training config dict, or a path to a ``config.yaml``
            file.

        Returns
        -------
        FrameDetector
            Model built from config, in eval mode.
        """
        if isinstance(config, DetectorConfig):
            cfg: dict = config.model_dump()
        else:
            with open_anypath(str(config)) as f:
                cfg = yaml.safe_load(f)

        labels = cfg["labels"]
        model = create_detector_from_config(cfg, labels=labels)
        model.pooling_temperature = float(cfg.get("pooling_temperature", 1.0))
        return model.eval()  # type: ignore[return-value]

    def run(
        self,
        audio: np.ndarray,
        batch_size: int = 32,
        device: str | torch.device | None = None,
        overlap: float | None = None,
    ) -> DetectorOutput:
        """Run inference on a batch of variable-length audio files.

        Splits each recording into windows, processes them in batches, applies
        sigmoid to get probabilities, and concatenates the frame outputs. All
        recordings in the batch must share the same number of samples.

        When overlap is specified, uses overlapping windows and only keeps
        the center frames from each window to reduce edge artifacts.

        Parameters
        ----------
        audio : np.ndarray
            Batched audio waveform at ``self.sample_rate``, shape ``(batch, samples)``.
        batch_size : int
            Number of windows to process at once.
        device : str | torch.device | None
            Device to run inference on. If ``None``, uses the model's device.
        overlap : float | None
            Fraction of window to overlap (``0.0`` to ``<1.0``). ``None`` means
            no overlap.

        Returns
        -------
        DetectorOutput
            Frame-level predictions with `predictions` of shape ``(batch, time, classes)``.

        Raises
        ------
        ValueError
            If audio is not 2-D.
        ValueError
            If overlap is not in ``[0.0, 1.0)``.
        """
        if audio.ndim != 2:
            raise ValueError(f"Expected 2D audio array [batch, samples], got shape {audio.shape}")

        if device is None:
            device = next(self.parameters()).device

        window_samples = int(self.window_duration * self.sample_rate)
        frames_per_window = int(self.window_duration * self.frame_rate)

        if overlap is None or overlap == 0.0:
            left_margin_frames = 0
            right_margin_frames = 0
        else:
            if not (0.0 <= overlap < 1.0):
                raise ValueError(f"overlap must be in [0.0, 1.0), got {overlap}")
            total_margin = int(round(overlap * frames_per_window))
            total_margin = max(2, min(total_margin, frames_per_window - 1))
            left_margin_frames = total_margin // 2
            right_margin_frames = total_margin - left_margin_frames

        all_probs = self._run_windowed_inference(
            audio, window_samples, frames_per_window, left_margin_frames, right_margin_frames, batch_size, device
        )

        return DetectorOutput(
            predictions=all_probs,
            frame_rate=self.frame_rate,
            class_names=self.labels,
        )

    def run_as_classifier(
        self,
        audio: np.ndarray,
        batch_size: int = 32,
        device: str | torch.device | None = None,
        overlap: float | None = None,
    ) -> MultiLabelClassifierOutput:
        """Run inference and pool frame predictions to clip-level scores.

        Runs frame-level inference via `run` and pools the frame predictions
        over time with tempered pooling at `self.pooling_temperature`.

        Parameters
        ----------
        audio : np.ndarray
            Batched audio waveform at ``self.sample_rate``, shape ``(batch, samples)``.
        batch_size : int
            Number of windows to process at once.
        device : str | torch.device | None
            Device to run inference on. If ``None``, uses the model's device.
        overlap : float | None
            Fraction of window to overlap (``0.0`` to ``<1.0``). ``None`` means
            no overlap.

        Returns
        -------
        MultiLabelClassifierOutput
            Clip-level predictions with `predictions` of shape ``(batch, classes)``.
        """
        output = self.run(audio, batch_size=batch_size, device=device, overlap=overlap)
        frame_probs = torch.from_numpy(output.predictions).float()  # (B, T, C)
        clip_probs = tempered_pooling(frame_probs, temperature=self.pooling_temperature, dim=1)  # (B, C)
        return MultiLabelClassifierOutput(predictions=clip_probs.numpy(), class_names=self.labels)

    def _run_windowed_inference(
        self,
        audio: np.ndarray,
        window_samples: int,
        frames_per_window: int,
        left_margin_frames: int,
        right_margin_frames: int,
        batch_size: int,
        device: str | torch.device,
    ) -> np.ndarray:
        """Run windowed inference with optional overlap over a batch of recordings.

        All recordings in the batch are assumed to have the same number of samples,
        so the windowing and reassembly are identical across the batch axis.

        Parameters
        ----------
        audio : np.ndarray
            Batched audio waveform of shape ``(batch, samples)``.
        window_samples : int
            Number of samples per window.
        frames_per_window : int
            Number of output frames per window.
        left_margin_frames : int
            Frames to discard from the left edge of middle windows.
        right_margin_frames : int
            Frames to discard from the right edge of middle windows.
        batch_size : int
            Number of windows per forward pass (counted across the whole batch).
        device : str | torch.device
            Device to run inference on.

        Returns
        -------
        np.ndarray
            Frame probabilities of shape ``(batch, total_frames, num_classes)``.
        """
        batch = audio.shape[0]
        total_margin = left_margin_frames + right_margin_frames
        keep_frames = frames_per_window - total_margin
        hop_samples = (
            window_samples if total_margin == 0 else int(round(window_samples * keep_frames / frames_per_window))
        )
        audio_len = audio.shape[1]

        if audio_len <= window_samples:
            n_windows = 1
        else:
            n_windows = 1 + int(np.ceil((audio_len - window_samples) / hop_samples))

        required_len = (n_windows - 1) * hop_samples + window_samples
        n_pad_samples = max(0, required_len - audio_len)
        if n_pad_samples > 0:
            audio = np.pad(audio, ((0, 0), (0, n_pad_samples)), mode="constant")

        if total_margin == 0:
            windows = audio.reshape(batch, n_windows, window_samples)
        else:
            windows = np.lib.stride_tricks.sliding_window_view(audio, window_samples, axis=1)[:, ::hop_samples]
            windows = windows[:, :n_windows]

        # Flatten the batch and window axes so a single forward pass spans recordings.
        flat_windows = windows.reshape(batch * n_windows, window_samples)

        all_window_probs = []
        self.eval()
        with torch.no_grad():
            for i in range(0, flat_windows.shape[0], batch_size):
                batch_windows = flat_windows[i : i + batch_size]
                if not batch_windows.flags["C_CONTIGUOUS"] or not batch_windows.flags["WRITEABLE"]:
                    batch_windows = batch_windows.copy()
                forward_batch = torch.from_numpy(batch_windows).float().to(device)
                logits = self.forward(forward_batch)
                probs = torch.sigmoid(logits)
                all_window_probs.append(probs.cpu().numpy())

        # (batch, n_windows, frames_per_window, num_classes)
        all_window_probs_arr = np.concatenate(all_window_probs, axis=0).reshape(batch, n_windows, frames_per_window, -1)

        if total_margin == 0:
            all_probs = all_window_probs_arr.reshape(batch, n_windows * frames_per_window, -1)
        else:
            kept_frames_list = []
            for i in range(n_windows):
                is_first = i == 0
                is_last = i == n_windows - 1
                start = 0 if is_first else left_margin_frames
                end = frames_per_window if is_last else frames_per_window - right_margin_frames
                kept_frames_list.append(all_window_probs_arr[:, i, start:end])
            all_probs = np.concatenate(kept_frames_list, axis=1)

        if n_pad_samples > 0:
            n_pad_frames = int(np.ceil(n_pad_samples / self.sample_rate * self.frame_rate))
            if 0 < n_pad_frames < all_probs.shape[1]:
                all_probs = all_probs[:, :-n_pad_frames]

        return all_probs

    def freeze_encoder(self) -> None:
        """Freeze encoder parameters."""
        self.encoder.freeze()

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters for fine-tuning."""
        self.encoder.unfreeze()


def create_beats_detector(
    encoder_name: str,
    labels: list[str],
    sample_rate: int,
    window_duration: float = 5.0,
    aggregation: str = "average",
    head: str = "linear",
    hidden_dim: int | None = None,
) -> FrameDetector:
    """Create a FrameDetector model with BEATs encoder.

    Parameters
    ----------
    encoder_name : str
        Name of the encoder model to load via avex.
    labels : list[str]
        List of label strings for classification outputs.
    sample_rate : int
        Audio sample rate in Hz.
    window_duration : float
        Duration of input windows in seconds.
    aggregation : str
        Aggregation strategy. One of ``"average"``, ``"all_frames"``, ``"concat"``.
    head : str
        Classifier head type. One of ``"linear"`` or ``"mlp"``.
    hidden_dim : int | None
        Hidden dimension for MLP head.

    Returns
    -------
    FrameDetector
        Configured FrameDetector with BEATSEncoder.

    Raises
    ------
    ValueError
        If `aggregation` is not a recognised strategy.
    RuntimeError
        If `encoder_name` requires a checkpoint but none was registered.
    """
    from sound_event_detection.models.encoders.beats import AGGREGATION_STRATEGIES

    if aggregation not in AGGREGATION_STRATEGIES:
        raise ValueError(
            f"Unknown aggregation strategy '{aggregation}'. Must be one of: {list(AGGREGATION_STRATEGIES.keys())}"
        )

    import sound_event_detection.models.avex_model_registration as _avex_reg  # registers beats_ssl

    encoder_ckpt = _avex_reg._REGISTERED_CHECKPOINT_PATHS.get(encoder_name)
    if encoder_name in _avex_reg._CHECKPOINT_REQUIRED and encoder_ckpt is None:
        raise RuntimeError(
            f"No checkpoint path registered for encoder '{encoder_name}'. "
            "Training cannot proceed — the encoder would have random weights. "
            "Check avex installation or avex_model_registration.py."
        )
    raw_model: nn.Module = load_model(  # type: ignore[assignment]
        encoder_name, return_features_only=True, device="cpu", checkpoint_path=encoder_ckpt
    )

    encoder = BEATSEncoder(
        model=raw_model,
        sample_rate=sample_rate,
        window_duration=window_duration,
        aggregation=aggregation,
    )

    return FrameDetector(encoder=encoder, labels=labels, head=head, hidden_dim=hidden_dim)


def create_dual_stream_detector(
    backbone_name: str,
    labels: list[str],
    sample_rate: int,
    window_duration: float,
    output_dim: int,
    output_frame_rate: float | None = None,
    aggregation: str = "concat",
    cnn_config: dict | None = None,
    head: str = "linear",
    hidden_dim: int | None = None,
) -> FrameDetector:
    """Create a FrameDetector with DualStreamEncoder (BEATs + CNN).

    Parameters
    ----------
    backbone_name : str
        Name of the BEATs model to load via avex.
    labels : list[str]
        List of label strings for classification outputs.
    sample_rate : int
        Audio sample rate in Hz.
    window_duration : float
        Duration of input windows in seconds.
    output_dim : int
        Output dimension for the fused embeddings.
    output_frame_rate : float | None
        Target output frame_rate in Hz. If ``None``, uses the higher of the two
        encoders' native frame_rates.
    aggregation : str
        BEATs aggregation strategy.
    cnn_config : dict | None
        Optional dict with CNN parameters (``hop_length``, ``win_length``,
        ``n_fft``, ``n_mels``).
    head : str
        Classifier head type.
    hidden_dim : int | None
        Hidden dimension for MLP head.

    Returns
    -------
    FrameDetector
        Configured model with DualStreamEncoder.
    """
    raw_model: nn.Module = load_model(backbone_name, return_features_only=True, device="cpu")  # type: ignore[assignment]
    beats_encoder = BEATSEncoder(
        model=raw_model,
        sample_rate=sample_rate,
        window_duration=window_duration,
        aggregation=aggregation,
    )

    cnn_kwargs: dict = {"sample_rate": sample_rate, "window_duration": window_duration}
    if cnn_config:
        cnn_kwargs.update(cnn_config)
    cnn_encoder = CNNEncoder(**cnn_kwargs)

    encoder = DualStreamEncoder(
        encoder_1=beats_encoder,
        encoder_2=cnn_encoder,
        output_dim=output_dim,
        output_frame_rate=output_frame_rate,
        freeze_on_warmup=(True, False),
    )

    return FrameDetector(encoder=encoder, labels=labels, head=head, hidden_dim=hidden_dim)


def create_detector_from_config(
    config: dict,
    labels: list[str] | None,
) -> FrameDetector:
    """Create detector based on ``config['encoder']['type']``.

    Parameters
    ----------
    config : dict
        Training config dict. Must have an ``'encoder'`` section with a ``'type'``
        key. Supported types: ``"beats"``, ``"dual_stream"``,
        ``"audioprotopnet_frame"``.
    labels : list[str] | None
        List of label strings. Required for all encoder types.

    Returns
    -------
    FrameDetector
        Configured model based on encoder type.

    Raises
    ------
    ValueError
        If the encoder type is unknown.
    ValueError
        If `labels` is ``None`` and the encoder type requires caller-supplied labels.
    """
    encoder_config = config["encoder"]
    encoder_type = encoder_config["type"]

    if labels is None:
        raise ValueError(f"labels is required for encoder type '{encoder_type}'")

    classifier_config = config.get("classifier", {})
    head = classifier_config.get("head", "linear")
    hidden_dim = classifier_config.get("hidden_dim")

    if encoder_type == "beats":
        return create_beats_detector(
            encoder_name=encoder_config["name"],
            labels=labels,
            sample_rate=config["sample_rate"],
            window_duration=config["target_duration"],
            aggregation=encoder_config.get("aggregation", "average"),
            head=head,
            hidden_dim=hidden_dim,
        )
    elif encoder_type == "dual_stream":
        return create_dual_stream_detector(
            backbone_name=encoder_config["name"],
            labels=labels,
            sample_rate=config["sample_rate"],
            window_duration=config["target_duration"],
            output_dim=encoder_config["output_dim"],
            output_frame_rate=encoder_config.get("output_frame_rate"),
            aggregation=encoder_config.get("aggregation", "concat"),
            cnn_config=encoder_config.get("cnn"),
            head=head,
            hidden_dim=hidden_dim,
        )
    elif encoder_type == "audioprotopnet_frame":
        from sound_event_detection.models.audioprotopnet_frame_detector import (
            create_audioprotopnet_frame_detector,
        )

        return create_audioprotopnet_frame_detector(
            num_classes=len(labels),
            num_prototypes=encoder_config["num_prototypes"],
            labels=labels,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
