"""BEATs encoder wrapper for frame-level sound event detection."""

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Self

import torch
import torch.nn as nn
from avex import load_model
from pydantic import Field, computed_field

from esp_research.protocols.encoder import AudioEncoderConfig, AudioEncoderOutput


class BEATSEncoderConfig(AudioEncoderConfig):
    """Configuration for BEATSEncoder.

    Attributes
    ----------
    model_name : str
        Name of the BEATs model to load via avex. Only required when using
        `from_config` to load a model; defaults to ``""`` for direct construction.
    sample_rate : int
        Expected input audio sample rate in Hz.
    window_duration : float
        Expected input window duration in seconds.
    aggregation : str
        Aggregation strategy for frequency patches. One of ``"average"``,
        ``"all_frames"``, ``"concat"``.
    output_frame_rate : float
        Output frame_rate in Hz, computed from `sample_rate`, `window_duration`,
        and `aggregation`.
    """

    HIDDEN_DIM: ClassVar[int] = 768
    NUM_FREQ_PATCHES: ClassVar[int] = 8
    FBANK_WINDOW_SAMPLES: ClassVar[int] = 400
    FBANK_HOP_SAMPLES: ClassVar[int] = 160
    PATCH_SIZE: ClassVar[int] = 16

    model_name: str = ""
    sample_rate: int
    window_duration: float
    aggregation: str = Field(default="average")

    @computed_field
    @property
    def output_frame_rate(self) -> float:
        """Output frame_rate in Hz, accounting for aggregation strategy."""
        base = compute_beats_frame_rate(self.sample_rate, self.window_duration)
        return base * AGGREGATION_STRATEGIES[self.aggregation]["frame_rate_multiplier"]


def compute_beats_frame_rate(sample_rate: int, window_duration: float) -> float:
    """Compute the actual output frame_rate of BEATs encoder.

    BEATs produces a specific number of frames per window due to:
    1. fbank with sample_frequency=16000 (assumed), frame_shift=10ms
    2. 16x16 patching with floor division

    Note: BEATs assumes input is at 16kHz. When we feed audio at a different
    sample rate, it treats the samples as if they were at 16kHz, which affects
    the effective temporal resolution.

    Parameters
    ----------
    sample_rate : int
        Input audio sample rate in Hz.
    window_duration : float
        Window duration in seconds.

    Returns
    -------
    float
        Actual output frame_rate in Hz for the ``"average"`` aggregation strategy.
    """
    window_samples = int(window_duration * sample_rate)
    t_fbank = (window_samples - BEATSEncoderConfig.FBANK_WINDOW_SAMPLES) // BEATSEncoderConfig.FBANK_HOP_SAMPLES + 1
    t_patched = t_fbank // BEATSEncoderConfig.PATCH_SIZE
    return t_patched / window_duration


# ============= AGGREGATION FUNCTIONS =============


def _aggregate_average(embeddings: torch.Tensor) -> torch.Tensor:
    """Average BEATs patch embeddings over the frequency dimension.

    Args:
        embeddings: Raw patch embeddings [batch, total_patches, hidden_dim]

    Returns:
        Averaged embeddings [batch, time_patches, hidden_dim]
    """
    batch_size, _, hidden_dim = embeddings.shape
    return embeddings.reshape(batch_size, -1, BEATSEncoderConfig.NUM_FREQ_PATCHES, hidden_dim).mean(dim=2)


def _aggregate_all_frames(embeddings: torch.Tensor) -> torch.Tensor:
    """Treat all patches as individual time frames (no aggregation).

    Args:
        embeddings: Raw patch embeddings [batch, total_patches, hidden_dim]

    Returns:
        Unchanged embeddings [batch, total_patches, hidden_dim]
    """
    return embeddings


def _aggregate_concat(embeddings: torch.Tensor) -> torch.Tensor:
    """Concatenate frequency patches into a single embedding per time step.

    Args:
        embeddings: Raw patch embeddings [batch, total_patches, hidden_dim]

    Returns:
        Concatenated embeddings [batch, time_patches, hidden_dim * num_freq_patches]
    """
    batch_size, _, hidden_dim = embeddings.shape
    return embeddings.reshape(batch_size, -1, BEATSEncoderConfig.NUM_FREQ_PATCHES, hidden_dim).reshape(
        batch_size, -1, BEATSEncoderConfig.NUM_FREQ_PATCHES * hidden_dim
    )


AGGREGATION_STRATEGIES = {
    "average": {
        "fn": _aggregate_average,
        "frame_rate_multiplier": 1,
        "hidden_dim_multiplier": 1,
    },
    "all_frames": {
        "fn": _aggregate_all_frames,
        "frame_rate_multiplier": BEATSEncoderConfig.NUM_FREQ_PATCHES,
        "hidden_dim_multiplier": 1,
    },
    "concat": {
        "fn": _aggregate_concat,
        "frame_rate_multiplier": 1,
        "hidden_dim_multiplier": BEATSEncoderConfig.NUM_FREQ_PATCHES,
    },
}


class BEATSEncoder(nn.Module):
    """BEATs encoder that produces frame-level embeddings.

    Wraps a pretrained BEATs model and handles aggregation of patch embeddings
    across the frequency dimension. Implements the `AudioEncoder` protocol from
    `esp_research`.

    Attributes
    ----------
    output_dim : int
        Dimension of output embeddings (depends on aggregation strategy).
    output_frame_rate : float
        Output frame_rate in Hz (depends on aggregation strategy).
    sample_rate : int
        Expected input sample rate in Hz.
    window_duration : float
        Expected input window duration in seconds.
    """

    config_class = BEATSEncoderConfig

    def __init__(
        self,
        model: nn.Module,
        sample_rate: int,
        window_duration: float,
        aggregation: str = "average",
    ) -> None:
        """Initialize BEATSEncoder.

        Parameters
        ----------
        model : nn.Module
            Pretrained BEATs model from avex, loaded with ``return_features_only=True``.
        sample_rate : int
            Expected input audio sample rate in Hz.
        window_duration : float
            Expected input window duration in seconds.
        aggregation : str
            Aggregation strategy. One of ``"average"``, ``"all_frames"``, ``"concat"``.

        Raises
        ------
        ValueError
            If an unknown aggregation strategy is provided.
        """
        super().__init__()

        if aggregation not in AGGREGATION_STRATEGIES:
            raise ValueError(
                f"Unknown aggregation strategy '{aggregation}'. Must be one of: {list(AGGREGATION_STRATEGIES.keys())}"
            )

        self._model = model
        self._sample_rate = sample_rate
        self._window_duration = window_duration

        strategy = AGGREGATION_STRATEGIES[aggregation]
        self._aggregate_fn: Callable[[torch.Tensor], torch.Tensor] = strategy["fn"]
        self._output_dim = BEATSEncoderConfig.HIDDEN_DIM * strategy["hidden_dim_multiplier"]
        self._output_frame_rate = BEATSEncoderConfig(
            sample_rate=sample_rate, window_duration=window_duration, aggregation=aggregation
        ).output_frame_rate

    @classmethod
    def from_config(cls, config: BEATSEncoderConfig | Path | str) -> Self:
        """Instantiate from a config object or path to a JSON config file.

        Parameters
        ----------
        config : BEATSEncoderConfig | Path | str
            Config object or path to a JSON file containing the config.

        Returns
        -------
        BEATSEncoder
            Initialized encoder with pretrained BEATs weights loaded via avex.
        """
        if not isinstance(config, BEATSEncoderConfig):
            config = BEATSEncoderConfig.model_validate_json(Path(config).read_text())
        raw_model: nn.Module = load_model(config.model_name, return_features_only=True, device="cpu")  # type: ignore[assignment]
        return cls(
            model=raw_model,
            sample_rate=config.sample_rate,
            window_duration=config.window_duration,
            aggregation=config.aggregation,
        )

    @property
    def output_dim(self) -> int:
        """Dimension of output frame embeddings."""
        return self._output_dim

    @property
    def output_frame_rate(self) -> float:
        """Output frame_rate in Hz."""
        return self._output_frame_rate

    @property
    def sample_rate(self) -> int:
        """Expected input sample rate in Hz."""
        return self._sample_rate

    @property
    def window_duration(self) -> float:
        """Expected input window duration in seconds."""
        return self._window_duration

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode audio waveform to frame-level embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Audio waveform of shape ``(batch, samples)``.

        Returns
        -------
        torch.Tensor
            Frame embeddings of shape ``(batch, frames, output_dim)``.
        """
        embeddings = self._model(x, padding_mask=None)
        return self._aggregate_fn(embeddings)

    def encode(self, waveform: torch.Tensor, padding_mask: torch.Tensor) -> AudioEncoderOutput:
        """Encode audio waveform, satisfying the `AudioEncoder` protocol.

        Parameters
        ----------
        waveform : torch.Tensor
            Audio waveform of shape ``(batch, samples)``.
        padding_mask : torch.Tensor
            Input padding mask of shape ``(batch, samples)``. Not used internally
            since BEATs processes fixed-length windows, but accepted for protocol
            compatibility.

        Returns
        -------
        AudioEncoderOutput
            Encoder output with ``embeddings`` of shape ``(batch, frames, output_dim)``
            and an all-false ``padding_mask`` (no output frames are padded).
        """
        embeddings = self.forward(waveform)
        out_mask = torch.zeros(embeddings.shape[:2], dtype=torch.bool, device=embeddings.device)
        return AudioEncoderOutput(embeddings=embeddings, padding_mask=out_mask)

    def freeze(self) -> None:
        """Freeze all encoder parameters."""
        for param in self._model.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all encoder parameters."""
        for param in self._model.parameters():
            param.requires_grad = True
