"""Dual-stream encoder combining transformer backbone with CNN.

Based on the DASM (Detect Any Sound Model) architecture. Combines a semantic-rich
transformer encoder (e.g., BEATs) with a high-temporal-resolution CNN encoder,
projecting both to a common dimension and interpolating to a common frame_rate.

Architecture:
    encoder_1 (e.g., BEATs) ──► project ──► interpolate ──┐
                                                          ├──► weighted sum ──► LayerNorm ──► output
    encoder_2 (e.g., CNN)  ──► project ──► interpolate ──┘
"""

from pathlib import Path
from typing import Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field

from esp_research.protocols.encoder import AudioEncoder, AudioEncoderConfig, AudioEncoderOutput
from sound_event_detection.models.encoders.beats import BEATSEncoder, BEATSEncoderConfig
from sound_event_detection.models.encoders.cnn import CNNEncoder, CNNEncoderConfig


class DualStreamEncoderConfig(AudioEncoderConfig):
    """Configuration for DualStreamEncoder.

    Attributes
    ----------
    beats : BEATSEncoderConfig
        Configuration for the BEATs transformer encoder.
    cnn : CNNEncoderConfig
        Configuration for the CNN encoder.
    output_dim : int
        Dimension to project both encoders to.
    output_frame_rate : float | None
        Target output frame_rate in Hz. If ``None``, uses the higher of the two
        encoders' native frame_rates.
    freeze_on_warmup : tuple[bool, bool]
        Which encoders to freeze during warmup ``(freeze_beats, freeze_cnn)``.
    """

    beats: BEATSEncoderConfig
    cnn: CNNEncoderConfig
    output_dim: int
    output_frame_rate: float | None = Field(default=None)
    freeze_on_warmup: tuple[bool, bool] = Field(default=(True, False))


class DualStreamEncoder(nn.Module):
    """Dual-stream encoder that fuses two audio encoders.

    Combines embeddings from two encoders by:
    1. Projecting each to a common output dimension
    2. Interpolating each to a common output frame_rate
    3. Combining via learnable weighted sum
    4. Applying LayerNorm

    Implements the `AudioEncoder` protocol from `esp_research`.

    Attributes
    ----------
    encoder_1 : AudioEncoder
        First encoder (typically transformer backbone like BEATs).
    encoder_2 : AudioEncoder
        Second encoder (typically CNN for high temporal resolution).
    output_dim : int
        Dimension of output embeddings.
    output_frame_rate : float
        Output frame_rate in Hz.
    """

    config_class = DualStreamEncoderConfig

    def __init__(
        self,
        encoder_1: AudioEncoder,
        encoder_2: AudioEncoder,
        output_dim: int,
        output_frame_rate: float | None = None,
        merge_weight_init: float = 0.0,
        freeze_on_warmup: tuple[bool, bool] = (True, True),
    ) -> None:
        """Initialize DualStreamEncoder.

        Parameters
        ----------
        encoder_1 : AudioEncoder
            First encoder implementing the `AudioEncoder` protocol.
        encoder_2 : AudioEncoder
            Second encoder implementing the `AudioEncoder` protocol.
            Must have the same `sample_rate` and `window_duration` as `encoder_1`.
        output_dim : int
            Dimension to project both encoders to.
        output_frame_rate : float | None
            Target frame_rate in Hz. If ``None``, uses the higher of the two
            encoders' native frame_rates.
        merge_weight_init : float
            Initial value for learnable merge weight (before sigmoid).
        freeze_on_warmup : tuple[bool, bool]
            Which encoders to freeze during warmup. Projection layers and merge
            weight are always left trainable.

        Raises
        ------
        ValueError
            If encoders have different `sample_rate` or `window_duration`.
        """
        super().__init__()

        if encoder_1.sample_rate != encoder_2.sample_rate:
            raise ValueError(
                f"Encoders must have same sample_rate. Got {encoder_1.sample_rate} and {encoder_2.sample_rate}"
            )
        if encoder_1.window_duration != encoder_2.window_duration:
            raise ValueError(
                f"Encoders must have same window_duration. "
                f"Got {encoder_1.window_duration} and {encoder_2.window_duration}"
            )

        self.encoder_1 = encoder_1
        self.encoder_2 = encoder_2
        self._freeze_on_warmup = freeze_on_warmup
        self._output_dim = output_dim
        if output_frame_rate is None:
            output_frame_rate = max(encoder_1.output_frame_rate, encoder_2.output_frame_rate)
        self._output_frame_rate = output_frame_rate
        self._sample_rate = encoder_1.sample_rate
        self._window_duration = encoder_1.window_duration

        self.proj_1 = nn.Linear(encoder_1.output_dim, output_dim)
        self.proj_2 = nn.Linear(encoder_2.output_dim, output_dim)

        self.merge_weight = nn.Parameter(torch.tensor([merge_weight_init]))
        self.norm = nn.LayerNorm(output_dim)

    @classmethod
    def from_config(cls, config: DualStreamEncoderConfig | Path | str) -> Self:
        """Instantiate from a config object or path to a JSON config file.

        Parameters
        ----------
        config : DualStreamEncoderConfig | Path | str
            Config object or path to a JSON file containing the config.

        Returns
        -------
        DualStreamEncoder
            Initialized encoder with BEATs and CNN sub-encoders.
        """
        if not isinstance(config, DualStreamEncoderConfig):
            config = DualStreamEncoderConfig.model_validate_json(Path(config).read_text())
        beats = BEATSEncoder.from_config(config.beats)
        cnn = CNNEncoder.from_config(config.cnn)
        return cls(
            encoder_1=beats,
            encoder_2=cnn,
            output_dim=config.output_dim,
            output_frame_rate=config.output_frame_rate,
            freeze_on_warmup=config.freeze_on_warmup,
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

    def _interpolate_to_target(self, x: torch.Tensor) -> torch.Tensor:
        """Interpolate embeddings to target frame_rate.

        Parameters
        ----------
        x : torch.Tensor
            Embeddings of shape ``(batch, frames, dim)``.

        Returns
        -------
        torch.Tensor
            Interpolated embeddings of shape ``(batch, target_frames, dim)``.
        """
        target_frames = int(self._window_duration * self._output_frame_rate)
        if x.shape[1] == target_frames:
            return x
        x = x.transpose(1, 2)
        x = F.interpolate(x, size=target_frames, mode="nearest")
        return x.transpose(1, 2)

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
        mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
        emb_1 = self.encoder_1.encode(x, mask).embeddings  # [B, T1, D1]
        emb_2 = self.encoder_2.encode(x, mask).embeddings  # [B, T2, D2]

        emb_1 = self.proj_1(emb_1)
        emb_2 = self.proj_2(emb_2)

        emb_1 = self._interpolate_to_target(emb_1)
        emb_2 = self._interpolate_to_target(emb_2)

        w = torch.sigmoid(self.merge_weight)
        fused = (1.0 - w) * emb_1 + w * emb_2
        return self.norm(fused)

    def encode(self, waveform: torch.Tensor, padding_mask: torch.Tensor) -> AudioEncoderOutput:
        """Encode audio waveform, satisfying the `AudioEncoder` protocol.

        Parameters
        ----------
        waveform : torch.Tensor
            Audio waveform of shape ``(batch, samples)``.
        padding_mask : torch.Tensor
            Input padding mask. Not used internally (fixed-length windows), but
            accepted for protocol compatibility.

        Returns
        -------
        AudioEncoderOutput
            Encoder output with ``embeddings`` of shape ``(batch, frames, output_dim)``
            and an all-false ``padding_mask``.
        """
        embeddings = self.forward(waveform)
        out_mask = torch.zeros(embeddings.shape[:2], dtype=torch.bool, device=embeddings.device)
        return AudioEncoderOutput(embeddings=embeddings, padding_mask=out_mask)

    def freeze(self) -> None:
        """Freeze encoders according to `freeze_on_warmup`. Projections stay trainable."""
        if self._freeze_on_warmup[0]:
            self.encoder_1.freeze()
        if self._freeze_on_warmup[1]:
            self.encoder_2.freeze()

    def unfreeze(self) -> None:
        """Unfreeze both encoders."""
        self.encoder_1.unfreeze()
        self.encoder_2.unfreeze()

    def freeze_encoders(self) -> None:
        """Freeze only the encoder parameters, keep projections trainable."""
        self.encoder_1.freeze()
        self.encoder_2.freeze()

    def unfreeze_encoders(self) -> None:
        """Unfreeze encoder parameters."""
        self.encoder_1.unfreeze()
        self.encoder_2.unfreeze()
