"""Lightweight CNN encoder for high temporal resolution sound event detection.

Based on the CNN architecture from DASM (Detect Any Sound Model). The CNN operates
on mel spectrograms and provides higher temporal resolution than transformer-based
encoders like BEATs.

Architecture:
    - 10 convolutional layers with ReLU activation
    - Filters: [16, 16, 32, 32, 64, 64, 128, 128, 256, 384]
    - Time pooling: 4x total (reduces temporal dimension by factor of 4)
    (for 10ms hop at 32kHz and 5s windows: ~25 Hz output)
    - Frequency pooling: 128x total (collapses 128 mel bins to 1)
    - Output: [batch, time_frames, 384]
"""

from pathlib import Path
from typing import Self

import torch
import torch.nn as nn
import torchaudio
from pydantic import Field

from esp_research.protocols.encoder import AudioEncoderConfig, AudioEncoderOutput

# Default CNN configuration matching DASM
DEFAULT_CNN_FILTERS = [16, 16, 32, 32, 64, 64, 128, 128, 256, 384]
DEFAULT_CNN_POOLING = [
    (2, 2),
    (1, 1),
    (2, 2),
    (1, 1),
    (1, 2),
    (1, 2),
    (1, 2),
    (1, 2),
    (1, 2),
    (1, 1),
]

DEFAULT_N_MELS = 128
DEFAULT_N_FFT = 1024
DEFAULT_WIN_LENGTH = 800  # 25ms at 32kHz
DEFAULT_HOP_LENGTH = 320  # 10ms at 32kHz


class CNNEncoderConfig(AudioEncoderConfig):
    """Configuration for CNNEncoder.

    Attributes
    ----------
    sample_rate : int
        Expected input audio sample rate in Hz.
    window_duration : float
        Expected input window duration in seconds.
    n_mels : int
        Number of mel frequency bins.
    n_fft : int
        FFT size for spectrogram computation.
    win_length : int
        STFT window length in samples.
    hop_length : int
        STFT hop length in samples.
    """

    sample_rate: int
    window_duration: float
    n_mels: int = Field(default=DEFAULT_N_MELS)
    n_fft: int = Field(default=DEFAULT_N_FFT)
    win_length: int = Field(default=DEFAULT_WIN_LENGTH)
    hop_length: int = Field(default=DEFAULT_HOP_LENGTH)


class CNNEncoder(nn.Module):
    """CNN encoder that produces frame-level embeddings from audio waveforms.

    Computes mel spectrogram from raw audio, then processes through a lightweight
    CNN with ReLU activation. Implements the `AudioEncoder` protocol from
    `esp_research`.

    The CNN collapses the frequency dimension while preserving (downsampled)
    temporal structure, making it suitable for sound event detection.

    Architecture based on DASM (Detect Any Sound Model):
        - Conv2d + BatchNorm + ReLU blocks
        - Pooling that reduces time by 4x and collapses frequency to 1
        - Output dimension is the last filter count (default 384)
    """

    config_class = CNNEncoderConfig

    def __init__(
        self,
        sample_rate: int,
        window_duration: float,
        n_mels: int = DEFAULT_N_MELS,
        n_fft: int = DEFAULT_N_FFT,
        win_length: int | None = None,
        hop_length: int | None = None,
        nb_filters: list[int] | None = None,
        pooling: list[tuple[int, int]] | None = None,
        dropout: float = 0.0,
    ) -> None:
        """Initialize CNNEncoder.

        Parameters
        ----------
        sample_rate : int
            Expected input audio sample rate in Hz.
        window_duration : float
            Expected input window duration in seconds.
        n_mels : int
            Number of mel frequency bins.
        n_fft : int
            FFT size for spectrogram computation.
        win_length : int | None
            Window length for STFT in samples. Defaults to 800.
        hop_length : int | None
            Hop length for STFT in samples. Defaults to 320.
        nb_filters : list[int] | None
            List of CNN filter counts per layer.
        pooling : list[tuple[int, int]] | None
            List of ``(time_pool, freq_pool)`` tuples per layer.
        dropout : float
            Dropout probability in CNN.

        Raises
        ------
        ValueError
            If `n_mels` is not divisible by the total frequency pooling factor.
        ValueError
            If pooling does not collapse the frequency dimension to 1.
        """
        super().__init__()

        self._sample_rate = sample_rate
        self._window_duration = window_duration
        self._n_mels = n_mels
        self._n_fft = n_fft
        self._win_length = win_length if win_length is not None else DEFAULT_WIN_LENGTH
        self._hop_length = hop_length if hop_length is not None else DEFAULT_HOP_LENGTH

        if nb_filters is None:
            nb_filters = DEFAULT_CNN_FILTERS
        if pooling is None:
            pooling = DEFAULT_CNN_POOLING

        assert len(nb_filters) == len(pooling), "nb_filters and pooling must have same length"

        self._nb_filters = nb_filters
        self._pooling = pooling

        freq_pooling_factor = 1
        for _, f_pool in pooling:
            freq_pooling_factor *= f_pool

        if n_mels % freq_pooling_factor != 0:
            raise ValueError(
                f"n_mels ({n_mels}) must be divisible by total frequency pooling factor ({freq_pooling_factor})"
            )

        final_freq_dim = n_mels // freq_pooling_factor
        if final_freq_dim != 1:
            raise ValueError(
                f"CNN pooling must collapse frequency to 1, but {n_mels} / {freq_pooling_factor} = {final_freq_dim}. "
                f"Adjust pooling or n_mels."
            )

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=self._win_length,
            hop_length=self._hop_length,
            n_mels=n_mels,
            power=2.0,
        )

        self.cnn = self._build_cnn(nb_filters, pooling, dropout)
        self._output_frame_rate = self._compute_output_frame_rate()

    @classmethod
    def from_config(cls, config: CNNEncoderConfig | Path | str) -> Self:
        """Instantiate from a config object or path to a JSON config file.

        Parameters
        ----------
        config : CNNEncoderConfig | Path | str
            Config object or path to a JSON file containing the config.

        Returns
        -------
        CNNEncoder
            Initialized encoder.
        """
        if not isinstance(config, CNNEncoderConfig):
            config = CNNEncoderConfig.model_validate_json(Path(config).read_text())
        return cls(
            sample_rate=config.sample_rate,
            window_duration=config.window_duration,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
        )

    def _build_cnn(
        self,
        nb_filters: list[int],
        pooling: list[tuple[int, int]],
        dropout: float,
    ) -> nn.Sequential:
        """Build the CNN layers.

        Each block: Conv2d -> BatchNorm -> ReLU -> [Dropout] -> [AvgPool2d]

        Returns
        -------
        nn.Sequential
            Sequential CNN module with specified architecture.
        """
        layers: list[nn.Module] = []
        in_channels = 1

        for out_channels, pool in zip(nb_filters, pooling, strict=False):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            if pool != (1, 1):
                layers.append(nn.AvgPool2d(pool))

            in_channels = out_channels

        return nn.Sequential(*layers)

    def _compute_output_frame_rate(self) -> float:
        """Compute actual output frame_rate based on mel and CNN parameters.

        Returns
        -------
        float
            Output frame_rate in frames per second.
        """
        window_samples = int(self._window_duration * self._sample_rate)
        mel_frames = (window_samples // self._hop_length) + 1

        time_pooling = 1
        for t_pool, _ in self._pooling:
            time_pooling *= t_pool

        output_frames = mel_frames // time_pooling
        return output_frames / self._window_duration

    @property
    def output_dim(self) -> int:
        """Dimension of output frame embeddings."""
        return self._nb_filters[-1]

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
        mel = self.mel_spectrogram(x)
        mel = torch.log(mel + 1e-7)
        mel = mel.unsqueeze(1).transpose(2, 3)
        x = self.cnn(mel)
        x = x.squeeze(-1).permute(0, 2, 1)
        return x

    def encode(self, waveform: torch.Tensor, padding_mask: torch.Tensor) -> AudioEncoderOutput:
        """Encode audio waveform, satisfying the `AudioEncoder` protocol.

        Parameters
        ----------
        waveform : torch.Tensor
            Audio waveform of shape ``(batch, samples)``.
        padding_mask : torch.Tensor
            Input padding mask of shape ``(batch, samples)``. Not used internally
            since the CNN processes fixed-length windows, but accepted for protocol
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
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
