"""Protocols and base classes for audio encoders."""

from dataclasses import dataclass
from typing import Generic, Protocol, Self, Type, TypeVar, runtime_checkable

import torch
from pydantic import BaseModel

from esp_research.types import AnyPathOrStr


class AudioEncoderConfig(BaseModel):
    """Base configuration class for ESP Audio encoders."""

    pass


# TODO: do we actually need AudoEncoderConfig? can use basemodel?

AudioEncoderConfigT = TypeVar("AudioEncoderConfigT", bound=AudioEncoderConfig)


@dataclass
class AudioEncoderOutput:
    """Output from an audio encoder."""

    embeddings: torch.Tensor  # (batch, time, embed_dim)
    padding_mask: torch.Tensor  # (batch, time) - True where padded


@runtime_checkable
class AudioEncoder(Protocol, Generic[AudioEncoderConfigT]):
    """Protocol for generic audio encoders."""

    # TODO: copy comment from TrainableModel
    config_class: Type[AudioEncoderConfigT]

    # TODO: make this common
    @classmethod
    def from_config(cls, config: AudioEncoderConfigT | AnyPathOrStr) -> Self: ...

    def encode(
        self,
        waveform: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> AudioEncoderOutput:
        """Encode raw audio waveform into embeddings.

        Parameters
        ----------
        waveform : torch.Tensor
            Raw audio waveform with shape (batch, time).
        padding_mask : torch.Tensor
            Boolean mask where True denotes padding elements. Shape: (batch, time).

        Returns
        -------
        AudioEncoderOutput
            Encoder output containing frame-level embeddings and padding mask.
        """
        ...
