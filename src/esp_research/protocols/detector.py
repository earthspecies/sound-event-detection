"""Detector protocol and output data structures for frame-level sound event detection models."""

from pathlib import Path
from typing import Annotated, Generic, Protocol, Self, Type, TypeVar, runtime_checkable

import numpy as np
from pydantic import BaseModel, Field, model_validator
from pydantic.functional_validators import PlainValidator

from .checkpointing import CheckpointLoadable
from .classifier import ClassNames, MultiLabelClassifierOutput


def _validate_predictions(v: object) -> np.ndarray:
    if not isinstance(v, np.ndarray):
        raise ValueError(f"Expected np.ndarray, got {type(v).__name__}")
    if v.ndim != 3:
        raise ValueError(f"Expected 3D predictions array [batch, time, classes], got shape {v.shape}")
    if not np.all((v >= 0) & (v <= 1)):
        raise ValueError("All probability values must be between 0.0 and 1.0")
    return v


PredictionsArray = Annotated[np.ndarray, PlainValidator(_validate_predictions)]


class DetectorOutput(BaseModel):
    """Output from a detector model containing probability scores.

    Attributes
    ----------
    predictions : np.ndarray
        Array of shape ``(batch, time, classes)`` where ``time`` is the number of frames
        and ``classes`` is the number of output classes. Values are probabilities in [0, 1].
    frame_rate : float
        Frame rate in Hz (frames per second).
    class_names : list[str]
        Names of the output classes, length equal to ``predictions.shape[2]``.
    """

    predictions: PredictionsArray
    frame_rate: float = Field(gt=0)
    class_names: ClassNames

    @model_validator(mode="after")
    def _check_class_names_length(self) -> "DetectorOutput":
        """Ensure ``class_names`` length matches the number of prediction classes.

        Returns
        -------
        DetectorOutput
            The validated model.

        Raises
        ------
        ValueError
            If ``len(class_names)`` differs from ``predictions.shape[2]``.
        """
        n_classes = self.predictions.shape[2]
        if len(self.class_names) != n_classes:
            raise ValueError(f"class_names has length {len(self.class_names)} but predictions has {n_classes} classes")
        return self


class DetectorConfig(BaseModel):
    """Common configuration for all detector models.

    Attributes
    ----------
    labels : list[str]
        Output class labels.
    sample_rate : int
        Expected input audio sample rate in Hz.
    """

    labels: list[str]
    sample_rate: int


DetectorConfigT = TypeVar("DetectorConfigT", bound=DetectorConfig)


@runtime_checkable
class Detector(CheckpointLoadable, Protocol, Generic[DetectorConfigT]):
    """Protocol for detector models that can run frame-level inference on audio.

    Extends `CheckpointLoadable` from `esp_research`, so implementations must
    also provide `from_checkpoint_dir`.

    Attributes
    ----------
    config_class : Type[DetectorConfigT]
        The Pydantic config class for this detector.
    labels : list[str]
        Output class labels.
    sample_rate : int
        Expected input audio sample rate in Hz.
    frame_rate : float
        Output frame_rate in frames per second.
    """

    config_class: Type[DetectorConfigT]
    labels: list[str]
    sample_rate: int
    frame_rate: float

    def run(
        self,
        audio: np.ndarray,
        *args: object,
        **kwargs: object,
    ) -> DetectorOutput:
        """Run inference on a batch of audio files.

        Parameters
        ----------
        audio : np.ndarray
            Batched waveform of shape ``(batch, n_samples)`` at `self.sample_rate`.
        *args : object
            Additional positional arguments specific to the implementing detector.
        **kwargs : object
            Additional keyword arguments specific to the implementing detector.

        Returns
        -------
        DetectorOutput
            Frame-level predictions with `predictions` of shape ``(batch, time, classes)``.
        """
        ...

    def run_as_classifier(
        self,
        audio: np.ndarray,
        *args: object,
        **kwargs: object,
    ) -> MultiLabelClassifierOutput:
        """Run inference and pool frame predictions to clip-level scores.

        Produces one probability per class per recording by pooling the
        frame-level predictions over time. The pooling strategy is defined by
        the implementing detector.

        Parameters
        ----------
        audio : np.ndarray
            Batched waveform of shape ``(batch, n_samples)`` at `self.sample_rate`.
        *args : object
            Additional positional arguments specific to the implementing detector.
        **kwargs : object
            Additional keyword arguments specific to the implementing detector.

        Returns
        -------
        MultiLabelClassifierOutput
            Clip-level predictions with `predictions` of shape ``(batch, classes)``.
        """
        ...

    @classmethod
    def from_checkpoint_dir(cls, checkpoint_dir: Path | str, config: DetectorConfig | Path | str) -> Self:
        """Load from a training results folder.

        Parameters
        ----------
        checkpoint_dir : Path | str
            Local path or cloud URI (``gs://...``, ``r2://...``) to the
            results folder (contains ``config.yaml`` and checkpoint).
        config : DetectorConfig | Path | str
            Training config as a `DetectorConfig` (or subclass) model, or
            path to ``config.yaml``.

        Returns
        -------
        Self
            Loaded model ready for inference.
        """
        ...

    @classmethod
    def from_config(cls, config: DetectorConfig | Path | str) -> Self:
        """Load from a config alone, without a training results folder.

        Parameters
        ----------
        config : DetectorConfig | Path | str
            Training config as a `DetectorConfig` (or subclass) model, or
            path to ``config.yaml``.

        Returns
        -------
        Self
            Loaded model ready for inference.
        """
        ...
