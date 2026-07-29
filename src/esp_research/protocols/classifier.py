"""Output data structures for multi-label clip-level classification."""

from collections.abc import Iterable
from typing import Annotated

import numpy as np
from pydantic import BaseModel, model_validator
from pydantic.functional_validators import PlainValidator


def _validate_clip_predictions(v: object) -> np.ndarray:
    if not isinstance(v, np.ndarray):
        raise ValueError(f"Expected np.ndarray, got {type(v).__name__}")
    if v.ndim != 2:
        raise ValueError(f"Expected 2D predictions array [batch, classes], got shape {v.shape}")
    if not np.all((v >= 0) & (v <= 1)):
        raise ValueError("All probability values must be between 0.0 and 1.0")
    return v


def _validate_class_names(v: object) -> list[str]:
    if isinstance(v, str) or not isinstance(v, Iterable):
        raise ValueError(f"Expected an iterable of strings, got {type(v).__name__}")
    names = list(v)
    if not all(isinstance(name, str) for name in names):
        raise ValueError("All class names must be strings")
    return names


ClipPredictionsArray = Annotated[np.ndarray, PlainValidator(_validate_clip_predictions)]
ClassNames = Annotated[list[str], PlainValidator(_validate_class_names)]


class MultiLabelClassifierOutput(BaseModel):
    """Output from a multi-label clip-level classifier containing probability scores.

    Predictions are a single score per class per clip, with no time dimension:
    each class is scored independently, so the per-clip probabilities need not
    sum to one (multi-label rather than single-label classification).

    Attributes
    ----------
    predictions : np.ndarray
        Array of shape ``(batch, classes)`` where ``classes`` is the number of
        output classes. Values are probabilities in [0, 1].
    class_names : list[str]
        Names of the output classes, length equal to ``predictions.shape[1]``.
    """

    predictions: ClipPredictionsArray
    class_names: ClassNames

    @model_validator(mode="after")
    def _check_class_names_length(self) -> "MultiLabelClassifierOutput":
        """Ensure ``class_names`` length matches the number of prediction classes.

        Returns
        -------
        MultiLabelClassifierOutput
            The validated model.

        Raises
        ------
        ValueError
            If ``len(class_names)`` differs from ``predictions.shape[1]``.
        """
        n_classes = self.predictions.shape[1]
        if len(self.class_names) != n_classes:
            raise ValueError(f"class_names has length {len(self.class_names)} but predictions has {n_classes} classes")
        return self
