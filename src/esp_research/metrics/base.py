"""Metric base classes and utilities"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial, total_ordering
from typing import Any, Sequence

from pydantic import BaseModel, Field, field_validator

from .score_functions import (
    ComputesScore,
    get_scorer,
)


class MetricConfig(BaseModel):
    """Metric Config class"""

    name: str = Field(description="Name of the scoring function to be used")
    scorer_kwargs: dict = Field(
        description="Optional keyword arguments to pass to the scorer function",
        default_factory=dict,
    )
    higher_is_better: bool = Field(description="Indicates if higher metric values are better", default=True)
    metric_prefix: str = Field(description="Useful for namespacing, e.g., 'train_', 'val_'", default="")
    metric_suffix: str = Field(description="Useful for namespacing, e.g., '_epoch1', '_step5'", default="")

    @field_validator("name")
    def validate_scorer(cls, v: str) -> str:
        try:
            get_scorer(v)
            return v
        except LookupError as e:
            raise ValueError(f"Scorer '{v}' not found in registry.") from e


@total_ordering
@dataclass(frozen=True, slots=True)
class MetricOutput:
    """Metric value from computation.

    Allows comparison between MetricOutput instances based on the `value` attribute.
    Uses `higher_is_better` to determine comparison direction.

    Attributes
    ----------
    name : str
        The name of the metric.
    value : float
        The computed metric value.
    higher_is_better : bool
        If True, higher values are considered better (e.g., accuracy).
        If False, lower values are considered better (e.g., loss).
    """

    name: str
    value: float
    higher_is_better: bool = True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MetricOutput):
            return NotImplemented
        return self.value == other.value

    def __lt__(self, other: MetricOutput) -> bool:
        if not isinstance(other, MetricOutput):
            return NotImplemented
        if self.higher_is_better != other.higher_is_better:
            raise ValueError("Cannot compare metrics with different higher_is_better semantics")
        if self.higher_is_better:
            return self.value < other.value
        return self.value > other.value

    def __hash__(self) -> int:
        return hash((self.name, self.value, self.higher_is_better))

    def to_dict(self) -> dict[str, Any]:
        """Convert MetricOutput to a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the MetricOutput.
        """
        return {
            "name": self.name,
            "value": self.value,
            "higher_is_better": self.higher_is_better,
        }

    @property
    def log(self) -> dict[str, Any]:
        """Prepare MetricOutput for logging.

        Returns
        -------
        dict[str, Any]
            A dictionary suitable for logging.
        """
        return {self.name: self.value}


@dataclass(slots=True)
class Metric:
    """A Metric wraps a scoring function but does not store predictions and targets.

    Useful for generating scores when all predictions and targets are available.

    Attributes
    ----------
    scorer : ComputesScore
        A callable that computes the score given predictions and targets.
    name : str
        The name of the metric.
    higher_is_better : bool
        If True, higher values are considered better (e.g., accuracy).

    Examples
    --------
    >>> from esp_research.metrics import Metric, MetricConfig
    >>> config = MetricConfig(name="accuracy", higher_is_better=True)
    >>> accuracy_metric = Metric.from_config(config)
    >>> preds, targets = [0, 1, 1, 0], [0, 1, 0, 0]
    >>> result = accuracy_metric.compute_score(preds, targets)
    >>> print(result)
    MetricOutput(name='accuracy', value=0.75, higher_is_better=True)
    """

    scorer: ComputesScore
    name: str
    higher_is_better: bool = True

    def compute_score(
        self,
        predictions: Sequence[Any],
        targets: Sequence[Any] | None = None,
    ) -> MetricOutput:
        """Compute the score for the given predictions and targets.

        Parameters
        ----------
        predictions : Sequence[Any]
            Model predictions.
        targets : Sequence[Any] | None
            Ground truth targets. Can be None for reference-free metrics.

        Returns
        -------
        MetricOutput
            The computed metric result.
        """
        score_value = self.scorer(predictions, targets)
        return MetricOutput(
            name=self.name,
            value=score_value,
            higher_is_better=self.higher_is_better,
        )

    @classmethod
    def from_config(cls, cfg: MetricConfig | dict) -> Metric:
        """Create a Metric instance from a config.

        Parameters
        ----------
        cfg : MetricConfig | dict
            Metric configuration.

        Returns
        -------
        Metric
            A configured metric instance.
        """
        if isinstance(cfg, dict):
            cfg = MetricConfig.model_validate(cfg)

        scorer = partial(get_scorer(cfg.name), **cfg.scorer_kwargs)
        name = cfg.metric_prefix + cfg.name + cfg.metric_suffix
        return cls(scorer=scorer, name=name, higher_is_better=cfg.higher_is_better)

    def __repr__(self) -> str:
        return f"Metric(name={self.name}, higher_is_better={self.higher_is_better})"
