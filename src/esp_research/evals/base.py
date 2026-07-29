"""Defines the api for each eval"""

from enum import Enum

from alp_data import DatasetConfig
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass

from esp_research.metrics import MetricConfig
from esp_research.protocols.eval import EvaluatesModelOnTasks
from esp_research.utils.registry import Registry

evals_registry = Registry[EvaluatesModelOnTasks]("evaluator", discriminator="type")


def validate_has_expected_model_output(cls: type) -> None:
    """Validate that an evaluator class defines `expected_model_output`.

    The attribute must be a Pydantic BaseModel subclass so it can
    describe the model contract via its JSON schema.

    Parameters
    ----------
    cls : type
        The evaluator class being registered.

    Raises
    ------
    TypeError
        If `expected_model_output` is missing or not a Pydantic BaseModel.
    """
    attr = getattr(cls, "expected_model_output", None)
    if attr is None:
        raise TypeError(
            f"Evaluator '{cls.__name__}' must define an `expected_model_output` "
            f"class attribute pointing to a Pydantic BaseModel that describes "
            f"the expected model output schema."
        )
    if not isinstance(attr, type) or not issubclass(attr, BaseModel):
        raise TypeError(
            f"Evaluator '{cls.__name__}.expected_model_output' must be a Pydantic BaseModel subclass, got {attr!r}."
        )


class TaskType(str, Enum):
    """Enum for different task types, useful for categorizing tasks.
    String enums are used for easy serialization in config files.
    """

    CLASSIFICATION = "classification"
    DETECTION = "detection"
    CAPTIONING = "captioning"
    FRAME_DETECTION = "frame_detection"
    EVENT_DETECTION = "event_detection"
    CALL_TYPE_CLASSIFICATION = "call_type_classification"
    INDIVIDUAL_ID = "individual_id"


@dataclass
class TargetDatasetSplit:
    """Reference to a specific dataset split for metric mapping.
    'version' is reserved for future use.

    Attributes
    ----------
    dataset_name : str
        The name of the dataset.
    split : str
        The specific split within the dataset.
    version : str | None
        Optional version identifier for the split. Reserved for future use.
    """

    dataset_name: str
    split: str
    version: str | None = None


class EvalTaskConfig(BaseModel):
    """Task Config class.

    A task is a collection of datasets and metrics used to evaluate a model on a specific task.
    Each metric may be applied to one or more of the dataset-split-versions as defined in the
    metric_to_dataset_mapping.s

    Attributes
    ----------
    task_type : TaskType
        The type of task (e.g., classification, detection).
    datasets : list[DatasetConfig]
        A list of dataset configurations for the task.
    metrics : list[MetricConfig]
        A list of metric configurations to be computed for the task.
    metric_to_dataset_mapping : dict[str, list[TargetDatasetSplit]]
        A mapping from metric names to specific dataset targets. Metrics not in this mapping
        apply to ALL datasets.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=False,
        validate_assignment=True,
        str_strip_whitespace=True,
        extra="forbid",
    )

    task_type: TaskType
    datasets: list[DatasetConfig]
    metrics: list[MetricConfig]
    metric_to_dataset_mapping: dict[str, list[TargetDatasetSplit]] = Field(
        default_factory=dict,
        description=(
            "Maps metric names to specific dataset targets. Metrics not in this mapping apply to all datasets."
        ),
    )

    @model_validator(mode="after")
    def validate_metric_to_dataset_mapping(self) -> "EvalTaskConfig":
        """Validate that metric_to_dataset_mapping targets exist in datasets.

        Returns
        -------
        TaskConfig
            The validated TaskConfig instance.

        Raises
        ------
        ValueError
            If a metric has an empty target list or references a non-existent dataset/split.
        """
        if not self.metric_to_dataset_mapping:
            return self

        valid_targets = {(ds.dataset_name, ds.split) for ds in self.datasets}

        for metric_name, targets in self.metric_to_dataset_mapping.items():
            if not targets:
                raise ValueError(
                    f"Empty target list for metric '{metric_name}'.Remove from mapping to apply to all datasets."
                )

            for target in targets:
                if (target.dataset_name, target.split) not in valid_targets:
                    raise ValueError(
                        f"Invalid target ({target.dataset_name}, {target.split}) "
                        f"for metric '{metric_name}'. Target does not exist in datasets."
                    )

        return self

    def get_datasets_for_metric(self, metric_name: str) -> list[DatasetConfig]:
        """Return the datasets a metric should be applied to.

        Parameters
        ----------
        metric_name : str
            The name of the metric.

        Returns
        -------
        list[DatasetConfig]
            The datasets the metric should be applied to. If the metric is not
            in metric_to_dataset_mapping, returns all datasets.
        """
        if metric_name not in self.metric_to_dataset_mapping:
            return self.datasets

        targets = {(t.dataset_name, t.split) for t in self.metric_to_dataset_mapping[metric_name]}
        return [ds_cfg for ds_cfg in self.datasets if (ds_cfg.dataset_name, ds_cfg.split) in targets]

    # TODO: better to have dataset_name: str, split: str, version: str as parameters?
    def get_metrics_for_dataset(self, dataset: DatasetConfig) -> list[MetricConfig]:
        """Return the metrics that should be applied to a dataset.

        Parameters
        ----------
        dataset : DatasetConfig
            The dataset configuration.

        Returns
        -------
        list[MetricConfig]
            The metrics that should be applied to the dataset.
        """
        result = []
        for metric in self.metrics:
            if metric.name not in self.metric_to_dataset_mapping:
                # Not in mapping = applies to all datasets
                result.append(metric)
            else:
                targets = {(t.dataset_name, t.split) for t in self.metric_to_dataset_mapping[metric.name]}
                if (dataset.dataset_name, dataset.split) in targets:
                    result.append(metric)
        return result
