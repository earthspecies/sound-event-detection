import pytest
from esp_research.metrics import MetricConfig
from esp_research.evals.base import (
    TaskType,
    EvalTaskConfig,
    DatasetConfig,
    TargetDatasetSplit,
)


def test_task_type_enum() -> None:
    """Test TaskType enum values."""
    assert TaskType.CLASSIFICATION.value == "classification"
    assert TaskType.DETECTION.value == "detection"
    assert TaskType.CAPTIONING.value == "captioning"


def test_task_config_construction() -> None:
    """Test constructing a EvalTaskConfig."""
    dataset_config = DatasetConfig(dataset_name="beans", split="cbi_val")
    metric_config = MetricConfig(
        name="accuracy",
    )
    task_config = EvalTaskConfig(
        task_type=TaskType.CLASSIFICATION,
        datasets=[dataset_config],
        metrics=[metric_config],
    )

    assert task_config.task_type == TaskType.CLASSIFICATION
    assert len(task_config.datasets) == 1
    assert len(task_config.metrics) == 1
    # Empty mapping means all metrics apply to all datasets
    assert task_config.metric_to_dataset_mapping == {}


def test_task_config_with_explicit_mapping() -> None:
    """Test EvalTaskConfig with explicit metric_to_dataset_mapping."""
    ds1 = DatasetConfig(dataset_name="beans", split="cbi_val")
    ds2 = DatasetConfig(dataset_name="beans", split="esc50")
    metric_accuracy = MetricConfig(name="accuracy")
    metric_f1 = MetricConfig(name="f1")

    task_config = EvalTaskConfig(
        task_type=TaskType.CLASSIFICATION,
        datasets=[ds1, ds2],
        metrics=[metric_accuracy, metric_f1],
        metric_to_dataset_mapping={
            "accuracy": [TargetDatasetSplit(dataset_name="beans", split="cbi_val")],
        },
    )

    # accuracy only applies to cbi_val, f1 applies to all (not in mapping)
    assert len(task_config.metric_to_dataset_mapping) == 1
    assert task_config.metric_to_dataset_mapping["accuracy"][0].split == "cbi_val"


def test_task_config_validation_invalid_target() -> None:
    """Test that invalid targets in mapping raise ValueError."""
    dataset_config = DatasetConfig(dataset_name="beans", split="cbi_val")
    metric_config = MetricConfig(name="accuracy")

    with pytest.raises(ValueError, match="Invalid target"):
        EvalTaskConfig(
            task_type=TaskType.CLASSIFICATION,
            datasets=[dataset_config],
            metrics=[metric_config],
            metric_to_dataset_mapping={
                "accuracy": [TargetDatasetSplit(dataset_name="beans", split="nonexistent")],
            },
        )


def test_task_config_validation_empty_target_list() -> None:
    """Test that empty target list raises ValueError."""
    dataset_config = DatasetConfig(dataset_name="beans", split="cbi_val")
    metric_config = MetricConfig(name="accuracy")

    with pytest.raises(ValueError, match="Empty target list"):
        EvalTaskConfig(
            task_type=TaskType.CLASSIFICATION,
            datasets=[dataset_config],
            metrics=[metric_config],
            metric_to_dataset_mapping={
                "accuracy": [],
            },
        )


def test_get_datasets_for_metric() -> None:
    """Test get_datasets_for_metric helper method."""
    ds1 = DatasetConfig(dataset_name="beans", split="cbi_val")
    ds2 = DatasetConfig(dataset_name="beans", split="esc50")
    metric_accuracy = MetricConfig(name="accuracy")
    metric_f1 = MetricConfig(name="f1")

    task_config = EvalTaskConfig(
        task_type=TaskType.CLASSIFICATION,
        datasets=[ds1, ds2],
        metrics=[metric_accuracy, metric_f1],
        metric_to_dataset_mapping={
            "accuracy": [TargetDatasetSplit(dataset_name="beans", split="cbi_val")],
        },
    )

    # accuracy only applies to cbi_val
    accuracy_datasets = task_config.get_datasets_for_metric("accuracy")
    assert len(accuracy_datasets) == 1
    assert accuracy_datasets[0].split == "cbi_val"

    # f1 not in mapping, applies to all datasets
    f1_datasets = task_config.get_datasets_for_metric("f1")
    assert len(f1_datasets) == 2


def test_get_metrics_for_dataset() -> None:
    """Test get_metrics_for_dataset helper method."""
    ds1 = DatasetConfig(dataset_name="beans", split="cbi_val")
    ds2 = DatasetConfig(dataset_name="beans", split="esc50")
    metric_accuracy = MetricConfig(name="accuracy")
    metric_f1 = MetricConfig(name="f1")

    task_config = EvalTaskConfig(
        task_type=TaskType.CLASSIFICATION,
        datasets=[ds1, ds2],
        metrics=[metric_accuracy, metric_f1],
        metric_to_dataset_mapping={
            "accuracy": [TargetDatasetSplit(dataset_name="beans", split="cbi_val")],
        },
    )

    # cbi_val gets both accuracy (explicit) and f1 (not in mapping = all)
    cbi_metrics = task_config.get_metrics_for_dataset(ds1)
    assert len(cbi_metrics) == 2
    assert {m.name for m in cbi_metrics} == {"accuracy", "f1"}

    # esc50 only gets f1 (accuracy is restricted to cbi_val)
    esc_metrics = task_config.get_metrics_for_dataset(ds2)
    assert len(esc_metrics) == 1
    assert esc_metrics[0].name == "f1"
