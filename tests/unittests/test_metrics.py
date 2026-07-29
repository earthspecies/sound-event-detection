"""Unit tests for esp_research.metrics.metrics module."""

import pytest

from esp_research.metrics import (
    Metric,
    MetricConfig,
    MetricOutput,
)


class TestMetricConfig:
    """Tests for MetricConfig class."""

    def test_valid_metric_config(self) -> None:
        """Test creating a valid MetricConfig."""
        config = MetricConfig(
            name="accuracy",
            higher_is_better=True,
            scorer_kwargs={"normalize": True},
        )
        assert config.name == "accuracy"
        assert config.scorer_kwargs == {"normalize": True}
        assert config.higher_is_better is True

    def test_metric_config_without_kwargs(self) -> None:
        """Test creating a MetricConfig without scorer_kwargs."""
        config = MetricConfig(name="accuracy", higher_is_better=True)
        assert config.name == "accuracy"
        assert config.scorer_kwargs == {}

    def test_invalid_name(self) -> None:
        """Test that invalid scorer names raise ValueError."""
        with pytest.raises(
            ValueError, match="Scorer 'nonexistent_scorer' not found in registry."
        ):
            MetricConfig(name="nonexistent_scorer", higher_is_better=True)

    def test_metric_config_from_dict(self) -> None:
        """Test creating MetricConfig from dictionary."""
        config_dict = {
            "name": "accuracy",
            "higher_is_better": True,
            "scorer_kwargs": {"normalize": False},
        }
        config = MetricConfig.model_validate(config_dict)
        assert config.name == "accuracy"
        assert config.scorer_kwargs == {"normalize": False}


class TestMetricOutput:
    """Tests for MetricOutput class."""

    def test_valid_metric_output(self) -> None:
        """Test creating a valid MetricOutput."""
        output = MetricOutput(name="accuracy", value=0.95, higher_is_better=True)
        assert output.name == "accuracy"
        assert output.value == 0.95
        assert output.higher_is_better is True

    def test_metric_output_with_zero_value(self) -> None:
        """Test MetricOutput with zero value."""
        output = MetricOutput(name="loss", value=0.0, higher_is_better=False)
        assert output.name == "loss"
        assert output.value == 0.0
        assert output.higher_is_better is False

    def test_metric_output_comparison_higher_is_better(self) -> None:
        """Test comparison when higher_is_better=True."""
        output1 = MetricOutput(name="accuracy", value=0.8, higher_is_better=True)
        output2 = MetricOutput(name="accuracy", value=0.9, higher_is_better=True)
        assert output1 < output2
        assert output2 > output1

    def test_metric_output_comparison_lower_is_better(self) -> None:
        """Test comparison when higher_is_better=False (e.g., loss)."""
        output1 = MetricOutput(name="loss", value=0.8, higher_is_better=False)
        output2 = MetricOutput(name="loss", value=0.9, higher_is_better=False)
        # Lower loss is better, so 0.8 > 0.9 in "goodness"
        assert output1 > output2
        assert output2 < output1

    def test_metric_output_is_frozen(self) -> None:
        """Test that MetricOutput is immutable."""
        output = MetricOutput(name="accuracy", value=0.95, higher_is_better=True)
        with pytest.raises(AttributeError):
            output.value = 0.5  # type: ignore[misc]

    def test_metric_output_to_dict(self) -> None:
        """Test conversion of MetricOutput to dictionary."""
        output = MetricOutput(name="accuracy", value=0.95, higher_is_better=True)
        output_dict = output.to_dict()
        expected_dict = {
            "name": "accuracy",
            "value": 0.95,
            "higher_is_better": True,
        }
        assert output_dict == expected_dict

    def test_log_property(self) -> None:
        """Test the log property of MetricOutput."""
        output = MetricOutput(name="accuracy", value=0.95, higher_is_better=True)
        log_dict = output.log
        expected_log = {"accuracy": 0.95}
        assert log_dict == expected_log


class TestMetric:
    """Tests for Metric class."""

    def test_stateless_metric_computation(self) -> None:
        """Test stateless metric computation."""
        config = MetricConfig(name="accuracy", higher_is_better=True)
        metric = Metric.from_config(config)

        output = metric.compute_score([0, 1, 1, 0], [0, 1, 0, 0])

        assert output.name == "accuracy"
        assert abs(output.value - 0.75) < 1e-6  # Accuracy should be 75%
        assert output.higher_is_better is True
