import pytest

from esp_research.metrics import get_scorer, list_scorers, register_scorer


@pytest.fixture
def matching_predictions_and_targets() -> tuple[list[int], list[int]]:
    predictions = [0, 1, 1, 0]
    targets = predictions
    return predictions, targets


@pytest.fixture
def nonmatching_predictions_and_targets() -> tuple[list[int], list[int]]:
    predictions = [0, 1, 1, 0]
    targets = [1, 0, 0, 1]
    return predictions, targets


@pytest.fixture
def multilabel_predictions_and_targets() -> tuple[list[list[int]], list[list[int]]]:
    predictions = [[1, 0, 1], [0, 1, 1], [1, 1, 0]]
    targets = [[1, 0, 0], [0, 1, 1], [1, 0, 1]]
    return predictions, targets


def test_accuracy_scorer(
    matching_predictions_and_targets,
    nonmatching_predictions_and_targets,
) -> None:
    accuracy_scorer = get_scorer("accuracy")

    predictions, targets = matching_predictions_and_targets
    score = accuracy_scorer(predictions, targets)
    assert score == 1.0

    predictions, targets = nonmatching_predictions_and_targets
    score = accuracy_scorer(predictions, targets)
    assert score == 0.0


def test_register_scorer() -> None:
    @register_scorer
    def dummy_scorer(predictions, targets):
        return 42

    retrieved_scorer = get_scorer("dummy_scorer")
    assert retrieved_scorer is dummy_scorer
    score = retrieved_scorer([], [])
    assert score == 42


def test_list_scorers() -> None:
    scorers = list_scorers()
    assert "accuracy" in scorers
    assert "f1" in scorers
    assert "random" not in scorers


def test_f1_macro_score(multilabel_predictions_and_targets) -> None:
    f1_scorer = get_scorer("f1")

    predictions, targets = multilabel_predictions_and_targets
    score = f1_scorer(predictions, targets, average="macro")
    expected_score = 0.7222222
    assert abs(score - expected_score) < 1e-6
