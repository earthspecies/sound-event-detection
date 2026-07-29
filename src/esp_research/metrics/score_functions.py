"""Score functions module.

This module provides a registry for scoreres and implementations
for various scoring functions used in eval tasks.

Usage:
- Register a new scoring function using the `@register_scorer` decorator.
- Retrieve a scoring function by name using `get_scorer(score_name)`.
- Compute score by calling the function

Example
-------
>>> from esp_research.metrics import get_scorer
>>> accuracy_scorer = get_scorer("accuracy")
>>> predictions = [0, 1, 1, 0]
>>> targets = [0, 1, 0, 0]
>>> score = accuracy_scorer(predictions, targets)
>>> print(score)
0.75
"""

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics import (
    average_precision_score as ap_score,
)

from esp_research.protocols.eval import ComputesScore

from .utils import handle_multilabel_indicator_array

_SCORE_REGISTRY = {}


def get_scorer(score_name: str) -> ComputesScore | None:
    """Retrieve a scoring function by name with optional keyword arguments.

    Parameters
    ----------
    score_name : str
        The name of the scoring function to retrieve.

    Returns
    -------
    callable | None
        The scoring function if found, otherwise None.

    Raises
    ------
    LookupError
        If the scoring function is not found in the registry.
    """
    scorer = _SCORE_REGISTRY.get(score_name, None)
    if not scorer:
        raise LookupError(f"Scorer '{score_name}' not found in registry.")
    return scorer


def register_scorer(func: ComputesScore) -> ComputesScore:
    """Decorator to register a scoring function.

    Parameters
    ----------
    func : callable
        The scoring function to register.

    Returns
    -------
    callable
        The original scoring function.

    Raises
    ------
    ValueError
        If a scorer with the same name is already registered.
    """
    if func.__name__ in _SCORE_REGISTRY:
        raise ValueError(f"Scorer '{func.__name__}' is already registered.")

    _SCORE_REGISTRY[func.__name__] = func
    return func


def _common_validation(predictions: Sequence[Any], targets: Sequence[Any] | None) -> None:
    if predictions is None or len(predictions) == 0:
        raise ValueError("Predictions array is empty.")
    if targets is None or len(targets) == 0:
        raise ValueError("Targets array is empty.")
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length.")


@register_scorer
def accuracy(
    predictions: list[float | int] | np.ndarray,
    targets: list[float | int] | np.ndarray,
) -> float:
    """Wraps sklearn's accuracy_score.

    Parameters
    ----------
    predictions: list[float | int] | np.ndarray
        List of model predictions
    targets: list[float | int] | np.ndarray
        List of ground truth labels

    Returns
    -------
    accuracy: float
        Accuracy score for the predictions

    Example
    -------
    >>> predictions = [0, 1, 1, 0]
    >>> targets = [0, 1, 0, 0]
    >>> accuracy(predictions, targets)
    0.75
    >>> # Multilabel case
    >>> accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))
    0.5

    Raises
    ------
    ValueError
        If predictions and targets have different shapes.
    """
    _common_validation(predictions, targets)
    # Ensure predictions and targets are numpy arrays
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape.")

    return accuracy_score(targets, predictions)


@register_scorer
def precision(
    predictions: list[list[int]] | np.ndarray,
    targets: list[list[int]] | np.ndarray,
    sample_weights: list[float] | None = None,
    zero_division: int = 0,
    average: str = "macro",
) -> float:
    """Computes the precision for single / multi-label classification.
    and then averaged. Wraps sklearn's precision_score.

    Parameters
    ----------
    predictions: list[list[int]] | np.ndarray
        Model predictions, expected shape (num_samples, num_classes)
    targets: list[list[int]] | np.ndarray
        Ground truth labels, can be label indicators (list[list[int]]) or binary matrix,
        expected shape (num_samples, num_classes)
    sample_weights: Optional[list[float]]
        Optional list of sample weights, length should match number of samples
    zero_division: int
        Value to return when there is a zero division, default is np.nan
    average: str
        Averaging method for precision calculation, default is 'macro'

    Returns
    -------
    precision: float
        precision score for the predictions

    Example
    -------
    >>> predictions = [[1, 0], [0, 1], [1, 0]]
    >>> targets = [[1, 0], [0, 1], [1, 0]]
    >>> precision(predictions, targets)
    1.0
    >>> # Using label indicators for targets
    >>> predictions = [[1, 0, 0], [0, 1, 1], [1, 0, 1]]
    >>> targets = [[0], [1, 2], [0, 2]]
    >>> precision(predictions, targets)
    1.0

    Raises
    ------
    ValueError
        If average is None.
    """
    _common_validation(predictions, targets)
    # Ensure predictions and targets are numpy arrays
    predictions = handle_multilabel_indicator_array(predictions)
    targets = handle_multilabel_indicator_array(targets)

    if average is None:
        raise ValueError("Score functions dont return arrays. Please set average parameter.")

    return precision_score(
        targets,
        predictions,
        sample_weight=sample_weights,
        zero_division=zero_division,
        average=average,
    )


@register_scorer
def average_precision(
    predictions: list[list[int]] | np.ndarray,
    targets: list[list[int]] | np.ndarray,
    sample_weights: list[float] | None = None,
    average: str = "macro",
) -> float:
    """Computes the average precision score for multi-label classification.
    Wraps sklearn's average_precision_score. This function calculates
    the recall difference weighted precision for a number of thresholds

    Parameters
    ----------
    predictions: list[list[int]] | np.ndarray
        Model predictions, expected shape (num_samples, num_classes)
    targets: list[list[int]] | np.ndarray
        Ground truth labels, expected shape (num_samples, num_classes)
    sample_weights: Optional[list[float]]
        Optional list of sample weights, length should match number of samples
    average: str
        Averaging method for average precision calculation, default is 'macro'

    Returns
    -------
    average_precision: float
        Average precision score for the predictions

    Example
    -------
    >>> predictions = [[1, 0], [0, 1], [1, 0]]
    >>> targets = [[1, 0], [0, 1], [1, 0]]
    >>> average_precision(predictions, targets)
    1.0

    Raises
    ------
    ValueError
        If average is None.
    """
    _common_validation(predictions, targets)
    # Ensure predictions and targets are numpy arrays
    # average_precision_score doesn't support multilabel label indicators
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape.")

    if average is None:
        raise ValueError("Score functions dont return arrays. Please set average parameter.")

    return ap_score(
        targets,
        predictions,
        sample_weight=sample_weights,
        average=average,
    )


@register_scorer
def recall(
    predictions: list[list[int]] | np.ndarray,
    targets: list[list[int]] | np.ndarray,
    sample_weights: list[float] | None = None,
    zero_division: int | float | str = np.nan,
    average: str = "macro",
) -> float:
    """Computes the average recall for multi-label classification.
    Recall is calculated for each class, optionally weighted by sample_weights
    and then averaged. Wraps sklearn's recall_score.

    Parameters
    ----------
    predictions: list[list[int]] | np.ndarray
        Model predictions, expected shape (num_samples, num_classes)
    targets: list[list[int]] | np.ndarray
        Ground truth labels, can be label indicators (list[list[int]]) or binary matrix,
        expected shape (num_samples, num_classes)
    sample_weights: Optional[list[float]]
        Optional list of sample weights, length should match number of samples
    zero_division: int | float | str
        Value to return when there is a zero division, default is np.nan
    average: str
        Averaging method for recall calculation, default is 'macro'

    Returns
    -------
    recall: float
        recall score for the predictions

    Example
    -------
    >>> predictions = [[1, 0], [0, 1], [1, 0]]
    >>> targets = [[1, 0], [0, 1], [1, 0]]
    >>> recall(predictions, targets, average="macro")
    1.0
    >>> # Using label indicators for targets
    >>> predictions = [[1, 0, 0], [0, 1, 1], [1, 0, 1]]
    >>> targets = [[0], [1, 2], [0, 2]]
    >>> recall(predictions, targets)
    1.0
    """
    _common_validation(predictions, targets)
    # Ensure predictions and targets are numpy arrays
    predictions = handle_multilabel_indicator_array(predictions)
    targets = handle_multilabel_indicator_array(targets)

    return recall_score(
        targets,
        predictions,
        sample_weight=sample_weights,
        zero_division=zero_division,
        average=average,
    )


@register_scorer
def f1(
    predictions: list[list[int]] | np.ndarray,
    targets: list[list[int]] | np.ndarray,
    zero_division: int | float | str = np.nan,
    average: str = "macro",
) -> float:
    """Computes the F1 score for classification.
    Wraps sklearn's f1_score.

    Parameters
    ----------
    predictions: list[list[int]] | np.ndarray,
        List of model predictions (0 or 1)
    targets: list[list[int]] | np.ndarray,
        List of ground truth labels (0 or 1)
    zero_division: int | float | str
        Value to return when there is a zero division, default is np.nan
    average: str
        Averaging method for F1 calculation, default is 'macro'

    Returns
    -------
    f1_score: float
        F1 score for the predictions

    Example
    -------
    >>> predictions = [0, 1, 1, 0]
    >>> targets = [0, 1, 0, 0]
    >>> f1(predictions, targets, average="binary")
    0.6666666666666666
    """
    _common_validation(predictions, targets)
    # Ensure predictions and targets are numpy arrays
    predictions = handle_multilabel_indicator_array(predictions)
    targets = handle_multilabel_indicator_array(targets)

    return f1_score(
        targets,
        predictions,
        zero_division=zero_division,
        average=average,
    )


def list_scorers() -> list[str]:
    """List all registered scoring functions.

    Returns
    -------
    list[str]
        List of registered scoring function names.
    """
    return list(_SCORE_REGISTRY.keys())
