"""Utility functions for score functions."""

from typing import Iterable

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix as mcm
from sklearn.preprocessing import MultiLabelBinarizer


def make_multihot(arr: list[Iterable[int]]) -> np.ndarray:
    """Make multi-hot encoding from list of label indices.

    Parameters
    ----------
    arr : list[Iterable[int]]
        A list where each element is a list or tuple of indices for a single
        sample, common for targets/predictions in a multilabel classification setting.

    Returns
    -------
    np.ndarray
        A 2D numpy array where each row is a multi-hot encoded vector representing the
        labels for a sample.

    Example
    -------
    >>> arr = [[0, 2], [1], [0, 1, 2]]
    >>> make_multihot(arr)
    array([[1, 0, 1],
           [0, 1, 0],
           [1, 1, 1]])
    """
    mlb = MultiLabelBinarizer(sparse_output=False)
    return mlb.fit_transform(arr)


def handle_multilabel_indicator_array(arr: list[list[int]] | np.ndarray) -> np.ndarray:
    """Ensure the input array is a multi-label indicator array.

    If the input is a 1D array-like of label indices, it converts it to a multi-hot
    encoded 2D array. If it's already a 2D multi-label binary array, it returns it
    as is.

    Parameters
    ----------
    arr : list[list[int]] | np.ndarray
        Input array which can be either a list of label indices or a multi-label
        indicator array.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the multi-label indicator matrix.
    """
    try:
        arr = np.asarray(arr)
    except Exception as e:
        # check that error is caused by an inhomegenous array
        # which happens when targets are label indicators
        if not isinstance(e, ValueError) or "setting an array element with a sequence" not in str(e):
            raise e

        arr = make_multihot(arr)

    return arr


def multilabel_confusion_matrix(
    predictions: list[list[int]] | np.ndarray,
    targets: list[list[int]] | np.ndarray,
    sample_weights: list[float] | None = None,
) -> np.ndarray:
    """Computes the multilabel confusion matrix for multi-label classification.
    Wraps sklearn's multilabel_confusion_matrix.

    In multilabel confusion matrix M, the count of true negatives is M[i][0, 0],
    false negatives is M[i][1,0], true positives is M[i][1,1] and false positives
    is M[i][0,1] where i is the class index.

    Parameters
    ----------
    predictions: list[list[int]] | np.ndarray
        Model predictions, expected shape (num_samples, num_classes) or (num_samples,)
        if predictions are label indices
    targets: list[list[int]] | np.ndarray
        Ground truth labels, expected shape (num_samples, num_classes) or (num_samples,)
        if targets are label indices
    sample_weights: Optional[list[float]]
        Optional list of sample weights, length should match number of samples

    Returns
    -------
    conf_matrix: np.ndarray
        Multilabel confusion matrix of shape (num_classes, 2, 2)

    Example
    -------
    >>> predictions = [[1, 0], [0, 1], [1, 0]]
    >>> targets = [[1, 0], [0, 1], [1, 0]]
    >>> m = multilabel_confusion_matrix(predictions, targets)
    >>> assert m.shape == (2, 2, 2)
    >>> print(m[0])
    [[1 0]
     [0 2]]
    """
    # Ensure predictions and targets are numpy arrays
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    return mcm(targets, predictions, sample_weight=sample_weights)
