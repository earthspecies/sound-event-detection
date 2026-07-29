"""Counting TP/FP/FN for SED evaluation."""

from typing import Dict

import numpy as np
import numpy.typing as npt
from pandas import DataFrame

from sound_event_detection.evaluation.matching import match_events


def get_frame_tpfpfn_counts(
    gt_array: npt.NDArray,
    preds_array: npt.NDArray[np.bool_],
    frame_rate: float,
) -> npt.NDArray[np.float64]:
    """Compute per-class TP/FP/FN counts from frame-level predictions.

    Parameters
    ----------
    gt_array : ndarray of shape (T, C)
        Ground-truth frame occupancy per class; can be fractional.
    preds_array : ndarray of shape (T, C), bool
        Thresholded predictions per frame and class (True = positive).
    frame_rate : float
        Frames per second, used to convert results to seconds.

    Returns
    -------
    ndarray of shape (C, 3), float64
        Per-class counts [TP, FP, FN], summed over frames (in seconds).

    Raises
    ------
    ValueError
        If gt_array is not the same shape as preds_array.
    """
    if gt_array.shape != preds_array.shape:
        raise ValueError(f"gt_array and preds_array must have same shape, got {gt_array.shape} vs {preds_array.shape}")

    T, C = gt_array.shape
    preds_f = preds_array.astype(np.float32)
    gt_f = gt_array.astype(np.float32)

    out = np.zeros((C, 3), dtype=np.float64)

    tp = np.sum(preds_f * gt_f, axis=0)
    fp = np.sum(preds_f, axis=0) - tp
    fn = np.sum(gt_f, axis=0) - tp

    out[:, 0] = tp
    out[:, 1] = fp
    out[:, 2] = fn

    out /= float(frame_rate)
    return out


def get_tpfpfn_counts(
    gt: DataFrame,
    pred: DataFrame,
    iou_threshold: float,
    label_to_idx: Dict[str, int],
    annotation_col: str = "Species",
) -> npt.NDArray[np.floating]:
    """Count TP/FP/FN detections using IoU-based matching.

    Parameters
    ----------
    gt : pd.DataFrame
        Ground-truth selection table with columns
        ``"Begin Time (s)"``, ``"End Time (s)"``, and `annotation_col`.
    pred : pd.DataFrame
        Predicted selection table with the same columns.
    iou_threshold : float
        Minimum IoU required to consider a GT/predicted event as a match.
    label_to_idx : dict[str, int]
        Mapping from class label to integer class index.
    annotation_col : str
        Name of the column containing class labels.

    Returns
    -------
    np.ndarray of shape (n_classes, 3)
        Per-class counts of [TP, FP, FN].
    """
    n_classes = len(label_to_idx.keys())

    ref = np.array(gt[["Begin Time (s)", "End Time (s)"]]).T
    est = np.array(pred[["Begin Time (s)", "End Time (s)"]]).T
    matched = match_events(ref, est, min_iou=iou_threshold)

    out = np.zeros((n_classes, 3))

    def _label_to_idx(x: str) -> int:
        """Map a class label to its index.

        Parameters
        ----------
        x : str
            Label string.

        Returns
        -------
        int
            Class index.

        Raises
        ------
        ValueError
            If `x` is not a key in `label_to_idx`.
        """
        if x in label_to_idx:
            return label_to_idx[x]
        else:
            raise ValueError(f"Unrecognized class label: {x}")

    pred_label_idxs = np.array(pred[annotation_col].map(_label_to_idx)).astype(int)
    gt_label_idxs = np.array(gt[annotation_col].map(_label_to_idx)).astype(int)

    for p in matched:
        gt_label_idx = gt_label_idxs[p[0]]
        pred_label_idx = pred_label_idxs[p[1]]

        if gt_label_idx == pred_label_idx:
            out[pred_label_idx, 0] += 1

    n_pred = np.zeros((n_classes,))
    for pred_label_idx in pred_label_idxs:
        if pred_label_idx >= 0:
            n_pred[pred_label_idx] += 1

    n_gt = np.zeros((n_classes,))
    for gt_label_idx in gt_label_idxs:
        if gt_label_idx >= 0:
            n_gt[gt_label_idx] += 1

    out[:, 1] = n_pred - out[:, 0]
    out[:, 2] = n_gt - out[:, 0]

    return out
