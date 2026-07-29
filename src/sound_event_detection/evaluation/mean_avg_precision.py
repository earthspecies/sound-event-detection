"""Mean average precision metrics."""

from typing import Dict

import numpy as np
import sklearn


def get_map_from_results_by_threshold(
    prec_rec_by_threshold: np.ndarray,
    label_to_idx: Dict[str, int],
    class_mask: np.ndarray | None = None,
) -> Dict[str, float | Dict[str, float]]:
    """Compute AP for each class and mAP across classes from precision/recall curves.

    Parameters
    ----------
    prec_rec_by_threshold : np.ndarray of shape (T, C, 2)
        Precision/recall per threshold and class; last axis is [precision, recall].
    label_to_idx : dict[str, int]
        Mapping from label name to index.
    class_mask : np.ndarray of shape (C,), optional
        Boolean mask where True means include that class in the mAP macro-average.
        AP is still computed for all classes; only the mAP average is filtered.
        If None, all classes are included.

    Returns
    -------
    dict
        ``{"AP_per_class": {class_name: AP_value, ...}, "mAP": mean_AP_value}``.
    """
    results: Dict[str, float | Dict[str, float]] = {"AP_per_class": {}}
    for cl in label_to_idx:
        i = label_to_idx[cl]

        precisions = prec_rec_by_threshold[:, i, 0]
        recalls = prec_rec_by_threshold[:, i, 1]

        inds = np.argsort(recalls)
        recalls = recalls[inds]
        precisions = precisions[inds]

        interpolated_precision = np.maximum.accumulate(precisions[::-1])[::-1]

        ap = sklearn.metrics.auc(recalls, interpolated_precision)
        results["AP_per_class"][cl] = float(ap)

    if class_mask is not None:
        all_aps = [results["AP_per_class"][cl] for cl in label_to_idx if class_mask[label_to_idx[cl]]]
    else:
        all_aps = [results["AP_per_class"][cl] for cl in label_to_idx]
    mAP = np.mean(all_aps)
    results["mAP"] = float(mAP)
    return results


def prec_rec_metrics_by_threshold(results_by_threshold: np.ndarray) -> np.ndarray:
    """Convert TP/FP/FN counts to precision/recall for each threshold and class.

    Parameters
    ----------
    results_by_threshold : np.ndarray of shape (T, C, 3)
        Counts per threshold and class; last axis is [TP, FP, FN].

    Returns
    -------
    np.ndarray of shape (T, C, 2)
        Precision and recall per threshold and class; last axis is [precision, recall].
    """
    in_shape = np.shape(results_by_threshold)
    out = np.zeros((in_shape[0], in_shape[1], 2))

    prec_num = results_by_threshold[:, :, 0]
    prec_denom = results_by_threshold[:, :, 0] + results_by_threshold[:, :, 1]
    precision = np.divide(
        prec_num.astype(np.float32),
        prec_denom.astype(np.float32),
        out=np.ones_like(prec_num).astype(np.float32),
        where=prec_denom != 0,
    )
    out[:, :, 0] = precision

    rec_num = results_by_threshold[:, :, 0]
    rec_denom = results_by_threshold[:, :, 0] + results_by_threshold[:, :, 2]
    recall = np.divide(
        rec_num.astype(np.float32),
        rec_denom.astype(np.float32),
        out=np.ones_like(rec_num).astype(np.float32),
        where=rec_denom != 0,
    )
    out[:, :, 1] = recall

    precision_at_zero_recall_is_one = np.zeros((1, np.shape(out)[1], 2), dtype=np.float32)
    precision_at_zero_recall_is_one[:, :, 0] = 1.0

    out = np.concatenate([precision_at_zero_recall_is_one, out])
    return out
