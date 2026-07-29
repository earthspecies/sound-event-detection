"""
Frame-level scoring for sound event detection models.

Provides `score_file`, the per-file scoring building block used by the
`sed-eval` evaluator (`sound_event_detection.evaluation.evaluator`). The entry
point for running evaluations is the `sed-eval` CLI
(`sound_event_detection.evaluation.cli`), which talks to a detector server over
HTTP.
"""

import numpy as np
import pandas as pd

from sound_event_detection.evaluation.metrics import Scorer
from sound_event_detection.utils.postprocessing import (
    postprocess_frame_predictions,
    postprocess_selection_table_by_threshold,
)
from sound_event_detection.utils.reformatters import frames_to_dur, frames_to_selection_table_by_threshold


def score_file(
    preds: np.ndarray,
    frame_rate: float,
    pred_labels: list[str],
    gt_selection_table: "pd.DataFrame",
    scorer: Scorer,
    gt_labels: list[str],
    species_column: str = "Species",
    postprocessing_config: dict | None = None,
) -> tuple[np.ndarray, list[str], dict]:
    """Postprocess frame predictions for one file and update the scorer.

    Parameters
    ----------
    preds : np.ndarray
        Frame-level predictions, shape (T, K).
    frame_rate : float
        Prediction frame_rate in Hz.
    pred_labels : list[str]
        Label names for each of the K prediction columns.
    gt_selection_table : pd.DataFrame
        Ground-truth selection table for this file.
    scorer : Scorer
        Scorer instance to update (mutated in place).
    gt_labels : list[str]
        Canonical GT class labels for this dataset.
    species_column : str
        Column name for species in selection tables.
    postprocessing_config : dict or None
        Post-processing config (merge_max_gap, min_event_duration, nms).

    Returns
    -------
    tuple[np.ndarray, list[str], dict]
        ``(filtered_preds, filtered_labels, pred_st_by_threshold)`` for optional
        downstream use.
    """
    filtered_preds, filtered_labels = postprocess_frame_predictions(
        preds,
        pred_labels,
        postprocessing_config,
        classes_to_keep=gt_labels,
    )

    pred_st_by_threshold = frames_to_selection_table_by_threshold(
        filtered_preds,
        filtered_labels,
        frame_rate,
        scorer.thresholds,
        annotation_col=species_column,
        probs=filtered_preds,
    )

    pred_st_by_threshold = postprocess_selection_table_by_threshold(
        pred_st_by_threshold,
        postprocessing_config,
        classes_to_keep=gt_labels,
        annotation_col=species_column,
    )

    # Compute selection tables at the metric-specific thresholds (for call-rate/duration
    # metrics). These go through the same postprocessing as the mAP selection tables.
    metric_st = frames_to_selection_table_by_threshold(
        filtered_preds,
        filtered_labels,
        frame_rate,
        scorer.thresholds_for_thresholded_metrics,
        annotation_col=species_column,
        probs=filtered_preds,
    )
    metric_st = postprocess_selection_table_by_threshold(
        metric_st,
        postprocessing_config,
        classes_to_keep=gt_labels,
        annotation_col=species_column,
    )

    duration_sec = frames_to_dur(preds, frame_rate)

    scorer.update_from_selection_table_by_threshold(
        pred_st_by_threshold,
        gt_selection_table,
        duration_sec=duration_sec,
        selection_tables_for_thresholded_metrics=metric_st,
    )

    return filtered_preds, filtered_labels, pred_st_by_threshold
