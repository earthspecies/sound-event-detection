"""
Metrics for sound event detection evaluation.

Provides `Scorer`, which accumulates per-file TP/FP/FN counts across a
threshold grid and computes frame-level and event-level mAP-style metrics,
as well as per-class and macro precision, recall, and F1 at specific
thresholds (`thresholded_f1`).
"""

from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from sound_event_detection.evaluation.counting import get_frame_tpfpfn_counts, get_tpfpfn_counts
from sound_event_detection.evaluation.mean_avg_precision import (
    get_map_from_results_by_threshold,
    prec_rec_metrics_by_threshold,
)
from sound_event_detection.utils.reformatters import selection_table_to_frames


def _compute_event_counts_for_threshold(
    thr: float,
    prediction_selection_table_by_threshold: Dict[float, pd.DataFrame],
    ground_truth_selection_table: pd.DataFrame,
    label_to_idx: Dict[str, int],
    annotation_col: str,
    iou_threshold: float,
) -> np.ndarray:
    """Compute per-class TP/FP/FN event counts for a single detection threshold.

    Parameters
    ----------
    thr : float
        Detection threshold.
    prediction_selection_table_by_threshold : dict[float, pd.DataFrame]
        Mapping from threshold to predicted selection table.
    ground_truth_selection_table : pd.DataFrame
        Ground-truth selection table.
    label_to_idx : dict[str, int]
        Mapping from class label to class index.
    annotation_col : str
        Column name for class labels.
    iou_threshold : float
        IoU threshold for matching.

    Returns
    -------
    np.ndarray of shape (C, 3)
        Per-class counts [TP, FP, FN].
    """
    prediction_selection_table = prediction_selection_table_by_threshold[thr]

    return get_tpfpfn_counts(
        ground_truth_selection_table,
        prediction_selection_table,
        iou_threshold,
        label_to_idx,
        annotation_col=annotation_col,
    )


def _compute_frame_counts_for_threshold(
    thr: float,
    prediction_selection_table_by_threshold: Dict[float, pd.DataFrame],
    label_to_idx: Dict[str, int],
    discretization_frame_rate: float,
    annotation_col: str,
    gt_array: np.ndarray,
) -> np.ndarray:
    """Compute per-class TP/FP/FN frame counts for a single detection threshold.

    Parameters
    ----------
    thr : float
        Detection threshold.
    prediction_selection_table_by_threshold : dict[float, pd.DataFrame]
        Mapping from threshold to predicted selection table.
    label_to_idx : dict[str, int]
        Mapping from class label to class index.
    discretization_frame_rate : float
        Frame rate in Hz for discretizing events.
    annotation_col : str
        Column name for class labels.
    gt_array : np.ndarray
        Ground-truth frame array.

    Returns
    -------
    np.ndarray of shape (C, 3)
        Per-class counts [TP, FP, FN].
    """
    prediction_selection_table = prediction_selection_table_by_threshold[thr]
    target_duration_frames = np.shape(gt_array)[0]

    pred_array = selection_table_to_frames(
        prediction_selection_table,
        target_duration_frames,
        discretization_frame_rate,
        label_to_idx,
        annotation_col=annotation_col,
    )

    return get_frame_tpfpfn_counts(gt_array, pred_array, discretization_frame_rate)


class Scorer:
    """Accumulate counts across files and compute frame- and event-based mAP-style metrics.

    Call `update_from_selection_table_by_threshold` once per file, then call
    `get_results` to retrieve the accumulated metrics.

    Parameters
    ----------
    dataset_ontology : Sequence[str]
        Ordered list of label strings defining the dataset class set.
    n_thresholds : int
        Number of thresholds in [min_threshold, 1] for mAP calculation.
    min_threshold : float
        Minimum threshold value. Default is 0.0.
    annotation_col : str
        Column name in selection tables for the event label.
    discretization_frame_rate : float
        Frame rate (Hz) for discretizing event tables into frame arrays.
    iou_thresholds : Iterable[float]
        IoU thresholds for event-based matching.
    thresholds_for_thresholded_metrics : Iterable[float]
        Detection thresholds at which to compute per-class and macro F1, precision, and recall.
    """

    def __init__(
        self,
        dataset_ontology: Sequence[str],
        n_thresholds: int = 101,
        min_threshold: float = 0.0,
        annotation_col: str = "Species",
        discretization_frame_rate: float = 100.0,
        iou_thresholds: Iterable[float] = (0.2, 0.5),
        thresholds_for_thresholded_metrics: Iterable[float] = (0.5,),
    ) -> None:
        self.dataset_ontology: List[str] = list(dataset_ontology)
        self.dataset_label_to_idx: Dict[str, int] = {x: i for i, x in enumerate(dataset_ontology)}
        self.thresholds: np.ndarray = np.round(np.linspace(min_threshold, 1.0, num=n_thresholds, endpoint=True), 3)
        self.annotation_col: str = annotation_col
        self.iou_thresholds: List[float] = list(iou_thresholds)
        self.discretization_frame_rate: float = discretization_frame_rate
        self.thresholds_for_thresholded_metrics: List[float] = list(thresholds_for_thresholded_metrics)

        self.result_counts: Dict[str, np.ndarray | None] = {}
        self.result_counts["frame_counts"] = None
        for iou_threshold in self.iou_thresholds:
            self.result_counts[f"event_counts_iou_{iou_threshold}"] = None

        self.localization_result_counts: Dict[str, np.ndarray | None] = {}
        self.localization_result_counts["frame_counts"] = None
        for iou_threshold in self.iou_thresholds:
            self.localization_result_counts[f"event_counts_iou_{iou_threshold}"] = None

        self.thresholded_counts: Dict[float, Dict[str, np.ndarray | None]] = {
            t: {
                "frame_counts": None,
                **{f"event_counts_iou_{iou}": None for iou in self.iou_thresholds},
            }
            for t in self.thresholds_for_thresholded_metrics
        }

    def _update_counts_array(
        self,
        counts_array_name: str,
        new_counts_to_add: np.ndarray,
        *,
        counts_dict: Dict[str, np.ndarray | None] | None = None,
    ) -> None:
        """Accumulate per-threshold per-class count arrays into a counts dict.

        Parameters
        ----------
        counts_array_name : str
            Key identifying which accumulated array to update.
        new_counts_to_add : np.ndarray
            Array of counts to add to the stored accumulator.
        counts_dict : dict or None
            The counts dictionary to update. Defaults to `self.result_counts`.
        """
        if counts_dict is None:
            counts_dict = self.result_counts
        if counts_dict[counts_array_name] is None:
            counts_dict[counts_array_name] = new_counts_to_add
        else:
            counts_dict[counts_array_name] += new_counts_to_add

    def _update_counts_for_frame_map(
        self,
        prediction_selection_table_by_threshold: Dict[float, pd.DataFrame],
        ground_truth_selection_table: pd.DataFrame,
        duration_sec: float,
        counts_dict: Dict[str, np.ndarray | None] | None = None,
    ) -> None:
        """Update accumulated frame-count arrays for a single file.

        Parameters
        ----------
        prediction_selection_table_by_threshold : dict[float, pd.DataFrame]
            Mapping from threshold to predicted selection table.
        ground_truth_selection_table : pd.DataFrame
            Ground-truth selection table.
        duration_sec : float
            Duration of the underlying audio in seconds.
        counts_dict : dict or None
            The counts dictionary to update. Defaults to `self.result_counts`.
        """
        target_duration_frames = int(duration_sec * self.discretization_frame_rate)
        gt_array = selection_table_to_frames(
            ground_truth_selection_table,
            target_duration_frames,
            self.discretization_frame_rate,
            self.dataset_label_to_idx,
            annotation_col=self.annotation_col,
        )

        results = [
            _compute_frame_counts_for_threshold(
                thr,
                prediction_selection_table_by_threshold,
                self.dataset_label_to_idx,
                self.discretization_frame_rate,
                self.annotation_col,
                gt_array,
            )
            for thr in self.thresholds
        ]

        counts_by_threshold = np.stack(results, axis=0).astype(np.float32)
        self._update_counts_array("frame_counts", counts_by_threshold, counts_dict=counts_dict)

    def _update_counts_for_event_map(
        self,
        prediction_selection_table_by_threshold: Dict[float, pd.DataFrame],
        ground_truth_selection_table: pd.DataFrame,
        iou_threshold: float,
        counts_dict: Dict[str, np.ndarray | None] | None = None,
    ) -> None:
        """Update accumulated event-count arrays for a single file and IoU threshold.

        Parameters
        ----------
        prediction_selection_table_by_threshold : dict[float, pd.DataFrame]
            Mapping from threshold to predicted selection table.
        ground_truth_selection_table : pd.DataFrame
            Ground-truth selection table.
        iou_threshold : float
            IoU threshold for matching.
        counts_dict : dict or None
            The counts dictionary to update. Defaults to `self.result_counts`.
        """
        results = [
            _compute_event_counts_for_threshold(
                thr,
                prediction_selection_table_by_threshold,
                ground_truth_selection_table,
                self.dataset_label_to_idx,
                self.annotation_col,
                iou_threshold,
            )
            for thr in self.thresholds
        ]

        counts_by_threshold = np.stack(results, axis=0).astype(np.float32)
        self._update_counts_array(f"event_counts_iou_{iou_threshold}", counts_by_threshold, counts_dict=counts_dict)

    def _update_all_counts(
        self,
        prediction_selection_table_by_threshold: Dict[float, pd.DataFrame],
        ground_truth_selection_table: pd.DataFrame,
        duration_sec: float,
        counts_dict: Dict[str, np.ndarray | None] | None = None,
    ) -> None:
        """Update both frame- and event-based accumulated counts for a single file.

        Parameters
        ----------
        prediction_selection_table_by_threshold : dict[float, pd.DataFrame]
            Mapping from threshold to predicted selection table.
        ground_truth_selection_table : pd.DataFrame
            Ground-truth selection table.
        duration_sec : float
            Duration of the underlying audio in seconds.
        counts_dict : dict or None
            The counts dictionary to update. Defaults to `self.result_counts`.
        """
        self._update_counts_for_frame_map(
            prediction_selection_table_by_threshold, ground_truth_selection_table, duration_sec, counts_dict=counts_dict
        )
        for iou_threshold in self.iou_thresholds:
            self._update_counts_for_event_map(
                prediction_selection_table_by_threshold,
                ground_truth_selection_table,
                iou_threshold,
                counts_dict=counts_dict,
            )

    def _update_thresholded_counts(
        self,
        metric_st_filtered: Dict[float, pd.DataFrame],
        gt_filtered: pd.DataFrame,
        duration_sec: float,
    ) -> None:
        """Accumulate per-class TP/FP/FN at each threshold in `thresholds_for_thresholded_metrics`.

        Parameters
        ----------
        metric_st_filtered : dict[float, pd.DataFrame]
            Predicted selection tables keyed by each threshold in
            ``thresholds_for_thresholded_metrics``, already filtered to GT species.
        gt_filtered : pd.DataFrame
            Ground-truth selection table (Unknown-excluded).
        duration_sec : float
            Duration of the audio clip in seconds.
        """
        target_duration_frames = int(duration_sec * self.discretization_frame_rate)
        gt_array = selection_table_to_frames(
            gt_filtered,
            target_duration_frames,
            self.discretization_frame_rate,
            self.dataset_label_to_idx,
            annotation_col=self.annotation_col,
        )
        for t in self.thresholds_for_thresholded_metrics:
            frame_counts = _compute_frame_counts_for_threshold(
                t,
                metric_st_filtered,
                self.dataset_label_to_idx,
                self.discretization_frame_rate,
                self.annotation_col,
                gt_array,
            ).astype(np.float32)
            if self.thresholded_counts[t]["frame_counts"] is None:
                self.thresholded_counts[t]["frame_counts"] = frame_counts
            else:
                self.thresholded_counts[t]["frame_counts"] += frame_counts

            for iou in self.iou_thresholds:
                event_counts = _compute_event_counts_for_threshold(
                    t,
                    metric_st_filtered,
                    gt_filtered,
                    self.dataset_label_to_idx,
                    self.annotation_col,
                    iou,
                ).astype(np.float32)
                key = f"event_counts_iou_{iou}"
                if self.thresholded_counts[t][key] is None:
                    self.thresholded_counts[t][key] = event_counts
                else:
                    self.thresholded_counts[t][key] += event_counts

    def update_from_selection_table_by_threshold(
        self,
        prediction_selection_table_by_threshold: Dict[float, pd.DataFrame],
        ground_truth_selection_table: pd.DataFrame,
        duration_sec: float,
        selection_tables_for_thresholded_metrics: Dict[float, pd.DataFrame] | None = None,
    ) -> None:
        """Ingest per-threshold prediction selection tables for a single file.

        Parameters
        ----------
        prediction_selection_table_by_threshold : dict[float, pd.DataFrame]
            Mapping from threshold to predicted selection table.
        ground_truth_selection_table : pd.DataFrame
            Ground-truth selection table.
        duration_sec : float
            Duration of the underlying audio in seconds.
        selection_tables_for_thresholded_metrics : dict or None
            Mapping from each threshold in ``thresholds_for_thresholded_metrics`` to a
            predicted selection table, used for F1/precision/recall computation. Falls back
            to the closest entry in `self.thresholds` if not provided.

        Raises
        ------
        ValueError
            If the provided thresholds don't match those expected by the scorer.
        """
        for threshold in self.thresholds:
            if threshold not in prediction_selection_table_by_threshold:
                raise ValueError("Mismatch between thresholds expected and received")

            st = prediction_selection_table_by_threshold[threshold]
            prediction_selection_table_by_threshold[threshold] = st[
                st[self.annotation_col].isin(self.dataset_ontology)
            ].copy()

        gt_filtered = ground_truth_selection_table[ground_truth_selection_table[self.annotation_col] != "Unknown"]

        self._update_all_counts(prediction_selection_table_by_threshold, gt_filtered, duration_sec)

        gt_species = set(gt_filtered[self.annotation_col].unique())
        localization_preds = {
            thr: df[df[self.annotation_col].isin(gt_species)].copy()
            for thr, df in prediction_selection_table_by_threshold.items()
        }
        self._update_all_counts(
            localization_preds, gt_filtered, duration_sec, counts_dict=self.localization_result_counts
        )

        if selection_tables_for_thresholded_metrics is None:
            warnings.warn(
                "selection_tables_for_thresholded_metrics not provided; "
                "using closest thresholds from self.thresholds for F1/precision/recall metrics.",
                UserWarning,
                stacklevel=2,
            )
            selection_tables_for_thresholded_metrics = {
                t: prediction_selection_table_by_threshold[
                    float(self.thresholds[np.argmin(np.abs(self.thresholds - t))])
                ]
                for t in self.thresholds_for_thresholded_metrics
            }

        metric_st_filtered = {
            t: df[df[self.annotation_col].isin(gt_species)].copy()
            for t, df in selection_tables_for_thresholded_metrics.items()
        }
        self._update_thresholded_counts(metric_st_filtered, gt_filtered, duration_sec)

    def get_state(self) -> Dict[str, Dict]:
        """Export the accumulated count arrays so evaluation can be checkpointed.

        The returned structure fully describes the scorer's progress: feeding it
        back into a freshly constructed scorer (with the same configuration) via
        `load_state` reproduces the current results exactly, because all counts
        are additive.

        Returns
        -------
        dict
            ``{"result_counts": {...}, "localization_result_counts": {...},
            "thresholded_counts": {str(threshold): {...}}}``. Each leaf is a
            `numpy.ndarray` or ``None``. Threshold keys are stringified so the
            structure round-trips through JSON.
        """
        return {
            "result_counts": dict(self.result_counts),
            "localization_result_counts": dict(self.localization_result_counts),
            "thresholded_counts": {str(t): dict(counts) for t, counts in self.thresholded_counts.items()},
        }

    def load_state(self, state: Dict[str, Dict]) -> None:
        """Restore accumulated counts produced by `get_state`, in place.

        Parameters
        ----------
        state : dict
            State previously produced by `get_state`. Each leaf must be a
            `numpy.ndarray` or ``None``; threshold keys may be strings (as
            serialized) or floats.

        Raises
        ------
        ValueError
            If `state` lacks counts for a threshold in this scorer's
            `thresholds_for_thresholded_metrics`.
        """
        self.result_counts = dict(state["result_counts"])
        self.localization_result_counts = dict(state["localization_result_counts"])

        thresholded = state["thresholded_counts"]
        restored: Dict[float, Dict[str, np.ndarray | None]] = {}
        for t in self.thresholds_for_thresholded_metrics:
            if str(t) in thresholded:
                restored[t] = dict(thresholded[str(t)])
            elif t in thresholded:
                restored[t] = dict(thresholded[t])
            else:
                raise ValueError(f"Checkpointed scorer state is missing counts for threshold {t}.")
        self.thresholded_counts = restored

    @staticmethod
    def _compute_prf_from_counts(
        counts: np.ndarray,
        label_to_idx: Dict[str, int],
        class_mask: np.ndarray | None,
    ) -> Dict:
        """Compute precision, recall, and F1 from per-class TP/FP/FN counts.

        Parameters
        ----------
        counts : np.ndarray of shape (C, 3)
            Per-class [TP, FP, FN] counts.
        label_to_idx : dict[str, int]
            Mapping from class label to index.
        class_mask : np.ndarray of shape (C,) or None
            Boolean mask selecting classes included in the macro average.
            If None, all classes are included.

        Returns
        -------
        dict
            Keys: ``precision_per_class``, ``recall_per_class``, ``f1_per_class``
            (each a dict from label to float), plus ``macro_precision``,
            ``macro_recall``, ``macro_f1``.
        """
        tp = counts[:, 0]
        fp = counts[:, 1]
        fn = counts[:, 2]

        with np.errstate(invalid="ignore", divide="ignore"):
            prec_denom = tp + fp
            precision = np.where(prec_denom > 0, tp / prec_denom, 1.0)

            rec_denom = tp + fn
            recall = np.where(rec_denom > 0, tp / rec_denom, 1.0)

            f1_denom = precision + recall
            f1 = np.where(f1_denom > 0, 2.0 * precision * recall / f1_denom, 0.0)

        idx_to_label = {v: k for k, v in label_to_idx.items()}
        n = len(label_to_idx)
        precision_per_class = {idx_to_label[i]: float(precision[i]) for i in range(n)}
        recall_per_class = {idx_to_label[i]: float(recall[i]) for i in range(n)}
        f1_per_class = {idx_to_label[i]: float(f1[i]) for i in range(n)}

        mask = class_mask if class_mask is not None else np.ones(n, dtype=bool)
        macro_precision = float(np.mean(precision[mask])) if mask.any() else float("nan")
        macro_recall = float(np.mean(recall[mask])) if mask.any() else float("nan")
        macro_f1 = float(np.mean(f1[mask])) if mask.any() else float("nan")

        return {
            "precision_per_class": precision_per_class,
            "recall_per_class": recall_per_class,
            "f1_per_class": f1_per_class,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
        }

    def _get_thresholded_f1_results(self, ignore_absent: bool = True) -> Dict:
        """Compute precision, recall, and F1 at each threshold in `thresholds_for_thresholded_metrics`.

        Parameters
        ----------
        ignore_absent : bool
            If True, exclude classes absent from GT from the macro average.

        Returns
        -------
        dict
            ``{"threshold_0.5": {"frame": {...}, "event_iou_0.2": {...}, ...}, ...}``.
        """
        frame_counts_all = self.result_counts["frame_counts"]
        if ignore_absent and frame_counts_all is not None:
            class_mask: np.ndarray | None = (frame_counts_all[:, :, 0] + frame_counts_all[:, :, 2]).max(axis=0) > 0
        else:
            class_mask = None

        results: Dict = {}
        for t in self.thresholds_for_thresholded_metrics:
            tkey = f"threshold_{t}"
            results[tkey] = {}

            frame_counts = self.thresholded_counts[t]["frame_counts"]
            if frame_counts is not None:
                results[tkey]["frame"] = self._compute_prf_from_counts(
                    frame_counts, self.dataset_label_to_idx, class_mask
                )

            for iou in self.iou_thresholds:
                event_counts = self.thresholded_counts[t][f"event_counts_iou_{iou}"]
                if event_counts is not None:
                    results[tkey][f"event_iou_{iou}"] = self._compute_prf_from_counts(
                        event_counts, self.dataset_label_to_idx, class_mask
                    )

        return results

    def get_results(self, ignore_absent: bool = True) -> Dict[str, Dict]:
        """Compute mAP-style summary metrics from accumulated counts.

        Parameters
        ----------
        ignore_absent : bool
            If True, exclude ground-truth-absent classes from the macro average.

        Returns
        -------
        dict[str, dict]
            Dictionary containing "frame_map", "event_map_iou_{iou}",
            "localization_frame_map", "localization_event_map_iou_{iou}",
            and "thresholded_f1".

        Raises
        ------
        ValueError
            If `update_from_frames` / `update_from_selection_table_by_threshold`
            has not been called at least once, or if only some count arrays are
            populated (e.g. state restored from a checkpoint saved under
            different `iou_thresholds`).
        """
        if self.result_counts["frame_counts"] is None:
            raise ValueError("Scorer must be updated on one file before results are obtained")

        missing = [
            key
            for counts in (self.result_counts, self.localization_result_counts)
            for key, arr in counts.items()
            if arr is None
        ]
        if missing:
            raise ValueError(
                f"Scorer has no accumulated counts for {missing} although other counts are populated. "
                "Updates always populate all counts together, so this state was likely restored from "
                "a checkpoint saved under different frame_eval settings (e.g. iou_thresholds). "
                "Resume with the original eval config, or start a fresh run."
            )

        results: Dict[str, Dict] = {}

        frame_counts = self.result_counts["frame_counts"]
        frame_class_mask = (frame_counts[:, :, 0] + frame_counts[:, :, 2]).max(axis=0) > 0 if ignore_absent else None
        prec_rec_by_threshold = prec_rec_metrics_by_threshold(frame_counts)
        results["frame_map"] = get_map_from_results_by_threshold(
            prec_rec_by_threshold, self.dataset_label_to_idx, class_mask=frame_class_mask
        )

        for iou_threshold in self.iou_thresholds:
            event_counts = self.result_counts[f"event_counts_iou_{iou_threshold}"]
            event_class_mask = (
                (event_counts[:, :, 0] + event_counts[:, :, 2]).max(axis=0) > 0 if ignore_absent else None
            )
            prec_rec_by_threshold = prec_rec_metrics_by_threshold(event_counts)
            results[f"event_map_iou_{iou_threshold}"] = get_map_from_results_by_threshold(
                prec_rec_by_threshold, self.dataset_label_to_idx, class_mask=event_class_mask
            )

        loc_frame_counts = self.localization_result_counts["frame_counts"]
        loc_frame_class_mask = (
            (loc_frame_counts[:, :, 0] + loc_frame_counts[:, :, 2]).max(axis=0) > 0 if ignore_absent else None
        )
        prec_rec_by_threshold = prec_rec_metrics_by_threshold(loc_frame_counts)
        results["localization_frame_map"] = get_map_from_results_by_threshold(
            prec_rec_by_threshold, self.dataset_label_to_idx, class_mask=loc_frame_class_mask
        )

        for iou_threshold in self.iou_thresholds:
            loc_event_counts = self.localization_result_counts[f"event_counts_iou_{iou_threshold}"]
            loc_event_class_mask = (
                (loc_event_counts[:, :, 0] + loc_event_counts[:, :, 2]).max(axis=0) > 0 if ignore_absent else None
            )
            prec_rec_by_threshold = prec_rec_metrics_by_threshold(loc_event_counts)
            results[f"localization_event_map_iou_{iou_threshold}"] = get_map_from_results_by_threshold(
                prec_rec_by_threshold, self.dataset_label_to_idx, class_mask=loc_event_class_mask
            )

        results["thresholded_f1"] = self._get_thresholded_f1_results(ignore_absent)

        return results
