"""
Post-processing for sound event detection predictions.

Two independent layers, each operating on its own data type:

**Frame-level** (``postprocess_frame_predictions``):
    Median-filters raw probability arrays and optionally slices to a subset
    of classes.  Input/output: ``np.ndarray`` of shape (T, C).

**Selection-table-level** (``postprocess_selection_table_by_threshold``):
    Chains atomic DataFrame → DataFrame transforms on a
    ``dict[float, pd.DataFrame]`` of threshold-keyed selection tables:
    filter_classes → merge_nearby_events → remove_short_events → hard_nms.
    Also available as ``postprocess_selection_table`` for a single table.

This module is a postprocessing utility used by evaluation and inference — it
does not affect training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

# ============= Frame-level postprocessing: ndarray → (ndarray, list[str]) =============


def postprocess_frame_predictions(
    preds: np.ndarray,
    class_names: list[str],
    postprocessing_config: dict | None = None,
    classes_to_keep: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Apply frame-level postprocessing: median filtering and class column slicing.

    Parameters
    ----------
    preds : np.ndarray, shape (T, C)
        Frame-level class probabilities in [0, 1].
    class_names : list[str]
        Class labels corresponding to the C columns.
    postprocessing_config : dict or None
        Optional dict. Recognised key:
            - ``median_filter_size`` (int): kernel size for temporal median
              filtering. Values ≤ 1 or absent → no filtering.
    classes_to_keep : list[str] or None
        If provided, slice columns to only these classes.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        ``(filtered_preds, filtered_class_names)``.
    """
    if postprocessing_config is None:
        postprocessing_config = {}

    median_kernel = postprocessing_config.get("median_filter_size", 0)

    if median_kernel > 1:
        median_kernel = median_kernel | 1  # scipy requires odd kernel size
        preds = median_filter(preds, size=(median_kernel, 1))

    if classes_to_keep is not None:
        keep_set = set(classes_to_keep)
        keep_indices = [i for i, c in enumerate(class_names) if c in keep_set]
        preds = preds[:, keep_indices]
        class_names = [class_names[i] for i in keep_indices]

    return preds, class_names


# ============= Atomic ST-level transforms: DataFrame → DataFrame =============


def filter_classes(
    selection_table: pd.DataFrame,
    classes_to_keep: list[str],
    annotation_col: str = "Species",
) -> pd.DataFrame:
    """Keep only rows whose annotation is in `classes_to_keep`.

    Parameters
    ----------
    selection_table : pd.DataFrame
        Selection table with an `annotation_col` column.
    classes_to_keep : list[str]
        Allowed class labels.
    annotation_col : str
        Column containing class labels.

    Returns
    -------
    pd.DataFrame
        Filtered selection table (copy).
    """
    return selection_table[selection_table[annotation_col].isin(classes_to_keep)].copy().reset_index(drop=True)


def merge_nearby_events(
    selection_table: pd.DataFrame,
    max_gap: float,
    annotation_col: str = "Species",
) -> pd.DataFrame:
    """Merge same-species events whose gap is smaller than `max_gap` seconds.

    Parameters
    ----------
    selection_table : pd.DataFrame
        Must contain "Begin Time (s)", "End Time (s)", and `annotation_col`.
    max_gap : float
        Maximum gap in seconds to trigger a merge. Set to 0 to disable.
    annotation_col : str
        Column containing class labels.

    Returns
    -------
    pd.DataFrame
        Merged selection table (copy, reset index).
    """
    if len(selection_table) == 0 or max_gap <= 0:
        return selection_table.copy().reset_index(drop=True)

    has_score = "Score" in selection_table.columns
    merged_rows: list[dict] = []

    for species, group in selection_table.groupby(annotation_col, sort=False):
        group = group.sort_values("Begin Time (s)").reset_index(drop=True)

        cur_begin = group["Begin Time (s)"].iloc[0]
        cur_end = group["End Time (s)"].iloc[0]
        cur_score = group["Score"].iloc[0] if has_score else None

        for j in range(1, len(group)):
            next_begin = group["Begin Time (s)"].iloc[j]
            next_end = group["End Time (s)"].iloc[j]
            next_score = group["Score"].iloc[j] if has_score else None

            if next_begin - cur_end < max_gap:
                cur_end = max(cur_end, next_end)
                if has_score:
                    cur_score = max(cur_score, next_score)  # type: ignore[arg-type]
            else:
                row = {"Begin Time (s)": cur_begin, "End Time (s)": cur_end, annotation_col: species}
                if has_score:
                    row["Score"] = cur_score
                merged_rows.append(row)
                cur_begin = next_begin
                cur_end = next_end
                cur_score = next_score

        row = {"Begin Time (s)": cur_begin, "End Time (s)": cur_end, annotation_col: species}
        if has_score:
            row["Score"] = cur_score
        merged_rows.append(row)

    return pd.DataFrame(merged_rows).reset_index(drop=True)


def remove_short_events(
    selection_table: pd.DataFrame,
    min_duration: float,
) -> pd.DataFrame:
    """Drop events shorter than `min_duration` seconds.

    Parameters
    ----------
    selection_table : pd.DataFrame
        Must contain "Begin Time (s)" and "End Time (s)".
    min_duration : float
        Minimum event duration in seconds.

    Returns
    -------
    pd.DataFrame
        Filtered selection table (copy, reset index).
    """
    if len(selection_table) == 0 or min_duration <= 0:
        return selection_table.copy().reset_index(drop=True)

    durations = selection_table["End Time (s)"] - selection_table["Begin Time (s)"]
    return selection_table[durations >= min_duration].copy().reset_index(drop=True)


def hard_nms(
    selection_table: pd.DataFrame,
    iou_threshold: float,
) -> pd.DataFrame:
    """Apply hard-NMS to a selection table with temporal IoU.

    Sort events by Score descending, then greedily suppress any remaining
    event whose temporal IoU with a higher-scoring kept event is >=
    ``iou_threshold``. Operates across all classes.

    Parameters
    ----------
    selection_table : pd.DataFrame
        Must contain columns: "Begin Time (s)", "End Time (s)", "Score".
    iou_threshold : float
        Suppress events with temporal IoU >= this value against a
        higher-scoring event.

    Returns
    -------
    pd.DataFrame
        Filtered selection table. Same columns as input.
    """
    if len(selection_table) == 0:
        return selection_table.copy()

    begins = selection_table["Begin Time (s)"].values
    ends = selection_table["End Time (s)"].values
    scores = selection_table["Score"].values

    order = np.argsort(-scores)
    keep_mask = np.ones(len(scores), dtype=bool)

    for i in range(len(order)):
        idx = order[i]
        if not keep_mask[idx]:
            continue

        for j in range(i + 1, len(order)):
            jdx = order[j]
            if not keep_mask[jdx]:
                continue

            inter_start = max(begins[idx], begins[jdx])
            inter_end = min(ends[idx], ends[jdx])
            intersection = max(0.0, inter_end - inter_start)

            if intersection == 0.0:
                continue

            union = (ends[idx] - begins[idx]) + (ends[jdx] - begins[jdx]) - intersection
            iou = intersection / union

            if iou >= iou_threshold:
                keep_mask[jdx] = False

    return selection_table[keep_mask].copy().reset_index(drop=True)


# ============= ST-level transforms: single table and by-threshold =============


def postprocess_selection_table(
    selection_table: pd.DataFrame,
    postprocessing_config: dict | None = None,
    classes_to_keep: list[str] | None = None,
    annotation_col: str = "Species",
) -> pd.DataFrame:
    """Apply selection-table postprocessing to a single table.

    Chains (each step conditional on the relevant config key):
      1. ``filter_classes`` — keep only `classes_to_keep`
      2. ``merge_nearby_events`` — merge same-species events with small gaps
      3. ``remove_short_events`` — drop events below a minimum duration
      4. ``hard_nms`` — suppress overlapping weaker events

    Parameters
    ----------
    selection_table : pd.DataFrame
        Predicted selection table with columns "Begin Time (s)",
        "End Time (s)", `annotation_col`, and "Score" (required for NMS / merge).
    postprocessing_config : dict or None
        Optional dict with keys:
            - ``merge_max_gap`` (float): max gap in seconds for merging.
            - ``min_event_duration`` (float): minimum event duration in seconds.
            - ``nms`` (dict): with key ``iou_threshold`` (float).
    classes_to_keep : list[str] or None
        If provided, filter to only these classes before merge / NMS.
    annotation_col : str
        Column containing class labels.

    Returns
    -------
    pd.DataFrame
        Postprocessed selection table.
    """
    if postprocessing_config is None:
        postprocessing_config = {}

    merge_gap = postprocessing_config.get("merge_max_gap", 0)
    min_dur = postprocessing_config.get("min_event_duration", 0)
    nms_config = postprocessing_config.get("nms", None)

    st = selection_table

    if classes_to_keep is not None:
        st = filter_classes(st, classes_to_keep, annotation_col)

    if merge_gap > 0:
        st = merge_nearby_events(st, max_gap=merge_gap, annotation_col=annotation_col)

    if min_dur > 0:
        st = remove_short_events(st, min_duration=min_dur)

    if nms_config is not None:
        st = hard_nms(st, iou_threshold=nms_config["iou_threshold"])

    return st


def postprocess_selection_table_by_threshold(
    st_by_threshold: dict[float, pd.DataFrame],
    postprocessing_config: dict | None = None,
    classes_to_keep: list[str] | None = None,
    annotation_col: str = "Species",
) -> dict[float, pd.DataFrame]:
    """Apply selection-table postprocessing to every threshold.

    Calls :func:`postprocess_selection_table` on each threshold's table.

    Parameters
    ----------
    st_by_threshold : dict[float, pd.DataFrame]
        Mapping from detection threshold → predicted selection table.
    postprocessing_config : dict or None
        Optional dict with merge/NMS keys (see :func:`postprocess_selection_table`).
    classes_to_keep : list[str] or None
        If provided, filter each table to only these classes before merge / NMS.
    annotation_col : str
        Column containing class labels.

    Returns
    -------
    dict[float, pd.DataFrame]
        Postprocessed selection tables, same keys as input.
    """
    return {
        thr: postprocess_selection_table(
            st,
            postprocessing_config=postprocessing_config,
            classes_to_keep=classes_to_keep,
            annotation_col=annotation_col,
        )
        for thr, st in st_by_threshold.items()
    }
