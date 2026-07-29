"""Utility functions for going between raster and list-based annotation formats."""

from typing import Dict, List

import numpy as np
import pandas as pd

from esp_research.protocols.detector import DetectorOutput


def detector_output_to_dataframe(output: DetectorOutput, index: int = 0) -> pd.DataFrame:
    """Return one batch item's detector predictions as a DataFrame.

    Parameters
    ----------
    output : DetectorOutput
        Frame-level detector output with `predictions` of shape ``(batch, time, classes)``.
    index : int
        Index along the batch dimension to extract.

    Returns
    -------
    pd.DataFrame
        DataFrame of shape ``(time, classes)`` with columns named by `output.class_names`.
    """
    return pd.DataFrame(output.predictions[index], columns=list(output.class_names))


def frames_to_dur(x: np.ndarray, frame_rate: float) -> float:
    """Return the duration in seconds for a frame array.

    Parameters
    ----------
    x : np.ndarray
        Array of shape (..., T, C).
    frame_rate : float
        Frame rate in Hz.

    Returns
    -------
    float
        Duration in seconds.
    """
    n_frames = np.shape(x)[-2]
    return n_frames / frame_rate


def frames_to_selection_table(
    x: pd.DataFrame | np.ndarray,
    labels: List,
    frame_rate: float,
    annotation_col: str = "Species",
    probs: np.ndarray | None = None,
) -> pd.DataFrame:
    """Convert per-frame boolean predictions (T x C) into event intervals.

    Detects onsets/offsets of contiguous True runs per class.

    Parameters
    ----------
    x : pd.DataFrame or np.ndarray, shape (T, C)
        Boolean per-frame predictions. Non-bool inputs are cast to bool.
    labels : list
        Sequence of class labels corresponding to the C columns.
    frame_rate : float
        Frame rate in Hz used to convert frame indices to seconds.
    annotation_col : str
        Name of the output column containing class labels.
    probs : np.ndarray or None
        Raw probability array of shape (T, C). When provided, each event gets a
        "Score" column containing the mean probability over its time span.

    Returns
    -------
    pd.DataFrame
        Columns: "Begin Time (s)", "End Time (s)", `annotation_col`, and optionally "Score".

    Raises
    ------
    ValueError
        If `x` is not 2-D or does not have bool dtype.
    RuntimeError
        If start/end detection produces mismatched runs.
    """
    X = np.asarray(x)
    if X.ndim != 2:
        raise ValueError("x must be 2-D (T x C).")

    if X.size == 0:
        return pd.DataFrame(columns=["Begin Time (s)", "End Time (s)", annotation_col])

    if X.dtype != bool:
        raise ValueError(f"X should have dtype = bool, got dtype = {X.dtype}")

    # Pad with a row of False at top and bottom so every run has a start and end
    X_pad = np.pad(X, ((1, 1), (0, 0)), mode="constant", constant_values=False)

    # Transitions: +1 at onsets, -1 at offsets
    D = np.diff(X_pad.view(np.int8), axis=0)  # (T+1, C), values in {-1, 0, +1}

    start_t, start_c = np.where(D == 1)
    end_t, end_c = np.where(D == -1)

    # lexsort to make sure starts match up with ends
    idxstart = np.lexsort((start_t, start_c))
    idxend = np.lexsort((end_t, end_c))
    start_t = start_t[idxstart]
    start_c = start_c[idxstart]
    end_t = end_t[idxend]
    end_c = end_c[idxend]

    if not (start_c.shape == end_c.shape and np.array_equal(start_c, end_c)):
        raise RuntimeError("Mismatched start/end detection; check input array.")

    begin_sec = start_t / frame_rate
    end_sec = end_t / frame_rate
    labels_arr = np.asarray(labels)
    annots = labels_arr[start_c]

    out = pd.DataFrame(
        {
            "Begin Time (s)": begin_sec,
            "End Time (s)": end_sec,
            annotation_col: annots,
        }
    )

    if probs is not None:
        probs = np.asarray(probs)
        assert probs.min() >= 0.0 and probs.max() <= 1.0, (
            f"probs must be in [0, 1], got range [{probs.min()}, {probs.max()}]"
        )
        scores = np.array([probs[s:e, c].mean() for s, e, c in zip(start_t, end_t, start_c, strict=False)])
        out["Score"] = scores

    durs = out["End Time (s)"] - out["Begin Time (s)"]
    if len(durs) > 0:
        if not (durs.min() > 0):
            raise RuntimeError("Negative durs, check input array.")

    out.sort_values(["Begin Time (s)", "End Time (s)"], kind="mergesort", inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def events_array_to_frames(
    begin_times: np.ndarray,
    end_times: np.ndarray,
    label_indices: np.ndarray,
    num_frames: int,
    num_classes: int,
    frame_rate: float,
) -> np.ndarray:
    """Create multi-hot frame-level labels from parallel numpy arrays.

    Parameters
    ----------
    begin_times : np.ndarray
        Event begin times in seconds, shape (N,).
    end_times : np.ndarray
        Event end times in seconds, shape (N,), parallel to begin_times.
    label_indices : np.ndarray
        Integer class index for each event, shape (N,).
    num_frames : int
        Number of output frames.
    num_classes : int
        Total number of classes.
    frame_rate : float
        Frame rate in Hz.

    Returns
    -------
    np.ndarray
        Binary array of shape (num_frames, num_classes), dtype float32.
    """
    frame_labels = np.zeros((num_frames, num_classes), dtype=np.float32)

    if len(begin_times) == 0:
        return frame_labels

    start_frames = np.clip((begin_times * frame_rate).astype(int), 0, num_frames)
    end_frames = np.clip(np.ceil(end_times * frame_rate).astype(int), 0, num_frames)

    for start, end, idx in zip(start_frames, end_frames, label_indices, strict=False):
        frame_labels[start:end, idx] = True

    return frame_labels


def selection_table_to_frames(
    selection_table: pd.DataFrame,
    output_num_frames: int,
    output_frame_rate: float,
    label_to_idx: dict[str, int],
    annotation_col: str = "Species",
) -> np.ndarray:
    """Create multi-hot frame-level labels from a selection table.

    Parameters
    ----------
    selection_table : pd.DataFrame
        Selection table with columns: "Begin Time (s)", "End Time (s)", and `annotation_col`.
    output_num_frames : int
        Duration of output, in frames.
    output_frame_rate : float
        Frame rate for output labels in Hz.
    label_to_idx : dict[str, int]
        Mapping from species name to class index.
    annotation_col : str
        Column name containing species labels.

    Returns
    -------
    np.ndarray
        Binary array of shape (output_num_frames, num_classes) where num_classes is
        len(label_to_idx).
    """
    num_classes = len(label_to_idx)

    if len(selection_table) == 0:
        return np.zeros((output_num_frames, num_classes), dtype=np.float32)

    label_indices = selection_table[annotation_col].map(label_to_idx).values.astype(int)

    return events_array_to_frames(
        begin_times=selection_table["Begin Time (s)"].values,
        end_times=selection_table["End Time (s)"].values,
        label_indices=label_indices,
        num_frames=output_num_frames,
        num_classes=num_classes,
        frame_rate=output_frame_rate,
    )


def frames_to_selection_table_by_threshold(
    preds: np.ndarray,
    class_names: List[str],
    preds_frame_rate: float,
    thresholds: np.ndarray,
    annotation_col: str = "Species",
    probs: np.ndarray | None = None,
) -> Dict[float, pd.DataFrame]:
    """Convert frame-level probabilities into selection tables across a range of thresholds.

    Parameters
    ----------
    preds : np.ndarray
        Frame-level probabilities of shape (T, C).
    class_names : list[str]
        Class names corresponding to the C columns.
    preds_frame_rate : float
        Frame rate in Hz.
    thresholds : np.ndarray
        Detection thresholds in [0, 1].
    annotation_col : str
        Column name for class labels in the output tables.
    probs : np.ndarray or None
        Raw probabilities forwarded to `frames_to_selection_table` for per-event
        Score computation. Typically the same as `preds`.

    Returns
    -------
    dict[float, pd.DataFrame]
        Mapping from threshold → predicted selection table.
    """
    pred_st_by_threshold = {}

    for threshold in thresholds:
        thresholded_preds = preds >= threshold
        st = frames_to_selection_table(
            thresholded_preds,
            class_names,
            preds_frame_rate,
            annotation_col=annotation_col,
            probs=probs,
        )
        pred_st_by_threshold[threshold] = st

    return pred_st_by_threshold
