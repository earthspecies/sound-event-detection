import numpy as np
import pandas as pd

from sound_event_detection.utils.reformatters import (
    events_array_to_frames,
    frames_to_selection_table,
    selection_table_to_frames,
)

# ============= events_array_to_frames tests =============


def test_events_array_to_frames_empty_returns_zeros() -> None:
    out = events_array_to_frames(
        begin_times=np.array([]),
        end_times=np.array([]),
        label_indices=np.array([], dtype=np.intp),
        num_frames=10,
        num_classes=3,
        frame_rate=5.0,
    )
    assert out.shape == (10, 3)
    assert out.dtype == np.float32
    assert np.all(out == 0.0)


def test_events_array_to_frames_single_event() -> None:
    # One event: class 1, 0.2s–0.6s at 10 Hz → frames 2–6
    out = events_array_to_frames(
        begin_times=np.array([0.2]),
        end_times=np.array([0.6]),
        label_indices=np.array([1], dtype=np.intp),
        num_frames=10,
        num_classes=3,
        frame_rate=10.0,
    )
    assert out.shape == (10, 3)
    assert np.all(out[:2, 1] == 0.0)
    assert np.all(out[2:6, 1] == 1.0)
    assert np.all(out[6:, 1] == 0.0)
    assert np.all(out[:, 0] == 0.0)
    assert np.all(out[:, 2] == 0.0)


def test_events_array_to_frames_clamps_to_num_frames() -> None:
    # Event that extends past the end of the array
    out = events_array_to_frames(
        begin_times=np.array([0.8]),
        end_times=np.array([2.0]),  # past end of 1-second / 10-frame window
        label_indices=np.array([0], dtype=np.intp),
        num_frames=10,
        num_classes=2,
        frame_rate=10.0,
    )
    assert np.all(out[8:10, 0] == 1.0)
    assert np.all(out[:8, 0] == 0.0)


def test_events_array_to_frames_matches_selection_table_to_frames() -> None:
    # Both paths must produce identical results for the same logical input
    frame_rate = 5.0
    label_to_idx = {"a": 0, "b": 1}
    st = pd.DataFrame({
        "Begin Time (s)": [0.0, 0.6],
        "End Time (s)":   [0.4, 1.2],
        "Species":        ["a", "b"],
    })

    expected = selection_table_to_frames(
        selection_table=st,
        output_num_frames=10,
        output_frame_rate=frame_rate,
        label_to_idx=label_to_idx,
        annotation_col="Species",
    )

    actual = events_array_to_frames(
        begin_times=st["Begin Time (s)"].values,
        end_times=st["End Time (s)"].values,
        label_indices=np.array([label_to_idx[s] for s in st["Species"]], dtype=np.intp),
        num_frames=10,
        num_classes=2,
        frame_rate=frame_rate,
    )

    np.testing.assert_array_equal(actual, expected)


def test_events_array_to_frames_exported_from_utils() -> None:
    from sound_event_detection.utils import events_array_to_frames as f
    assert callable(f)


# ============= frames_to_selection_table tests =============


def test_frames_to_selection_table_empty_input_returns_empty_df_with_columns() -> None:
    x = np.zeros((0, 3), dtype=bool)
    out = frames_to_selection_table(x, labels=["a", "b", "c"], frame_rate=10.0, annotation_col="Species")
    assert list(out.columns) == ["Begin Time (s)", "End Time (s)", "Species"]
    assert len(out) == 0


def test_frames_to_selection_table_detects_contiguous_runs_multiple_classes_and_sorts() -> None:
    # T=6, C=2, frame_hz=2 => each frame = 0.5 sec
    # Class 0: True on frames [1,2] => [0.5, 1.5)
    #          True on frame [4]    => [2.0, 2.5)
    # Class 1: True on frames [0]   => [0.0, 0.5)
    #          True on frames [3,4,5] => [1.5, 3.0)
    x = np.array(
        [
            [False, True],
            [True, False],
            [True, False],
            [False, True],
            [True, True],
            [False, True],
        ],
        dtype=bool,
    )
    labels = ["c0", "c1"]
    out = frames_to_selection_table(x, labels=labels, frame_rate=2.0, annotation_col="Species")

    expected = pd.DataFrame(
        {
            "Begin Time (s)": [0.0, 0.5, 1.5, 2.0],
            "End Time (s)":   [0.5, 1.5, 3.0, 2.5],
            "Species":        ["c1", "c0", "c1", "c0"],
        }
    )

    # exact equality after sorting/reset is expected
    pd.testing.assert_frame_equal(out, expected, check_dtype=False)


def test_selection_table_to_frames_empty_returns_zeros() -> None:
    st = pd.DataFrame(columns=["Begin Time (s)", "End Time (s)", "Species"])
    out = selection_table_to_frames(
        selection_table=st,
        output_num_frames=10,
        output_frame_rate=5.0,
        label_to_idx={"a": 0, "b": 1},
        annotation_col="Species",
    )
    assert out.shape == (10, 2)
    assert out.dtype == np.float32
    assert np.all(out == 0.0)


def test_selection_table_to_frames_overlapping_events_same_class_union() -> None:
    # Two overlapping events for class 'a' should result in union in frames
    st = pd.DataFrame(
        {
            "Begin Time (s)": [0.0, 0.4],
            "End Time (s)": [0.6, 1.0],
            "Species": ["a", "a"],
        }
    )
    out = selection_table_to_frames(
        selection_table=st,
        output_num_frames=10,
        output_frame_rate=10.0,  # 0.1 sec/frame
        label_to_idx={"a": 0},
        annotation_col="Species",
    )
    # Event1: [0.0,0.6] -> frames [0:int?] end ceil(6)=6 => [0:6]
    # Event2: [0.4,1.0] -> start int(4)=4, end ceil(10)=10 => [4:10]
    expected = np.zeros((10, 1), dtype=np.float32)
    expected[0:10, 0] = 1.0
    np.testing.assert_array_equal(out, expected)


def test_round_trip_frames_to_selection_table_to_frames_exact_on_frame_boundaries() -> None:
    """
    Round-trip property (exact) for frame-aligned events:

        frames(bool) -> selection_table -> frames(float32)

    This is exact when event boundaries land on frame indices (which is how
    frames_to_selection_table defines begin/end), and selection_table_to_frames
    uses int(begin*hz) and ceil(end*hz).
    """
    frame_hz = 4.0  # 0.25s per frame
    T = 12
    labels = ["a", "b", "c"]
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    # Construct frame-level truth with several disjoint contiguous runs per class.
    # All boundaries are exact frame boundaries by construction.
    X = np.zeros((T, len(labels)), dtype=bool)

    # class a: [1,4) and [9,11)
    X[1:4, 0] = True
    X[9:11, 0] = True

    # class b: [0,1) and [6,9)
    X[0:1, 1] = True
    X[6:9, 1] = True

    # class c: empty (no events)

    st = frames_to_selection_table(X, labels=labels, frame_rate=frame_hz, annotation_col="Species")
    X_rt = selection_table_to_frames(
        selection_table=st,
        output_num_frames=T,
        output_frame_rate=frame_hz,
        label_to_idx=label_to_idx,
        annotation_col="Species",
    )

    # selection_table_to_frames returns float32 {0.0, 1.0}; compare to original mask.
    np.testing.assert_array_equal(X_rt, X.astype(np.float32))


def test_round_trip_selection_table_to_frames_to_selection_table_exact_on_frame_boundaries() -> None:
    """
    Round-trip property (exact) for a frame-aligned selection table:

        selection_table -> frames -> selection_table

    This is exact when begin/end are multiples of 1/frame_hz, and there are no overlapping events from one class
    """
    frame_hz = 5.0  # 0.2s per frame
    T = 20
    labels = ["a", "b"]
    label_to_idx = {"a": 0, "b": 1}

    # Begin/End are exact frame boundaries (multiples of 0.2s).
    st = pd.DataFrame(
        {
            "Begin Time (s)": [0.0, 0.4, 2.0, 2.6],
            "End Time (s)":   [0.2, 1.0, 2.4, 3.0],
            "Species":        ["b", "a", "a", "b"],
        }
    )

    frames = selection_table_to_frames(
        selection_table=st,
        output_num_frames=T,
        output_frame_rate=frame_hz,
        label_to_idx=label_to_idx,
        annotation_col="Species",
    )

    # Convert back. frames_to_selection_table requires bool dtype.
    st_rt = frames_to_selection_table(frames.astype(bool), labels=labels, frame_rate=frame_hz, annotation_col="Species")

    # Canonicalize: sort and reset index for stable comparison
    st_sorted = st.sort_values(["Begin Time (s)", "End Time (s)", "Species"], kind="mergesort").reset_index(drop=True)
    st_rt_sorted = st_rt.sort_values(["Begin Time (s)", "End Time (s)", "Species"], kind="mergesort").reset_index(drop=True)

    pd.testing.assert_frame_equal(st_rt_sorted, st_sorted, check_dtype=False)
