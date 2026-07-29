"""Tests for inference post-processing (median filtering, merge, remove short, hard-NMS)."""

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import median_filter

from sound_event_detection.utils.postprocessing import (
    filter_classes,
    hard_nms,
    merge_nearby_events,
    postprocess_frame_predictions,
    postprocess_selection_table_by_threshold,
    remove_short_events,
)
from sound_event_detection.utils.reformatters import (
    frames_to_selection_table,
    frames_to_selection_table_by_threshold,
)

# ============= median filtering (scipy.ndimage.median_filter) =============


class TestMedianFilterPredictions:
    """Tests for median_filter with size=(kernel_size, 1) on probability arrays."""

    @staticmethod
    def _filter(preds: np.ndarray, kernel_size: int) -> np.ndarray:
        """Apply per-column median filter, same as postprocess_predictions uses."""
        return median_filter(preds, size=(kernel_size, 1))

    def test_no_op_when_kernel_is_1(self) -> None:
        preds = np.array([[0.8, 0.1], [0.1, 0.9], [0.7, 0.2]], dtype=np.float32)
        result = self._filter(preds, kernel_size=1)
        np.testing.assert_allclose(result, preds, atol=1e-6)

    def test_smooths_single_frame_spike(self) -> None:
        # One isolated high-prob frame surrounded by low-prob should be smoothed
        preds = np.array(
            [[0.0], [0.0], [0.9], [0.0], [0.0]],
            dtype=np.float32,
        )
        result = self._filter(preds, kernel_size=3)
        assert result[2, 0] < 0.5, "Isolated spike should be smoothed below threshold"

    def test_smooths_single_frame_dip(self) -> None:
        # One isolated low-prob frame in a high-prob run should be filled
        preds = np.array(
            [[0.9], [0.9], [0.1], [0.9], [0.9]],
            dtype=np.float32,
        )
        result = self._filter(preds, kernel_size=3)
        assert result[2, 0] > 0.5, "Single-frame dip should be smoothed up"

    def test_preserves_long_runs(self) -> None:
        # A long high-prob run should be preserved
        preds = np.zeros((20, 1), dtype=np.float32)
        preds[5:15, 0] = 0.9
        result = self._filter(preds, kernel_size=3)
        np.testing.assert_allclose(result[6:14, 0], 0.9, atol=1e-6)  # interior preserved
        np.testing.assert_allclose(result[:4, 0], 0.0, atol=1e-6)    # exterior preserved

    def test_classes_are_independent(self) -> None:
        # Filtering one class should not affect another
        preds = np.zeros((10, 2), dtype=np.float32)
        preds[3, 0] = 0.9  # spike in class 0
        preds[2:8, 1] = 0.9  # long run in class 1
        result = self._filter(preds, kernel_size=3)
        assert result[3, 0] < 0.5, "Spike in class 0 should be smoothed"
        np.testing.assert_allclose(result[3:7, 1], 0.9, atol=1e-6)  # interior of class 1 preserved

    def test_output_dtype_matches_input(self) -> None:
        preds = np.array([[0.5], [0.5]], dtype=np.float32)
        result = self._filter(preds, kernel_size=3)
        assert result.dtype == np.float32


# ============= frames_to_selection_table (Score column) =============


class TestFramesToSelectionTableScore:
    def test_score_column_present_when_probs_given(self) -> None:
        binary = np.array(
            [[True, False], [True, False], [False, True]],
            dtype=bool,
        )
        probs = np.array(
            [[0.8, 0.1], [0.9, 0.2], [0.3, 0.7]],
            dtype=np.float32,
        )
        st = frames_to_selection_table(binary, ["a", "b"], frame_rate=1.0, probs=probs)
        assert "Score" in st.columns

    def test_score_column_absent_when_no_probs(self) -> None:
        binary = np.array([[True], [True]], dtype=bool)
        st = frames_to_selection_table(binary, ["a"], frame_rate=1.0)
        assert "Score" not in st.columns

    def test_score_is_mean_probability(self) -> None:
        # Single event spanning frames 0-2 for class 0
        binary = np.array(
            [[True], [True], [True], [False]],
            dtype=bool,
        )
        probs = np.array(
            [[0.6], [0.8], [1.0], [0.1]],
            dtype=np.float32,
        )
        st = frames_to_selection_table(binary, ["a"], frame_rate=1.0, probs=probs)
        assert len(st) == 1
        expected_score = (0.6 + 0.8 + 1.0) / 3
        assert st["Score"].iloc[0] == pytest.approx(expected_score)

    def test_empty_input_with_probs(self) -> None:
        binary = np.zeros((0, 2), dtype=bool)
        probs = np.zeros((0, 2), dtype=np.float32)
        st = frames_to_selection_table(binary, ["a", "b"], frame_rate=1.0, probs=probs)
        assert len(st) == 0


# ============= hard_nms =============


class TestHardNMS:
    def _make_st(self, rows: list[tuple[float, float, str, float]]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=["Begin Time (s)", "End Time (s)", "Species", "Score"])

    def test_empty_table(self) -> None:
        st = self._make_st([])
        result = hard_nms(st, iou_threshold=0.5)
        assert len(result) == 0

    def test_non_overlapping_events_unchanged(self) -> None:
        st = self._make_st([
            (0.0, 1.0, "a", 0.9),
            (2.0, 3.0, "b", 0.8),
        ])
        result = hard_nms(st, iou_threshold=0.5)
        assert len(result) == 2
        # Scores should be unchanged (no overlap)
        assert result["Score"].iloc[0] == pytest.approx(0.9, abs=1e-6)
        assert result["Score"].iloc[1] == pytest.approx(0.8, abs=1e-6)

    def test_heavily_overlapping_weaker_event_killed(self) -> None:
        # Two events identical in time, weaker one should be killed
        st = self._make_st([
            (0.0, 1.0, "a", 0.9),
            (0.0, 1.0, "b", 0.3),
        ])
        result = hard_nms(st, iou_threshold=0.5)
        # IoU = 1.0 >= 0.5, so weaker event is killed
        assert len(result) == 1
        assert result["Species"].iloc[0] == "a"

    def test_iou_just_below_threshold_both_survive(self) -> None:
        # Two events with partial overlap, IoU just below threshold
        # Event A: [0, 2], Event B: [1, 4]
        # intersection = 1.0, union = 4.0, IoU = 0.25
        st = self._make_st([
            (0.0, 2.0, "a", 0.9),
            (1.0, 4.0, "b", 0.8),
        ])
        result = hard_nms(st, iou_threshold=0.3)
        # IoU = 0.25 < 0.3, so both survive
        assert len(result) == 2

    def test_iou_at_threshold_is_suppressed(self) -> None:
        # Two events with IoU exactly at threshold
        # Event A: [0, 2], Event B: [1, 3]
        # intersection = 1.0, union = 3.0, IoU = 1/3
        st = self._make_st([
            (0.0, 2.0, "a", 0.9),
            (1.0, 3.0, "b", 0.8),
        ])
        result = hard_nms(st, iou_threshold=1.0 / 3.0)
        # IoU = 1/3 >= 1/3, so weaker event is killed
        assert len(result) == 1
        assert result["Species"].iloc[0] == "a"

    def test_same_class_overlapping_events(self) -> None:
        # Hard-NMS operates across all classes, including same-class
        st = self._make_st([
            (0.0, 1.0, "a", 0.9),
            (0.0, 1.0, "a", 0.3),
        ])
        result = hard_nms(st, iou_threshold=0.5)
        # The weaker duplicate should be killed
        assert len(result) == 1

    def test_chain_suppression(self) -> None:
        # Three events all overlapping: only the strongest survives
        st = self._make_st([
            (0.0, 1.0, "a", 0.9),
            (0.0, 1.0, "b", 0.7),
            (0.0, 1.0, "c", 0.5),
        ])
        result = hard_nms(st, iou_threshold=0.5)
        assert len(result) == 1
        assert result["Species"].iloc[0] == "a"


# ============= postprocess_frame_predictions =============


class TestPostprocessFramePredictions:
    def test_no_config_returns_unchanged(self) -> None:
        """With no config, preds and class names are returned unchanged."""
        preds = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
        labels = ["a", "b"]

        result_preds, result_labels = postprocess_frame_predictions(preds, labels)

        np.testing.assert_allclose(result_preds, preds, atol=1e-6)
        assert result_labels == labels

    def test_median_filter_removes_blip(self) -> None:
        """A single-frame spike should be smoothed by median filtering."""
        preds = np.zeros((10, 1), dtype=np.float32)
        preds[5, 0] = 0.9  # single-frame blip

        config = {"median_filter_size": 3}
        result_preds, result_labels = postprocess_frame_predictions(
            preds, ["a"], postprocessing_config=config,
        )

        assert result_preds[5, 0] < 0.5, "Spike should be smoothed below threshold"
        assert result_labels == ["a"]

    def test_median_kernel_1_is_noop(self) -> None:
        """Kernel size 1 should not alter predictions (> 1 guard)."""
        preds = np.array([[0.0], [0.9], [0.0]], dtype=np.float32)
        config = {"median_filter_size": 1}

        result_preds, _ = postprocess_frame_predictions(preds, ["a"], postprocessing_config=config)
        np.testing.assert_allclose(result_preds, preds, atol=1e-6)

    def test_classes_to_keep_slices_columns(self) -> None:
        """Should slice array columns to only kept classes."""
        preds = np.random.rand(5, 4).astype(np.float32)
        labels = ["a", "b", "c", "d"]

        result_preds, result_labels = postprocess_frame_predictions(
            preds, labels, classes_to_keep=["b", "d"],
        )

        assert result_labels == ["b", "d"]
        assert result_preds.shape == (5, 2)
        np.testing.assert_allclose(result_preds[:, 0], preds[:, 1], atol=1e-6)  # "b"
        np.testing.assert_allclose(result_preds[:, 1], preds[:, 3], atol=1e-6)  # "d"

    def test_classes_to_keep_none_keeps_all(self) -> None:
        preds = np.random.rand(3, 2).astype(np.float32)
        result_preds, result_labels = postprocess_frame_predictions(preds, ["a", "b"])
        assert result_preds.shape == (3, 2)
        assert result_labels == ["a", "b"]


# ============= filter_classes =============


class TestFilterClasses:
    def _make_st(self, rows: list[tuple]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=["Begin Time (s)", "End Time (s)", "Species", "Score"])

    def test_keeps_only_specified_classes(self) -> None:
        st = self._make_st([
            (0.0, 1.0, "a", 0.9),
            (1.0, 2.0, "b", 0.8),
            (2.0, 3.0, "c", 0.7),
        ])
        result = filter_classes(st, ["a", "c"])
        assert list(result["Species"]) == ["a", "c"]

    def test_empty_table(self) -> None:
        st = self._make_st([])
        result = filter_classes(st, ["a"])
        assert len(result) == 0


# ============= merge_nearby_events =============


class TestMergeNearbyEvents:
    def _make_st(self, rows: list[tuple]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=["Begin Time (s)", "End Time (s)", "Species", "Score"])

    def test_no_merge_when_gap_exceeds_threshold(self) -> None:
        st = self._make_st([
            (0.0, 1.0, "a", 0.9),
            (2.0, 3.0, "a", 0.8),
        ])
        result = merge_nearby_events(st, max_gap=0.5)
        assert len(result) == 2

    def test_merge_when_gap_below_threshold(self) -> None:
        st = self._make_st([
            (0.0, 1.0, "a", 0.7),
            (1.2, 2.0, "a", 0.9),
        ])
        result = merge_nearby_events(st, max_gap=0.5)
        assert len(result) == 1
        assert result["Begin Time (s)"].iloc[0] == pytest.approx(0.0)
        assert result["End Time (s)"].iloc[0] == pytest.approx(2.0)
        assert result["Score"].iloc[0] == pytest.approx(0.9)  # max score

    def test_chain_merge_three_events(self) -> None:
        """Three events each within max_gap of the next → all merge into one."""
        st = self._make_st([
            (0.0, 1.0, "a", 0.5),
            (1.1, 2.0, "a", 0.9),
            (2.1, 3.0, "a", 0.7),
        ])
        result = merge_nearby_events(st, max_gap=0.2)
        assert len(result) == 1
        assert result["Begin Time (s)"].iloc[0] == pytest.approx(0.0)
        assert result["End Time (s)"].iloc[0] == pytest.approx(3.0)
        assert result["Score"].iloc[0] == pytest.approx(0.9)

    def test_different_species_not_merged(self) -> None:
        st = self._make_st([
            (0.0, 1.0, "a", 0.9),
            (1.1, 2.0, "b", 0.8),
        ])
        result = merge_nearby_events(st, max_gap=0.5)
        assert len(result) == 2

    def test_empty_table(self) -> None:
        st = self._make_st([])
        result = merge_nearby_events(st, max_gap=0.5)
        assert len(result) == 0

    def test_zero_max_gap_is_noop(self) -> None:
        st = self._make_st([
            (0.0, 1.0, "a", 0.9),
            (1.0, 2.0, "a", 0.8),
        ])
        result = merge_nearby_events(st, max_gap=0)
        assert len(result) == 2

    def test_without_score_column(self) -> None:
        """Should work without a Score column."""
        st = pd.DataFrame([
            {"Begin Time (s)": 0.0, "End Time (s)": 1.0, "Species": "a"},
            {"Begin Time (s)": 1.1, "End Time (s)": 2.0, "Species": "a"},
        ])
        result = merge_nearby_events(st, max_gap=0.5)
        assert len(result) == 1
        assert "Score" not in result.columns


# ============= remove_short_events =============


class TestRemoveShortEvents:
    def _make_st(self, rows: list[tuple]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=["Begin Time (s)", "End Time (s)", "Species", "Score"])

    def test_removes_short_events(self) -> None:
        st = self._make_st([
            (0.0, 0.05, "a", 0.9),   # 50ms — too short
            (1.0, 2.0, "b", 0.8),    # 1s — long enough
        ])
        result = remove_short_events(st, min_duration=0.1)
        assert len(result) == 1
        assert result["Species"].iloc[0] == "b"

    def test_keeps_events_at_boundary(self) -> None:
        st = self._make_st([
            (0.0, 0.1, "a", 0.9),   # exactly at min_duration
        ])
        result = remove_short_events(st, min_duration=0.1)
        assert len(result) == 1

    def test_empty_table(self) -> None:
        st = self._make_st([])
        result = remove_short_events(st, min_duration=0.1)
        assert len(result) == 0

    def test_zero_min_duration_is_noop(self) -> None:
        st = self._make_st([
            (0.0, 0.01, "a", 0.9),
        ])
        result = remove_short_events(st, min_duration=0)
        assert len(result) == 1


# ============= postprocess_selection_table_by_threshold =============


class TestPostprocessSelectionTableByThreshold:
    def _make_st(self, rows: list[tuple]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=["Begin Time (s)", "End Time (s)", "Species", "Score"])

    def test_no_config_returns_unchanged(self) -> None:
        st = self._make_st([(0.0, 1.0, "a", 0.9)])
        st_by_thr = {0.5: st}
        result = postprocess_selection_table_by_threshold(st_by_thr)
        assert len(result[0.5]) == 1

    def test_nms_suppresses_weak_duplicate(self) -> None:
        """Overlapping events: weak one killed by NMS."""
        st = self._make_st([
            (0.0, 1.0, "a", 0.9),
            (0.0, 1.0, "b", 0.3),
        ])
        config = {"nms": {"iou_threshold": 0.5}}
        result = postprocess_selection_table_by_threshold({0.5: st}, postprocessing_config=config)
        assert len(result[0.5]) == 1
        assert result[0.5]["Species"].iloc[0] == "a"

    def test_class_filter_before_nms(self) -> None:
        """Non-ontology class removed before NMS so it can't suppress valid class."""
        st = self._make_st([
            (0.0, 1.0, "junk", 0.95),
            (0.0, 1.0, "valid", 0.8),
        ])
        config = {"nms": {"iou_threshold": 0.5}}
        result = postprocess_selection_table_by_threshold(
            {0.5: st}, postprocessing_config=config, classes_to_keep=["valid"],
        )
        assert len(result[0.5]) == 1
        assert result[0.5]["Species"].iloc[0] == "valid"

    def test_full_chain(self) -> None:
        """filter → merge → remove_short → NMS all applied in order."""
        st = self._make_st([
            (0.0, 0.5, "a", 0.9),     # will merge with next
            (0.55, 1.0, "a", 0.7),    # gap=0.05 < 0.1, merges → [0, 1.0]
            (0.0, 0.02, "a", 0.6),    # short event, removed by min_duration
            (0.0, 1.0, "junk", 0.95),  # filtered out by classes_to_keep
            (0.0, 1.0, "b", 0.3),     # will be killed by NMS (overlaps merged "a")
        ])
        config = {
            "merge_max_gap": 0.1,
            "min_event_duration": 0.05,
            "nms": {"iou_threshold": 0.5},
        }
        result = postprocess_selection_table_by_threshold(
            {0.5: st}, postprocessing_config=config,
            classes_to_keep=["a", "b"],
        )
        # Should have 1 event: merged "a" [0, 1.0], "b" killed by NMS
        assert len(result[0.5]) == 1
        assert result[0.5]["Species"].iloc[0] == "a"

    def test_multiple_thresholds(self) -> None:
        """Pipeline applied independently to each threshold."""
        preds = np.array([[0.3], [0.7], [0.9]], dtype=np.float32)
        thresholds = np.array([0.2, 0.5, 0.8])

        st_by_thr = frames_to_selection_table_by_threshold(
            preds, ["a"], 1.0, thresholds, probs=preds,
        )
        result = postprocess_selection_table_by_threshold(st_by_thr)

        assert set(result.keys()) == {0.2, 0.5, 0.8}
        # More events at lower thresholds
        assert len(result[0.2]) >= len(result[0.5])
        assert len(result[0.5]) >= len(result[0.8])


# ============= End-to-end pipeline =============


class TestEndToEndPipeline:
    def test_frame_then_st_postprocessing(self) -> None:
        """Smoke test: postprocess_frame_predictions → by_threshold → postprocess_selection_table_by_threshold."""
        preds = np.zeros((20, 3), dtype=np.float32)
        preds[2:8, 0] = 0.9   # class "a" active
        preds[5, 0] = 0.1     # single-frame dip (will be filled by median filter)
        preds[2:8, 1] = 0.6   # class "b" overlaps with "a"
        preds[15:18, 2] = 0.8  # class "c" separate

        labels = ["a", "b", "c"]
        config = {"median_filter_size": 3, "nms": {"iou_threshold": 0.8}}

        # Step 1: frame-level
        filtered_preds, filtered_labels = postprocess_frame_predictions(
            preds, labels, postprocessing_config=config, classes_to_keep=["a", "c"],
        )
        assert filtered_labels == ["a", "c"]
        assert filtered_preds.shape[1] == 2

        # Step 2: frames → STs
        st_by_thr = frames_to_selection_table_by_threshold(
            filtered_preds, filtered_labels, 1.0, np.array([0.5]),
            probs=filtered_preds,
        )

        # Step 3: ST-level postprocessing
        result = postprocess_selection_table_by_threshold(
            st_by_thr, postprocessing_config=config,
            classes_to_keep=["a", "c"],
        )
        assert 0.5 in result
        species = set(result[0.5]["Species"])
        assert species <= {"a", "c"}, "Only kept classes should remain"


# ============= frames_to_selection_table_by_threshold probs forwarding =============


class TestByThresholdProbs:
    def test_score_column_present_when_probs_given(self) -> None:
        preds = np.array([[0.8, 0.1], [0.9, 0.2]], dtype=np.float32)
        result = frames_to_selection_table_by_threshold(
            preds, ["a", "b"], 1.0, np.array([0.5]), probs=preds,
        )
        assert "Score" in result[0.5].columns

    def test_score_column_absent_when_no_probs(self) -> None:
        preds = np.array([[0.8], [0.9]], dtype=np.float32)
        result = frames_to_selection_table_by_threshold(
            preds, ["a"], 1.0, np.array([0.5]),
        )
        assert "Score" not in result[0.5].columns
