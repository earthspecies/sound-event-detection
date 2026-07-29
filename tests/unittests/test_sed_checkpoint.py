"""Unit tests for SED evaluation checkpointing and resume."""

import numpy as np
import pytest

from sound_event_detection.evaluation.checkpoint import (
    DatasetProgress,
    load_progress,
    load_scorer_state,
    save_progress,
    save_scorer_state,
)
from sound_event_detection.evaluation.metrics import Scorer


def _make_scorer() -> Scorer:
    """Build a small scorer for state round-trip tests.

    Returns
    -------
    Scorer
        A scorer over a 3-class ontology with one IoU and one thresholded metric.
    """
    return Scorer(
        dataset_ontology=["a", "b", "c"],
        n_thresholds=11,
        iou_thresholds=[0.5],
        thresholds_for_thresholded_metrics=[0.5],
    )


def test_progress_latest_record_wins(tmp_path) -> None:
    save_progress(tmp_path, DatasetProgress("frame", "ds:test", 10, is_complete=False, state_file="scorer_state/x.npz"))
    save_progress(
        tmp_path,
        DatasetProgress("frame", "ds:test", 20, is_complete=True, state_file="scorer_state/y.npz", results={"mAP": 0.5}),
    )

    loaded = load_progress(tmp_path)

    assert set(loaded) == {"frame/ds:test"}
    prog = loaded["frame/ds:test"]
    assert prog.n_completed == 20
    assert prog.is_complete
    assert prog.state_file == "scorer_state/y.npz"
    assert prog.results == {"mAP": 0.5}


def test_progress_serializes_numpy_results(tmp_path) -> None:
    results = {"frame_map": {"mAP": np.float32(0.25)}, "class_AP": np.array([0.1, 0.2], dtype=np.float32)}
    save_progress(tmp_path, DatasetProgress("clip", "ds:test", 3, is_complete=True, results=results))

    prog = load_progress(tmp_path)["clip/ds:test"]

    # float32 -> JSON list loses precision; the point is that numpy types serialize.
    assert prog.results["frame_map"]["mAP"] == pytest.approx(0.25)
    assert prog.results["class_AP"] == pytest.approx([0.1, 0.2], abs=1e-6)
    assert isinstance(prog.results["class_AP"], list)


def test_load_progress_missing_dir(tmp_path) -> None:
    assert load_progress(tmp_path / "does_not_exist") == {}


def test_scorer_state_roundtrip(tmp_path) -> None:
    scorer = _make_scorer()
    # Inject accumulated counts to mimic a partially-evaluated dataset.
    scorer.result_counts["frame_counts"] = np.arange(11 * 3 * 3, dtype=np.float32).reshape(11, 3, 3)
    scorer.thresholded_counts[0.5]["frame_counts"] = np.ones((3, 3), dtype=np.float32)

    state_file = save_scorer_state(tmp_path, "ds:test", 5, scorer)
    assert (tmp_path / state_file).exists()

    restored = _make_scorer()
    load_scorer_state(tmp_path, state_file, restored)

    np.testing.assert_array_equal(restored.result_counts["frame_counts"], scorer.result_counts["frame_counts"])
    np.testing.assert_array_equal(
        restored.thresholded_counts[0.5]["frame_counts"], scorer.thresholded_counts[0.5]["frame_counts"]
    )
    # Counts that were never accumulated round-trip back to None, not a stale array.
    assert restored.result_counts["event_counts_iou_0.5"] is None
    assert restored.localization_result_counts["frame_counts"] is None


def test_scorer_state_roundtrip_empty(tmp_path) -> None:
    scorer = _make_scorer()  # all counts None
    state_file = save_scorer_state(tmp_path, "empty", 0, scorer)

    restored = _make_scorer()
    load_scorer_state(tmp_path, state_file, restored)

    assert all(v is None for v in restored.result_counts.values())


def test_load_scorer_state_rejects_mismatched_config(tmp_path) -> None:
    # Snapshot saved under iou [0.2, 0.5]; restoring into a scorer configured
    # with only [0.5] must fail loudly instead of silently dropping counts.
    scorer = Scorer(
        dataset_ontology=["a", "b", "c"],
        n_thresholds=11,
        iou_thresholds=[0.2, 0.5],
        thresholds_for_thresholded_metrics=[0.5],
    )
    scorer.result_counts["event_counts_iou_0.2"] = np.ones((11, 3, 3), dtype=np.float32)
    state_file = save_scorer_state(tmp_path, "ds:test", 5, scorer)

    restored = _make_scorer()  # iou_thresholds=[0.5] only
    with pytest.raises(ValueError, match="does not expect"):
        load_scorer_state(tmp_path, state_file, restored)


def test_get_results_rejects_partially_restored_counts() -> None:
    # Mimics resuming a checkpoint saved under fewer iou_thresholds: frame
    # counts restore but the new threshold's event counts stay None.
    scorer = _make_scorer()
    scorer.result_counts["frame_counts"] = np.ones((11, 3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="different frame_eval settings"):
        scorer.get_results()
