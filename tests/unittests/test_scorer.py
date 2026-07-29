"""Tests for the Scorer class in sound_event_detection.evaluation.metrics."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from sound_event_detection.evaluation.counting import get_tpfpfn_counts
from sound_event_detection.evaluation.metrics import Scorer
from sound_event_detection.utils.reformatters import frames_to_selection_table_by_threshold

ANNOTATION_COL = "Species"


def _st(rows: List[tuple[float, float, str]]) -> pd.DataFrame:
    """Build a Raven-style selection table from (begin, end, label) tuples."""
    return pd.DataFrame(rows, columns=["Begin Time (s)", "End Time (s)", ANNOTATION_COL])


def _make_frame_predictions(
    *,
    duration_sec: float,
    frame_rate: float,
    labels: List[str],
    events: Dict[str, List[tuple[float, float, float]]],
    background_score: float = 0.0,
) -> np.ndarray:
    """
    Create a dense framewise prediction array shaped (T, C).

    events maps label -> list of (begin_sec, end_sec, score).
    """
    T = int(round(duration_sec * frame_rate))
    C = len(labels)
    out = np.full((T, C), background_score, dtype=np.float32)

    label_to_col = {lab: i for i, lab in enumerate(labels)}
    for lab, spans in events.items():
        j = label_to_col[lab]
        for b, e, s in spans:
            t_b = int(np.floor(b * frame_rate))
            t_e = int(np.ceil(e * frame_rate))
            t_b = max(t_b, 0)
            t_e = min(t_e, T)
            out[t_b:t_e, j] = s
    return out


@pytest.fixture
def thresholds_3() -> np.ndarray:
    # Matches Scorer(n_thresholds=3) default linspace rounding: [0.0, 0.5, 1.0]
    return np.array([0.0, 0.5, 1.0], dtype=np.float32)


def test_more_than_two_thresholds_shape_and_keys(thresholds_3: np.ndarray) -> None:
    """Scorer should accumulate counts across >2 thresholds with correct shape."""
    scorer = Scorer(
        dataset_ontology=["A", "B"],
        n_thresholds=3,
        annotation_col=ANNOTATION_COL,
        discretization_frame_rate=10.0,
        iou_thresholds=[0.2, 0.5],
    )

    duration = 2.0
    fr = 10.0
    labels = ["A", "B"]
    preds = _make_frame_predictions(
        duration_sec=duration,
        frame_rate=fr,
        labels=labels,
        events={"A": [(0.2, 1.2, 0.9)], "B": [(0.5, 1.5, 0.8)]},
        background_score=0.1,
    )
    gt = _st([(0.2, 1.2, "A"), (0.5, 1.5, "B")])

    pred_st_by_thr = frames_to_selection_table_by_threshold(
        preds, labels, fr, scorer.thresholds, annotation_col=ANNOTATION_COL
    )

    scorer.update_from_selection_table_by_threshold(pred_st_by_thr, gt, duration_sec=duration)

    assert scorer.result_counts["frame_counts"] is not None
    assert scorer.result_counts["frame_counts"].shape[0] == 3  # n_thresholds
    assert scorer.result_counts["frame_counts"].shape[1] == 2  # C
    assert scorer.result_counts["frame_counts"].shape[2] == 3  # TP/FP/FN

    for iou in scorer.iou_thresholds:
        k = f"event_counts_iou_{iou}"
        assert scorer.result_counts[k] is not None
        assert scorer.result_counts[k].shape == (3, 2, 3)


def test_multi_class_correct_scores_perfect_predictions() -> None:
    """
    With >1 classes and perfect predictions, both frame_map and event_map should be 1.0.
    """
    scorer = Scorer(
        dataset_ontology=["A", "B"],
        n_thresholds=3,
        annotation_col=ANNOTATION_COL,
        discretization_frame_rate=10.0,
        iou_thresholds=[0.2, 0.5],
    )

    duration = 2.0
    fr = 10.0
    labels = ["A", "B"]

    preds = _make_frame_predictions(
        duration_sec=duration,
        frame_rate=fr,
        labels=labels,
        events={"A": [(0.2, 1.2, 0.9)], "B": [(0.5, 1.5, 0.9)]},
        background_score=0.0,
    )
    gt = _st([(0.2, 1.2, "A"), (0.5, 1.5, "B")])

    pred_st_by_thr = frames_to_selection_table_by_threshold(
        preds, labels, fr, scorer.thresholds, annotation_col=ANNOTATION_COL
    )
    scorer.update_from_selection_table_by_threshold(pred_st_by_thr, gt, duration_sec=duration)

    res = scorer.get_results()
    assert res["frame_map"]['mAP'] == pytest.approx(1.0, abs=1e-6)
    assert res["event_map_iou_0.2"]['mAP'] == pytest.approx(1.0, abs=1e-6)
    assert res["event_map_iou_0.5"]['mAP'] == pytest.approx(1.0, abs=1e-6)


def test_multi_file_updates_accumulate_counts() -> None:
    """
    Calling update multiple times should accumulate counts; perfect predictions remain perfect.
    """
    scorer = Scorer(
        dataset_ontology=["A", "B"],
        n_thresholds=3,
        annotation_col=ANNOTATION_COL,
        discretization_frame_rate=10.0,
        iou_thresholds=[0.2, 0.5],
    )

    duration = 2.0
    fr = 10.0
    labels = ["A", "B"]

    # File 1
    preds1 = _make_frame_predictions(
        duration_sec=duration,
        frame_rate=fr,
        labels=labels,
        events={"A": [(0.2, 1.0, 0.9)], "B": [(1.0, 1.8, 0.9)]},
        background_score=0.0,
    )
    gt1 = _st([(0.2, 1.0, "A"), (1.0, 1.8, "B")])
    pred_st_by_thr1 = frames_to_selection_table_by_threshold(
        preds1, labels, fr, scorer.thresholds, annotation_col=ANNOTATION_COL
    )
    scorer.update_from_selection_table_by_threshold(pred_st_by_thr1, gt1, duration_sec=duration)

    frame_counts_after_1 = scorer.result_counts["frame_counts"].copy()
    event_counts_after_1 = {iou: scorer.result_counts[f"event_counts_iou_{iou}"].copy() for iou in scorer.iou_thresholds}

    # File 2
    preds2 = _make_frame_predictions(
        duration_sec=duration,
        frame_rate=fr,
        labels=labels,
        events={"A": [(0.0, 0.5, 0.9)], "B": [(0.5, 1.0, 0.9)]},
        background_score=0.0,
    )
    gt2 = _st([(0.0, 0.5, "A"), (0.5, 1.0, "B")])
    pred_st_by_thr2 = frames_to_selection_table_by_threshold(
        preds2, labels, fr, scorer.thresholds, annotation_col=ANNOTATION_COL
    )
    scorer.update_from_selection_table_by_threshold(pred_st_by_thr2, gt2, duration_sec=duration)

    # Counts should have increased (roughly doubled in aggregate sense).
    assert scorer.result_counts["frame_counts"].sum() > frame_counts_after_1.sum()
    for iou in scorer.iou_thresholds:
        k = f"event_counts_iou_{iou}"
        assert scorer.result_counts[k].sum() > event_counts_after_1[iou].sum()

    # Still perfect overall.
    res = scorer.get_results()

    assert res["frame_map"]['mAP'] == pytest.approx(1.0, abs=1e-6)
    assert res["event_map_iou_0.2"]['mAP'] == pytest.approx(1.0, abs=1e-6)
    assert res["event_map_iou_0.5"]['mAP'] == pytest.approx(1.0, abs=1e-6)


def test_iou_thresholds_match_at_02_not_at_05() -> None:
    """
    Construct a case where IoU ~= 0.4 so it matches at 0.2 but not at 0.5.
    """
    scorer = Scorer(
        dataset_ontology=["A"],
        n_thresholds=3,
        annotation_col=ANNOTATION_COL,
        discretization_frame_rate=10.0,
        iou_thresholds=[0.2, 0.5],
    )

    duration = 2.0

    # GT event: [0.0, 1.0]
    gt = _st([(0.0, 1.0, "A")])

    # Pred event: [0.0, 0.4]
    # IoU = intersection 0.4 / union 1.0 = 0.4  => >=0.2 match, <0.5 no match.
    pred_st_by_thr: Dict[float, pd.DataFrame] = {
        thr: _st([(0.0, 0.4, "A")]) for thr in scorer.thresholds
    }

    # Update and then directly validate the event-count arrays using the same counting primitive.
    scorer.update_from_selection_table_by_threshold(pred_st_by_thr, gt, duration_sec=duration)

    # Pick one threshold (they're identical here), check that at IoU=0.2 we get TP=1, FP=0, FN=0,
    # while at IoU=0.5 we get TP=0, FP=1, FN=1.
    thr0 = float(scorer.thresholds[0])

    expected_02 = get_tpfpfn_counts(
        gt, pred_st_by_thr[thr0], 0.2, scorer.dataset_label_to_idx, annotation_col=ANNOTATION_COL
    )
    expected_05 = get_tpfpfn_counts(
        gt, pred_st_by_thr[thr0], 0.5, scorer.dataset_label_to_idx, annotation_col=ANNOTATION_COL
    )

    # Sanity: expected arrays are (C,3) with C=1.
    assert expected_02.shape == (1, 3)
    assert expected_05.shape == (1, 3)

    # Validate accumulated arrays match the per-threshold expected values (they're constant over thresholds).
    # result_counts is (n_thresholds, C, 3)
    np.testing.assert_allclose(scorer.result_counts["event_counts_iou_0.2"][0], expected_02.astype(np.float32))
    np.testing.assert_allclose(scorer.result_counts["event_counts_iou_0.5"][0], expected_05.astype(np.float32))

    # Additionally ensure the two IoU settings differ in the expected direction.
    tp_02, fp_02, fn_02 = scorer.result_counts["event_counts_iou_0.2"][0, 0]
    tp_05, fp_05, fn_05 = scorer.result_counts["event_counts_iou_0.5"][0, 0]
    assert tp_02 == 1.0
    assert fn_02 == 0.0
    assert tp_05 == 0.0
    assert fn_05 == 1.0


def test_predictions_outside_ontology_are_dropped_update_from_selection_table_by_threshold() -> None:
    """
    If predicted labels include classes outside the dataset ontology, they should be dropped
    """
    scorer = Scorer(
        dataset_ontology=["A"],
        n_thresholds=3,
        annotation_col=ANNOTATION_COL,
        discretization_frame_rate=10.0,
        iou_thresholds=[0.2, 0.5],
    )

    duration = 2.0
    gt = _st([(0.0, 1.0, "A")])

    # Predictions include an extra class "C" outside ontology.
    pred_with_extra: Dict[float, pd.DataFrame] = {
        thr: _st([(0.0, 1.0, "A"), (0.0, 2.0, "C")]) for thr in scorer.thresholds
    }
    pred_filtered: Dict[float, pd.DataFrame] = {
        thr: _st([(0.0, 1.0, "A")]) for thr in scorer.thresholds
    }

    scorer_extra = Scorer(
        dataset_ontology=["A"],
        n_thresholds=3,
        annotation_col=ANNOTATION_COL,
        discretization_frame_rate=10.0,
        iou_thresholds=[0.2, 0.5],
    )
    scorer_filtered = Scorer(
        dataset_ontology=["A"],
        n_thresholds=3,
        annotation_col=ANNOTATION_COL,
        discretization_frame_rate=10.0,
        iou_thresholds=[0.2, 0.5],
    )

    scorer_extra.update_from_selection_table_by_threshold(pred_with_extra, gt, duration_sec=duration)
    scorer_filtered.update_from_selection_table_by_threshold(pred_filtered, gt, duration_sec=duration)

    assert scorer_extra.get_results() == scorer_filtered.get_results()
    np.testing.assert_allclose(scorer_extra.result_counts["frame_counts"], scorer_filtered.result_counts["frame_counts"])
    for iou in scorer_extra.iou_thresholds:
        k = f"event_counts_iou_{iou}"
        np.testing.assert_allclose(scorer_extra.result_counts[k], scorer_filtered.result_counts[k])


def test_ignore_absent_excludes_absent_classes_from_map() -> None:
    """
    With ignore_absent=True, classes with no GT positives are excluded from the mAP macro-average.
    AP_per_class should still contain entries for all classes regardless.
    """
    scorer = Scorer(
        dataset_ontology=["A", "B", "C"],
        n_thresholds=3,
        annotation_col=ANNOTATION_COL,
        discretization_frame_rate=10.0,
        iou_thresholds=[0.5],
    )

    duration = 2.0
    fr = 10.0
    labels = ["A", "B", "C"]

    # Predictions for A and B; C has predictions but no GT (absent class).
    preds = _make_frame_predictions(
        duration_sec=duration,
        frame_rate=fr,
        labels=labels,
        events={"A": [(0.0, 1.0, 0.9)], "B": [(1.0, 1.5, 0.9)], "C": [(0.0, 1.0, 0.9)]},
        background_score=0.0,
    )
    gt = _st([(0.0, 1.0, "A"), (1.0, 2.0, "B")])  # no GT for C

    pred_st_by_thr = frames_to_selection_table_by_threshold(
        preds, labels, fr, scorer.thresholds, annotation_col=ANNOTATION_COL
    )
    scorer.update_from_selection_table_by_threshold(pred_st_by_thr, gt, duration_sec=duration)

    res_ignore = scorer.get_results(ignore_absent=True)
    res_all = scorer.get_results(ignore_absent=False)

    # AP_per_class should contain all 3 classes in both cases (mask only affects mAP).
    for res in [res_ignore, res_all]:
        assert set(res["frame_map"]["AP_per_class"].keys()) == {"A", "B", "C"}

    ap_a = res_ignore["frame_map"]["AP_per_class"]["A"]
    ap_b = res_ignore["frame_map"]["AP_per_class"]["B"]
    ap_c = res_ignore["frame_map"]["AP_per_class"]["C"]

    # ignore_absent=True: mAP is the mean of present classes (A and B) only.
    assert res_ignore["frame_map"]["mAP"] == pytest.approx((ap_a + ap_b) / 2, abs=1e-6)

    # ignore_absent=False: mAP includes C.
    assert res_all["frame_map"]["mAP"] == pytest.approx((ap_a + ap_b + ap_c) / 3, abs=1e-6)

    # The two mAPs must differ since absent C's AP != mean(AP_A, AP_B).
    assert res_ignore["frame_map"]["mAP"] != pytest.approx(res_all["frame_map"]["mAP"])


# ---------------------------------------------------------------------------
# Thresholded F1 / precision / recall helper
# ---------------------------------------------------------------------------

def _make_scorer(*, n_thresholds: int = 3, thresholds_for_thresholded_metrics=(0.5,)) -> Scorer:
    return Scorer(
        dataset_ontology=["A", "B"],
        n_thresholds=n_thresholds,
        annotation_col=ANNOTATION_COL,
        discretization_frame_rate=10.0,
        iou_thresholds=[0.5],
        thresholds_for_thresholded_metrics=thresholds_for_thresholded_metrics,
    )


# ---------------------------------------------------------------------------
# Thresholded F1 / precision / recall tests
# ---------------------------------------------------------------------------


def test_thresholded_f1_keys_present() -> None:
    """get_results should include a 'thresholded_f1' key with expected sub-structure."""
    scorer = _make_scorer(thresholds_for_thresholded_metrics=[0.5])
    duration = 60.0
    gt = _st([(0.0, 1.0, "A")])
    pred_st = gt.copy()
    pred_st_by_thr = {thr: pred_st.copy() for thr in scorer.thresholds}

    scorer.update_from_selection_table_by_threshold(
        pred_st_by_thr, gt, duration_sec=duration,
        selection_tables_for_thresholded_metrics={0.5: pred_st},
    )
    res = scorer.get_results()

    assert "thresholded_f1" in res
    assert "threshold_0.5" in res["thresholded_f1"]
    thr_res = res["thresholded_f1"]["threshold_0.5"]
    assert "frame" in thr_res
    assert "event_iou_0.5" in thr_res

    for subkey in ("frame", "event_iou_0.5"):
        m = thr_res[subkey]
        assert "precision_per_class" in m
        assert "recall_per_class" in m
        assert "f1_per_class" in m
        assert "macro_precision" in m
        assert "macro_recall" in m
        assert "macro_f1" in m
        assert isinstance(m["macro_f1"], float)
        assert set(m["f1_per_class"].keys()) == {"A", "B"}


def test_thresholded_f1_perfect_predictions() -> None:
    """Perfect predictions should yield macro_f1 == 1.0 for both frame and event metrics."""
    scorer = Scorer(
        dataset_ontology=["A", "B"],
        n_thresholds=3,
        annotation_col=ANNOTATION_COL,
        discretization_frame_rate=10.0,
        iou_thresholds=[0.2, 0.5],
        thresholds_for_thresholded_metrics=[0.5],
    )
    duration = 2.0
    fr = 10.0
    labels = ["A", "B"]
    preds = _make_frame_predictions(
        duration_sec=duration,
        frame_rate=fr,
        labels=labels,
        events={"A": [(0.2, 1.2, 0.9)], "B": [(0.5, 1.5, 0.9)]},
        background_score=0.0,
    )
    gt = _st([(0.2, 1.2, "A"), (0.5, 1.5, "B")])

    pred_st_by_thr = frames_to_selection_table_by_threshold(
        preds, labels, fr, scorer.thresholds, annotation_col=ANNOTATION_COL
    )
    metric_st = frames_to_selection_table_by_threshold(
        preds, labels, fr, np.array([0.5]), annotation_col=ANNOTATION_COL
    )
    scorer.update_from_selection_table_by_threshold(
        pred_st_by_thr, gt, duration_sec=duration,
        selection_tables_for_thresholded_metrics=metric_st,
    )

    res = scorer.get_results()
    thr_res = res["thresholded_f1"]["threshold_0.5"]
    assert thr_res["frame"]["macro_f1"] == pytest.approx(1.0, abs=1e-6)
    assert thr_res["event_iou_0.2"]["macro_f1"] == pytest.approx(1.0, abs=1e-6)
    assert thr_res["event_iou_0.5"]["macro_f1"] == pytest.approx(1.0, abs=1e-6)


def test_thresholded_f1_zero_recall_all_missed() -> None:
    """With no predictions at threshold 0.5, recall should be 0.0 and F1 should be 0.0."""
    scorer = _make_scorer(thresholds_for_thresholded_metrics=[0.5])
    duration = 60.0
    gt = _st([(0.0, 1.0, "A")])
    empty = _st([])
    pred_st_by_thr = {thr: empty.copy() for thr in scorer.thresholds}

    scorer.update_from_selection_table_by_threshold(
        pred_st_by_thr, gt, duration_sec=duration,
        selection_tables_for_thresholded_metrics={0.5: empty},
    )
    res = scorer.get_results()
    m = res["thresholded_f1"]["threshold_0.5"]["frame"]

    assert m["recall_per_class"]["A"] == pytest.approx(0.0, abs=1e-6)
    assert m["f1_per_class"]["A"] == pytest.approx(0.0, abs=1e-6)


def test_thresholded_f1_multi_threshold() -> None:
    """Results should be reported separately for each threshold in thresholds_for_thresholded_metrics."""
    scorer = _make_scorer(n_thresholds=3, thresholds_for_thresholded_metrics=[0.3, 0.7])
    duration = 60.0
    gt = _st([(0.0, 1.0, "A")])
    pred = gt.copy()
    empty = _st([])
    pred_st_by_thr = {thr: pred.copy() for thr in scorer.thresholds}

    scorer.update_from_selection_table_by_threshold(
        pred_st_by_thr, gt, duration_sec=duration,
        selection_tables_for_thresholded_metrics={
            0.3: pred,    # detected → F1 = 1.0 for event
            0.7: empty,   # missed  → F1 = 0.0 for event
        },
    )
    res = scorer.get_results()
    assert "threshold_0.3" in res["thresholded_f1"]
    assert "threshold_0.7" in res["thresholded_f1"]
    # At 0.3 threshold: prediction present → recall > 0 → F1 > 0
    assert res["thresholded_f1"]["threshold_0.3"]["event_iou_0.5"]["macro_f1"] > 0.0
    # At 0.7 threshold: no prediction → recall = 0 → F1 = 0
    assert res["thresholded_f1"]["threshold_0.7"]["event_iou_0.5"]["macro_f1"] == pytest.approx(0.0, abs=1e-6)


def test_thresholded_f1_ignore_absent() -> None:
    """Absent classes should be excluded from macro averages when ignore_absent=True."""
    scorer = Scorer(
        dataset_ontology=["A", "B", "C"],
        n_thresholds=3,
        annotation_col=ANNOTATION_COL,
        discretization_frame_rate=10.0,
        iou_thresholds=[0.5],
        thresholds_for_thresholded_metrics=[0.5],
    )
    duration = 60.0
    # Only A and B have GT; C is absent.
    gt = _st([(0.0, 1.0, "A"), (1.0, 2.0, "B")])
    pred = gt.copy()
    pred_st_by_thr = {thr: pred.copy() for thr in scorer.thresholds}

    scorer.update_from_selection_table_by_threshold(
        pred_st_by_thr, gt, duration_sec=duration,
        selection_tables_for_thresholded_metrics={0.5: pred},
    )
    res_ignore = scorer.get_results(ignore_absent=True)
    res_all = scorer.get_results(ignore_absent=False)

    # All three labels appear in per-class dicts regardless of ignore_absent.
    for res in (res_ignore, res_all):
        assert set(res["thresholded_f1"]["threshold_0.5"]["frame"]["f1_per_class"].keys()) == {"A", "B", "C"}

    # With ignore_absent=True, absent class C (F1=0) is excluded from macro, so macro_f1 is higher.
    assert res_ignore["thresholded_f1"]["threshold_0.5"]["frame"]["macro_f1"] >= \
        res_all["thresholded_f1"]["threshold_0.5"]["frame"]["macro_f1"]
