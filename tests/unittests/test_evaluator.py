from __future__ import annotations

import torch

from sound_event_detection.evaluation.classification_eval_helpers import compute_multilabel_metrics

_NUM_CLASSES = 4
_N = 40
_CLASS_NAMES = ["a", "b", "c", "d"]

_EXPECTED_KEYS = {
    "cmAP",
    "cmAP5",
    "mAP",
    "pcmAP",
    "MultilabelAUROC",
    "T1Accuracy",
    "T3Accuracy",
    "class_AP",
    "class_AP_masked",
}


def test_compute_multilabel_metrics_returns_all_keys():
    torch.manual_seed(0)
    preds = torch.rand(_N, _NUM_CLASSES)
    targets = torch.randint(0, 2, (_N, _NUM_CLASSES))
    results = compute_multilabel_metrics(preds, targets, _NUM_CLASSES, _CLASS_NAMES)
    assert set(results.keys()) == _EXPECTED_KEYS


def test_compute_multilabel_metrics_per_class_keyed_by_class_names():
    torch.manual_seed(0)
    preds = torch.rand(_N, _NUM_CLASSES)
    targets = torch.randint(0, 2, (_N, _NUM_CLASSES))
    results = compute_multilabel_metrics(preds, targets, _NUM_CLASSES, _CLASS_NAMES)
    assert set(results["class_AP"].keys()) == set(_CLASS_NAMES)
    assert set(results["class_AP_masked"].keys()) == set(_CLASS_NAMES)


def test_compute_multilabel_metrics_perfect_predictions():
    """Perfect predictions should yield cmAP == 1.0."""
    targets = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 1, 1], [0, 1, 1]])
    preds = targets.float()
    results = compute_multilabel_metrics(preds, targets, 3, ["a", "b", "c"])
    assert results["cmAP"] == 1.0
