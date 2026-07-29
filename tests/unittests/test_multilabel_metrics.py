from __future__ import annotations

import pytest
import torch

from sound_event_detection.evaluation.classification_metrics import (
    TopKAccuracy,
    cmAP,
    cmAP5,
    mAP,
    mAUROC,
    pcmAP,
)

_NUM_LABELS = 4
_N = 60  # 10 positives per class keeps all classes above the cmAP5 sample threshold


@pytest.fixture
def perfect_preds():
    """Structured targets with 10 positives per class; preds == targets (float)."""
    targets = torch.zeros(_N, _NUM_LABELS, dtype=torch.int)
    for c in range(_NUM_LABELS):
        targets[c * 10 : c * 10 + 10, c] = 1
    return targets.float(), targets


@pytest.fixture
def random_preds():
    """Seeded random probabilities and binary integer targets."""
    torch.manual_seed(42)
    preds = torch.rand(_N, _NUM_LABELS)
    targets = torch.randint(0, 2, (_N, _NUM_LABELS))
    return preds, targets


# --- cmAP ---

def test_cmAP_perfect_score(perfect_preds):
    preds, targets = perfect_preds
    result = cmAP(num_labels=_NUM_LABELS)(preds, targets)
    torch.testing.assert_close(result, torch.tensor(1.0))


def test_cmAP_score_in_range(random_preds):
    preds, targets = random_preds
    result = cmAP(num_labels=_NUM_LABELS)(preds, targets)
    assert 0.0 <= result.item() <= 1.0


# --- mAP ---

def test_mAP_perfect_score(perfect_preds):
    preds, targets = perfect_preds
    result = mAP(num_labels=_NUM_LABELS)(preds, targets)
    torch.testing.assert_close(result, torch.tensor(1.0))


def test_mAP_score_in_range(random_preds):
    preds, targets = random_preds
    result = mAP(num_labels=_NUM_LABELS)(preds, targets)
    assert 0.0 <= result.item() <= 1.0


# --- cmAP5 ---

def test_cmAP5_perfect_score(perfect_preds):
    preds, targets = perfect_preds
    metric = cmAP5(num_labels=_NUM_LABELS, sample_threshold=5)
    metric.update(preds, targets)
    torch.testing.assert_close(metric.compute(), torch.tensor(1.0))


def test_cmAP5_score_in_range(random_preds):
    preds, targets = random_preds
    metric = cmAP5(num_labels=_NUM_LABELS, sample_threshold=5)
    metric.update(preds, targets)
    result = metric.compute()
    assert 0.0 <= result.item() <= 1.0


def test_cmAP5_accumulates_across_updates(perfect_preds):
    """Calling update twice then compute is equivalent to one update with all data."""
    preds, targets = perfect_preds
    half = _N // 2

    metric_split = cmAP5(num_labels=_NUM_LABELS, sample_threshold=5)
    metric_split.update(preds[:half], targets[:half])
    metric_split.update(preds[half:], targets[half:])

    metric_full = cmAP5(num_labels=_NUM_LABELS, sample_threshold=5)
    metric_full.update(preds, targets)

    torch.testing.assert_close(metric_split.compute(), metric_full.compute())


# --- pcmAP ---

def test_pcmAP_perfect_score(perfect_preds):
    preds, targets = perfect_preds
    result = pcmAP(num_labels=_NUM_LABELS, padding_factor=5)(preds, targets)
    torch.testing.assert_close(result, torch.tensor(1.0))


def test_pcmAP_score_in_range(random_preds):
    preds, targets = random_preds
    result = pcmAP(num_labels=_NUM_LABELS, padding_factor=5)(preds, targets)
    assert 0.0 <= result.item() <= 1.0


# --- mAUROC ---

def test_mAUROC_perfect_score(perfect_preds):
    preds, targets = perfect_preds
    result = mAUROC(num_labels=_NUM_LABELS)(preds, targets)
    torch.testing.assert_close(result, torch.tensor(1.0))


def test_mAUROC_score_in_range(random_preds):
    preds, targets = random_preds
    result = mAUROC(num_labels=_NUM_LABELS)(preds, targets)
    assert 0.0 <= result.item() <= 1.0


# --- TopKAccuracy ---

def test_topk1_perfect():
    targets = torch.zeros(10, _NUM_LABELS)
    targets[torch.arange(10), torch.arange(10) % _NUM_LABELS] = 1
    preds = targets.clone()

    metric = TopKAccuracy(topk=1)
    metric.update(preds, targets)
    torch.testing.assert_close(metric.compute(), torch.tensor(1.0))


def test_topk3_at_least_as_good_as_topk1(random_preds):
    preds, targets = random_preds
    targets = targets.float()
    # Only keep rows with at least one positive label
    mask = targets.sum(dim=1) > 0
    preds, targets = preds[mask], targets[mask]

    m1 = TopKAccuracy(topk=1)
    m1.update(preds, targets)
    m3 = TopKAccuracy(topk=3)
    m3.update(preds, targets)

    assert m3.compute().item() >= m1.compute().item()


def test_topk1_exclude_nocalls_ignores_all_negative_rows():
    """All-negative rows are excluded from the denominator when include_nocalls=False."""
    preds = torch.tensor([[0.9, 0.1], [0.1, 0.1]])
    targets = torch.tensor([[1, 0], [0, 0]], dtype=torch.float)

    metric = TopKAccuracy(topk=1, include_nocalls=False)
    metric.update(preds, targets)
    # Only row 0 counts (positive row predicted correctly) → 1/1 = 1.0
    torch.testing.assert_close(metric.compute(), torch.tensor(1.0))


def test_topk1_include_nocalls_low_confidence_counts_as_correct():
    """All-negative row with max pred below threshold is counted as correct."""
    preds = torch.tensor([[0.9, 0.1], [0.1, 0.1]])
    targets = torch.tensor([[1, 0], [0, 0]], dtype=torch.float)

    metric = TopKAccuracy(topk=1, include_nocalls=True, threshold=0.5)
    metric.update(preds, targets)
    # Row 0: predicted correctly. Row 1: all-negative, max pred 0.1 < 0.5 → correct.
    torch.testing.assert_close(metric.compute(), torch.tensor(1.0))


def test_topk1_include_nocalls_high_confidence_counts_as_wrong():
    """All-negative row with max pred >= threshold is counted as incorrect."""
    preds = torch.tensor([[0.9, 0.1], [0.8, 0.1]])
    targets = torch.tensor([[1, 0], [0, 0]], dtype=torch.float)

    metric = TopKAccuracy(topk=1, include_nocalls=True, threshold=0.5)
    metric.update(preds, targets)
    # Row 0: correct. Row 1: all-negative but max pred 0.8 >= 0.5 → wrong.
    torch.testing.assert_close(metric.compute(), torch.tensor(0.5))
