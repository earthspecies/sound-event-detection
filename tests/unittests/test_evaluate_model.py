from __future__ import annotations

import torch

from sound_event_detection.evaluation.classification_eval_helpers import remap_to_target_labels


def test_remap_reorders_columns() -> None:
    """Matches the doctest example: source [A,B,C] → target [C,A,D]."""
    preds = torch.tensor([[0.1, 0.2, 0.3]])
    result = remap_to_target_labels(preds, ["A", "B", "C"], ["C", "A", "D"])
    torch.testing.assert_close(result, torch.tensor([[0.3, 0.1, 0.0]]))


def test_remap_missing_target_label_is_zero() -> None:
    preds = torch.tensor([[0.5, 0.8]])
    result = remap_to_target_labels(preds, ["A", "B"], ["A", "C", "B"])
    torch.testing.assert_close(result, torch.tensor([[0.5, 0.0, 0.8]]))


def test_remap_output_shape() -> None:
    preds = torch.randn(4, 3)
    result = remap_to_target_labels(preds, ["A", "B", "C"], ["C", "B"])
    assert result.shape == (4, 2)


def test_remap_batch_of_one() -> None:
    preds = torch.tensor([[0.7]])
    result = remap_to_target_labels(preds, ["A"], ["A"])
    torch.testing.assert_close(result, torch.tensor([[0.7]]))


def test_remap_all_missing_produces_zeros() -> None:
    preds = torch.ones(3, 2)
    result = remap_to_target_labels(preds, ["A", "B"], ["C", "D"])
    torch.testing.assert_close(result, torch.zeros(3, 2))
