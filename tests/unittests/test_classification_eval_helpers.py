"""Unit tests for the clip-evaluation helpers.

The dataset loop itself lives in `SedEvaluator._evaluate_clip_dataset` and is
tested in `test_evaluator_clip.py`; these tests cover the pure helpers.
"""

from collections.abc import Iterator

import numpy as np
import torch

from sound_event_detection.evaluation.classification_eval_helpers import (
    clip_dataset_target_labels,
    multi_hot_targets,
    normalize_species_list,
)


class FakeDataset:
    """Fake dataset yielding items with audio and species_list."""

    def __init__(self, items: list[dict], sample_rate: int = 32000) -> None:
        self._items = items
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[dict]:
        return iter(self._items)


def _clip(species_list: list[str], sample_rate: int = 32000) -> dict:
    return {"audio": np.zeros(sample_rate * 5, dtype=np.float32), "species_list": species_list}


def test_clip_dataset_target_labels_sorted_union() -> None:
    dataset = FakeDataset(
        [
            _clip(["b", "a"]),
            _clip(["c"]),
            _clip([]),
            _clip(["a"]),
        ]
    )
    assert clip_dataset_target_labels(dataset) == ["a", "b", "c"]


def test_normalize_species_list_coerces_scalars() -> None:
    assert normalize_species_list(None) == []
    assert normalize_species_list("a") == ["a"]
    assert normalize_species_list(["a", "b"]) == ["a", "b"]


def test_multi_hot_targets_marks_known_species() -> None:
    targets = multi_hot_targets([["a", "c"], [], ["b"]], ["a", "b", "c"])

    assert targets.dtype == torch.int32
    torch.testing.assert_close(targets, torch.tensor([[1, 0, 1], [0, 0, 0], [0, 1, 0]], dtype=torch.int32))


def test_multi_hot_targets_ignores_unknown_species() -> None:
    targets = multi_hot_targets([["a", "species_z"]], ["a", "b"])

    torch.testing.assert_close(targets, torch.tensor([[1, 0]], dtype=torch.int32))
