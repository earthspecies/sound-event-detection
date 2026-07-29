"""Unit tests for `SedEvaluator._evaluate_clip_dataset`.

The dataset loader is monkeypatched and the model is a fake that satisfies the
full `DetectorClient` protocol, so no network, model, or real dataset is
needed. Mirrors the fake-collaborator style of `test_denoising_detector.py`.
"""

from __future__ import annotations

import types
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest

from esp_research.protocols.classifier import MultiLabelClassifierOutput
from sound_event_detection.adapters.dispatch import DetectorClient
from sound_event_detection.evaluation import evaluator as evaluator_mod
from sound_event_detection.evaluation.config import ClipDatasetEntry
from sound_event_detection.evaluation.evaluator import SedEvaluator

_SR = 32000
_WINDOW = 5.0
_LABELS = ["species_a", "species_b", "species_c"]

_EXPECTED_METRIC_KEYS = {
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


class FakeClient:
    """Full-surface `DetectorClient` returning fixed clip probabilities."""

    def __init__(self, labels: list[str], clip_probs: list[float], sample_rate: int = _SR) -> None:
        self.labels = labels
        self._clip_probs = np.asarray(clip_probs, dtype=np.float32)  # (C,)
        self.sample_rate = sample_rate
        self.frame_rate = 10.0
        self.window_duration = _WINDOW
        self.server_config = {"labels": labels, "sample_rate": sample_rate}
        self.classifier_calls: list[dict] = []

    def run(self, audio: np.ndarray, *args: object, **kwargs: object) -> None:
        raise AssertionError("clip evaluation must use run_as_classifier, not run")

    def run_as_classifier(
        self, audio: np.ndarray, batch_size: int = 32, **kwargs: object
    ) -> MultiLabelClassifierOutput:
        self.classifier_calls.append({"shape": audio.shape, "batch_size": batch_size})
        preds = np.tile(self._clip_probs, (audio.shape[0], 1))
        return MultiLabelClassifierOutput(predictions=preds, class_names=self.labels)

    def describe_summary(self) -> dict:
        return {
            "n_labels": len(self.labels),
            "sample_rate": self.sample_rate,
            "frame_rate": self.frame_rate,
            "window_duration": self.window_duration,
        }

    def close(self) -> None:
        pass


class FakeDataset:
    """Fake alp-data dataset: items with audio + species_list, plus name/split."""

    def __init__(self, items: list[dict], sample_rate: int = _SR, name: str = "fake_clip", split: str = "test") -> None:
        self._items = items
        self.sample_rate = sample_rate
        self.info = types.SimpleNamespace(name=name)
        self.split = split

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[dict]:
        return iter(self._items)


def _clip(species_list: list[str], n_samples: int = int(_WINDOW * _SR)) -> dict:
    return {"audio": np.zeros(n_samples, dtype=np.float32), "species_list": species_list}


def _evaluator(tmp_path: Path, **overrides: object) -> SedEvaluator:
    kwargs: dict = {
        "clip_datasets": [ClipDatasetEntry(config="unused.yml")],
        "batch_size": 2,
        "output_dir": str(tmp_path / "results"),
    }
    kwargs.update(overrides)
    return SedEvaluator(**kwargs)


@pytest.fixture
def patched_dataset(monkeypatch: pytest.MonkeyPatch) -> Callable[[FakeDataset], None]:
    def patch(dataset: FakeDataset) -> None:
        monkeypatch.setattr(evaluator_mod, "dataset_from_config", lambda config: (dataset, None))

    return patch


def test_fake_client_satisfies_detector_client_protocol() -> None:
    assert isinstance(FakeClient(_LABELS, [0.5, 0.5, 0.5]), DetectorClient)


def test_evaluate_clip_dataset_metrics_and_remap(
    tmp_path: Path, patched_dataset: Callable[[FakeDataset], None]
) -> None:
    """Predictions are remapped to gt_labels order and scored, without mutating the client."""
    # species_a high, others low. The client's label order is reversed relative
    # to the (sorted) gt ontology derived from the dataset, exercising the remap.
    client = FakeClient(list(reversed(_LABELS)), [0.05, 0.1, 0.9])
    dataset = FakeDataset([_clip(["species_a"]), _clip(["species_a"]), _clip(["species_b"]), _clip(["species_c"])])
    patched_dataset(dataset)

    result = _evaluator(tmp_path).evaluate(client)

    ds_results = result.details["clip_datasets"]["fake_clip:test"]
    assert _EXPECTED_METRIC_KEYS <= set(ds_results.keys())
    assert ds_results["n_files"] == 4
    assert ds_results["n_classes"] == 3
    assert ds_results["gt_coverage"] == 1.0
    # species_a (high) lands in its gt column after remap; clips 0,1 (GT
    # species_a) are top-1 correct, clips 2,3 are not -> top-1 == 2/4.
    assert ds_results["T1Accuracy"] == pytest.approx(0.5)
    # batch_size=2 over 4 clips -> two classifier calls of 2 window-length clips.
    assert [call["shape"] for call in client.classifier_calls] == [(2, int(_WINDOW * _SR))] * 2
    # The frame pattern: the client is read, never mutated.
    assert not hasattr(client, "target_labels")


def test_evaluate_clip_dataset_pads_short_clips(tmp_path: Path, patched_dataset: Callable[[FakeDataset], None]) -> None:
    client = FakeClient(_LABELS, [0.9, 0.1, 0.05])
    # Three species so the metric suite's top-3 accuracy has enough classes.
    dataset = FakeDataset([_clip(["species_a", "species_b", "species_c"], n_samples=1000)])
    patched_dataset(dataset)

    _evaluator(tmp_path).evaluate(client)

    # The short clip is padded up to the client's window length before the call.
    assert client.classifier_calls[0]["shape"] == (1, int(_WINDOW * _SR))


def test_evaluate_clip_dataset_rejects_sample_rate_mismatch(
    tmp_path: Path, patched_dataset: Callable[[FakeDataset], None]
) -> None:
    client = FakeClient(_LABELS, [0.9, 0.1, 0.05])
    patched_dataset(FakeDataset([_clip(["species_a"])], sample_rate=16000))

    with pytest.raises(ValueError, match="sample rate"):
        _evaluator(tmp_path).evaluate(client)


def test_evaluate_clip_dataset_rejects_empty_dataset(
    tmp_path: Path, patched_dataset: Callable[[FakeDataset], None]
) -> None:
    client = FakeClient(_LABELS, [0.9, 0.1, 0.05])
    patched_dataset(FakeDataset([]))

    # A pointed error, not torch.cat's opaque "expected a non-empty list".
    with pytest.raises(ValueError, match="yielded no items"):
        _evaluator(tmp_path).evaluate(client)


def test_evaluate_clip_dataset_skips_completed_checkpoint(
    tmp_path: Path, patched_dataset: Callable[[FakeDataset], None]
) -> None:
    dataset = FakeDataset([_clip(["species_a"]), _clip(["species_b", "species_c"])])
    patched_dataset(dataset)
    checkpoint_dir = str(tmp_path / "ckpt")

    first_client = FakeClient(_LABELS, [0.9, 0.1, 0.05])
    first = _evaluator(tmp_path, checkpoint_dir=checkpoint_dir).evaluate(first_client)

    second_client = FakeClient(_LABELS, [0.9, 0.1, 0.05])
    second = _evaluator(tmp_path, checkpoint_dir=checkpoint_dir).evaluate(second_client)

    # The completed dataset is skipped: cached results, no model calls.
    assert second_client.classifier_calls == []
    ds_first = first.details["clip_datasets"]["fake_clip:test"]
    ds_second = second.details["clip_datasets"]["fake_clip:test"]
    assert ds_second["cmAP"] == pytest.approx(ds_first["cmAP"])
    assert ds_second["n_files"] == ds_first["n_files"]
