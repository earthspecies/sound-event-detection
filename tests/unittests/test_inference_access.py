"""Tests for row-based access to heavy LSI outputs."""

import numpy as np
import pytest

from esp_research.protocols.detector import DetectorOutput
from sound_event_detection.inference.access import load_denoised, load_frame_preds, load_stems, read_item
from sound_event_detection.inference.engine import save_shard
from sound_event_detection.inference.result import ItemResult, Stem


def _preds(frames: int, class_names: list[str], seed: int) -> DetectorOutput:
    """Batch-1 predictions on the float16 grid and above the codec threshold (lossless round-trip)."""
    rng = np.random.default_rng(seed)
    values = (0.5 + 0.5 * rng.random((frames, len(class_names)))).astype(np.float16).astype(np.float32)
    return DetectorOutput(predictions=values[np.newaxis], frame_rate=8.0, class_names=class_names)


def _audio(samples: int, seed: int) -> np.ndarray:
    """Audio on the int16 PCM grid (lossless through the FLAC codec)."""
    rng = np.random.default_rng(seed)
    return (rng.integers(-32768, 32768, size=samples) / 32767.0).astype(np.float32)


def _write_run(tmp_path):
    """Write a one-shard run with a full stems-rung ItemResult; return (row, item)."""
    item = ItemResult(
        preds=_preds(6, ["robin", "wren"], seed=1),
        denoised=_audio(400, seed=2),
        stems=(
            Stem(audio=_audio(400, seed=3), preds=_preds(6, ["robin", "wren"], seed=4)),
            Stem(audio=_audio(400, seed=5), preds=_preds(6, ["robin", "wren"], seed=6)),
        ),
    )
    shard_path = str(tmp_path / "shard_0000.npz")
    save_shard(shard_path, [("rec.wav", item.to_arrays())], job_index=0)
    row = {"audio_path": "rec.wav", "canonical_name": "robin", "lsi_shard": shard_path}
    return row, item


def test_read_item_round_trips_full_result(tmp_path):
    row, item = _write_run(tmp_path)
    got = read_item(row)
    np.testing.assert_array_equal(got.preds.predictions, item.preds.predictions)
    np.testing.assert_array_equal(got.denoised, item.denoised)
    assert len(got.stems) == 2
    np.testing.assert_array_equal(got.stems[0].audio, item.stems[0].audio)


def test_typed_loaders(tmp_path):
    row, item = _write_run(tmp_path)
    np.testing.assert_array_equal(load_frame_preds(row).predictions, item.preds.predictions)
    np.testing.assert_array_equal(load_denoised(row), item.denoised)
    assert len(load_stems(row)) == 2


def test_explicit_id_column(tmp_path):
    row, item = _write_run(tmp_path)
    row2 = {"my_id": "rec.wav", "lsi_shard": row["lsi_shard"]}
    np.testing.assert_array_equal(load_frame_preds(row2, id_column="my_id").predictions, item.preds.predictions)


def test_empty_pointer_raises(tmp_path):
    row, _ = _write_run(tmp_path)
    row["lsi_shard"] = ""
    with pytest.raises(ValueError, match="empty"):
        read_item(row)


def test_missing_pointer_column_raises(tmp_path):
    row, _ = _write_run(tmp_path)
    del row["lsi_shard"]
    with pytest.raises(KeyError, match="lsi_shard"):
        read_item(row)


def test_id_not_in_shard_raises(tmp_path):
    row, _ = _write_run(tmp_path)
    row["audio_path"] = "not_there.wav"
    with pytest.raises(KeyError, match="not_there.wav"):
        read_item(row)


def test_uninferrable_id_column_raises(tmp_path):
    row, _ = _write_run(tmp_path)
    row = {"weird": "rec.wav", "lsi_shard": row["lsi_shard"]}
    with pytest.raises(ValueError, match="could not infer id_column"):
        read_item(row)
