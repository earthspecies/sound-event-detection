"""Tests for the sharded LSI engine: framing, sharding math, skip, and errors."""

import json

import numpy as np
import pytest

from esp_research.protocols.detector import DetectorOutput
from sound_event_detection.inference.engine import (
    LINEAGE_FILENAME,
    list_shards,
    read_lineage,
    read_shard,
    run_sharded,
    save_shard,
    write_lineage,
)
from sound_event_detection.inference.result import ItemResult


class _FakeDataset:
    """Minimal sequence dataset of ``{"id", "audio"}`` items."""

    def __init__(self, n: int) -> None:
        self._items = [{"id": f"rec{i:03d}", "audio": np.full(8, i, dtype=np.float32)} for i in range(n)]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        return self._items[idx]


def _process(item: dict) -> dict[str, np.ndarray]:
    frames = int(item["audio"][0]) % 3 + 1
    values = np.full((frames, 2), 0.5, dtype=np.float32)
    output = DetectorOutput(predictions=values[np.newaxis], frame_rate=5.0, class_names=["a", "b"])
    return ItemResult(preds=output).to_arrays()


def test_single_job_writes_and_reads_back(tmp_path) -> None:
    dataset = _FakeDataset(5)
    run_sharded(dataset, "id", _process, str(tmp_path), files_per_shard=2)

    shards = sorted(tmp_path.glob("shard_*.npz"))
    assert [p.name for p in shards] == ["shard_0000.npz", "shard_0001.npz", "shard_0002.npz"]

    all_ids = []
    for shard in shards:
        for file_id, arrays in read_shard(str(shard)).items():
            all_ids.append(file_id)
            ItemResult.from_arrays(arrays)  # decodes without error
    assert all_ids == [f"rec{i:03d}" for i in range(5)]


def test_shard_ranges_partition_dataset_across_jobs(tmp_path) -> None:
    dataset = _FakeDataset(10)  # 5 shards of 2
    for job in range(3):
        run_sharded(dataset, "id", _process, str(tmp_path), files_per_shard=2, job_index=job, num_jobs=3)

    ids = []
    for shard in sorted(tmp_path.glob("shard_*.npz")):
        ids.extend(read_shard(str(shard)).keys())
    assert ids == [f"rec{i:03d}" for i in range(10)]  # every file exactly once, in order


def test_skip_existing_is_idempotent(tmp_path) -> None:
    dataset = _FakeDataset(4)
    run_sharded(dataset, "id", _process, str(tmp_path), files_per_shard=2)
    shard0 = tmp_path / "shard_0000.npz"
    mtime = shard0.stat().st_mtime_ns

    def _boom(_item: dict) -> dict[str, np.ndarray]:
        raise AssertionError("process must not run for existing shards")

    run_sharded(dataset, "id", _boom, str(tmp_path), files_per_shard=2)
    assert shard0.stat().st_mtime_ns == mtime  # not rewritten


def test_incomplete_shard_is_recomputed_on_resubmit(tmp_path) -> None:
    # A shard left short by a failing file (the poisoned-server case writes an
    # empty one) must not be mistaken for finished: resubmit recomputes it.
    dataset = _FakeDataset(2)  # one shard of 2, so "complete" needs both files
    calls = {"n": 0}

    def _first_file_fails_once(item: dict) -> dict[str, np.ndarray]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("transient failure")
        return _process(item)

    run_sharded(dataset, "id", _first_file_fails_once, str(tmp_path), files_per_shard=2)
    assert len(read_shard(str(tmp_path / "shard_0000.npz"))) == 1  # short: one file dropped

    run_sharded(dataset, "id", _process, str(tmp_path), files_per_shard=2)  # resubmit, all succeed
    assert list(read_shard(str(tmp_path / "shard_0000.npz")).keys()) == ["rec000", "rec001"]


def test_per_item_error_is_logged_and_isolated(tmp_path) -> None:
    dataset = _FakeDataset(3)

    def _sometimes_fails(item: dict) -> dict[str, np.ndarray]:
        if item["id"] == "rec001":
            raise ValueError("bad file")
        return _process(item)

    run_sharded(dataset, "id", _sometimes_fails, str(tmp_path), files_per_shard=3)

    ids = list(read_shard(str(tmp_path / "shard_0000.npz")).keys())
    assert ids == ["rec000", "rec002"]  # the failing file is dropped, others survive

    error_log = tmp_path / "errors_job_000.jsonl"
    logged = [json.loads(line) for line in error_log.read_text().splitlines()]
    assert len(logged) == 1
    assert logged[0]["file_id"] == "rec001" and "bad file" in logged[0]["error"]


def test_temp_file_is_cleaned_up_on_success(tmp_path) -> None:
    run_sharded(_FakeDataset(2), "id", _process, str(tmp_path), files_per_shard=2)
    assert not list(tmp_path.glob("*.tmp*"))


@pytest.mark.parametrize(
    "kwargs",
    [{"files_per_shard": 0}, {"files_per_shard": 2, "num_jobs": 0}, {"files_per_shard": 2, "job_index": 3, "num_jobs": 3}],
)
def test_invalid_args_raise(tmp_path, kwargs) -> None:
    with pytest.raises(ValueError):
        run_sharded(_FakeDataset(4), "id", _process, str(tmp_path), **kwargs)


def test_read_lineage_missing_returns_none(tmp_path) -> None:
    assert read_lineage(str(tmp_path)) is None


def test_write_lineage_records_stage_fields_and_provenance(tmp_path) -> None:
    write_lineage(str(tmp_path), "run", {"detail": "preds"})
    record = read_lineage(str(tmp_path))
    assert (tmp_path / LINEAGE_FILENAME).exists()
    assert record["stage"] == "run"
    assert record["detail"] == "preds"
    assert "git_commit" in record and "created_utc" in record
    assert "parent" not in record  # a root stage has no parent


def test_write_lineage_chains_parent(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pp_dir = run_dir / "postprocessed_thr0.50"
    pp_dir.mkdir()
    write_lineage(str(run_dir), "run", {"detail": "preds"})
    write_lineage(str(pp_dir), "postprocess", {"threshold": 0.5}, parent_dir=str(run_dir))

    record = read_lineage(str(pp_dir))
    assert record["stage"] == "postprocess"
    # The upstream run record is embedded whole, so the chain is traceable.
    assert record["parent"]["stage"] == "run"
    assert record["parent"]["detail"] == "preds"


def test_write_lineage_parent_absent_fails_open(tmp_path) -> None:
    # Chaining onto an input dir with no lineage file records parent = None.
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    write_lineage(str(out_dir), "postprocess", {"threshold": 0.5}, parent_dir=str(tmp_path))
    assert read_lineage(str(out_dir))["parent"] is None


def test_write_lineage_is_idempotent(tmp_path) -> None:
    write_lineage(str(tmp_path), "run", {"n": 1})
    write_lineage(str(tmp_path), "run", {"n": 2})  # a re-run must not overwrite the original
    assert read_lineage(str(tmp_path))["n"] == 1


def test_list_shards_sorts_by_index_and_ignores_others(tmp_path) -> None:
    empty = ItemResult(
        preds=DetectorOutput(predictions=np.zeros((1, 1, 1), dtype=np.float32), frame_rate=1.0, class_names=["a"])
    ).to_arrays()
    for idx in (2, 0, 10):  # out of order, and a multi-digit index
        save_shard(str(tmp_path / f"shard_{idx:04d}.npz"), [("x", empty)], 0)
    (tmp_path / "config.yaml").write_text("noise")  # a non-shard file must be ignored

    listed = list_shards(str(tmp_path))
    assert [idx for idx, _ in listed] == [0, 2, 10]
    assert all(path.endswith(f"shard_{idx:04d}.npz") for idx, path in listed)
