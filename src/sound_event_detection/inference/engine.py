"""Model-agnostic sharded inference engine.

`run_sharded` is the invariant loop: it assigns a contiguous range of shards to
this job, skips shards already complete (holding a result for every file they
were assigned), calls a producer `process` on each dataset item, and writes one
compressed ``.npz`` per shard via an atomic temp-then-rename. Per-item failures
are caught and logged to a per-job JSONL sidecar so one bad recording never
aborts a shard.

The engine owns the *container* — the shard framing (`file_ids` plus a
per-recording ``i/`` key prefix) and the on-disk write — while the producer owns
the *encoding* (each `process(item)` returns one recording's flat
``{key: array}`` dict, e.g. from `ItemResult.to_arrays`). There is no injected
writer: a second container format would be added as an explicit alternative, not
a strategy object.

`read_shard` is the inverse of the framing: it splits a shard back into
``{file_id: {key: array}}`` so the downstream re-gate can feed each recording to
`ItemResult.from_arrays`.

Paths may be local or cloud (``gs://``, ``s3://``, ``r2://``); cloud access goes
through `alp_data.io`.
"""

import datetime
import io
import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import yaml
from alp_data.io import anypath, exists, filesystem_from_path

__all__ = [
    "LINEAGE_FILENAME",
    "SHARD_RE",
    "git_commit",
    "list_shards",
    "read_lineage",
    "read_shard",
    "run_sharded",
    "save_shard",
    "write_lineage",
    "write_text",
]

#: Filename of the per-stage lineage record written into each stage's output
#: directory (see `write_lineage`).
LINEAGE_FILENAME = "lineage.yaml"

#: A producer: maps one dataset item to its flat ``{key: array}`` encoding.
Process = Callable[[Mapping], dict[str, np.ndarray]]

#: Regex capturing the zero-padded index of a ``shard_NNNN.npz`` filename.
SHARD_RE = re.compile(r"shard_(\d+)\.npz$")


def git_commit() -> str | None:
    """Return the current ``HEAD`` commit hash, or ``None`` if unavailable.

    Used to stamp run provenance into the lineage records written by every LSI
    stage (see `write_lineage`). Never raises: any failure (not a git checkout,
    git missing) yields ``None``.

    Returns
    -------
    str or None
        The 40-char commit hash, or ``None`` if it could not be determined.
    """
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 — provenance is best-effort, never fatal
        return None


def _ensure_dir(prefix: str) -> None:
    """Create a local directory if needed (cloud prefixes are implicit)."""
    if isinstance(anypath(prefix), Path):
        Path(prefix).mkdir(parents=True, exist_ok=True)


def _read_bytes(path: str) -> bytes:
    """Read a file's bytes (cloud or local).

    Returns
    -------
    bytes
        The file contents.
    """
    if not isinstance(anypath(path), Path):
        fs = filesystem_from_path(path)
        with fs.open(fs._strip_protocol(path), "rb") as handle:
            return handle.read()
    return Path(path).read_bytes()


def _write_bytes_atomic(path: str, data: bytes, job_index: int) -> None:
    """Write bytes to `path` atomically via a temp file then rename (cloud or local).

    A crash mid-write leaves only the temp file, so the completeness check never
    sees a partially written shard. The temp name includes `job_index` so
    concurrent jobs never collide on it.

    Parameters
    ----------
    path : str
        Destination path (local or cloud).
    data : bytes
        Payload to write.
    job_index : int
        This job's index, used to make the temp path unique.
    """
    tmp_path = f"{path}.tmp.job{job_index:03d}"
    if not isinstance(anypath(path), Path):
        fs = filesystem_from_path(path)
        tmp_stripped = fs._strip_protocol(tmp_path)
        with fs.open(tmp_stripped, "wb") as handle:
            handle.write(data)
        fs.mv(tmp_stripped, fs._strip_protocol(path))
    else:
        Path(tmp_path).write_bytes(data)
        os.replace(tmp_path, path)


def write_text(path: str, content: str) -> None:
    """Write a UTF-8 text file (cloud or local)."""
    encoded = content.encode("utf-8")
    if not isinstance(anypath(path), Path):
        fs = filesystem_from_path(path)
        with fs.open(fs._strip_protocol(path), "wb") as handle:
            handle.write(encoded)
    else:
        local = Path(path)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(encoded)


def read_lineage(directory: str) -> dict | None:
    """Read a stage's lineage record from `directory`, or ``None`` if absent.

    The inverse of `write_lineage`: reads the ``lineage.yaml`` written into a
    stage's output directory. Downstream stages call this on their input
    directory to embed the upstream record as their `parent` (see
    `write_lineage`); a missing file fails open (returns ``None``) so a stage run
    over output from an older, lineage-less run still succeeds.

    Parameters
    ----------
    directory : str
        Directory (local or cloud) that may hold a `LINEAGE_FILENAME` record.

    Returns
    -------
    dict or None
        The parsed lineage record, or ``None`` if no lineage file is present.
    """
    path = f"{directory.rstrip('/')}/{LINEAGE_FILENAME}"
    if not exists(path):
        return None
    return yaml.safe_load(_read_bytes(path).decode("utf-8"))


def write_lineage(
    out_dir: str,
    stage: str,
    fields: Mapping[str, object],
    *,
    parent_dir: str | None = None,
) -> None:
    """Write a chained lineage record for one pipeline stage, unless one exists.

    Every LSI stage (``run`` -> ``postprocess`` -> ``features``) drops a
    `LINEAGE_FILENAME` into its output directory recording the stage name, its
    resolved `fields` (config, resolved parameters), the git commit, and a UTC
    timestamp. When `parent_dir` is given, the upstream stage's lineage (read
    from that directory via `read_lineage`) is embedded under a ``parent`` key,
    so a downstream output can be traced back through every stage to the run
    that produced it (model + dataset + git commit). The write is skipped if a
    lineage file already exists, so re-running a stage never overwrites the
    original provenance.

    Parameters
    ----------
    out_dir : str
        The stage's output directory (local or cloud) to write the record into.
    stage : str
        Short stage name recorded under ``stage`` (e.g. ``"run"``,
        ``"postprocess"``, ``"features"``).
    fields : Mapping[str, object]
        Stage-specific, YAML-serializable payload (e.g. the resolved config and
        parameters) merged into the record after ``stage``.
    parent_dir : str or None
        Input directory of the upstream stage. When given, its lineage record is
        embedded under ``parent`` (``None`` there if the upstream directory has
        no lineage file). ``None`` marks a root stage (the run) with no parent.
    """
    path = f"{out_dir.rstrip('/')}/{LINEAGE_FILENAME}"
    if exists(path):
        return
    record: dict[str, object] = {"stage": stage, **dict(fields)}
    if parent_dir is not None:
        record["parent"] = read_lineage(parent_dir)
    record["git_commit"] = git_commit()
    record["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_text(path, yaml.dump(record, default_flow_style=False, sort_keys=False))


def _pack_shard(items: Sequence[tuple[str, Mapping[str, np.ndarray]]]) -> dict[str, np.ndarray]:
    """Frame per-recording arrays into one shard array dict.

    Parameters
    ----------
    items : Sequence[tuple[str, Mapping[str, np.ndarray]]]
        ``(file_id, arrays)`` pairs, one per recording, in shard order.

    Returns
    -------
    dict[str, np.ndarray]
        ``{"file_ids": (N,) str}`` plus every recording's arrays namespaced
        under an ``f"{i}/"`` prefix (``i`` is the position in `items`).
    """
    packed: dict[str, np.ndarray] = {"file_ids": np.array([file_id for file_id, _ in items], dtype=np.str_)}
    for i, (_, arrays) in enumerate(items):
        for key, value in arrays.items():
            packed[f"{i}/{key}"] = value
    return packed


def save_shard(path: str, items: Sequence[tuple[str, Mapping[str, np.ndarray]]], job_index: int) -> None:
    """Compress and atomically write one shard of packed recordings."""
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **_pack_shard(items))
    _write_bytes_atomic(path, buffer.getvalue(), job_index)


def read_shard(path: str) -> dict[str, dict[str, np.ndarray]]:
    """Read a shard back into ``{file_id: {key: array}}`` (inverse of the framing).

    Undoes `_pack_shard`: splits the ``i/`` prefix and groups arrays by
    recording. The per-recording dict is exactly what `ItemResult.from_arrays`
    consumes.

    Parameters
    ----------
    path : str
        Shard ``.npz`` path (local or cloud).

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        File id -> that recording's flat array dict, in shard order.
    """
    data = np.load(io.BytesIO(_read_bytes(path)), allow_pickle=False)
    file_ids = [str(file_id) for file_id in data["file_ids"]]
    groups: list[dict[str, np.ndarray]] = [{} for _ in file_ids]
    for key in data.files:
        if key == "file_ids":
            continue
        index_str, sub_key = key.split("/", 1)
        groups[int(index_str)][sub_key] = data[key]
    return {file_ids[i]: groups[i] for i in range(len(file_ids))}


def _shard_file_count(path: str) -> int:
    """Return the number of recordings stored in a shard, reading only its index.

    Lets `run_sharded` tell a fully-produced shard from a short one left behind by
    a crash or a shard where every file errored: only the ``file_ids`` array is
    decompressed, not the heavy per-recording payload.

    Parameters
    ----------
    path : str
        Shard ``.npz`` path (local or cloud).

    Returns
    -------
    int
        Count of recordings framed into the shard.
    """
    data = np.load(io.BytesIO(_read_bytes(path)), allow_pickle=False)
    return len(data["file_ids"])


def list_shards(run_dir: str) -> list[tuple[int, str]]:
    """List ``shard_NNNN.npz`` files in `run_dir`, sorted by index.

    Works on local and cloud (``gs://``/``s3://``/``r2://``) directories. Used by
    the shard-consuming stages (`postprocess`, `features`) to enumerate a run's
    shards and to pair shards across sibling directories by their shared index.

    Parameters
    ----------
    run_dir : str
        Directory (local or cloud) holding ``shard_*.npz`` files.

    Returns
    -------
    list[tuple[int, str]]
        ``(index, path)`` pairs sorted by index.
    """
    if not isinstance(anypath(run_dir), Path):
        scheme = run_dir.split("://", 1)[0]
        fs = filesystem_from_path(run_dir)
        stripped = fs._strip_protocol(run_dir).rstrip("/")
        paths = [f"{scheme}://{match}" for match in fs.glob(f"{stripped}/shard_*.npz")]
    else:
        paths = [str(path) for path in Path(run_dir).expanduser().glob("shard_*.npz")]
    shards: list[tuple[int, str]] = []
    for path in paths:
        match = SHARD_RE.search(path)
        if match:
            shards.append((int(match.group(1)), path))
    return sorted(shards)


def run_sharded(
    dataset: object,
    id_column: str,
    process: Process,
    out_dir: str,
    files_per_shard: int,
    job_index: int = 0,
    num_jobs: int = 1,
) -> None:
    """Run a producer over this job's shard range and write ``.npz`` shards.

    The dataset is split into ``ceil(len / files_per_shard)`` contiguous shards;
    this job handles the range ``[total*job_index/num_jobs,
    total*(job_index+1)/num_jobs)``. A shard is skipped only once it holds a
    result for every file it was assigned, so a job killed mid-shard — or one
    whose shard came up short because files errored out — is redone on resubmit
    rather than mistaken for finished (a shard with legitimately skipped files
    stays short and is re-attempted, cheaply, on each resubmit). Each item is
    fetched, keyed by ``str(item[id_column])``, and passed to `process`; per-item
    exceptions are recorded to ``errors_job_{job_index:03d}.jsonl`` in `out_dir`
    and skipped.

    Parameters
    ----------
    dataset : object
        A sequence-like dataset: ``len(dataset)`` and ``dataset[i]`` yield an
        item mapping containing `id_column` and whatever `process` reads.
    id_column : str
        Item key used as the file identifier stored in each shard.
    process : Process
        Producer mapping one item to its flat ``{key: array}`` encoding.
    out_dir : str
        Output directory for shards and the error log (local or cloud).
    files_per_shard : int
        Number of recordings per shard.
    job_index : int
        Zero-based index of this job within a parallel array.
    num_jobs : int
        Total number of parallel jobs.

    Raises
    ------
    ValueError
        If `files_per_shard` or `num_jobs` is not positive, or `job_index` is
        outside ``[0, num_jobs)``.
    """
    if files_per_shard <= 0:
        raise ValueError(f"files_per_shard must be positive, got {files_per_shard}")
    if num_jobs <= 0:
        raise ValueError(f"num_jobs must be positive, got {num_jobs}")
    if not 0 <= job_index < num_jobs:
        raise ValueError(f"job_index {job_index} out of range [0, {num_jobs})")

    _ensure_dir(out_dir)
    n = len(dataset)
    total_shards = max(1, (n + files_per_shard - 1) // files_per_shard)
    job_shard_start = (total_shards * job_index) // num_jobs
    job_shard_end = (total_shards * (job_index + 1)) // num_jobs

    error_log_path = os.path.join(out_dir, f"errors_job_{job_index:03d}.jsonl")
    error_lines: list[str] = []

    print(
        f"[lsi] {n} files, {total_shards} shards ({files_per_shard}/shard) | "
        f"job {job_index + 1}/{num_jobs}: shards {job_shard_start:04d}-{job_shard_end - 1:04d}",
        flush=True,
    )

    shards_done = 0
    files_done = 0
    for shard_idx in range(job_shard_start, job_shard_end):
        shard_path = os.path.join(out_dir, f"shard_{shard_idx:04d}.npz")
        file_start = shard_idx * files_per_shard
        file_end = min(file_start + files_per_shard, n)
        expected = file_end - file_start
        if exists(shard_path) and _shard_file_count(shard_path) >= expected:
            print(f"[lsi] skip complete shard {shard_idx:04d}", flush=True)
            shards_done += 1
            continue

        print(f"[lsi] shard {shard_idx:04d}: files {file_start}-{file_end - 1}", flush=True)

        shard_start = time.perf_counter()
        items: list[tuple[str, dict[str, np.ndarray]]] = []
        for file_idx in range(file_start, file_end):
            item = dataset[file_idx]
            file_id = str(item[id_column])
            try:
                items.append((file_id, process(item)))
            except Exception as exc:  # noqa: BLE001 — isolate one bad file, keep the shard going
                error_lines.append(json.dumps({"file_id": file_id, "index": file_idx, "error": str(exc)}))
                print(f"[lsi] error on {file_id} (idx {file_idx}): {exc}", flush=True)

        save_shard(shard_path, items, job_index)
        elapsed = time.perf_counter() - shard_start
        per_file = elapsed / len(items) if items else float("nan")
        print(
            f"[lsi] saved shard {shard_idx:04d} ({len(items)} files) in {elapsed:.1f}s ({per_file:.1f}s/file)",
            flush=True,
        )
        shards_done += 1
        files_done += len(items)

    print(f"[lsi] done: {shards_done} shards, {files_done} files, {len(error_lines)} errors", flush=True)
    if error_lines:
        write_text(error_log_path, "\n".join(error_lines) + "\n")
        print(f"[lsi] errors logged to {error_log_path}", flush=True)
