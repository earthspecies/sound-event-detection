"""Feature stage CLI: enrich selection tables with per-event acoustic features.

Entry point: ``uv run sed-lsi-features``.

Commands
--------
- ``sed-lsi-features --config <path> [--run-dir DIR] [--postprocessing SUBDIR] [--job-index N] [--num-jobs M]``
    enrich this job's slice of a run's selection-table shards with features.
- ``sed-lsi-features describe``
    print the feature-stage config schema.

The third stage of the LSI pipeline (produce `ItemResult` shards -> postprocess
-> **features** -> attach). It consumes two sibling products of a run:

- the `ItemResult` shards (``<run_dir>/shard_*.npz``) for the **focal audio**, and
- a postprocessed selection-table directory
  (``<run_dir>/postprocessed_.../shard_*.npz``) for the **event spans**,

and writes, 1:1 with those shards, an **enriched selection table** per recording:
the same TSV of detected events with the `v0minimal` acoustic feature columns
(see `features_v0minimal.FEATURE_COLS`) appended — one feature vector per event,
computed on the high-pass-filtered focal audio sliced to that event's span.

Audio source
------------
A ``denoised``/``stems`` run (a `DenoisingDetector`) stores a denoised focal
waveform in each shard; features are computed on that. A ``preds`` run (a plain
`Detector`) stores no audio, so features are instead computed on the **original
source audio**, re-read at feature time from the dataset the run was run over
(recovered from the run's ``lineage.yaml``, or from ``input.dataset``). The two
sources differ by construction — a ``preds`` run's features come from un-gated,
full-band source audio — so feature values are not comparable across rungs.

This is a separate stage rather than part of `postprocess` on purpose:
`postprocess` is deliberately audio-free and swept constantly (over thresholds /
NMS), while feature computation must decode audio and run STFTs and is
recomputed far less often. Composing over `postprocess`'s output (the spans)
keeps both stages small and independently runnable. Because the enriched table
is still a `selection_table` TSV string stored 1:1 with the shards, the
`AttachLSISelectionTables` transform attaches it unchanged — point its
``postprocessing`` at this ``features_...`` subdirectory.

Aggregation to a per-recording feature (e.g. the mean over a focal species'
events) is a downstream step at dataframe-assembly time, not baked in here: the
stored unit keeps per-event fidelity.

Output layout::

    <run_dir>/postprocessed_thr0.50_.../
        shard_0000.npz                     ← selection tables (spans), from postprocess
        features_v0minimal/
            shard_0000.npz                 ← enriched selection tables (1:1)
            lineage.yaml

Config format (see `LsiFeaturesConfig`)::

    input:
      run_dir: gs://bucket/lsi/run_name       # dir holding the ItemResult shards
      postprocessing: postprocessed_thr0.50    # subdir under run_dir with the spans
      dataset:                                 # optional; preds runs only, else from lineage
        config: configs/data/inference/xeno_canto.yml  # source-audio dataset the run was run over
    features:
      version: v0minimal                       # optional; only supported value

Usage::

    uv run sed-lsi-features --config config.yml \\
        --job-index $SLURM_ARRAY_TASK_ID --num-jobs 32
    # or override the run dir / postprocessing subdir from the CLI:
    uv run sed-lsi-features --config config.yml \\
        --run-dir DIR --postprocessing postprocessed_thr0.50

The ``features_...`` output directory also gets a ``lineage.yaml`` (written by
job 0) recording the resolved config, the git commit, a UTC timestamp, and —
chained under ``parent`` — the postprocess stage's lineage (which itself chains
back to the run), so an enriched selection table traces back through every stage
(see `write_lineage`).
"""

import io
import json
import os
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
from alp_data import dataset_from_config
from alp_data.io import anypath, exists
from pydantic import ValidationError

from esp_research.logging import logger
from sound_event_detection.inference.config import LsiFeaturesConfig
from sound_event_detection.inference.engine import (
    list_shards,
    read_lineage,
    read_shard,
    save_shard,
    write_lineage,
    write_text,
)
from sound_event_detection.inference.features_v0minimal import (
    FEATURE_COLS,
    compute_v0minimal_features,
    highpass_filter,
)
from sound_event_detection.inference.result import decode_audio

__all__ = ["add_features_to_selection_table", "cli"]

#: Focal-quality scalars carried through from the postprocess shard so the
#: features directory stays a superset drop-in for the attach transform.
_QUALITY_KEYS = ("focal_confidence", "focal_max_stems")


def add_features_to_selection_table(
    selection_table: str,
    audio: np.ndarray,
    sample_rate: float,
) -> str:
    """Append per-event `v0minimal` acoustic features to a selection-table TSV.

    Parses the selection table, high-pass filters `audio` once, then for each
    event slices the filtered waveform to the event's ``[Begin Time (s),
    End Time (s)]`` span and computes the features on that slice (with
    ``duration_s`` taken from the span, not the clip length). The feature columns
    (`FEATURE_COLS`) are appended to the table and it is re-serialized as a TSV
    string. An empty selection table (no events) gains empty feature columns so
    the schema matches non-empty tables.

    Parameters
    ----------
    selection_table : str
        A selection table serialized with ``to_csv(sep="\\t", index=False)``,
        with at least ``Begin Time (s)`` and ``End Time (s)`` columns (the format
        produced by `postprocess`).
    audio : np.ndarray
        Mono waveform of the recording — the stored denoised focal track, or the
        original source audio for a ``preds`` run — shape ``(samples,)``.
    sample_rate : float
        Sample rate of `audio` in Hz.

    Returns
    -------
    str
        The enriched selection table serialized with ``to_csv(sep="\\t",
        index=False)`` — the original columns plus `FEATURE_COLS`.
    """
    table = pd.read_csv(io.StringIO(selection_table), sep="\t")
    if table.empty:
        for col in FEATURE_COLS:
            table[col] = pd.Series(dtype=float)
        return table.to_csv(sep="\t", index=False)

    filtered = highpass_filter(np.asarray(audio, dtype=np.float32), sample_rate)
    n_samples = len(filtered)
    rows: list[dict[str, float]] = []
    for begin, end in zip(table["Begin Time (s)"], table["End Time (s)"], strict=True):
        begin_s = float(begin)
        end_s = float(end)
        start_sample = max(0, int(round(begin_s * sample_rate)))
        end_sample = min(n_samples, int(round(end_s * sample_rate)))
        clip = filtered[start_sample:end_sample]
        rows.append(compute_v0minimal_features(clip, sample_rate, end_s - begin_s))

    features = pd.DataFrame(rows, columns=list(FEATURE_COLS), index=table.index)
    enriched = pd.concat([table, features], axis=1)
    return enriched.to_csv(sep="\t", index=False)


def _decode_denoised(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, int]:
    """Decode a recording's stored denoised waveform and its sample rate.

    Reads the FLAC header for the true sample rate (rather than assuming the
    default) so features are computed against the rate the audio was stored at.

    Parameters
    ----------
    arrays : dict[str, np.ndarray]
        One recording's flat array dict from `read_shard`.

    Returns
    -------
    tuple[np.ndarray, int]
        The float32 waveform in [-1, 1] and its sample rate in Hz.

    Raises
    ------
    KeyError
        If the recording has no ``denoised`` track. Callers branch on the key's
        presence before calling, so this is a defensive internal guard.
    """
    import soundfile as sf

    if "denoised" not in arrays:
        raise KeyError("no denoised track in this recording's arrays")
    data = np.asarray(arrays["denoised"], dtype=np.uint8)
    pcm, sample_rate = sf.read(io.BytesIO(data.tobytes()), dtype="int16")
    return decode_audio(pcm), int(sample_rate)


class _SourceAudioResolver:
    """Map a recording's `file_id` back to its original source audio.

    For a ``preds`` run the shards store no audio, so the feature stage recovers
    each recording's waveform from the dataset the run was run over. The shard's
    ``file_id`` is the string value of the run's ``id_column``; this resolver
    scans the dataset's backend once (without decoding audio) to build a cached
    ``{file_id: row index}`` map, then decodes only the needed recordings via the
    dataset's own item access — reusing every per-dataset path / resample rule so
    the audio matches what the detector saw.

    Parameters
    ----------
    dataset : object
        An `alp_data` dataset (from `dataset_from_config`), indexable by integer
        position (yielding an item with ``audio`` / ``sample_rate``) and exposing
        a ``_data`` backend of undecoded rows.
    id_column : str
        Item key whose string value keyed each recording's shard (the `file_id`).
    """

    def __init__(self, dataset: object, id_column: str) -> None:
        self._dataset = dataset
        self._id_column = id_column
        self._index: dict[str, int] | None = None

    def _index_map(self) -> dict[str, int]:
        """Build (once) and return the ``{file_id: row index}`` map.

        Iterates the dataset's backend rows, which does not decode audio; the
        first row seen for a given id wins, mirroring the producer's shard order.

        Returns
        -------
        dict[str, int]
            Mapping from a recording's `file_id` to its dataset row index.
        """
        if self._index is None:
            mapping: dict[str, int] = {}
            for i, row in enumerate(self._dataset._data):
                mapping.setdefault(str(row[self._id_column]), i)
            self._index = mapping
        return self._index

    def resolve(self, file_id: str) -> tuple[np.ndarray, float]:
        """Return the recording's mono source waveform and its sample rate.

        Parameters
        ----------
        file_id : str
            The recording's shard key (the run's ``id_column`` value).

        Returns
        -------
        tuple[np.ndarray, float]
            The mono float32 waveform and its sample rate in Hz.

        Raises
        ------
        KeyError
            If `file_id` is not present in the dataset under ``id_column``.
        """
        index = self._index_map().get(file_id)
        if index is None:
            raise KeyError(f"file_id {file_id!r} not found in dataset via id_column {self._id_column!r}")
        item = self._dataset[index]
        audio = np.asarray(item["audio"], dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=0)
        return audio, float(item["sample_rate"])


def _resolve_dataset_ref(run_dir: str, config: LsiFeaturesConfig) -> tuple[str, str | None]:
    """Resolve the source-audio dataset config path and id column for a run.

    Prefers the explicit ``input.dataset`` override; otherwise recovers both from
    the run's ``lineage.yaml`` (the dataset config path and the resolved id column
    the run keyed its shards by).

    Parameters
    ----------
    run_dir : str
        The run directory holding the `ItemResult` shards and its ``lineage.yaml``.
    config : LsiFeaturesConfig
        The feature-stage config (its ``input.dataset`` override is used if set).

    Returns
    -------
    tuple[str, str | None]
        The `alp_data` dataset config path and the id column (``None`` to fall
        back to the dataset's originals-path column).

    Raises
    ------
    click.UsageError
        If no override is set and the run's lineage is missing or records no
        dataset config, so a ``preds`` run's source audio cannot be located.
    """
    if config.input.dataset is not None:
        return config.input.dataset.config, config.input.dataset.id_column

    record = read_lineage(run_dir) or {}
    dataset_cfg = record.get("run_config", {}).get("dataset", {})
    cfg_path = dataset_cfg.get("config")
    if not cfg_path:
        raise click.UsageError(
            f"this run stored no denoised audio (a preds run) and its source dataset could not be located: "
            f"{run_dir!r} has no usable lineage.yaml. Re-run with a denoising detector, or set input.dataset "
            "in the feature config to point at the source-audio dataset."
        )
    id_column = record.get("id_column") or dataset_cfg.get("id_column")
    return cfg_path, id_column


def _build_source_resolver(run_dir: str, config: LsiFeaturesConfig) -> _SourceAudioResolver:
    """Build the source-audio resolver for a ``preds`` run.

    Resolves the dataset reference (override or run lineage) via
    `_resolve_dataset_ref` — which raises `click.UsageError` if it cannot be
    located — rebuilds the dataset, and picks the id column (falling back to the
    dataset's originals-path column, matching the run's own default).

    Parameters
    ----------
    run_dir : str
        The run directory (for its lineage, when no override is set).
    config : LsiFeaturesConfig
        The feature-stage config.

    Returns
    -------
    _SourceAudioResolver
        A resolver mapping each recording's `file_id` to its source audio.
    """
    cfg_path, id_column = _resolve_dataset_ref(run_dir, config)
    dataset, _ = dataset_from_config(cfg_path)
    return _SourceAudioResolver(dataset, id_column or dataset._originals_path_column)


def _load_config(path: Path) -> LsiFeaturesConfig:
    """Load the feature-stage config from a YAML file.

    Parameters
    ----------
    path : Path
        Path to the feature config YAML (the ``--config`` file).

    Returns
    -------
    LsiFeaturesConfig
        The validated feature config.

    Raises
    ------
    click.UsageError
        If the file is not a valid feature config (e.g. an unsupported feature
        version or an unknown top-level key).
    """
    try:
        return LsiFeaturesConfig.from_sources(yaml_file=path)
    except ValidationError as exc:
        raise click.UsageError(f"{path}: {exc}") from exc


def _run_features(
    config_path: Path,
    *,
    run_dir_override: str | None,
    postprocessing_override: str | None,
    job_index: int,
    num_jobs: int,
) -> None:
    """Enrich this job's selection-table shards of a run with acoustic features.

    Parameters
    ----------
    config_path : Path
        Path to the feature config YAML.
    run_dir_override : str or None
        If given, overrides the config's ``input.run_dir`` (the `ItemResult`
        shard directory).
    postprocessing_override : str or None
        If given, overrides the config's ``input.postprocessing`` (the
        selection-table subdir under the run dir).
    job_index : int
        Zero-based index of this job in the array.
    num_jobs : int
        Total number of parallel jobs.

    Raises
    ------
    click.UsageError
        If no run directory or no postprocessing subdir is given, or if a
        ``preds`` run (no stored audio) is enriched but its source-audio dataset
        cannot be located (see `_resolve_dataset_ref`).
    ValueError
        If `num_jobs` is not positive, `job_index` is out of range, no
        ``shard_*.npz`` files are found, or a selection-table shard has no
        matching `ItemResult` shard.
    """
    if num_jobs <= 0:
        raise ValueError(f"num_jobs must be positive, got {num_jobs}")
    if not 0 <= job_index < num_jobs:
        raise ValueError(f"job_index {job_index} out of range [0, {num_jobs})")

    config = _load_config(config_path)
    run_dir = run_dir_override or config.input.run_dir
    if not run_dir:
        raise click.UsageError("no run directory: set input.run_dir in the config or pass --run-dir")
    postprocessing = postprocessing_override or config.input.postprocessing
    if not postprocessing:
        raise click.UsageError(
            "no selection-table subdir: set input.postprocessing in the config or pass --postprocessing"
        )
    run_dir = run_dir if not isinstance(anypath(run_dir), Path) else str(Path(run_dir).expanduser())

    version = config.features.version

    st_dir = f"{run_dir.rstrip('/')}/{postprocessing}"
    item_shards = dict(list_shards(run_dir))
    st_shards = list_shards(st_dir)
    if not st_shards:
        raise ValueError(f"no shard_*.npz found under {st_dir!r}")

    out_dir = f"{st_dir.rstrip('/')}/features_{version}"
    if job_index == 0:
        write_lineage(
            out_dir,
            "features",
            # Record the resolved run_dir / postprocessing subdir explicitly so the
            # record is complete even when they came from the CLI overrides.
            {
                "config": config.model_dump(),
                "run_dir": run_dir,
                "postprocessing": postprocessing,
                "version": version,
            },
            parent_dir=st_dir,
        )

    total = len(st_shards)
    start = (total * job_index) // num_jobs
    end = (total * (job_index + 1)) // num_jobs
    print(
        f"[features] {total} shards | job {job_index + 1}/{num_jobs}: shards [{start}, {end}) "
        f"| version={version} -> {out_dir}",
        flush=True,
    )

    # Built lazily on the first recording that lacks a denoised track (a preds
    # run), so denoised/stems runs never touch the source dataset.
    source_resolver: _SourceAudioResolver | None = None

    error_lines: list[str] = []
    shards_done = 0
    files_done = 0
    for shard_idx, st_path in st_shards[start:end]:
        out_path = f"{out_dir}/shard_{shard_idx:04d}.npz"
        if exists(out_path):
            print(f"[features] skip existing shard {shard_idx:04d}", flush=True)
            shards_done += 1
            continue
        if shard_idx not in item_shards:
            raise ValueError(f"selection-table shard {shard_idx:04d} has no matching ItemResult shard in {run_dir!r}")

        shard_start = time.perf_counter()
        item_arrays = read_shard(item_shards[shard_idx])
        st_arrays_by_id = read_shard(st_path)
        # A preds run stores no audio; build the source-audio resolver once (outside
        # the per-recording try, so a missing dataset fails fast rather than being
        # logged per recording as a spurious per-file failure).
        needs_source = any(fid in item_arrays and "denoised" not in item_arrays[fid] for fid in st_arrays_by_id)
        if needs_source and source_resolver is None:
            source_resolver = _build_source_resolver(run_dir, config)

        items: list[tuple[str, dict[str, np.ndarray]]] = []
        for file_id, st_arrays in st_arrays_by_id.items():
            try:
                # 1:1 shards share file_ids; a missing entry KeyErrors here and is
                # isolated and logged like any other per-recording failure.
                arrays = item_arrays[file_id]
                if "denoised" in arrays:
                    audio, sample_rate = _decode_denoised(arrays)
                else:
                    audio, sample_rate = source_resolver.resolve(file_id)
                enriched = add_features_to_selection_table(str(st_arrays["selection_table"]), audio, sample_rate)
                entry: dict[str, np.ndarray] = {"selection_table": np.array(enriched, dtype=np.str_)}
                for key in _QUALITY_KEYS:
                    if key in st_arrays:
                        entry[key] = st_arrays[key]
                items.append((file_id, entry))
            except Exception as exc:  # noqa: BLE001 — isolate one bad recording, keep the shard going
                error_lines.append(json.dumps({"file_id": file_id, "shard": shard_idx, "error": str(exc)}))
                print(f"[features] error on {file_id} (shard {shard_idx:04d}): {exc}", flush=True)

        save_shard(out_path, items, job_index)
        elapsed = time.perf_counter() - shard_start
        print(f"[features] saved shard {shard_idx:04d} ({len(items)} files) in {elapsed:.1f}s", flush=True)
        shards_done += 1
        files_done += len(items)

    print(f"[features] done: {shards_done} shards, {files_done} files, {len(error_lines)} errors", flush=True)
    if error_lines:
        error_log_path = os.path.join(out_dir, f"errors_job_{job_index:03d}.jsonl")
        write_text(error_log_path, "\n".join(error_lines) + "\n")
        print(f"[features] errors logged to {error_log_path}", flush=True)


@click.group(invoke_without_command=True)
@click.option(
    "--config",
    "config_path",
    required=False,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the feature config YAML. Required for a run.",
)
@click.option(
    "--run-dir",
    default=None,
    help="Override input.run_dir (the ItemResult shard directory).",
)
@click.option(
    "--postprocessing",
    default=None,
    help="Override input.postprocessing (the selection-table subdir under the run dir).",
)
@click.option("--job-index", type=int, default=0, help="Zero-based index of this job in the array.")
@click.option("--num-jobs", type=int, default=1, help="Total number of parallel jobs.")
@click.pass_context
def cli(
    ctx: click.Context,
    config_path: Path | None,
    run_dir: str | None,
    postprocessing: str | None,
    job_index: int,
    num_jobs: int,
) -> None:
    """ESP Research — LSI feature stage (enrich selection tables with features).

    Enrich this job's shard range::

        sed-lsi-features --config <path> [--run-dir DIR] [--postprocessing SUBDIR] [--job-index N] [--num-jobs M]

    Raises
    ------
    click.UsageError
        If ``--config`` is missing for a run.
    """
    if ctx.invoked_subcommand is not None:
        return

    if config_path is None:
        raise click.UsageError("--config is required for a run.")
    _run_features(
        config_path,
        run_dir_override=run_dir,
        postprocessing_override=postprocessing,
        job_index=job_index,
        num_jobs=num_jobs,
    )


@cli.command()
def describe() -> None:
    """Print the LSI feature-stage config schema."""
    logger.info("LSI Feature Config Schema (--config):")
    logger.info(json.dumps(LsiFeaturesConfig.model_json_schema(), indent=2))


if __name__ == "__main__":
    cli()
