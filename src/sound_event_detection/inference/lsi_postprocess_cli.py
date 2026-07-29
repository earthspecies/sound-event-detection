"""Postprocessing stage CLI: LSI `ItemResult` shards -> selection-table shards.

Entry point: ``uv run sed-lsi-postprocess``.

Commands
--------
- ``sed-lsi-postprocess --config <path> [--run-dir DIR] [--job-index N] [--num-jobs M]``
    postprocess this job's slice of a run's `ItemResult` shards.
- ``sed-lsi-postprocess describe``
    print the postprocess config schema.

The second stage of the LSI pipeline (produce `ItemResult` shards -> postprocess
-> attach). It reads the combined framewise predictions stored in each
``shard_*.npz`` and turns them into a per-recording selection table (a TSV string
of detected events), written back as ``shard_*.npz`` **1:1 with the input shards**
(same index / basename). That 1:1 correspondence is the contract the
`AttachLSISelectionTables` transform relies on to resolve a row's heavy shard from
its selection-table shard without any manifest.

The event math is entirely reused, not reimplemented:

- `sound_event_detection.utils.reformatters.frames_to_selection_table` turns a
  thresholded ``(T, C)`` prediction array into event intervals (with a per-event
  ``Score`` = mean probability), and
- `sound_event_detection.utils.postprocessing.postprocess_selection_table`
  applies the merge -> min-duration -> NMS chain.

One detection threshold is baked into a run; the output directory name encodes the
threshold and postprocessing params (`_output_dir_name`), so re-thresholding is
just a re-run of this cheap stage into a sibling directory.

**By default this does nothing but threshold at 0.5** — no merge, no min-duration,
no NMS, no filtering. Those steps only run when their keys are set, so a bare
config yields plain 0.5-thresholded selection tables (dir ``postprocessed_thr0.50``).

Filtering (a class allowlist and geography priors) is applied **before** the
merge / min-duration / NMS chain, so a detection that geography would reject can
never win NMS over a real one. Geography filtering is off unless `geo_filter` is
set; it needs `geopandas`/`shapely` (imported only then), reads each recording's
latitude/longitude from its `ItemResult` shard, and globs its range maps
(``*.gpkg``) from the required `range_map_dir` (checked to exist at the outset).
When a run stored
per-frame focal quality (`focal_detprob`/`focal_nstems`, from a denoising run),
they are pooled per recording into ``focal_confidence`` (mean) and
``focal_max_stems`` (max) scalars alongside the selection table, for downstream
quality filtering.

Config format (everything except a run directory is optional; see
`LsiPostprocessConfig`)::

    input:
      run_dir: gs://bucket/lsi/run_name    # dir holding the ItemResult shard_*.npz
                                           # (or pass --run-dir to postprocess many)
    postprocessing:
      threshold: 0.5                       # optional; default 0.5
      merge_max_gap: 1.0                   # optional; seconds, off unless set
      min_event_duration: 0.01             # optional; seconds, off unless set
      nms: 0.8                             # optional; IoU threshold, off unless set
      annotation_col: Species              # optional; label column name
      allowed_classes: [Species A, ...]    # optional; drop rows outside this set (pre-NMS)
      geo_filter: true                     # optional; drop out-of-range species (pre-NMS)
      range_map_dir: geography/range_maps  # required when geo_filter; dir of *.gpkg range maps

Usage::

    uv run sed-lsi-postprocess --config config.yml \\
        --job-index $SLURM_ARRAY_TASK_ID --num-jobs 32
    # or override the run dir (postprocess several runs with one config):
    uv run sed-lsi-postprocess --config config.yml --run-dir DIR

Each run's output directory also gets a ``lineage.yaml`` (written by job 0)
recording the resolved config, the git commit, a UTC timestamp, and — chained
under ``parent`` — the run's own lineage, so a selection table traces back to the
model and dataset that produced it (see `write_lineage`).
"""

import json
import os
import time
import warnings
from pathlib import Path

import click
import numpy as np
import pandas as pd
from alp_data.io import anypath, exists, filesystem_from_path
from pydantic import ValidationError

from esp_research.logging import logger
from esp_research.protocols.detector import DetectorOutput
from sound_event_detection.inference.config import LsiPostprocessConfig
from sound_event_detection.inference.engine import (
    list_shards,
    read_shard,
    save_shard,
    write_lineage,
    write_text,
)
from sound_event_detection.inference.result import decode_preds
from sound_event_detection.utils.postprocessing import postprocess_selection_table
from sound_event_detection.utils.reformatters import frames_to_selection_table

__all__ = ["cli", "postprocess_and_convert_detector_output_to_selection_table"]

#: Glob pattern for the range-map files consulted by the geography filter.
_RANGE_MAP_GLOB = "*.gpkg"

#: Cap on geography-filter warnings so a location-less run does not flood the log.
_geo_filter_warning_count = [0]
_GEO_FILTER_WARNING_LIMIT = 100


def _output_dir_name(threshold: float, pp: dict, geo_filter: bool = False) -> str:
    """Build the postprocessed-selection-table directory name from the run's params.

    The name encodes every step that shaped the tables — threshold, merge,
    min-duration, NMS, and geography filtering — so each distinct postprocessing
    lives in its own sibling directory and re-thresholding never overwrites.

    Parameters
    ----------
    threshold : float
        Detection threshold baked into this run.
    pp : dict
        Postprocessing config as passed to `postprocess_selection_table` (keys
        ``merge_max_gap``, ``min_event_duration``, ``nms``); only enabled steps
        appear in the name.
    geo_filter : bool
        Whether geography filtering was applied (adds a ``geo`` token).

    Returns
    -------
    str
        Directory name, e.g. ``postprocessed_thr0.50_merge1.00_minDur0.01_nms0.80_geo``.
    """
    parts = [f"thr{threshold:.2f}"]
    merge_gap = pp.get("merge_max_gap", 0)
    min_dur = pp.get("min_event_duration", 0)
    nms = pp.get("nms")
    if merge_gap and merge_gap > 0:
        parts.append(f"merge{merge_gap:.2f}")
    if min_dur and min_dur > 0:
        parts.append(f"minDur{min_dur:.2f}")
    if nms is not None:
        parts.append(f"nms{nms['iou_threshold']:.2f}")
    if geo_filter:
        parts.append("geo")
    return "postprocessed_" + "_".join(parts)


def _dir_exists(path: str) -> bool:
    """Return whether `path` is an existing directory (local or cloud).

    Parameters
    ----------
    path : str
        Directory path (local path or cloud URI, e.g. ``gs://...``).

    Returns
    -------
    bool
        ``True`` if the path exists and is a directory.
    """
    if not isinstance(anypath(path), Path):
        fs = filesystem_from_path(path)
        return fs.isdir(fs._strip_protocol(path).rstrip("/"))
    return Path(path).expanduser().is_dir()


def _list_range_map_files(range_map_dir: str) -> list[str]:
    """Glob the ``*.gpkg`` range-map files in `range_map_dir`, sorted by path.

    Works on local and cloud (``gs://``/``s3://``/``r2://``) directories, mirroring
    `list_shards`.

    Parameters
    ----------
    range_map_dir : str
        Directory (local path or cloud URI) holding the range-map files.

    Returns
    -------
    list[str]
        Paths of the ``*.gpkg`` files in `range_map_dir`, sorted.
    """
    if not isinstance(anypath(range_map_dir), Path):
        scheme = range_map_dir.split("://", 1)[0]
        fs = filesystem_from_path(range_map_dir)
        stripped = fs._strip_protocol(range_map_dir).rstrip("/")
        paths = [f"{scheme}://{match}" for match in fs.glob(f"{stripped}/{_RANGE_MAP_GLOB}")]
    else:
        paths = [str(path) for path in Path(range_map_dir).expanduser().glob(_RANGE_MAP_GLOB)]
    return sorted(paths)


def _load_range_maps(range_map_files: list[str]) -> "object":
    """Load and GBIF-convert the range maps from the given files.

    Imported lazily by the geography filter so the default postprocessing path
    never needs `geopandas`.

    Parameters
    ----------
    range_map_files : list[str]
        Paths of the ``*.gpkg`` range-map files to load (from
        `_list_range_map_files`).

    Returns
    -------
    geopandas.GeoDataFrame
        Concatenated range maps with an added ``gbif_name`` column; rows whose
        ``name`` could not be mapped to a GBIF canonical name are dropped.
    """
    import geopandas as gpd
    from alp_data.discover import GBIFConverter

    print(f"[postprocess] loading {len(range_map_files)} range-map files...", flush=True)
    gdf = pd.concat([gpd.read_file(path) for path in range_map_files], ignore_index=True)

    converter = GBIFConverter()
    gbif_names = []
    for name in gdf["name"]:
        info, ok = converter(name)
        gbif_names.append(info["canonicalName"] if ok else None)
    gdf = gdf.copy()
    gdf["gbif_name"] = gbif_names
    gdf = gdf[gdf["gbif_name"].notna()].reset_index(drop=True)
    print(f"[postprocess] range maps loaded: {len(gdf)} rows after GBIF conversion", flush=True)
    return gdf


def filter_by_geography(
    selection_table: pd.DataFrame,
    range_map_gdf: "object",
    latitude: float,
    longitude: float,
    annotation_col: str = "Species",
) -> pd.DataFrame:
    """Drop detections for species whose range maps exclude the recording location.

    Fails open: a species is removed only on positive out-of-range evidence —
    the recording has valid coordinates **and** the species' range map exists
    and excludes the recording point. If the recording has no lat/long, geography
    filtering is skipped entirely (the table is returned unchanged); a species
    absent from the range maps is kept (missing data is not evidence of absence).
    Warnings are capped at `_GEO_FILTER_WARNING_LIMIT` so a coordinate-less run
    does not flood the log.

    Parameters
    ----------
    selection_table : pd.DataFrame
        Detection table with an annotation column (e.g. ``"Species"``).
    range_map_gdf : geopandas.GeoDataFrame
        Range maps with a ``gbif_name`` column and geometry, from `_load_range_maps`.
    latitude : float
        Decimal latitude of the recording (WGS84); ``nan`` if unknown.
    longitude : float
        Decimal longitude of the recording (WGS84); ``nan`` if unknown.
    annotation_col : str
        Column holding species names. Default ``"Species"``.

    Returns
    -------
    pd.DataFrame
        The selection table with out-of-range species removed. Unchanged when
        the recording has no valid coordinates.
    """
    from shapely.geometry import Point

    def _warn(message: str) -> None:
        if _geo_filter_warning_count[0] < _GEO_FILTER_WARNING_LIMIT:
            warnings.warn(message, stacklevel=2)
            _geo_filter_warning_count[0] += 1

    if selection_table.empty:
        return selection_table

    # No coordinates -> can't evaluate range membership, so filter nothing.
    if np.isnan(latitude) or np.isnan(longitude):
        _warn("[geo_filter] no valid lat/long for recording; skipping geography filter.")
        return selection_table

    import geopandas as gpd

    point_wgs84 = gpd.GeoDataFrame(geometry=[Point(longitude, latitude)], crs="EPSG:4326")
    point_geom = point_wgs84.to_crs(range_map_gdf.crs).geometry[0]

    rows_to_drop = []
    for species in selection_table[annotation_col].unique():
        species_rows = range_map_gdf[range_map_gdf["gbif_name"] == species]
        if species_rows.empty:
            # No range map for this species -> can't confirm out-of-range; keep it.
            _warn(f"[geo_filter] '{species}' not found in range maps; keeping.")
            continue
        if not any(geom.contains(point_geom) for geom in species_rows.geometry):
            _warn(f"[geo_filter] '{species}' range excludes the recording location; removing.")
            rows_to_drop.append(species)

    if rows_to_drop:
        selection_table = selection_table[~selection_table[annotation_col].isin(rows_to_drop)]
    return selection_table


def postprocess_and_convert_detector_output_to_selection_table(
    preds: DetectorOutput,
    threshold: float,
    pp_config: dict,
    annotation_col: str = "Species",
    *,
    allowed_classes: set[str] | None = None,
    range_map_gdf: "object | None" = None,
    latitude: float = float("nan"),
    longitude: float = float("nan"),
) -> str:
    """Turn one recording's combined predictions into a selection-table TSV string.

    Thresholds the ``(T, C)`` probabilities and extracts events with a per-event
    ``Score`` via `frames_to_selection_table`. Then, **before** the merge /
    min-duration / NMS chain, applies any filtering — a class allowlist and
    geography priors — so a detection that filtering would reject can never win
    NMS over a real one. Finally applies `postprocess_selection_table` and
    serializes the result as a tab-separated table with a header (the exact
    format the `selection_table` dataset column holds). An empty result is a
    header-only line (with a ``Score`` column, kept consistent with non-empty
    tables).

    Parameters
    ----------
    preds : DetectorOutput
        Combined framewise predictions, shape ``(1, T, C)``, in [0, 1].
    threshold : float
        Binarization threshold applied before event extraction.
    pp_config : dict
        Postprocessing config forwarded to `postprocess_selection_table`
        (``merge_max_gap`` / ``min_event_duration`` / ``nms``).
    annotation_col : str
        Name of the label column in the output table.
    allowed_classes : set[str] or None
        When given, rows whose `annotation_col` is not in this set are dropped
        (before postprocessing). ``None`` keeps all classes.
    range_map_gdf : geopandas.GeoDataFrame or None
        Range maps for geography filtering (from `_load_range_maps`). When given,
        out-of-range species are dropped (before postprocessing) using
        `latitude`/`longitude`. ``None`` disables geography filtering.
    latitude : float
        Recording latitude (WGS84) for geography filtering; ``nan`` if unknown.
    longitude : float
        Recording longitude (WGS84) for geography filtering; ``nan`` if unknown.

    Returns
    -------
    str
        The selection table serialized with ``to_csv(sep="\\t", index=False)``.
    """
    values = np.asarray(preds.predictions[0], dtype=np.float32)  # (T, C)
    labels = list(preds.class_names)
    binary = values >= threshold
    table = frames_to_selection_table(
        binary, labels, preds.frame_rate, annotation_col=annotation_col, probs=values if values.size else None
    )
    if "Score" not in table.columns:
        # Empty tables (no class survived encoding) lack a Score column; add it so
        # the schema matches non-empty tables and NMS never KeyErrors.
        table["Score"] = pd.Series(dtype=float)
    # Filter BEFORE postprocessing so a rejected class can't win NMS over a real one.
    if allowed_classes is not None:
        table = table[table[annotation_col].isin(allowed_classes)]
    if range_map_gdf is not None:
        table = filter_by_geography(table, range_map_gdf, latitude, longitude, annotation_col)
    table = postprocess_selection_table(table, pp_config, annotation_col=annotation_col)
    return table.to_csv(sep="\t", index=False)


def _combined_preds(arrays: dict[str, np.ndarray]) -> DetectorOutput:
    """Decode only the combined-prediction group from a recording's shard arrays.

    Avoids `ItemResult.from_arrays`, which would also FLAC-decode the (unused)
    denoised and stem audio for every recording.

    Parameters
    ----------
    arrays : dict[str, np.ndarray]
        One recording's flat array dict from `read_shard`.

    Returns
    -------
    DetectorOutput
        The combined framewise predictions.
    """
    prefix = "preds_"
    group = {key[len(prefix) :]: value for key, value in arrays.items() if key.startswith(prefix)}
    return decode_preds(group)


def _pool_quality(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Pool a recording's per-frame focal-quality tracks into scalar summaries.

    Present only for denoising runs (which store the `focal_detprob` /
    `focal_nstems` tracks); a plain-predictions run returns an empty dict.

    Parameters
    ----------
    arrays : dict[str, np.ndarray]
        One recording's flat array dict from `read_shard`.

    Returns
    -------
    dict[str, np.ndarray]
        ``{"focal_confidence", "focal_max_stems"}`` as float32 scalars when the
        focal-quality tracks are present, else ``{}``. `focal_confidence` is the
        mean focal detection probability over frames where the focal species was
        detected (``nan`` if it never was); `focal_max_stems` is the peak number
        of stems combined in any frame.
    """
    out: dict[str, np.ndarray] = {}
    if "focal_detprob" in arrays:
        detprob = np.asarray(arrays["focal_detprob"], dtype=np.float32)
        all_nan = detprob.size == 0 or bool(np.all(np.isnan(detprob)))
        confidence = np.float32("nan") if all_nan else np.float32(np.nanmean(detprob))
        out["focal_confidence"] = np.asarray(confidence, dtype=np.float32)
    if "focal_nstems" in arrays:
        nstems = np.asarray(arrays["focal_nstems"], dtype=np.float32)
        peak = np.float32(nstems.max()) if nstems.size else np.float32(0.0)
        out["focal_max_stems"] = np.asarray(peak, dtype=np.float32)
    return out


def _load_config(path: Path) -> LsiPostprocessConfig:
    """Load the postprocess config from a YAML file.

    Parameters
    ----------
    path : Path
        Path to the postprocess config YAML (the ``--config`` file).

    Returns
    -------
    LsiPostprocessConfig
        The validated postprocess config.

    Raises
    ------
    click.UsageError
        If the file is not a valid postprocess config (e.g. an unknown top-level
        key or a wrong-typed field).
    """
    try:
        return LsiPostprocessConfig.from_sources(yaml_file=path)
    except ValidationError as exc:
        raise click.UsageError(f"{path}: {exc}") from exc


def _run_postprocess(
    config_path: Path,
    *,
    run_dir_override: str | None,
    job_index: int,
    num_jobs: int,
) -> None:
    """Postprocess this job's shard range of a run into selection-table shards.

    Parameters
    ----------
    config_path : Path
        Path to the postprocess config YAML.
    run_dir_override : str or None
        If given, overrides the config's ``input.run_dir`` (postprocess a
        different run with the same config).
    job_index : int
        Zero-based index of this job in the array.
    num_jobs : int
        Total number of parallel jobs.

    Raises
    ------
    click.UsageError
        If no run directory is given (neither ``input.run_dir`` nor ``--run-dir``),
        or if geography filtering is enabled but ``postprocessing.range_map_dir``
        does not exist or holds no ``*.gpkg`` range maps.
    ValueError
        If `num_jobs` is not positive, `job_index` is out of range, or no
        ``shard_*.npz`` files are found in the run directory.
    """
    if num_jobs <= 0:
        raise ValueError(f"num_jobs must be positive, got {num_jobs}")
    if not 0 <= job_index < num_jobs:
        raise ValueError(f"job_index {job_index} out of range [0, {num_jobs})")

    config = _load_config(config_path)
    run_dir = run_dir_override or config.input.run_dir
    if not run_dir:
        raise click.UsageError("no run directory: set input.run_dir in the config or pass --run-dir")
    run_dir = run_dir if not isinstance(anypath(run_dir), Path) else str(Path(run_dir).expanduser())

    # Everything but the run dir is optional; a bare config means "threshold at 0.5,
    # no merge / min-duration / NMS / filtering".
    pp = config.postprocessing
    threshold = pp.threshold
    annotation_col = pp.annotation_col
    allowed_classes = set(pp.allowed_classes) if pp.allowed_classes else None
    geo_filter = pp.geo_filter

    # Resolve and validate the range-map directory up front (before any shard work)
    # so a misconfigured geography filter fails fast. `range_map_dir` is guaranteed
    # set when geo_filter is on (PostprocessingConfig validates this on load).
    range_map_files: list[str] = []
    if geo_filter:
        range_map_dir = pp.range_map_dir
        if not _dir_exists(range_map_dir):
            raise click.UsageError(f"postprocessing.range_map_dir does not exist: {range_map_dir!r}")
        range_map_files = _list_range_map_files(range_map_dir)
        if not range_map_files:
            raise click.UsageError(
                f"no {_RANGE_MAP_GLOB!r} range maps found in postprocessing.range_map_dir: {range_map_dir!r}"
            )

    pp_config: dict = {}
    if pp.merge_max_gap:
        pp_config["merge_max_gap"] = pp.merge_max_gap
    if pp.min_event_duration:
        pp_config["min_event_duration"] = pp.min_event_duration
    if pp.nms is not None:
        pp_config["nms"] = {"iou_threshold": pp.nms}

    shards = list_shards(run_dir)
    if not shards:
        raise ValueError(f"no shard_*.npz found under {run_dir!r}")

    # Load range maps once per job (only when geography filtering is enabled).
    range_map_gdf = _load_range_maps(range_map_files) if geo_filter else None

    out_dir = f"{run_dir.rstrip('/')}/{_output_dir_name(threshold, pp_config, geo_filter=geo_filter)}"
    if job_index == 0:
        write_lineage(
            out_dir,
            "postprocess",
            # Record the resolved run_dir explicitly so the record is complete
            # even when it came from --run-dir rather than the config.
            {"config": config.model_dump(), "run_dir": run_dir, "threshold": threshold, "postprocessing": pp_config},
            parent_dir=run_dir,
        )

    total = len(shards)
    start = (total * job_index) // num_jobs
    end = (total * (job_index + 1)) // num_jobs
    print(
        f"[postprocess] {total} shards | job {job_index + 1}/{num_jobs}: shards [{start}, {end}) "
        f"| threshold={threshold} pp={pp_config} -> {out_dir}",
        flush=True,
    )

    error_lines: list[str] = []
    shards_done = 0
    files_done = 0
    for shard_idx, shard_path in shards[start:end]:
        out_path = f"{out_dir}/shard_{shard_idx:04d}.npz"
        if exists(out_path):
            print(f"[postprocess] skip existing shard {shard_idx:04d}", flush=True)
            shards_done += 1
            continue

        shard_start = time.perf_counter()
        items: list[tuple[str, dict[str, np.ndarray]]] = []
        for file_id, arrays in read_shard(shard_path).items():
            try:
                tsv = postprocess_and_convert_detector_output_to_selection_table(
                    _combined_preds(arrays),
                    threshold,
                    pp_config,
                    annotation_col,
                    allowed_classes=allowed_classes,
                    range_map_gdf=range_map_gdf,
                    latitude=float(arrays["latitude"]) if "latitude" in arrays else float("nan"),
                    longitude=float(arrays["longitude"]) if "longitude" in arrays else float("nan"),
                )
                entry: dict[str, np.ndarray] = {"selection_table": np.array(tsv, dtype=np.str_)}
                entry.update(_pool_quality(arrays))
                items.append((file_id, entry))
            except Exception as exc:  # noqa: BLE001 — isolate one bad recording, keep the shard going
                error_lines.append(json.dumps({"file_id": file_id, "shard": shard_idx, "error": str(exc)}))
                print(f"[postprocess] error on {file_id} (shard {shard_idx:04d}): {exc}", flush=True)

        save_shard(out_path, items, job_index)
        elapsed = time.perf_counter() - shard_start
        print(f"[postprocess] saved shard {shard_idx:04d} ({len(items)} files) in {elapsed:.1f}s", flush=True)
        shards_done += 1
        files_done += len(items)

    print(f"[postprocess] done: {shards_done} shards, {files_done} files, {len(error_lines)} errors", flush=True)
    if error_lines:
        error_log_path = os.path.join(out_dir, f"errors_job_{job_index:03d}.jsonl")
        write_text(error_log_path, "\n".join(error_lines) + "\n")
        print(f"[postprocess] errors logged to {error_log_path}", flush=True)


@click.group(invoke_without_command=True)
@click.option(
    "--config",
    "config_path",
    required=False,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the postprocess config YAML. Required for a run.",
)
@click.option(
    "--run-dir",
    default=None,
    help="Override input.run_dir (postprocess a different run with the same config).",
)
@click.option("--job-index", type=int, default=0, help="Zero-based index of this job in the array.")
@click.option("--num-jobs", type=int, default=1, help="Total number of parallel jobs.")
@click.pass_context
def cli(
    ctx: click.Context,
    config_path: Path | None,
    run_dir: str | None,
    job_index: int,
    num_jobs: int,
) -> None:
    """ESP Research — LSI postprocess stage (ItemResult shards -> selection tables).

    Postprocess this job's shard range::

        sed-lsi-postprocess --config <path> [--run-dir DIR] [--job-index N] [--num-jobs M]

    Raises
    ------
    click.UsageError
        If ``--config`` is missing for a run.
    """
    if ctx.invoked_subcommand is not None:
        return

    if config_path is None:
        raise click.UsageError("--config is required for a run.")
    _run_postprocess(config_path, run_dir_override=run_dir, job_index=job_index, num_jobs=num_jobs)


@cli.command()
def describe() -> None:
    """Print the LSI postprocess config schema."""
    logger.info("LSI Postprocess Config Schema (--config):")
    logger.info(json.dumps(LsiPostprocessConfig.model_json_schema(), indent=2))


if __name__ == "__main__":
    cli()
