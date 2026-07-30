"""CLI for config-driven large-scale inference (LSI).

Entry point: ``uv run sed-lsi``.

Commands
--------
- ``sed-lsi --run-config <path> --httpclient-config <path> [--job-index N] [--num-jobs M]``
    run this job's slice of the dataset.
- ``sed-lsi describe``
    print the run-config schema and the http-client config schema.

Builds a dataset from the run config and a detector client from the http-client
config, wraps the client in a `process` closure selected by the output detail
level, and runs the sharded engine over this job's slice of the dataset.

Splits between a run config (*what* to run) and an http-client config (*where*
the served model lives and *how* to reach it), mirroring `sed-eval`. The
``--run-config`` YAML (`LsiRunConfig`) names the dataset and the sharded output;
it says nothing about the model. The ``--httpclient-config`` YAML holds the pure
http-client config consumed by `detector_client_from_config` (``url`` plus
optional ``timeout`` / ``retries`` / ``auth``); the kind of client is
auto-detected from the server, so the same file shape reaches a plain detector
server (``sed-server``) or a denoising detector server (``sed-denoising-server``). The
``denoised`` / ``stems`` detail rungs require the ``url`` to point at a
``sed-denoising-server`` server.

The pipeline is dataset-agnostic: any `alp_data` dataset is named by config, the
file-identifier column defaults to the dataset's own originals-path column, and
the focal-species column (needed only for the ``denoised`` / ``stems`` rungs) is
named by config too. Nothing here is specific to a particular corpus.
"""

import json
import time
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path

import click
import numpy as np
from alp_data import dataset_from_config
from alp_data.io import anypath
from pydantic import ValidationError

import sound_event_detection.data.transforms  # noqa: F401 — registers custom transforms
from esp_research.adapters.client_config import HttpClientConfig
from esp_research.logging import logger
from esp_research.protocols.detector import DetectorOutput
from sound_event_detection.adapters.dispatch import DetectorClient, detector_client_from_config
from sound_event_detection.inference.config import LsiRunConfig
from sound_event_detection.inference.engine import run_sharded, write_lineage
from sound_event_detection.inference.result import Detail, ItemResult, Stem

__all__ = ["cli", "make_process"]

#: Rolling-mean logging cadence for `_StageStats`: after the first few per-item
#: lines, one summary line is printed every this-many items.
_LOG_EVERY = 20


class _StageStats:
    """Accumulate per-stage timings and log a per-item / rolling-mean breakdown.

    A single instance is shared by the `process` closure across all items in a
    job. Each `record` call adds one item's stage times to the running totals
    and, for the first few items and then every `_LOG_EVERY` items, prints both
    that item's breakdown and the rolling mean. The logging style mirrors
    `evaluation.py`'s per-file lines so a bottleneck (e.g. the client-side stem
    resample) is visible in the job log without extra tooling.

    Parameters
    ----------
    log_every : int
        Cadence for rolling-mean summary lines after the first five items.
    """

    def __init__(self, log_every: int = _LOG_EVERY) -> None:
        self._log_every = log_every
        self._totals: dict[str, float] = {}
        self._count = 0
        self._audio_total = 0.0

    def record(self, timings: Mapping[str, float], audio_seconds: float) -> None:
        """Add one item's stage times to the totals and log periodically.

        Parameters
        ----------
        timings : Mapping[str, float]
            Stage name -> wall seconds for this item.
        audio_seconds : float
            Duration of this item's audio, for a real-time-factor sense.
        """
        self._count += 1
        self._audio_total += audio_seconds
        for stage, seconds in timings.items():
            self._totals[stage] = self._totals.get(stage, 0.0) + seconds

        if self._count > 5 and self._count % self._log_every != 0:
            return
        item_total = sum(timings.values())
        per_item = " ".join(f"{stage}={seconds:.2f}s" for stage, seconds in timings.items())
        mean = " ".join(f"{stage}={total / self._count:.2f}s" for stage, total in self._totals.items())
        print(
            f"[lsi] item {self._count}: audio={audio_seconds:.1f}s {per_item} total={item_total:.2f}s "
            f"| mean/item ({self._count}): {mean}",
            flush=True,
        )


def make_process(
    model: DetectorClient,
    focal_column: str | None,
    detail: Detail,
    preds_threshold: float,
    latitude_column: str = "latitudeDecimal",
    longitude_column: str = "longitudeDecimal",
    max_audio_seconds: float | None = None,
) -> Callable[[Mapping], dict[str, np.ndarray]]:
    """Build the per-item `process` closure for the engine.

    The returned closure runs the model on one item's audio and encodes an
    `ItemResult` at the requested detail rung:

    - ``preds`` -> `model.run` on any `DetectorClient`; combined predictions only.
    - ``denoised`` -> `model.separate_and_detect`; combined predictions + a
      denoised waveform gated at the model's threshold.
    - ``stems`` -> the same core, additionally storing every stem (audio +
      predictions) so any threshold re-gates downstream.

    Every rung also stores the recording's latitude/longitude (for downstream
    geo filtering); the denoising rungs additionally store the per-frame quality
    tracks from `StemDetections.quality` (focal detection probability and
    gated-stem count) for downstream quality filtering.

    Parameters
    ----------
    model : DetectorClient
        A detector client (used via `run` on the ``preds`` rung). For
        ``denoised`` / ``stems`` it must additionally expose
        `separate_and_detect`, `labels`, and `threshold` (i.e. a
        `ServedDenoisingDetectorClient` reaching a denoising detector
        server) — denoising is an add-on on top of the `DetectorClient` run
        surface.
    focal_column : str or None
        Item key holding the focal-species label. Required for ``denoised`` /
        ``stems``; unused for ``preds``.
    detail : Detail
        The detail rung to emit.
    preds_threshold : float
        Max-probability threshold forwarded to the prediction codec.
    latitude_column : str
        Item key holding decimal latitude. Missing values (or a missing column)
        are stored as ``nan`` (a one-time warning is emitted if the column is
        absent). Default ``"latitudeDecimal"``.
    longitude_column : str
        Item key holding decimal longitude; same handling as `latitude_column`.
        Default ``"longitudeDecimal"``.
    max_audio_seconds : float | None
        Raise (so the engine skips + logs the item) before touching the model
        when the audio is longer than this. ``None`` disables the guard.

    Returns
    -------
    Callable[[Mapping], dict[str, np.ndarray]]
        The producer passed to `run_sharded`.

    Raises
    ------
    ValueError
        If `detail` requires denoising but `model` lacks `separate_and_detect`,
        or if `detail` requires a focal species but `focal_column` is ``None``.

    Notes
    -----
    The closure keeps a shared `_StageStats` that logs a per-stage timing
    breakdown (per item for the first few, then a rolling mean) so the pipeline
    bottleneck is visible in the job log. Stages are ``run`` / ``encode`` for the
    ``preds`` rung and ``separate`` / ``resample`` / ``detect`` / ``denoise`` /
    ``encode`` for the denoising rungs.
    """
    if detail != "preds":
        if not hasattr(model, "separate_and_detect"):
            raise ValueError(
                f"detail={detail!r} requires a denoising detector server "
                "(a client with separate_and_detect; point model.url at a sed-denoising-server server)"
            )
        if focal_column is None:
            raise ValueError(f"detail={detail!r} requires dataset.focal_column to be set")

    sample_rate = getattr(model, "sample_rate", 0) or 1
    stats = _StageStats()
    warned_latlon = {"done": False}

    def read_latlon(item: Mapping) -> tuple[float, float]:
        """Read decimal lat/long from `item`, or ``nan`` (warning once if absent).

        Returns
        -------
        tuple[float, float]
            ``(latitude, longitude)``; each is ``nan`` when its column is absent
            or its value is ``None``.
        """
        if (latitude_column not in item or longitude_column not in item) and not warned_latlon["done"]:
            warnings.warn(
                f"lat/long columns {latitude_column!r}/{longitude_column!r} not in dataset items; storing nan",
                stacklevel=2,
            )
            warned_latlon["done"] = True
        lat = item.get(latitude_column)
        lon = item.get(longitude_column)
        return (
            float(lat) if lat is not None else float("nan"),
            float(lon) if lon is not None else float("nan"),
        )

    def process(item: Mapping) -> dict[str, np.ndarray]:
        audio = np.asarray(item["audio"], dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=0)
        audio_seconds = audio.shape[0] / sample_rate
        if max_audio_seconds is not None and audio_seconds > max_audio_seconds:
            raise ValueError(
                f"audio {audio_seconds / 60:.1f} min exceeds max_audio_seconds="
                f"{max_audio_seconds:.0f}s; skipped (over-long files can crash the separator)"
            )
        timings: dict[str, float] = {}
        latitude, longitude = read_latlon(item)

        if detail == "preds":
            start = time.perf_counter()
            preds = model.run(audio[np.newaxis, :])
            timings["run"] = time.perf_counter() - start
            arrays = ItemResult(preds=preds, latitude=latitude, longitude=longitude).to_arrays(
                preds_threshold, sample_rate
            )
            timings["encode"] = time.perf_counter() - start - timings["run"]
            stats.record(timings, audio_seconds)
            return arrays

        core = model.separate_and_detect(audio, timings=timings)  # fills separate / resample / detect
        focal_label = item[focal_column]
        if focal_label not in model.labels:
            raise ValueError(f"focal label {focal_label!r} is not among the detector labels")
        focal_idx = model.labels.index(focal_label)

        derive_start = time.perf_counter()
        denoised = core.denoise([focal_idx], model.threshold)
        focal_detprob, focal_nstems = core.quality([focal_idx], model.threshold)
        stems: tuple[Stem, ...] = ()
        if detail == "stems":
            stems = tuple(
                Stem(
                    audio=stem_audio,
                    preds=DetectorOutput(
                        predictions=stem_preds[np.newaxis], frame_rate=core.frame_rate, class_names=core.labels
                    ),
                )
                for stem_audio, stem_preds in core.stem_pairs()
            )
        combined = core.combined()
        timings["denoise"] = time.perf_counter() - derive_start

        encode_start = time.perf_counter()
        arrays = ItemResult(
            preds=combined,
            denoised=denoised,
            stems=stems,
            latitude=latitude,
            longitude=longitude,
            focal_detprob=focal_detprob,
            focal_nstems=focal_nstems,
        ).to_arrays(preds_threshold, sample_rate)
        timings["encode"] = time.perf_counter() - encode_start
        stats.record(timings, audio_seconds)
        return arrays

    return process


def _load_http_client_config(path: Path) -> HttpClientConfig:
    """Load the http-client config consumed by `detector_client_from_config`.

    Parameters
    ----------
    path : Path
        Path to the http-client config YAML (the ``--httpclient-config`` file).

    Returns
    -------
    HttpClientConfig
        The validated http-client config (``url`` plus optional ``timeout`` /
        ``retries`` / ``auth``).

    Raises
    ------
    click.UsageError
        If the file is not a valid http-client config (e.g. missing ``url`` or
        containing unknown keys).
    """
    try:
        return HttpClientConfig.from_sources(yaml_file=path)
    except ValidationError as exc:
        raise click.UsageError(f"{path}: {exc}") from exc


def _load_run_config(path: Path) -> LsiRunConfig:
    """Load the LSI run config (dataset + output) from a YAML file.

    Parameters
    ----------
    path : Path
        Path to the run config YAML (the ``--run-config`` file).

    Returns
    -------
    LsiRunConfig
        The validated run config.

    Raises
    ------
    click.UsageError
        If the file is not a valid run config (e.g. a missing required key, an
        invalid ``output.detail``, or a leftover model connection key).
    """
    try:
        return LsiRunConfig.from_sources(yaml_file=path)
    except ValidationError as exc:
        raise click.UsageError(f"{path}: {exc}") from exc


def _model_lineage(http_client_config: HttpClientConfig, model: DetectorClient) -> dict:
    """Build the ``model`` block of the run lineage record.

    Combines the http-client connection config (``url`` / ``timeout`` /
    ``retries``, never ``auth``) with the served model's identity: the raw
    ``GET /`` metadata the client captured on connect (`server_config`), which
    carries the model ``type``, weight SHA-256(s), and serving-process git
    commit for a `sed-server` / `sed-denoising-server` server that exposes them.

    Falls back to the previous behaviour — recording only the connection config
    — with a warning when the server did not expose that identity (e.g. an
    older server whose ``GET /`` lacks a ``git_commit`` marker), so a run
    against such a server still succeeds.

    Parameters
    ----------
    http_client_config : HttpClientConfig
        The connection config used to reach the server.
    model : DetectorClient
        The connected client, whose `server_config` holds the server's
        ``GET /`` metadata.

    Returns
    -------
    dict
        ``{"client": <connection config>, "server": <GET / identity>}`` when the
        server exposed its identity, else just the connection config (the
        previous behaviour).
    """
    client_config = http_client_config.model_dump(exclude={"auth"})
    server_config = getattr(model, "server_config", None) or {}
    if "git_commit" in server_config:
        return {"client": client_config, "server": server_config}
    warnings.warn(
        "detector server did not expose model identity in its GET / metadata "
        "(no 'git_commit'); recording connection config only in lineage. Serve with an "
        "updated sed-server / sed-denoising-server to capture the model type, weight SHA-256, and git commit.",
        stacklevel=2,
    )
    return client_config


def _run_lsi(
    run_config_path: Path,
    httpclient_config_path: Path,
    *,
    job_index: int,
    num_jobs: int,
    output_dir: Path | None = None,
) -> None:
    """Build the model and dataset, then run this job's slice of sharded LSI.

    Parameters
    ----------
    run_config_path : Path
        Path to the LSI run config YAML (dataset + output).
    httpclient_config_path : Path
        Path to the http-client config YAML (the running server's ``url``).
    job_index : int
        Zero-based index of this job in the array.
    num_jobs : int
        Total number of parallel jobs.
    output_dir : Path | None
        If given, overrides the run config's ``output.dir``.
    """
    run_config = _load_run_config(run_config_path)
    if output_dir is not None:
        run_config.output.dir = str(output_dir)
    http_client_config = _load_http_client_config(httpclient_config_path)

    dataset_config = run_config.dataset
    output_config = run_config.output
    detail = output_config.detail

    dataset, _ = dataset_from_config(dataset_config.config)
    id_column = dataset_config.id_column or dataset._originals_path_column
    print(f"[lsi] dataset {dataset.info.name}: {len(dataset)} files | id_column={id_column} | detail={detail}")

    model = detector_client_from_config(http_client_config, labels=None)
    process = make_process(
        model,
        dataset_config.focal_column,
        detail,
        output_config.preds_threshold,
        latitude_column=dataset_config.latitude_column,
        longitude_column=dataset_config.longitude_column,
        max_audio_seconds=run_config.max_audio_seconds,
    )

    raw_dir = output_config.dir
    out_dir = raw_dir if not isinstance(anypath(raw_dir), Path) else str(Path(raw_dir).expanduser())
    if job_index == 0:
        # The run is the root of the lineage chain (no parent); downstream stages
        # read this record as their `parent` (see write_lineage).
        write_lineage(
            out_dir,
            "run",
            {
                "run_config": run_config.model_dump(),
                "model": _model_lineage(http_client_config, model),
                "detail": detail,
                "id_column": id_column,
            },
        )

    try:
        run_sharded(
            dataset=dataset,
            id_column=id_column,
            process=process,
            out_dir=out_dir,
            files_per_shard=output_config.files_per_shard,
            job_index=job_index,
            num_jobs=num_jobs,
        )
    finally:
        model.close()


@click.group(invoke_without_command=True)
@click.option(
    "--run-config",
    "run_config_path",
    required=False,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the LSI run config YAML (dataset + output). Required for a run.",
)
@click.option(
    "--httpclient-config",
    "httpclient_config_path",
    required=False,
    type=click.Path(exists=True, path_type=Path),
    help=(
        "Path to the http-client config YAML consumed by detector_client_from_config "
        "(url, timeout, retries, auth). Required for a run."
    ),
)
@click.option("--job-index", type=int, default=0, help="Zero-based index of this job in the array.")
@click.option("--num-jobs", type=int, default=1, help="Total number of parallel jobs.")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Override the run config's output.dir for this run.",
)
@click.pass_context
def cli(
    ctx: click.Context,
    run_config_path: Path | None,
    httpclient_config_path: Path | None,
    job_index: int,
    num_jobs: int,
    output_dir: Path | None,
) -> None:
    """ESP Research — sound event detection large-scale inference (LSI) CLI.

    Run a slice of the dataset::

        sed-lsi --run-config <path> --httpclient-config <path> [--job-index N] [--num-jobs M]

    Raises
    ------
    click.UsageError
        If required options are missing.
    """
    if ctx.invoked_subcommand is not None:
        return

    if run_config_path is None or httpclient_config_path is None:
        raise click.UsageError("--run-config and --httpclient-config are required for a run.")
    _run_lsi(
        run_config_path,
        httpclient_config_path,
        job_index=job_index,
        num_jobs=num_jobs,
        output_dir=output_dir,
    )


@cli.command()
def describe() -> None:
    """Print the LSI run-config schema and the http-client config schema."""
    logger.info("LSI Run Config Schema (--run-config):")
    logger.info(json.dumps(LsiRunConfig.model_json_schema(), indent=2))
    logger.info("\nHTTP Client Config Schema (--httpclient-config):")
    logger.info(json.dumps(HttpClientConfig.model_json_schema(), indent=2))


if __name__ == "__main__":
    cli()
