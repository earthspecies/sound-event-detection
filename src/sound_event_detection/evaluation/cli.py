"""CLI for running and inspecting the SED evaluation.

Entry point: ``uv run sed-eval``.

Commands
--------
- ``sed-eval --eval-config <path> --httpclient-config <path> [--checkpoint-dir <path>]``
    start a new evaluation.
- ``sed-eval --resume <checkpoint-dir>``
    resume an evaluation from its checkpoint directory.
- ``sed-eval describe``
    print the expected model-output schema and the eval config schema.

Splits between an eval config (what to evaluate) and an http-client config
(where the served model lives and how to reach it). The ``--httpclient-config``
file holds the pure http-client config consumed by `detector_client_from_config`
(``url``, optional ``timeout`` / ``retries`` / ``auth``); the kind of client is
auto-detected from the server, so the same file shape reaches a plain
detector server (``sed-server``) or a denoising detector server
(``sed-denoising-server``).
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

import click
from pydantic import ValidationError
from rich.table import Table

from esp_research.adapters.client_config import HttpClientConfig
from esp_research.evals.base import validate_has_expected_model_output
from esp_research.logging import logger
from esp_research.protocols.eval import EvalResult
from sound_event_detection.adapters.dispatch import (
    DetectorClient,
    detector_client_from_config,
)

from .config import SedEvalConfig
from .evaluator import SedEvaluator

_EVAL_NAME = "sed"
_EVAL_CONFIG_FILENAME = "eval_config.yml"
_HTTPCLIENT_CONFIG_FILENAME = "httpclient_config.yml"
_SERVER_CONFIG_FILENAME = "server_config.json"
_PROGRESS_FILENAME = "progress.jsonl"


@click.group(invoke_without_command=True)
@click.option(
    "--eval-config",
    "eval_config_path",
    required=False,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the eval configuration YAML file. Required for new runs.",
)
@click.option(
    "--httpclient-config",
    "httpclient_config_path",
    required=False,
    type=click.Path(exists=True, path_type=Path),
    help=(
        "Path to the http-client config YAML consumed by detector_client_from_config "
        "(url, timeout, retries, auth). Required for new runs."
    ),
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for resumable checkpoints. Auto-generated under checkpoints/sed/ if not specified.",
)
@click.option(
    "--resume",
    "resume_dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Resume from an existing checkpoint directory. Configuration is loaded from the checkpoint.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Override the eval config's output_dir for this run (e.g. a per-model results folder).",
)
@click.pass_context
def cli(
    ctx: click.Context,
    eval_config_path: Path | None,
    httpclient_config_path: Path | None,
    checkpoint_dir: Path | None,
    resume_dir: Path | None,
    output_dir: Path | None,
) -> None:
    """ESP Research — sound event detection evaluation CLI.

    Start a new evaluation::

        sed-eval --eval-config <path> --httpclient-config <path>

    Resume an existing evaluation::

        sed-eval --resume <checkpoint-dir>

    ``--output-dir`` may be combined with either mode.

    Raises
    ------
    click.UsageError
        If required options are missing or conflicting options are provided.
    """
    if ctx.invoked_subcommand is not None:
        return

    if resume_dir is not None:
        if eval_config_path or httpclient_config_path or checkpoint_dir:
            raise click.UsageError(
                "--resume cannot be combined with --eval-config, --httpclient-config, or --checkpoint-dir."
            )
        _resume_evaluation(resume_dir, output_dir=output_dir)
    else:
        if eval_config_path is None or httpclient_config_path is None:
            raise click.UsageError(
                "--eval-config and --httpclient-config are required for new runs. "
                "To resume, use --resume <checkpoint-dir>."
            )
        _run_evaluation(eval_config_path, httpclient_config_path, checkpoint_dir, output_dir=output_dir)


@cli.command()
def describe() -> None:
    """Print the expected model-output schema and the eval config schema."""
    logger.info(_EVAL_NAME)
    logger.info("\nExpected Model Output Schema (the server's /run response):")
    logger.info(json.dumps(SedEvaluator.expected_model_output.model_json_schema(), indent=2))
    logger.info("\nSED Eval Config Schema:")
    logger.info(json.dumps(SedEvalConfig.model_json_schema(), indent=2))


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


def _run_evaluation(
    eval_config_path: Path,
    httpclient_config_path: Path,
    checkpoint_dir: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """Start a new SED evaluation run.

    Parameters
    ----------
    eval_config_path : Path
        Path to the eval configuration YAML file.
    httpclient_config_path : Path
        Path to the http-client config YAML file.
    checkpoint_dir : Path | None
        Directory for checkpoints. Falls back to the config's ``checkpoint_dir``,
        else auto-generated under ``checkpoints/sed/``.
    output_dir : Path | None
        If given, overrides the eval config's ``output_dir``.

    Raises
    ------
    click.UsageError
        If ``checkpoint_dir`` already contains data from a previous run, or the
        model config is malformed.
    SystemExit
        If evaluator validation fails.
    """
    try:
        validate_has_expected_model_output(SedEvaluator)
    except TypeError as exc:
        logger.error(f"Validation error: {exc}")
        raise SystemExit(1) from exc

    eval_config = SedEvalConfig.from_sources(yaml_file=eval_config_path)
    if output_dir is not None:
        eval_config.output_dir = str(output_dir)
    http_client_config = _load_http_client_config(httpclient_config_path)

    if checkpoint_dir is None:
        if eval_config.checkpoint_dir is not None:
            checkpoint_dir = Path(eval_config.checkpoint_dir)
        else:
            checkpoint_dir = Path("checkpoints/sed") / datetime.now().strftime("%Y%m%d_%H%M%S")
    elif (checkpoint_dir / _PROGRESS_FILENAME).exists():
        raise click.UsageError(
            f"{checkpoint_dir} already contains checkpoint data from a previous run.\n"
            "To resume it, use:  sed-eval --resume " + str(checkpoint_dir) + "\n"
            "To start fresh, omit --checkpoint-dir and a new directory will be created automatically."
        )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    eval_config.checkpoint_dir = str(checkpoint_dir)
    click.echo(f"Checkpoint directory: {checkpoint_dir}")

    # Save configs so the run can be resumed without re-supplying them.
    shutil.copy2(eval_config_path, checkpoint_dir / _EVAL_CONFIG_FILENAME)
    shutil.copy2(httpclient_config_path, checkpoint_dir / _HTTPCLIENT_CONFIG_FILENAME)

    client = detector_client_from_config(http_client_config)
    _validate_model_config(client.server_config, checkpoint_dir)
    _execute_evaluation(eval_config, client)


def _resume_evaluation(checkpoint_dir: Path, output_dir: Path | None = None) -> None:
    """Resume a SED evaluation from an existing checkpoint directory.

    The model config is loaded from the checkpoint; if a server moved to a new
    host:port since the run started, edit the ``url`` in the checkpoint's
    ``httpclient_config.yml``.

    Parameters
    ----------
    checkpoint_dir : Path
        Path to the checkpoint directory created by a previous run.
    output_dir : Path | None
        If given, overrides the saved eval config's ``output_dir``.

    Raises
    ------
    click.UsageError
        If the checkpoint directory is missing required config files or the
        model config is malformed.
    """
    eval_cfg_path = checkpoint_dir / _EVAL_CONFIG_FILENAME
    httpclient_cfg_path = checkpoint_dir / _HTTPCLIENT_CONFIG_FILENAME

    if not eval_cfg_path.exists():
        raise click.UsageError(f"No {_EVAL_CONFIG_FILENAME} found in {checkpoint_dir}.")
    if not httpclient_cfg_path.exists():
        raise click.UsageError(f"No {_HTTPCLIENT_CONFIG_FILENAME} found in {checkpoint_dir}.")

    eval_config = SedEvalConfig.from_sources(yaml_file=eval_cfg_path)
    eval_config.checkpoint_dir = str(checkpoint_dir)
    if output_dir is not None:
        eval_config.output_dir = str(output_dir)
    http_client_config = _load_http_client_config(httpclient_cfg_path)

    click.echo(f"Resuming from checkpoint directory: {checkpoint_dir}")
    client = detector_client_from_config(http_client_config)
    _validate_model_config(client.server_config, checkpoint_dir)
    _execute_evaluation(eval_config, client)


def _validate_model_config(server_config: dict, checkpoint_dir: Path) -> None:
    """Check the client's live model identity against the checkpoint.

    On a fresh run the live identity (`DetectorClient.server_config`) is saved.
    On a resume it is compared with the saved one so results from a changed
    model are never mixed.

    Parameters
    ----------
    server_config : dict
        The client's live model identity (for a served detector, the ``GET /``
        payload).
    checkpoint_dir : Path
        Directory where ``server_config.json`` is read/written.

    Raises
    ------
    click.UsageError
        If the live config differs from the saved one on a resume.
    """
    config_path = checkpoint_dir / _SERVER_CONFIG_FILENAME
    if not config_path.exists():
        config_path.write_text(json.dumps(server_config, indent=2))
        logger.info("Server config saved to %s", config_path)
    else:
        saved = json.loads(config_path.read_text())
        if saved != server_config:
            raise click.UsageError(
                "Server config has changed since this checkpoint was created. "
                "Resuming would mix results from different models. "
                "To start fresh, use a new --checkpoint-dir."
            )
        logger.info("Server config matches checkpoint — resuming.")


def _execute_evaluation(eval_config: SedEvalConfig, client: DetectorClient) -> None:
    """Build the evaluator and run the evaluation.

    Parameters
    ----------
    eval_config : SedEvalConfig
        Fully resolved evaluation configuration.
    client : DetectorClient
        Connected detector client; closed when the run ends.
    """
    evaluator = SedEvaluator.from_config(eval_config)
    logger.info(f"Running {_EVAL_NAME}")
    try:
        result = evaluator.evaluate(client)
    finally:
        client.close()
    _print_result_summary(_EVAL_NAME, result)


def _print_result_summary(eval_name: str, result: EvalResult) -> None:
    """Print evaluation results as a rich table.

    Parameters
    ----------
    eval_name : str
        Name of the evaluator that produced the result.
    result : EvalResult
        The evaluation result object.
    """
    logger.info("Results")
    logger.info(f"Eval: {eval_name}")
    logger.info(f"Aggregate score: {result.value:.4f}")

    metric_outputs = getattr(result, "metric_outputs", None)
    if metric_outputs:
        table = Table(title="Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")
        for m in metric_outputs:
            table.add_row(m.name, f"{m.value:.4f}")
        logger.info(table)
