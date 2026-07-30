"""Convenience launchers for the detector servers.

Wraps ``uvicorn`` so the servers can be started with ``uv run sed-server`` (the
unified detector server) or ``uv run sed-denoising-server`` (the denoising
detector server) instead of spelling out the ``uv run uvicorn ...:app``
incantations. The model to serve is selected through the ``SED_MODEL_CONFIG``
environment variable (see `sound_event_detection.serving.serve_detector` and
`sound_event_detection.serving.serve_denoising_detector`).

Example
-------
Deploy with::

    SED_MODEL_CONFIG=configs/birdcode/models/birdcode_esp_research.yml \\
        uv run sed-server --host localhost --port 8100

    SED_MODEL_CONFIG=configs/birdcode/models/denoising_detector.yml \\
        uv run sed-denoising-server --host localhost --port 8110
"""

import click
import uvicorn

_APP_IMPORT_STRING = "sound_event_detection.serving.serve_detector:app"
_DENOISING_APP_IMPORT_STRING = "sound_event_detection.serving.serve_denoising_detector:app"


@click.command()
@click.option("--host", default="localhost", show_default=True, help="Interface to bind the server to.")
@click.option("--port", default=8100, show_default=True, type=int, help="Port to bind the server to.")
@click.option("--workers", default=1, show_default=True, type=int, help="Number of worker processes.")
@click.option(
    "--reload",
    is_flag=True,
    default=False,
    help="Enable auto-reload on code changes (development only; incompatible with --workers > 1).",
)
@click.option(
    "--log-level",
    default="info",
    show_default=True,
    type=click.Choice(["critical", "error", "warning", "info", "debug", "trace"]),
    help="Uvicorn log level.",
)
def main(host: str, port: int, workers: int, reload: bool, log_level: str) -> None:
    """Launch the unified detector server via uvicorn.

    The model to serve is chosen by the ``SED_MODEL_CONFIG`` environment
    variable, which must point to a model-config YAML.

    Parameters
    ----------
    host : str
        Interface to bind the server to.
    port : int
        Port to bind the server to.
    workers : int
        Number of worker processes to run.
    reload : bool
        Whether to enable auto-reload on code changes.
    log_level : str
        Uvicorn log level.
    """
    uvicorn.run(_APP_IMPORT_STRING, host=host, port=port, workers=workers, reload=reload, log_level=log_level)


@click.command()
@click.option("--host", default="localhost", show_default=True, help="Interface to bind the server to.")
@click.option("--port", default=8110, show_default=True, type=int, help="Port to bind the server to.")
@click.option("--workers", default=1, show_default=True, type=int, help="Number of worker processes.")
@click.option(
    "--reload",
    is_flag=True,
    default=False,
    help="Enable auto-reload on code changes (development only; incompatible with --workers > 1).",
)
@click.option(
    "--log-level",
    default="info",
    show_default=True,
    type=click.Choice(["critical", "error", "warning", "info", "debug", "trace"]),
    help="Uvicorn log level.",
)
def denoising_main(host: str, port: int, workers: int, reload: bool, log_level: str) -> None:
    """Launch the denoising detector server via uvicorn.

    The model config is chosen by the ``SED_MODEL_CONFIG`` environment
    variable, which must point to a ``type: denoising_detector`` model-config
    YAML naming the wrapped detector and separator servers (which must already
    be running when this server starts).

    Parameters
    ----------
    host : str
        Interface to bind the server to.
    port : int
        Port to bind the server to.
    workers : int
        Number of worker processes to run.
    reload : bool
        Whether to enable auto-reload on code changes.
    log_level : str
        Uvicorn log level.
    """
    uvicorn.run(_DENOISING_APP_IMPORT_STRING, host=host, port=port, workers=workers, reload=reload, log_level=log_level)


if __name__ == "__main__":
    main()
