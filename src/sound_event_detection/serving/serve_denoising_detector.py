"""Denoising detector server driven by a model-config file.

Serves a `DenoisingDetector` — a model wrapping a detector client and a
source-separator client — behind the standard detector contract (``GET /``,
``POST /run``, ``POST /run_as_classifier``) plus the denoising-specific
``POST /separate_and_detect`` used by large-scale inference. The model config
is a YAML pointed to by the ``SED_MODEL_CONFIG`` environment variable::

    type: denoising_detector
    detector: {url: http://localhost:8100, timeout: 300}   # detector server
    separator: {url: http://localhost:8200, timeout: 300}  # BirdMixIt server
    threshold: 0.5
    resampling_method: torchaudio_kaiser_fast

Deploy with::

    SED_MODEL_CONFIG=configs/birdcode/models/denoising_detector.yml \\
        uv run sed.denoising_app --host localhost --port 8110

Both backend servers must be up when this server starts: the wrapped clients
fetch their metadata at construction (which happens at lifespan startup, by
convention detector on 8100, BirdMixIt on 8200, this server on 8110).

``GET /`` merges the model's composed `server_config` (``type:
denoising_detector``, nested detector identity, separator identity — including
its ``weights_sha256`` when the separator server exposes one, else ``None`` —
``threshold``, ``resampling_method``, and this serving process's
``git_commit``) into the standard payload; that ``type`` key
(`DENOISING_DETECTOR_TYPE`) is what `detector_client_from_config` auto-detects
on to hand back a `ServedDenoisingDetectorClient`.

``/separate_and_detect`` speaks raw binary, like the BirdMixIt server's
``/separate_file_binary``: the request body is raw float32 PCM
(``application/octet-stream``, layout in query parameters) and the response
body is the raw stems + predictions with the layout in ``x-``-prefixed
headers — the stem payload is ``n_stems`` times the whole recording, so
base64-in-JSON would inflate it by a third and force both ends through a
giant JSON document.
"""

import json
import os
from pathlib import Path

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool

from sound_event_detection.denoising.denoising_detector import (
    DEFAULT_DETECTOR_BATCH_SIZE,
    DENOISING_DETECTOR_TYPE,
    DenoisingDetector,
    DenoisingDetectorConfig,
)
from sound_event_detection.serving.server import MODEL_ERRORS, _http_error, create_app, git_head_commit


def _decode_mono_audio(raw: bytes, samples: int) -> np.ndarray:
    """Validate a raw float32 mono audio request body.

    Parameters
    ----------
    raw : bytes
        Raw little-endian float32 PCM request body.
    samples : int
        Expected number of samples in the recording.

    Returns
    -------
    np.ndarray
        Audio of shape ``(samples,)``.

    Raises
    ------
    HTTPException
        422 if `samples` is not positive or the body length does not equal
        ``samples * 4`` bytes.
    """
    expected = samples * 4
    if samples <= 0 or len(raw) != expected:
        raise HTTPException(
            status_code=422,
            detail=f"Audio body length {len(raw)} B != samples*4 ({samples}*4 = {expected} B).",
        )

    return np.frombuffer(raw, dtype=np.float32)


def add_denoising_routes(app: FastAPI) -> FastAPI:
    """Register the denoising-specific routes on a detector-serving app.

    Adds ``POST /separate_and_detect`` next to the standard contract from
    `create_app`. Factored out of the module-level app assembly so tests can
    build an app around a fake model without touching ``SED_MODEL_CONFIG``.

    Parameters
    ----------
    app : FastAPI
        App built by `create_app` whose ``app.state.model`` is (or will be at
        lifespan startup) a `DenoisingDetector`.

    Returns
    -------
    FastAPI
        The same app, with the denoising routes registered.
    """

    @app.post("/separate_and_detect")
    async def separate_and_detect(
        request: Request,
        samples: int,
        batch_size: int = DEFAULT_DETECTOR_BATCH_SIZE,
        overlap: float | None = None,
    ) -> Response:
        """Separate one recording into stems and detect over them.

        The request body is the raw float32 mono recording
        (``application/octet-stream``); `samples`, `batch_size`, and
        `overlap` arrive as query parameters. The model runs in the thread
        pool so the event loop (and ``/health``) stays responsive.

        Parameters
        ----------
        request : Request
            Request whose body is raw little-endian float32 PCM of
            `samples` samples at the model's (separator's) sample rate.
        samples : int
            Number of samples in the recording.
        batch_size : int
            Windows per detector forward pass, forwarded to the wrapped
            detector. Defaults to `DEFAULT_DETECTOR_BATCH_SIZE`.
        overlap : float | None
            Fraction of window overlap forwarded to the wrapped detector.

        Returns
        -------
        Response
            ``application/octet-stream`` body of the raw float32 stems
            followed by the raw float16 per-stem predictions, with the layout
            in headers: ``x-stems-shape`` (``n_stems,samples``),
            ``x-preds-shape`` (``n_stems,frames,classes``), ``x-preds-dtype``
            (``float16``), ``x-frame-rate``, ``x-sample-rate``, and
            ``x-timings`` (JSON ``{"separate", "resample", "detect"}``
            seconds). Stems stay float32 — they are audio, reused for the
            denoised waveform — while the predictions are float16 like
            ``/run``'s (probabilities in [0, 1]).

        Raises
        ------
        HTTPException
            422 if the body length does not equal ``samples * 4`` bytes or
            the model rejects the arguments; 502/503 if a wrapped backend
            server fails or is unreachable (see `_http_error`).
        """
        audio = _decode_mono_audio(await request.body(), samples)

        timings: dict[str, float] = {}
        try:
            core = await run_in_threadpool(
                app.state.model.separate_and_detect, audio, batch_size=batch_size, timings=timings, overlap=overlap
            )
        except MODEL_ERRORS as exc:
            status, detail = _http_error(exc)
            raise HTTPException(status_code=status, detail=detail) from exc

        stems = np.ascontiguousarray(core.stems, dtype=np.float32)
        preds = np.ascontiguousarray(core.stem_preds, dtype=np.float16)
        return Response(
            content=stems.tobytes() + preds.tobytes(),
            media_type="application/octet-stream",
            headers={
                "x-stems-shape": ",".join(str(dim) for dim in stems.shape),
                "x-preds-shape": ",".join(str(dim) for dim in preds.shape),
                "x-preds-dtype": "float16",
                "x-frame-rate": str(core.frame_rate),
                "x-sample-rate": str(core.sample_rate),
                "x-timings": json.dumps(timings),
            },
        )

    return app


def _build_model() -> DenoisingDetector:
    """Build the denoising detector to serve from the ``SED_MODEL_CONFIG`` file.

    Returns
    -------
    DenoisingDetector
        The model, with both wrapped clients connected to their servers.

    Raises
    ------
    RuntimeError
        If ``SED_MODEL_CONFIG`` is not set.
    ValueError
        If the config file is not a YAML mapping, its ``type`` is not
        `DENOISING_DETECTOR_TYPE`, or it lacks a ``detector`` or
        ``separator`` block.
    """
    config_path = os.environ.get("SED_MODEL_CONFIG")
    if not config_path:
        raise RuntimeError("SED_MODEL_CONFIG environment variable is not set.")

    with open(Path(config_path).expanduser(), "r") as f:
        model_config = yaml.safe_load(f)

    if not isinstance(model_config, dict):
        raise ValueError(f"Model config {config_path} must be a YAML mapping, got {type(model_config).__name__}.")

    model_type = model_config.get("type")
    if model_type != DENOISING_DETECTOR_TYPE:
        raise ValueError(f"Unknown model type {model_type!r} in {config_path}. Expected {DENOISING_DETECTOR_TYPE!r}.")

    missing = [key for key in ("detector", "separator") if key not in model_config]
    if missing:
        raise ValueError(f"Model config {config_path} is missing required key(s): {missing}.")

    config = DenoisingDetectorConfig(
        detector=model_config["detector"],
        separator=model_config["separator"],
        **{key: model_config[key] for key in ("threshold", "resampling_method") if key in model_config},
    )
    detector = DenoisingDetector.from_config(config)
    # Composed detector/separator identity is built in __init__; add this
    # serving process's commit (the frame leg's own commit rides along inside
    # detector.server_config["detector"], captured from that backend's GET /).
    detector.server_config["git_commit"] = git_head_commit()
    print(
        f"Serving denoising detector: {len(detector.labels)} classes, "
        f"sample_rate={detector.sample_rate}, frame_rate={detector.frame_rate}, "
        f"threshold={detector.threshold}, resampling_method={detector.resampling_method}.",
        flush=True,
    )
    return detector


app = add_denoising_routes(create_app(model_factory=_build_model, describe_extras=lambda model: model.server_config))
