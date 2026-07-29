"""FastAPI server wrapping a frame-level `Detector` (e.g. `FrameDetector`).

Exposes a process that satisfies the `Detector` protocol over HTTP so the
evaluation pipeline can run inference remotely instead of loading the model
in-process. Endpoints:

- ``GET  /``       active configuration (labels, sample_rate, frame_rate, window_duration)
- ``GET  /health`` liveness check (only returns 200 once the model is loaded)
- ``GET  /labels`` ordered list of class labels matching prediction columns
- ``POST /run``    send base64 float32 audio, receive base64 float16 frame predictions
- ``POST /run_as_classifier`` send base64 float32 audio, receive base64 float16 clip predictions

This module is library code: `create_app` builds the app around an injected
detector (or factory), and `serve.py` is the deployable entry point that
selects the detector from a model-config file (``SED_MODEL_CONFIG``).

Environment variables:

- ``SED_DEVICE``  ``"cpu"`` or ``"cuda"`` for `load_frame_detector`
  (default: cuda if available, else cpu).
"""

import base64
import hashlib
import os
import subprocess
import warnings
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import httpx
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from esp_research.checkpointing import load_checkpoint_dir
from esp_research.protocols.detector import Detector
from sound_event_detection.models.frame_detector import FrameDetector


class RunRequest(BaseModel):
    """Request body for the ``/run`` endpoint.

    Attributes
    ----------
    audio : str
        Base64-encoded little-endian float32 PCM for the whole batch, laid out
        as `batch` recordings of `samples` samples each (row-major).
    batch : int
        Number of recordings in the batch.
    samples : int
        Number of samples per recording.
    batch_size : int | None
        Number of windows the detector processes per forward pass. When
        ``None`` (default) the served model's own default applies (e.g. a
        `DenoisingDetector`'s deliberately small
        `DEFAULT_DETECTOR_BATCH_SIZE`), so the wire default never overrides a
        model's memory-motivated choice.
    overlap : float | None
        Fraction of window overlap forwarded to the detector.
    """

    audio: str
    batch: int
    samples: int
    batch_size: int | None = None
    overlap: float | None = None


def _decode_request_audio(body: RunRequest) -> np.ndarray:
    """Decode and validate the base64 audio payload from a request.

    Parameters
    ----------
    body : RunRequest
        Request carrying base64 float32 audio and its `batch`/`samples` layout.

    Returns
    -------
    np.ndarray
        Audio of shape ``(batch, samples)``.

    Raises
    ------
    HTTPException
        422 if `audio` is not valid base64 or the decoded length does not equal
        ``batch * samples * 4`` bytes.
    """
    try:
        raw = base64.b64decode(body.audio)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="audio is not valid base64.") from exc

    expected = body.batch * body.samples * 4
    if body.batch <= 0 or body.samples <= 0 or len(raw) != expected:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Decoded audio length {len(raw)} B != batch*samples*4 ({body.batch}*{body.samples}*4 = {expected} B)."
            ),
        )

    return np.frombuffer(raw, dtype=np.float32).reshape(body.batch, body.samples)


#: Model/backend errors a route maps to HTTP status codes. Served models may
#: wrap remote backends (a `SlidingWindowDetector` over a classifier server, a
#: `DenoisingDetector` over detector + separator servers), so a bad argument or
#: a dead backend surfaces as an `httpx` error, not a `ValueError`; catching
#: this tuple and mapping it via `_http_error` keeps a route from turning a
#: backend failure into a bare 500. `httpx.TransportError` also covers
#: `httpx.TimeoutException`.
MODEL_ERRORS = (ValueError, httpx.HTTPStatusError, httpx.TransportError)


def _http_error(exc: Exception) -> tuple[int, str]:
    """Map a model/backend error to an HTTP status code and detail string.

    Mapping a backend 4xx to 422 keeps a deterministic bad request
    non-retryable for the calling client, while 502/503 keep genuine backend
    failures in the client's retryable set (see `esp_research.adapters`'
    retry policy).

    Parameters
    ----------
    exc : Exception
        An exception from `MODEL_ERRORS` raised by the served model.

    Returns
    -------
    tuple[int, str]
        The HTTP status code and detail: 422 for a `ValueError` or a backend
        4xx (invalid arguments), 502 for a backend 5xx, 503 when the backend
        is unreachable or timed out.
    """
    if isinstance(exc, ValueError):
        return 422, str(exc)
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if 400 <= status < 500:
            return 422, f"Backend server rejected the request ({status}): {exc.response.text}"
        return 502, f"Backend server error ({status})."
    return 503, f"Backend server unreachable: {exc}"


def load_frame_detector(folder: str, device: str | None = None) -> Detector:
    """Load a trained frame detector from a checkpoint directory.

    Parameters
    ----------
    folder : str
        Checkpoint directory containing ``manifest.json`` (passed to
        `load_checkpoint_dir`).
    device : str | None
        Torch device (``"cpu"`` or ``"cuda"``). When ``None``, reads ``SED_DEVICE``
        and falls back to cuda-if-available else cpu.

    Returns
    -------
    Detector
        The loaded model, in eval mode on `device`.
    """
    if device is None:
        device = os.environ.get("SED_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    return load_checkpoint_dir(folder).objects["model"].eval().to(device)


def load_frame_detector_from_hf(repo_id: str, revision: str | None = None, device: str | None = None) -> Detector:
    """Load a trained frame detector from a HuggingFace Hub repository.

    Parameters
    ----------
    repo_id : str
        HuggingFace Hub repo id (e.g. ``"EarthSpeciesProject/sed-birdcode"``), passed
        to `FrameDetector.from_hf_hub`.
    revision : str | None
        Git revision (branch, tag, or commit) to download. When ``None``, uses the
        repository's default branch.
    device : str | None
        Torch device (``"cpu"`` or ``"cuda"``). When ``None``, reads ``SED_DEVICE``
        and falls back to cuda-if-available else cpu.

    Returns
    -------
    Detector
        The loaded model, in eval mode on `device`.
    """
    if device is None:
        device = os.environ.get("SED_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    return FrameDetector.from_hf_hub(repo_id, revision=revision).eval().to(device)


def git_head_commit() -> str | None:
    """Return the serving process's current ``HEAD`` commit, or ``None``.

    Stamped into a served model's identity (`server_config`, surfaced by
    ``GET /``) so a downstream lineage record captures which checkout produced
    the served predictions. Best-effort: any failure (not a git checkout, git
    missing) yields ``None``, never raising — provenance must not stop a server
    from starting.

    Returns
    -------
    str or None
        The 40-char commit hash, or ``None`` if it could not be determined.
    """
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 — provenance is best-effort, never fatal
        return None


def state_dict_sha256(model: torch.nn.Module) -> str | None:
    """Return a SHA-256 digest over a model's weights, or ``None`` on failure.

    Hashes every `state_dict` entry in sorted key order (each key name followed
    by its tensor bytes, moved to CPU and made contiguous), so the digest is a
    stable identity of the loaded weights independent of the on-disk
    serialization format or the device the model sits on. Two servers that
    loaded byte-identical weights hash to the same value even from different
    checkpoint paths. Best-effort: any failure (e.g. an exotic tensor dtype
    numpy cannot view) yields ``None`` with a warning, so serving never fails
    on provenance.

    Parameters
    ----------
    model : torch.nn.Module
        The loaded model whose `state_dict` is hashed.

    Returns
    -------
    str or None
        The 64-char hex digest, or ``None`` if it could not be computed.
    """
    try:
        digest = hashlib.sha256()
        state = model.state_dict()
        for key in sorted(state):
            digest.update(key.encode("utf-8"))
            digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    except Exception as exc:  # noqa: BLE001 — provenance is best-effort, never fatal
        warnings.warn(f"could not compute model weights SHA-256: {exc}", stacklevel=2)
        return None


def create_app(
    model: Detector | None = None,
    model_factory: Callable[[], Detector] | None = None,
    describe_extras: Callable[[Detector], dict] | None = None,
) -> FastAPI:
    """Build the FastAPI app that serves a frame-level detector.

    Construction is cheap: it does not load weights. The model is loaded (or
    the injected `model` adopted) when the app's lifespan starts. Because
    FastAPI serves no request until lifespan startup completes, a 200 from
    ``/health`` reliably signals that the model is loaded.

    Parameters
    ----------
    model : Detector | None
        Pre-built detector to serve. Injecting a model is primarily for
        testing; the production path (`serve.py`) uses `model_factory`. The
        caller keeps ownership: an injected model is not closed on shutdown.
    model_factory : Callable[[], Detector] | None
        Factory called once at startup to build the detector. Lets servers that
        wrap a remote backend (e.g. a sliding-window detector over a classifier
        server) defer construction until lifespan, so the backend need not be up
        at import time. Ignored when `model` is provided. The app owns the
        factory-built model and calls its `close` (when present) on shutdown.
    describe_extras : Callable[[Detector], dict] | None
        Optional hook merging extra keys into the ``GET /`` payload, called
        with the loaded model. Lets a server expose model-specific identity
        (e.g. the denoising server's composed `server_config`) without
        re-registering the route. When ``None`` (default) the payload is the
        standard four fields, unchanged.

    Returns
    -------
    FastAPI
        The configured application.

    Raises
    ------
    ValueError
        If neither `model` nor `model_factory` is provided.
    """
    if model is None and model_factory is None:
        raise ValueError("create_app requires either a model or a model_factory.")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if model is not None:
            app.state.model = model
        else:
            app.state.model = model_factory()

        loaded = app.state.model
        print(
            f"Frame detector server ready: {len(loaded.labels)} classes, "
            f"sample_rate={loaded.sample_rate}, frame_rate={loaded.frame_rate}.",
            flush=True,
        )
        try:
            yield
        finally:
            # The app owns factory-built models, so it releases their
            # resources (e.g. a DenoisingDetector's wrapped HTTP clients) on
            # shutdown; an injected `model` stays the caller's to close.
            if model is None:
                close = getattr(loaded, "close", None)
                if close is not None:
                    close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    def describe() -> dict:
        """Return the active configuration (used by `HttpClient.describe()`).

        Returns
        -------
        dict
            ``{"labels", "sample_rate", "frame_rate", "window_duration"}``,
            merged with the `describe_extras` keys when the hook is set.
        """
        loaded = app.state.model
        return {
            "labels": list(loaded.labels),
            "sample_rate": loaded.sample_rate,
            "frame_rate": loaded.frame_rate,
            "window_duration": loaded.window_duration,
            **(describe_extras(loaded) if describe_extras is not None else {}),
        }

    @app.get("/health")
    def health() -> dict:
        """Return a liveness marker; served only after the model is loaded.

        Returns
        -------
        dict
            ``{"status": "ok"}``.
        """
        return {"status": "ok"}

    @app.get("/labels")
    def labels() -> list[str]:
        """Return the ordered class labels matching prediction columns.

        Returns
        -------
        list[str]
            Class labels in prediction-column order.
        """
        return list(app.state.model.labels)

    @app.post("/run")
    def run(body: RunRequest) -> dict:
        """Run frame-level inference on a batch of equal-length recordings.

        Parameters
        ----------
        body : RunRequest
            Base64 float32 audio plus its `batch`/`samples` layout and the
            forwarded `batch_size`/`overlap` arguments.

        Returns
        -------
        dict
            ``{"predictions": <base64 float16>, "shape": [batch, time, classes],
            "dtype": "float16", "frame_rate": float}`` where predictions are
            probabilities in [0, 1].

        Raises
        ------
        HTTPException
            422 if `audio` is not valid base64, if the decoded length does not
            equal ``batch * samples * 4`` bytes, or if the detector rejects
            the arguments (e.g. an out-of-range `overlap`); 502/503 if a
            wrapped backend server fails or is unreachable (see `_http_error`).
        """
        audio = _decode_request_audio(body)

        kwargs: dict = {"overlap": body.overlap}
        if body.batch_size is not None:
            kwargs["batch_size"] = body.batch_size
        try:
            output = app.state.model.run(audio, **kwargs)
        except MODEL_ERRORS as exc:
            status, detail = _http_error(exc)
            raise HTTPException(status_code=status, detail=detail) from exc

        # Send predictions as float16 to halve the payload; they are
        # probabilities in [0, 1] that get thresholded downstream, so the
        # precision loss is immaterial. The dtype is reported so the client
        # decodes with the matching type.
        preds = np.ascontiguousarray(output.predictions, dtype=np.float16)
        return {
            "predictions": base64.b64encode(preds.tobytes()).decode("ascii"),
            "shape": list(preds.shape),
            "dtype": "float16",
            "frame_rate": output.frame_rate,
        }

    @app.post("/run_as_classifier")
    def run_as_classifier(body: RunRequest) -> dict:
        """Run clip-level classification on a batch of equal-length recordings.

        Parameters
        ----------
        body : RunRequest
            Base64 float32 audio plus its `batch`/`samples` layout and the
            forwarded `batch_size`/`overlap` arguments.

        Returns
        -------
        dict
            ``{"predictions": <base64 float16>, "shape": [batch, classes],
            "dtype": "float16"}`` where predictions are clip-level
            probabilities in [0, 1].

        Raises
        ------
        HTTPException
            422 if `audio` is not valid base64, if the decoded length does not
            equal ``batch * samples * 4`` bytes, or if the detector rejects
            the arguments (e.g. an out-of-range `overlap`); 502/503 if a
            wrapped backend server fails or is unreachable (see `_http_error`).
        """
        audio = _decode_request_audio(body)

        kwargs: dict = {"overlap": body.overlap}
        if body.batch_size is not None:
            kwargs["batch_size"] = body.batch_size
        try:
            output = app.state.model.run_as_classifier(audio, **kwargs)
        except MODEL_ERRORS as exc:
            status, detail = _http_error(exc)
            raise HTTPException(status_code=status, detail=detail) from exc

        # Clip predictions are 2-D (batch, classes); float16 halves the payload.
        preds = np.ascontiguousarray(output.predictions, dtype=np.float16)
        return {
            "predictions": base64.b64encode(preds.tobytes()).decode("ascii"),
            "shape": list(preds.shape),
            "dtype": "float16",
        }

    return app
