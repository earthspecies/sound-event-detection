"""FastAPI server wrapping the BEATs-SL-All multi-label classifier.

Loads the ``esp_aves2_sl_beats_all`` model via avex (backbone + trained
classification head) and exposes it over HTTP so the evaluation pipeline can run
inference remotely instead of loading the model in-process. The server emits raw
classifier logits and the raw classifier label list; converting those labels to
GBIF scientific names and applying an activation stays the client's job (see
`create_beats_sl_all_detector`).

Endpoints (mirroring `audioprotopnet-server/server.py`):

- ``GET  /``       active configuration (labels, sample_rate)
- ``GET  /health`` liveness check (only returns 200 once the model is loaded)
- ``GET  /labels`` ordered list of class labels matching logit indices
- ``POST /logits`` send ``{"audio": "<base64 float32 PCM>", "num_windows": N}``,
  receive ``{"logits": [[float, ...]]}`` of shape ``[N, n_classes]``

Deploy with::

    uvicorn sound_event_detection.serving.sl_beats_all_server:app \\
        --host localhost --port 8200

Environment variables:

- ``SED_DEVICE``  ``"cpu"`` or ``"cuda"`` (default: cuda if available, else cpu).
"""

import base64
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_NAME = "esp_aves2_sl_beats_all"
SAMPLE_RATE = 16000


class LogitsRequest(BaseModel):
    """Request body for the ``/logits`` endpoint.

    Attributes
    ----------
    audio : str
        Base64-encoded little-endian float32 PCM for the whole batch, laid out
        as `num_windows` clips of equal length (row-major).
    num_windows : int
        Number of clips in the batch. The per-clip sample count is inferred by
        reshaping the decoded buffer to ``(num_windows, -1)`` — the model
        accepts variable-length windows, so it is not fixed by the server.
    """

    audio: str
    num_windows: int


def create_app(
    model: Callable[[torch.Tensor], torch.Tensor] | None = None,
    labels: list[str] | None = None,
) -> FastAPI:
    """Build the FastAPI app that serves the BEATs-SL-All classifier.

    Construction is cheap: it does not read environment variables or load
    weights. The model is loaded (or the injected `model`/`labels` adopted) when
    the app's lifespan starts. Because FastAPI serves no request until lifespan
    startup completes, a 200 from ``/health`` reliably signals that the model is
    loaded.

    Parameters
    ----------
    model : Callable[[torch.Tensor], torch.Tensor] | None
        Pre-built classifier mapping audio ``(B, samples)`` to logits ``(B, K)``.
        When ``None`` (the production path), the model is loaded at startup via
        ``avex.load_model(MODEL_NAME)``. Injecting a model is primarily for
        testing and must be paired with `labels`.
    labels : list[str] | None
        Ordered class labels matching the model's logit indices. Required when
        `model` is injected; ignored on the production path (labels come from
        ``avex.load_label_mapping(MODEL_NAME)``).

    Returns
    -------
    FastAPI
        The configured application.

    Raises
    ------
    ValueError
        If `model` is injected without `labels`.
    """
    if model is not None and labels is None:
        raise ValueError("labels must be provided when injecting a model.")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        device = torch.device(os.environ.get("SED_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

        if model is not None:
            app.state.model = model
            app.state.labels = labels
        else:
            from avex import load_label_mapping, load_model

            print(f"Loading {MODEL_NAME} model...", flush=True)
            # Pass the device as a string: avex forwards it unchanged to
            # safetensors.load_file(), whose Rust backend rejects torch.device
            # objects and only accepts strings like "cuda"/"cpu".
            loaded = load_model(MODEL_NAME, device=str(device))
            loaded.eval()

            mapping = load_label_mapping(MODEL_NAME)
            index_to_label = {v: k for k, v in mapping["label_to_index"].items()}
            app.state.model = loaded
            app.state.labels = [index_to_label[i] for i in range(len(index_to_label))]

        app.state.device = device
        print(
            f"BEATs-SL-All server ready: {len(app.state.labels)} classes on {device}.",
            flush=True,
        )
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    def describe() -> dict:
        """Return the active configuration (used by `HttpClient.describe()`).

        Returns
        -------
        dict
            ``{"labels", "sample_rate"}``.
        """
        return {"labels": list(app.state.labels), "sample_rate": SAMPLE_RATE}

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
    def labels_endpoint() -> list[str]:
        """Return the ordered class labels matching logit indices.

        Returns
        -------
        list[str]
            Class labels in logit-column order.
        """
        return list(app.state.labels)

    @app.post("/logits")
    def logits(body: LogitsRequest) -> dict:
        """Run the classifier on a batch of equal-length mono clips.

        Parameters
        ----------
        body : LogitsRequest
            Base64 float32 audio plus the batch size `num_windows`. The decoded
            buffer is reshaped to ``(num_windows, samples)`` row-major.

        Returns
        -------
        dict
            ``{"logits": [[float, ...]]}`` of shape ``[num_windows, n_classes]``
            (raw, pre-activation logits).

        Raises
        ------
        HTTPException
            422 if `audio` is not valid base64, if `num_windows` is not positive,
            or if the decoded length is not a multiple of ``num_windows * 4`` bytes.
        """
        try:
            raw = base64.b64decode(body.audio)
        except Exception as exc:
            raise HTTPException(status_code=422, detail="audio is not valid base64.") from exc

        if body.num_windows <= 0 or len(raw) == 0 or len(raw) % (body.num_windows * 4) != 0:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Decoded audio length {len(raw)} B is not a positive multiple of "
                    f"num_windows*4 ({body.num_windows}*4 = {body.num_windows * 4} B)."
                ),
            )

        audio = np.frombuffer(raw, dtype=np.float32).reshape(body.num_windows, -1)
        tensor = torch.from_numpy(audio.copy()).to(app.state.device)

        with torch.no_grad():
            out = app.state.model(tensor)

        return {"logits": out.cpu().tolist()}

    return app


app = create_app()
