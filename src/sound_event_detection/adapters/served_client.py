"""Client-side adapter for talking to a served detector over HTTP.

The SED evaluation talks to a model that runs behind a FastAPI server (a
trained `FrameDetector`, or a baseline wrapped in a `SlidingWindowDetector` —
see `sound_event_detection.serving`). The transport is an
`esp_research.adapters.HttpClient` per route; everything model-specific
(weights, config, checkpointing, sliding-window logic) lives server-side.

`ServedDetectorClient` is exactly that — *a client*. It exposes the runtime
surface the scoring helpers consume (`run`, `run_as_classifier`, `labels`,
`sample_rate`, `frame_rate`, `window_duration`) but it is
**not** a `Detector`: it has no `config_class`, no `from_checkpoint_dir`, and no
`from_config` in the `Detector` sense. The `Detector` protocol describes what a
*model* delivers and how it is configured/checkpointed; that contract belongs
to the model on the server, not to the network client used to reach it.

The wire contract (see `sound_event_detection.serving.server`) is::

    GET  /                    -> {labels, sample_rate, frame_rate, window_duration}
    POST /run                 -> SedRunResponse
    POST /run_as_classifier   -> SedRunResponse (predictions pooled to clip level)
"""

import base64
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field

from esp_research.adapters import HttpClient
from esp_research.adapters.client_config import HttpClientConfig
from esp_research.protocols.classifier import MultiLabelClassifierOutput
from esp_research.protocols.detector import DetectorOutput

_RUN_ROUTE = "run"
_CLASSIFIER_ROUTE = "run_as_classifier"


class SedRunResponse(BaseModel):
    """Wire schema for the server's ``/run`` and ``/run_as_classifier`` responses.

    This doubles as the evaluator's ``expected_model_output``: it documents the
    contract any server must satisfy to be evaluated, and is surfaced by the
    ``sed-eval describe`` command.

    Attributes
    ----------
    predictions : str
        Base64-encoded little-endian prediction array (probabilities in [0, 1]).
        For ``/run`` it decodes to shape ``(batch, time, classes)``; for
        ``/run_as_classifier`` to ``(batch, classes)``.
    shape : list[int]
        Shape of the decoded prediction array.
    dtype : str
        Numpy dtype of the wire payload (e.g. ``"float16"``). The server may
        downcast to halve payload size; the client restores ``float32``.
    frame_rate : float
        Output frame rate in Hz. Present for ``/run``; ``run_as_classifier``
        responses may omit it (it carries no clip-level meaning).
    """

    predictions: str
    shape: list[int]
    dtype: str = "float32"
    frame_rate: float | None = Field(default=None, gt=0)


def _decode_predictions(response: dict) -> np.ndarray:
    """Decode the base64 prediction payload shared by both server routes.

    Parameters
    ----------
    response : dict
        Parsed JSON response with ``predictions`` (base64), ``shape``, and an
        optional ``dtype`` (defaults to ``"float32"``).

    Returns
    -------
    np.ndarray
        Decoded predictions cast to ``float32``, reshaped to ``response["shape"]``.
    """
    wire_dtype = np.dtype(response.get("dtype", "float32"))
    return (
        np.frombuffer(base64.b64decode(response["predictions"]), dtype=wire_dtype)
        .reshape(response["shape"])
        .astype(np.float32)
    )


def decode_run_response(response: dict, class_names: list[str]) -> DetectorOutput:
    """Build a `DetectorOutput` from a ``/run`` response.

    Parameters
    ----------
    response : dict
        Parsed JSON response from the server's ``/run`` endpoint.
    class_names : list[str]
        Output class labels, in prediction-column order (from ``GET /``).

    Returns
    -------
    DetectorOutput
        Frame-level predictions of shape ``(batch, time, classes)``.
    """
    return DetectorOutput(
        predictions=_decode_predictions(response),
        frame_rate=response["frame_rate"],
        class_names=class_names,
    )


def decode_classifier_response(response: dict, class_names: list[str]) -> MultiLabelClassifierOutput:
    """Build a `MultiLabelClassifierOutput` from a ``/run_as_classifier`` response.

    Parameters
    ----------
    response : dict
        Parsed JSON response from the server's ``/run_as_classifier`` endpoint.
    class_names : list[str]
        Output class labels, in prediction-column order (from ``GET /``).

    Returns
    -------
    MultiLabelClassifierOutput
        Clip-level predictions of shape ``(batch, classes)``.
    """
    return MultiLabelClassifierOutput(predictions=_decode_predictions(response), class_names=class_names)


def _post_audio(client: HttpClient, audio: np.ndarray, batch_size: int | None, overlap: float | None) -> dict:
    """POST a batch of audio to a route-bound client and return the JSON response.

    Parameters
    ----------
    client : HttpClient
        Route-bound client (``/run`` or ``/run_as_classifier``).
    audio : np.ndarray
        Batched waveform of shape ``(batch, samples)``.
    batch_size : int | None
        Number of windows the server processes per forward pass. ``None`` is
        omitted from the payload so the served model's own default applies.
    overlap : float | None
        Fraction of window overlap forwarded to the server-side detector.

    Returns
    -------
    dict
        Parsed JSON response.

    Raises
    ------
    ValueError
        If `audio` is not a 2-D array.
    """
    if audio.ndim != 2:
        raise ValueError(f"Expected 2D audio array [batch, samples], got shape {audio.shape}")

    batch, samples = audio.shape
    payload: dict = {
        "audio": np.ascontiguousarray(audio, dtype=np.float32).tobytes(),
        "batch": batch,
        "samples": samples,
        "overlap": overlap,
    }
    if batch_size is not None:
        payload["batch_size"] = batch_size
    return client(payload)


@dataclass
class ServedDetectorClient:
    """Adapter exposing a served detector to the scoring helpers — a client, not a model.

    Holds one route-bound `HttpClient` per server endpoint and decodes their
    responses. Construction is plain data (handy for tests); use `from_config`
    to connect to a real server and populate the metadata from ``GET /``.

    Attributes
    ----------
    run_client : HttpClient
        Client bound to the server's ``/run`` route.
    classifier_client : HttpClient
        Client bound to the server's ``/run_as_classifier`` route.
    labels : list[str]
        Output class labels, as reported by the server.
    sample_rate : int
        Expected input audio sample rate in Hz.
    frame_rate : float
        Output frame rate in frames per second.
    window_duration : float
        Detector input window duration in seconds.
    server_config : dict
        Raw ``GET /`` payload, kept so the CLI can compare the live server
        against a checkpoint without a second fetch.
    """

    run_client: HttpClient
    classifier_client: HttpClient
    labels: list[str]
    sample_rate: int
    frame_rate: float
    window_duration: float
    server_config: dict

    @classmethod
    def from_config(cls, config: HttpClientConfig, meta: dict | None = None) -> "ServedDetectorClient":
        """Connect to the server described by `config` and fetch its metadata.

        Parameters
        ----------
        config : HttpClientConfig
            Connection settings for the server. Its `route` is ignored: the
            ``/run`` and ``/run_as_classifier`` clients are derived from copies
            of the config with the route overridden. Its `audio_key` is forced
            to ``"audio"`` (the wire contract's audio key), so a bare
            ``HttpClientConfig(url=...)`` connects a working client.
        meta : dict or None
            The server's ``GET /`` payload, when the caller already fetched it
            (e.g. `detector_client_from_config` describes once to pick the
            client class). When ``None`` (default) it is fetched here.

        Returns
        -------
        ServedDetectorClient
            Connected client with metadata populated from ``GET /``. If the
            metadata fetch or its parsing fails, the just-opened route
            clients are closed before the error propagates.
        """
        config = config.model_copy(update={"audio_key": "audio"})
        run_client = HttpClient.from_config(config.model_copy(update={"route": _RUN_ROUTE}))
        classifier_client = HttpClient.from_config(config.model_copy(update={"route": _CLASSIFIER_ROUTE}))
        try:
            if meta is None:
                meta = run_client.describe()  # describe() targets the base URL, not the route
            return cls(
                run_client=run_client,
                classifier_client=classifier_client,
                labels=list(meta["labels"]),
                sample_rate=int(meta["sample_rate"]),
                frame_rate=float(meta["frame_rate"]),
                window_duration=float(meta["window_duration"]),
                server_config=meta,
            )
        except Exception:
            run_client.close()
            classifier_client.close()
            raise

    def describe_summary(self) -> dict:
        """Return a compact, serialisable summary of the served model's metadata.

        Returns
        -------
        dict
            ``{"n_labels", "sample_rate", "frame_rate", "window_duration"}`` —
            recorded in the results file to identify which model produced them.
        """
        return {
            "n_labels": len(self.labels),
            "sample_rate": self.sample_rate,
            "frame_rate": self.frame_rate,
            "window_duration": self.window_duration,
        }

    def run(self, audio: np.ndarray, batch_size: int | None = None, overlap: float | None = None) -> DetectorOutput:
        """Run frame-level inference on a batch of equal-length recordings.

        Parameters
        ----------
        audio : np.ndarray
            Batched waveform of shape ``(batch, samples)`` at `self.sample_rate`.
        batch_size : int | None
            Number of windows the server processes per forward pass. When
            ``None`` (default) the served model's own default applies.
        overlap : float | None
            Fraction of window overlap forwarded to the server-side detector.

        Returns
        -------
        DetectorOutput
            Frame-level predictions of shape ``(batch, time, classes)``.
        """
        return decode_run_response(_post_audio(self.run_client, audio, batch_size, overlap), self.labels)

    def run_as_classifier(
        self, audio: np.ndarray, batch_size: int | None = None, overlap: float | None = None
    ) -> MultiLabelClassifierOutput:
        """Run clip-level classification on a batch of recordings.

        Parameters
        ----------
        audio : np.ndarray
            Batched waveform of shape ``(batch, samples)`` at `self.sample_rate`.
        batch_size : int | None
            Number of windows the server processes per forward pass. When
            ``None`` (default) the served model's own default applies.
        overlap : float | None
            Fraction of window overlap forwarded to the server-side detector.

        Returns
        -------
        MultiLabelClassifierOutput
            Clip-level predictions of shape ``(batch, classes)``.
        """
        return decode_classifier_response(_post_audio(self.classifier_client, audio, batch_size, overlap), self.labels)

    def close(self) -> None:
        """Close the underlying route-bound HTTP clients."""
        self.run_client.close()
        self.classifier_client.close()
