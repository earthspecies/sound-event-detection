"""Client-side adapter for talking to a served denoising detector over HTTP.

`ServedDenoisingDetectorClient` extends `ServedDetectorClient` with the
denoising-specific surface of a served `DenoisingDetector` (see
`sound_event_detection.serving.serve_denoising_detector`): the `separate_and_detect`
call that hands back a `StemDetections`, plus the model's `threshold`,
`resampling_method`, and `n_stems` read from ``GET /``. Everything derived
from a `StemDetections` (`combined` / `denoise` / `quality` / `stem_pairs`)
stays pure client-side numpy — only the separate -> detect core crosses the
network.

The wire contract adds one route to the standard detector contract, speaking
raw binary because the stem payload is ``n_stems`` times the whole
recording::

    POST /separate_and_detect?samples=&batch_size=&overlap=
        request body:  raw float32 PCM (application/octet-stream)
        response body: raw float32 stems + raw float16 stem predictions,
                       layout in x-stems-shape / x-preds-shape /
                       x-preds-dtype / x-frame-rate / x-sample-rate /
                       x-timings headers
"""

import json
import time
from collections.abc import MutableMapping
from dataclasses import dataclass

import httpx
import numpy as np

from esp_research.adapters import HttpClient
from esp_research.adapters.client_config import HttpClientConfig
from sound_event_detection.adapters.served_client import _CLASSIFIER_ROUTE, _RUN_ROUTE, ServedDetectorClient
from sound_event_detection.denoising.denoising_detector import (
    DEFAULT_DETECTOR_BATCH_SIZE,
    DENOISING_DETECTOR_TYPE,
    StemDetections,
)

_SEPARATE_AND_DETECT_ROUTE = "separate_and_detect"


class SeparateAndDetectTransport:
    """Raw-binary HTTP transport for the ``/separate_and_detect`` route.

    The route's payloads are raw ``application/octet-stream`` bodies (see the
    module docstring), which `esp_research.adapters.HttpClient` cannot speak
    (it posts and parses JSON), so this transport owns its own `httpx.Client`
    — mirroring `BirdMixItClient`'s binary wire — and reproduces the same
    transient-failure retry policy as `HttpClient`.

    Parameters
    ----------
    url : str
        Base URL of the denoising server (e.g. ``http://host:8110``).
    timeout : float
        Request timeout in seconds.
    retries : int
        Retry attempts on transient failures (429/5xx statuses and transport
        errors; timeouts are not retried).
    headers : dict[str, str] | None
        Extra headers for every request (e.g. auth).
    """

    def __init__(
        self, url: str, timeout: float = 300.0, retries: int = 3, headers: dict[str, str] | None = None
    ) -> None:
        self.url = f"{url.rstrip('/')}/{_SEPARATE_AND_DETECT_ROUTE}"
        self._retries = retries
        self._client = httpx.Client(timeout=httpx.Timeout(timeout), headers=headers or {})

    def post(self, audio_bytes: bytes, params: dict) -> httpx.Response:
        """POST raw audio bytes to the route and return the raw response.

        Parameters
        ----------
        audio_bytes : bytes
            Contiguous little-endian float32 PCM to send as the body.
        params : dict
            Query parameters (``samples``, ``batch_size``, optional
            ``overlap``).

        Returns
        -------
        httpx.Response
            The successful (2xx) response, undecoded.

        Raises
        ------
        httpx.TimeoutException
            If a request times out (not retried).
        httpx.HTTPStatusError
            For a non-retryable status (e.g. 422), or a retryable
            status/transport failure that persists after all retry attempts
            (the last such error is re-raised, so a persistent transport
            failure surfaces as its `httpx.TransportError`).
        """
        last_exc: httpx.HTTPStatusError | httpx.TransportError | None = None

        for _ in range(1 + self._retries):
            try:
                response = self._client.post(
                    self.url, params=params, content=audio_bytes, headers={"Content-Type": "application/octet-stream"}
                )
                response.raise_for_status()
                return response
            except httpx.TimeoutException:
                raise
            except httpx.TransportError as exc:
                last_exc = exc
            except httpx.HTTPStatusError as exc:
                if response.status_code in (429, 500, 502, 503, 504):
                    last_exc = exc
                else:
                    raise

        raise last_exc  # type: ignore[misc]

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()


@dataclass
class ServedDenoisingDetectorClient(ServedDetectorClient):
    """Adapter exposing a served denoising detector — a client, not a model.

    Everything from `ServedDetectorClient` (the standard detector contract)
    plus the denoising add-ons large-scale inference consumes:
    `separate_and_detect`, `threshold`, and `n_stems`.

    Attributes
    ----------
    separate_client : SeparateAndDetectTransport
        Raw-binary transport bound to the server's ``/separate_and_detect``
        route.
    threshold : float
        The served model's default focal probability threshold for the
        denoised waveform, as reported by the server.
    resampling_method : str
        The served model's stem -> detector-rate resampling method.
    n_stems : int
        Number of stems the served model's separator produces.
    """

    separate_client: SeparateAndDetectTransport
    threshold: float
    resampling_method: str
    n_stems: int

    @classmethod
    def from_config(cls, config: HttpClientConfig, meta: dict | None = None) -> "ServedDenoisingDetectorClient":
        """Connect to the denoising server described by `config`.

        Parameters
        ----------
        config : HttpClientConfig
            Connection settings for the server. Its `route` is ignored: the
            route-bound clients are derived from copies of the config. Its
            `audio_key` is forced to ``"audio"`` (the wire contract's audio
            key), so a bare ``HttpClientConfig(url=...)`` connects a working
            client.
        meta : dict or None
            The server's ``GET /`` payload, when the caller already fetched it
            (e.g. `detector_client_from_config` describes once to pick the
            client class). When ``None`` (default) it is fetched here.

        Returns
        -------
        ServedDenoisingDetectorClient
            Connected client with metadata populated from ``GET /``. If the
            metadata fetch, its validation, or its parsing fails, the
            just-opened clients are closed before the error propagates.

        Raises
        ------
        ValueError
            If the server's metadata does not identify a denoising detector
            (its ``GET /`` payload lacks ``type == "denoising_detector"``).
        """
        config = config.model_copy(update={"audio_key": "audio"})
        run_client = HttpClient.from_config(config.model_copy(update={"route": _RUN_ROUTE}))
        classifier_client = HttpClient.from_config(config.model_copy(update={"route": _CLASSIFIER_ROUTE}))
        separate_client = SeparateAndDetectTransport(
            url=config.url,
            timeout=config.timeout,
            retries=config.retries,
            headers={config.auth.header: config.auth.value} if config.auth is not None else None,
        )
        try:
            if meta is None:
                meta = run_client.describe()  # describe() targets the base URL, not the route
            if meta.get("type") != DENOISING_DETECTOR_TYPE:
                raise ValueError(
                    f"Server at {config.url} is not a denoising detector (GET / reported type "
                    f"{meta.get('type')!r}); point the client at a sed.denoising_app server."
                )
            return cls(
                run_client=run_client,
                classifier_client=classifier_client,
                labels=list(meta["labels"]),
                sample_rate=int(meta["sample_rate"]),
                frame_rate=float(meta["frame_rate"]),
                window_duration=float(meta["window_duration"]),
                server_config=meta,
                separate_client=separate_client,
                threshold=float(meta["threshold"]),
                resampling_method=str(meta["resampling_method"]),
                n_stems=int(meta["separator"]["n_stems"]),
            )
        except Exception:
            run_client.close()
            classifier_client.close()
            separate_client.close()
            raise

    def separate_and_detect(
        self,
        audio: np.ndarray,
        batch_size: int = DEFAULT_DETECTOR_BATCH_SIZE,
        timings: MutableMapping[str, float] | None = None,
        overlap: float | None = None,
    ) -> StemDetections:
        """Separate a whole recording into stems and detect over them, remotely.

        Mirrors `DenoisingDetector.separate_and_detect`: the server separates,
        resamples, and detects; this client reconstructs the `StemDetections`
        core locally so all its derivations (`combined` / `denoise` /
        `quality` / `stem_pairs`) run client-side.

        Parameters
        ----------
        audio : np.ndarray
            Mono waveform of shape ``(samples,)`` at `self.sample_rate`.
        batch_size : int
            Windows per detector forward pass, forwarded to the served model.
            See `DEFAULT_DETECTOR_BATCH_SIZE`.
        timings : MutableMapping[str, float] or None
            Optional out-parameter for stage-level timing. The server-side
            stage times (``"separate"`` / ``"resample"`` / ``"detect"``) are
            added to any existing values, so a single mapping can accumulate
            across recordings (matching
            `DenoisingDetector.separate_and_detect`); the round-trip overhead
            beyond those stages (transfer, serialization, queueing) is added
            under ``"wire"`` so the network cost stays visible in stage
            stats.
        overlap : float | None
            Fraction of window overlap forwarded to the served model's
            wrapped detector.

        Returns
        -------
        StemDetections
            The whole-file stems and their per-stem framewise predictions.
            The stems array is a read-only view over the response buffer (no
            copy); all `StemDetections` derivations are read-only.

        Raises
        ------
        ValueError
            If `audio` is not a 1-D array.
        """
        if audio.ndim != 1:
            raise ValueError(f"Expected 1D audio array [samples], got shape {audio.shape}")

        params: dict = {"samples": int(audio.shape[0]), "batch_size": batch_size}
        if overlap is not None:
            params["overlap"] = overlap

        start = time.perf_counter()
        response = self.separate_client.post(np.ascontiguousarray(audio, dtype=np.float32).tobytes(), params)
        wall = time.perf_counter() - start

        stems_shape = tuple(int(dim) for dim in response.headers["x-stems-shape"].split(","))
        preds_shape = tuple(int(dim) for dim in response.headers["x-preds-shape"].split(","))
        preds_dtype = np.dtype(response.headers.get("x-preds-dtype", "float32"))
        stems_count = int(np.prod(stems_shape))
        stems = np.frombuffer(response.content, dtype=np.float32, count=stems_count).reshape(stems_shape)
        stem_preds = (
            np.frombuffer(response.content, dtype=preds_dtype, count=int(np.prod(preds_shape)), offset=stems_count * 4)
            .reshape(preds_shape)
            .astype(np.float32)
        )

        server_timings: dict[str, float] = json.loads(response.headers.get("x-timings", "{}"))
        if timings is not None:
            for stage, seconds in server_timings.items():
                timings[stage] = timings.get(stage, 0.0) + seconds
            timings["wire"] = timings.get("wire", 0.0) + max(0.0, wall - sum(server_timings.values()))

        return StemDetections(
            stems=stems,
            stem_preds=stem_preds,
            frame_rate=float(response.headers["x-frame-rate"]),
            labels=list(self.labels),
            sample_rate=int(response.headers["x-sample-rate"]),
        )

    def describe_summary(self) -> dict:
        """Return a compact, serialisable summary of the served model's metadata.

        Returns
        -------
        dict
            ``{"n_labels", "sample_rate", "frame_rate", "window_duration",
            "n_stems"}`` — recorded in the results file to identify which
            model produced them (matching `DenoisingDetector.describe_summary`).
        """
        return {**super().describe_summary(), "n_stems": self.n_stems}

    def close(self) -> None:
        """Close the underlying route-bound HTTP clients."""
        super().close()
        self.separate_client.close()
