"""Synchronous HTTP client that implements the `InferenceModel` protocol."""

from typing import Any

import httpx
import numpy as np

from esp_research.adapters.client_config import HttpAuthConfig, HttpClientConfig
from esp_research.logging import logger
from esp_research.utils.audio import bytes_to_base64_string, coerce_audio_bytes


class HttpClient:
    """Thin HTTP client for communicating with adapter service apps.

    Sends JSON POST requests to a remote adapter service and returns the
    JSON response as-is.  Optionally base64-encodes audio data before
    sending.

    Parameters
    ----------
    url : str
        Base URL of the adapter service (e.g. ``"http://localhost:8001"``).
    route : str | None
        Specific route for inference POST requests (e.g.
        ``"beans-zero-forgiving"``).  The full POST URL is
        ``{url}/{route}``.  When ``None``, requests are sent to ``url``.
    auth : HttpAuthConfig | None
        Optional authentication configuration.
    timeout : float
        Request timeout in seconds.
    retries : int
        Number of retry attempts on transient failures (5xx, 429).
    audio_key : str | None
        Key in `input_data` whose value holds raw audio bytes.  When set
        the bytes are base64-encoded and the original key is replaced
        with the encoded string in the request payload.
    audio_format : str | None
        Audio format label (e.g. ``"wav"``, ``"mp3"``).  When set, an
        ``audio_format`` key is added to the request payload alongside
        the encoded audio.
    """

    config_class = HttpClientConfig

    def __init__(
        self,
        url: str,
        route: str | None = None,
        auth: HttpAuthConfig | None = None,
        timeout: float = 30.0,
        retries: int = 3,
        audio_key: str | None = None,
        audio_format: str | None = None,
    ) -> None:
        self._base_url = url.rstrip("/")
        self._url = f"{self._base_url}/{route}" if route else self._base_url
        self._retries = retries
        self._audio_key = audio_key
        self._audio_format = audio_format

        headers: dict[str, str] = {}
        if auth is not None:
            headers[auth.header] = auth.value

        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            headers=headers,
        )

    @classmethod
    def from_config(cls, config: HttpClientConfig) -> "HttpClient":
        """Create an `HttpClient` from an `HttpClientConfig`.

        Parameters
        ----------
        config : HttpClientConfig
            The HTTP client configuration.

        Returns
        -------
        HttpClient
            The configured client instance.
        """
        return cls(
            url=config.url,
            route=config.route,
            auth=config.auth,
            timeout=config.timeout,
            retries=config.retries,
            audio_key=config.audio_key,
            audio_format=config.audio_format,
        )

    def __call__(self, input_data: dict[str, Any] | Any = None, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Send a single inference request.

        If ``audio_key`` is configured and present in `input_data`, the
        raw audio bytes are base64-encoded in-place before the request
        is sent.

        Parameters
        ----------
        input_data : dict[str, Any] | Any
            The input data to send as JSON.
        **kwargs : Any
            Additional keyword arguments merged into `input_data` when it
            is a dict.

        Returns
        -------
        dict[str, Any]
            The JSON response from the server.
        """
        body = self._prepare_request(input_data, **kwargs)
        return self._post_with_retry(body)

    def _prepare_request(self, input_data: dict[str, Any] | Any, **kwargs: Any) -> dict[str, Any] | Any:  # noqa: ANN401
        """Build the request body, encoding audio if configured.

        Parameters
        ----------
        input_data : dict[str, Any] | Any
            Caller-supplied data.
        **kwargs : Any
            Additional keyword arguments merged into the request body when
            `input_data` is a dict.

        Returns
        -------
        dict[str, Any] | Any
            The JSON-serialisable request body.  Non-dict inputs are
            returned unchanged.
        """
        if input_data is None:
            input_data = kwargs
        elif isinstance(input_data, dict) and kwargs:
            input_data = {**input_data, **kwargs}

        if not isinstance(input_data, dict):
            return input_data

        data = dict(input_data)

        if self._audio_key is not None and self._audio_key in data:
            audio_val = data[self._audio_key]
            if isinstance(audio_val, list) and audio_val and isinstance(audio_val[0], (bytes, list, np.ndarray)):
                # Batch: each element is its own audio sample (bytes / list-of-floats / ndarray)
                data[self._audio_key] = [
                    bytes_to_base64_string(coerce_audio_bytes(a, self._audio_key)) for a in audio_val
                ]
            else:
                data[self._audio_key] = bytes_to_base64_string(coerce_audio_bytes(audio_val, self._audio_key))
            if self._audio_format is not None:
                data["audio_format"] = self._audio_format

        return data

    def _post_with_retry(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST JSON with retry on transient failures.

        Retries on HTTP 429 (rate limit) and 5xx (server error) status codes.

        Parameters
        ----------
        body : dict[str, Any]
            The JSON request body.

        Returns
        -------
        dict[str, Any]
            The parsed JSON response.

        Raises
        ------
        httpx.TimeoutException
            If the request times out (not retried).
        httpx.HTTPStatusError
            If the request fails after all retry attempts.
        """
        last_exc: httpx.HTTPStatusError | httpx.TransportError | None = None

        for _ in range(1 + self._retries):
            try:
                response = self._client.post(self._url, json=body)
                response.raise_for_status()
                return response.json()
            except httpx.TimeoutException:
                raise
            except httpx.TransportError as exc:
                logger.warning("Transport error, retrying. Detail: %s", exc)
                last_exc = exc
            except httpx.HTTPStatusError as exc:
                response_body = response.text
                if response.status_code in (429, 500, 502, 503, 504):
                    logger.warning(
                        "Transient error (status %d), retrying. Response body: %s",
                        response.status_code,
                        response_body,
                    )
                    last_exc = exc
                else:
                    logger.error(
                        "Non-retriable error (status %d). Response body: %s",
                        response.status_code,
                        response_body,
                    )
                    raise

        raise last_exc  # type: ignore[misc]

    def describe(self) -> dict[str, Any]:
        """Fetch the adapter's active configuration from ``GET /``.

        Returns
        -------
        dict[str, Any]
            The adapter's active configuration as returned by ``GET /``.
        """
        response = self._client.get(self._base_url)
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "HttpClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
