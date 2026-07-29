"""Tests for HttpClient and HttpClientConfig."""

import base64
import json
from typing import Any

import httpx
import numpy as np
import pytest
from pydantic import ValidationError

from esp_research.adapters import HttpAuthConfig, HttpClient, HttpClientConfig


class TestHttpClientConfig:
    """Tests for `HttpClientConfig` validation."""

    def test_minimal_config(self) -> None:
        cfg = HttpClientConfig(url="http://localhost:8000/predict")

        assert cfg.url == "http://localhost:8000/predict"
        assert cfg.auth is None
        assert cfg.timeout == 30.0
        assert cfg.retries == 3
        assert cfg.audio_key is None
        assert cfg.audio_format is None

    def test_full_config(self) -> None:
        cfg = HttpClientConfig(
            url="https://api.openai.com/v1/responses",
            auth=HttpAuthConfig(header="Authorization", value="Bearer sk-test"),
            timeout=60.0,
            retries=5,
            audio_key="audio_input",
            audio_format="wav",
        )

        assert cfg.auth is not None
        assert cfg.auth.header == "Authorization"
        assert cfg.auth.value == "Bearer sk-test"
        assert cfg.timeout == 60.0
        assert cfg.retries == 5
        assert cfg.audio_key == "audio_input"
        assert cfg.audio_format == "wav"

    def test_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            HttpClientConfig(url="http://localhost", timeout=0)

    def test_retries_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            HttpClientConfig(url="http://localhost", retries=-1)


class TestHttpClientConstruction:
    """Tests for `HttpClient` construction from config."""

    def test_from_config_minimal(self) -> None:
        cfg = HttpClientConfig(url="http://localhost:8000/predict")
        client = HttpClient.from_config(cfg)

        assert client._url == "http://localhost:8000/predict"
        assert client._retries == 3
        assert client._audio_key is None
        assert client._audio_format is None

    def test_from_config_with_auth(self) -> None:
        cfg = HttpClientConfig(
            url="http://localhost:8000",
            auth=HttpAuthConfig(header="X-API-Key", value="secret"),
        )
        client = HttpClient.from_config(cfg)

        assert client._client.headers["X-API-Key"] == "secret"

    def test_from_config_with_audio(self) -> None:
        cfg = HttpClientConfig(
            url="http://localhost:8000",
            audio_key="audio_input",
            audio_format="mp3",
        )
        client = HttpClient.from_config(cfg)

        assert client._audio_key == "audio_input"
        assert client._audio_format == "mp3"

    def test_context_manager(self) -> None:
        cfg = HttpClientConfig(url="http://localhost:8000")
        with HttpClient.from_config(cfg) as client:
            assert isinstance(client, HttpClient)


def _make_client_with_transport(
    transport: httpx.MockTransport,
    **kwargs: Any,
) -> HttpClient:
    """Create a client with a mock transport for testing.

    Returns
    -------
    HttpClient
        Client whose requests are served by `transport`.
    """
    client = HttpClient(
        url="http://test/predict",
        retries=0,
        **kwargs,
    )
    client._client = httpx.Client(transport=transport)
    return client


class TestHttpClientRequests:
    """Tests for `HttpClient` request handling and retries."""

    def test_call_sends_json_and_returns_response(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"predictions": ["cat"]})

        client = _make_client_with_transport(httpx.MockTransport(handler))
        result = client({"audio": "data"})

        assert result == {"predictions": ["cat"]}

    def test_call_with_kwargs(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"ok": True})

        client = _make_client_with_transport(httpx.MockTransport(handler))
        client(audio="data", instruction="classify")

        assert captured["body"] == {"audio": "data", "instruction": "classify"}

    def test_call_kwargs_merged_with_input_data(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"ok": True})

        client = _make_client_with_transport(httpx.MockTransport(handler))
        client({"audio": "data"}, instruction="classify")

        assert captured["body"] == {"audio": "data", "instruction": "classify"}

    def test_retry_on_server_error(self) -> None:
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return httpx.Response(503)
            return httpx.Response(200, json={"ok": True})

        client = HttpClient(url="http://test/predict", retries=3)
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        result = client({"data": 1})

        assert result == {"ok": True}
        assert call_count == 3

    def test_retry_on_rate_limit(self) -> None:
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(429)
            return httpx.Response(200, json={"ok": True})

        client = HttpClient(url="http://test/predict", retries=1)
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        result = client({"data": 1})

        assert result == {"ok": True}
        assert call_count == 2

    def test_no_retry_on_client_error(self) -> None:
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(400)

        client = HttpClient(url="http://test/predict", retries=3)
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        with pytest.raises(httpx.HTTPStatusError):
            client({"data": 1})

        assert call_count == 1

    def test_raises_after_retries_exhausted(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        client = HttpClient(url="http://test/predict", retries=2)
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        with pytest.raises(httpx.HTTPStatusError):
            client({"data": 1})


class TestHttpClientAudio:
    """Tests for audio encoding in `HttpClient`."""

    @staticmethod
    def _make_capturing_client(
        captured: list[dict[str, Any]],
        **kwargs: Any,
    ) -> HttpClient:
        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={"result": "ok"})

        client = HttpClient(url="http://test/predict", retries=0, **kwargs)
        client._client = httpx.Client(transport=httpx.MockTransport(handler))
        return client

    def test_bytes_audio_encoded(self) -> None:
        captured: list[dict[str, Any]] = []
        client = self._make_capturing_client(captured, audio_key="audio")

        raw = b"fake_audio"
        client({"audio": raw, "text": "hello"})

        assert captured[0]["audio"] == base64.b64encode(raw).decode("utf-8")
        assert captured[0]["text"] == "hello"

    def test_numpy_audio_encoded(self) -> None:
        captured: list[dict[str, Any]] = []
        client = self._make_capturing_client(captured, audio_key="audio")

        arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        client({"audio": arr})

        expected = base64.b64encode(arr.tobytes()).decode("utf-8")
        assert captured[0]["audio"] == expected

    def test_list_audio_encoded(self) -> None:
        captured: list[dict[str, Any]] = []
        client = self._make_capturing_client(captured, audio_key="audio")

        data = [1.0, 2.0, 3.0]
        client({"audio": data})

        expected = base64.b64encode(np.array(data, dtype=np.float32).tobytes()).decode("utf-8")
        assert captured[0]["audio"] == expected

    def test_audio_format_included_when_set(self) -> None:
        captured: list[dict[str, Any]] = []
        client = self._make_capturing_client(captured, audio_key="audio", audio_format="wav")

        client({"audio": b"data"})

        assert captured[0]["audio_format"] == "wav"

    def test_audio_format_omitted_when_not_set(self) -> None:
        captured: list[dict[str, Any]] = []
        client = self._make_capturing_client(captured, audio_key="audio")

        client({"audio": b"data"})

        assert "audio_format" not in captured[0]

    def test_no_audio_key_passes_data_through(self) -> None:
        captured: list[dict[str, Any]] = []
        client = self._make_capturing_client(captured)

        client({"audio": "already_encoded", "text": "hello"})

        assert captured[0] == {"audio": "already_encoded", "text": "hello"}

    def test_missing_audio_key_in_data_passes_through(self) -> None:
        captured: list[dict[str, Any]] = []
        client = self._make_capturing_client(captured, audio_key="audio")

        client({"text": "no audio here"})

        assert captured[0] == {"text": "no audio here"}
