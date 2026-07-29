"""Unit tests for the served-detector client adapter and decode helpers."""

import base64

import numpy as np
import pytest

from esp_research.adapters import HttpClient, HttpClientConfig
from sound_event_detection.adapters.served_client import (
    SedRunResponse,
    ServedDetectorClient,
    decode_classifier_response,
    decode_run_response,
)


def _encode(arr: np.ndarray, dtype: str = "float16") -> dict:
    """Build a server-style base64 response dict for a prediction array.

    Parameters
    ----------
    arr : np.ndarray
        Prediction array to encode.
    dtype : str
        Wire dtype to encode with.

    Returns
    -------
    dict
        ``{"predictions", "shape", "dtype"}``.
    """
    encoded = np.ascontiguousarray(arr, dtype=dtype)
    return {
        "predictions": base64.b64encode(encoded.tobytes()).decode("ascii"),
        "shape": list(encoded.shape),
        "dtype": dtype,
    }


class _FakeRoute:
    """Callable stand-in for a route-bound HttpClient that returns a canned response."""

    def __init__(self, response: dict) -> None:
        self.response = response
        self.last_payload: dict | None = None
        self.closed = False

    def __call__(self, payload: dict) -> dict:
        self.last_payload = payload
        return self.response

    def close(self) -> None:
        self.closed = True


def _make_client(meta: dict, run_resp: dict, clf_resp: dict) -> tuple[ServedDetectorClient, _FakeRoute, _FakeRoute]:
    """Build a `ServedDetectorClient` wired to fake route clients.

    Parameters
    ----------
    meta : dict
        Server metadata (``GET /`` payload).
    run_resp : dict
        Canned ``/run`` response.
    clf_resp : dict
        Canned ``/run_as_classifier`` response.

    Returns
    -------
    tuple[ServedDetectorClient, _FakeRoute, _FakeRoute]
        The client and its two fake route clients.
    """
    run_client = _FakeRoute(run_resp)
    classifier_client = _FakeRoute(clf_resp)
    client = ServedDetectorClient(
        run_client=run_client,  # type: ignore[arg-type]
        classifier_client=classifier_client,  # type: ignore[arg-type]
        labels=list(meta["labels"]),
        sample_rate=meta["sample_rate"],
        frame_rate=meta["frame_rate"],
        window_duration=meta["window_duration"],
        server_config=meta,
    )
    return client, run_client, classifier_client


def test_decode_run_response_roundtrip() -> None:
    preds = np.random.default_rng(0).random((1, 5, 3)).astype(np.float32)
    response = {**_encode(preds), "frame_rate": 100.0}

    out = decode_run_response(response, ["a", "b", "c"])

    assert out.predictions.shape == (1, 5, 3)
    assert out.frame_rate == 100.0
    assert out.class_names == ["a", "b", "c"]
    np.testing.assert_allclose(out.predictions, preds.astype(np.float16).astype(np.float32))


def test_decode_classifier_response_roundtrip() -> None:
    clip = np.random.default_rng(1).random((2, 3)).astype(np.float32)

    out = decode_classifier_response(_encode(clip), ["a", "b", "c"])

    assert out.predictions.shape == (2, 3)
    assert out.class_names == ["a", "b", "c"]


def test_sed_run_response_schema_validates() -> None:
    preds = np.zeros((1, 2, 2), dtype=np.float16)
    response = {**_encode(preds), "frame_rate": 25.0}

    parsed = SedRunResponse(**response)

    assert parsed.dtype == "float16"
    assert parsed.frame_rate == 25.0
    assert parsed.shape == [1, 2, 2]


def test_from_config_binds_run_and_classifier_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    meta = {"labels": ["a", "b"], "sample_rate": 16000, "frame_rate": 25.0, "window_duration": 3.0}
    monkeypatch.setattr(HttpClient, "describe", lambda self: meta)

    client = ServedDetectorClient.from_config(HttpClientConfig(url="http://localhost:9"))

    assert client.run_client._url == "http://localhost:9/run"
    assert client.classifier_client._url == "http://localhost:9/run_as_classifier"
    assert client.labels == ["a", "b"]
    assert client.sample_rate == 16000
    assert client.frame_rate == 25.0
    assert client.window_duration == 3.0
    assert client.server_config == meta
    client.close()


def test_served_detector_client_run_and_classify() -> None:
    preds = np.random.default_rng(2).random((2, 4, 3)).astype(np.float32)
    clip = np.random.default_rng(3).random((2, 3)).astype(np.float32)
    meta = {"labels": ["a", "b", "c"], "sample_rate": 32000, "frame_rate": 50.0, "window_duration": 5.0}
    client, run_client, classifier_client = _make_client(meta, {**_encode(preds), "frame_rate": 50.0}, _encode(clip))

    assert client.labels == ["a", "b", "c"]
    assert client.sample_rate == 32000
    assert client.frame_rate == 50.0
    assert client.window_duration == 5.0
    assert client.describe_summary()["n_labels"] == 3

    audio = np.zeros((2, 32000), dtype=np.float32)
    out = client.run(audio, batch_size=8, overlap=0.5)
    assert out.predictions.shape == (2, 4, 3)

    payload = run_client.last_payload
    assert payload["batch"] == 2
    assert payload["samples"] == 32000
    assert payload["batch_size"] == 8
    assert payload["overlap"] == 0.5
    assert isinstance(payload["audio"], (bytes, bytearray))

    clf = client.run_as_classifier(audio)
    assert clf.predictions.shape == (2, 3)

    client.close()
    assert run_client.closed
    assert classifier_client.closed


def test_served_detector_client_rejects_non_2d_audio() -> None:
    meta = {"labels": ["a"], "sample_rate": 16000, "frame_rate": 10.0, "window_duration": 1.0}
    client, _, _ = _make_client(meta, _encode(np.zeros((1, 1, 1), np.float32)), _encode(np.zeros((1, 1), np.float32)))

    with pytest.raises(ValueError, match="2D audio"):
        client.run(np.zeros((16000,), dtype=np.float32))


def test_default_batch_size_is_omitted_so_the_model_default_applies() -> None:
    meta = {"labels": ["a"], "sample_rate": 16000, "frame_rate": 10.0, "window_duration": 1.0}
    client, run_client, classifier_client = _make_client(
        meta, {**_encode(np.zeros((1, 1, 1), np.float32)), "frame_rate": 10.0}, _encode(np.zeros((1, 1), np.float32))
    )
    audio = np.zeros((1, 16000), dtype=np.float32)

    client.run(audio)
    client.run_as_classifier(audio)

    assert "batch_size" not in run_client.last_payload
    assert "batch_size" not in classifier_client.last_payload


def test_from_config_forces_the_wire_audio_key(monkeypatch: pytest.MonkeyPatch) -> None:
    meta = {"labels": ["a"], "sample_rate": 16000, "frame_rate": 25.0, "window_duration": 3.0}
    monkeypatch.setattr(HttpClient, "describe", lambda self: meta)

    # A bare config (audio_key defaults to None) must still yield a client
    # whose route clients base64-encode the audio bytes.
    client = ServedDetectorClient.from_config(HttpClientConfig(url="http://localhost:9"))

    assert client.run_client._audio_key == "audio"
    assert client.classifier_client._audio_key == "audio"
    client.close()


def test_from_config_closes_clients_when_the_metadata_fetch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[str] = []

    def failing_describe(self: HttpClient) -> dict:
        raise RuntimeError("server exploded")

    monkeypatch.setattr(HttpClient, "describe", failing_describe)
    monkeypatch.setattr(HttpClient, "close", lambda self: closed.append(self._url))

    with pytest.raises(RuntimeError, match="server exploded"):
        ServedDetectorClient.from_config(HttpClientConfig(url="http://localhost:9"))

    assert closed == ["http://localhost:9/run", "http://localhost:9/run_as_classifier"]
