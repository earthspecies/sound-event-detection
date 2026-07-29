"""Unit tests for the served denoising-detector client adapter."""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from esp_research.adapters.client_config import HttpClientConfig
from sound_event_detection.adapters import served_denoising_client as sdc
from sound_event_detection.adapters.served_denoising_client import (
    SeparateAndDetectTransport,
    ServedDenoisingDetectorClient,
)
from sound_event_detection.denoising.denoising_detector import StemDetections

_LABELS = ["a", "b", "c"]
_SEP_SR = 22050
_N_STEMS = 4
_FRAME_RATE = 20.0

_META = {
    "labels": _LABELS,
    "sample_rate": _SEP_SR,
    "frame_rate": _FRAME_RATE,
    "window_duration": 5.0,
    "type": "denoising_detector",
    "detector": {"labels": _LABELS, "sample_rate": 32000},
    "separator": {"sample_rate": _SEP_SR, "n_stems": _N_STEMS},
    "threshold": 0.3,
    "resampling_method": "torchaudio_kaiser_fast",
}


class _FakeRoute:
    """Callable stand-in for a route-bound HttpClient that returns a canned response."""

    def __init__(self, response: dict | None = None) -> None:
        self.response = response or {}
        self.last_payload: dict | None = None
        self.closed = False

    def __call__(self, payload: dict) -> dict:
        self.last_payload = payload
        return self.response

    def close(self) -> None:
        self.closed = True


class _FakeSeparateTransport:
    """Stand-in for `SeparateAndDetectTransport` returning a canned raw response."""

    def __init__(self, response: httpx.Response | None = None) -> None:
        self.response = response
        self.last_content: bytes | None = None
        self.last_params: dict | None = None
        self.closed = False

    def post(self, audio_bytes: bytes, params: dict) -> httpx.Response:
        self.last_content = audio_bytes
        self.last_params = params
        assert self.response is not None, "test posted without a canned response"
        return self.response

    def close(self) -> None:
        self.closed = True


def _separate_response(stems: np.ndarray, stem_preds: np.ndarray, timings: dict | None = None) -> httpx.Response:
    """Build a server-style raw-binary ``/separate_and_detect`` response.

    Returns
    -------
    httpx.Response
        Raw float32 stems + float16 preds body with the layout headers.
    """
    stems32 = np.ascontiguousarray(stems, dtype=np.float32)
    preds16 = np.ascontiguousarray(stem_preds, dtype=np.float16)
    return httpx.Response(
        200,
        content=stems32.tobytes() + preds16.tobytes(),
        headers={
            "x-stems-shape": ",".join(str(dim) for dim in stems32.shape),
            "x-preds-shape": ",".join(str(dim) for dim in preds16.shape),
            "x-preds-dtype": "float16",
            "x-frame-rate": str(_FRAME_RATE),
            "x-sample-rate": str(_SEP_SR),
            "x-timings": json.dumps(timings or {}),
        },
    )


def _make_client(
    separate_response: httpx.Response | None = None,
) -> tuple[ServedDenoisingDetectorClient, _FakeSeparateTransport]:
    separate_client = _FakeSeparateTransport(separate_response)
    client = ServedDenoisingDetectorClient(
        run_client=_FakeRoute(),  # type: ignore[arg-type]
        classifier_client=_FakeRoute(),  # type: ignore[arg-type]
        labels=list(_LABELS),
        sample_rate=_SEP_SR,
        frame_rate=_FRAME_RATE,
        window_duration=5.0,
        server_config=dict(_META),
        separate_client=separate_client,  # type: ignore[arg-type]
        threshold=0.3,
        resampling_method="torchaudio_kaiser_fast",
        n_stems=_N_STEMS,
    )
    return client, separate_client


class _FakeHttpClient:
    """Fake `HttpClient` recording configs and answering `describe` with a canned meta."""

    instances: list["_FakeHttpClient"] = []
    meta: dict = {}

    def __init__(self, config: HttpClientConfig) -> None:
        self.config = config
        self.closed = False
        type(self).instances.append(self)

    @classmethod
    def from_config(cls, config: HttpClientConfig) -> "_FakeHttpClient":
        return cls(config)

    def describe(self) -> dict:
        return dict(type(self).meta)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_http(monkeypatch: pytest.MonkeyPatch) -> type:
    _FakeHttpClient.instances = []
    _FakeHttpClient.meta = dict(_META)
    monkeypatch.setattr(sdc, "HttpClient", _FakeHttpClient)
    return _FakeHttpClient


def test_from_config_binds_routes_and_reads_meta(fake_http: type) -> None:
    client = ServedDenoisingDetectorClient.from_config(HttpClientConfig(url="http://localhost:9"))

    routes = [instance.config.route for instance in fake_http.instances]
    assert routes == ["run", "run_as_classifier"]
    # The audio key is forced even on a bare config, so the JSON routes
    # base64-encode the audio bytes.
    assert all(instance.config.audio_key == "audio" for instance in fake_http.instances)
    assert isinstance(client.separate_client, SeparateAndDetectTransport)
    assert client.separate_client.url == "http://localhost:9/separate_and_detect"
    assert client.labels == _LABELS
    assert client.sample_rate == _SEP_SR
    assert client.frame_rate == _FRAME_RATE
    assert client.window_duration == 5.0
    assert client.threshold == 0.3
    assert client.resampling_method == "torchaudio_kaiser_fast"
    assert client.n_stems == _N_STEMS
    assert client.server_config == _META
    client.close()


def test_from_config_rejects_non_denoising_meta(fake_http: type) -> None:
    plain_meta = {"labels": _LABELS, "sample_rate": 32000, "frame_rate": 10.0, "window_duration": 5.0}

    with pytest.raises(ValueError, match="not a denoising detector"):
        ServedDenoisingDetectorClient.from_config(HttpClientConfig(url="http://localhost:9"), meta=plain_meta)

    assert all(instance.closed for instance in fake_http.instances)


def test_from_config_closes_clients_when_meta_is_malformed(fake_http: type) -> None:
    # A version-skewed denoising server: right type, but no threshold key.
    skewed = {key: value for key, value in _META.items() if key != "threshold"}

    with pytest.raises(KeyError, match="threshold"):
        ServedDenoisingDetectorClient.from_config(HttpClientConfig(url="http://localhost:9"), meta=skewed)

    assert all(instance.closed for instance in fake_http.instances)


def test_separate_and_detect_roundtrip() -> None:
    rng = np.random.default_rng(0)
    stems = rng.random((_N_STEMS, 32)).astype(np.float32)
    stem_preds = rng.random((_N_STEMS, 3, len(_LABELS))).astype(np.float32)
    client, separate_client = _make_client(_separate_response(stems, stem_preds))
    audio = np.linspace(-1.0, 1.0, 32, dtype=np.float32)

    core = client.separate_and_detect(audio, batch_size=16, overlap=0.5)

    assert separate_client.last_params == {"samples": 32, "batch_size": 16, "overlap": 0.5}
    np.testing.assert_array_equal(np.frombuffer(separate_client.last_content, dtype=np.float32), audio)

    assert isinstance(core, StemDetections)
    assert core.frame_rate == _FRAME_RATE
    assert core.sample_rate == _SEP_SR
    assert core.labels == _LABELS
    np.testing.assert_array_equal(core.stems, stems)  # float32, bit-exact across the raw wire
    np.testing.assert_allclose(core.stem_preds, stem_preds.astype(np.float16).astype(np.float32))
    # Derivations run client-side on the reconstructed core.
    np.testing.assert_allclose(core.combined().predictions[0], core.stem_preds.max(axis=0))


def test_separate_and_detect_default_batch_size_is_eight() -> None:
    stems = np.zeros((_N_STEMS, 8), dtype=np.float32)
    preds = np.zeros((_N_STEMS, 1, len(_LABELS)), dtype=np.float32)
    client, separate_client = _make_client(_separate_response(stems, preds))

    client.separate_and_detect(np.zeros(8, dtype=np.float32))

    assert separate_client.last_params == {"samples": 8, "batch_size": 8}  # no overlap key when None


def test_separate_and_detect_merges_timings_additively() -> None:
    stems = np.zeros((_N_STEMS, 8), dtype=np.float32)
    preds = np.zeros((_N_STEMS, 1, len(_LABELS)), dtype=np.float32)
    client, _ = _make_client(_separate_response(stems, preds, timings={"separate": 1.0, "detect": 0.5}))

    timings: dict[str, float] = {"separate": 0.5}
    client.separate_and_detect(np.zeros(8, dtype=np.float32), timings=timings)
    client.separate_and_detect(np.zeros(8, dtype=np.float32), timings=timings)

    wire = timings.pop("wire")
    assert wire >= 0.0  # round-trip overhead beyond the server stages stays visible
    assert timings == {"separate": 2.5, "detect": 1.0}


def test_separate_and_detect_rejects_non_1d_audio() -> None:
    client, _ = _make_client()
    with pytest.raises(ValueError, match="1D"):
        client.separate_and_detect(np.zeros((1, 8), dtype=np.float32))


def test_describe_summary_includes_n_stems() -> None:
    client, _ = _make_client()
    assert client.describe_summary() == {
        "n_labels": len(_LABELS),
        "sample_rate": _SEP_SR,
        "frame_rate": _FRAME_RATE,
        "window_duration": 5.0,
        "n_stems": _N_STEMS,
    }


def test_close_closes_all_three_routes() -> None:
    client, separate_client = _make_client()

    client.close()

    assert client.run_client.closed  # type: ignore[attr-defined]
    assert client.classifier_client.closed  # type: ignore[attr-defined]
    assert separate_client.closed
