"""Unit tests for the config-driven client dispatcher.

`detector_client_from_config` takes an `HttpClientConfig`, describes the
server once, and connects the client class matching the server kind. The
`HttpClient` sites (the dispatcher's probe and both served-client classes) are
replaced with fakes that answer ``describe`` with canned server metadata, so no
network or model is needed.
"""

from __future__ import annotations

import pytest

from esp_research.adapters.client_config import HttpClientConfig
from sound_event_detection.adapters import dispatch
from sound_event_detection.adapters import served_client as sc
from sound_event_detection.adapters import served_denoising_client as sdc
from sound_event_detection.adapters.served_client import ServedDetectorClient
from sound_event_detection.adapters.served_denoising_client import ServedDenoisingDetectorClient

_LABELS = ["a", "b", "c"]

#: A plain detector server's ``GET /`` payload — the base four fields only.
_DETECTOR_META = {"labels": _LABELS, "sample_rate": 32000, "frame_rate": 10.0, "window_duration": 5.0}

#: A denoising detector server's ``GET /`` payload — the base fields merged
#: with the model's composed `server_config`.
_DENOISING_META = {
    "labels": _LABELS,
    "sample_rate": 22050,
    "frame_rate": 10.0,
    "window_duration": 5.0,
    "type": "denoising_detector",
    "detector": _DETECTOR_META,
    "separator": {"sample_rate": 22050, "n_stems": 4},
    "threshold": 0.3,
    "resampling_method": "torchaudio_kaiser_fast",
}


def _make_fake_http_client(meta: dict) -> type:
    """Build a fake `HttpClient` class answering `describe` with `meta`.

    The class records every instance and counts `describe` calls so tests can
    assert the dispatcher describes the server exactly once.

    Returns
    -------
    type
        The fake `HttpClient` class.
    """

    class FakeHttpClient:
        describe_calls = 0
        instances: list["FakeHttpClient"] = []

        def __init__(self, config: object = None) -> None:
            self.config = config
            self.closed = False
            type(self).instances.append(self)

        @classmethod
        def from_config(cls, config: object) -> "FakeHttpClient":
            return cls(config)

        def describe(self) -> dict:
            type(self).describe_calls += 1
            return meta

        def close(self) -> None:
            self.closed = True

    return FakeHttpClient


@pytest.fixture
def url() -> str:
    return "http://localhost:9999"


def _patch_http_clients(monkeypatch: pytest.MonkeyPatch, meta: dict) -> type:
    """Point every `HttpClient` site at a fake answering `describe` with `meta`.

    Returns
    -------
    type
        The fake class, for asserting on its instances and describe count.
    """
    fake = _make_fake_http_client(meta)
    monkeypatch.setattr(sc, "HttpClient", fake)
    monkeypatch.setattr(sdc, "HttpClient", fake)
    return fake


def test_plain_server_returns_served_detector_client(url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_http_clients(monkeypatch, _DETECTOR_META)

    model = dispatch.detector_client_from_config(HttpClientConfig(url=url))

    assert isinstance(model, ServedDetectorClient)
    assert not isinstance(model, ServedDenoisingDetectorClient)
    assert model.labels == _LABELS
    assert model.frame_rate == 10.0
    assert model.server_config == _DETECTOR_META


def test_transport_fields_are_forwarded(url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_http_clients(monkeypatch, _DETECTOR_META)

    model = dispatch.detector_client_from_config(HttpClientConfig(url=url, timeout=60.0, retries=5))

    assert model.run_client.config.timeout == 60.0
    assert model.run_client.config.retries == 5


def test_describe_is_called_exactly_once(url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _patch_http_clients(monkeypatch, _DENOISING_META)

    dispatch.detector_client_from_config(HttpClientConfig(url=url))

    assert fake.describe_calls == 1


def test_probe_client_is_closed(url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _patch_http_clients(monkeypatch, _DETECTOR_META)

    dispatch.detector_client_from_config(HttpClientConfig(url=url))

    # The first instance is the dispatcher's probe; it must not leak.
    assert fake.instances[0].closed


def test_stale_audio_key_is_tolerated(url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_http_clients(monkeypatch, _DETECTOR_META)

    # Configs saved by pre-serving eval checkpoints carry audio_key; they must
    # keep resuming (the wire contract fixes the key to "audio" regardless).
    model = dispatch.detector_client_from_config(HttpClientConfig(url=url, audio_key="audio"))

    assert model.run_client.config.audio_key == "audio"


def test_timeout_defaults_to_config_default(url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_http_clients(monkeypatch, _DETECTOR_META)

    # The dispatcher no longer overrides timeout; a bare config keeps
    # HttpClientConfig's own default.
    model = dispatch.detector_client_from_config(HttpClientConfig(url=url))

    assert model.run_client.config.timeout == 30.0


def test_denoising_server_returns_denoising_client(url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_http_clients(monkeypatch, _DENOISING_META)

    model = dispatch.detector_client_from_config(HttpClientConfig(url=url))

    assert isinstance(model, ServedDenoisingDetectorClient)
    assert model.labels == _LABELS
    assert model.sample_rate == 22050  # the separator's rate, per the denoising server
    assert model.threshold == 0.3
    assert model.resampling_method == "torchaudio_kaiser_fast"
    assert model.n_stems == 4
    assert model.server_config == _DENOISING_META


def test_labels_mismatch_raises_and_closes(url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _patch_http_clients(monkeypatch, _DETECTOR_META)

    with pytest.raises(ValueError, match="do not match the server's labels"):
        dispatch.detector_client_from_config(HttpClientConfig(url=url), labels=["not", "the", "labels"])

    assert all(instance.closed for instance in fake.instances)


def test_matching_labels_pass(url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_http_clients(monkeypatch, _DENOISING_META)

    model = dispatch.detector_client_from_config(HttpClientConfig(url=url), labels=list(_LABELS))

    assert model.labels == _LABELS


def test_denoising_client_rejects_plain_server(url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _patch_http_clients(monkeypatch, _DETECTOR_META)

    with pytest.raises(ValueError, match="not a denoising detector"):
        ServedDenoisingDetectorClient.from_config(HttpClientConfig(url=url, audio_key="audio"))

    assert all(instance.closed for instance in fake.instances)
