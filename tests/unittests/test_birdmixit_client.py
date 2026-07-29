"""Unit tests for `BirdMixItClient`.

`HttpClient` is replaced with a fake that mimics the BirdMixIt server's wire
contract (base64 float32 stems + shape + dtype), so no network is needed. The
fake echoes each block across stems, which lets tests verify block ordering is
preserved when large batches are split across requests.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

from sound_event_detection.denoising import birdmixit_client as bc

_SAMPLE_RATE = 22050
_N_STEMS = 4
_MAX_BATCH = 4


class FakeHttpClient:
    """Stand-in for `esp_research.adapters.HttpClient`.

    `describe` returns canned server metadata; `__call__` echoes each input
    block across `_N_STEMS` stems and records every request payload so tests
    can assert on request splitting.
    """

    calls: list[dict] = []

    def __init__(self, url: str, route: str | None = None, audio_key: str | None = None, timeout: float = 30.0) -> None:
        self.url = url
        self.route = route
        self.audio_key = audio_key
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def describe(self) -> dict:
        return {
            "sample_rate": _SAMPLE_RATE,
            "n_stems": _N_STEMS,
            "max_batch_size": _MAX_BATCH,
            "model": "fake",
            "stub": True,
        }

    def __call__(self, payload: dict) -> dict:
        FakeHttpClient.calls.append(payload)
        samples = payload["samples"]
        if "batch" not in payload:  # /separate_file: whole mono file -> (n_stems, samples)
            mono = np.frombuffer(payload["audio"], dtype=np.float32).reshape(samples)
            stems = np.repeat(mono[np.newaxis, :], _N_STEMS, axis=0).astype(np.float32)
            shape = [_N_STEMS, samples]
        else:  # /separate: (batch, samples) -> (batch, n_stems, samples)
            batch = payload["batch"]
            blocks = np.frombuffer(payload["audio"], dtype=np.float32).reshape(batch, samples)
            stems = np.repeat(blocks[:, np.newaxis, :], _N_STEMS, axis=1).astype(np.float32)
            shape = [batch, _N_STEMS, samples]
        return {
            "stems": base64.b64encode(stems.tobytes()).decode("ascii"),
            "shape": shape,
            "dtype": "float32",
            "sample_rate": _SAMPLE_RATE,
        }


def _echo_stems(content: bytes, params: dict) -> np.ndarray:
    # Mirror the server: echo each block across _N_STEMS stems.
    samples = params["samples"]
    if "batch" not in params:  # /separate_file_binary: whole mono -> (n_stems, samples)
        mono = np.frombuffer(content, dtype=np.float32).reshape(samples)
        stems = np.repeat(mono[np.newaxis, :], _N_STEMS, axis=0)
    else:  # /separate_binary: (batch, samples) -> (batch, n_stems, samples)
        blocks = np.frombuffer(content, dtype=np.float32).reshape(params["batch"], samples)
        stems = np.repeat(blocks[:, np.newaxis, :], _N_STEMS, axis=1)
    return np.ascontiguousarray(stems, dtype=np.float32)


class FakeBinaryResponse:
    """Minimal stand-in for an `httpx.Response` from a binary separate endpoint."""

    def __init__(self, content: bytes, headers: dict) -> None:
        self.content = content
        self.headers = headers

    def raise_for_status(self) -> None:
        return None


class FakeBinaryClient:
    """Stand-in for the `httpx.Client` used by `BirdMixItClient._post_binary`.

    Records each POST (url/params/content) and echoes the input across `_N_STEMS`
    stems as a raw float32 body with the shape/dtype/sample-rate headers.
    """

    calls: list[dict] = []

    def __init__(self, timeout: object = None) -> None:
        self.timeout = timeout
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def post(
        self, url: str, params: dict | None = None, content: bytes | None = None, headers: dict | None = None
    ) -> FakeBinaryResponse:
        FakeBinaryClient.calls.append({"url": url, "params": params, "content": content, "headers": headers})
        stems = _echo_stems(content or b"", params or {})
        return FakeBinaryResponse(
            stems.tobytes(),
            {
                "x-shape": ",".join(str(dim) for dim in stems.shape),
                "x-dtype": "float32",
                "x-sample-rate": str(_SAMPLE_RATE),
            },
        )


@pytest.fixture
def url():
    return "http://localhost:9999"


@pytest.fixture
def patched_client(monkeypatch):
    FakeHttpClient.calls = []
    monkeypatch.setattr(bc, "HttpClient", FakeHttpClient)
    return FakeHttpClient


@pytest.fixture
def patched_binary(monkeypatch):
    FakeBinaryClient.calls = []
    monkeypatch.setattr(bc.httpx, "Client", FakeBinaryClient)
    return FakeBinaryClient


def test_init_populates_attrs_from_describe(url, patched_client):
    client = bc.BirdMixItClient(url)
    assert client.sample_rate == _SAMPLE_RATE
    assert client.n_stems == _N_STEMS
    assert client.max_batch_size == _MAX_BATCH


def test_separate_serializes_audio_and_decodes_stems(url, patched_client):
    # base64-JSON path (binary=False).
    client = bc.BirdMixItClient(url, binary=False)
    audio = np.linspace(0.0, 1.0, 3 * 8, dtype=np.float32).reshape(3, 8)

    stems = client.separate(audio)

    assert stems.shape == (3, _N_STEMS, 8)
    assert stems.dtype == np.float32
    # Fake echoes each block across stems.
    for s in range(_N_STEMS):
        np.testing.assert_array_equal(stems[:, s, :], audio)

    assert len(patched_client.calls) == 1
    payload = patched_client.calls[0]
    assert payload["batch"] == 3
    assert payload["samples"] == 8
    sent = np.frombuffer(payload["audio"], dtype=np.float32).reshape(3, 8)
    np.testing.assert_array_equal(sent, audio)


def test_separate_splits_batches_over_max_batch_size(url, patched_client):
    client = bc.BirdMixItClient(url, binary=False)
    # 10 blocks with max_batch_size=4 -> requests of 4, 4, 2.
    audio = np.arange(10 * 5, dtype=np.float32).reshape(10, 5)

    stems = client.separate(audio)

    assert stems.shape == (10, _N_STEMS, 5)
    # Ordering preserved across the split/concat.
    np.testing.assert_array_equal(stems[:, 0, :], audio)
    assert [c["batch"] for c in patched_client.calls] == [4, 4, 2]


def test_separate_rejects_non_2d_audio(url, patched_client):
    client = bc.BirdMixItClient(url)
    with pytest.raises(ValueError, match="2D"):
        client.separate(np.zeros(8, dtype=np.float32))


def test_separate_file_serializes_audio_and_decodes_stems(url, patched_client):
    client = bc.BirdMixItClient(url, binary=False)
    audio = np.linspace(0.0, 1.0, 200, dtype=np.float32)

    stems = client.separate_file(audio)

    assert stems.shape == (_N_STEMS, 200)
    assert stems.dtype == np.float32
    # Fake echoes the whole file across stems.
    for s in range(_N_STEMS):
        np.testing.assert_array_equal(stems[s], audio)

    payload = patched_client.calls[-1]
    assert payload["samples"] == 200 and "batch" not in payload
    np.testing.assert_array_equal(np.frombuffer(payload["audio"], dtype=np.float32), audio)


def test_separate_file_rejects_non_1d_audio(url, patched_client):
    client = bc.BirdMixItClient(url)
    with pytest.raises(ValueError, match="1D"):
        client.separate_file(np.zeros((2, 8), dtype=np.float32))


def test_default_is_binary(url, patched_client):
    assert bc.BirdMixItClient(url).binary is True


def test_separate_binary_posts_raw_audio_and_decodes_stems(url, patched_client, patched_binary):
    # Default binary=True: raw octet-stream body to /separate_binary, layout as
    # query params, raw stems + shape header back.
    client = bc.BirdMixItClient(url)
    audio = np.linspace(0.0, 1.0, 3 * 8, dtype=np.float32).reshape(3, 8)

    stems = client.separate(audio)

    assert stems.shape == (3, _N_STEMS, 8)
    assert stems.dtype == np.float32
    for s in range(_N_STEMS):
        np.testing.assert_array_equal(stems[:, s, :], audio)

    assert len(patched_binary.calls) == 1
    call = patched_binary.calls[0]
    assert call["url"].endswith("/separate_binary")
    assert call["headers"]["Content-Type"] == "application/octet-stream"
    assert call["params"] == {"batch": 3, "samples": 8}
    np.testing.assert_array_equal(np.frombuffer(call["content"], dtype=np.float32).reshape(3, 8), audio)


def test_separate_binary_splits_batches_over_max_batch_size(url, patched_client, patched_binary):
    client = bc.BirdMixItClient(url)
    audio = np.arange(10 * 5, dtype=np.float32).reshape(10, 5)

    stems = client.separate(audio)

    assert stems.shape == (10, _N_STEMS, 5)
    np.testing.assert_array_equal(stems[:, 0, :], audio)
    assert [c["params"]["batch"] for c in patched_binary.calls] == [4, 4, 2]
    assert all(c["url"].endswith("/separate_binary") for c in patched_binary.calls)


def test_separate_file_binary_posts_raw_audio_and_decodes_stems(url, patched_client, patched_binary):
    client = bc.BirdMixItClient(url)
    audio = np.linspace(0.0, 1.0, 200, dtype=np.float32)

    stems = client.separate_file(audio)

    assert stems.shape == (_N_STEMS, 200)
    assert stems.dtype == np.float32
    for s in range(_N_STEMS):
        np.testing.assert_array_equal(stems[s], audio)

    call = patched_binary.calls[-1]
    assert call["url"].endswith("/separate_file_binary")
    assert call["params"] == {"samples": 200} and "batch" not in call["params"]
    np.testing.assert_array_equal(np.frombuffer(call["content"], dtype=np.float32), audio)


def test_close_closes_all_transport_clients(url, patched_client, patched_binary):
    client = bc.BirdMixItClient(url)

    client.close()

    assert client._client.closed
    assert client._file_client.closed
    assert client._binary_client.closed
