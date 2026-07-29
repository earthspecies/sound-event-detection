"""Unit tests for the frame-detector serving app.

Exercises the FastAPI app via `TestClient` with a stub detector injected, so no
GPU, weights, or network are needed.
"""

from __future__ import annotations

import base64

import httpx
import numpy as np
import torch
from fastapi.testclient import TestClient

from esp_research.protocols.classifier import MultiLabelClassifierOutput
from esp_research.protocols.detector import DetectorOutput
from sound_event_detection.serving.server import create_app, git_head_commit, state_dict_sha256


class StubDetector:
    """Minimal detector with deterministic, input-dependent inference methods."""

    def __init__(self) -> None:
        self.labels = ["a", "b", "c"]
        self.sample_rate = 32000
        self.frame_rate = 10.0
        self.window_duration = 5.0

    def run(self, audio: np.ndarray, batch_size: int = 32, overlap: float | None = None) -> DetectorOutput:
        if overlap is not None and not (0.0 <= overlap < 1.0):
            raise ValueError(f"overlap must be in [0.0, 1.0), got {overlap}")
        # Predictions encode the first sample of each recording so tests can
        # confirm the server decoded + reshaped the audio row-major.
        col0 = np.clip(audio[:, 0], 0.0, 1.0).astype(np.float32)
        preds = np.broadcast_to(col0[:, None, None], (audio.shape[0], 2, len(self.labels))).astype(np.float32)
        return DetectorOutput(predictions=preds, frame_rate=self.frame_rate, class_names=self.labels)

    def run_as_classifier(
        self, audio: np.ndarray, batch_size: int = 32, overlap: float | None = None
    ) -> MultiLabelClassifierOutput:
        if overlap is not None and not (0.0 <= overlap < 1.0):
            raise ValueError(f"overlap must be in [0.0, 1.0), got {overlap}")
        # Clip-level: encode the first sample of each recording into every class.
        col0 = np.clip(audio[:, 0], 0.0, 1.0).astype(np.float32)
        preds = np.broadcast_to(col0[:, None], (audio.shape[0], len(self.labels))).astype(np.float32)
        return MultiLabelClassifierOutput(predictions=preds, class_names=self.labels)


def _client() -> TestClient:
    return TestClient(create_app(model=StubDetector()))


def test_health():
    with _client() as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_describe_and_labels():
    with _client() as client:
        describe = client.get("/").json()
        labels = client.get("/labels").json()
    assert describe == {
        "labels": ["a", "b", "c"],
        "sample_rate": 32000,
        "frame_rate": 10.0,
        "window_duration": 5.0,
    }
    assert labels == ["a", "b", "c"]


def test_describe_merges_extras_identity():
    """`describe_extras` keys are merged into ``GET /`` alongside the base four."""
    extras = {"type": "frame", "model_folder": "/ckpt", "weights_sha256": "deadbeef", "git_commit": "abc"}
    app = create_app(model=StubDetector(), describe_extras=lambda _model: extras)
    with TestClient(app) as client:
        describe = client.get("/").json()
    assert describe == {
        "labels": ["a", "b", "c"],
        "sample_rate": 32000,
        "frame_rate": 10.0,
        "window_duration": 5.0,
        **extras,
    }


def test_state_dict_sha256_is_stable_and_weight_sensitive():
    """The digest is deterministic for identical weights and changes when they do."""
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    first = state_dict_sha256(model)
    assert isinstance(first, str) and len(first) == 64
    assert state_dict_sha256(model) == first  # deterministic across calls

    reloaded = torch.nn.Linear(4, 2)
    reloaded.load_state_dict(model.state_dict())
    assert state_dict_sha256(reloaded) == first  # byte-identical weights → same digest

    with torch.no_grad():
        model.weight[0, 0] += 1.0
    assert state_dict_sha256(model) != first  # a changed weight → different digest


def test_git_head_commit_returns_hash_or_none():
    """Best-effort commit stamp: a 40-char hash in a checkout, else ``None``."""
    commit = git_head_commit()
    assert commit is None or (isinstance(commit, str) and len(commit) == 40)


def test_run_roundtrip_decodes_predictions():
    batch, samples = 2, 8
    audio = np.linspace(0.0, 1.0, batch * samples, dtype=np.float32).reshape(batch, samples)
    body = {
        "audio": base64.b64encode(audio.tobytes()).decode("ascii"),
        "batch": batch,
        "samples": samples,
        "batch_size": 4,
        "overlap": None,
    }
    with _client() as client:
        resp = client.post("/run", json=body)

    assert resp.status_code == 200
    data = resp.json()
    assert data["shape"] == [batch, 2, 3]
    assert data["dtype"] == "float16"
    assert data["frame_rate"] == 10.0

    preds = np.frombuffer(base64.b64decode(data["predictions"]), dtype=np.float16).reshape(data["shape"])
    # Server must have reshaped row-major: preds[i,0,0] == audio[i, 0]
    # (within float16 precision, since predictions are sent as float16).
    np.testing.assert_allclose(preds[:, 0, 0].astype(np.float32), audio[:, 0], rtol=0, atol=1e-3)


def test_run_as_classifier_roundtrip_decodes_predictions():
    batch, samples = 2, 8
    audio = np.linspace(0.0, 1.0, batch * samples, dtype=np.float32).reshape(batch, samples)
    body = {
        "audio": base64.b64encode(audio.tobytes()).decode("ascii"),
        "batch": batch,
        "samples": samples,
        "batch_size": 4,
        "overlap": None,
    }
    with _client() as client:
        resp = client.post("/run_as_classifier", json=body)

    assert resp.status_code == 200
    data = resp.json()
    assert data["shape"] == [batch, 3]
    assert data["dtype"] == "float16"
    assert "frame_rate" not in data

    preds = np.frombuffer(base64.b64decode(data["predictions"]), dtype=np.float16).reshape(data["shape"])
    # Server reshaped row-major: preds[i, 0] == audio[i, 0] (float16 precision).
    np.testing.assert_allclose(preds[:, 0].astype(np.float32), audio[:, 0], rtol=0, atol=1e-3)


def test_run_as_classifier_rejects_invalid_overlap():
    batch, samples = 1, 8
    audio = np.zeros(batch * samples, dtype=np.float32)
    body = {
        "audio": base64.b64encode(audio.tobytes()).decode("ascii"),
        "batch": batch,
        "samples": samples,
        "overlap": 1.5,
    }
    with _client() as client:
        resp = client.post("/run_as_classifier", json=body)
    assert resp.status_code == 422


def test_run_rejects_bad_length():
    audio = np.zeros(10, dtype=np.float32)  # 10 floats, but we claim 2*100
    body = {
        "audio": base64.b64encode(audio.tobytes()).decode("ascii"),
        "batch": 2,
        "samples": 100,
    }
    with _client() as client:
        resp = client.post("/run", json=body)
    assert resp.status_code == 422


def test_run_rejects_invalid_overlap():
    batch, samples = 1, 8
    audio = np.zeros(batch * samples, dtype=np.float32)
    body = {
        "audio": base64.b64encode(audio.tobytes()).decode("ascii"),
        "batch": batch,
        "samples": samples,
        "overlap": 1.5,
    }
    with _client() as client:
        resp = client.post("/run", json=body)
    assert resp.status_code == 422


class RecordingStub(StubDetector):
    """Stub whose own batch_size default differs from the old wire default."""

    def __init__(self) -> None:
        super().__init__()
        self.run_calls: list[dict] = []
        self.closed = False

    def run(self, audio: np.ndarray, batch_size: int = 8, overlap: float | None = None) -> DetectorOutput:
        self.run_calls.append({"batch_size": batch_size, "overlap": overlap})
        return super().run(audio, batch_size=batch_size, overlap=overlap)

    def close(self) -> None:
        self.closed = True


def _body(batch: int = 1, samples: int = 8, **extra: object) -> dict:
    audio = np.zeros(batch * samples, dtype=np.float32)
    return {"audio": base64.b64encode(audio.tobytes()).decode("ascii"), "batch": batch, "samples": samples, **extra}


def test_run_without_batch_size_uses_the_model_default():
    model = RecordingStub()
    with TestClient(create_app(model=model)) as client:
        client.post("/run", json=_body())
        client.post("/run", json=_body(batch_size=32))

    # No wire value -> the model's own default (8 here); explicit values pass through.
    assert [call["batch_size"] for call in model.run_calls] == [8, 32]


class BackendErrorStub(StubDetector):
    """Stub raising the httpx errors a wrapped backend client would raise."""

    def __init__(self, exc: Exception) -> None:
        super().__init__()
        self._exc = exc

    def run(self, audio: np.ndarray, batch_size: int = 8, overlap: float | None = None) -> DetectorOutput:
        raise self._exc


def _status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://backend:1/run")
    response = httpx.Response(status_code, request=request, text="backend detail")
    return httpx.HTTPStatusError("boom", request=request, response=response)


def test_backend_4xx_maps_to_422():
    with TestClient(create_app(model=BackendErrorStub(_status_error(422)))) as client:
        resp = client.post("/run", json=_body())
    assert resp.status_code == 422
    assert "backend detail" in resp.json()["detail"]


def test_backend_5xx_maps_to_502():
    with TestClient(create_app(model=BackendErrorStub(_status_error(500)))) as client:
        resp = client.post("/run", json=_body())
    assert resp.status_code == 502


def test_backend_unreachable_maps_to_503():
    with TestClient(create_app(model=BackendErrorStub(httpx.ConnectError("refused")))) as client:
        resp = client.post("/run", json=_body())
    assert resp.status_code == 503


def test_lifespan_closes_factory_built_model():
    model = RecordingStub()
    with TestClient(create_app(model_factory=lambda: model)) as client:
        client.get("/health")
    assert model.closed


def test_lifespan_leaves_injected_model_open():
    model = RecordingStub()
    with TestClient(create_app(model=model)) as client:
        client.get("/health")
    assert not model.closed
