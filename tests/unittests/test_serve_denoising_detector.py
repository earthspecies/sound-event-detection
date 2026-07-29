"""Unit tests for the denoising detector serving app.

Exercises the FastAPI app (`create_app` + `add_denoising_routes`) via
`TestClient` with a stub denoising model injected, so no backend servers are
needed. `_build_model` is exercised against a temp model-config YAML with
`DenoisingDetector.from_config` stubbed out.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from esp_research.adapters.client_config import HttpClientConfig
from esp_research.protocols.classifier import MultiLabelClassifierOutput
from esp_research.protocols.detector import DetectorOutput
from sound_event_detection.denoising.denoising_detector import StemDetections
from sound_event_detection.serving import serve_denoising_detector
from sound_event_detection.serving.serve_denoising_detector import add_denoising_routes
from sound_event_detection.serving.server import create_app

_LABELS = ["a", "b", "c"]
_SEP_SR = 22050
_N_STEMS = 4
_FRAME_RATE = 20.0


class StubDenoisingModel:
    """Minimal denoising model with deterministic, recordable methods."""

    def __init__(self) -> None:
        self.labels = list(_LABELS)
        self.sample_rate = _SEP_SR
        self.frame_rate = _FRAME_RATE
        self.window_duration = 5.0
        self.threshold = 0.3
        self.resampling_method = "torchaudio_kaiser_fast"
        self.server_config = {
            "type": "denoising_detector",
            "detector": {"labels": list(_LABELS), "sample_rate": 32000},
            "separator": {"sample_rate": _SEP_SR, "n_stems": _N_STEMS},
            "threshold": 0.3,
            "resampling_method": "torchaudio_kaiser_fast",
        }
        self.separate_calls: list[dict] = []

    def run(self, audio: np.ndarray, batch_size: int = 8, overlap: float | None = None) -> DetectorOutput:
        preds = np.full((audio.shape[0], 2, len(self.labels)), 0.5, dtype=np.float32)
        return DetectorOutput(predictions=preds, frame_rate=self.frame_rate, class_names=self.labels)

    def run_as_classifier(
        self, audio: np.ndarray, batch_size: int = 8, overlap: float | None = None
    ) -> MultiLabelClassifierOutput:
        preds = np.full((audio.shape[0], len(self.labels)), 0.5, dtype=np.float32)
        return MultiLabelClassifierOutput(predictions=preds, class_names=self.labels)

    def separate_and_detect(
        self,
        audio: np.ndarray,
        batch_size: int = 8,
        timings: dict | None = None,
        overlap: float | None = None,
    ) -> StemDetections:
        if overlap is not None and not (0.0 <= overlap < 1.0):
            raise ValueError(f"overlap must be in [0.0, 1.0), got {overlap}")
        self.separate_calls.append({"samples": audio.shape[0], "batch_size": batch_size, "overlap": overlap})
        if timings is not None:
            for stage, seconds in (("separate", 1.0), ("resample", 0.25), ("detect", 0.5)):
                timings[stage] = timings.get(stage, 0.0) + seconds
        # Stem 0 echoes the input; predictions encode the stem index so the
        # roundtrip can confirm shapes and ordering.
        stems = np.zeros((_N_STEMS, audio.shape[0]), dtype=np.float32)
        stems[0] = audio
        stem_preds = np.broadcast_to(
            (np.arange(_N_STEMS, dtype=np.float32) / 10.0)[:, None, None],
            (_N_STEMS, 3, len(self.labels)),
        ).copy()
        return StemDetections(
            stems=stems,
            stem_preds=stem_preds,
            frame_rate=self.frame_rate,
            labels=self.labels,
            sample_rate=self.sample_rate,
        )


def _client(model: StubDenoisingModel) -> TestClient:
    app = add_denoising_routes(create_app(model=model, describe_extras=lambda m: m.server_config))
    return TestClient(app)


def _post_separate(client: TestClient, audio: np.ndarray, **params: object) -> httpx.Response:
    """POST raw float32 audio to /separate_and_detect with layout query params."""
    return client.post(
        "/separate_and_detect",
        params={"samples": audio.shape[0], **params},
        content=np.ascontiguousarray(audio, dtype=np.float32).tobytes(),
        headers={"Content-Type": "application/octet-stream"},
    )


def test_describe_merges_denoising_identity() -> None:
    model = StubDenoisingModel()
    with _client(model) as client:
        describe = client.get("/").json()

    assert describe == {
        "labels": _LABELS,
        "sample_rate": _SEP_SR,
        "frame_rate": _FRAME_RATE,
        "window_duration": 5.0,
        **model.server_config,
    }
    assert describe["type"] == "denoising_detector"  # the auto-detect discriminator


def test_standard_routes_still_served() -> None:
    audio = np.zeros((2, 8), dtype=np.float32)
    body = {
        "audio": base64.b64encode(audio.tobytes()).decode("ascii"),
        "batch": 2,
        "samples": 8,
    }
    with _client(StubDenoisingModel()) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.post("/run", json=body).status_code == 200
        assert client.post("/run_as_classifier", json=body).status_code == 200


def test_separate_and_detect_roundtrip() -> None:
    model = StubDenoisingModel()
    audio = np.linspace(-0.5, 0.5, 16, dtype=np.float32)

    with _client(model) as client:
        resp = _post_separate(client, audio)

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/octet-stream"
    stems_shape = tuple(int(dim) for dim in resp.headers["x-stems-shape"].split(","))
    preds_shape = tuple(int(dim) for dim in resp.headers["x-preds-shape"].split(","))
    assert stems_shape == (_N_STEMS, 16)
    assert preds_shape == (_N_STEMS, 3, len(_LABELS))
    assert resp.headers["x-preds-dtype"] == "float16"
    assert float(resp.headers["x-frame-rate"]) == _FRAME_RATE
    assert int(resp.headers["x-sample-rate"]) == _SEP_SR
    assert json.loads(resp.headers["x-timings"]) == {"separate": 1.0, "resample": 0.25, "detect": 0.5}

    stems_count = int(np.prod(stems_shape))
    stems = np.frombuffer(resp.content, dtype=np.float32, count=stems_count).reshape(stems_shape)
    np.testing.assert_array_equal(stems[0], audio)  # stems cross the wire as raw float32, bit-exact
    np.testing.assert_array_equal(stems[1:], 0.0)

    preds = np.frombuffer(
        resp.content, dtype=np.float16, count=int(np.prod(preds_shape)), offset=stems_count * 4
    ).reshape(preds_shape)
    np.testing.assert_allclose(preds[:, 0, 0].astype(np.float32), np.arange(_N_STEMS) / 10.0, rtol=0, atol=1e-3)


def test_separate_and_detect_forwards_defaults_and_args() -> None:
    model = StubDenoisingModel()
    audio = np.zeros(8, dtype=np.float32)

    with _client(model) as client:
        _post_separate(client, audio)
        _post_separate(client, audio, batch_size=16, overlap=0.5)

    assert model.separate_calls[0] == {"samples": 8, "batch_size": 8, "overlap": None}
    assert model.separate_calls[1] == {"samples": 8, "batch_size": 16, "overlap": 0.5}


def test_separate_and_detect_rejects_bad_length() -> None:
    audio = np.zeros(8, dtype=np.float32)
    with _client(StubDenoisingModel()) as client:
        resp = _post_separate(client, audio, samples=100)
    assert resp.status_code == 422


def test_separate_and_detect_rejects_non_positive_samples() -> None:
    with _client(StubDenoisingModel()) as client:
        resp = client.post(
            "/separate_and_detect",
            params={"samples": 0},
            content=b"",
            headers={"Content-Type": "application/octet-stream"},
        )
    assert resp.status_code == 422


def test_separate_and_detect_rejects_model_value_error() -> None:
    audio = np.zeros(8, dtype=np.float32)
    with _client(StubDenoisingModel()) as client:
        resp = _post_separate(client, audio, overlap=1.5)
    assert resp.status_code == 422


class BackendErrorModel(StubDenoisingModel):
    """Stub raising the httpx errors a wrapped backend client would raise."""

    def __init__(self, exc: Exception) -> None:
        super().__init__()
        self._exc = exc

    def separate_and_detect(
        self,
        audio: np.ndarray,
        batch_size: int = 8,
        timings: dict | None = None,
        overlap: float | None = None,
    ) -> StemDetections:
        raise self._exc


def _status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://backend:1/run")
    response = httpx.Response(status_code, request=request, text="backend detail")
    return httpx.HTTPStatusError("boom", request=request, response=response)


def test_separate_and_detect_maps_backend_4xx_to_422() -> None:
    audio = np.zeros(8, dtype=np.float32)
    with _client(BackendErrorModel(_status_error(422))) as client:
        resp = _post_separate(client, audio)
    assert resp.status_code == 422
    assert "backend detail" in resp.json()["detail"]


def test_separate_and_detect_maps_backend_5xx_to_502() -> None:
    audio = np.zeros(8, dtype=np.float32)
    with _client(BackendErrorModel(_status_error(500))) as client:
        resp = _post_separate(client, audio)
    assert resp.status_code == 502


def test_separate_and_detect_maps_unreachable_backend_to_503() -> None:
    audio = np.zeros(8, dtype=np.float32)
    with _client(BackendErrorModel(httpx.ConnectError("refused"))) as client:
        resp = _post_separate(client, audio)
    assert resp.status_code == 503


def test_build_model_reads_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "type": "denoising_detector",
        "detector": {"url": "http://d:1"},
        "separator": {"url": "http://s:2"},
        "threshold": 0.4,
        "resampling_method": "torchaudio_kaiser_fast",
    }
    config_path = tmp_path / "denoising.yml"
    config_path.write_text(yaml.dump(config))
    monkeypatch.setenv("SED_MODEL_CONFIG", str(config_path))

    model = StubDenoisingModel()
    captured: dict = {}

    def fake_from_config(cfg: object, labels: list[str] | None = None) -> StubDenoisingModel:
        captured["config"] = cfg
        return model

    monkeypatch.setattr(serve_denoising_detector.DenoisingDetector, "from_config", fake_from_config)

    assert serve_denoising_detector._build_model() is model
    assert captured["config"].detector == HttpClientConfig(url="http://d:1")
    assert captured["config"].separator == {"url": "http://s:2"}
    assert captured["config"].threshold == 0.4
    assert captured["config"].resampling_method == "torchaudio_kaiser_fast"


def test_build_model_rejects_wrong_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "frame.yml"
    config_path.write_text(yaml.dump({"type": "frame", "model_folder": "x"}))
    monkeypatch.setenv("SED_MODEL_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="Expected 'denoising_detector'"):
        serve_denoising_detector._build_model()


def test_build_model_rejects_non_mapping_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "empty.yml"
    config_path.write_text("# comments only\n")
    monkeypatch.setenv("SED_MODEL_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="YAML mapping"):
        serve_denoising_detector._build_model()


def test_build_model_rejects_missing_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "partial.yml"
    config_path.write_text(yaml.dump({"type": "denoising_detector", "detector": {"url": "http://d:1"}}))
    monkeypatch.setenv("SED_MODEL_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="separator"):
        serve_denoising_detector._build_model()


def test_build_model_requires_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SED_MODEL_CONFIG", raising=False)

    with pytest.raises(RuntimeError, match="SED_MODEL_CONFIG"):
        serve_denoising_detector._build_model()
