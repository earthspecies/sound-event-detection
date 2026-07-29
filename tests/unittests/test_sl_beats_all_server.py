"""Unit tests for the BEATs-SL-All serving app.

Exercises the FastAPI app via `TestClient` with a stub classifier injected, so no
avex model, GPU, or network is needed.
"""

from __future__ import annotations

import base64

import numpy as np
import torch
from fastapi.testclient import TestClient

from sound_event_detection.serving.sl_beats_all_server import SAMPLE_RATE, create_app

_LABELS = ["Turdus merula", "Erithacus rubecula", "Wind noise"]


def _stub_model(audio: torch.Tensor) -> torch.Tensor:
    """Classifier whose logits encode each clip's first sample.

    Returns logits of shape ``(B, len(_LABELS))`` where every column equals the
    clip's first sample, so tests can confirm the server reshaped the flat
    buffer row-major before calling the model.
    """
    col0 = audio[:, 0]
    return col0[:, None].expand(audio.shape[0], len(_LABELS)).contiguous()


def _client() -> TestClient:
    return TestClient(create_app(model=_stub_model, labels=_LABELS))


def test_health():
    with _client() as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_describe_and_labels():
    with _client() as client:
        describe = client.get("/").json()
        labels = client.get("/labels").json()
    assert describe == {"labels": _LABELS, "sample_rate": SAMPLE_RATE}
    assert labels == _LABELS


def test_logits_roundtrip_decodes_input():
    num_windows, samples = 2, 8
    audio = np.linspace(0.0, 1.0, num_windows * samples, dtype=np.float32).reshape(num_windows, samples)
    body = {
        "audio": base64.b64encode(audio.tobytes()).decode("ascii"),
        "num_windows": num_windows,
    }
    with _client() as client:
        resp = client.post("/logits", json=body)

    assert resp.status_code == 200
    logits = np.asarray(resp.json()["logits"], dtype=np.float32)
    assert logits.shape == (num_windows, len(_LABELS))
    # Stub encodes each clip's first sample into every column; this also proves
    # the server reshaped row-major to (num_windows, samples).
    np.testing.assert_allclose(logits[:, 0], audio[:, 0], rtol=0, atol=1e-6)


def test_logits_rejects_length_not_multiple_of_num_windows():
    # 10 float32 samples cannot be split evenly into 3 clips.
    audio = np.zeros(10, dtype=np.float32)
    body = {
        "audio": base64.b64encode(audio.tobytes()).decode("ascii"),
        "num_windows": 3,
    }
    with _client() as client:
        resp = client.post("/logits", json=body)
    assert resp.status_code == 422
