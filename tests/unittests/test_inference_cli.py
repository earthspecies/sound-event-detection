"""Unit tests for the LSI CLI helpers.

Exercises `_model_lineage` — the ``model`` block of the run lineage record —
and the `make_process` over-long-audio guard, with lightweight fakes, so no
server, dataset, or network is needed.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from esp_research.adapters.client_config import HttpAuthConfig, HttpClientConfig
from esp_research.protocols.detector import DetectorOutput
from sound_event_detection.inference.cli import _model_lineage, make_process


def _config() -> HttpClientConfig:
    return HttpClientConfig(
        url="http://server:8100",
        timeout=120.0,
        retries=2,
        auth=HttpAuthConfig(header="Authorization", value="secret-token"),
    )


def test_model_lineage_records_server_identity_and_drops_auth():
    """When the server exposes identity, both connection and server blocks are recorded."""
    server_config = {
        "type": "frame",
        "model_folder": "/ckpt",
        "weights_sha256": "deadbeef",
        "git_commit": "abc123",
        "labels": ["a", "b"],
    }
    model = SimpleNamespace(server_config=server_config)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning expected on the enriched path
        record = _model_lineage(_config(), model)

    assert record["server"] == server_config
    assert record["client"]["url"] == "http://server:8100"
    assert record["client"]["timeout"] == 120.0
    assert record["client"]["retries"] == 2
    assert "auth" not in record["client"]  # auth is never written to lineage


def test_model_lineage_falls_back_and_warns_without_identity():
    """A server whose GET / lacks a git_commit marker → previous behaviour + warning."""
    # server_config present (labels etc.) but no identity marker.
    model = SimpleNamespace(server_config={"labels": ["a", "b"], "sample_rate": 32000})

    with pytest.warns(UserWarning, match="did not expose model identity"):
        record = _model_lineage(_config(), model)

    # Fallback is the previous behaviour: the plain connection-config dump.
    assert record == _config().model_dump(exclude={"auth"})
    assert "server" not in record
    assert "auth" not in record


def test_model_lineage_falls_back_when_server_config_missing():
    """A client with no server_config at all still falls back cleanly (with a warning)."""
    model = SimpleNamespace()  # no server_config attribute

    with pytest.warns(UserWarning, match="did not expose model identity"):
        record = _model_lineage(_config(), model)

    assert record["url"] == "http://server:8100"
    assert "server" not in record


class _FakeDetector:
    """A detector client whose `run` fails if it is ever reached."""

    sample_rate = 10

    def run(self, audio: np.ndarray) -> DetectorOutput:
        frames = audio.shape[-1]
        return DetectorOutput(
            predictions=np.full((1, frames, 1), 0.5, dtype=np.float32), frame_rate=5.0, class_names=["a"]
        )


def _preds_process(max_audio_seconds: float | None):
    return make_process(_FakeDetector(), None, "preds", 0.05, max_audio_seconds=max_audio_seconds)


def test_make_process_skips_over_long_audio():
    """An over-long recording raises before the model runs (so the engine logs+skips it)."""
    process = _preds_process(max_audio_seconds=1.0)  # cap 1 s; audio below is 3 s
    with pytest.raises(ValueError, match="max_audio_seconds"):
        process({"audio": np.zeros(30, dtype=np.float32)})


def test_make_process_passes_audio_within_cap():
    """Audio at or under the cap reaches the model and encodes normally."""
    process = _preds_process(max_audio_seconds=1.0)  # cap 1 s; audio below is 0.5 s
    arrays = process({"audio": np.zeros(5, dtype=np.float32), "latitudeDecimal": 1.0, "longitudeDecimal": 2.0})
    assert arrays  # produced a non-empty encoding rather than raising
