"""Unit tests for `DenoisingDetector` and `StemDetections`.

The detector and separator collaborators are replaced with fakes so no server
or model is needed:

- `FakeSeparator.separate_file` routes all signal into stem 0 and leaves the
  other stems silent, so the denoised waveform should recover the (gated) input.
- `FakeDetector` reports a constant probability for any recording that carries
  energy and zero for a silent one, with a frame count proportional to the input
  length — which makes the max-over-stems combine and the focal gate exactly
  predictable.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from esp_research.adapters.client_config import HttpClientConfig
from esp_research.protocols.classifier import MultiLabelClassifierOutput
from esp_research.protocols.detector import Detector, DetectorOutput
from sound_event_detection.adapters.dispatch import DetectorClient
from sound_event_detection.denoising.denoising_detector import (
    DenoisingDetector,
    DenoisingDetectorConfig,
    StemDetections,
)
from sound_event_detection.denoising.source_separator import SourceSeparatorClient

_SEP_SR = 22050
_DET_SR = 32000
_N_STEMS = 4
_FRAMERATE = 20.0
_LABELS = ["species_a", "species_b", "species_c"]


class FakeSeparator:
    """Separator that puts all signal in stem 0 and zeros in the rest."""

    def __init__(self) -> None:
        self.sample_rate = _SEP_SR
        self.n_stems = _N_STEMS

    def separate_file(self, audio: np.ndarray) -> np.ndarray:
        stems = np.zeros((_N_STEMS, audio.shape[0]), dtype=np.float32)
        stems[0, :] = audio
        return stems


class FakeDetector:
    """Detector reporting `energetic_prob` for energetic recordings, else 0.

    The frame count is proportional to the input length, mimicking a real
    detector that windows the whole recording internally. Exposes the full
    `DetectorClient` surface.
    """

    def __init__(self, energetic_prob: float = 1.0) -> None:
        self.sample_rate = _DET_SR
        self.frame_rate = _FRAMERATE
        self.labels = list(_LABELS)
        self.window_duration = 5.0
        self.server_config = {
            "labels": list(_LABELS),
            "sample_rate": _DET_SR,
            "frame_rate": _FRAMERATE,
            "window_duration": 5.0,
        }
        self.energetic_prob = energetic_prob
        self.calls: list[dict] = []
        self.classifier_calls: list[dict] = []
        self.closed = False

    def run(self, audio: np.ndarray, batch_size: int = 32, overlap: float | None = None) -> DetectorOutput:
        self.calls.append({"shape": audio.shape, "batch_size": batch_size, "overlap": overlap})
        n_frames = max(1, int(round(audio.shape[1] / self.sample_rate * self.frame_rate)))
        has_energy = np.abs(audio).max(axis=1) > 1e-6  # (batch,)
        per_recording = np.where(has_energy, self.energetic_prob, 0.0).astype(np.float32)
        preds = np.broadcast_to(
            per_recording[:, np.newaxis, np.newaxis], (audio.shape[0], n_frames, len(self.labels))
        ).copy()
        return DetectorOutput(predictions=preds, frame_rate=_FRAMERATE, class_names=self.labels)

    def run_as_classifier(
        self, audio: np.ndarray, batch_size: int = 32, overlap: float | None = None
    ) -> MultiLabelClassifierOutput:
        self.classifier_calls.append({"shape": audio.shape, "batch_size": batch_size, "overlap": overlap})
        has_energy = np.abs(audio).max(axis=1) > 1e-6  # (batch,)
        per_recording = np.where(has_energy, self.energetic_prob, 0.0).astype(np.float32)
        preds = np.repeat(per_recording[:, np.newaxis], len(self.labels), axis=1)
        return MultiLabelClassifierOutput(predictions=preds, class_names=self.labels)

    def describe_summary(self) -> dict:
        return {
            "n_labels": len(self.labels),
            "sample_rate": self.sample_rate,
            "frame_rate": self.frame_rate,
            "window_duration": self.window_duration,
        }

    def close(self) -> None:
        self.closed = True


def _make_detector(**kwargs: object) -> DenoisingDetector:
    return DenoisingDetector(FakeDetector(**kwargs), FakeSeparator())


def _expected_frames(n_samples: int) -> int:
    """Frames the fake detector emits for `n_samples` input at the separator rate.

    The pipeline resamples the stems to `_DET_SR` before detection, so the frame
    count follows the resampled length.

    Returns
    -------
    int
        Number of frames the fake detector emits for `n_samples` input.
    """
    det_samples = int(round(n_samples / _SEP_SR * _DET_SR))
    return max(1, int(round(det_samples / _DET_SR * _FRAMERATE)))


def test_reads_config_from_collaborators() -> None:
    det = _make_detector()
    assert det.sample_rate == _SEP_SR  # separator's rate is the input contract
    assert det.frame_rate == _FRAMERATE
    assert det.labels == _LABELS


def test_unknown_resampling_method_raises() -> None:
    with pytest.raises(ValueError, match="Unknown resampling_method"):
        DenoisingDetector(FakeDetector(), FakeSeparator(), resampling_method="bogus")  # type: ignore[arg-type]


def test_batch_size_defaults_to_eight_and_is_forwarded() -> None:
    detector = FakeDetector()
    det = DenoisingDetector(detector, FakeSeparator())

    det.separate_and_detect(np.ones(_SEP_SR, dtype=np.float32))

    assert detector.calls, "detector was not called"
    assert all(call["batch_size"] == 8 for call in detector.calls)


def test_batch_size_is_forwarded_per_call() -> None:
    detector = FakeDetector()
    det = DenoisingDetector(detector, FakeSeparator())

    det.run(np.ones((1, _SEP_SR), dtype=np.float32), batch_size=4)
    det.separate_and_detect(np.ones(_SEP_SR, dtype=np.float32), batch_size=16)

    assert [call["batch_size"] for call in detector.calls] == [4, 16]


def test_satisfies_detector_client_but_not_detector() -> None:
    det = _make_detector()
    # Exposes the full client surface (run/run_as_classifier/describe_summary/
    # close + labels/sample_rate/frame_rate/window_duration/server_config)...
    assert isinstance(det, DetectorClient)
    assert isinstance(FakeDetector(), DetectorClient)
    # ...but owns no weights / checkpoint API, so it is NOT a Detector.
    assert not isinstance(det, Detector)


def test_overlap_is_forwarded_to_detector() -> None:
    detector = FakeDetector()
    det = DenoisingDetector(detector, FakeSeparator())

    det.run(np.ones((1, _SEP_SR), dtype=np.float32), overlap=0.25)
    det.separate_and_detect(np.ones(_SEP_SR, dtype=np.float32), overlap=0.5)

    assert [call["overlap"] for call in detector.calls] == [0.25, 0.5]


def test_from_config_bundles_two_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    import sound_event_detection.adapters.dispatch as dispatch_mod
    import sound_event_detection.denoising.birdmixit_client as birdmixit_mod

    detector = FakeDetector()
    captured: dict = {}

    def fake_detector_from_config(
        http_client_config: HttpClientConfig, labels: list[str] | None = None
    ) -> FakeDetector:
        captured["detector_cfg"] = http_client_config
        captured["labels"] = labels
        return detector

    def fake_birdmixit(**kwargs: object) -> FakeSeparator:
        captured["separator_kwargs"] = kwargs
        return FakeSeparator()

    monkeypatch.setattr(dispatch_mod, "detector_client_from_config", fake_detector_from_config)
    monkeypatch.setattr(birdmixit_mod, "BirdMixItClient", fake_birdmixit)

    config = DenoisingDetectorConfig(
        detector={"url": "http://detector:8100", "timeout": 60.0},
        separator={"url": "http://separator:8200", "timeout": 5.0, "binary": False},
        threshold=0.3,
        resampling_method="torchaudio_kaiser_fast",
    )
    det = DenoisingDetector.from_config(config, labels=list(_LABELS))

    assert isinstance(det, DenoisingDetector)
    assert det.threshold == 0.3
    assert det.resampling_method == "torchaudio_kaiser_fast"
    assert det.labels == _LABELS
    # The detector block is coerced to an HttpClientConfig on the config and
    # passed through the shared dispatcher, which forwards the labels check.
    assert captured["detector_cfg"] == HttpClientConfig(url="http://detector:8100", timeout=60.0)
    assert captured["labels"] == _LABELS
    assert captured["separator_kwargs"] == {"url": "http://separator:8200", "timeout": 5.0, "binary": False}


def test_from_config_rejects_stale_detector_type_key() -> None:
    # The detector block is an HttpClientConfig (extra="forbid"), so a stale
    # `type` key is rejected when the config is constructed, before any
    # connection is attempted.
    with pytest.raises(ValidationError):
        DenoisingDetectorConfig(
            detector={"type": "detector", "url": "http://detector:8100"},
            separator={"url": "http://separator:8200"},
        )


def test_from_config_rejects_unknown_separator_keys() -> None:
    # `retries` is valid for a detector http-client config but not for the
    # separator (only url/timeout/binary), so it must be rejected — before any
    # connection is attempted.
    config = DenoisingDetectorConfig(
        detector={"url": "http://d:1"},
        separator={"url": "http://s:2", "retries": 3},
    )
    with pytest.raises(ValueError, match="Unknown separator config key"):
        DenoisingDetector.from_config(config)


def test_from_config_requires_separator_url() -> None:
    config = DenoisingDetectorConfig(detector={"url": "http://d:1"}, separator={"timeout": 5.0})
    with pytest.raises(ValueError, match="Separator config must contain 'url'"):
        DenoisingDetector.from_config(config)


def test_from_config_closes_detector_when_separator_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import sound_event_detection.adapters.dispatch as dispatch_mod
    import sound_event_detection.denoising.birdmixit_client as birdmixit_mod

    detector = FakeDetector()

    def failing_birdmixit(**kwargs: object) -> FakeSeparator:
        raise ConnectionError("separator server is down")

    monkeypatch.setattr(dispatch_mod, "detector_client_from_config", lambda cfg, labels=None: detector)
    monkeypatch.setattr(birdmixit_mod, "BirdMixItClient", failing_birdmixit)

    config = DenoisingDetectorConfig(detector={"url": "http://d:1"}, separator={"url": "http://s:2"})
    with pytest.raises(ConnectionError, match="separator server is down"):
        DenoisingDetector.from_config(config)

    assert detector.closed


def test_separate_and_detect_returns_whole_file_core() -> None:
    det = _make_detector(energetic_prob=0.7)
    audio = np.ones(_SEP_SR, dtype=np.float32)  # 1 s

    core = det.separate_and_detect(audio)

    assert isinstance(core, StemDetections)
    assert core.stems.shape == (_N_STEMS, _SEP_SR)  # stems stay at the separator rate
    assert core.stem_preds.shape == (_N_STEMS, _expected_frames(_SEP_SR), len(_LABELS))
    assert core.sample_rate == _SEP_SR
    assert core.frame_rate == _FRAMERATE
    assert core.labels == _LABELS
    # Only stem 0 carries signal, so only its predictions are energetic.
    np.testing.assert_allclose(core.stem_preds[0], 0.7, rtol=0, atol=1e-6)
    np.testing.assert_allclose(core.stem_preds[1:], 0.0, rtol=0, atol=1e-6)


def test_separate_and_detect_rejects_non_1d_audio() -> None:
    det = _make_detector()
    with pytest.raises(ValueError, match="1D"):
        det.separate_and_detect(np.zeros((1, _SEP_SR), dtype=np.float32))


def test_combined_is_max_over_stems() -> None:
    det = _make_detector(energetic_prob=0.6)
    core = det.separate_and_detect(np.ones(_SEP_SR, dtype=np.float32))

    combined = core.combined()

    assert combined.predictions.shape == (1, _expected_frames(_SEP_SR), len(_LABELS))
    # Max over stems: stem 0 is 0.6, the rest are 0.
    np.testing.assert_allclose(combined.predictions, 0.6, rtol=0, atol=1e-6)


def test_denoise_gates_and_recovers_stem_zero() -> None:
    det = _make_detector(energetic_prob=1.0)
    audio = np.linspace(0.1, 1.0, _SEP_SR, dtype=np.float32)
    core = det.separate_and_detect(audio)

    waveform = core.denoise(focal_idxs=[0], threshold=0.5)

    # Stem 0 carries the signal and its focal prob (1.0) clears the threshold,
    # so the denoised waveform is the input; other stems are silent.
    assert waveform.shape == (_SEP_SR,)
    np.testing.assert_allclose(waveform, audio, rtol=0, atol=1e-6)


def test_denoise_silences_below_threshold() -> None:
    det = _make_detector(energetic_prob=0.3)
    core = det.separate_and_detect(np.ones(_SEP_SR, dtype=np.float32))

    waveform = core.denoise(focal_idxs=[0], threshold=0.5)

    # Focal prob 0.3 < 0.5 everywhere -> gate is all zero -> silence.
    np.testing.assert_array_equal(waveform, np.zeros(_SEP_SR, dtype=np.float32))


def test_stem_pairs_yields_audio_and_preds_per_stem() -> None:
    det = _make_detector(energetic_prob=1.0)
    core = det.separate_and_detect(np.ones(_SEP_SR, dtype=np.float32))

    pairs = list(core.stem_pairs())

    assert len(pairs) == _N_STEMS
    for i, (audio, preds) in enumerate(pairs):
        np.testing.assert_array_equal(audio, core.stems[i])
        np.testing.assert_array_equal(preds, core.stem_preds[i])


def test_run_returns_framewise_output() -> None:
    det = _make_detector(energetic_prob=0.7)
    audio = np.ones((1, _SEP_SR), dtype=np.float32)

    output = det.run(audio)

    assert isinstance(output, DetectorOutput)
    assert output.frame_rate == _FRAMERATE
    assert output.class_names == _LABELS
    assert output.predictions.shape == (1, _expected_frames(_SEP_SR), len(_LABELS))
    np.testing.assert_allclose(output.predictions, 0.7, rtol=0, atol=1e-6)


def test_run_handles_a_batch_of_recordings() -> None:
    det = _make_detector(energetic_prob=0.5)
    audio = np.ones((3, _SEP_SR), dtype=np.float32)

    output = det.run(audio)

    assert output.predictions.shape == (3, _expected_frames(_SEP_SR), len(_LABELS))
    np.testing.assert_allclose(output.predictions, 0.5, rtol=0, atol=1e-6)


def test_run_rejects_non_2d_audio() -> None:
    det = _make_detector()
    with pytest.raises(ValueError, match="2D"):
        det.run(np.zeros(_SEP_SR, dtype=np.float32))


def test_run_as_classifier_takes_max_over_stems() -> None:
    det = _make_detector(energetic_prob=0.8)
    # Recording 0 carries signal (stem 0 energetic -> per-stem scores
    # [0.8, 0, 0, 0], max 0.8); recording 1 is silent (all stems 0).
    audio = np.stack([np.ones(_SEP_SR, dtype=np.float32), np.zeros(_SEP_SR, dtype=np.float32)])

    output = det.run_as_classifier(audio)

    assert isinstance(output, MultiLabelClassifierOutput)
    assert output.predictions.shape == (2, len(_LABELS))
    assert output.class_names == _LABELS
    np.testing.assert_allclose(output.predictions[0], 0.8, rtol=0, atol=1e-6)
    np.testing.assert_allclose(output.predictions[1], 0.0, rtol=0, atol=1e-6)


def test_run_as_classifier_stacks_stems_across_recordings() -> None:
    detector = FakeDetector()
    det = DenoisingDetector(detector, FakeSeparator())

    det.run_as_classifier(np.ones((3, _SEP_SR), dtype=np.float32), batch_size=8, overlap=0.25)

    # All 3 * 4 stems are stacked and chunked batch_size rows at a time
    # (8 + 4), instead of one underfilled call per recording, with kwargs
    # forwarded to every call.
    assert [call["shape"][0] for call in detector.classifier_calls] == [8, _N_STEMS]
    assert all(call["batch_size"] == 8 and call["overlap"] == 0.25 for call in detector.classifier_calls)


def test_run_as_classifier_default_batch_size_is_eight() -> None:
    detector = FakeDetector()
    det = DenoisingDetector(detector, FakeSeparator())

    det.run_as_classifier(np.ones((1, _SEP_SR), dtype=np.float32))

    assert detector.classifier_calls[-1]["batch_size"] == 8


def test_run_as_classifier_rejects_non_2d_audio() -> None:
    det = _make_detector()
    with pytest.raises(ValueError, match="2D"):
        det.run_as_classifier(np.zeros(_SEP_SR, dtype=np.float32))


def test_window_duration_and_server_config_compose() -> None:
    detector = FakeDetector()
    det = DenoisingDetector(detector, FakeSeparator(), threshold=0.3, resampling_method="torchaudio_kaiser_fast")

    assert det.window_duration == detector.window_duration
    assert det.server_config == {
        "type": "denoising_detector",
        "detector": detector.server_config,
        # A plain separator exposes no weight hash, so it fills in as None
        # ("nan" over JSON) — the fallback for a backend that can't identify itself.
        "separator": {"sample_rate": _SEP_SR, "n_stems": _N_STEMS, "weights_sha256": None},
        "threshold": 0.3,
        "resampling_method": "torchaudio_kaiser_fast",
    }


def test_server_config_surfaces_separator_weight_hash() -> None:
    separator = FakeSeparator()
    separator.server_config = {"sample_rate": _SEP_SR, "n_stems": _N_STEMS, "weights_sha256": "abc123"}
    det = DenoisingDetector(FakeDetector(), separator)

    assert det.server_config["separator"]["weights_sha256"] == "abc123"


def test_describe_summary_reports_composite_metadata() -> None:
    det = _make_detector()
    assert det.describe_summary() == {
        "n_labels": len(_LABELS),
        "sample_rate": _SEP_SR,
        "frame_rate": _FRAMERATE,
        "window_duration": 5.0,
        "n_stems": _N_STEMS,
    }


class ClosableFakeSeparator(FakeSeparator):
    """Separator variant that records `close` calls."""

    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_close_closes_detector_and_closable_separator() -> None:
    detector = FakeDetector()
    separator = ClosableFakeSeparator()
    det = DenoisingDetector(detector, separator)

    det.close()

    assert detector.closed
    assert separator.closed


def test_close_tolerates_separator_without_close() -> None:
    detector = FakeDetector()
    det = DenoisingDetector(detector, FakeSeparator())

    det.close()  # FakeSeparator has no close(); must not raise

    assert detector.closed


def _stem_detections_with_focal(focal_by_stem: np.ndarray) -> StemDetections:
    """Build a `StemDetections` whose focal (class 0) probs are `focal_by_stem`.

    `focal_by_stem` has shape ``(n_stems, frames)``; the other two classes are
    left at zero. Stem audio is irrelevant to `quality`, so it is zeroed.

    Returns
    -------
    StemDetections
        A core whose class-0 (focal) per-stem probabilities are `focal_by_stem`
        and whose other classes are zero.
    """
    n_stems, frames = focal_by_stem.shape
    stem_preds = np.zeros((n_stems, frames, len(_LABELS)), dtype=np.float32)
    stem_preds[:, :, 0] = focal_by_stem
    return StemDetections(
        stems=np.zeros((n_stems, 10), dtype=np.float32),
        stem_preds=stem_preds,
        frame_rate=_FRAMERATE,
        labels=list(_LABELS),
        sample_rate=_SEP_SR,
    )


def test_quality_min_over_gated_stems_and_nan_when_none_gated() -> None:
    # Frame 0: both stems clear 0.5 -> min(0.9, 0.7); frame 1: neither -> nan;
    # frame 2: both clear -> min(0.6, 0.6).
    focal = np.array([[0.9, 0.2, 0.6], [0.7, 0.4, 0.6]], dtype=np.float32)
    core = _stem_detections_with_focal(focal)

    detection_prob, n_gated = core.quality(focal_idxs=[0], threshold=0.5)

    assert detection_prob.shape == (3,)
    assert n_gated.shape == (3,)
    np.testing.assert_allclose(detection_prob[[0, 2]], [0.7, 0.6], rtol=0, atol=1e-6)
    assert np.isnan(detection_prob[1])
    np.testing.assert_array_equal(n_gated, [2.0, 0.0, 2.0])


def test_quality_counts_only_stems_over_threshold() -> None:
    # Frame 0: only stem 0 clears -> count 1, min is stem 0's prob.
    focal = np.array([[0.8, 0.9], [0.3, 0.95]], dtype=np.float32)
    core = _stem_detections_with_focal(focal)

    detection_prob, n_gated = core.quality(focal_idxs=[0], threshold=0.5)

    np.testing.assert_array_equal(n_gated, [1.0, 2.0])
    np.testing.assert_allclose(detection_prob, [0.8, 0.9], rtol=0, atol=1e-6)


def _stem_detections_with_two_focal(class0: np.ndarray, class1: np.ndarray) -> StemDetections:
    """Build a `StemDetections` with class-0 and class-1 per-stem probs set.

    Both arrays have shape ``(n_stems, frames)``; the remaining class is zero.
    Stem audio is irrelevant to `quality`, so it is zeroed.

    Returns
    -------
    StemDetections
        A core whose class-0 and class-1 per-stem probabilities are `class0`
        and `class1` and whose other class is zero.
    """
    n_stems, frames = class0.shape
    stem_preds = np.zeros((n_stems, frames, len(_LABELS)), dtype=np.float32)
    stem_preds[:, :, 0] = class0
    stem_preds[:, :, 1] = class1
    return StemDetections(
        stems=np.zeros((n_stems, 10), dtype=np.float32),
        stem_preds=stem_preds,
        frame_rate=_FRAMERATE,
        labels=list(_LABELS),
        sample_rate=_SEP_SR,
    )


def test_quality_union_gates_over_multiple_focal_classes() -> None:
    # One stem, two frames. Class 0 clears only on frame 0; class 1 only on
    # frame 1. A single-class gate sees one frame; the union sees both.
    class0 = np.array([[0.9, 0.1]], dtype=np.float32)
    class1 = np.array([[0.2, 0.8]], dtype=np.float32)
    core = _stem_detections_with_two_focal(class0, class1)

    _, n_gated_single = core.quality(focal_idxs=[0], threshold=0.5)
    np.testing.assert_array_equal(n_gated_single, [1.0, 0.0])

    # Union of classes 0 and 1: both frames gated; per-frame prob is the max
    # over the two classes for the (single) gated-in stem.
    detection_prob, n_gated_union = core.quality(focal_idxs=[0, 1], threshold=0.5)
    np.testing.assert_array_equal(n_gated_union, [1.0, 1.0])
    np.testing.assert_allclose(detection_prob, [0.9, 0.8], rtol=0, atol=1e-6)


def test_fake_separator_satisfies_source_separator_protocol() -> None:
    assert isinstance(FakeSeparator(), SourceSeparatorClient)
