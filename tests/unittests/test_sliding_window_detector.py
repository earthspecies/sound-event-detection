"""Unit tests for SlidingWindowDetector.

Tests the core sliding window, activation, averaging, and label remapping logic
using mock classifiers (no TF Hub / real models needed).
"""

import numpy as np
import pytest
import torch

from esp_research.protocols.classifier import MultiLabelClassifierOutput
from esp_research.protocols.detector import Detector, DetectorOutput
from sound_event_detection.models.sliding_window_detector import SlidingWindowDetector
from sound_event_detection.utils.reformatters import detector_output_to_dataframe

# ============= Fixtures =============


def _make_constant_classifier(n_classes: int = 3, logit_value: float = 0.0):
    """Classifier that returns constant logits for every window."""

    def classify_fn(audio: torch.Tensor) -> torch.Tensor:
        batch_size = audio.shape[0]
        return torch.full((batch_size, n_classes), logit_value)

    return classify_fn


def _make_per_class_classifier(logits_per_class: list[float]):
    """Classifier that returns fixed per-class logits for every window.

    Useful for testing label remapping: each class gets a distinct value
    so we can verify which columns end up in the output.
    """
    t = torch.tensor(logits_per_class, dtype=torch.float32)

    def classify_fn(audio: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(0).expand(audio.shape[0], -1)

    return classify_fn


def _make_identity_classifier():
    """Classifier where logit[i] = window_index. Useful for testing averaging.

    Returns a classify_fn and a list to track call count.
    """
    call_count = [0]

    def classify_fn(audio: torch.Tensor) -> torch.Tensor:
        batch_size = audio.shape[0]
        logits = []
        for _ in range(batch_size):
            # Each window gets a unique value based on call order
            logits.append(torch.full((3,), float(call_count[0])))
            call_count[0] += 1
        return torch.stack(logits)

    return classify_fn, call_count


@pytest.fixture
def detector_no_overlap() -> SlidingWindowDetector:
    """Detector with hop_size == window_size (no overlap)."""
    return SlidingWindowDetector(
        classify_fn=_make_constant_classifier(n_classes=3, logit_value=0.0),
        classifier_labels=["a", "b", "c"],
        labels=["a", "b", "c"],
        sample_rate=100,
        window_size=1.0,
        hop_size=1.0,
    )


@pytest.fixture
def detector_with_overlap() -> SlidingWindowDetector:
    """Detector with hop_size = 0.5 * window_size (50% overlap)."""
    return SlidingWindowDetector(
        classify_fn=_make_constant_classifier(n_classes=3, logit_value=0.0),
        classifier_labels=["a", "b", "c"],
        labels=["a", "b", "c"],
        sample_rate=100,
        window_size=1.0,
        hop_size=0.5,
    )


# ============= Protocol conformance =============


def test_satisfies_detector_protocol(detector_no_overlap):
    """SlidingWindowDetector must satisfy the Detector protocol."""
    assert isinstance(detector_no_overlap, Detector)


def test_has_required_attributes(detector_no_overlap):
    """Check labels, sample_rate, frame_rate are set correctly."""
    assert detector_no_overlap.labels == ["a", "b", "c"]
    assert detector_no_overlap.sample_rate == 100
    assert detector_no_overlap.frame_rate == 1.0  # 1 / hop_size


def test_server_config_composes_classifier_identity():
    """`server_config` folds in the wrapped classifier's identity, SHA and all."""
    detector = SlidingWindowDetector(
        classify_fn=_make_constant_classifier(n_classes=3, logit_value=0.0),
        classifier_labels=["a", "b", "c"],
        labels=["a", "b", "c"],
        sample_rate=100,
        window_size=1.0,
        hop_size=1.0,
        detector_type="perch2",
        classifier_server_config={
            "type": "perch2_backend",
            "labels": ["a", "b", "c"],
            "sample_rate": 32000,
            "weights_sha256": "cafef00d",
        },
    )
    assert detector.server_config == {
        "type": "perch2",
        "window_size": 1.0,
        "hop_size": 1.0,
        "analysis_window": 1.0,
        "classifier": {
            "type": "perch2_backend",
            "weights_sha256": "cafef00d",
            "git_commit": None,
            "sample_rate": 32000,
            "n_labels": 3,
        },
    }


def test_server_config_fills_none_when_classifier_identity_absent():
    """A backend that only reports labels/sample_rate leaves SHA/commit as None."""
    detector = SlidingWindowDetector(
        classify_fn=_make_constant_classifier(n_classes=3, logit_value=0.0),
        classifier_labels=["a", "b", "c"],
        labels=["a", "b", "c"],
        sample_rate=100,
        window_size=1.0,
        hop_size=1.0,
    )
    assert detector.server_config["type"] is None
    assert detector.server_config["classifier"] == {
        "type": None,
        "weights_sha256": None,
        "git_commit": None,
        "sample_rate": None,
        "n_labels": None,
    }


# ============= Validation =============


def test_rejects_hop_greater_than_window():
    with pytest.raises(ValueError, match="hop_size.*must be <= analysis_window"):
        SlidingWindowDetector(
            classify_fn=_make_constant_classifier(),
            classifier_labels=["a", "b", "c"],
            labels=["a", "b", "c"],
            sample_rate=100,
            window_size=1.0,
            hop_size=2.0,
        )


def test_rejects_invalid_activation():
    with pytest.raises(ValueError, match="activation"):
        SlidingWindowDetector(
            classify_fn=_make_constant_classifier(),
            classifier_labels=["a", "b", "c"],
            labels=["a", "b", "c"],
            sample_rate=100,
            window_size=1.0,
            hop_size=1.0,
            activation="relu",
        )


def test_rejects_1d_audio(detector_no_overlap):
    with pytest.raises(ValueError, match="Expected 2D audio"):
        detector_no_overlap.run(np.zeros(100))


# ============= Output shape and type =============


def test_output_is_detector_output(detector_no_overlap):
    audio = np.zeros((1, 300), dtype=np.float32)  # 3 seconds at sr=100
    result = detector_no_overlap.run(audio)
    assert isinstance(result, DetectorOutput)


def test_output_shape_no_overlap(detector_no_overlap):
    """3 seconds of audio, 1s window, 1s hop → 3 frames."""
    audio = np.zeros((1, 300), dtype=np.float32)
    result = detector_no_overlap.run(audio)
    assert result.predictions.shape == (1, 3, 3)
    assert result.frame_rate == 1.0


def test_output_shape_with_overlap(detector_with_overlap):
    """3 seconds of audio, 1s window, 0.5s hop → 5 frames."""
    audio = np.zeros((1, 300), dtype=np.float32)
    result = detector_with_overlap.run(audio)
    assert result.predictions.shape == (1, 5, 3)
    assert result.frame_rate == 2.0


def test_output_shape_short_audio():
    """Audio shorter than one window → padded, still 1 frame."""
    detector = SlidingWindowDetector(
        classify_fn=_make_constant_classifier(),
        classifier_labels=["a", "b", "c"],
        labels=["a", "b", "c"],
        sample_rate=100,
        window_size=1.0,
        hop_size=1.0,
    )
    audio = np.zeros((1, 50), dtype=np.float32)  # 0.5 seconds
    result = detector.run(audio)
    assert result.predictions.shape == (1, 1, 3)


# ============= run_as_classifier (max pooling over time) =============


def _make_window_mean_classifier(n_classes: int = 3):
    """Classifier whose per-class logit is the window's mean amplitude.

    Deterministic in the input (not call order), so ``run`` and
    ``run_as_classifier`` agree, while varying across windows of different
    amplitude so max-over-time differs from mean-over-time.
    """

    def classify_fn(audio: torch.Tensor) -> torch.Tensor:
        means = audio.mean(dim=1, keepdim=True)
        return means.expand(audio.shape[0], n_classes)

    return classify_fn


def test_run_as_classifier_returns_classifier_output(detector_no_overlap):
    audio = np.zeros((1, 300), dtype=np.float32)
    result = detector_no_overlap.run_as_classifier(audio)
    assert isinstance(result, MultiLabelClassifierOutput)


def test_run_as_classifier_shape_and_labels(detector_no_overlap):
    """Clip-level predictions are (batch, classes) with the output labels."""
    audio = np.zeros((2, 300), dtype=np.float32)
    result = detector_no_overlap.run_as_classifier(audio)
    assert result.predictions.shape == (2, 3)
    assert result.class_names == ["a", "b", "c"]


def test_run_as_classifier_max_pools_over_time():
    """Clip score is the max frame probability over time."""
    detector = SlidingWindowDetector(
        classify_fn=_make_window_mean_classifier(),
        classifier_labels=["a", "b", "c"],
        labels=["a", "b", "c"],
        sample_rate=100,
        window_size=1.0,
        hop_size=1.0,
        activation="sigmoid",
    )
    # Three 1s windows of increasing amplitude → increasing frame probabilities.
    audio = np.concatenate(
        [np.full((1, 100), 0.1), np.full((1, 100), 0.5), np.full((1, 100), 0.9)], axis=1
    ).astype(np.float32)

    frame_out = detector.run(audio)
    clip_out = detector.run_as_classifier(audio)

    np.testing.assert_array_almost_equal(clip_out.predictions, frame_out.predictions.max(axis=1))
    # The largest-amplitude (last) window dominates the max.
    np.testing.assert_array_almost_equal(clip_out.predictions[0], frame_out.predictions[0, -1])


def test_run_as_classifier_rejects_1d_audio(detector_no_overlap):
    with pytest.raises(ValueError, match="Expected 2D audio"):
        detector_no_overlap.run_as_classifier(np.zeros(100))


# ============= Softmax activation =============


def test_softmax_produces_valid_probabilities():
    """Softmax output should sum to 1 and be in [0, 1]."""
    detector = SlidingWindowDetector(
        classify_fn=_make_constant_classifier(n_classes=4, logit_value=1.0),
        classifier_labels=["a", "b", "c", "d"],
        labels=["a", "b", "c", "d"],
        sample_rate=100,
        window_size=1.0,
        hop_size=1.0,
        activation="softmax",
    )
    audio = np.zeros((1, 100), dtype=np.float32)
    result = detector.run(audio)
    row = result.predictions[0, 0]
    np.testing.assert_allclose(row.sum(), 1.0, atol=1e-6)
    assert np.all(row >= 0) and np.all(row <= 1)


def test_sigmoid_produces_valid_probabilities():
    """Sigmoid output should be in [0, 1] but NOT sum to 1."""
    detector = SlidingWindowDetector(
        classify_fn=_make_constant_classifier(n_classes=3, logit_value=0.0),
        classifier_labels=["a", "b", "c"],
        labels=["a", "b", "c"],
        sample_rate=100,
        window_size=1.0,
        hop_size=1.0,
        activation="sigmoid",
    )
    audio = np.zeros((1, 100), dtype=np.float32)
    result = detector.run(audio)
    row = result.predictions[0, 0]
    assert np.all(row >= 0) and np.all(row <= 1)
    np.testing.assert_allclose(row, 0.5, atol=1e-6)  # sigmoid(0) = 0.5


# ============= Overlap averaging =============


def test_overlap_averaging_values():
    """With 50% overlap, middle frames should average two windows' predictions."""
    classify_fn, call_count = _make_identity_classifier()
    detector = SlidingWindowDetector(
        classify_fn=classify_fn,
        classifier_labels=["a", "b", "c"],
        labels=["a", "b", "c"],
        sample_rate=100,
        window_size=1.0,
        hop_size=0.5,
        activation="sigmoid",  # sigmoid so values stay distinct
    )
    # 2 seconds → windows at [0,1), [0.5,1.5), [1,2) → 3 windows, 3 frames
    audio = np.zeros((1, 200), dtype=np.float32)
    result = detector.run(audio)

    # Window 0 → logit 0 → sigmoid(0)=0.5
    # Window 1 → logit 1 → sigmoid(1)≈0.7311
    # Window 2 → logit 2 → sigmoid(2)≈0.8808
    # frames_per_window = 1.0/0.5 = 2
    # Frame 0: covered by window 0 only → sigmoid(0) = 0.5
    # Frame 1: covered by windows 0 and 1 → avg(sigmoid(0), sigmoid(1))
    # Frame 2: covered by windows 1 and 2 → avg(sigmoid(1), sigmoid(2))
    preds = result.predictions[0]
    # Implementation averages in logit space, then applies sigmoid once.
    # sigmoid(avg(logits)) != avg(sigmoid(logits)) due to nonlinearity.
    np.testing.assert_allclose(preds[0, 0], 1 / (1 + np.exp(-0.0)), atol=1e-4)  # sigmoid(avg(0))
    np.testing.assert_allclose(preds[1, 0], 1 / (1 + np.exp(-0.5)), atol=1e-4)  # sigmoid(avg(0, 1))
    np.testing.assert_allclose(preds[2, 0], 1 / (1 + np.exp(-1.5)), atol=1e-4)  # sigmoid(avg(1, 2))


def test_no_overlap_no_averaging():
    """With hop_size == window_size, each frame gets one window's prediction."""
    classify_fn, _ = _make_identity_classifier()
    detector = SlidingWindowDetector(
        classify_fn=classify_fn,
        classifier_labels=["a", "b", "c"],
        labels=["a", "b", "c"],
        sample_rate=100,
        window_size=1.0,
        hop_size=1.0,
        activation="sigmoid",
    )
    audio = np.zeros((1, 200), dtype=np.float32)
    result = detector.run(audio)
    preds = result.predictions[0]

    sig0 = 1 / (1 + np.exp(0))
    sig1 = 1 / (1 + np.exp(-1))
    np.testing.assert_allclose(preds[0, 0], sig0, atol=1e-4)
    np.testing.assert_allclose(preds[1, 0], sig1, atol=1e-4)

# ============= Framerate =============


def test_frame_rate_matches_hop_size():
    for hop_size in [0.25, 0.5, 1.0, 2.5]:
        detector = SlidingWindowDetector(
            classify_fn=_make_constant_classifier(),
            classifier_labels=["a", "b", "c"],
            labels=["a", "b", "c"],
            sample_rate=100,
            window_size=5.0,
            hop_size=hop_size,
        )
        assert detector.frame_rate == pytest.approx(1.0 / hop_size)


# ============= Label remapping =============


def test_remap_identity():
    """When labels == classifier_labels, output matches directly."""
    detector = SlidingWindowDetector(
        classify_fn=_make_per_class_classifier([1.0, 2.0, 3.0]),
        classifier_labels=["a", "b", "c"],
        labels=["a", "b", "c"],
        sample_rate=100,
        window_size=1.0,
        hop_size=1.0,
        activation="sigmoid",
    )
    result = detector.run(np.zeros((1, 100), dtype=np.float32))
    df = detector_output_to_dataframe(result)
    row = df.iloc[0]
    assert list(df.columns) == ["a", "b", "c"]
    # sigmoid(1) < sigmoid(2) < sigmoid(3)
    assert row["a"] < row["b"] < row["c"]


def test_remap_labels_none_defaults_to_classifier_labels():
    """When labels=None, output uses classifier_labels."""
    detector = SlidingWindowDetector(
        classify_fn=_make_per_class_classifier([1.0, 2.0]),
        classifier_labels=["x", "y"],
        labels=None,
        sample_rate=100,
        window_size=1.0,
        hop_size=1.0,
        activation="sigmoid",
    )
    assert detector.labels == ["x", "y"]
    result = detector.run(np.zeros((1, 100), dtype=np.float32))
    assert result.class_names == ["x", "y"]


def test_remap_subset_reorders():
    """Output labels are a reordered subset of classifier labels."""
    detector = SlidingWindowDetector(
        classify_fn=_make_per_class_classifier([1.0, 2.0, 3.0]),
        classifier_labels=["a", "b", "c"],
        labels=["c", "a"],
        sample_rate=100,
        window_size=1.0,
        hop_size=1.0,
        activation="sigmoid",
    )
    result = detector.run(np.zeros((1, 100), dtype=np.float32))
    df = detector_output_to_dataframe(result)
    row = df.iloc[0]
    assert list(df.columns) == ["c", "a"]
    # "c" column should have sigmoid(3), "a" should have sigmoid(1)
    expected_c = 1 / (1 + np.exp(-3.0))
    expected_a = 1 / (1 + np.exp(-1.0))
    np.testing.assert_allclose(row["c"], expected_c, atol=1e-6)
    np.testing.assert_allclose(row["a"], expected_a, atol=1e-6)


def test_remap_missing_label_gets_zero():
    """A label not in classifier_labels gets zero probability."""
    detector = SlidingWindowDetector(
        classify_fn=_make_per_class_classifier([1.0, 2.0]),
        classifier_labels=["a", "b"],
        labels=["a", "b", "missing"],
        sample_rate=100,
        window_size=1.0,
        hop_size=1.0,
        activation="sigmoid",
    )
    result = detector.run(np.zeros((1, 100), dtype=np.float32))
    row = detector_output_to_dataframe(result).iloc[0]
    assert row["missing"] == 0.0
    assert row["a"] > 0.0


def test_remap_duplicate_classifier_labels_takes_max():
    """When classifier has duplicate labels, output takes the max probability."""
    # Classifier has "a" at indices 0 and 2 with different logits
    detector = SlidingWindowDetector(
        classify_fn=_make_per_class_classifier([1.0, 2.0, 5.0]),
        classifier_labels=["a", "b", "a"],
        labels=["a", "b"],
        sample_rate=100,
        window_size=1.0,
        hop_size=1.0,
        activation="sigmoid",
    )
    result = detector.run(np.zeros((1, 100), dtype=np.float32))
    row = detector_output_to_dataframe(result).iloc[0]
    # "a" maps to classifier columns 0 (logit=1) and 2 (logit=5)
    # After sigmoid: sigmoid(1)~0.731, sigmoid(5)~0.993 -> max is sigmoid(5)
    expected_a = 1 / (1 + np.exp(-5.0))
    expected_b = 1 / (1 + np.exp(-2.0))
    np.testing.assert_allclose(row["a"], expected_a, atol=1e-6)
    np.testing.assert_allclose(row["b"], expected_b, atol=1e-6)


def test_remap_all_missing():
    """When no output labels match the classifier, all outputs are zero."""
    detector = SlidingWindowDetector(
        classify_fn=_make_per_class_classifier([1.0, 2.0]),
        classifier_labels=["a", "b"],
        labels=["x", "y", "z"],
        sample_rate=100,
        window_size=1.0,
        hop_size=1.0,
        activation="sigmoid",
    )
    result = detector.run(np.zeros((1, 100), dtype=np.float32))
    np.testing.assert_array_equal(result.predictions, 0.0)


# ============= Analysis window (sub-window padding) =============


def _make_shape_recording_classifier(n_classes: int = 3):
    """Classifier that records the shape of each input batch for verification."""
    shapes = []

    def classify_fn(audio: torch.Tensor) -> torch.Tensor:
        shapes.append(audio.shape)
        return torch.zeros(audio.shape[0], n_classes)

    return classify_fn, shapes


def test_analysis_window_default_equals_window_size():
    """When analysis_window is None, it defaults to window_size."""
    detector = SlidingWindowDetector(
        classify_fn=_make_constant_classifier(),
        classifier_labels=["a", "b", "c"],
        labels=["a", "b", "c"],
        sample_rate=100,
        window_size=5.0,
        hop_size=5.0,
    )
    assert detector.analysis_window == 5.0


def test_analysis_window_pads_to_classifier_size():
    """Clips should be padded to window_size samples even when analysis_window is smaller."""
    classify_fn, shapes = _make_shape_recording_classifier()
    detector = SlidingWindowDetector(
        classify_fn=classify_fn,
        classifier_labels=["a", "b", "c"],
        labels=["a", "b", "c"],
        sample_rate=100,
        window_size=5.0,
        hop_size=1.0,
        analysis_window=1.0,
    )
    audio = np.zeros((1, 300), dtype=np.float32)  # 3 seconds
    detector.run(audio)
    # Each batch should have samples == window_size * sample_rate = 500
    for shape in shapes:
        assert shape[1] == 500  # classifier sees 5s of audio


def test_analysis_window_higher_frame_rate():
    """analysis_window < window_size should produce more frames."""
    detector = SlidingWindowDetector(
        classify_fn=_make_constant_classifier(),
        classifier_labels=["a", "b", "c"],
        labels=["a", "b", "c"],
        sample_rate=100,
        window_size=5.0,
        hop_size=1.0,
        analysis_window=1.0,
    )
    assert detector.frame_rate == 1.0
    audio = np.zeros((1, 500), dtype=np.float32)  # 5 seconds
    result = detector.run(audio)
    # 5s audio, 1s analysis window, 1s hop → 5 frames
    assert result.predictions.shape[1] == 5


def test_analysis_window_left_pads_with_zeros():
    """The padding should be on the left (audio at the end of the classifier window)."""

    def classify_fn(audio: torch.Tensor) -> torch.Tensor:
        # Return the mean of the first half and second half as "logits"
        n = audio.shape[1]
        left_mean = audio[:, : n // 2].mean(dim=1, keepdim=True)
        right_mean = audio[:, n // 2 :].mean(dim=1, keepdim=True)
        return torch.cat([left_mean, right_mean], dim=1)

    detector = SlidingWindowDetector(
        classify_fn=classify_fn,
        classifier_labels=["left", "right"],
        labels=["left", "right"],
        sample_rate=100,
        window_size=2.0,  # classifier expects 2s = 200 samples
        hop_size=1.0,
        analysis_window=1.0,  # we extract 1s, left-pad 1s of zeros
        activation="sigmoid",
    )
    # Audio of all ones
    audio = np.ones((1, 100), dtype=np.float32)  # 1 second
    result = detector.run(audio)
    row = detector_output_to_dataframe(result).iloc[0]
    # Left half = zeros (padding), right half = ones (audio)
    # sigmoid(0) = 0.5, sigmoid(1) ≈ 0.731
    np.testing.assert_allclose(row["left"], 0.5, atol=1e-6)
    assert row["right"] > 0.6


def test_analysis_window_rejects_greater_than_window_size():
    with pytest.raises(ValueError, match="analysis_window.*must be <= window_size"):
        SlidingWindowDetector(
            classify_fn=_make_constant_classifier(),
            classifier_labels=["a", "b", "c"],
            labels=["a", "b", "c"],
            sample_rate=100,
            window_size=1.0,
            hop_size=1.0,
            analysis_window=2.0,
        )


def test_analysis_window_rejects_hop_greater_than_analysis():
    with pytest.raises(ValueError, match="hop_size.*must be <= analysis_window"):
        SlidingWindowDetector(
            classify_fn=_make_constant_classifier(),
            classifier_labels=["a", "b", "c"],
            labels=["a", "b", "c"],
            sample_rate=100,
            window_size=5.0,
            hop_size=2.0,
            analysis_window=1.0,
        )
