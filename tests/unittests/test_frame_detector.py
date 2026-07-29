"""Tests for FrameDetector and create_beats_detector factory function.

Tests the FrameDetector class which combines an audio encoder with a linear
classification head for frame-level sound event detection.
"""

import numpy as np
import pytest
import torch

from esp_research.protocols.classifier import MultiLabelClassifierOutput
from sound_event_detection.models.encoders import (
    BEATSEncoder,
    BEATSEncoderConfig,
    compute_beats_frame_rate,
)
from sound_event_detection.models.frame_detector import (
    FrameDetector,
    create_beats_detector,
)
from sound_event_detection.utils.pooling import tempered_pooling
from sound_event_detection.utils.reformatters import detector_output_to_dataframe

# --- Constants ---

ENCODER_NAME = "esp_aves2_sl_beats_all"
TEST_LABELS = ["species_1", "species_2", "species_3"]
SAMPLE_RATE = 32000
WINDOW_DURATION = 5.0


# --- Fixtures ---


@pytest.fixture(scope="module")
def raw_beats_model():
    """Load the raw BEATs model."""
    from avex import load_model

    return load_model(ENCODER_NAME, return_features_only=True, device="cpu")


@pytest.fixture(scope="module")
def beats_encoder(raw_beats_model):
    """Create a BEATSEncoder."""
    return BEATSEncoder(
        model=raw_beats_model,
        sample_rate=SAMPLE_RATE,
        window_duration=WINDOW_DURATION,
        aggregation="average",
    )


@pytest.fixture(scope="module")
def frame_detector(beats_encoder):
    """Create a FrameDetector with BEATSEncoder."""
    detector = FrameDetector(encoder=beats_encoder, labels=TEST_LABELS)
    detector.eval()
    return detector


@pytest.fixture(scope="module")
def detector_from_factory():
    """Create a FrameDetector using the factory function."""
    detector = create_beats_detector(
        encoder_name=ENCODER_NAME,
        labels=TEST_LABELS,
        sample_rate=SAMPLE_RATE,
        window_duration=WINDOW_DURATION,
        aggregation="average",
    )
    detector.eval()
    return detector


# --- Tests for FrameDetector Initialization ---


class TestFrameDetectorInit:
    """Tests for FrameDetector initialization."""

    def test_has_encoder(self, frame_detector):
        """Test that detector has encoder attribute."""
        assert hasattr(frame_detector, "encoder")
        assert isinstance(frame_detector.encoder, torch.nn.Module)

    def test_has_classifier(self, frame_detector):
        """Test that detector has classifier attribute."""
        assert hasattr(frame_detector, "classifier")
        assert isinstance(frame_detector.classifier, torch.nn.Linear)

    def test_classifier_output_size(self, frame_detector):
        """Test classifier output size matches number of labels."""
        assert frame_detector.classifier.out_features == len(TEST_LABELS)

    def test_classifier_input_size(self, frame_detector):
        """Test classifier input size matches encoder output dim."""
        assert frame_detector.classifier.in_features == frame_detector.encoder.output_dim

    def test_labels_stored(self, frame_detector):
        """Test that labels are stored correctly."""
        assert frame_detector.labels == TEST_LABELS

    def test_classifier_bias_initialized(self, frame_detector):
        """Test that classifier bias is initialized for low prior probability."""
        # With prior_prob=0.01, bias should be negative (sigmoid → ~0.01)
        assert (frame_detector.classifier.bias.data < 0).all()

        # Check that sigmoid of bias gives approximately 0.01
        prior_prob = torch.sigmoid(frame_detector.classifier.bias.data).mean().item()
        assert 0.005 < prior_prob < 0.02


# --- Tests for FrameDetector Properties ---


class TestFrameDetectorProperties:
    """Tests for FrameDetector property delegation to encoder."""

    def test_frame_rate_property(self, frame_detector):
        """Test frame_rate property delegates to encoder."""
        assert frame_detector.frame_rate == frame_detector.encoder.output_frame_rate
        assert frame_detector.frame_rate == compute_beats_frame_rate(SAMPLE_RATE, WINDOW_DURATION)

    def test_sample_rate_property(self, frame_detector):
        """Test sample_rate property delegates to encoder."""
        assert frame_detector.sample_rate == frame_detector.encoder.sample_rate
        assert frame_detector.sample_rate == SAMPLE_RATE

    def test_window_duration_property(self, frame_detector):
        """Test window_duration property delegates to encoder."""
        assert frame_detector.window_duration == frame_detector.encoder.window_duration
        assert frame_detector.window_duration == WINDOW_DURATION


# --- Tests for FrameDetector Forward Pass ---


class TestFrameDetectorForward:
    """Tests for FrameDetector forward pass."""

    def test_forward_output_shape(self, frame_detector):
        """Test forward pass output shape."""
        batch_size = 2
        window_samples = int(WINDOW_DURATION * SAMPLE_RATE)
        audio = torch.randn(batch_size, window_samples)

        with torch.no_grad():
            output = frame_detector(audio)

        assert output.ndim == 3
        assert output.shape[0] == batch_size
        assert output.shape[1] == 62  # frames for 5s at 32kHz
        assert output.shape[2] == len(TEST_LABELS)

    def test_forward_returns_logits(self, frame_detector):
        """Test that forward returns logits (unbounded values)."""
        window_samples = int(WINDOW_DURATION * SAMPLE_RATE)
        audio = torch.randn(1, window_samples)

        with torch.no_grad():
            output = frame_detector(audio)

        # Logits can be any value, not bounded to [0, 1]
        assert torch.isfinite(output).all()
        # With initialized bias, logits should be mostly negative
        assert output.mean() < 0

    def test_forward_batch_consistency(self, frame_detector):
        """Test forward pass is consistent across batch."""
        window_samples = int(WINDOW_DURATION * SAMPLE_RATE)
        single_audio = torch.randn(1, window_samples)
        batch_audio = single_audio.repeat(3, 1)

        with torch.no_grad():
            single_output = frame_detector(single_audio)
            batch_output = frame_detector(batch_audio)

        for i in range(3):
            torch.testing.assert_close(batch_output[i], single_output[0], rtol=1e-5, atol=1e-5)

    def test_forward_different_audio_different_output(self, frame_detector):
        """Test that different audio produces different output."""
        window_samples = int(WINDOW_DURATION * SAMPLE_RATE)
        audio1 = torch.randn(1, window_samples)
        audio2 = torch.randn(1, window_samples)

        with torch.no_grad():
            output1 = frame_detector(audio1)
            output2 = frame_detector(audio2)

        assert not torch.allclose(output1, output2)


# --- Tests for create_beats_detector Factory ---


class TestCreateBeatsDetector:
    """Tests for create_beats_detector factory function."""

    def test_creates_frame_detector(self, detector_from_factory):
        """Test that factory creates a FrameDetector."""
        assert isinstance(detector_from_factory, FrameDetector)

    def test_creates_beats_encoder(self, detector_from_factory):
        """Test that factory creates a BEATSEncoder."""
        assert isinstance(detector_from_factory.encoder, BEATSEncoder)

    def test_correct_labels(self, detector_from_factory):
        """Test that factory sets labels correctly."""
        assert detector_from_factory.labels == TEST_LABELS

    def test_correct_sample_rate(self, detector_from_factory):
        """Test that factory sets sample_rate correctly."""
        assert detector_from_factory.sample_rate == SAMPLE_RATE

    def test_correct_window_duration(self, detector_from_factory):
        """Test that factory sets window_duration correctly."""
        assert detector_from_factory.window_duration == WINDOW_DURATION

    def test_correct_frame_rate(self, detector_from_factory):
        """Test that factory computes frame_rate correctly."""
        expected = compute_beats_frame_rate(SAMPLE_RATE, WINDOW_DURATION)
        assert detector_from_factory.frame_rate == expected

    def test_invalid_aggregation_raises(self):
        """Test that invalid aggregation raises ValueError."""
        with pytest.raises(ValueError, match="Unknown aggregation strategy"):
            create_beats_detector(
                encoder_name=ENCODER_NAME,
                labels=TEST_LABELS,
                sample_rate=SAMPLE_RATE,
                aggregation="invalid",
            )


class TestCreateBeatsDetectorAggregations:
    """Tests for create_beats_detector with different aggregation strategies."""

    @pytest.fixture(scope="class")
    def average_detector(self):
        """Create detector with average aggregation."""
        return create_beats_detector(
            encoder_name=ENCODER_NAME,
            labels=TEST_LABELS,
            sample_rate=SAMPLE_RATE,
            aggregation="average",
        )

    @pytest.fixture(scope="class")
    def all_frames_detector(self):
        """Create detector with all_frames aggregation."""
        return create_beats_detector(
            encoder_name=ENCODER_NAME,
            labels=TEST_LABELS,
            sample_rate=SAMPLE_RATE,
            aggregation="all_frames",
        )

    @pytest.fixture(scope="class")
    def concat_detector(self):
        """Create detector with concat aggregation."""
        return create_beats_detector(
            encoder_name=ENCODER_NAME,
            labels=TEST_LABELS,
            sample_rate=SAMPLE_RATE,
            aggregation="concat",
        )

    def test_average_frame_rate(self, average_detector):
        """Test average aggregation frame_rate."""
        expected = compute_beats_frame_rate(SAMPLE_RATE, WINDOW_DURATION)
        assert average_detector.frame_rate == expected

    def test_all_frames_frame_rate(self, all_frames_detector):
        """Test all_frames aggregation has 8x frame_rate."""
        base = compute_beats_frame_rate(SAMPLE_RATE, WINDOW_DURATION)
        assert all_frames_detector.frame_rate == base * BEATSEncoderConfig.NUM_FREQ_PATCHES

    def test_concat_frame_rate(self, concat_detector):
        """Test concat aggregation frame_rate."""
        expected = compute_beats_frame_rate(SAMPLE_RATE, WINDOW_DURATION)
        assert concat_detector.frame_rate == expected

    def test_average_classifier_input(self, average_detector):
        """Test average aggregation classifier input size."""
        assert average_detector.classifier.in_features == BEATSEncoderConfig.HIDDEN_DIM

    def test_all_frames_classifier_input(self, all_frames_detector):
        """Test all_frames aggregation classifier input size."""
        assert all_frames_detector.classifier.in_features == BEATSEncoderConfig.HIDDEN_DIM

    def test_concat_classifier_input(self, concat_detector):
        """Test concat aggregation has 8x classifier input."""
        expected = BEATSEncoderConfig.HIDDEN_DIM * BEATSEncoderConfig.NUM_FREQ_PATCHES
        assert concat_detector.classifier.in_features == expected

    def test_average_output_shape(self, average_detector):
        """Test average aggregation output shape."""
        audio = torch.randn(1, int(WINDOW_DURATION * SAMPLE_RATE))
        with torch.no_grad():
            output = average_detector(audio)
        assert output.shape == (1, 62, len(TEST_LABELS))

    def test_all_frames_output_shape(self, all_frames_detector):
        """Test all_frames aggregation output shape (8x frames)."""
        audio = torch.randn(1, int(WINDOW_DURATION * SAMPLE_RATE))
        with torch.no_grad():
            output = all_frames_detector(audio)
        assert output.shape == (1, 496, len(TEST_LABELS))

    def test_concat_output_shape(self, concat_detector):
        """Test concat aggregation output shape."""
        audio = torch.randn(1, int(WINDOW_DURATION * SAMPLE_RATE))
        with torch.no_grad():
            output = concat_detector(audio)
        assert output.shape == (1, 62, len(TEST_LABELS))


# --- Tests for Freeze/Unfreeze ---


class TestFrameDetectorFreezeUnfreeze:
    """Tests for freeze_encoder and unfreeze_encoder methods."""

    def test_freeze_encoder(self, frame_detector):
        """Test that freeze_encoder freezes encoder parameters."""
        frame_detector.unfreeze_encoder()
        assert all(p.requires_grad for p in frame_detector.encoder.parameters())

        frame_detector.freeze_encoder()

        assert not any(p.requires_grad for p in frame_detector.encoder.parameters())
        # Classifier should still be trainable
        assert all(p.requires_grad for p in frame_detector.classifier.parameters())

    def test_unfreeze_encoder(self, frame_detector):
        """Test that unfreeze_encoder unfreezes encoder parameters."""
        frame_detector.freeze_encoder()
        assert not any(p.requires_grad for p in frame_detector.encoder.parameters())

        frame_detector.unfreeze_encoder()

        assert all(p.requires_grad for p in frame_detector.encoder.parameters())


# --- Tests for run ---


class TestRunInferenceOnFile:
    """Tests for run method."""

    def test_inference_returns_detector_output(self, frame_detector):
        """Test that inference returns DetectorOutput."""
        from esp_research.protocols.detector import DetectorOutput

        audio = np.random.randn(1, 5 * SAMPLE_RATE).astype(np.float32)
        result = frame_detector.run(audio)

        assert isinstance(result, DetectorOutput)

    def test_inference_returns_class_names(self, frame_detector):
        """Test that inference returns predictions with correct class names."""
        audio = np.random.randn(1, 5 * SAMPLE_RATE).astype(np.float32)
        result = frame_detector.run(audio)

        assert result.class_names == TEST_LABELS
        assert list(detector_output_to_dataframe(result).columns) == TEST_LABELS

    def test_inference_returns_correct_frame_rate(self, frame_detector):
        """Test that inference returns correct frame_rate."""
        audio = np.random.randn(1, 5 * SAMPLE_RATE).astype(np.float32)
        result = frame_detector.run(audio)

        assert result.frame_rate == frame_detector.frame_rate

    def test_inference_on_one_window(self, frame_detector):
        """Test inference on exactly one window of audio."""
        audio = np.random.randn(1, int(WINDOW_DURATION * SAMPLE_RATE)).astype(np.float32)
        result = frame_detector.run(audio)

        assert result.predictions.shape == (1, 62, len(TEST_LABELS))

    def test_inference_on_short_audio(self, frame_detector):
        """Test inference on audio shorter than one window."""
        audio = np.random.randn(1, 3 * SAMPLE_RATE).astype(np.float32)  # 3 seconds
        result = frame_detector.run(audio)

        # Should have predictions (padded to window size, then trimmed)
        assert result.predictions.shape[1] > 0

    def test_inference_on_long_audio(self, frame_detector):
        """Test inference on audio spanning multiple windows."""
        duration = 12  # seconds
        audio = np.random.randn(1, duration * SAMPLE_RATE).astype(np.float32)
        result = frame_detector.run(audio)

        expected_frames = int(duration * frame_detector.frame_rate)
        assert result.predictions.shape[1] == expected_frames

    def test_inference_on_exact_multiple_windows(self, frame_detector):
        """Test inference on audio that's exactly N windows."""
        n_windows = 3
        duration = n_windows * WINDOW_DURATION
        audio = np.zeros((1, int(duration * SAMPLE_RATE)), dtype=np.float32)
        result = frame_detector.run(audio)

        expected_frames = n_windows * 62
        assert result.predictions.shape[1] == expected_frames

    def test_inference_returns_probabilities(self, frame_detector):
        """Test that inference returns probabilities in [0, 1]."""
        audio = np.random.randn(1, 5 * SAMPLE_RATE).astype(np.float32)
        result = frame_detector.run(audio)

        probs = result.predictions
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_inference_invalid_audio_shape(self, frame_detector):
        """Test that non-2D audio raises ValueError."""
        audio = np.random.randn(5 * SAMPLE_RATE).astype(np.float32)  # 1D

        with pytest.raises(ValueError, match="Expected 2D audio"):
            frame_detector.run(audio)

    def test_inference_batched_recordings(self, frame_detector):
        """Test inference on a batch of recordings stacks independently."""
        audio = np.random.randn(3, 5 * SAMPLE_RATE).astype(np.float32)
        result = frame_detector.run(audio)

        assert result.predictions.shape == (3, 62, len(TEST_LABELS))

        # Each row must match running that recording on its own.
        for i in range(3):
            single = frame_detector.run(audio[i : i + 1])
            np.testing.assert_array_almost_equal(result.predictions[i], single.predictions[0])

    def test_inference_with_overlap(self, frame_detector):
        """Test inference with overlapping windows."""
        audio = np.random.randn(1, 15 * SAMPLE_RATE).astype(np.float32)
        result = frame_detector.run(audio, overlap=0.5)

        # Should still produce correct number of frames
        expected_frames = int(15 * frame_detector.frame_rate)
        assert result.predictions.shape[1] == expected_frames

    def test_inference_overlap_invalid(self, frame_detector):
        """Test that invalid overlap raises ValueError."""
        audio = np.random.randn(1, 10 * SAMPLE_RATE).astype(np.float32)

        with pytest.raises(ValueError, match="overlap must be"):
            frame_detector.run(audio, overlap=1.5)

    def test_inference_batch_size(self, frame_detector):
        """Test inference with different batch sizes produces same result."""
        audio = np.random.randn(1, 20 * SAMPLE_RATE).astype(np.float32)

        result_batch1 = frame_detector.run(audio, batch_size=1)
        result_batch4 = frame_detector.run(audio, batch_size=4)

        np.testing.assert_array_almost_equal(
            result_batch1.predictions,
            result_batch4.predictions,
        )


# --- Tests for run_as_classifier ---


class TestRunAsClassifier:
    """Tests for the run_as_classifier method."""

    def test_returns_classifier_output(self, frame_detector):
        """run_as_classifier returns a MultiLabelClassifierOutput."""
        audio = np.random.randn(1, 5 * SAMPLE_RATE).astype(np.float32)
        result = frame_detector.run_as_classifier(audio)

        assert isinstance(result, MultiLabelClassifierOutput)

    def test_output_shape_and_class_names(self, frame_detector):
        """Predictions are clip-level (batch, classes) with the model's labels."""
        audio = np.random.randn(3, 5 * SAMPLE_RATE).astype(np.float32)
        result = frame_detector.run_as_classifier(audio)

        assert result.predictions.shape == (3, len(TEST_LABELS))
        assert result.class_names == TEST_LABELS

    def test_returns_probabilities(self, frame_detector):
        """Clip-level predictions are probabilities in [0, 1]."""
        audio = np.random.randn(2, 8 * SAMPLE_RATE).astype(np.float32)
        result = frame_detector.run_as_classifier(audio)

        assert (result.predictions >= 0).all()
        assert (result.predictions <= 1).all()

    def test_default_pooling_temperature(self, frame_detector):
        """A FrameDetector defaults to a pooling temperature of 1.0."""
        assert frame_detector.pooling_temperature == 1.0

    def test_pools_run_output_with_pooling_temperature(self, beats_encoder):
        """run_as_classifier equals tempered pooling of run() at the model's temperature."""
        detector = FrameDetector(encoder=beats_encoder, labels=TEST_LABELS, pooling_temperature=3.0)
        detector.eval()
        audio = np.random.randn(2, 6 * SAMPLE_RATE).astype(np.float32)

        clip = detector.run_as_classifier(audio)
        frames = detector.run(audio).predictions
        expected = tempered_pooling(torch.from_numpy(frames).float(), temperature=3.0, dim=1).numpy()

        assert detector.pooling_temperature == 3.0
        np.testing.assert_array_almost_equal(clip.predictions, expected)

    def test_invalid_audio_shape(self, frame_detector):
        """Non-2D audio raises ValueError (forwarded from run)."""
        audio = np.random.randn(5 * SAMPLE_RATE).astype(np.float32)  # 1D

        with pytest.raises(ValueError, match="Expected 2D audio"):
            frame_detector.run_as_classifier(audio)


# --- Tests for Timing Accuracy ---


class TestInferenceTimingAccuracy:
    """Tests for timing accuracy in inference."""

    def test_frame_count_for_5_minutes(self, frame_detector):
        """Test frame count is correct for 5 minutes of audio."""
        duration = 300  # 5 minutes
        audio = np.zeros((1, duration * SAMPLE_RATE), dtype=np.float32)
        result = frame_detector.run(audio)

        # 60 windows × 62 frames = 3720 frames
        expected_frames = 60 * 62
        assert result.predictions.shape[1] == expected_frames

    def test_last_frame_time_is_accurate(self, frame_detector):
        """Test that last frame corresponds to correct time."""
        duration = 300
        audio = np.zeros((1, duration * SAMPLE_RATE), dtype=np.float32)
        result = frame_detector.run(audio)

        n_frames = result.predictions.shape[1]
        last_frame_time = (n_frames - 1) / result.frame_rate

        # Should be just under 300s
        assert last_frame_time > 299.0
        assert last_frame_time < 300.0


# --- Tests for Gradient Flow ---


class TestGradientFlow:
    """Tests for gradient flow through the model."""

    def test_gradients_flow_to_classifier(self, beats_encoder):
        """Test that gradients flow to classifier."""
        detector = FrameDetector(encoder=beats_encoder, labels=TEST_LABELS)
        detector.train()

        audio = torch.randn(1, int(WINDOW_DURATION * SAMPLE_RATE))
        output = detector(audio)
        loss = output.sum()
        loss.backward()

        assert detector.classifier.weight.grad is not None
        assert detector.classifier.weight.grad.abs().sum() > 0

    def test_gradients_flow_to_encoder_when_unfrozen(self, beats_encoder):
        """Test that gradients flow to encoder when unfrozen."""
        detector = FrameDetector(encoder=beats_encoder, labels=TEST_LABELS)
        detector.train()
        detector.unfreeze_encoder()

        audio = torch.randn(1, int(WINDOW_DURATION * SAMPLE_RATE))
        output = detector(audio)
        loss = output.sum()
        loss.backward()

        # Check that at least some encoder parameters have gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in detector.encoder.parameters())
        assert has_grad

    def test_no_encoder_gradients_when_frozen(self, beats_encoder):
        """Test that encoder parameters have requires_grad=False when frozen.

        Note: Setting requires_grad=False prevents parameters from being updated
        by the optimizer, but gradients may still flow through the computation
        graph during backward(). This matches the behavior of BeatsDetector.
        """
        detector = FrameDetector(encoder=beats_encoder, labels=TEST_LABELS)
        detector.train()
        detector.freeze_encoder()

        # All encoder parameters should have requires_grad=False
        assert not any(p.requires_grad for p in detector.encoder.parameters())

        # Classifier should still be trainable
        assert all(p.requires_grad for p in detector.classifier.parameters())


# --- Tests for Edge Cases ---


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_label(self, raw_beats_model):
        """Test detector with single label."""
        encoder = BEATSEncoder(
            model=raw_beats_model,
            sample_rate=SAMPLE_RATE,
            window_duration=WINDOW_DURATION,
            aggregation="average",
        )
        detector = FrameDetector(encoder=encoder, labels=["single_species"])

        assert detector.classifier.out_features == 1

        audio = torch.randn(1, int(WINDOW_DURATION * SAMPLE_RATE))
        with torch.no_grad():
            output = detector(audio)

        assert output.shape == (1, 62, 1)

    def test_many_labels(self, raw_beats_model):
        """Test detector with many labels."""
        encoder = BEATSEncoder(
            model=raw_beats_model,
            sample_rate=SAMPLE_RATE,
            window_duration=WINDOW_DURATION,
            aggregation="average",
        )
        labels = [f"species_{i}" for i in range(100)]
        detector = FrameDetector(encoder=encoder, labels=labels)

        assert detector.classifier.out_features == 100

        audio = torch.randn(1, int(WINDOW_DURATION * SAMPLE_RATE))
        with torch.no_grad():
            output = detector(audio)

        assert output.shape == (1, 62, 100)

    def test_detector_is_nn_module(self, frame_detector):
        """Test that FrameDetector is a proper nn.Module."""
        assert isinstance(frame_detector, torch.nn.Module)
        assert hasattr(frame_detector, "parameters")
        assert hasattr(frame_detector, "eval")
        assert hasattr(frame_detector, "train")

    def test_detector_can_be_moved_to_device(self, frame_detector):
        """Test that detector can be moved between devices."""
        detector_cpu = frame_detector.to("cpu")
        assert next(detector_cpu.parameters()).device.type == "cpu"

        # Test forward still works
        audio = torch.randn(1, int(WINDOW_DURATION * SAMPLE_RATE))
        with torch.no_grad():
            output = detector_cpu(audio)

        assert output.shape == (1, 62, len(TEST_LABELS))

    def test_very_short_audio_inference(self, frame_detector):
        """Test inference on very short audio (< 1 second)."""
        audio = np.random.randn(1, int(0.5 * SAMPLE_RATE)).astype(np.float32)
        result = frame_detector.run(audio)

        # Should produce some frames (padded)
        assert result.predictions.shape[1] > 0

    def test_silent_audio_inference(self, frame_detector):
        """Test inference on silent audio."""
        audio = np.zeros((1, 5 * SAMPLE_RATE), dtype=np.float32)
        result = frame_detector.run(audio)

        assert result.predictions.shape[1] == 62
        # Probabilities should be valid
        assert (result.predictions >= 0).all()
        assert (result.predictions <= 1).all()
