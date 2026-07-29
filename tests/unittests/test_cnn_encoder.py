"""Tests for CNNEncoder.

Tests the CNNEncoder class which produces frame-level embeddings from audio
waveforms via mel spectrogram and lightweight CNN processing.
"""

import pytest
import torch

from sound_event_detection.models.encoders.cnn import (
    CNNEncoder,
)

# --- Constants ---

SAMPLE_RATE = 32000
WINDOW_DURATION = 5.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION)


# --- Fixtures ---


@pytest.fixture(scope="module")
def cnn_encoder():
    """Create a CNNEncoder with default settings."""
    encoder = CNNEncoder(
        sample_rate=SAMPLE_RATE,
        window_duration=WINDOW_DURATION,
    )
    encoder.eval()
    return encoder


# --- Tests for CNNEncoder Initialization ---


class TestCNNEncoderInit:
    """Tests for CNNEncoder initialization."""

    def test_has_mel_spectrogram(self, cnn_encoder):
        """Test that encoder has mel spectrogram transform."""
        assert hasattr(cnn_encoder, "mel_spectrogram")

    def test_has_cnn(self, cnn_encoder):
        """Test that encoder has CNN module."""
        assert hasattr(cnn_encoder, "cnn")
        assert isinstance(cnn_encoder.cnn, torch.nn.Sequential)

    def test_output_dim(self, cnn_encoder):
        """Test output dimension is last filter count."""
        assert cnn_encoder.output_dim == 384

    def test_sample_rate_stored(self, cnn_encoder):
        """Test sample rate is stored correctly."""
        assert cnn_encoder.sample_rate == SAMPLE_RATE

    def test_window_duration_stored(self, cnn_encoder):
        """Test window duration is stored correctly."""
        assert cnn_encoder.window_duration == WINDOW_DURATION

    def test_invalid_freq_pooling_raises(self):
        """Test that invalid frequency pooling raises error."""
        # Pooling that doesn't collapse 128 to 1 (only 4x freq pooling total)
        bad_pooling = [(2, 2), (2, 2), (1, 1)]
        bad_filters = [32, 64, 128]
        with pytest.raises(ValueError, match="collapse frequency to 1"):
            CNNEncoder(
                sample_rate=SAMPLE_RATE,
                window_duration=WINDOW_DURATION,
                nb_filters=bad_filters,
                pooling=bad_pooling,
            )

    def test_mismatched_n_mels_raises(self):
        """Test that n_mels not divisible by pooling raises error."""
        with pytest.raises(ValueError, match="divisible by"):
            CNNEncoder(
                sample_rate=SAMPLE_RATE,
                window_duration=WINDOW_DURATION,
                n_mels=100,  # Not divisible by 128
            )

    def test_custom_filters(self):
        """Test CNNEncoder with custom filter configuration."""
        filters = [32, 64, 128]
        pooling = [(2, 2), (2, 2), (2, 32)]  # Collapse 128 -> 1
        encoder = CNNEncoder(
            sample_rate=SAMPLE_RATE,
            window_duration=WINDOW_DURATION,
            nb_filters=filters,
            pooling=pooling,
        )
        assert encoder.output_dim == 128


# --- Tests for CNNEncoder Properties ---


class TestCNNEncoderProperties:
    """Tests for CNNEncoder property delegation."""

    def test_output_frame_rate_positive(self, cnn_encoder):
        """Test that output frame_rate is positive."""
        assert cnn_encoder.output_frame_rate > 0

    def test_output_frame_rate_exact(self, cnn_encoder):
        """Test that output frame_rate matches the current CNN architecture."""
        assert cnn_encoder.output_frame_rate == pytest.approx(25.0, abs=0.1)


# --- Tests for CNNEncoder Forward Pass ---


class TestCNNEncoderForward:
    """Tests for CNNEncoder forward pass."""

    def test_forward_output_shape(self, cnn_encoder):
        """Test forward pass output shape."""
        batch_size = 2
        audio = torch.randn(batch_size, WINDOW_SAMPLES)

        with torch.no_grad():
            output = cnn_encoder(audio)

        assert output.ndim == 3
        assert output.shape[0] == batch_size
        assert output.shape[2] == cnn_encoder.output_dim

    def test_forward_frame_count(self, cnn_encoder):
        """Test that frame count matches expected from frame_rate."""
        audio = torch.randn(1, WINDOW_SAMPLES)

        with torch.no_grad():
            output = cnn_encoder(audio)

        expected_frames = int(WINDOW_DURATION * cnn_encoder.output_frame_rate)
        # Allow small tolerance due to floor divisions
        assert abs(output.shape[1] - expected_frames) <= 1

    def test_forward_deterministic(self, cnn_encoder):
        """Test that forward pass is deterministic in eval mode."""
        audio = torch.randn(1, WINDOW_SAMPLES)

        with torch.no_grad():
            output1 = cnn_encoder(audio)
            output2 = cnn_encoder(audio)

        torch.testing.assert_close(output1, output2)

    def test_forward_batch_consistency(self, cnn_encoder):
        """Test that batched forward is consistent with single forward."""
        single_audio = torch.randn(1, WINDOW_SAMPLES)
        batch_audio = single_audio.repeat(3, 1)

        with torch.no_grad():
            single_output = cnn_encoder(single_audio)
            batch_output = cnn_encoder(batch_audio)

        for i in range(3):
            torch.testing.assert_close(batch_output[i], single_output[0])

    def test_forward_different_audio_different_output(self, cnn_encoder):
        """Test that different audio produces different output."""
        audio1 = torch.randn(1, WINDOW_SAMPLES)
        audio2 = torch.randn(1, WINDOW_SAMPLES)

        with torch.no_grad():
            output1 = cnn_encoder(audio1)
            output2 = cnn_encoder(audio2)

        assert not torch.allclose(output1, output2)


# --- Tests for Freeze/Unfreeze ---


class TestCNNEncoderFreezeUnfreeze:
    """Tests for freeze and unfreeze methods."""

    def test_freeze(self):
        """Test that freeze freezes all parameters."""
        encoder = CNNEncoder(sample_rate=SAMPLE_RATE, window_duration=WINDOW_DURATION)
        encoder.freeze()

        for param in encoder.parameters():
            assert not param.requires_grad

    def test_unfreeze(self):
        """Test that unfreeze unfreezes all parameters."""
        encoder = CNNEncoder(sample_rate=SAMPLE_RATE, window_duration=WINDOW_DURATION)
        encoder.freeze()
        encoder.unfreeze()

        for param in encoder.parameters():
            assert param.requires_grad


# --- Tests for Different Sample Rates ---


class TestCNNEncoderSampleRates:
    """Tests for CNNEncoder with different sample rates."""

    @pytest.mark.parametrize("sample_rate", [16000, 32000, 44100])
    def test_different_sample_rates(self, sample_rate):
        """Test encoder works with different sample rates."""
        encoder = CNNEncoder(
            sample_rate=sample_rate,
            window_duration=WINDOW_DURATION,
        )
        encoder.eval()

        window_samples = int(sample_rate * WINDOW_DURATION)
        audio = torch.randn(1, window_samples)

        with torch.no_grad():
            output = encoder(audio)

        assert output.ndim == 3
        assert output.shape[0] == 1
        assert output.shape[2] == 384

    @pytest.mark.parametrize("window_duration", [1.0, 5.0, 10.0])
    def test_different_window_durations(self, window_duration):
        """Test encoder works with different window durations."""
        encoder = CNNEncoder(
            sample_rate=SAMPLE_RATE,
            window_duration=window_duration,
        )
        encoder.eval()

        window_samples = int(SAMPLE_RATE * window_duration)
        audio = torch.randn(1, window_samples)

        with torch.no_grad():
            output = encoder(audio)

        # Frame count should scale with window duration
        expected_frames = int(window_duration * encoder.output_frame_rate)
        assert abs(output.shape[1] - expected_frames) <= 1


# --- Tests for Gradient Flow ---


class TestCNNEncoderGradients:
    """Tests for gradient flow through CNNEncoder."""

    def test_gradients_flow(self):
        """Test that gradients flow through encoder."""
        encoder = CNNEncoder(sample_rate=SAMPLE_RATE, window_duration=WINDOW_DURATION)
        encoder.train()

        audio = torch.randn(2, WINDOW_SAMPLES, requires_grad=True)
        output = encoder(audio)
        loss = output.sum()
        loss.backward()

        # Check that CNN parameters have gradients
        for name, param in encoder.cnn.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_frozen_no_gradients(self):
        """Test that frozen encoder doesn't accumulate gradients."""
        encoder = CNNEncoder(sample_rate=SAMPLE_RATE, window_duration=WINDOW_DURATION)
        encoder.freeze()
        encoder.train()

        audio = torch.randn(2, WINDOW_SAMPLES)
        output = encoder(audio)

        # Output should not require grad when encoder is frozen
        assert not output.requires_grad

        # Frozen parameters should have requires_grad=False
        for param in encoder.parameters():
            assert not param.requires_grad
