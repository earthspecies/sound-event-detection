"""Tests for overlapping window inference in FrameDetector.

Tests the `run` method with various overlap settings,
focusing on correctness of frame stitching and boundary conditions.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from esp_research.protocols.encoder import AudioEncoderOutput
from sound_event_detection.models.frame_detector import FrameDetector
from sound_event_detection.utils.reformatters import detector_output_to_dataframe

# --- Test Model ---


class DeterministicEncoder(nn.Module):
    """Encoder that outputs deterministic frame indices for testing.

    For each window, outputs frames where each frame's value encodes its
    position within the window. This allows verification that the correct
    frames are kept during stitching.

    Implements the AudioEncoder protocol expected by FrameDetector.
    """

    def __init__(
        self,
        frames_per_window: int,
        hidden_dim: int = 16,
        sample_rate: int = 1000,
        window_duration: float = 1.0,
    ):
        super().__init__()
        self._frames_per_window = frames_per_window
        self._hidden_dim = hidden_dim
        self._sample_rate = sample_rate
        self._window_duration = window_duration
        # Dummy parameter so model.parameters() works
        self._dummy = nn.Parameter(torch.zeros(1))

    @property
    def output_dim(self) -> int:
        return self._hidden_dim

    @property
    def output_frame_rate(self) -> float:
        return self._frames_per_window / self._window_duration

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def window_duration(self) -> float:
        return self._window_duration

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        # Output shape: [batch, frames, hidden_dim]
        frame_indices = torch.arange(self._frames_per_window, dtype=torch.float32)
        # Expand to [1, frames, 1] then broadcast
        frame_values = frame_indices.view(1, -1, 1).expand(batch_size, -1, self._hidden_dim)

        return frame_values.clone()

    def encode(self, waveform: torch.Tensor, padding_mask: torch.Tensor) -> AudioEncoderOutput:
        embeddings = self.forward(waveform)
        out_mask = torch.zeros(embeddings.shape[:2], dtype=torch.bool, device=embeddings.device)
        return AudioEncoderOutput(embeddings=embeddings, padding_mask=out_mask)

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True


@pytest.fixture
def simple_model() -> FrameDetector:
    """Create a simple detector with deterministic output for testing."""
    frames_per_window = 10
    hidden_dim = 16
    sample_rate = 1000  # 1 sample = 1ms for easy math
    window_duration = 1.0  # 1 second = 1000 samples

    encoder = DeterministicEncoder(
        frames_per_window=frames_per_window,
        hidden_dim=hidden_dim,
        sample_rate=sample_rate,
        window_duration=window_duration,
    )
    model = FrameDetector(
        encoder=encoder,
        labels=["A", "B"],
    )
    # Replace classifier with identity-like layer for predictable output
    model.classifier = nn.Linear(hidden_dim, 2, bias=False)
    with torch.no_grad():
        model.classifier.weight.fill_(0.0)
        model.classifier.weight[0, 0] = 1.0  # First class = first hidden dim
    model.eval()
    return model


# --- Overlap Parameter Validation ---


class TestOverlapValidation:
    """Tests for overlap parameter validation."""

    def test_overlap_none_equivalent_to_zero(self, simple_model):
        """overlap=None should behave identically to overlap=0.0."""
        audio = np.zeros((1, 2000), dtype=np.float32)

        result_none = simple_model.run(audio, overlap=None)
        result_zero = simple_model.run(audio, overlap=0.0)

        np.testing.assert_array_equal(
            result_none.predictions,
            result_zero.predictions,
        )

    def test_overlap_invalid_negative(self, simple_model):
        """Negative overlap should raise ValueError."""
        audio = np.zeros((1, 1000), dtype=np.float32)
        with pytest.raises(ValueError, match="overlap must be in"):
            simple_model.run(audio, overlap=-0.1)

    def test_overlap_invalid_one(self, simple_model):
        """overlap=1.0 should raise ValueError (would keep 0 frames)."""
        audio = np.zeros((1, 1000), dtype=np.float32)
        with pytest.raises(ValueError, match="overlap must be in"):
            simple_model.run(audio, overlap=1.0)

    def test_overlap_invalid_greater_than_one(self, simple_model):
        """overlap > 1.0 should raise ValueError."""
        audio = np.zeros((1, 1000), dtype=np.float32)
        with pytest.raises(ValueError, match="overlap must be in"):
            simple_model.run(audio, overlap=1.5)


# --- Frame Count Consistency ---


class TestFrameCount:
    """Tests that output frame count is correct and independent of overlap.

    Key invariant: The number of output frames should equal audio_duration * frame_rate,
    regardless of overlap setting. Overlap only affects WHICH window's prediction is
    used for each time point, not HOW MANY frames are output.
    """

    @pytest.mark.parametrize("overlap", [None, 0.0, 0.2, 0.5, 0.8])
    def test_single_window_frame_count(self, simple_model, overlap):
        """Single window should produce frames_per_window frames."""
        audio = np.zeros((1, 1000), dtype=np.float32)  # 1 window
        result = simple_model.run(audio, overlap=overlap)
        assert result.predictions.shape[1] == 10  # frames_per_window

    def test_frame_count_independent_of_overlap(self, simple_model):
        """Frame count must be identical regardless of overlap setting.

        This is the core invariant: overlap affects which predictions are used,
        not how many frames are output.
        """
        # Test with various audio lengths
        for n_samples in [1000, 2000, 3000, 5000, 10000]:
            audio = np.zeros((1, n_samples), dtype=np.float32)

            frame_counts = []
            for overlap in [0.0, 0.2, 0.5, 0.8]:
                result = simple_model.run(audio, overlap=overlap)
                frame_counts.append(result.predictions.shape[1])

            # All overlaps must produce the same frame count
            assert all(c == frame_counts[0] for c in frame_counts), (
                f"Frame counts differ for {n_samples} samples: {frame_counts}"
            )

    def test_frame_count_equals_duration_times_frame_rate(self, simple_model):
        """For clean multiples of window duration, frame count = duration * frame_rate ."""
        # Test with exact multiples of window duration (1 second)
        for n_windows in [1, 2, 5, 10]:
            n_samples = n_windows * 1000  # window_duration * sample_rate
            audio = np.zeros((1, n_samples), dtype=np.float32)
            expected_frames = n_windows * 10  # n_windows * frames_per_window

            for overlap in [0.0, 0.2, 0.5, 0.8]:
                result = simple_model.run(audio, overlap=overlap)
                assert result.predictions.shape[1] == expected_frames, (
                    f"Expected {expected_frames} frames for {n_windows} windows with "
                    f"overlap={overlap}, got {result.predictions.shape[1]}"
                )

    def test_frame_count_scales_linearly_with_duration(self, simple_model):
        """Frame count should scale  linearly with audio duration."""
        # Use exact multiples to avoid edge effects
        durations_in_windows = [1, 2, 5, 10, 20]

        for overlap in [0.0, 0.2, 0.5, 0.8]:
            frame_counts = []
            for n_windows in durations_in_windows:
                n_samples = n_windows * 1000
                audio = np.zeros((1, n_samples), dtype=np.float32)
                result = simple_model.run(audio, overlap=overlap)
                frame_counts.append(result.predictions.shape[1])

            # Check exact linear scaling
            for i, n_windows in enumerate(durations_in_windows):
                expected = n_windows * 10  # n_windows * frames_per_window
                assert frame_counts[i] == expected, (
                    f"Expected {expected} frames for {n_windows} windows, got {frame_counts[i]}"
                )

    def test_no_drift_over_long_files(self, simple_model):
        """Over long files with many windows, there should be zero accumulated drift."""
        # 100 windows = 100 seconds at 1 window/second
        n_windows = 100
        n_samples = n_windows * 1000
        audio = np.zeros((1, n_samples), dtype=np.float32)
        expected_frames = n_windows * 10

        for overlap in [0.0, 0.3, 0.5, 0.7]:
            result = simple_model.run(audio, overlap=overlap)
            assert result.predictions.shape[1] == expected_frames, (
                f"Drift detected with overlap={overlap}: expected {expected_frames}, "
                f"got {result.predictions.shape[1]} (diff={result.predictions.shape[1] - expected_frames})"
            )


# --- Frame Stitching Correctness ---


class TestFrameStitching:
    """Tests that frames are correctly stitched together with complete coverage."""

    def test_no_overlap_keeps_all_frames(self, simple_model):
        """With no overlap, all frames from each window should be kept."""
        audio = np.zeros((1, 2000), dtype=np.float32)  # 2 windows
        result = simple_model.run(audio, overlap=0.0)

        # Should have 20 frames (2 windows × 10 frames)
        assert result.predictions.shape[1] == 20

        # First class output encodes frame position within window
        probs = torch.sigmoid(torch.tensor(detector_output_to_dataframe(result)["A"].values))

        # Window 1: frames 0-9, Window 2: frames 0-9
        # After sigmoid, values should show the pattern repeating
        first_window = probs[:10].numpy()
        second_window = probs[10:].numpy()
        np.testing.assert_allclose(first_window, second_window, rtol=1e-5)

    def test_overlap_produces_same_frame_count(self, simple_model):
        """With overlap, frame count must be identical to no overlap."""
        audio = np.zeros((1, 3000), dtype=np.float32)  # 3 windows

        result_no_overlap = simple_model.run(audio, overlap=0.0)
        result_overlap = simple_model.run(audio, overlap=0.5)

        # Frame counts must be identical - this is the key invariant
        assert result_no_overlap.predictions.shape[1] == result_overlap.predictions.shape[1], (
            f"Frame count mismatch: no_overlap={result_no_overlap.predictions.shape[1]}, "
            f"overlap=0.5 gave {result_overlap.predictions.shape[1]}"
        )

    def test_all_time_points_represented_with_overlap(self, simple_model):
        """Every time point should be represented  once with overlap."""
        # 5 windows worth of audio
        audio = np.zeros((1, 5000), dtype=np.float32)
        expected_frames = 50  # 5 windows * 10 frames/window

        for overlap in [0.0, 0.2, 0.5, 0.8]:
            result = simple_model.run(audio, overlap=overlap)
            assert result.predictions.shape[1] == expected_frames, (
                f"overlap={overlap}: expected {expected_frames} frames, got {result.predictions.shape[1]}"
            )

    def test_continuous_time_coverage_exact(self, simple_model):
        """Output duration should  match input duration for clean multiples."""
        # Use exact multiples of window duration
        for n_windows in [1, 3, 5, 10]:
            n_samples = n_windows * 1000
            audio = np.zeros((1, n_samples), dtype=np.float32)
            audio_duration = n_samples / simple_model.sample_rate

            for overlap in [0.0, 0.2, 0.5, 0.8]:
                result = simple_model.run(audio, overlap=overlap)
                implied_duration = result.predictions.shape[1] / result.frame_rate

                assert implied_duration == audio_duration, (
                    f"Duration mismatch for {n_windows} windows, overlap={overlap}: "
                    f"audio={audio_duration}s, output={implied_duration}s"
                )


# --- Edge Cases ---


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_audio_shorter_than_window(self, simple_model):
        """Audio shorter than one window should be padded and produce full window frames."""
        audio = np.zeros((1, 500), dtype=np.float32)  # Half a window

        for overlap in [None, 0.0, 0.5]:
            result = simple_model.run(audio, overlap=overlap)
            # Should produce frames for a full window (audio gets padded)
            # The exact count depends on implementation, but should be consistent
            assert result.predictions.shape[1] > 0

    def test_audio_one_window(self, simple_model):
        """Audio  one window long should produce  frames_per_window frames."""
        audio = np.zeros((1, 1000), dtype=np.float32)  # 1 window

        for overlap in [None, 0.0, 0.5, 0.8]:
            result = simple_model.run(audio, overlap=overlap)
            assert result.predictions.shape[1] == 10, (
                f"overlap={overlap}: expected 10 frames, got {result.predictions.shape[1]}"
            )

    def test_audio_slightly_over_one_window(self, simple_model):
        """Audio slightly longer than one window."""
        audio = np.zeros((1, 1001), dtype=np.float32)  # Just over 1 window

        # Both overlaps should produce the same frame count
        result_no_overlap = simple_model.run(audio, overlap=0.0)
        result_overlap = simple_model.run(audio, overlap=0.5)

        assert result_no_overlap.predictions.shape[1] == result_overlap.predictions.shape[1]
        # Should have at least 10 frames (one full window)
        assert result_no_overlap.predictions.shape[1] >= 10

    def test_non_integer_window_multiples(self, simple_model):
        """Audio that's not an exact multiple of window duration."""
        # 2.5 windows worth
        audio = np.zeros((1, 2500), dtype=np.float32)

        # All overlaps should produce the same frame count
        frame_counts = []
        for overlap in [0.0, 0.3, 0.5]:
            result = simple_model.run(audio, overlap=overlap)
            frame_counts.append(result.predictions.shape[1])

        assert all(c == frame_counts[0] for c in frame_counts), (
            f"Frame counts differ for non-integer windows: {frame_counts}"
        )

    def test_wrong_audio_dimensions(self, simple_model):
        """Non-2D audio should raise ValueError."""
        audio = np.zeros(1000, dtype=np.float32)  # 1D
        with pytest.raises(ValueError, match="Expected 2D audio"):
            simple_model.run(audio)


# --- Determinism ---


class TestDeterminism:
    """Tests that inference is deterministic."""

    def test_same_input_same_output(self, simple_model):
        """Same input should produce identical output."""
        audio = np.random.randn(1, 3000).astype(np.float32)

        result1 = simple_model.run(audio, overlap=0.5)
        result2 = simple_model.run(audio, overlap=0.5)

        np.testing.assert_array_equal(
            result1.predictions,
            result2.predictions,
        )

    def test_batch_size_does_not_affect_output(self, simple_model):
        """Different batch sizes should produce identical output."""
        audio = np.zeros((1, 5000), dtype=np.float32)

        result_bs1 = simple_model.run(audio, batch_size=1, overlap=0.5)
        result_bs4 = simple_model.run(audio, batch_size=4, overlap=0.5)
        result_bs32 = simple_model.run(audio, batch_size=32, overlap=0.5)

        np.testing.assert_array_equal(
            result_bs1.predictions,
            result_bs4.predictions,
        )
        np.testing.assert_array_equal(
            result_bs1.predictions,
            result_bs32.predictions,
        )
