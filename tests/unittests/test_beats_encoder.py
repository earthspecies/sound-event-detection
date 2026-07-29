"""Tests for BEATSEncoder.

Tests the BEATSEncoder class which wraps pretrained BEATs models and handles
aggregation of patch embeddings. Uses real BEATs encoder from avex
to ensure accurate integration testing.
"""

import pytest
import torch

from sound_event_detection.models.encoders import (
    AGGREGATION_STRATEGIES,
    BEATSEncoder,
    BEATSEncoderConfig,
    compute_beats_frame_rate,
)

# --- Constants ---

# Encoder name for avex API
ENCODER_NAME = "esp_aves2_sl_beats_all"

# Standard test configuration
SAMPLE_RATE = 32000
WINDOW_DURATION = 5.0


# --- Fixtures ---


@pytest.fixture(scope="module")
def raw_beats_model():
    """Load the raw BEATs model using avex API.

    Uses module scope to avoid downloading the model multiple times.
    """
    from avex import load_model

    model = load_model(ENCODER_NAME, return_features_only=True, device="cpu")
    return model


@pytest.fixture(scope="module")
def beats_encoder(raw_beats_model):
    """Create a BEATSEncoder with default (average) aggregation."""
    encoder = BEATSEncoder(
        model=raw_beats_model,
        sample_rate=SAMPLE_RATE,
        window_duration=WINDOW_DURATION,
        aggregation="average",
    )
    encoder.eval()
    return encoder


@pytest.fixture(scope="module")
def all_frames_encoder(raw_beats_model):
    """Create a BEATSEncoder with all_frames aggregation."""
    encoder = BEATSEncoder(
        model=raw_beats_model,
        sample_rate=SAMPLE_RATE,
        window_duration=WINDOW_DURATION,
        aggregation="all_frames",
    )
    encoder.eval()
    return encoder


@pytest.fixture(scope="module")
def concat_encoder(raw_beats_model):
    """Create a BEATSEncoder with concat aggregation."""
    encoder = BEATSEncoder(
        model=raw_beats_model,
        sample_rate=SAMPLE_RATE,
        window_duration=WINDOW_DURATION,
        aggregation="concat",
    )
    encoder.eval()
    return encoder


# --- Tests for BEATSEncoder Properties ---


class TestBEATSEncoderProperties:
    """Tests for BEATSEncoder property accessors."""

    def test_output_dim_average(self, beats_encoder):
        """Test output_dim for average aggregation."""
        assert beats_encoder.output_dim == BEATSEncoderConfig.HIDDEN_DIM
        assert beats_encoder.output_dim == 768

    def test_output_dim_all_frames(self, all_frames_encoder):
        """Test output_dim for all_frames aggregation (same as average)."""
        assert all_frames_encoder.output_dim == BEATSEncoderConfig.HIDDEN_DIM

    def test_output_dim_concat(self, concat_encoder):
        """Test output_dim for concat aggregation (8x larger)."""
        expected = BEATSEncoderConfig.HIDDEN_DIM * BEATSEncoderConfig.NUM_FREQ_PATCHES
        assert concat_encoder.output_dim == expected
        assert concat_encoder.output_dim == 768 * 8

    def test_output_frame_rate_average(self, beats_encoder):
        """Test output_frame_rate for average aggregation."""
        expected = compute_beats_frame_rate(SAMPLE_RATE, WINDOW_DURATION)
        assert beats_encoder.output_frame_rate == expected
        assert beats_encoder.output_frame_rate == 12.4

    def test_output_frame_rate_all_frames(self, all_frames_encoder):
        """Test output_frame_rate for all_frames aggregation (8x higher)."""
        base = compute_beats_frame_rate(SAMPLE_RATE, WINDOW_DURATION)
        expected = base * BEATSEncoderConfig.NUM_FREQ_PATCHES
        assert all_frames_encoder.output_frame_rate == expected
        assert all_frames_encoder.output_frame_rate == 12.4 * 8

    def test_output_frame_rate_concat(self, concat_encoder):
        """Test output_frame_rate for concat aggregation (same as average)."""
        expected = compute_beats_frame_rate(SAMPLE_RATE, WINDOW_DURATION)
        assert concat_encoder.output_frame_rate == expected

    def test_sample_rate(self, beats_encoder):
        """Test sample_rate property."""
        assert beats_encoder.sample_rate == SAMPLE_RATE

    def test_window_duration(self, beats_encoder):
        """Test window_duration property."""
        assert beats_encoder.window_duration == WINDOW_DURATION


# --- Tests for BEATSEncoder Initialization ---


class TestBEATSEncoderInit:
    """Tests for BEATSEncoder initialization."""

    def test_invalid_aggregation_raises_error(self, raw_beats_model):
        """Test that invalid aggregation strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown aggregation strategy"):
            BEATSEncoder(
                model=raw_beats_model,
                sample_rate=SAMPLE_RATE,
                window_duration=WINDOW_DURATION,
                aggregation="invalid_strategy",
            )

    def test_all_valid_aggregations_accepted(self, raw_beats_model):
        """Test that all valid aggregation strategies are accepted."""
        for aggregation in AGGREGATION_STRATEGIES.keys():
            encoder = BEATSEncoder(
                model=raw_beats_model,
                sample_rate=SAMPLE_RATE,
                window_duration=WINDOW_DURATION,
                aggregation=aggregation,
            )
            assert encoder is not None

    def test_different_sample_rates(self, raw_beats_model):
        """Test encoder with different sample rates."""
        for sr in [16000, 32000, 48000]:
            encoder = BEATSEncoder(
                model=raw_beats_model,
                sample_rate=sr,
                window_duration=WINDOW_DURATION,
                aggregation="average",
            )
            assert encoder.sample_rate == sr
            # Framerate should scale with sample rate
            assert encoder.output_frame_rate == compute_beats_frame_rate(sr, WINDOW_DURATION)

    def test_different_window_durations(self, raw_beats_model):
        """Test encoder with different window durations."""
        for duration in [3.0, 5.0, 10.0]:
            encoder = BEATSEncoder(
                model=raw_beats_model,
                sample_rate=SAMPLE_RATE,
                window_duration=duration,
                aggregation="average",
            )
            assert encoder.window_duration == duration


# --- Tests for BEATSEncoder Forward Pass ---


class TestBEATSEncoderForward:
    """Tests for BEATSEncoder forward pass."""

    def test_forward_output_shape_average(self, beats_encoder):
        """Test forward pass output shape with average aggregation."""
        batch_size = 2
        window_samples = int(WINDOW_DURATION * SAMPLE_RATE)
        audio = torch.randn(batch_size, window_samples)

        with torch.no_grad():
            output = beats_encoder(audio)

        # Should be [batch, frames, hidden_dim]
        assert output.ndim == 3
        assert output.shape[0] == batch_size
        assert output.shape[1] == 62  # frames for 5s at 32kHz
        assert output.shape[2] == BEATSEncoderConfig.HIDDEN_DIM

    def test_forward_output_shape_all_frames(self, all_frames_encoder):
        """Test forward pass output shape with all_frames aggregation."""
        batch_size = 2
        window_samples = int(WINDOW_DURATION * SAMPLE_RATE)
        audio = torch.randn(batch_size, window_samples)

        with torch.no_grad():
            output = all_frames_encoder(audio)

        # Should be [batch, frames * 8, hidden_dim]
        assert output.ndim == 3
        assert output.shape[0] == batch_size
        assert output.shape[1] == 62 * BEATSEncoderConfig.NUM_FREQ_PATCHES  # 496 frames
        assert output.shape[2] == BEATSEncoderConfig.HIDDEN_DIM

    def test_forward_output_shape_concat(self, concat_encoder):
        """Test forward pass output shape with concat aggregation."""
        batch_size = 2
        window_samples = int(WINDOW_DURATION * SAMPLE_RATE)
        audio = torch.randn(batch_size, window_samples)

        with torch.no_grad():
            output = concat_encoder(audio)

        # Should be [batch, frames, hidden_dim * 8]
        assert output.ndim == 3
        assert output.shape[0] == batch_size
        assert output.shape[1] == 62  # same frames as average
        assert output.shape[2] == BEATSEncoderConfig.HIDDEN_DIM * BEATSEncoderConfig.NUM_FREQ_PATCHES

    def test_forward_batch_consistency(self, beats_encoder):
        """Test that forward pass is consistent across batch."""
        window_samples = int(WINDOW_DURATION * SAMPLE_RATE)
        single_audio = torch.randn(1, window_samples)
        batch_audio = single_audio.repeat(3, 1)

        with torch.no_grad():
            single_output = beats_encoder(single_audio)
            batch_output = beats_encoder(batch_audio)

        # All batch outputs should be identical
        for i in range(3):
            torch.testing.assert_close(
                batch_output[i], single_output[0], rtol=1e-5, atol=1e-5
            )

    def test_forward_output_is_finite(self, beats_encoder):
        """Test that forward pass returns finite values."""
        window_samples = int(WINDOW_DURATION * SAMPLE_RATE)
        audio = torch.randn(1, window_samples)

        with torch.no_grad():
            output = beats_encoder(audio)

        assert torch.isfinite(output).all()
        assert output.abs().sum() > 0  # Not all zeros

    def test_forward_deterministic_in_eval_mode(self, beats_encoder):
        """Test that forward pass is deterministic in eval mode."""
        window_samples = int(WINDOW_DURATION * SAMPLE_RATE)
        audio = torch.randn(1, window_samples)

        beats_encoder.eval()
        with torch.no_grad():
            output1 = beats_encoder(audio)
            output2 = beats_encoder(audio)

        torch.testing.assert_close(output1, output2)


# --- Tests for Frame Count Accuracy ---


class TestFrameCountAccuracy:
    """Tests for frame count accuracy across different configurations."""

    def test_frame_count_matches_frame_rate(self, beats_encoder):
        """Test that actual frame count matches computed frame_rate."""
        window_samples = int(WINDOW_DURATION * SAMPLE_RATE)
        audio = torch.randn(1, window_samples)

        with torch.no_grad():
            output = beats_encoder(audio)

        expected_frames = int(beats_encoder.output_frame_rate * WINDOW_DURATION)
        assert output.shape[1] == expected_frames

    def test_frame_count_different_durations(self, raw_beats_model):
        """Test frame count accuracy for different window durations."""
        for duration in [3.0, 5.0, 10.0]:
            encoder = BEATSEncoder(
                model=raw_beats_model,
                sample_rate=SAMPLE_RATE,
                window_duration=duration,
                aggregation="average",
            )

            window_samples = int(duration * SAMPLE_RATE)
            audio = torch.randn(1, window_samples)

            with torch.no_grad():
                output = encoder(audio)

            expected_frames = int(encoder.output_frame_rate * duration)
            assert output.shape[1] == expected_frames, (
                f"Duration {duration}s: expected {expected_frames} frames, got {output.shape[1]}"
            )

    def test_all_aggregations_frame_count(self, raw_beats_model):
        """Test frame count for all aggregation strategies."""
        window_samples = int(WINDOW_DURATION * SAMPLE_RATE)
        audio = torch.randn(1, window_samples)

        for aggregation, strategy in AGGREGATION_STRATEGIES.items():
            encoder = BEATSEncoder(
                model=raw_beats_model,
                sample_rate=SAMPLE_RATE,
                window_duration=WINDOW_DURATION,
                aggregation=aggregation,
            )

            with torch.no_grad():
                output = encoder(audio)

            expected_frames = int(encoder.output_frame_rate * WINDOW_DURATION)
            assert output.shape[1] == expected_frames, (
                f"{aggregation}: expected {expected_frames} frames, got {output.shape[1]}"
            )


# --- Tests for Freeze/Unfreeze ---


class TestBEATSEncoderFreezeUnfreeze:
    """Tests for freeze and unfreeze methods."""

    def test_freeze(self, beats_encoder):
        """Test that freeze freezes all parameters."""
        beats_encoder.unfreeze()  # Ensure unfrozen first
        assert all(p.requires_grad for p in beats_encoder.parameters())

        beats_encoder.freeze()

        assert not any(p.requires_grad for p in beats_encoder.parameters())

    def test_unfreeze(self, beats_encoder):
        """Test that unfreeze unfreezes all parameters."""
        beats_encoder.freeze()
        assert not any(p.requires_grad for p in beats_encoder.parameters())

        beats_encoder.unfreeze()

        assert all(p.requires_grad for p in beats_encoder.parameters())

    def test_freeze_unfreeze_preserves_output(self, beats_encoder):
        """Test that freeze/unfreeze doesn't affect output."""
        window_samples = int(WINDOW_DURATION * SAMPLE_RATE)
        audio = torch.randn(1, window_samples)

        beats_encoder.eval()
        with torch.no_grad():
            output_before = beats_encoder(audio).clone()

        beats_encoder.freeze()
        with torch.no_grad():
            output_frozen = beats_encoder(audio).clone()

        beats_encoder.unfreeze()
        with torch.no_grad():
            output_unfrozen = beats_encoder(audio)

        torch.testing.assert_close(output_before, output_frozen)
        torch.testing.assert_close(output_before, output_unfrozen)


# --- Tests for compute_beats_frame_rate ---


class TestComputeBeatsFramerate:
    """Tests for the compute_beats_frame_rate function."""

    def test_frame_rate_32khz_5s_window(self):
        """Test frame_rate calculation for 32kHz audio with 5s window."""
        frame_rate = compute_beats_frame_rate(sample_rate=32000, window_duration=5.0)
        # 32000 * 5 = 160000 samples
        # t_fbank = (160000 - 400) // 160 + 1 = 998
        # t_patched = 998 // 16 = 62
        # frame_rate = 62 / 5.0 = 12.4 Hz
        assert frame_rate == 12.4

    def test_frame_rate_16khz_5s_window(self):
        """Test frame_rate calculation for 16kHz audio with 5s window."""
        frame_rate = compute_beats_frame_rate(sample_rate=16000, window_duration=5.0)
        # 16000 * 5 = 80000 samples
        # t_fbank = (80000 - 400) // 160 + 1 = 498
        # t_patched = 498 // 16 = 31
        # frame_rate = 31 / 5.0 = 6.2 Hz
        assert frame_rate == 6.2

    def test_frame_rate_32khz_10s_window(self):
        """Test frame_rate calculation for 32kHz audio with 10s window."""
        frame_rate = compute_beats_frame_rate(sample_rate=32000, window_duration=10.0)
        # 32000 * 10 = 320000 samples
        # t_fbank = (320000 - 400) // 160 + 1 = 1998
        # t_patched = 1998 // 16 = 124
        # frame_rate = 124 / 10.0 = 12.4 Hz
        assert frame_rate == 12.4

    def test_frame_rate_consistency_across_window_sizes(self):
        """Test that frame_rate is approximately consistent across window sizes."""
        fr_5s = compute_beats_frame_rate(sample_rate=32000, window_duration=5.0)
        fr_10s = compute_beats_frame_rate(sample_rate=32000, window_duration=10.0)
        fr_20s = compute_beats_frame_rate(sample_rate=32000, window_duration=20.0)

        # All should be approximately 12.4 Hz for 32kHz audio
        assert abs(fr_5s - 12.4) < 0.1
        assert abs(fr_10s - 12.4) < 0.1
        assert abs(fr_20s - 12.4) < 0.1

    def test_frame_rate_produces_correct_frame_count(self):
        """Test that frame_rate * duration gives correct number of frames."""
        sample_rate = 32000
        window_duration = 5.0
        frame_rate = compute_beats_frame_rate(sample_rate, window_duration)

        expected_frames = int(frame_rate * window_duration)
        assert expected_frames == 62

    def test_frame_rate_not_theoretical_value(self):
        """Test that we don't use the theoretical (incorrect) frame_rate.

        The old theoretical formula was: SR_SCALE_FACTOR * sample_rate
        where SR_SCALE_FACTOR = 100 / 16 / 16000 = 0.000390625
        For 32kHz: 0.000390625 * 32000 = 12.5 Hz (WRONG!)
        """
        theoretical_frame_rate = (100 / 16 / 16000) * 32000  # 12.5 Hz
        actual_frame_rate = compute_beats_frame_rate(sample_rate=32000, window_duration=5.0)

        assert theoretical_frame_rate != actual_frame_rate
        assert theoretical_frame_rate == 12.5  # The old incorrect value
        assert actual_frame_rate == 12.4  # The correct value

    def test_frames_per_window_is_integer(self):
        """Test that frames per window is a clean integer."""
        sample_rate = 32000
        window_duration = 5.0
        frame_rate = compute_beats_frame_rate(sample_rate, window_duration)

        frames_per_window = frame_rate * window_duration

        assert frames_per_window == int(frames_per_window)
        assert frames_per_window == 62


# --- Tests for Timing Accuracy ---


class TestTimingAccuracy:
    """Tests for timing accuracy with the correct frame_rate."""

    def test_no_cumulative_offset_over_long_audio(self):
        """Test that predictions don't accumulate timing offset.

        This was the original bug: using 12.5 Hz instead of 12.4 Hz caused
        predictions to appear ~2.4 seconds early on 300-second audio.
        """
        sample_rate = 32000
        window_duration = 5.0
        audio_duration = 300.0  # 5 minutes

        frame_rate = compute_beats_frame_rate(sample_rate, window_duration)

        n_windows = int(audio_duration / window_duration)
        frames_per_window = int(frame_rate * window_duration)
        total_frames = n_windows * frames_per_window

        implied_duration = total_frames / frame_rate

        # With correct frame_rate, should be exact
        assert implied_duration == audio_duration

    def test_old_frame_rate_would_cause_offset(self):
        """Verify that the old incorrect frame_rate WOULD cause timing issues."""
        sample_rate = 32000
        window_duration = 5.0
        audio_duration = 300.0

        old_frame_rate = 12.5  # Old incorrect value

        n_windows = int(audio_duration / window_duration)
        actual_frame_rate = compute_beats_frame_rate(sample_rate, window_duration)
        actual_frames_per_window = int(actual_frame_rate * window_duration)
        total_frames = n_windows * actual_frames_per_window

        implied_duration_old = total_frames / old_frame_rate

        assert implied_duration_old < audio_duration
        offset = audio_duration - implied_duration_old
        assert abs(offset - 2.4) < 0.01  # ~2.4 second offset

    def test_frame_to_time_conversion_accuracy(self):
        """Test that frame-to-time conversion is accurate at various positions."""
        frame_rate = compute_beats_frame_rate(sample_rate=32000, window_duration=5.0)

        test_times = [0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0]

        for expected_time in test_times:
            frame_idx = int(expected_time * frame_rate)
            recovered_time = frame_idx / frame_rate

            frame_duration = 1.0 / frame_rate
            assert abs(recovered_time - expected_time) <= frame_duration


# --- Tests for AGGREGATION_STRATEGIES Registry ---


class TestAggregationStrategies:
    """Tests for the AGGREGATION_STRATEGIES registry."""

    def test_all_strategies_exist(self):
        """Verify all expected strategies are in the registry."""
        expected_strategies = ["average", "all_frames", "concat"]

        for strategy_name in expected_strategies:
            assert strategy_name in AGGREGATION_STRATEGIES

    def test_strategies_have_required_keys(self):
        """Each strategy should have fn, frame_rate_multiplier, hidden_dim_multiplier."""
        required_keys = ["fn", "frame_rate_multiplier", "hidden_dim_multiplier"]

        for name, strategy in AGGREGATION_STRATEGIES.items():
            for key in required_keys:
                assert key in strategy, f"Strategy '{name}' missing key '{key}'"

    @pytest.mark.parametrize(
        "strategy_name,expected_frame_rate_mult,expected_hidden_mult",
        [
            ("average", 1, 1),
            ("all_frames", BEATSEncoderConfig.NUM_FREQ_PATCHES, 1),
            ("concat", 1, BEATSEncoderConfig.NUM_FREQ_PATCHES),
        ],
    )
    def test_strategy_multipliers(self, strategy_name, expected_frame_rate_mult, expected_hidden_mult):
        """Verify each strategy has correct multipliers."""
        strategy = AGGREGATION_STRATEGIES[strategy_name]

        assert strategy["frame_rate_multiplier"] == expected_frame_rate_mult
        assert strategy["hidden_dim_multiplier"] == expected_hidden_mult


# --- Tests for Patch Ordering ---


class TestPatchOrdering:
    """Integration tests confirming BEATs patch ordering from waveform inputs.

    BEATs' extract_features pipeline applies these shape manipulations to
    produce the patch sequence fed to the transformer:

        fbank [B, T_fbank, 128]
        → unsqueeze(1)             [B, 1, T_fbank, 128]
        → Conv2d patch_embedding   [B, D, T_patched, F_patched]  ← spatial layout known here
        → reshape(B, D, -1)        [B, D, T*F]                   ← time-outer, freq-inner
        → transpose(1, 2)          [B, T*F, D]                   ← sequence to transformer

    By stopping at the Conv2d output we have ground-truth spatial knowledge of each
    patch's (time, freq) position, from which we derive the expected aggregation
    results and compare to what _aggregate_average / _aggregate_concat produce on the
    equivalent flattened sequence.
    """

    def _get_patch_embedding(
        self, raw_beats_model: torch.nn.Module, waveform: torch.Tensor
    ) -> torch.Tensor:
        """Run fbank preprocessing and patch embedding (Conv2d), not including transformer.

        raw_beats_model is an avex Model wrapper; the actual BEATs
        instance with preprocess() and patch_embedding is at .backbone.
        """
        beats = raw_beats_model.backbone
        with torch.no_grad():
            fbank = beats.preprocess(waveform).float()  # [B, T_fbank, 128]
            fbank = fbank.unsqueeze(1)                  # [B, 1, T_fbank, 128]
            patches = beats.patch_embedding(fbank)      # [B, D, T_patched, F_patched]
        return patches

    def test_time_patch_count_matches_expected(self, raw_beats_model):
        """T_patched from the Conv2d must match the formula used in compute_beats_frame_rate.

        For a 5 s waveform at 32 kHz:
            window_samples = 160000
            t_fbank  = (160000 - BEATSEncoderConfig.FBANK_WINDOW_SAMPLES) // BEATSEncoderConfig.FBANK_HOP_SAMPLES + 1 = 998
            t_patched = 998 // BEATSEncoderConfig.PATCH_SIZE = 62
        """
        waveform = torch.zeros(1, int(WINDOW_DURATION * SAMPLE_RATE))
        patches = self._get_patch_embedding(raw_beats_model, waveform)

        window_samples = int(WINDOW_DURATION * SAMPLE_RATE)
        t_fbank = (window_samples - BEATSEncoderConfig.FBANK_WINDOW_SAMPLES) // BEATSEncoderConfig.FBANK_HOP_SAMPLES + 1
        expected_t_patched = t_fbank // BEATSEncoderConfig.PATCH_SIZE

        assert patches.shape[2] == expected_t_patched
        assert patches.shape[3] == BEATSEncoderConfig.NUM_FREQ_PATCHES

    def test_aggregate_average_groups_by_time_step(self, raw_beats_model):
        """_aggregate_average must average the F freq patches belonging to each time step.

        Ground truth: average the Conv2d output [B, D, T, F] over the freq dimension
        and permute to [B, T, D].  The flat sequence is reconstructed from the Conv2d
        output via the same reshape/transpose as extract_features, so any mismatch
        indicates the aggregation is grouping patches incorrectly.
        """
        aggregate_fn = AGGREGATION_STRATEGIES["average"]["fn"]

        waveform = torch.randn(1, int(WINDOW_DURATION * SAMPLE_RATE))
        patches = self._get_patch_embedding(raw_beats_model, waveform)  # [1, D, T, F]

        B, D, T, F = patches.shape
        assert F == BEATSEncoderConfig.NUM_FREQ_PATCHES

        # Ground truth: average over the freq dim of the spatial layout [B, D, T, F]
        expected = patches.mean(dim=3).permute(0, 2, 1)  # [1, T, D]

        # Replicate the extract_features flatten + transpose to get the sequence
        flat = patches.reshape(B, D, -1).transpose(1, 2)  # [1, T*F, D]

        result = aggregate_fn(flat)  # [1, T, D]

        torch.testing.assert_close(result, expected)

    def test_aggregate_concat_groups_by_time_step(self, raw_beats_model):
        """_aggregate_concat must concatenate the F freq patches belonging to each time step.

        Ground truth: permute Conv2d output to [B, T, F, D] then flatten F and D.
        """
        aggregate_fn = AGGREGATION_STRATEGIES["concat"]["fn"]

        waveform = torch.randn(1, int(WINDOW_DURATION * SAMPLE_RATE))
        patches = self._get_patch_embedding(raw_beats_model, waveform)  # [1, D, T, F]

        B, D, T, F = patches.shape
        assert F == BEATSEncoderConfig.NUM_FREQ_PATCHES

        # Ground truth: for each time step, concatenate freq patch embeddings along D
        expected = patches.permute(0, 2, 3, 1).reshape(B, T, F * D)  # [1, T, F*D]

        # Replicate the extract_features flatten + transpose to get the sequence
        flat = patches.reshape(B, D, -1).transpose(1, 2)  # [1, T*F, D]

        result = aggregate_fn(flat)  # [1, T, F*D]

        torch.testing.assert_close(result, expected)


# --- Tests for Edge Cases ---


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_single_batch(self, beats_encoder):
        """Test with batch size of 1."""
        window_samples = int(WINDOW_DURATION * SAMPLE_RATE)
        audio = torch.randn(1, window_samples)

        with torch.no_grad():
            output = beats_encoder(audio)

        assert output.shape[0] == 1
        assert output.shape[1] == 62
        assert output.shape[2] == BEATSEncoderConfig.HIDDEN_DIM

    def test_large_batch(self, beats_encoder):
        """Test with larger batch size."""
        batch_size = 16
        window_samples = int(WINDOW_DURATION * SAMPLE_RATE)
        audio = torch.randn(batch_size, window_samples)

        with torch.no_grad():
            output = beats_encoder(audio)

        assert output.shape[0] == batch_size
        assert output.shape[1] == 62

    def test_silent_audio(self, beats_encoder):
        """Test with silent (zero) audio."""
        window_samples = int(WINDOW_DURATION * SAMPLE_RATE)
        audio = torch.zeros(1, window_samples)

        with torch.no_grad():
            output = beats_encoder(audio)

        assert output.shape == (1, 62, BEATSEncoderConfig.HIDDEN_DIM)
        assert torch.isfinite(output).all()

    def test_loud_audio(self, beats_encoder):
        """Test with very loud audio (large amplitude)."""
        window_samples = int(WINDOW_DURATION * SAMPLE_RATE)
        audio = torch.randn(1, window_samples) * 100  # Very loud

        with torch.no_grad():
            output = beats_encoder(audio)

        assert output.shape == (1, 62, BEATSEncoderConfig.HIDDEN_DIM)
        assert torch.isfinite(output).all()

    def test_constant_audio(self, beats_encoder):
        """Test with constant (DC) audio."""
        window_samples = int(WINDOW_DURATION * SAMPLE_RATE)
        audio = torch.ones(1, window_samples) * 0.5

        with torch.no_grad():
            output = beats_encoder(audio)

        assert output.shape == (1, 62, BEATSEncoderConfig.HIDDEN_DIM)
        assert torch.isfinite(output).all()

    def test_short_window_duration(self, raw_beats_model):
        """Test with shorter window duration."""
        duration = 1.0
        encoder = BEATSEncoder(
            model=raw_beats_model,
            sample_rate=SAMPLE_RATE,
            window_duration=duration,
            aggregation="average",
        )

        window_samples = int(duration * SAMPLE_RATE)
        audio = torch.randn(1, window_samples)

        with torch.no_grad():
            output = encoder(audio)

        expected_frames = int(encoder.output_frame_rate * duration)
        assert output.shape[1] == expected_frames

    def test_encoder_is_nn_module(self, beats_encoder):
        """Test that BEATSEncoder is a proper nn.Module."""
        assert isinstance(beats_encoder, torch.nn.Module)
        assert hasattr(beats_encoder, 'parameters')
        assert hasattr(beats_encoder, 'eval')
        assert hasattr(beats_encoder, 'train')

    def test_encoder_can_be_moved_to_device(self, beats_encoder):
        """Test that encoder can be moved between devices."""
        # Move to CPU (should always work)
        encoder_cpu = beats_encoder.to("cpu")
        assert next(encoder_cpu.parameters()).device.type == "cpu"

        # Test forward still works after move
        window_samples = int(WINDOW_DURATION * SAMPLE_RATE)
        audio = torch.randn(1, window_samples)

        with torch.no_grad():
            output = encoder_cpu(audio)

        assert output.shape == (1, 62, BEATSEncoderConfig.HIDDEN_DIM)


# --- Tests for Raw BEATs Output Shape ---


class TestRawBEATsOutputShape:
    """Test the actual raw BEATs encoder output shapes (before aggregation)."""

    def test_beats_raw_output_shape(self, raw_beats_model):
        """Test that raw BEATs produces expected number of patches for 5s audio."""
        window_samples = int(WINDOW_DURATION * SAMPLE_RATE)
        audio = torch.randn(1, window_samples)

        with torch.no_grad():
            embeddings = raw_beats_model(audio, padding_mask=None)

        # Expected patches: 62 time × 8 freq = 496
        expected_time_patches = 62
        expected_freq_patches = BEATSEncoderConfig.NUM_FREQ_PATCHES
        expected_total = expected_time_patches * expected_freq_patches

        assert embeddings.shape[0] == 1  # Batch size
        assert embeddings.shape[1] == expected_total, (
            f"Expected {expected_total} patches, got {embeddings.shape[1]}"
        )
        assert embeddings.shape[2] == BEATSEncoderConfig.HIDDEN_DIM

    def test_beats_output_scales_with_duration(self, raw_beats_model):
        """Test that BEATs output scales correctly with audio duration."""
        results = {}
        for duration in [5.0, 10.0]:
            samples = int(duration * SAMPLE_RATE)
            audio = torch.randn(1, samples)

            with torch.no_grad():
                embeddings = raw_beats_model(audio, padding_mask=None)

            results[duration] = embeddings.shape[1]

        # 10s should produce approximately 2x patches of 5s
        ratio = results[10.0] / results[5.0]
        assert 1.9 < ratio < 2.1, f"Expected ~2x patches, got {ratio:.2f}x"


# --- Integration Tests for Training/Inference Consistency ---


class TestTrainingInferenceConsistency:
    """Test that encoder frame_rate is consistent with training data preparation."""

    def test_frame_rate_matches_actual_encoder_output(self, raw_beats_model):
        """Test that stored frame_rate matches actual frames produced by encoder."""
        for aggregation in ["average", "all_frames", "concat"]:
            encoder = BEATSEncoder(
                model=raw_beats_model,
                sample_rate=SAMPLE_RATE,
                window_duration=WINDOW_DURATION,
                aggregation=aggregation,
            )

            audio = torch.randn(1, int(WINDOW_DURATION * SAMPLE_RATE))

            with torch.no_grad():
                output = encoder(audio)

            actual_frames = output.shape[1]
            expected_frames = int(encoder.output_frame_rate * WINDOW_DURATION)

            assert actual_frames == expected_frames, (
                f"{aggregation}: expected {expected_frames} frames, got {actual_frames}"
            )
