"""Tests for shared audio-handling utilities."""

import base64

import numpy as np
import pytest

from esp_research.utils.audio import bytes_to_base64_string as encode_audio
from esp_research.utils.audio import coerce_audio_bytes, float_array_to_wav


class TestCoerceAudioBytes:
    """Unit tests for `coerce_audio_bytes`."""

    def test_bytes_passthrough(self) -> None:
        raw = b"raw_audio_data"
        assert coerce_audio_bytes(raw, "audio") == raw

    def test_empty_bytes_passthrough(self) -> None:
        assert coerce_audio_bytes(b"", "audio") == b""

    def test_list_of_floats(self) -> None:
        data = [0.1, 0.2, 0.3]
        result = coerce_audio_bytes(data, "audio")
        expected = np.array(data, dtype=np.float32).tobytes()
        assert result == expected

    def test_numpy_float32_array(self) -> None:
        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = coerce_audio_bytes(arr, "audio")
        assert result == arr.tobytes()

    def test_numpy_float64_array_cast_to_float32(self) -> None:
        arr = np.array([1.0, 2.0], dtype=np.float64)
        result = coerce_audio_bytes(arr, "audio")
        expected = arr.astype(np.float32).tobytes()
        assert result == expected

    def test_empty_list_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Empty audio array"):
            coerce_audio_bytes([], "audio")

    def test_empty_numpy_array_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Empty audio array"):
            coerce_audio_bytes(np.array([], dtype=np.float32), "audio")

    def test_non_floating_numpy_raises_type_error(self) -> None:
        arr = np.array([1, 2, 3], dtype=np.int32)
        with pytest.raises(TypeError, match="floating-point dtype"):
            coerce_audio_bytes(arr, "audio")

    def test_non_floating_numpy_error_includes_key_name(self) -> None:
        arr = np.array([1], dtype=np.int64)
        with pytest.raises(TypeError, match="my_audio"):
            coerce_audio_bytes(arr, "my_audio")

    def test_unsupported_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="audio_field"):
            coerce_audio_bytes("not bytes", "audio_field")

    def test_unsupported_type_error_includes_key_name(self) -> None:
        with pytest.raises(TypeError, match="my_key"):
            coerce_audio_bytes(42, "my_key")


class TestEncodeAudio:
    """Unit tests for `encode_audio`."""

    def test_round_trip(self) -> None:
        raw = b"hello audio world"
        encoded = encode_audio(raw)
        assert base64.b64decode(encoded) == raw

    def test_returns_string(self) -> None:
        assert isinstance(encode_audio(b"data"), str)

    def test_empty_bytes(self) -> None:
        encoded = encode_audio(b"")
        assert base64.b64decode(encoded) == b""


class TestFloat32RoundTrip:
    """Verify the client→server float32 encoding round-trip.

    The sender calls `coerce_audio_bytes` then `encode_audio`; the receiver
    calls `base64.b64decode` then `np.frombuffer(..., dtype=np.float32)`.
    The recovered array must be exactly equal to the original float32 data.
    """

    def _round_trip(self, audio_data: object) -> np.ndarray:
        raw_bytes = coerce_audio_bytes(audio_data, "audio")
        b64 = encode_audio(raw_bytes)
        decoded_bytes = base64.b64decode(b64)
        return np.frombuffer(decoded_bytes, dtype=np.float32)

    def test_numpy_float32_array(self) -> None:
        original = np.array([0.0, 0.1, -0.5, 1.0, -1.0], dtype=np.float32)
        recovered = self._round_trip(original)
        np.testing.assert_array_equal(recovered, original)

    def test_list_of_floats(self) -> None:
        data = [0.0, 0.1, -0.5, 1.0, -1.0]
        original = np.array(data, dtype=np.float32)
        recovered = self._round_trip(data)
        np.testing.assert_array_equal(recovered, original)

    def test_float64_input_cast_to_float32(self) -> None:
        # coerce_audio_bytes casts float64 → float32 before serialising;
        # the recovered array should match the cast result, not the original float64.
        original_f64 = np.array([0.1, 0.5, -0.3], dtype=np.float64)
        expected = original_f64.astype(np.float32)
        recovered = self._round_trip(original_f64)
        np.testing.assert_array_equal(recovered, expected)

    def test_single_sample(self) -> None:
        original = np.array([0.42], dtype=np.float32)
        recovered = self._round_trip(original)
        np.testing.assert_array_equal(recovered, original)

    def test_boundary_values(self) -> None:
        original = np.array([-1.0, 1.0, 0.0], dtype=np.float32)
        recovered = self._round_trip(original)
        np.testing.assert_array_equal(recovered, original)


class TestFloatArrayToWav:
    """Unit tests for `float_array_to_wav`."""

    def test_passthrough_bytes(self) -> None:
        raw = b"RIFF....WAVEfmt "
        assert float_array_to_wav(raw, "audio", 16000) == raw

    def test_float_list_to_wav(self) -> None:
        data = [[0.0, 0.5, -0.5]]
        wav_bytes = float_array_to_wav(data, "audio", 16000)
        assert wav_bytes.startswith(b"RIFF") and b"WAVE" in wav_bytes

    def test_numpy_float_array_to_wav(self) -> None:
        arr = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        wav_bytes = float_array_to_wav(arr, "audio", 16000)
        assert wav_bytes.startswith(b"RIFF") and b"WAVE" in wav_bytes
