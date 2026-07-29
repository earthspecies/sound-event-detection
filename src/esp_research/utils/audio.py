"""Shared audio-handling utilities."""

import base64
import io
import wave

import numpy as np


def coerce_audio_bytes(audio_data: object, audio_key: str) -> bytes:
    """Coerce audio data from various formats into raw bytes.

    Parameters
    ----------
    audio_data : object
        Raw bytes, a list of floats, or a NumPy float array.
    audio_key : str
        Name of the audio field, used in error messages.

    Returns
    -------
    bytes
        Raw audio bytes.

    Raises
    ------
    TypeError
        If the data is a NumPy array with a non-floating dtype, or if
        the data is not bytes, list, or NumPy array.
    ValueError
        If the list or array is empty.
    """
    if isinstance(audio_data, (list, np.ndarray)):
        if len(audio_data) == 0:
            raise ValueError("Empty audio array provided")

        if isinstance(audio_data, list):
            audio_array = np.array(audio_data, dtype=np.float32)
        else:
            if not np.issubdtype(audio_data.dtype, np.floating):
                raise TypeError(f"Numpy array in {audio_key} must have a floating-point dtype.")
            audio_array = audio_data.astype(np.float32)

        return audio_array.tobytes()

    if not isinstance(audio_data, bytes):
        raise TypeError(f"{audio_key} must contain raw audio bytes.")

    return audio_data


def float_array_to_wav(audio_data: object, audio_key: str, sample_rate: int) -> bytes:
    """Convert float audio samples to WAV-formatted bytes.

    Parameters
    ----------
    audio_data : object
        A list of floats or a NumPy float array of audio samples in the
        range [-1.0, 1.0], or raw ``bytes`` (returned as-is since they
        are assumed to already be in a file format).
    audio_key : str
        Name of the audio field, used in error messages.
    sample_rate : int
        Sample rate in Hz (e.g. 16000, 44100).

    Returns
    -------
    bytes
        WAV file bytes (PCM 16-bit) if the input was float data, or
        the original bytes unchanged.

    Raises
    ------
    TypeError
        If the data is a NumPy array with a non-floating dtype, or if
        the data is not bytes, list, or NumPy array.
    ValueError
        If the list or array is empty.
    """
    if isinstance(audio_data, bytes):
        return audio_data

    if isinstance(audio_data, (list, np.ndarray)):
        if len(audio_data) == 0:
            raise ValueError("Empty audio array provided")

        if isinstance(audio_data, list):
            audio_array = np.array(audio_data, dtype=np.float32)
        else:
            if not np.issubdtype(audio_data.dtype, np.floating):
                raise TypeError(f"Numpy array in {audio_key} must have a floating-point dtype.")
            audio_array = audio_data.astype(np.float32)

        # Squeeze batch dimensions (e.g. from collation with batch_size=1).
        audio_array = audio_array.squeeze()

        # Clip and convert to 16-bit PCM
        audio_array = np.clip(audio_array, -1.0, 1.0)
        pcm16 = (audio_array * 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()

    raise TypeError(f"{audio_key} must contain raw audio bytes, a list of floats, or a NumPy array.")


def bytes_to_base64_string(data: bytes) -> str:
    """Base64-encode bytes to a UTF-8 string.

    Parameters
    ----------
    data : bytes
        Raw bytes to encode.

    Returns
    -------
    str
        Base64-encoded string.
    """
    return base64.b64encode(data).decode("utf-8")
