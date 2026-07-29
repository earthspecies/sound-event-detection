"""The ``v0minimal`` acoustic-feature definitions (pure math).

These two pure functions (`highpass_filter` and `compute_v0minimal_features`)
define the ``v0minimal`` acoustic feature set that the `features` stage computes
per detected event. They are a verbatim copy of the feature math from the
standalone ``earthspecies/sound-event-detection`` repository at commit
``6bed1c2`` ("add acoustic features",
``sound_event_detection/inference/add_acoustic_features.py``) — the definition
behind the original ``..._geo_features_v0minimal`` selection tables.

Vendored (rather than imported) because the source lives in a *separate*
repository whose top-level package is also called ``sound_event_detection`` and
is not a dependency of esp-research. The functions are pure (only
``librosa``/``numpy``/``scipy``), so copying them is safe and self-contained. If
the upstream feature definition changes, update this file from the same source
and note the new commit here.

STFT parameters: ``n_fft=1024``, ``hop_length=512``. Audio is high-pass filtered
at 50 Hz before any feature is computed.
"""

import librosa
import numpy as np
import scipy.signal

__all__ = [
    "FEATURE_COLS",
    "compute_v0minimal_features",
    "highpass_filter",
]

#: Feature column names, in order, as appended to the enriched selection table.
FEATURE_COLS: tuple[str, ...] = (
    "duration_s",
    "rms_amplitude",
    "zero_crossing_rate",
    "mean_dominant_freq_hz",
    "std_dominant_freq_hz",
    "dominant_freq_range_hz",
    "dominant_freq_slope",
    "mean_spectral_entropy",
    "mean_spectral_flatness",
    "mean_spectral_centroid_hz",
    "mean_spectral_bandwidth_hz",
    "mean_spectral_rolloff_hz",
    "mean_spectral_flux",
)

_N_FFT = 1024
_HOP_LENGTH = 512
_HIGHPASS_HZ = 50.0
_HIGHPASS_ORDER = 4


def highpass_filter(audio: np.ndarray, sr: float, cutoff_hz: float = _HIGHPASS_HZ) -> np.ndarray:
    """Apply a zero-phase Butterworth high-pass filter.

    Parameters
    ----------
    audio : np.ndarray, shape (T,)
        Mono audio signal.
    sr : float
        Sample rate in Hz.
    cutoff_hz : float
        High-pass cutoff frequency in Hz. Default 50.0.

    Returns
    -------
    np.ndarray
        Filtered audio, same shape as input. Returns input unchanged if the
        signal is too short for the filter's padding requirement.
    """
    nyq = sr / 2.0
    sos = scipy.signal.butter(_HIGHPASS_ORDER, cutoff_hz / nyq, btype="high", output="sos")
    min_len = 3 * (2 * sos.shape[0]) + 1  # sosfiltfilt padding requirement
    if len(audio) <= min_len:
        return audio
    return scipy.signal.sosfiltfilt(sos, audio).astype(audio.dtype)


def compute_v0minimal_features(
    clip: np.ndarray,
    sr: float,
    duration: float,
) -> dict[str, float]:
    """Compute v0minimal acoustic features for a single audio clip.

    The clip must already be high-pass filtered.

    Parameters
    ----------
    clip : np.ndarray, shape (T,)
        Mono audio clip.
    sr : float
        Sample rate in Hz.
    duration : float
        Event duration in seconds (end_time - begin_time from the selection table).

    Returns
    -------
    dict[str, float]
        Mapping of feature name to value. All spectral features are NaN for
        clips shorter than n_fft samples.
    """
    nan_row: dict[str, float] = {col: float("nan") for col in FEATURE_COLS}
    nan_row["duration_s"] = duration

    if len(clip) < _N_FFT:
        return nan_row

    clip_f64 = clip.astype(np.float64)
    rms = float(np.sqrt(np.mean(clip_f64**2)))

    zcr = float(np.mean(librosa.feature.zero_crossing_rate(clip, hop_length=_HOP_LENGTH)))

    stft = librosa.stft(clip, n_fft=_N_FFT, hop_length=_HOP_LENGTH)
    mag_spec = np.abs(stft)  # (F, T_frames) — amplitude spectrogram
    power_spec = mag_spec**2  # (F, T_frames) — power spectrogram
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_N_FFT)  # (F,) in Hz

    n_frames = power_spec.shape[1]

    # Dominant frequency: argmax energy per frame → Hz
    dom_bins = np.argmax(power_spec, axis=0)  # (T_frames,)
    dom_freqs = freqs[dom_bins]  # (T_frames,) in Hz

    mean_dom_freq = float(np.mean(dom_freqs))
    std_dom_freq = float(np.std(dom_freqs))
    dom_freq_range = float(dom_freqs.max() - dom_freqs.min()) if n_frames > 1 else 0.0

    if n_frames > 1:
        t_sec = np.arange(n_frames) * (_HOP_LENGTH / sr)
        dom_freq_slope = float(np.polyfit(t_sec, dom_freqs, 1)[0])
    else:
        dom_freq_slope = 0.0

    # Shannon entropy per frame: -sum(p * log(p)), then mean
    col_sums = power_spec.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums == 0.0, 1e-12, col_sums)
    p = np.clip(power_spec / col_sums, 1e-12, None)
    mean_entropy = float(np.mean(-np.sum(p * np.log(p), axis=0)))

    mean_flatness = float(np.mean(librosa.feature.spectral_flatness(S=mag_spec)))
    mean_centroid = float(np.mean(librosa.feature.spectral_centroid(S=mag_spec, sr=sr)))
    mean_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(S=mag_spec, sr=sr)))
    mean_rolloff = float(np.mean(librosa.feature.spectral_rolloff(S=mag_spec, sr=sr)))

    # Spectral flux: mean L2 norm of sequential power-spectrum frame differences
    if n_frames > 1:
        diffs = np.diff(power_spec, axis=1)
        mean_flux = float(np.mean(np.sqrt(np.sum(diffs**2, axis=0))))
    else:
        mean_flux = 0.0

    return {
        "duration_s": duration,
        "rms_amplitude": rms,
        "zero_crossing_rate": zcr,
        "mean_dominant_freq_hz": mean_dom_freq,
        "std_dominant_freq_hz": std_dom_freq,
        "dominant_freq_range_hz": dom_freq_range,
        "dominant_freq_slope": dom_freq_slope,
        "mean_spectral_entropy": mean_entropy,
        "mean_spectral_flatness": mean_flatness,
        "mean_spectral_centroid_hz": mean_centroid,
        "mean_spectral_bandwidth_hz": mean_bandwidth,
        "mean_spectral_rolloff_hz": mean_rolloff,
        "mean_spectral_flux": mean_flux,
    }
