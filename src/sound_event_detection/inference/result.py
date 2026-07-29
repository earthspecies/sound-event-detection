"""One recording's large-scale-inference output: the record and how it is stored.

One recording's inference output is an `ItemResult`: the combined framewise
predictions (always), an optional denoised waveform, optional per-source `Stem`s
(audio + framewise predictions), the recording's latitude/longitude, and
optional per-frame focal-quality tracks. The `to_arrays` / `from_arrays` pair is
the single owner of the flat-array key convention in both directions, so readers
go through `from_arrays` rather than hard-coding key names.

The detail ladder decides how much is kept versus re-derived downstream:

- ``preds`` -> combined predictions only (smallest).
- ``denoised`` -> combined predictions + a denoised waveform (threshold baked in).
- ``stems`` -> the per-stem core (stem audio + per-stem predictions) plus the
  baked denoised waveform, so `combined` and any-threshold `denoise` re-derive
  on read.

Encoding is lossy: predictions below a max-probability threshold are dropped and
the rest cast to float16 (`encode_preds`); audio is quantized to int16 PCM and
stored as FLAC (`encode_audio_flac`), which is lossless over that int16 and
compresses the silence-heavy audio well. Frame rates are stored float64 so they
round-trip exactly.
"""

import io
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import soundfile as sf

from esp_research.protocols.detector import DetectorOutput

__all__ = [
    "DEFAULT_AUDIO_SAMPLE_RATE",
    "DEFAULT_PREDS_THRESHOLD",
    "Detail",
    "ItemResult",
    "Stem",
    "decode_audio",
    "decode_audio_flac",
    "decode_preds",
    "encode_audio",
    "encode_audio_flac",
    "encode_preds",
]

#: The detail ladder: how much of the core to persist. ``preds`` is always
#: emitted; each higher rung keeps strictly more (see module docstring).
Detail = Literal["preds", "denoised", "stems"]

#: Default max-probability threshold for `encode_preds`: a class is dropped from
#: a stored prediction track if its maximum probability over all frames is below
#: this. Matches the old ``compress_predictions`` default.
DEFAULT_PREDS_THRESHOLD = 0.05

#: Full-scale value for the int16 PCM audio codec (`encode_audio` /
#: `decode_audio`).
_PCM_SCALE = 32767.0

#: Default sample rate written into the FLAC audio container. The separator (and
#: hence the stems / denoised waveform) run at 22.05 kHz.
DEFAULT_AUDIO_SAMPLE_RATE = 22050


def encode_preds(output: DetectorOutput, threshold: float = DEFAULT_PREDS_THRESHOLD) -> dict[str, np.ndarray]:
    """Compress a single-recording `DetectorOutput` into a self-contained array group.

    Drops classes whose maximum probability across all frames is below
    `threshold`, casts the surviving probabilities to float16, and keeps the
    matching class labels.

    Parameters
    ----------
    output : DetectorOutput
        Frame-level predictions of shape ``(1, frames, classes)`` (one recording).
    threshold : float
        Minimum max-probability for a class to be retained.

    Returns
    -------
    dict[str, np.ndarray]
        Group with keys ``predictions`` (``(frames, kept)`` float16),
        ``classes`` (``(kept,)`` unicode), and ``frame_rate`` (float64 scalar).

    Raises
    ------
    ValueError
        If `output.predictions` is not a single-recording ``(1, frames, classes)`` array.
    """
    preds = output.predictions
    if preds.ndim != 3 or preds.shape[0] != 1:
        raise ValueError(
            f"encode_preds expects single-recording predictions (1, frames, classes), got shape {preds.shape}"
        )

    values = preds[0]  # (frames, classes)
    class_names = list(output.class_names)
    if values.shape[1] > 0:
        keep = values.max(axis=0) >= threshold
    else:
        keep = np.zeros(0, dtype=bool)

    kept_values = np.ascontiguousarray(values[:, keep], dtype=np.float16)
    kept_names = [name for name, keep_it in zip(class_names, keep, strict=True) if keep_it]
    return {
        "predictions": kept_values,
        "classes": np.array(kept_names, dtype=np.str_),
        "frame_rate": np.asarray(output.frame_rate, dtype=np.float64),
    }


def decode_preds(group: Mapping[str, np.ndarray]) -> DetectorOutput:
    """Invert `encode_preds`, reconstructing a single-recording `DetectorOutput`.

    Parameters
    ----------
    group : Mapping[str, np.ndarray]
        Group as produced by `encode_preds`, with keys ``predictions``,
        ``classes``, and ``frame_rate``.

    Returns
    -------
    DetectorOutput
        Predictions of shape ``(1, frames, kept)`` (float16 preserved), with the
        stored frame_rate and kept class labels.
    """
    predictions = np.ascontiguousarray(group["predictions"])[np.newaxis]  # (1, frames, kept)
    class_names = [str(name) for name in group["classes"].tolist()]
    return DetectorOutput(predictions=predictions, frame_rate=float(group["frame_rate"]), class_names=class_names)


def encode_audio(audio: np.ndarray) -> np.ndarray:
    """Encode a float waveform in [-1, 1] as int16 PCM.

    Parameters
    ----------
    audio : np.ndarray
        Waveform, values expected in [-1, 1]. Out-of-range values are clipped.

    Returns
    -------
    np.ndarray
        int16 PCM samples, same shape as `audio`.
    """
    scaled = np.round(np.asarray(audio, dtype=np.float32) * _PCM_SCALE)
    return np.clip(scaled, -_PCM_SCALE - 1, _PCM_SCALE).astype(np.int16)


def decode_audio(pcm: np.ndarray) -> np.ndarray:
    """Invert `encode_audio`, reconstructing a float32 waveform in [-1, 1].

    Parameters
    ----------
    pcm : np.ndarray
        int16 PCM samples.

    Returns
    -------
    np.ndarray
        float32 waveform, same shape as `pcm`. Lossy by ~one int16 step versus
        the pre-encode signal.
    """
    return np.asarray(pcm, dtype=np.float32) / _PCM_SCALE


def encode_audio_flac(audio: np.ndarray, sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE) -> np.ndarray:
    """Quantize a float waveform to int16 PCM and encode it as FLAC bytes.

    FLAC is a lossless container over the int16 primitive (`encode_audio`), so
    the stored bytes decode back to exactly the int16 samples — the audio is no
    lossier than the plain int16 codec, but ~0.36x the size for silence-heavy
    recordings.

    Parameters
    ----------
    audio : np.ndarray
        Waveform, values expected in [-1, 1] (out-of-range values are clipped by
        `encode_audio`).
    sample_rate : int
        Sample rate written into the FLAC header.

    Returns
    -------
    np.ndarray
        1-D ``uint8`` array of the FLAC file bytes.
    """
    pcm = encode_audio(audio)
    buffer = io.BytesIO()
    sf.write(buffer, pcm, sample_rate, format="FLAC", subtype="PCM_16")
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8)


def decode_audio_flac(data: np.ndarray) -> np.ndarray:
    """Invert `encode_audio_flac`, reconstructing a float32 waveform in [-1, 1].

    Parameters
    ----------
    data : np.ndarray
        ``uint8`` array of FLAC file bytes, as produced by `encode_audio_flac`.

    Returns
    -------
    np.ndarray
        float32 waveform of shape ``(samples,)`` in [-1, 1] (the int16 samples
        divided by the PCM scale). Lossy by ~one int16 step versus the
        pre-encode signal, exactly as the int16 codec.
    """
    pcm, _sample_rate = sf.read(io.BytesIO(np.asarray(data, dtype=np.uint8).tobytes()), dtype="int16")
    return decode_audio(pcm)


@dataclass(frozen=True)
class Stem:
    """One separated source: its whole-file audio and framewise predictions.

    Attributes
    ----------
    audio : np.ndarray
        Whole-file stem waveform of shape ``(samples,)`` at the separator's
        sample rate.
    preds : DetectorOutput
        The detector's framewise predictions for this stem, shape
        ``(1, frames, classes)``.
    """

    audio: np.ndarray
    preds: DetectorOutput


@dataclass(frozen=True)
class ItemResult:
    """One recording's inference output, ready to encode to flat arrays.

    Optional fields express the detail ladder: `preds` alone is the ``preds``
    rung; adding `denoised` is the ``denoised`` rung; adding `stems` (with
    `denoised` baked in) is the ``stems`` rung.

    Attributes
    ----------
    preds : DetectorOutput
        Combined framewise predictions, shape ``(1, frames, classes)``. Always
        present.
    denoised : np.ndarray or None
        Focal-gated denoised waveform of shape ``(samples,)``, or ``None`` at
        the ``preds`` rung.
    stems : tuple[Stem, ...]
        Per-source stems, or empty except at the ``stems`` rung.
    latitude : float or None
        Recording latitude (decimal degrees). Stored on every shard for the
        downstream geo filter; ``None`` (written as ``nan``) when unknown.
    longitude : float or None
        Recording longitude (decimal degrees); ``None`` (written as ``nan``)
        when unknown.
    focal_detprob : np.ndarray or None
        Per-frame focal detection probability, shape ``(frames,)`` (may contain
        ``nan`` on frames with no gated-in stem). Set only at the denoised /
        stems rungs; ``None`` otherwise.
    focal_nstems : np.ndarray or None
        Per-frame count of gated-in stems, shape ``(frames,)``. Set only at the
        denoised / stems rungs; ``None`` otherwise.
    """

    preds: DetectorOutput
    denoised: np.ndarray | None = None
    stems: tuple[Stem, ...] = field(default_factory=tuple)
    latitude: float | None = None
    longitude: float | None = None
    focal_detprob: np.ndarray | None = None
    focal_nstems: np.ndarray | None = None

    def to_arrays(
        self,
        preds_threshold: float = DEFAULT_PREDS_THRESHOLD,
        audio_sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
    ) -> dict[str, np.ndarray]:
        """Encode this result to a flat ``{key: array}`` dict for one recording.

        This method is the sole owner of the per-recording key convention:
        ``preds_*`` for the combined track, ``latitude`` / ``longitude`` scalars
        (always written; ``nan`` when unknown), ``focal_detprob`` / ``focal_nstems``
        per-frame quality tracks (only when set), ``n_stems`` for the stem count,
        ``denoised`` for the waveform, and ``stem{i}_audio`` / ``stem{i}_preds_*``
        per stem. Audio keys hold FLAC bytes (`encode_audio_flac`). The engine
        namespaces these under a per-recording prefix when packing a shard.

        Parameters
        ----------
        preds_threshold : float
            Max-probability threshold forwarded to `encode_preds` for every
            prediction track (combined and per-stem).
        audio_sample_rate : int
            Sample rate written into the FLAC header for every audio track.

        Returns
        -------
        dict[str, np.ndarray]
            Flat array dict; see `from_arrays` for the inverse.
        """
        arrays: dict[str, np.ndarray] = {}
        for key, value in encode_preds(self.preds, preds_threshold).items():
            arrays[f"preds_{key}"] = value
        lat = float("nan") if self.latitude is None else float(self.latitude)
        lon = float("nan") if self.longitude is None else float(self.longitude)
        arrays["latitude"] = np.asarray(lat, dtype=np.float64)
        arrays["longitude"] = np.asarray(lon, dtype=np.float64)
        if self.focal_detprob is not None:
            arrays["focal_detprob"] = np.ascontiguousarray(self.focal_detprob, dtype=np.float32)
        if self.focal_nstems is not None:
            arrays["focal_nstems"] = np.ascontiguousarray(self.focal_nstems, dtype=np.float32)
        if self.denoised is not None:
            arrays["denoised"] = encode_audio_flac(self.denoised, audio_sample_rate)
        arrays["n_stems"] = np.asarray(len(self.stems), dtype=np.int64)
        for i, stem in enumerate(self.stems):
            arrays[f"stem{i}_audio"] = encode_audio_flac(stem.audio, audio_sample_rate)
            for key, value in encode_preds(stem.preds, preds_threshold).items():
                arrays[f"stem{i}_preds_{key}"] = value
        return arrays

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray]) -> "ItemResult":
        """Reconstruct an `ItemResult` from the flat arrays of `to_arrays`.

        The detail rung is inferred from the keys present: `denoised` iff a
        ``denoised`` key exists, `stems` from the ``n_stems`` count. Latitude /
        longitude default to ``nan`` when absent; the focal-quality tracks are
        ``None`` when absent. This is the entry point the downstream re-gate uses
        to read shards.

        Parameters
        ----------
        arrays : Mapping[str, np.ndarray]
            One recording's flat array dict (the engine strips the per-recording
            shard prefix before calling this).

        Returns
        -------
        ItemResult
            The reconstructed result. Prediction tracks carry only the classes
            that survived `encode_preds`; audio is decoded from FLAC bytes back
            to the int16-grid float waveform.
        """
        preds = decode_preds(_group(arrays, "preds_"))
        latitude = float(arrays["latitude"]) if "latitude" in arrays else float("nan")
        longitude = float(arrays["longitude"]) if "longitude" in arrays else float("nan")
        focal_detprob = np.asarray(arrays["focal_detprob"], dtype=np.float32) if "focal_detprob" in arrays else None
        focal_nstems = np.asarray(arrays["focal_nstems"], dtype=np.float32) if "focal_nstems" in arrays else None
        denoised = decode_audio_flac(arrays["denoised"]) if "denoised" in arrays else None
        n_stems = int(arrays["n_stems"]) if "n_stems" in arrays else 0
        stems = tuple(
            Stem(
                audio=decode_audio_flac(arrays[f"stem{i}_audio"]),
                preds=decode_preds(_group(arrays, f"stem{i}_preds_")),
            )
            for i in range(n_stems)
        )
        return cls(
            preds=preds,
            denoised=denoised,
            stems=stems,
            latitude=latitude,
            longitude=longitude,
            focal_detprob=focal_detprob,
            focal_nstems=focal_nstems,
        )


def _group(arrays: Mapping[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    """Return the sub-dict of `arrays` whose keys start with `prefix`, unprefixed.

    Parameters
    ----------
    arrays : Mapping[str, np.ndarray]
        Flat array dict.
    prefix : str
        Key prefix identifying one `encode_preds` group (e.g. ``"preds_"`` or
        ``"stem0_preds_"``).

    Returns
    -------
    dict[str, np.ndarray]
        Keys with `prefix` stripped, restoring the inner group key names
        (``predictions`` / ``classes`` / ``frame_rate``).
    """
    return {key[len(prefix) :]: value for key, value in arrays.items() if key.startswith(prefix)}
