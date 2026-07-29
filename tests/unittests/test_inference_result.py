"""Round-trip and codec tests for the LSI result DTO."""

import numpy as np
import pytest

from esp_research.protocols.detector import DetectorOutput
from sound_event_detection.inference.result import (
    ItemResult,
    Stem,
    decode_audio,
    decode_preds,
    encode_audio,
    encode_preds,
)


def _detector_output(frames: int, class_names: list[str], seed: int, floor: float = 0.5) -> DetectorOutput:
    """Build a single-recording DetectorOutput whose values are float16-exact and >= floor.

    Keeping values on the float16 grid and above the compression threshold makes
    the prediction codec lossless, so round-trips are bit-exact.
    """
    rng = np.random.default_rng(seed)
    values = floor + (1.0 - floor) * rng.random((frames, len(class_names)))
    values = values.astype(np.float16).astype(np.float32)  # snap to the float16 grid
    return DetectorOutput(predictions=values[np.newaxis], frame_rate=7.6, class_names=class_names)


def _int16_grid_audio(samples: int, seed: int) -> np.ndarray:
    """Audio lying exactly on the int16 PCM grid, so the audio codec is lossless here."""
    rng = np.random.default_rng(seed)
    return (rng.integers(-32768, 32768, size=samples) / 32767.0).astype(np.float32)


def _assert_output_equal(a: DetectorOutput, b: DetectorOutput) -> None:
    assert a.class_names == b.class_names
    assert a.frame_rate == b.frame_rate
    np.testing.assert_array_equal(a.predictions, b.predictions)


def _assert_item_equal(a: ItemResult, b: ItemResult) -> None:
    _assert_output_equal(a.preds, b.preds)
    if a.denoised is None:
        assert b.denoised is None
    else:
        np.testing.assert_array_equal(a.denoised, b.denoised)
    assert len(a.stems) == len(b.stems)
    for sa, sb in zip(a.stems, b.stems, strict=True):
        np.testing.assert_array_equal(sa.audio, sb.audio)
        _assert_output_equal(sa.preds, sb.preds)


def test_round_trip_preds() -> None:
    result = ItemResult(preds=_detector_output(5, ["a", "b", "c"], seed=1))
    _assert_item_equal(ItemResult.from_arrays(result.to_arrays()), result)


def test_round_trip_denoised() -> None:
    result = ItemResult(
        preds=_detector_output(5, ["a", "b", "c"], seed=2),
        denoised=_int16_grid_audio(200, seed=3),
    )
    _assert_item_equal(ItemResult.from_arrays(result.to_arrays()), result)


def test_round_trip_stems() -> None:
    labels = ["a", "b", "c"]
    stems = tuple(
        Stem(audio=_int16_grid_audio(200, seed=10 + i), preds=_detector_output(5, labels, seed=20 + i))
        for i in range(4)
    )
    result = ItemResult(
        preds=_detector_output(5, labels, seed=2),
        denoised=_int16_grid_audio(200, seed=3),
        stems=stems,
    )
    _assert_item_equal(ItemResult.from_arrays(result.to_arrays()), result)


def test_from_arrays_infers_detail_rung() -> None:
    labels = ["a", "b"]
    preds_only = ItemResult.from_arrays(ItemResult(preds=_detector_output(3, labels, seed=1)).to_arrays())
    assert preds_only.denoised is None and preds_only.stems == ()

    with_stems = ItemResult(
        preds=_detector_output(3, labels, seed=1),
        denoised=_int16_grid_audio(50, seed=2),
        stems=(Stem(audio=_int16_grid_audio(50, seed=3), preds=_detector_output(3, labels, seed=4)),),
    )
    decoded = ItemResult.from_arrays(with_stems.to_arrays())
    assert decoded.denoised is not None and len(decoded.stems) == 1


def test_encode_preds_drops_low_probability_classes() -> None:
    values = np.zeros((4, 3), dtype=np.float32)
    values[:, 0] = 0.9  # kept
    values[:, 1] = 0.01  # dropped (below default 0.05)
    values[:, 2] = 0.5  # kept
    output = DetectorOutput(predictions=values[np.newaxis], frame_rate=10.0, class_names=["hi", "lo", "mid"])

    group = encode_preds(output)
    assert group["classes"].tolist() == ["hi", "mid"]
    assert group["predictions"].shape == (4, 2)
    assert group["predictions"].dtype == np.float16

    decoded = decode_preds(group)
    assert decoded.class_names == ["hi", "mid"]


def test_encode_preds_all_dropped_is_empty() -> None:
    values = np.full((4, 2), 0.001, dtype=np.float32)
    output = DetectorOutput(predictions=values[np.newaxis], frame_rate=10.0, class_names=["x", "y"])
    decoded = decode_preds(encode_preds(output))
    assert decoded.class_names == []
    assert decoded.predictions.shape == (1, 4, 0)


def test_encode_preds_rejects_multi_batch() -> None:
    values = np.zeros((2, 4, 3), dtype=np.float32)
    output = DetectorOutput(predictions=values, frame_rate=10.0, class_names=["a", "b", "c"])
    with pytest.raises(ValueError, match="single-recording"):
        encode_preds(output)


def test_audio_codec_clips_and_is_near_lossless() -> None:
    audio = np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    decoded = decode_audio(encode_audio(audio))
    assert decoded.min() >= -1.0 - 1e-4 and decoded.max() <= 1.0 + 1e-4
    np.testing.assert_allclose(decoded, np.clip(audio, -1.0, 1.0), atol=1.0 / 32767.0)


def test_to_arrays_keys_per_detail_rung() -> None:
    labels = ["a", "b", "c"]
    base = {"preds_predictions", "preds_classes", "preds_frame_rate", "latitude", "longitude", "n_stems"}

    # preds rung: predictions + lat/long + n_stems, and nothing else.
    preds_keys = set(ItemResult(preds=_detector_output(5, labels, seed=1)).to_arrays())
    assert preds_keys == base

    # denoised rung: adds the waveform and (when supplied) the quality tracks.
    denoised_keys = set(
        ItemResult(
            preds=_detector_output(5, labels, seed=1),
            denoised=_int16_grid_audio(100, seed=2),
            focal_detprob=np.full(5, 0.7, dtype=np.float32),
            focal_nstems=np.full(5, 2.0, dtype=np.float32),
        ).to_arrays()
    )
    assert denoised_keys == base | {"denoised", "focal_detprob", "focal_nstems"}

    # stems rung: additionally stores each stem's audio + preds group.
    stems_keys = set(
        ItemResult(
            preds=_detector_output(5, labels, seed=1),
            denoised=_int16_grid_audio(100, seed=2),
            stems=(Stem(audio=_int16_grid_audio(100, seed=3), preds=_detector_output(5, labels, seed=4)),),
        ).to_arrays()
    )
    assert {"stem0_audio", "stem0_preds_predictions", "stem0_preds_classes", "stem0_preds_frame_rate"} <= stems_keys
    assert "denoised" in stems_keys


def test_preds_rung_has_no_audio_or_quality_keys() -> None:
    keys = set(ItemResult(preds=_detector_output(4, ["a"], seed=1)).to_arrays())
    assert not any(k.startswith("stem") for k in keys)
    assert "denoised" not in keys
    assert "focal_detprob" not in keys and "focal_nstems" not in keys


def test_lat_long_round_trip() -> None:
    result = ItemResult(preds=_detector_output(4, ["a", "b"], seed=1), latitude=42.5, longitude=-71.25)
    decoded = ItemResult.from_arrays(result.to_arrays())
    assert decoded.latitude == pytest.approx(42.5)
    assert decoded.longitude == pytest.approx(-71.25)


def test_lat_long_default_to_nan_when_absent() -> None:
    # Unset lat/long are stored as nan on every shard and round-trip to nan (not None).
    decoded = ItemResult.from_arrays(ItemResult(preds=_detector_output(4, ["a"], seed=1)).to_arrays())
    assert np.isnan(decoded.latitude) and np.isnan(decoded.longitude)


def test_quality_tracks_round_trip() -> None:
    detprob = np.array([0.6, np.nan, 0.9, 0.75], dtype=np.float32)  # nan where no stem gated in
    nstems = np.array([2.0, 0.0, 3.0, 1.0], dtype=np.float32)
    result = ItemResult(
        preds=_detector_output(4, ["a", "b"], seed=1),
        denoised=_int16_grid_audio(80, seed=2),
        focal_detprob=detprob,
        focal_nstems=nstems,
    )
    decoded = ItemResult.from_arrays(result.to_arrays())
    assert decoded.focal_detprob is not None and decoded.focal_nstems is not None
    assert decoded.focal_detprob.dtype == np.float32
    np.testing.assert_array_equal(decoded.focal_detprob, detprob)  # nan positions preserved
    np.testing.assert_array_equal(decoded.focal_nstems, nstems)


def test_quality_tracks_absent_on_preds_rung() -> None:
    decoded = ItemResult.from_arrays(ItemResult(preds=_detector_output(4, ["a"], seed=1)).to_arrays())
    assert decoded.focal_detprob is None and decoded.focal_nstems is None
