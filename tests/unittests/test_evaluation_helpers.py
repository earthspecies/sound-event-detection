import numpy as np
import pytest

from sound_event_detection.evaluation.counting import get_frame_tpfpfn_counts
from sound_event_detection.evaluation.matching import fast_intersect


def test_get_frame_tpfpfn_counts_basic_binary_counts_to_seconds() -> None:
    """
    Simple sanity check with binary gt and preds; verifies TP/FP/FN and frame_rate scaling.
    """
    frame_rate = 2.0  # 2 frames/sec => each frame contributes 0.5 seconds
    gt = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    preds = np.array(
        [
            [True, False],
            [True, True],
            [True, False],
            [False, True],
        ],
        dtype=bool,
    )

    out = get_frame_tpfpfn_counts(gt, preds, frame_rate=frame_rate)

    # Class 0:
    # tp frames = 2 (t0,t1), fp frames = 1 (t2), fn frames = 0
    # => seconds: [1.0, 0.5, 0.0]
    # Class 1:
    # tp frames = 1 (t1), fp frames = 1 (t3), fn frames = 1 (t2)
    # => seconds: [0.5, 0.5, 0.5]
    expected = np.array(
        [
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    ) / frame_rate

    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, expected, rtol=0.0, atol=1e-12)


def test_get_frame_tpfpfn_counts_fractional_gt() -> None:
    """
    gt_array can be fractional; ensure tp uses preds * gt, and fn/fp are consistent.
    """
    frame_rate = 10.0
    gt = np.array(
        [
            [0.5, 0.0],
            [1.0, 0.25],
            [0.0, 0.75],
        ],
        dtype=np.float32,
    )
    preds = np.array(
        [
            [True, False],
            [True, True],
            [False, True],
        ],
        dtype=bool,
    )

    out = get_frame_tpfpfn_counts(gt, preds, frame_rate=frame_rate)

    # Work in frames first.
    # Class 0: preds=[1,1,0], gt=[0.5,1,0]
    # tp=0.5+1=1.5, sum(pred)=2 => fp=0.5, sum(gt)=1.5 => fn=0.0
    # Class 1: preds=[0,1,1], gt=[0,0.25,0.75]
    # tp=0.25+0.75=1.0, sum(pred)=2 => fp=1.0, sum(gt)=1.0 => fn=0.0
    expected_frames = np.array(
        [
            [1.5, 0.5, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    expected = expected_frames / frame_rate

    np.testing.assert_allclose(out, expected, rtol=0.0, atol=1e-12)


def test_get_frame_tpfpfn_counts_raises_on_shape_mismatch() -> None:
    gt = np.zeros((3, 2), dtype=np.float32)
    preds = np.zeros((4, 2), dtype=bool)

    with pytest.raises(ValueError, match=r"must have same shape"):
        _ = get_frame_tpfpfn_counts(gt, preds, frame_rate=1.0)


def test_fast_intersect_property() -> None:
    rng = np.random.default_rng(0)

    # Generate random reference and estimated events
    n_ref = 200
    n_est = 250

    ref_on = rng.uniform(0.0, 100.0, size=n_ref)
    ref_off = ref_on + rng.uniform(0.0, 5.0, size=n_ref)
    ref = np.stack([ref_on, ref_off], axis=0)

    est_on = rng.uniform(0.0, 100.0, size=n_est)
    est_off = est_on + rng.uniform(0.0, 5.0, size=n_est)
    est = np.stack([est_on, est_off], axis=0)

    matches = fast_intersect(ref, est)

    # Assert the property from the docstring
    for i in range(n_ref):
        expected = {
            j
            for j in range(n_est)
            if (ref[0, i] <= est[1, j]) and (ref[1, i] >= est[0, j])
        }
        assert matches[i] == expected
