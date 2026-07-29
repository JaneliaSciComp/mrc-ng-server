import math

import numpy as np

from mrcng.downsample import block_mean


def _reference_block_mean(arr, factors):
    """Deliberately naive independent reference: one explicit Python loop per
    output voxel, float64 mean over whatever the block actually contains,
    rounded half away from zero. Shares no code with block_mean."""
    fz, fy, fx = factors
    out_shape = tuple(math.ceil(s / f) for s, f in zip(arr.shape, factors))
    out = np.empty(out_shape, dtype=arr.dtype)
    for iz in range(out_shape[0]):
        for iy in range(out_shape[1]):
            for ix in range(out_shape[2]):
                block = arr[iz * fz:(iz + 1) * fz,
                            iy * fy:(iy + 1) * fy,
                            ix * fx:(ix + 1) * fx]
                m = float(block.astype(np.float64).mean())
                out[iz, iy, ix] = math.floor(m + 0.5) if m >= 0 else math.ceil(m - 0.5)
    return out


def test_multi_axis_matches_independent_reference():
    # regression: reducing one axis at a time truncated the fractional mean back
    # to the int accumulator on every axis after the first, so ~32% of voxels
    # came out 1 low.
    rng = np.random.default_rng(0)
    arr = rng.integers(-3000, 3000, size=(8, 8, 8)).astype(np.int16)
    np.testing.assert_array_equal(block_mean(arr, (2, 2, 2)),
                                  _reference_block_mean(arr, (2, 2, 2)))


def test_multi_axis_matches_reference_on_non_divisible_shape():
    # short trailing block on all three axes at once
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 2000, size=(5, 7, 3)).astype(np.int16)
    np.testing.assert_array_equal(block_mean(arr, (2, 2, 2)),
                                  _reference_block_mean(arr, (2, 2, 2)))


def test_cascade_does_not_drift_downward_for_positive_data():
    # The pyramid builds level L from level L-1, so any per-level bias compounds.
    # The truncation bug drifted -0.25, -0.50, -0.76 DN across three levels.
    #
    # Some upward drift is the floor for an integer pyramid, not a defect: each
    # level must round to int16 to be storable, and round-half-away-from-zero
    # sends ties up. A sum of 8 ints is a tie 1/8 of the time, each worth +0.5,
    # so the floor is +1/16 = +0.0625 DN per level (measured: +0.063). Hence the
    # asymmetric bounds -- the sign is what distinguishes the two.
    rng = np.random.default_rng(2)
    lvl = rng.integers(1000, 3000, size=(32, 32, 32)).astype(np.int16)
    exact = lvl.astype(np.float64)
    for _ in range(3):
        lvl = block_mean(lvl, (2, 2, 2))
        n = exact.shape[0] // 2
        exact = exact.reshape(n, 2, n, 2, n, 2).mean(axis=(1, 3, 5))
        drift = lvl.mean() - exact.mean()
        assert -0.05 < drift < 0.25, f"drift {drift:+.4f} DN"


def test_constant_volume_downsamples_to_same_constant():
    arr = np.full((8, 8, 8), 7, dtype=np.int16)
    result = block_mean(arr, (2, 2, 2))
    assert result.shape == (4, 4, 4)
    assert np.all(result == 7)
    assert result.dtype == arr.dtype


def test_ramp_volume_matches_analytical_expectation():
    # ramp along the last axis, factor 2: [0,1,2,3] -> pair means [0.5, 2.5]
    # -> round half away from zero -> [1, 3]
    arr = np.arange(4, dtype=np.int16).reshape(1, 1, 4)
    result = block_mean(arr, (1, 1, 2))
    assert result.shape == (1, 1, 2)
    assert result.tolist() == [[[1, 3]]]


def test_non_divisible_edge_averages_actual_voxel_count():
    # size 5 with factor 2 -> blocks of size 2,2,1 (last block has 1 voxel)
    arr = np.array([10, 20, 30, 40, 50], dtype=np.int16).reshape(1, 1, 5)
    result = block_mean(arr, (1, 1, 2))
    # blocks: mean(10,20)=15, mean(30,40)=35, mean(50)=50
    assert result.tolist() == [[[15, 35, 50]]]


def test_no_overflow_for_int16_accumulation():
    arr = np.full((2, 2, 2), 32000, dtype=np.int16)
    result = block_mean(arr, (2, 2, 2))
    assert result.item() == 32000  # would overflow if summed in int16 before dividing


def test_float32_input_downsamples_correctly():
    arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(1, 1, 4)
    result = block_mean(arr, (1, 1, 2))
    np.testing.assert_allclose(result, [[[1.5, 3.5]]])
    assert result.dtype == np.float32


def test_rounds_half_away_from_zero_for_negative_values():
    arr = np.array([-1, -2], dtype=np.int16).reshape(1, 1, 2)
    result = block_mean(arr, (1, 1, 2))
    # mean = -1.5, round half away from zero -> -2
    assert result.item() == -2


def test_factors_apply_per_axis_zyx():
    # factor only on z axis (first axis of the (z,y,x) array)
    arr = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
    result = block_mean(arr, (2, 1, 1))
    assert result.shape == (1, 3, 4)
    expected = np.round((arr[0].astype(np.int32) + arr[1].astype(np.int32)) / 2).astype(np.int16)
    np.testing.assert_array_equal(result[0], expected)
