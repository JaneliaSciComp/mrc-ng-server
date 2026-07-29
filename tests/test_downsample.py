import numpy as np

from mrcng.downsample import block_mean


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
