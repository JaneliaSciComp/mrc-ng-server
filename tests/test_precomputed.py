import numpy as np
import pytest

from mrcng.precomputed import (
    plan_scales, build_info, chunk_name, parse_chunk_name,
    clip_chunk_to_scale, encode_chunk, ScaleLevel,
)


def test_plan_scales_isotropic_stops_at_min_axis_size():
    scales = plan_scales((256, 256, 256), min_axis_size=32, max_levels=10)
    assert scales[0].key == "1_1_1"
    assert scales[0].size == (256, 256, 256)
    # 256 -> 128 -> 64 -> 32 (stop, since 32 is not > min_axis_size)
    assert scales[-1].size == (32, 32, 32)


def test_plan_scales_anisotropic_pins_short_axis():
    scales = plan_scales((4096, 4096, 40), min_axis_size=32, max_levels=6)
    z_sizes = [lvl.size[2] for lvl in scales]
    assert z_sizes[0] == 40
    assert z_sizes[1] == 20  # one bin: 40 > 32
    assert all(z == 20 for z in z_sizes[1:])  # stops changing once <= min_axis_size


def test_plan_scales_uses_ceil_for_odd_sizes():
    scales = plan_scales((101, 101, 101), min_axis_size=32, max_levels=10)
    # level 1: factor (2,2,2), size = ceil(101/2) = 51
    assert scales[1].size == (51, 51, 51)
    assert scales[1].key == "2_2_2"


def test_plan_scales_respects_max_levels():
    scales = plan_scales((4096, 4096, 4096), min_axis_size=32, max_levels=3)
    assert len(scales) == 3


def test_build_info_converts_angstrom_to_nanometres():
    class FakeHdr:
        nx, ny, nz = 100, 100, 100
        dtype = np.dtype(np.int16)
        voxel_size_angstrom = (6.8, 6.8, 6.8)
        voxel_size_is_default = False

    scales = plan_scales((100, 100, 100), min_axis_size=32, max_levels=2)
    info = build_info(FakeHdr(), scales, chunk_size=(64, 64, 64))
    assert info["@type"] == "neuroglancer_multiscale_volume"
    assert info["data_type"] == "int16"
    assert info["voxel_size_is_default"] is False
    assert info["scales"][0]["resolution"] == pytest.approx([0.68, 0.68, 0.68])
    level1_factor = scales[1].factors
    expected_res = [0.68 * f for f in level1_factor]
    assert info["scales"][1]["resolution"] == pytest.approx(expected_res)


def test_build_info_surfaces_voxel_size_is_default():
    # Regression: a zero-cella file silently advertised a made-up 0.1nm voxel
    # size in info with nothing marking it as a fallback.
    class FakeHdr:
        nx, ny, nz = 8, 8, 8
        dtype = np.dtype(np.int16)
        voxel_size_angstrom = (1.0, 1.0, 1.0)
        voxel_size_is_default = True

    scales = plan_scales((8, 8, 8), min_axis_size=32, max_levels=1)
    info = build_info(FakeHdr(), scales, chunk_size=(64, 64, 64))
    assert info["voxel_size_is_default"] is True


def test_chunk_name_roundtrip():
    name = chunk_name(0, 64, 64, 128, 0, 32)
    assert name == "0-64_64-128_0-32"
    assert parse_chunk_name(name) == (0, 64, 64, 128, 0, 32)


def test_parse_chunk_name_rejects_malformed():
    with pytest.raises(ValueError):
        parse_chunk_name("not-a-chunk-name")


def test_clip_chunk_to_scale_clips_edge_chunk():
    scale = ScaleLevel(key="1_1_1", size=(100, 100, 40), factors=(1, 1, 1))
    clipped = clip_chunk_to_scale(scale, 0, 64, 0, 64, 0, 64)
    assert clipped == (0, 64, 0, 64, 0, 40)


def test_clip_chunk_to_scale_rejects_misaligned_request():
    scale = ScaleLevel(key="1_1_1", size=(100, 100, 40), factors=(1, 1, 1))
    with pytest.raises(ValueError):
        clip_chunk_to_scale(scale, 5, 69, 0, 64, 0, 64, chunk_size=(64, 64, 64))


def test_encode_chunk_byte_order():
    nx, ny, nz = 4, 3, 2
    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    # keep values within int16 range (x fastest, y next, z slowest, small factors)
    arr = (xx + 10 * yy + 100 * zz).astype("<i2")
    raw = encode_chunk(arr)
    assert raw == arr.tobytes()
    # spot-check voxel x=2,y=1,z=1 -> value 100 + 10 + 2, at flat index (z*ny*nx + y*nx + x)
    flat_index = (1 * ny * nx + 1 * nx + 2)
    value = int.from_bytes(raw[flat_index * 2: flat_index * 2 + 2], "little", signed=True)
    assert value == arr[1, 1, 2] == 100 + 10 + 2


def test_encode_chunk_accepts_single_byte_dtypes():
    # Regression: int8/uint8 (MRC mode 0) always report '|' for byteorder --
    # there's no endianness to get wrong when writing one byte -- but the
    # check rejected them anyway, so no mode-0 file could ever be served.
    arr = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
    assert encode_chunk(arr) == arr.tobytes()
    arr8 = np.arange(-4, 4, dtype=np.int8).reshape(2, 2, 2)
    assert encode_chunk(arr8) == arr8.tobytes()


def test_encode_chunk_rejects_non_contiguous():
    arr = np.zeros((4, 4, 4), dtype="<i2").T  # transposed -> not C-contiguous
    with pytest.raises(ValueError):
        encode_chunk(arr)
