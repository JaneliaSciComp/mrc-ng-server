import numpy as np
import pytest

from mrcng.mrcheader import parse_header
from mrcng.precomputed import plan_scales
from mrcng import omezarr


def _hdr(path, is_image_stack=False):
    import os
    fd = os.open(str(path), os.O_RDONLY)
    try:
        st = os.stat(fd)
        return parse_header(fd, st.st_size, st.st_mtime_ns, is_image_stack=is_image_stack)
    finally:
        os.close(fd)


def test_chunk_rel_path_is_level_c_z_y_x():
    assert omezarr.chunk_rel_path(1, 0, 2, 3) == "1/c/0/2/3"


def test_chunk_region_clips_to_level_and_rejects_outside():
    # size (x=20, y=10, z=5), chunk (8, 8, 8): x index 2 covers [16, 20)
    assert omezarr.chunk_region((20, 10, 5), (8, 8, 8), 0, 1, 2) == (16, 20, 8, 10, 0, 5)
    with pytest.raises(ValueError):
        omezarr.chunk_region((20, 10, 5), (8, 8, 8), 1, 0, 0)  # z origin 8 >= 5
    with pytest.raises(ValueError):
        omezarr.chunk_region((20, 10, 5), (8, 8, 8), 0, 0, 999999)


def test_pad_chunk_returns_same_object_when_full_and_zero_pads_otherwise():
    full = np.ones((8, 8, 8), dtype="<i2")
    assert omezarr.pad_chunk(full, (8, 8, 8)) is full

    edge = np.ones((5, 8, 3), dtype="<i2")
    out = omezarr.pad_chunk(edge, (8, 8, 8))
    assert out.shape == (8, 8, 8) and out.dtype == edge.dtype and out.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(out[:5, :, :3], 1)
    assert out[5:].sum() == 0 and out[:, :, 3:].sum() == 0


@pytest.mark.parametrize("np_dtype,zarr_name", [
    ("int8", "int8"), ("uint8", "uint8"), ("int16", "int16"), ("uint16", "uint16"), ("float32", "float32"),
])
def test_array_json_reverses_to_zyx_and_names_dtype(np_dtype, zarr_name):
    doc = omezarr.build_array_json((20, 10, 5), (8, 8, 4), np.dtype(np_dtype))
    assert doc["zarr_format"] == 3 and doc["node_type"] == "array"
    assert doc["shape"] == [5, 10, 20]
    assert doc["chunk_grid"] == {"name": "regular", "configuration": {"chunk_shape": [4, 8, 8]}}
    assert doc["chunk_key_encoding"] == {"name": "default", "configuration": {"separator": "/"}}
    assert doc["data_type"] == zarr_name
    assert doc["fill_value"] == 0
    assert doc["codecs"] == [{"name": "bytes", "configuration": {"endian": "little"}}]
    assert doc["dimension_names"] == ["z", "y", "x"]


def test_group_json_scale_and_translation_per_level(make_mrc_file):
    # 20 Å voxels -> 2.0 nm; level 1 factor 2 -> scale 4.0, translation 0.5 * 2.0 = 1.0
    path = make_mrc_file(shape=(64, 64, 64), mode=1, voxel_size_angstrom=(20.0, 20.0, 20.0))
    hdr = _hdr(path)
    scales = plan_scales((64, 64, 64), min_axis_size=16, max_levels=3)
    assert [s.key for s in scales] == ["1_1_1", "2_2_2", "4_4_4"]

    doc = omezarr.build_group_json(hdr, scales, name="tomo.mrc")
    assert doc["zarr_format"] == 3 and doc["node_type"] == "group"
    ome = doc["attributes"]["ome"]
    assert ome["version"] == "0.5"
    ms = ome["multiscales"][0]
    assert ms["name"] == "tomo.mrc"
    assert ms["axes"] == [
        {"name": "z", "type": "space", "unit": "nanometer"},
        {"name": "y", "type": "space", "unit": "nanometer"},
        {"name": "x", "type": "space", "unit": "nanometer"},
    ]
    ds = ms["datasets"]
    assert [d["path"] for d in ds] == ["0", "1", "2"]
    assert ds[0]["coordinateTransformations"] == [{"type": "scale", "scale": [2.0, 2.0, 2.0]}]
    assert ds[1]["coordinateTransformations"] == [
        {"type": "scale", "scale": [4.0, 4.0, 4.0]},
        {"type": "translation", "translation": [1.0, 1.0, 1.0]},
    ]
    assert ds[2]["coordinateTransformations"] == [
        {"type": "scale", "scale": [8.0, 8.0, 8.0]},
        {"type": "translation", "translation": [3.0, 3.0, 3.0]},
    ]
    assert doc["attributes"]["mrcng"] == {"voxel_size_is_default": False, "is_image_stack": False}


def test_group_json_image_stack_keeps_z_unbinned_and_untranslated(make_mrc_file):
    path = make_mrc_file(shape=(64, 64, 8), mode=1, voxel_size_angstrom=(20.0, 20.0, 20.0))
    hdr = _hdr(path, is_image_stack=True)
    scales = plan_scales((64, 64, 8), min_axis_size=16, max_levels=2, downsample_z=False)
    assert scales[1].key == "2_2_1"

    doc = omezarr.build_group_json(hdr, scales, name="stack.mrc")
    ds = doc["attributes"]["ome"]["multiscales"][0]["datasets"]
    # z is 1 nm per slice at level 0, unchanged at level 1, and never shifted
    assert ds[0]["coordinateTransformations"][0]["scale"] == [1.0, 2.0, 2.0]
    assert ds[1]["coordinateTransformations"][0]["scale"] == [1.0, 4.0, 4.0]
    assert ds[1]["coordinateTransformations"][1]["translation"] == [0.0, 1.0, 1.0]
    assert doc["attributes"]["mrcng"]["is_image_stack"] is True
