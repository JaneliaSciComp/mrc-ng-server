import os
import struct

import numpy as np
import mrcfile
import pytest

from mrcng.mrcheader import (
    parse_header, UnsupportedModeError, UnsupportedByteOrderError,
    NonStandardAxisOrderError, NonStandardGridSizeError, TruncatedFileError,
)


def _parse(path, **kwargs):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        st = os.stat(fd)
        return parse_header(fd, st.st_size, st.st_mtime_ns, **kwargs)
    finally:
        os.close(fd)


def test_basic_int16_header_matches_mrcfile(make_mrc_file):
    path = make_mrc_file(shape=(64, 32, 16), mode=1, voxel_size_angstrom=(2.0, 2.0, 4.0))
    hdr = _parse(path)
    with mrcfile.open(path, permissive=True) as mf:
        assert hdr.nx == mf.header.nx == 64
        assert hdr.ny == mf.header.ny == 32
        assert hdr.nz == mf.header.nz == 16
        assert hdr.dtype == mf.data.dtype
        np.testing.assert_allclose(hdr.voxel_size_angstrom, (2.0, 2.0, 4.0))


def test_odd_dimensions_and_extended_header(make_mrc_file):
    path = make_mrc_file(name="odd.mrc", shape=(101, 97, 53), mode=2, nsymbt=128)
    hdr = _parse(path)
    with mrcfile.open(path, permissive=True) as mf:
        assert hdr.data_offset == 1024 + 128 == mf.header.nsymbt + 1024
        assert (hdr.nx, hdr.ny, hdr.nz) == (mf.header.nx, mf.header.ny, mf.header.nz)


def test_anisotropic_volume(make_mrc_file):
    path = make_mrc_file(name="aniso.mrc", shape=(2048, 2048, 64), mode=1)
    hdr = _parse(path)
    assert (hdr.nx, hdr.ny, hdr.nz) == (2048, 2048, 64)


def test_unsupported_mode_raises(make_mrc_file):
    path = make_mrc_file(shape=(4, 4, 4), mode=1)
    with open(path, "r+b") as f:
        f.seek(12)
        f.write(struct.pack("<i", 3))
    with pytest.raises(UnsupportedModeError):
        _parse(path)


def test_non_standard_axis_order_raises(make_mrc_file):
    path = make_mrc_file(shape=(4, 4, 4), mode=1, mapc=2, mapr=1, maps=3)
    with pytest.raises(NonStandardAxisOrderError):
        _parse(path)


def test_truncated_file_raises(make_mrc_file):
    path = make_mrc_file(shape=(16, 16, 16), mode=1, truncate_bytes=100)
    with pytest.raises(TruncatedFileError):
        _parse(path)


def test_nxstart_nystart_nzstart_are_recorded(make_mrc_file):
    path = make_mrc_file(shape=(8, 8, 8), mode=1, nstart=(3, -5, 100))
    hdr = _parse(path)
    with mrcfile.open(path, permissive=True) as mf:
        assert (hdr.nxstart, hdr.nystart, hdr.nzstart) == (
            mf.header.nxstart, mf.header.nystart, mf.header.nzstart,
        ) == (3, -5, 100)


def test_mode12_float16_is_read_as_float16_and_served_as_float32(make_mrc_file):
    # Regression: mode 12 was rejected outright, so half-precision tomograms
    # (AreTomo/Warp write them to halve file size) 422'd on /info. Neuroglancer
    # allows only uint8/int8/uint16/int16/uint32/int32/uint64/float32 as
    # data_type -- and has no float16 in its DataType enum at all -- so the
    # served dtype must widen while the on-disk dtype stays float16, which is
    # what every byte offset is computed from.
    path = make_mrc_file(shape=(8, 8, 8), mode=12)
    hdr = _parse(path)
    assert hdr.dtype == np.dtype("<f2")
    assert hdr.dtype.itemsize == 2
    assert hdr.served_dtype == np.dtype("<f4")


def test_non_float16_modes_serve_their_on_disk_dtype(make_mrc_file):
    for mode in (0, 1, 2, 6):
        hdr = _parse(make_mrc_file(name=f"m{mode}.mrc", shape=(8, 8, 8), mode=mode))
        assert hdr.served_dtype == hdr.dtype, mode


def test_mismatched_grid_size_raises(make_mrc_file):
    # Common in image-stack MRC files: mz=1 regardless of nz. Silently using it
    # would divide the whole cell depth into a single voxel, making the z
    # scale bar nz times too large -- fail closed instead (sec 0).
    path = make_mrc_file(shape=(64, 64, 50), mode=1, grid_size=(64, 64, 1))
    with pytest.raises(NonStandardGridSizeError):
        _parse(path)


def test_grid_size_matching_sample_count_is_not_flagged_as_default(make_mrc_file):
    path = make_mrc_file(shape=(8, 8, 8), mode=1, grid_size=(8, 8, 8),
                         voxel_size_angstrom=(2.0, 2.0, 2.0))
    hdr = _parse(path)
    assert hdr.voxel_size_is_default is False
    np.testing.assert_allclose(hdr.voxel_size_angstrom, (2.0, 2.0, 2.0))


def test_single_axis_zero_cella_defaults_only_that_axis(make_mrc_file):
    # Regression: cella[2]==0 while cella[0]/cella[1] are populated (common
    # for per-section tilt-series stacks, where z isn't a real physical
    # sampling) fell through to voxel_size = cella / (mx,my,mz), producing a
    # literal 0.0 Angstrom z voxel size. Neuroglancer's precomputed format
    # rejects that outright: "Expected positive finite float but received: 0".
    path = make_mrc_file(shape=(8, 8, 8), mode=1, voxel_size_angstrom=(2.0, 2.0, 0.0))
    hdr = _parse(path)
    assert hdr.voxel_size_angstrom == (2.0, 2.0, 1.0)
    assert hdr.voxel_size_is_default is True


def test_zero_cella_falls_back_to_default_voxel_size(make_mrc_file):
    path = make_mrc_file(shape=(4, 4, 4), mode=1, voxel_size_angstrom=(0.0, 0.0, 0.0))
    hdr = _parse(path)
    assert hdr.voxel_size_angstrom == (1.0, 1.0, 1.0)
    assert hdr.voxel_size_is_default is True


def test_mode0_default_signed_agrees_with_mrcfile(make_mrc_file):
    path = make_mrc_file(shape=(4, 4, 4), mode=0)
    hdr = _parse(path)
    with mrcfile.open(path, permissive=True) as mf:
        assert hdr.dtype == mf.data.dtype == np.dtype(np.int8)


def test_mode0_imod_unsigned_diverges_from_mrcfile_default(make_mrc_file):
    # mrcfile always returns int8 for mode 0; we deliberately diverge when
    # an IMOD unsigned stamp is present (spec section 2.3).
    path = make_mrc_file(shape=(4, 4, 4), mode=0, imod_flags="unsigned")
    hdr = _parse(path)
    assert hdr.dtype == np.dtype(np.uint8)


def test_mode0_imod_signed_matches_default(make_mrc_file):
    path = make_mrc_file(shape=(4, 4, 4), mode=0, imod_flags="signed")
    hdr = _parse(path)
    assert hdr.dtype == np.dtype(np.int8)


def test_mode0_assume_mode0_cli_override(make_mrc_file):
    path = make_mrc_file(shape=(4, 4, 4), mode=0)
    hdr = _parse(path, assume_mode0="uint8")
    assert hdr.dtype == np.dtype(np.uint8)
    assert hdr.mode0_signedness_is_ambiguous is False


def test_mode0_no_stamp_no_override_is_flagged_ambiguous(make_mrc_file):
    path = make_mrc_file(shape=(4, 4, 4), mode=0)
    hdr = _parse(path)
    assert hdr.mode0_signedness_is_ambiguous is True


def test_mode0_imod_stamp_present_is_not_ambiguous(make_mrc_file):
    path = make_mrc_file(shape=(4, 4, 4), mode=0, imod_flags="signed")
    hdr = _parse(path)
    assert hdr.mode0_signedness_is_ambiguous is False


def test_non_mode0_is_never_ambiguous(make_mrc_file):
    path = make_mrc_file(shape=(4, 4, 4), mode=1)
    hdr = _parse(path)
    assert hdr.mode0_signedness_is_ambiguous is False
