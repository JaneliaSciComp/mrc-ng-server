import os

import numpy as np
import pytest

from mrcng.mrcheader import parse_header
from mrcng.reader import (
    pread_exact, read_chunk, choose_strategy, ReadStrategy,
    UnexpectedEOF, ChunkOutOfBounds,
)


def _open_and_parse(path):
    fd = os.open(str(path), os.O_RDONLY)
    st = os.stat(fd)
    hdr = parse_header(fd, st.st_size, st.st_mtime_ns)
    return fd, hdr


def test_pread_exact_reads_full_count(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"abcdefghij")
    fd = os.open(str(f), os.O_RDONLY)
    try:
        assert pread_exact(fd, 10, 0) == b"abcdefghij"
        assert pread_exact(fd, 4, 3) == b"defg"
    finally:
        os.close(fd)


def test_pread_exact_raises_on_short_read(tmp_path):
    f = tmp_path / "short.bin"
    f.write_bytes(b"abc")
    fd = os.open(str(f), os.O_RDONLY)
    try:
        with pytest.raises(UnexpectedEOF):
            pread_exact(fd, 10, 0)
    finally:
        os.close(fd)


def test_choose_strategy_threshold():
    assert choose_strategy(0, 1, itemsize=2, threshold=4096) == ReadStrategy.SPAN_WISE
    assert choose_strategy(0, 2048, itemsize=2, threshold=4096) == ReadStrategy.ROW_WISE


def test_read_chunk_matches_mrcfile_for_random_region(make_mrc_file):
    def fill(zz, yy, xx):
        return (xx + 1000 * yy + 1_000_000 * zz) % 30000

    path = make_mrc_file(shape=(40, 30, 20), mode=1, fill=fill)
    fd, hdr = _open_and_parse(path)
    try:
        import mrcfile
        with mrcfile.open(path, permissive=True) as mf:
            reference = mf.data  # shape (nz, ny, nx)

        arr = read_chunk(fd, hdr, x0=5, x1=25, y0=3, y1=20, z0=1, z1=10)
        np.testing.assert_array_equal(arr, reference[1:10, 3:20, 5:25])
    finally:
        os.close(fd)


def test_read_chunk_row_wise_and_span_wise_agree(make_mrc_file):
    def fill(zz, yy, xx):
        return (xx + 1000 * yy + 1_000_000 * zz) % 30000

    path = make_mrc_file(shape=(4096, 8, 8), mode=1, fill=fill)
    fd, hdr = _open_and_parse(path)
    try:
        row_wise = read_chunk(fd, hdr, 0, 4096, 0, 4, 0, 4, row_bytes_threshold=0)
        span_wise = read_chunk(fd, hdr, 0, 4096, 0, 4, 0, 4, row_bytes_threshold=10**9)
        np.testing.assert_array_equal(row_wise, span_wise)
    finally:
        os.close(fd)


def test_read_chunk_clips_out_of_bounds_raises(make_mrc_file):
    path = make_mrc_file(shape=(4, 4, 4), mode=1)
    fd, hdr = _open_and_parse(path)
    try:
        with pytest.raises(ChunkOutOfBounds):
            read_chunk(fd, hdr, x0=10, x1=20, y0=0, y1=4, z0=0, z1=4)
    finally:
        os.close(fd)


def test_read_chunk_clips_partial_edge(make_mrc_file):
    path = make_mrc_file(shape=(4, 4, 4), mode=1, fill=lambda zz, yy, xx: xx)
    fd, hdr = _open_and_parse(path)
    try:
        arr = read_chunk(fd, hdr, x0=2, x1=10, y0=0, y1=4, z0=0, z1=4)
        assert arr.shape == (4, 4, 2)  # clipped to nx=4
    finally:
        os.close(fd)
