import os
import time

import numpy as np

from mrcng.reader import read_chunk
from mrcng.server.fdcache import FdCache


def test_get_parses_header_once_and_reuses_fd(make_mrc_file):
    path = make_mrc_file(shape=(8, 8, 8), mode=1)
    cache = FdCache(max_size=8)
    try:
        with cache.open(path) as (fd1, hdr1):
            pass
        with cache.open(path) as (fd2, hdr2):
            pass
        assert hdr1 == hdr2
        assert fd1 == fd2
    finally:
        cache.close_all()


def test_replaced_file_misses_cache(make_mrc_file, tmp_path):
    path = make_mrc_file(shape=(8, 8, 8), mode=1)
    cache = FdCache(max_size=8)
    try:
        with cache.open(path) as (_, hdr1):
            pass

        time.sleep(0.01)
        os.remove(path)
        from tests.conftest import make_mrc
        make_mrc(path, shape=(16, 16, 16), mode=1)  # same path, different size/mtime

        with cache.open(path) as (_, hdr2):
            pass
        assert hdr2.nx == 16
        assert hdr1.nx == 8
    finally:
        cache.close_all()


def test_eviction_closes_oldest_fd(tmp_path):
    from tests.conftest import make_mrc
    cache = FdCache(max_size=2)
    paths = []
    for i in range(3):
        p = tmp_path / f"f{i}.mrc"
        make_mrc(p, shape=(4, 4, 4), mode=1)
        paths.append(p)
        with cache.open(p):
            pass

    key0 = cache._key_for(paths[0])
    assert key0 not in cache._entries
    cache.close_all()


def test_held_handle_survives_eviction_and_fd_reuse(tmp_path):
    """Regression: eviction used to os.close() an fd a request was still using.
    The number is recycled by the next os.open(), so the in-flight pread came
    back with another file's voxels and a 200 OK."""
    from tests.conftest import make_mrc
    a = tmp_path / "a.mrc"
    b = tmp_path / "b.mrc"
    make_mrc(a, shape=(8, 8, 8), mode=1, fill=lambda z, y, x: np.full_like(x, 111))
    make_mrc(b, shape=(8, 8, 8), mode=1, fill=lambda z, y, x: np.full_like(x, 222))

    cache = FdCache(max_size=1)
    try:
        with cache.open(a) as (fd_a, hdr_a):
            with cache.open(b):  # evicts a while the handle above is held
                pass
            # a new fd here would land on a's number if it had been closed
            decoy = os.open(str(b), os.O_RDONLY)
            try:
                assert decoy != fd_a
                arr = read_chunk(fd_a, hdr_a, 0, 8, 0, 8, 0, 8)
                assert np.unique(arr).tolist() == [111]
            finally:
                os.close(decoy)
    finally:
        cache.close_all()


def test_eviction_rate_warning_is_logged(tmp_path, caplog):
    # A high eviction rate means the working set of files exceeds
    # fd_cache_size, so every request is paying an open() -- the operator
    # needs to see this, not just an empty cache silently thrashing.
    import logging
    from tests.conftest import make_mrc

    cache = FdCache(max_size=2)
    with caplog.at_level(logging.WARNING, logger="mrcng.server"):
        for i in range(6):  # 4 evictions with max_size=2 -> one warning at count 2, one at 4
            p = tmp_path / f"f{i}.mrc"
            make_mrc(p, shape=(4, 4, 4), mode=1)
            with cache.open(p):
                pass
    cache.close_all()

    warnings = [r for r in caplog.records if r.name == "mrcng.server" and "evicted" in r.message]
    assert len(warnings) == 2


def test_evicted_fd_is_closed_once_the_last_holder_releases(tmp_path):
    from tests.conftest import make_mrc
    a = tmp_path / "a.mrc"
    b = tmp_path / "b.mrc"
    make_mrc(a, shape=(8, 8, 8), mode=1)
    make_mrc(b, shape=(8, 8, 8), mode=1)

    cache = FdCache(max_size=1)
    with cache.open(a) as (fd_a, _):
        with cache.open(b):
            pass
        os.fstat(fd_a)  # still open while held

    try:
        os.fstat(fd_a)
        raise AssertionError("fd should have been closed on release after eviction")
    except OSError:
        pass
    cache.close_all()
