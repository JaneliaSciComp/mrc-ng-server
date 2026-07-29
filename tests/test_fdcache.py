import os
import time

from mrcng.server.fdcache import FdCache


def test_get_parses_header_once_and_reuses_fd(make_mrc_file):
    path = make_mrc_file(shape=(8, 8, 8), mode=1)
    cache = FdCache(max_size=8)
    try:
        hdr1 = cache.get(path)
        fd1 = cache.fd_for(path)
        hdr2 = cache.get(path)
        fd2 = cache.fd_for(path)
        assert hdr1 == hdr2
        assert fd1 == fd2
    finally:
        cache.close_all()


def test_replaced_file_misses_cache(make_mrc_file, tmp_path):
    path = make_mrc_file(shape=(8, 8, 8), mode=1)
    cache = FdCache(max_size=8)
    try:
        hdr1 = cache.get(path)

        time.sleep(0.01)
        os.remove(path)
        from tests.conftest import make_mrc
        make_mrc(path, shape=(16, 16, 16), mode=1)  # same path, different size/mtime

        hdr2 = cache.get(path)
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
        cache.get(p)

    key0 = cache._key_for(paths[0])
    assert key0 not in cache._entries
    cache.close_all()
