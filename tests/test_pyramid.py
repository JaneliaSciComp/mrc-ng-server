import os
import fcntl

import numpy as np
import pytest

from mrcng.fingerprint import Params, read_fingerprint
from mrcng.paths import dataset_id, cache_dir_for
from mrcng.pyramid import build_one, BuildStatus


def _params(**overrides):
    base = dict(chunk_size=(8, 8, 8), downsample="mean", min_axis_size=8,
                max_levels=3, dtype="int16", encoding="raw")
    base.update(overrides)
    return Params(**base)


@pytest.fixture
def source_and_cache(tmp_path, make_mrc_file):
    source_root = tmp_path / "source"
    source_root.mkdir()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    def fill(zz, yy, xx):
        return (xx + 1000 * yy + 1_000_000 * zz) % 30000

    make_mrc_file(name="source/tomo.mrc", shape=(32, 32, 32), mode=1, fill=fill)
    return source_root, cache_root, "tomo.mrc"


def test_build_creates_fingerprint_and_scales(source_and_cache):
    source_root, cache_root, relpath = source_and_cache
    result = build_one(source_root, cache_root, relpath, _params())
    assert result.status == BuildStatus.BUILT
    assert result.levels_built >= 1

    ds_id = dataset_id(relpath)
    cache_dir = cache_dir_for(cache_root, ds_id)
    fp = read_fingerprint(cache_dir)
    assert fp is not None
    assert fp["source_relpath"] == relpath


def test_build_output_matches_in_memory_reference_downsample(source_and_cache):
    source_root, cache_root, relpath = source_and_cache
    build_one(source_root, cache_root, relpath, _params())

    import mrcfile
    from mrcng.downsample import block_mean
    from mrcng.precomputed import chunk_name

    with mrcfile.open(source_root / relpath, permissive=True) as mf:
        level0 = mf.data  # (nz, ny, nx)

    expected_level1 = block_mean(level0, (2, 2, 2))  # factors in (z,y,x) order
    ds_id = dataset_id(relpath)
    cache_dir = cache_dir_for(cache_root, ds_id)

    # first output chunk of level 2_2_2 (16x16x16 volume, chunk_size 8x8x8)
    name = chunk_name(0, 8, 0, 8, 0, 8)
    chunk_path = cache_dir / "2_2_2" / name
    assert chunk_path.exists()
    on_disk = np.fromfile(chunk_path, dtype="<i2").reshape(8, 8, 8)
    np.testing.assert_array_equal(on_disk, expected_level1[0:8, 0:8, 0:8])


def test_build_records_actual_file_dtype_not_the_callers_guess(tmp_path, make_mrc_file):
    # regression test: caller passes dtype="int16" (e.g. a CLI default meant
    # for a whole source tree), but this particular file is float32 -- the
    # written fingerprint must reflect the real per-file dtype, or the
    # server's later validate() call will wrongly see it as INCOMPATIBLE.
    source_root = tmp_path / "source"
    source_root.mkdir()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    make_mrc_file(name="source/tomo.mrc", shape=(16, 16, 16), mode=2)  # float32

    result = build_one(source_root, cache_root, "tomo.mrc", _params(dtype="int16"))
    assert result.status == BuildStatus.BUILT

    ds_id = dataset_id("tomo.mrc")
    cache_dir = cache_dir_for(cache_root, ds_id)
    fp = read_fingerprint(cache_dir)
    assert fp["params"]["dtype"] == "float32"


def test_skips_valid_cache_unless_forced(source_and_cache):
    source_root, cache_root, relpath = source_and_cache
    first = build_one(source_root, cache_root, relpath, _params())
    assert first.status == BuildStatus.BUILT

    second = build_one(source_root, cache_root, relpath, _params())
    assert second.status == BuildStatus.SKIPPED_VALID

    forced = build_one(source_root, cache_root, relpath, _params(), force=True)
    assert forced.status == BuildStatus.BUILT


def test_concurrent_build_reports_skipped_locked(source_and_cache):
    source_root, cache_root, relpath = source_and_cache
    ds_id = dataset_id(relpath)
    cache_dir = cache_dir_for(cache_root, ds_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / ".lock"
    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    try:
        result = build_one(source_root, cache_root, relpath, _params())
        assert result.status == BuildStatus.SKIPPED_LOCKED
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def test_killed_build_leaves_no_fingerprint_and_is_rebuilt(source_and_cache, monkeypatch):
    source_root, cache_root, relpath = source_and_cache

    from mrcng import pyramid as pyramid_module
    original = pyramid_module.write_fingerprint

    def boom(*args, **kwargs):
        raise RuntimeError("simulated crash before fingerprint write")

    monkeypatch.setattr(pyramid_module, "write_fingerprint", boom)
    with pytest.raises(RuntimeError):
        build_one(source_root, cache_root, relpath, _params())

    ds_id = dataset_id(relpath)
    cache_dir = cache_dir_for(cache_root, ds_id)
    assert read_fingerprint(cache_dir) is None

    monkeypatch.setattr(pyramid_module, "write_fingerprint", original)
    result = build_one(source_root, cache_root, relpath, _params())
    assert result.status == BuildStatus.BUILT
