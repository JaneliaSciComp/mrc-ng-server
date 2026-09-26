import json
import os
import fcntl

import numpy as np
import pytest

from mrcng.fingerprint import Params, Validity, read_fingerprint, validate
from mrcng.mrcheader import parse_header
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
    chunk_path = cache_dir / "precomputed" / "2_2_2" / name
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


def test_block_splitting_does_not_change_any_output_voxel(tmp_path, make_mrc_file):
    """The memory budget cuts the x range into pieces. Pieces land on output-chunk
    boundaries, which are factor-aligned, so a tiny budget and an unlimited one
    must produce byte-identical chunk trees. Misaligned pieces would silently
    change edge voxels of every piece."""
    source_root = tmp_path / "source"; source_root.mkdir()
    make_mrc_file(name="source/tomo.mrc", shape=(101, 37, 13), mode=1,
                  fill=lambda zz, yy, xx: (xx + 1000 * yy + 1_000_000 * zz) % 30000)

    trees = []
    for budget in (1, 1 << 30):  # 1 byte forces one output chunk per read
        cache_root = tmp_path / f"cache{budget}"
        cache_root.mkdir()
        build_one(source_root, cache_root, "tomo.mrc", _params(), max_block_bytes=budget)
        cache_dir = cache_dir_for(cache_root, dataset_id("tomo.mrc"))
        trees.append({
            p.relative_to(cache_dir).as_posix(): p.read_bytes()
            for p in sorted(cache_dir.rglob("*")) if p.is_file() and p.name != "fingerprint.json"
        })

    assert trees[0].keys() == trees[1].keys()
    assert len(trees[0]) > 1
    for name in trees[0]:
        assert trees[0][name] == trees[1][name], f"{name} differs between block budgets"


def test_level1_pass_reads_each_source_byte_about_once(tmp_path, make_mrc_file, monkeypatch):
    """Regression: building one output chunk at a time made each source read
    narrower than a page, so span-wise re-read the row prefix per x-chunk --
    16.5x the volume in bytes, 32x the syscalls."""
    import os
    from mrcng import reader as reader_module

    source_root = tmp_path / "source"; source_root.mkdir()
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    nx, ny, nz = 2048, 64, 16
    make_mrc_file(name="source/tomo.mrc", shape=(nx, ny, nz), mode=1)

    total = 0
    real_pread = os.pread

    def counting(fd, n, off):
        nonlocal total
        buf = real_pread(fd, n, off)
        total += len(buf)
        return buf

    monkeypatch.setattr(reader_module.os, "pread", counting)
    build_one(source_root, cache_root, "tomo.mrc", _params())

    volume_bytes = nx * ny * nz * 2
    assert total / volume_bytes < 1.2, f"read {total / volume_bytes:.1f}x the volume"


def test_skips_valid_cache_unless_forced(source_and_cache):
    source_root, cache_root, relpath = source_and_cache
    first = build_one(source_root, cache_root, relpath, _params())
    assert first.status == BuildStatus.BUILT

    second = build_one(source_root, cache_root, relpath, _params())
    assert second.status == BuildStatus.SKIPPED_VALID

    forced = build_one(source_root, cache_root, relpath, _params(), force=True)
    assert forced.status == BuildStatus.BUILT


def test_fingerprint_records_the_installed_version(source_and_cache):
    # Regression: GENERATOR_VERSION hardcoded the version number, so bumping
    # pyproject stamped every new fingerprint with the previous release. It is
    # the only record of which build wrote a cache entry, so it must not lie.
    from importlib.metadata import version

    source_root, cache_root, relpath = source_and_cache
    build_one(source_root, cache_root, relpath, _params())
    fp = read_fingerprint(cache_dir_for(cache_root, dataset_id(relpath)))
    assert fp["generator_version"] == f"mrc-pyramid {version('mrc-ng-server')}"


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


def test_image_stack_pyramid_never_bins_z(tmp_path, make_mrc_file):
    """End-to-end: the built cache must keep every slice at every level.

    Guards the wiring, not plan_scales (unit-tested separately): if the build
    stopped passing downsample_z it would write 2_2_2 chunks holding the mean
    of adjacent tilts, and the server -- which derives downsample_z from the
    same header -- would then advertise 2_2_1 and never find them.
    """
    source_root = tmp_path / "source"
    source_root.mkdir()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    # Classified a stack by glob, not by shape. Each slice is a distinct constant,
    # so a z-averaged level would be immediately visible.
    make_mrc_file(name="source/stack.mrc", shape=(256, 256, 8), mode=1,
                  fill=lambda zz, yy, xx: (zz + 1) * 100)
    result = build_one(source_root, cache_root, "stack.mrc",
                       _params(chunk_size=(64, 64, 64), min_axis_size=32),
                       stack_globs=("*stack.mrc",))
    assert result.status == BuildStatus.BUILT

    cache_dir = cache_dir_for(cache_root, dataset_id("stack.mrc"))
    fp = read_fingerprint(cache_dir)
    assert list(fp["scales"]) == ["2_2_1", "4_4_1"]

    info = json.loads((cache_dir / "precomputed" / "info").read_text())
    assert info["is_image_stack"] is True
    for scale in info["scales"]:
        assert scale["size"][2] == 8, scale["key"]
        assert scale["resolution"][2] == 1.0, scale["key"]

    # Level 1 holds all 8 original slice values, unaveraged.
    chunk = np.frombuffer((cache_dir / "precomputed" / "2_2_1" / "0-64_0-64_0-8").read_bytes(),
                          dtype="<i2").reshape(8, 64, 64)
    assert [int(chunk[z, 0, 0]) for z in range(8)] == [(z + 1) * 100 for z in range(8)]


def test_reclassifying_a_file_invalidates_its_cache(tmp_path, make_mrc_file):
    """A glob change must not leave the old classification's artifacts in play.

    The classification is an operator input, so nothing in the header or the
    source bytes changes when it flips -- without the fingerprint recording it,
    the entry would still read VALID and keep serving a z resolution and scale
    plan built from the other answer. This is also what makes a server/builder
    glob mismatch fail safe rather than silently.
    """
    source_root = tmp_path / "source"
    source_root.mkdir()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    make_mrc_file(name="source/s.mrc", shape=(256, 256, 8), mode=1,
                  fill=lambda zz, yy, xx: (zz + 1) * 100)
    params = _params(chunk_size=(64, 64, 64), min_axis_size=32)

    built = build_one(source_root, cache_root, "s.mrc", params, stack_globs=("*s.mrc",))
    assert built.status == BuildStatus.BUILT

    cache_dir = cache_dir_for(cache_root, dataset_id("s.mrc"))
    fp = read_fingerprint(cache_dir)
    assert fp["is_image_stack"] is True

    # Same file, same params, same everything except the operator's globs.
    fd = os.open(str(source_root / "s.mrc"), os.O_RDONLY)
    try:
        st = os.stat(fd)
        as_volume = parse_header(fd, st.st_size, st.st_mtime_ns, is_image_stack=False)
        assert validate(fp, as_volume, fd, params) == Validity.INCOMPATIBLE
        as_stack = parse_header(fd, st.st_size, st.st_mtime_ns, is_image_stack=True)
        assert validate(fp, as_stack, fd, params) == Validity.VALID
    finally:
        os.close(fd)

    # ...and a plain rebuild (no --force) picks the change up rather than skipping.
    rebuilt = build_one(source_root, cache_root, "s.mrc", params, stack_globs=())
    assert rebuilt.status == BuildStatus.BUILT
    assert read_fingerprint(cache_dir)["is_image_stack"] is False


def _read_zarr_chunk(cache_dir, level, kz, ky, kx, chunk_shape_zyx, dtype="<i2"):
    from mrcng.omezarr import chunk_rel_path
    raw = (cache_dir / "omezarr" / chunk_rel_path(level, kz, ky, kx)).read_bytes()
    return np.frombuffer(raw, dtype=dtype).reshape(chunk_shape_zyx)


def test_default_build_writes_both_layouts_and_records_formats(source_and_cache):
    source_root, cache_root, relpath = source_and_cache
    result = build_one(source_root, cache_root, relpath, _params())
    assert result.status == BuildStatus.BUILT
    assert result.formats == ("precomputed", "omezarr")

    cache_dir = cache_dir_for(cache_root, dataset_id(relpath))
    assert read_fingerprint(cache_dir)["formats"] == ["precomputed", "omezarr"]
    assert (cache_dir / "precomputed" / "info").is_file()
    assert (cache_dir / "omezarr" / "zarr.json").is_file()
    # metadata for every level including 0 (which has no chunks), chunks for 1..N
    assert (cache_dir / "omezarr" / "0" / "zarr.json").is_file()
    assert not (cache_dir / "omezarr" / "0" / "c").exists()
    assert (cache_dir / "omezarr" / "1" / "zarr.json").is_file()
    assert (cache_dir / "omezarr" / "1" / "c" / "0" / "0" / "0").is_file()
    # nothing from the old layout at the entry root
    assert not (cache_dir / "info").exists() and not (cache_dir / "2_2_2").exists()


def test_omezarr_chunks_equal_precomputed_chunks_after_padding(tmp_path, make_mrc_file):
    """Interior chunks are byte-identical; edge chunks equal the clipped
    precomputed chunk zero-padded to chunk_shape."""
    from mrcng.precomputed import chunk_name
    from mrcng.omezarr import pad_chunk

    source_root = tmp_path / "source"; source_root.mkdir()
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    # 36x20x12 -> level 1 is 18x10x6 with chunk 8: edges on every axis
    make_mrc_file(name="source/t.mrc", shape=(36, 20, 12), mode=1,
                  fill=lambda zz, yy, xx: (xx + 1000 * yy + 1_000_000 * zz) % 30000)
    build_one(source_root, cache_root, "t.mrc", _params())
    cache_dir = cache_dir_for(cache_root, dataset_id("t.mrc"))
    fp = read_fingerprint(cache_dir)
    assert fp["scales"]["2_2_2"] == [18, 10, 6]

    checked = 0
    for kz in range(1):
        for ky in range(2):
            for kx in range(3):
                x0, y0, z0 = kx * 8, ky * 8, kz * 8
                x1, y1, z1 = min(x0 + 8, 18), min(y0 + 8, 10), min(z0 + 8, 6)
                pre = np.frombuffer(
                    (cache_dir / "precomputed" / "2_2_2" / chunk_name(x0, x1, y0, y1, z0, z1)).read_bytes(),
                    dtype="<i2").reshape(z1 - z0, y1 - y0, x1 - x0)
                zarr_chunk = _read_zarr_chunk(cache_dir, 1, kz, ky, kx, (8, 8, 8))
                np.testing.assert_array_equal(zarr_chunk, pad_chunk(pre, (8, 8, 8)))
                checked += 1
    assert checked == 6


def test_omezarr_only_build_matches_both_formats_build(tmp_path, make_mrc_file):
    """Level 2 is downsampled from level 1's *cache files*. With only omezarr
    selected those files are padded, so the reader must slice the padding off
    or level 2 averages zeros into its edge voxels."""
    source_root = tmp_path / "source"; source_root.mkdir()
    make_mrc_file(name="source/t.mrc", shape=(36, 20, 12), mode=1,
                  fill=lambda zz, yy, xx: (xx + 1000 * yy + 1_000_000 * zz) % 30000)

    both_root = tmp_path / "both"; both_root.mkdir()
    only_root = tmp_path / "only"; only_root.mkdir()
    build_one(source_root, both_root, "t.mrc", _params())
    result = build_one(source_root, only_root, "t.mrc", _params(), formats=("omezarr",))
    assert result.formats == ("omezarr",)

    both_dir = cache_dir_for(both_root, dataset_id("t.mrc"))
    only_dir = cache_dir_for(only_root, dataset_id("t.mrc"))
    assert not (only_dir / "precomputed").exists()
    assert read_fingerprint(only_dir)["formats"] == ["omezarr"]

    both_tree = {p.relative_to(both_dir / "omezarr").as_posix(): p.read_bytes()
                 for p in sorted((both_dir / "omezarr").rglob("*")) if p.is_file()}
    only_tree = {p.relative_to(only_dir / "omezarr").as_posix(): p.read_bytes()
                 for p in sorted((only_dir / "omezarr").rglob("*")) if p.is_file()}
    assert any(k.startswith("2/c/") for k in both_tree)  # a level-2 actually exists
    assert only_tree == both_tree


def test_rebuild_with_different_formats_replaces_the_entry(source_and_cache):
    source_root, cache_root, relpath = source_and_cache
    build_one(source_root, cache_root, relpath, _params())
    cache_dir = cache_dir_for(cache_root, dataset_id(relpath))

    # same formats -> skipped; a different set is not "valid" for this request
    assert build_one(source_root, cache_root, relpath, _params()).status == BuildStatus.SKIPPED_VALID
    result = build_one(source_root, cache_root, relpath, _params(), formats=("omezarr",))
    assert result.status == BuildStatus.BUILT
    assert not (cache_dir / "precomputed").exists()
    assert read_fingerprint(cache_dir)["formats"] == ["omezarr"]

    result = build_one(source_root, cache_root, relpath, _params(), formats=("precomputed",))
    assert result.status == BuildStatus.BUILT
    assert not (cache_dir / "omezarr").exists()
    assert (cache_dir / "precomputed" / "info").is_file()


def test_sub_chunk_volume_produces_one_fully_padded_chunk(tmp_path, make_mrc_file):
    source_root = tmp_path / "source"; source_root.mkdir()
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    # 10x12x14 at chunk 8 and min_axis_size 4 -> level 1 is 5x6x7, a single all-edge chunk
    make_mrc_file(name="source/small.mrc", shape=(10, 12, 14), mode=1,
                  fill=lambda zz, yy, xx: np.full_like(xx, 7))
    build_one(source_root, cache_root, "small.mrc", _params(min_axis_size=4, max_levels=2))
    cache_dir = cache_dir_for(cache_root, dataset_id("small.mrc"))
    chunk = _read_zarr_chunk(cache_dir, 1, 0, 0, 0, (8, 8, 8))
    assert chunk.nbytes == 8 * 8 * 8 * 2
    np.testing.assert_array_equal(chunk[:7, :6, :5], 7)
    assert chunk[7:].sum() == 0 and chunk[:, 6:].sum() == 0 and chunk[:, :, 5:].sum() == 0


def test_image_stack_omezarr_keeps_every_slice(tmp_path, make_mrc_file):
    import json as _json
    source_root = tmp_path / "source"; source_root.mkdir()
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    make_mrc_file(name="source/stack.mrc", shape=(256, 256, 8), mode=1,
                  fill=lambda zz, yy, xx: (zz + 1) * 100)
    build_one(source_root, cache_root, "stack.mrc",
              _params(chunk_size=(64, 64, 64), min_axis_size=32), stack_globs=("*stack.mrc",))
    cache_dir = cache_dir_for(cache_root, dataset_id("stack.mrc"))

    arr1 = _json.loads((cache_dir / "omezarr" / "1" / "zarr.json").read_text())
    assert arr1["shape"] == [8, 128, 128]
    group = _json.loads((cache_dir / "omezarr" / "zarr.json").read_text())
    ds1 = group["attributes"]["ome"]["multiscales"][0]["datasets"][1]["coordinateTransformations"]
    assert ds1[0]["scale"][0] == 1.0 and ds1[1]["translation"][0] == 0.0

    chunk = _read_zarr_chunk(cache_dir, 1, 0, 0, 0, (64, 64, 64))
    assert [int(chunk[z, 0, 0]) for z in range(8)] == [(z + 1) * 100 for z in range(8)]
    assert chunk[8:].sum() == 0  # padded z


def test_omezarr_layout_opens_with_the_zarr_library(source_and_cache):
    """The one check that our hand-written zarr.json files are what a real
    Zarr v3 implementation expects. Level 0 has no chunks on disk by design,
    so it reads back as fill_value -- asserted here so nobody mistakes the
    cache for a standalone store."""
    import zarr
    from mrcng.downsample import block_mean
    import mrcfile

    source_root, cache_root, relpath = source_and_cache
    build_one(source_root, cache_root, relpath, _params())
    cache_dir = cache_dir_for(cache_root, dataset_id(relpath))

    group = zarr.open_group(str(cache_dir / "omezarr"), mode="r", zarr_format=3)
    ome = group.attrs["ome"]
    assert ome["version"] == "0.5"
    assert [d["path"] for d in ome["multiscales"][0]["datasets"]] == ["0", "1", "2"]

    with mrcfile.open(source_root / relpath, permissive=True) as mf:
        level0 = np.asarray(mf.data)
    level1 = group["1"]
    assert level1.shape == (16, 16, 16) and level1.dtype == np.dtype("int16")
    assert level1.chunks == (8, 8, 8)
    np.testing.assert_array_equal(level1[:], block_mean(level0, (2, 2, 2)))
    level2 = group["2"]
    np.testing.assert_array_equal(level2[:], block_mean(block_mean(level0, (2, 2, 2)), (2, 2, 2)))

    assert group["0"].shape == (32, 32, 32)
    assert group["0"][:].sum() == 0   # serving cache, not a standalone store
