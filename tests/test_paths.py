import os
import hashlib

import pytest

from mrcng.paths import dataset_id, cache_dir_for, resolve_source, PathNotAllowed


def test_dataset_id_is_deterministic_sha256_prefix():
    rel = "session42/tomo_0031.mrc"
    expected = hashlib.sha256(rel.encode()).hexdigest()[:16]
    assert dataset_id(rel) == expected
    assert dataset_id(rel) == dataset_id(rel)


@pytest.mark.parametrize("spelling", ["sub//t.mrc", "./sub/t.mrc", "sub/./t.mrc"])
def test_dataset_id_normalizes_equivalent_spellings(spelling):
    # Regression: hashing the raw request string meant a proxy or client that
    # joined URL segments naively (double slash, leading "./") landed on a
    # different, empty cache entry -- the whole pyramid silently vanished even
    # though resolve_source serves scale 0 from the same file either way.
    assert dataset_id(spelling) == dataset_id("sub/t.mrc")


def test_cache_dir_uses_two_char_prefix(tmp_path):
    ds_id = "abcdef0123456789"
    result = cache_dir_for(tmp_path, ds_id)
    assert result == tmp_path / "ab" / "abcdef0123456789"


def test_resolve_source_happy_path(tmp_path):
    (tmp_path / "sub").mkdir()
    f = tmp_path / "sub" / "a.mrc"
    f.write_bytes(b"x")
    assert resolve_source(tmp_path, "sub/a.mrc") == f.resolve()


@pytest.mark.parametrize("bad_relpath", [
    "../escape.mrc",
    "sub/../../escape.mrc",
    "/etc/passwd",
    "",
    "sub/\x00null.mrc",
])
def test_resolve_source_rejects_unsafe_paths(tmp_path, bad_relpath):
    with pytest.raises(PathNotAllowed):
        resolve_source(tmp_path, bad_relpath)


def test_resolve_source_rejects_missing_file(tmp_path):
    with pytest.raises(PathNotAllowed):
        resolve_source(tmp_path, "does_not_exist.mrc")


def test_resolve_source_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside_target.mrc"
    outside.write_bytes(b"secret")
    root = tmp_path / "root"
    root.mkdir()
    link = root / "escape.mrc"
    os.symlink(outside, link)
    try:
        with pytest.raises(PathNotAllowed):
            resolve_source(root, "escape.mrc")
    finally:
        outside.unlink()
