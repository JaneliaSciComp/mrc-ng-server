import os
import time

import pytest

from mrcng.mrcheader import parse_header
from mrcng.fingerprint import (
    Params, Validity, build_fingerprint,
    write_fingerprint, read_fingerprint, validate,
)


def _params(**overrides):
    base = dict(chunk_size=(64, 64, 64), downsample="mean", min_axis_size=32,
                max_levels=6, dtype="int16", encoding="raw")
    base.update(overrides)
    return Params(**base)


def _open(path):
    fd = os.open(str(path), os.O_RDONLY)
    st = os.stat(fd)
    return fd, parse_header(fd, st.st_size, st.st_mtime_ns)


def test_write_then_read_roundtrip(tmp_path, make_mrc_file):
    path = make_mrc_file(shape=(8, 8, 8), mode=1)
    fd, hdr = _open(path)
    try:
        fp = build_fingerprint(
            fd, hdr, relpath="tomo.mrc", params=_params(),
            scales=["2_2_2"], generator_version="mrc-pyramid 0.1.0",
            build_duration_s=1.23,
        )
        cache_dir = tmp_path / "cache_entry"
        cache_dir.mkdir()
        write_fingerprint(cache_dir, fp)
        loaded = read_fingerprint(cache_dir)
        assert loaded["source_relpath"] == "tomo.mrc"
        assert loaded["scales"] == ["2_2_2"]
    finally:
        os.close(fd)


def test_read_fingerprint_missing_returns_none(tmp_path):
    assert read_fingerprint(tmp_path / "nope") is None


@pytest.mark.parametrize("contents", ["null", "[]", '"a string"', "42"])
def test_read_fingerprint_non_object_json_returns_none(tmp_path, contents):
    # Regression: valid JSON that isn't a dict (e.g. a build crashed mid-write)
    # used to be returned as-is, and validate()'s fp.get(...) then raised
    # AttributeError instead of reading as "no cache".
    cache_dir = tmp_path / "entry"
    cache_dir.mkdir()
    (cache_dir / "fingerprint.json").write_text(contents)
    assert read_fingerprint(cache_dir) is None


def test_validate_valid_when_everything_matches(tmp_path, make_mrc_file):
    path = make_mrc_file(shape=(8, 8, 8), mode=1)
    fd, hdr = _open(path)
    try:
        fp = build_fingerprint(fd, hdr, "tomo.mrc", _params(), ["2_2_2"], "v0.1.0", 1.0)
        assert validate(fp, hdr, fd, _params()) == Validity.VALID
    finally:
        os.close(fd)


def test_validate_stale_when_source_mtime_changes(tmp_path, make_mrc_file):
    path = make_mrc_file(shape=(8, 8, 8), mode=1)
    fd, hdr = _open(path)
    try:
        fp = build_fingerprint(fd, hdr, "tomo.mrc", _params(), ["2_2_2"], "v0.1.0", 1.0)
    finally:
        os.close(fd)

    time.sleep(0.01)
    os.utime(path, None)  # bump mtime

    fd2 = os.open(str(path), os.O_RDONLY)
    try:
        st = os.stat(fd2)
        hdr2 = parse_header(fd2, st.st_size, st.st_mtime_ns)
        assert validate(fp, hdr2, fd2, _params()) == Validity.STALE
    finally:
        os.close(fd2)


def test_validate_incompatible_when_chunk_size_differs(tmp_path, make_mrc_file):
    path = make_mrc_file(shape=(8, 8, 8), mode=1)
    fd, hdr = _open(path)
    try:
        fp = build_fingerprint(fd, hdr, "tomo.mrc", _params(), ["2_2_2"], "v0.1.0", 1.0)
        different_params = _params(chunk_size=(32, 32, 32))
        assert validate(fp, hdr, fd, different_params) == Validity.INCOMPATIBLE
    finally:
        os.close(fd)


def test_validate_incompatible_when_schema_version_unknown(tmp_path, make_mrc_file):
    path = make_mrc_file(shape=(8, 8, 8), mode=1)
    fd, hdr = _open(path)
    try:
        fp = build_fingerprint(fd, hdr, "tomo.mrc", _params(), ["2_2_2"], "v0.1.0", 1.0)
        fp["schema_version"] = 999
        assert validate(fp, hdr, fd, _params()) == Validity.INCOMPATIBLE
    finally:
        os.close(fd)


def test_generator_version_change_alone_stays_valid(tmp_path, make_mrc_file):
    path = make_mrc_file(shape=(8, 8, 8), mode=1)
    fd, hdr = _open(path)
    try:
        fp = build_fingerprint(fd, hdr, "tomo.mrc", _params(), ["2_2_2"], "v0.1.0", 1.0)
        fp["generator_version"] = "mrc-pyramid 9.9.9"
        assert validate(fp, hdr, fd, _params()) == Validity.VALID
    finally:
        os.close(fd)
