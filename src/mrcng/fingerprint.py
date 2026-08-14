"""Fingerprint compute/write/validate. fingerprint.json is written last, after
every chunk and info are on disk and fsynced -- its presence is the only
signal a cache entry is complete."""
from __future__ import annotations

import enum
import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from mrcng.reader import pread_exact

SCHEMA_VERSION = 2

_ADDRESSING_FIELDS = ("chunk_size", "encoding", "dtype")

# Bump when a change alters what a build produces: the voxel size or data_type in
# info, the scale plan, the chunk bytes, or the encoding. Bumping invalidates
# every cache entry (Validity.OUTDATED) so they rebuild against the new
# behaviour. NOT bumping leaves the old artifacts served as though nothing
# changed, with no signal anywhere -- that is how a zero-cella-z tilt stack once
# served "resolution": [.., .., 0.0] for weeks after the fix landed (46e8a88).
# The modules this tracks are mrcheader.py, precomputed.py, downsample.py,
# pyramid.py and reader.py.
DERIVATION_VERSION = 1


@dataclass(frozen=True)
class Params:
    chunk_size: tuple[int, int, int]
    downsample: str
    min_axis_size: int
    max_levels: int
    dtype: str
    encoding: str


class Validity(enum.Enum):
    VALID = "valid"
    STALE = "stale"
    INCOMPATIBLE = "incompatible"
    OUTDATED = "outdated"


def compute_header_sha256(fd: int, data_offset: int) -> str:
    raw = pread_exact(fd, data_offset, 0)
    return hashlib.sha256(raw).hexdigest()


def build_fingerprint(fd: int, hdr, relpath: str, params: Params,
                       scales: dict[str, tuple[int, int, int]],
                       generator_version: str, build_duration_s: float) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": generator_version,
        "derivation_version": DERIVATION_VERSION,
        "source_relpath": relpath,
        "source_size": hdr.file_size,
        "source_mtime_ns": hdr.mtime_ns,
        "source_header_sha256": compute_header_sha256(fd, hdr.data_offset),
        "params": asdict(params),
        # key -> [sx, sy, sz]. The sizes let the server validate a requested
        # chunk extent without recomputing the scale plan, which is the last
        # place it would otherwise have to re-derive downsample_z and risk
        # disagreeing with the build that wrote these chunks.
        "scales": {k: list(v) for k, v in scales.items()},
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_duration_s": build_duration_s,
    }


def write_fingerprint(cache_dir: Path, fingerprint: dict) -> None:
    cache_dir = Path(cache_dir)
    path = cache_dir / "fingerprint.json"
    tmp_path = cache_dir / "fingerprint.json.tmp"
    with open(tmp_path, "w") as f:
        json.dump(fingerprint, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)

    dir_fd = os.open(str(cache_dir), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def read_fingerprint(cache_dir: Path) -> dict | None:
    path = Path(cache_dir) / "fingerprint.json"
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    # Valid JSON that isn't an object (e.g. a build crashed mid-write and left
    # `null` or `[]`) is corrupt, not a fingerprint. Ground rule: missing,
    # stale, incompatible, or corrupt must all read as "no cache".
    return data if isinstance(data, dict) else None


def validate(fp: dict, hdr, fd: int, current_params: Params) -> Validity:
    if fp.get("schema_version") != SCHEMA_VERSION:
        return Validity.INCOMPATIBLE

    if fp.get("derivation_version") != DERIVATION_VERSION:
        return Validity.OUTDATED

    fp_params = fp.get("params", {})
    current = asdict(current_params)
    for field in _ADDRESSING_FIELDS:
        fp_value = fp_params.get(field)
        cur_value = current.get(field)
        if field == "chunk_size" and fp_value is not None:
            fp_value = tuple(fp_value)
        if fp_value != cur_value:
            return Validity.INCOMPATIBLE

    if fp.get("source_size") != hdr.file_size or fp.get("source_mtime_ns") != hdr.mtime_ns:
        return Validity.STALE

    if fp.get("source_header_sha256") != compute_header_sha256(fd, hdr.data_offset):
        return Validity.STALE

    return Validity.VALID
