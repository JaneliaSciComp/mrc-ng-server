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

SCHEMA_VERSION = 1

_ADDRESSING_FIELDS = ("chunk_size", "encoding", "dtype")


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


def compute_header_sha256(fd: int, data_offset: int) -> str:
    raw = pread_exact(fd, data_offset, 0)
    return hashlib.sha256(raw).hexdigest()


def build_fingerprint(fd: int, hdr, relpath: str, params: Params, scales: list[str],
                       generator_version: str, build_duration_s: float) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": generator_version,
        "source_relpath": relpath,
        "source_size": hdr.file_size,
        "source_mtime_ns": hdr.mtime_ns,
        "source_header_sha256": compute_header_sha256(fd, hdr.data_offset),
        "params": asdict(params),
        "scales": list(scales),
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
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def validate(fp: dict, hdr, fd: int, current_params: Params) -> Validity:
    if fp.get("schema_version") != SCHEMA_VERSION:
        return Validity.INCOMPATIBLE

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
