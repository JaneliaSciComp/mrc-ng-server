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

# Two version numbers guard the cache and they answer different questions. Bump
# the wrong one and the cache is not invalidated at all.
#
#   SCHEMA_VERSION      the *shape* of fingerprint.json: which keys exist and what
#                       types their values have. Bump when build_fingerprint
#                       changes its output structure -- a key added, removed,
#                       renamed or retyped. v1 -> v2 did this: "scales" went from
#                       a list of keys to a key -> [sx, sy, sz] mapping.
#                       Mismatch => Validity.INCOMPATIBLE, i.e. "this code cannot
#                       reliably read that file".
#
#   DERIVATION_VERSION  the *content* a build produced: the values in info and the
#                       bytes in the chunk files. The fingerprint's shape is fine;
#                       what changed is the meaning of the artifacts sitting beside
#                       it. Mismatch => Validity.OUTDATED, i.e. "this file is
#                       readable but describes artifacts built by superseded code".
#
# Which one? Ask whether an existing fingerprint would still parse and compare
# correctly. If no, it is a schema change. If yes, but the info/chunks next to it
# are now wrong, it is a derivation change. A change that does both bumps both.
#
# SCHEMA_VERSION is checked first and already rejects every existing entry, so a
# schema bump alone is sufficient -- you never *have* to add a derivation bump on
# top of it, though it is harmless to. The converse does not hold: a derivation
# bump does nothing for a fingerprint whose shape this code cannot read.
#
# Only ever increment. Reusing or lowering either number silently revalidates
# caches built by code that no longer exists.
#
# "generator_version" is a third field, recorded for forensics and deliberately
# never compared: a release bump must not invalidate a whole corpus by itself.
SCHEMA_VERSION = 2

_ADDRESSING_FIELDS = ("chunk_size", "encoding", "dtype")

# Bump when a change alters what a build produces: the voxel size or data_type in
# info, the scale plan, the chunk bytes, or the encoding.
#
# It tracks the modules that decide what a build writes -- today mrcheader.py
# (voxel size, data_type, is_image_stack, byte offsets), precomputed.py
# (plan_scales, build_info, encode_chunk), downsample.py (the voxel arithmetic for
# every level >= 1), pyramid.py (which levels get written, the level-from-level
# cascade, downsample_z) and reader.py (the source voxels that get downsampled).
# Treat that as a consequence of the rule, not the rule itself: if you add a
# module, ask whether its code can change the values in info or the bytes in a
# chunk file. If it can, it is one of these, and changing it needs a bump.
#
# Not tracked, and why they fail differently: server/ only serves what already
# exists; cli.py parses arguments, and the params it passes are fingerprinted
# separately; benchmark.py only measures; this module describes and validates
# rather than producing. paths.py is the subtle one -- dataset_id sets where a
# cache lives, so changing it does not make entries stale, it orphans them, which
# is what `mrc-pyramid prune` is for.
#
# NOT bumping leaves the old artifacts served as though nothing changed, with no
# signal anywhere -- no failing test, no warning from `mrc-pyramid status`. That
# is how a zero-cella-z tilt stack once served "resolution": [.., .., 0.0] for
# weeks after the fix landed (46e8a88). When unsure, bump: a needless rebuild
# costs batch I/O, a missed one costs correctness nobody notices.
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
