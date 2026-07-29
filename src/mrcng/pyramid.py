"""Pyramid build orchestration -- used only by the mrc-pyramid CLI, never by
the server. Level 1 is built by reading directly from the source MRC, one
output chunk (and its corresponding source sub-volume) at a time. Every
level after that is built the same way but reads its source data from the
previous level's cache chunk files instead of the MRC -- so the total cost
is ~1.15 passes over the original volume, not N passes.
"""
from __future__ import annotations

import enum
import fcntl
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from mrcng.downsample import block_mean
from mrcng.fingerprint import (
    Params, build_fingerprint, write_fingerprint, read_fingerprint, validate, Validity,
)
from mrcng.mrcheader import parse_header
from mrcng.paths import resolve_source, dataset_id, cache_dir_for
from mrcng.precomputed import plan_scales, build_info, chunk_name
from mrcng.precomputed import encode_chunk
from mrcng.reader import read_chunk

_logger = logging.getLogger("mrcng.pyramid")

GENERATOR_VERSION = "mrc-pyramid 0.1.0"

# Peak RSS per worker is roughly 3x this: the int16 source block, plus the int32
# accumulator block_mean allocates over it. Budget accordingly when setting
# --jobs.
DEFAULT_MAX_BLOCK_BYTES = 256 << 20


class BuildStatus(enum.Enum):
    BUILT = "built"
    SKIPPED_VALID = "skipped_valid"
    SKIPPED_LOCKED = "skipped_locked"


@dataclass
class BuildResult:
    relpath: str
    dataset_id: str
    status: BuildStatus
    source_bytes: int = 0
    cache_bytes: int = 0
    levels_built: int = 0
    duration_s: float = 0.0
    voxel_size_is_default: bool = False


def _chunk_grid(size, chunk_size):
    """Yield (x0, x1, y0, y1, z0, z1) for every output chunk covering `size`."""
    sx, sy, sz = size
    cx, cy, cz = chunk_size
    for z0 in range(0, sz, cz):
        z1 = min(z0 + cz, sz)
        for y0 in range(0, sy, cy):
            y1 = min(y0 + cy, sy)
            for x0 in range(0, sx, cx):
                x1 = min(x0 + cx, sx)
                yield x0, x1, y0, y1, z0, z1


def _write_chunk(cache_dir: Path, scale_key: str, name: str, arr: np.ndarray) -> int:
    scale_dir = cache_dir / scale_key
    scale_dir.mkdir(parents=True, exist_ok=True)
    body = encode_chunk(np.ascontiguousarray(arr))
    (scale_dir / name).write_bytes(body)
    return len(body)


def _build_level_from_source(fd, hdr, cache_dir: Path, level0, level1, chunk_size,
                              max_block_bytes: int = DEFAULT_MAX_BLOCK_BYTES) -> int:
    """Stream level 1 out of the MRC one output chunk *row* at a time.

    Reading one output chunk at a time instead makes each source read only
    chunk_x * fx columns wide, which is far under a page for the usual 64/int16
    case, so reader.read_chunk picks span-wise and re-reads the whole row prefix
    once per x-chunk: 16.5x the volume in bytes and 32x the syscalls on a
    4096-wide tomogram. Full-width rows are over a page, so the read goes
    row-wise and each source byte is touched once.

    x pieces are cut on output-chunk boundaries, which are multiples of
    chunk_x * fx in source coordinates and therefore aligned to the block-mean
    grid -- so splitting for memory cannot change a single output voxel. The
    only short block is at the volume edge, exactly as when reading one chunk.
    """
    fx, fy, fz = level1.factors  # level0's factors are (1,1,1), so cumulative == per-step here
    cx, cy, cz = chunk_size
    sx, sy, sz = level1.size
    itemsize = hdr.dtype.itemsize
    cache_bytes = 0

    for z0 in range(0, sz, cz):
        z1 = min(z0 + cz, sz)
        for y0 in range(0, sy, cy):
            y1 = min(y0 + cy, sy)
            src_z0, src_z1 = z0 * fz, min(z1 * fz, level0.size[2])
            src_y0, src_y1 = y0 * fy, min(y1 * fy, level0.size[1])

            # bytes one source x-column costs across this band, then how many
            # whole output chunks of x fit in the budget (at least one)
            column_bytes = (src_z1 - src_z0) * (src_y1 - src_y0) * itemsize
            per_chunk_bytes = column_bytes * cx * fx
            step = max(1, max_block_bytes // per_chunk_bytes) * cx

            for px0 in range(0, sx, step):
                px1 = min(px0 + step, sx)
                src_x0, src_x1 = px0 * fx, min(px1 * fx, level0.size[0])

                block = read_chunk(fd, hdr, src_x0, src_x1, src_y0, src_y1, src_z0, src_z1)
                band = block_mean(block, (fz, fy, fx))
                del block

                for x0 in range(px0, px1, cx):
                    x1 = min(x0 + cx, sx)
                    name = chunk_name(x0, x1, y0, y1, z0, z1)
                    cache_bytes += _write_chunk(
                        cache_dir, level1.key, name, band[:, :, x0 - px0:x1 - px0],
                    )
    return cache_bytes


def _read_prev_level_region(cache_dir: Path, scale, chunk_size, dtype,
                             x0: int, x1: int, y0: int, y1: int, z0: int, z1: int) -> np.ndarray:
    """Assemble a (z1-z0, y1-y0, x1-x0) array from one or more of a previously
    built level's whole cache-chunk files (a source region spans at most 2
    previous-level chunks per axis, since the per-level step factor is 1 or 2
    and everything is grid-quantized in multiples of chunk_size)."""
    cx, cy, cz = chunk_size
    sx, sy, sz = scale.size
    out = np.empty((z1 - z0, y1 - y0, x1 - x0), dtype=dtype)

    z = z0
    while z < z1:
        block_z0, block_z1 = (z // cz) * cz, min((z // cz) * cz + cz, sz)
        take_z0, take_z1 = max(z, block_z0), min(z1, block_z1)

        y = y0
        while y < y1:
            block_y0, block_y1 = (y // cy) * cy, min((y // cy) * cy + cy, sy)
            take_y0, take_y1 = max(y, block_y0), min(y1, block_y1)

            x = x0
            while x < x1:
                block_x0, block_x1 = (x // cx) * cx, min((x // cx) * cx + cx, sx)
                take_x0, take_x1 = max(x, block_x0), min(x1, block_x1)

                name = chunk_name(block_x0, block_x1, block_y0, block_y1, block_z0, block_z1)
                raw = (cache_dir / scale.key / name).read_bytes()
                block = np.frombuffer(raw, dtype=dtype).reshape(
                    block_z1 - block_z0, block_y1 - block_y0, block_x1 - block_x0
                )
                out[
                    take_z0 - z0: take_z1 - z0,
                    take_y0 - y0: take_y1 - y0,
                    take_x0 - x0: take_x1 - x0,
                ] = block[
                    take_z0 - block_z0: take_z1 - block_z0,
                    take_y0 - block_y0: take_y1 - block_y0,
                    take_x0 - block_x0: take_x1 - block_x0,
                ]
                x = block_x1
            y = block_y1
        z = block_z1

    return out


def _build_level_from_previous(cache_dir: Path, prev_scale, next_scale, chunk_size, dtype) -> int:
    fx = next_scale.factors[0] // prev_scale.factors[0]
    fy = next_scale.factors[1] // prev_scale.factors[1]
    fz = next_scale.factors[2] // prev_scale.factors[2]
    cache_bytes = 0

    for x0, x1, y0, y1, z0, z1 in _chunk_grid(next_scale.size, chunk_size):
        src_x0, src_x1 = x0 * fx, min(x1 * fx, prev_scale.size[0])
        src_y0, src_y1 = y0 * fy, min(y1 * fy, prev_scale.size[1])
        src_z0, src_z1 = z0 * fz, min(z1 * fz, prev_scale.size[2])

        block = _read_prev_level_region(
            cache_dir, prev_scale, chunk_size, dtype, src_x0, src_x1, src_y0, src_y1, src_z0, src_z1,
        )
        downsampled = block_mean(block, (fz, fy, fx))
        cache_bytes += _write_chunk(cache_dir, next_scale.key, chunk_name(x0, x1, y0, y1, z0, z1), downsampled)
    return cache_bytes


def _fsync_tree(root: Path) -> None:
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            fd = os.open(os.path.join(dirpath, name), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        dir_fd = os.open(dirpath, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


def _open_source(source_root, relpath: str, assume_mode0: str | None = None):
    path = resolve_source(source_root, relpath)
    fd = os.open(str(path), os.O_RDONLY)
    st = os.stat(fd)
    hdr = parse_header(fd, st.st_size, st.st_mtime_ns, assume_mode0=assume_mode0)
    if hdr.mode0_signedness_is_ambiguous:
        _logger.warning(
            "%s: mode-0 signedness is ambiguous (no IMOD stamp), defaulting to "
            "int8; pass --assume-mode0 to override", path,
        )
    return fd, hdr


def build_one(source_root, cache_root, relpath: str, params: Params, force: bool = False,
              max_block_bytes: int = DEFAULT_MAX_BLOCK_BYTES,
              assume_mode0: str | None = None) -> BuildResult:
    source_root, cache_root = Path(source_root), Path(cache_root)
    start = time.monotonic()
    ds_id = dataset_id(relpath)
    cache_dir = cache_dir_for(cache_root, ds_id)

    fd, hdr = _open_source(source_root, relpath, assume_mode0)
    try:
        # dtype is a per-file property derived from the header, not a caller
        # chosen build setting -- a source tree can mix int16 tomograms with
        # float32 ones, so params.dtype must reflect *this* file, never a
        # value the caller guessed for the whole tree.
        params = replace(params, dtype=hdr.dtype.name)

        existing = read_fingerprint(cache_dir)
        if existing is not None and not force and validate(existing, hdr, fd, params) == Validity.VALID:
            return BuildResult(relpath, ds_id, BuildStatus.SKIPPED_VALID, source_bytes=hdr.file_size,
                               voxel_size_is_default=hdr.voxel_size_is_default)

        cache_dir.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(str(cache_dir / ".lock"), os.O_CREAT | os.O_RDWR)
        try:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return BuildResult(relpath, ds_id, BuildStatus.SKIPPED_LOCKED, source_bytes=hdr.file_size,
                                   voxel_size_is_default=hdr.voxel_size_is_default)

            fp_path = cache_dir / "fingerprint.json"
            if fp_path.exists():
                fp_path.unlink()

            # Drop every existing scale dir before rebuilding. A previous build
            # of a differently-shaped source leaves chunk files whose names are
            # not in the new grid; they survive an overwrite and stay readable
            # under the *new* valid fingerprint. Only the level dirs go -- the
            # flock we are holding lives on cache_dir/.lock.
            for child in cache_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)

            scales = plan_scales((hdr.nx, hdr.ny, hdr.nz), params.min_axis_size, params.max_levels)
            cache_bytes = 0
            levels_built = 0

            if len(scales) > 1:
                cache_bytes += _build_level_from_source(
                    fd, hdr, cache_dir, scales[0], scales[1], params.chunk_size, max_block_bytes,
                )
                levels_built += 1
                for i in range(2, len(scales)):
                    cache_bytes += _build_level_from_previous(
                        cache_dir, scales[i - 1], scales[i], params.chunk_size, hdr.dtype,
                    )
                    levels_built += 1

            info = build_info(hdr, scales, params.chunk_size, params.encoding)
            (cache_dir / "info").write_text(json.dumps(info))

            _fsync_tree(cache_dir)

            fp = build_fingerprint(
                fd, hdr, relpath, params,
                scales=[s.key for s in scales[1:]],
                generator_version=GENERATOR_VERSION,
                build_duration_s=time.monotonic() - start,
            )
            write_fingerprint(cache_dir, fp)

            return BuildResult(
                relpath, ds_id, BuildStatus.BUILT,
                source_bytes=hdr.file_size, cache_bytes=cache_bytes,
                levels_built=levels_built, duration_s=time.monotonic() - start,
                voxel_size_is_default=hdr.voxel_size_is_default,
            )
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
    finally:
        os.close(fd)
