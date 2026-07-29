"""Neuroglancer precomputed protocol: scale planning, info JSON, chunk
naming and raw encoding. Chunk-name bounds are x0-x1_y0-y1_z0-z1, clipped to
the scale's size -- edge chunks are smaller than chunk_size and must be
requested/served as their clipped extent, never padded."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

_CHUNK_RE = re.compile(r"^(\d+)-(\d+)_(\d+)-(\d+)_(\d+)-(\d+)$")


@dataclass(frozen=True)
class ScaleLevel:
    key: str
    size: tuple[int, int, int]
    factors: tuple[int, int, int]


def plan_scales(size0: tuple[int, int, int], min_axis_size: int = 32, max_levels: int = 6) -> list[ScaleLevel]:
    levels = [ScaleLevel(key="1_1_1", size=tuple(size0), factors=(1, 1, 1))]
    while len(levels) < max_levels:
        prev = levels[-1]
        step = tuple(2 if s > min_axis_size else 1 for s in prev.size)
        if step == (1, 1, 1):
            break
        new_factors = tuple(f * s for f, s in zip(prev.factors, step))
        new_size = tuple(math.ceil(s / st) for s, st in zip(prev.size, step))
        key = f"{new_factors[0]}_{new_factors[1]}_{new_factors[2]}"
        levels.append(ScaleLevel(key=key, size=new_size, factors=new_factors))
    return levels


def build_info(hdr, scales: list[ScaleLevel], chunk_size: tuple[int, int, int], encoding: str = "raw") -> dict:
    base_res_nm = tuple(a / 10.0 for a in hdr.voxel_size_angstrom)
    scale_entries = []
    for lvl in scales:
        resolution = [base_res_nm[i] * lvl.factors[i] for i in range(3)]
        scale_entries.append({
            "key": lvl.key,
            "size": list(lvl.size),
            "resolution": resolution,
            "voxel_offset": [0, 0, 0],
            "chunk_sizes": [list(chunk_size)],
            "encoding": encoding,
        })
    return {
        "@type": "neuroglancer_multiscale_volume",
        "type": "image",
        "data_type": str(hdr.dtype.name) if hasattr(hdr.dtype, "name") else str(hdr.dtype),
        "num_channels": 1,
        "scales": scale_entries,
    }


def chunk_name(x0: int, x1: int, y0: int, y1: int, z0: int, z1: int) -> str:
    return f"{x0}-{x1}_{y0}-{y1}_{z0}-{z1}"


def parse_chunk_name(name: str) -> tuple[int, int, int, int, int, int]:
    m = _CHUNK_RE.match(name)
    if not m:
        raise ValueError(f"malformed chunk name: {name!r}")
    x0, x1, y0, y1, z0, z1 = (int(g) for g in m.groups())
    return x0, x1, y0, y1, z0, z1


def clip_chunk_to_scale(
    scale: ScaleLevel, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int,
    chunk_size: tuple[int, int, int] | None = None,
) -> tuple[int, int, int, int, int, int]:
    sx, sy, sz = scale.size

    if chunk_size is not None:
        # Neuroglancer requests the already-clipped extent for edge chunks
        # (e.g. "0-64_0-64_0-40" for a volume with z-size 40), so the
        # requested x1/y1/z1 must equal the grid cell clipped to the volume
        # -- never the unclipped chunk_size width, and never padded.
        gx, gy, gz = chunk_size
        if x0 % gx != 0 or y0 % gy != 0 or z0 % gz != 0:
            raise ValueError(f"chunk origin not grid-aligned to {chunk_size}: {(x0, y0, z0)}")
        expected = (min(x0 + gx, sx), min(y0 + gy, sy), min(z0 + gz, sz))
        if (x1, y1, z1) != expected:
            raise ValueError(
                f"chunk extent {(x0, x1, y0, y1, z0, z1)} does not match the grid-clipped "
                f"extent {(x0, expected[0], y0, expected[1], z0, expected[2])} for scale size {scale.size}"
            )
        return x0, x1, y0, y1, z0, z1

    cx1, cy1, cz1 = min(x1, sx), min(y1, sy), min(z1, sz)
    if cx1 <= x0 or cy1 <= y0 or cz1 <= z0:
        raise ValueError(f"chunk request entirely out of bounds for scale size {scale.size}")
    return x0, cx1, y0, cy1, z0, cz1


def encode_chunk(arr: np.ndarray) -> bytes:
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError("array must be C-contiguous for precomputed raw encoding")
    if arr.dtype.byteorder not in ("<", "="):
        raise ValueError(f"array must be little-endian, got byteorder {arr.dtype.byteorder!r}")
    return arr.tobytes()
