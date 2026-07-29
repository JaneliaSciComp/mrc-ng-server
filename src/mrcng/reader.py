"""pread-based chunk extraction. Every read of an MRC file goes through
pread_exact; never call os.pread directly anywhere else in this codebase."""
from __future__ import annotations

import enum
import os

import numpy as np


class UnexpectedEOF(Exception):
    pass


class ChunkOutOfBounds(Exception):
    pass


class ReadStrategy(enum.Enum):
    ROW_WISE = "row_wise"
    SPAN_WISE = "span_wise"


def pread_exact(fd: int, count: int, offset: int) -> bytes:
    chunks = []
    got = 0
    while got < count:
        chunk = os.pread(fd, count - got, offset + got)
        if not chunk:
            raise UnexpectedEOF(
                f"unexpected EOF: got {got} of {count} bytes at offset {offset}"
            )
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def choose_strategy(x0: int, x1: int, itemsize: int, threshold: int) -> ReadStrategy:
    row_bytes = (x1 - x0) * itemsize
    return ReadStrategy.ROW_WISE if row_bytes >= threshold else ReadStrategy.SPAN_WISE


def read_chunk(
    fd: int, hdr, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int,
    row_bytes_threshold: int = 4096,
) -> np.ndarray:
    x0 = max(x0, 0); y0 = max(y0, 0); z0 = max(z0, 0)
    x1 = min(x1, hdr.nx); y1 = min(y1, hdr.ny); z1 = min(z1, hdr.nz)

    if x1 <= x0 or y1 <= y0 or z1 <= z0:
        raise ChunkOutOfBounds(f"empty clipped region: x[{x0}:{x1}] y[{y0}:{y1}] z[{z0}:{z1}]")

    itemsize = hdr.dtype.itemsize
    strategy = choose_strategy(x0, x1, itemsize, row_bytes_threshold)
    out = np.empty((z1 - z0, y1 - y0, x1 - x0), dtype=hdr.dtype)

    if strategy is ReadStrategy.ROW_WISE:
        row_len = x1 - x0
        for zi, z in enumerate(range(z0, z1)):
            for yi, y in enumerate(range(y0, y1)):
                offset = hdr.data_offset + (z * hdr.ny * hdr.nx + y * hdr.nx + x0) * itemsize
                raw = pread_exact(fd, row_len * itemsize, offset)
                out[zi, yi, :] = np.frombuffer(raw, dtype=hdr.dtype, count=row_len)
    else:
        span_len = x1  # from column 0 through x1, then slice out [x0:x1]
        for zi, z in enumerate(range(z0, z1)):
            for yi, y in enumerate(range(y0, y1)):
                offset = hdr.data_offset + (z * hdr.ny * hdr.nx + y * hdr.nx) * itemsize
                raw = pread_exact(fd, span_len * itemsize, offset)
                full_row = np.frombuffer(raw, dtype=hdr.dtype, count=span_len)
                out[zi, yi, :] = full_row[x0:x1]

    return out
