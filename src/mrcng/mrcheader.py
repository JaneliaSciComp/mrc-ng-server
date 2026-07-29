"""MRC2014 header parsing.

Data on disk is C-order with x fastest, y next, z slowest:
offset(x, y, z) = data_offset + (z*ny*nx + y*nx + x) * itemsize
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import numpy as np

HEADER_SIZE = 1024
IMOD_STAMP = 1146047817  # ASCII "IMOD" as little-endian int32, at byte 152
IMOD_SIGNED_BIT = 0x1

_MODE_DTYPES = {
    1: np.dtype("<i2"),
    2: np.dtype("<f4"),
    6: np.dtype("<u2"),
    12: np.dtype("<f2"),
}
_UNSUPPORTED_MODES = {3, 4}
_VALID_MODES = {0, 1, 2, 3, 4, 6, 12}


class MrcFormatError(Exception):
    pass


class UnsupportedModeError(MrcFormatError):
    pass


class UnsupportedByteOrderError(MrcFormatError):
    pass


class NonStandardAxisOrderError(MrcFormatError):
    pass


class TruncatedFileError(MrcFormatError):
    pass


@dataclass(frozen=True)
class MrcHeader:
    nx: int
    ny: int
    nz: int
    mode: int
    mx: int
    my: int
    mz: int
    nsymbt: int
    mapc: int
    mapr: int
    maps: int
    voxel_size_angstrom: tuple[float, float, float]
    voxel_size_is_default: bool
    dtype: np.dtype
    data_offset: int
    file_size: int
    mtime_ns: int

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return (self.nz, self.ny, self.nx)


def _dtype_for_mode(mode: int, raw: bytes, assume_mode0: str | None) -> np.dtype:
    if mode in _UNSUPPORTED_MODES:
        raise UnsupportedModeError(f"mode {mode} is unsupported (complex data)")
    if mode != 0:
        if mode not in _MODE_DTYPES:
            raise UnsupportedModeError(f"mode {mode} is not a recognised MRC mode")
        return _MODE_DTYPES[mode]

    if assume_mode0 is not None:
        return np.dtype(np.int8 if assume_mode0 == "int8" else np.uint8)

    imod_stamp = int.from_bytes(raw[152:156], "little", signed=True)
    if imod_stamp == IMOD_STAMP:
        imod_flags = int.from_bytes(raw[156:160], "little", signed=True)
        signed = bool(imod_flags & IMOD_SIGNED_BIT)
        return np.dtype(np.int8 if signed else np.uint8)

    return np.dtype(np.int8)  # default; agrees with mrcfile's own default


def parse_header(fd: int, file_size: int, mtime_ns: int, assume_mode0: str | None = None) -> MrcHeader:
    raw = _pread_header(fd)

    nx, ny, nz, mode = struct.unpack_from("<4i", raw, 0)
    if mode not in _VALID_MODES:
        _, _, _, be_mode = struct.unpack_from(">4i", raw, 0)
        if be_mode not in _VALID_MODES:
            raise UnsupportedByteOrderError(
                f"mode field ({mode}) is not a recognised value in either byte order"
            )
        raise UnsupportedByteOrderError("file appears to be big-endian; not supported in v1")

    if nx <= 0 or ny <= 0 or nz <= 0:
        raise MrcFormatError(f"non-positive dimensions: nx={nx}, ny={ny}, nz={nz}")

    mx, my, mz = struct.unpack_from("<3i", raw, 28)
    cella = struct.unpack_from("<3f", raw, 40)
    mapc, mapr, maps = struct.unpack_from("<3i", raw, 64)
    (nsymbt,) = struct.unpack_from("<i", raw, 92)

    if (mapc, mapr, maps) != (1, 2, 3):
        raise NonStandardAxisOrderError(f"mapc,mapr,maps = {(mapc, mapr, maps)}, expected (1, 2, 3)")

    if nsymbt < 0:
        raise MrcFormatError(f"negative nsymbt: {nsymbt}")

    dtype = _dtype_for_mode(mode, raw, assume_mode0)
    data_offset = HEADER_SIZE + nsymbt
    required = data_offset + nx * ny * nz * dtype.itemsize
    if required > file_size:
        raise TruncatedFileError(
            f"file is {file_size} bytes but header implies at least {required} bytes"
        )

    voxel_size_is_default = False
    if mx == 0 or my == 0 or mz == 0 or all(c == 0.0 for c in cella):
        voxel_size = (1.0, 1.0, 1.0)
        voxel_size_is_default = True
    else:
        voxel_size = (cella[0] / mx, cella[1] / my, cella[2] / mz)

    return MrcHeader(
        nx=nx, ny=ny, nz=nz, mode=mode,
        mx=mx, my=my, mz=mz, nsymbt=nsymbt,
        mapc=mapc, mapr=mapr, maps=maps,
        voxel_size_angstrom=voxel_size,
        voxel_size_is_default=voxel_size_is_default,
        dtype=dtype, data_offset=data_offset,
        file_size=file_size, mtime_ns=mtime_ns,
    )


def _pread_header(fd: int) -> bytes:
    chunks = []
    got = 0
    while got < HEADER_SIZE:
        chunk = os.pread(fd, HEADER_SIZE - got, got)
        if not chunk:
            raise TruncatedFileError("file shorter than the 1024-byte MRC header")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)
