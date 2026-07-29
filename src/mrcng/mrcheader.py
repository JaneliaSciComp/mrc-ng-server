"""MRC2014 header parsing.

Data on disk is C-order with x fastest, y next, z slowest:
offset(x, y, z) = data_offset + (z*ny*nx + y*nx + x) * itemsize
"""
from __future__ import annotations

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
}
# mode 12 (float16) is a recognised MRC mode -- kept in _VALID_MODES so the
# byte-order sanity check still treats it as a plausible little-endian mode
# field -- but it isn't in _MODE_DTYPES: the Neuroglancer precomputed protocol
# only allows uint8/int8/uint16/int16/uint32/int32/uint64/float32 as
# data_type, so serving float16 verbatim would produce an info file
# Neuroglancer can't render. Reject at open rather than guess a conversion.
_UNSUPPORTED_MODE_REASONS = {
    3: "complex data",
    4: "complex data",
    12: "float16 is not an allowed Neuroglancer precomputed data_type",
}
_UNSUPPORTED_MODES = set(_UNSUPPORTED_MODE_REASONS)
_VALID_MODES = {0, 1, 2, 3, 4, 6, 12}


class MrcFormatError(Exception):
    pass


class UnsupportedModeError(MrcFormatError):
    pass


class UnsupportedByteOrderError(MrcFormatError):
    pass


class NonStandardAxisOrderError(MrcFormatError):
    pass


class NonStandardGridSizeError(MrcFormatError):
    pass


class TruncatedFileError(MrcFormatError):
    pass


@dataclass(frozen=True)
class MrcHeader:
    nx: int
    ny: int
    nz: int
    mode: int
    nxstart: int
    nystart: int
    nzstart: int
    mx: int
    my: int
    mz: int
    nsymbt: int
    mapc: int
    mapr: int
    maps: int
    voxel_size_angstrom: tuple[float, float, float]
    voxel_size_is_default: bool
    mode0_signedness_is_ambiguous: bool
    dtype: np.dtype
    data_offset: int
    file_size: int
    mtime_ns: int

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return (self.nz, self.ny, self.nx)


def _dtype_for_mode(mode: int, raw: bytes, assume_mode0: str | None) -> tuple[np.dtype, bool]:
    """Returns (dtype, mode0_signedness_is_ambiguous)."""
    if mode in _UNSUPPORTED_MODES:
        raise UnsupportedModeError(f"mode {mode} is unsupported ({_UNSUPPORTED_MODE_REASONS[mode]})")
    if mode != 0:
        if mode not in _MODE_DTYPES:
            raise UnsupportedModeError(f"mode {mode} is not a recognised MRC mode")
        return _MODE_DTYPES[mode], False

    if assume_mode0 is not None:
        return np.dtype(np.int8 if assume_mode0 == "int8" else np.uint8), False

    imod_stamp = int.from_bytes(raw[152:156], "little", signed=True)
    if imod_stamp == IMOD_STAMP:
        imod_flags = int.from_bytes(raw[156:160], "little", signed=True)
        signed = bool(imod_flags & IMOD_SIGNED_BIT)
        return np.dtype(np.int8 if signed else np.uint8), False

    # No IMOD stamp and no override: signedness is genuinely ambiguous (sec 2).
    # Default to int8 (agrees with mrcfile's own default) but flag it so
    # callers -- which have the file path, unlike this function -- can warn.
    return np.dtype(np.int8), True


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

    nxstart, nystart, nzstart = struct.unpack_from("<3i", raw, 16)
    mx, my, mz = struct.unpack_from("<3i", raw, 28)
    cella = struct.unpack_from("<3f", raw, 40)
    mapc, mapr, maps = struct.unpack_from("<3i", raw, 64)
    (nsymbt,) = struct.unpack_from("<i", raw, 92)

    if (mapc, mapr, maps) != (1, 2, 3):
        raise NonStandardAxisOrderError(f"mapc,mapr,maps = {(mapc, mapr, maps)}, expected (1, 2, 3)")

    if nsymbt < 0:
        raise MrcFormatError(f"negative nsymbt: {nsymbt}")

    dtype, mode0_signedness_is_ambiguous = _dtype_for_mode(mode, raw, assume_mode0)
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
        if (mx, my, mz) != (nx, ny, nz):
            # Fail closed (sec 0) rather than silently computing a bogus
            # per-axis voxel size. This is a real-world case, not a
            # hypothetical: image-stack MRC files (each "z" an independent 2D
            # image, not a 3D volume) conventionally set mz=1 regardless of
            # nz, which would otherwise divide the whole cell depth into a
            # single voxel and make the z scale bar nz times too large.
            raise NonStandardGridSizeError(
                f"grid size (mx,my,mz)=({mx},{my},{mz}) does not match sample "
                f"count (nx,ny,nz)=({nx},{ny},{nz}); per-axis voxel size would "
                f"be unreliable (common in image-stack MRC files, e.g. mz=1)"
            )
        # A single axis's cella can be zero while the others are populated --
        # common for per-section tilt-series stacks, where z isn't a real
        # physical sampling. Left uncaught, cella[i]/m == 0.0 for that axis,
        # and Neuroglancer's precomputed format rejects a zero resolution
        # outright ("Expected positive finite float"). Default just that axis
        # to 1 Angstrom, same as the all-zero case, and flag it the same way.
        voxel_size = tuple(
            (c / m) if c != 0.0 else 1.0
            for c, m in zip(cella, (mx, my, mz))
        )
        voxel_size_is_default = any(c == 0.0 for c in cella)

    return MrcHeader(
        nx=nx, ny=ny, nz=nz, mode=mode,
        nxstart=nxstart, nystart=nystart, nzstart=nzstart,
        mx=mx, my=my, mz=mz, nsymbt=nsymbt,
        mapc=mapc, mapr=mapr, maps=maps,
        voxel_size_angstrom=voxel_size,
        voxel_size_is_default=voxel_size_is_default,
        mode0_signedness_is_ambiguous=mode0_signedness_is_ambiguous,
        dtype=dtype, data_offset=data_offset,
        file_size=file_size, mtime_ns=mtime_ns,
    )


def _pread_header(fd: int) -> bytes:
    from mrcng.reader import pread_exact, UnexpectedEOF
    try:
        return pread_exact(fd, HEADER_SIZE, 0)
    except UnexpectedEOF as e:
        raise TruncatedFileError("file shorter than the 1024-byte MRC header") from e
