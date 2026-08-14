"""MRC2014 header parsing.

Data on disk is C-order with x fastest, y next, z slowest:
offset(x, y, z) = data_offset + (z*ny*nx + y*nx + x) * itemsize
"""
from __future__ import annotations

import fnmatch
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
# float16 is a real MRC mode (AreTomo/Warp write half-precision tomograms to
# halve file size) but Neuroglancer has no float16 at all -- not in the
# precomputed data_type list, and no FLOAT16 in its DataType enum -- so it is
# widened to float32 on the way out. See MrcHeader.served_dtype.
_WIDEN_ON_SERVE = {np.dtype("<f2"): np.dtype("<f4")}

# An image stack (tilt series, gain reference, montage map) has a z axis that is
# a slice index, not a spatial sampling -- so cella_z carries no physical meaning
# and adjacent "slices" must never be averaged together.
#
# Nothing in the MRC header can tell you which kind of file this is. ispg is 0 on
# every file in the Janelia cryoET corpus, tomograms included, and writers agree
# on nothing else either: Relion stamps mz=nz with cella_z = nz * pixel_x, IMOD
# stamps mz=1 with a dummy cella_z, and only some stacks carry an extended header.
# Shape does not separate them either -- measured over 3648 corpus files, true 2D
# files span nz/max(nx,ny) 0.0001-0.2200 and true 3D files 0.1276-1.4120, so the
# classes overlap and no threshold can be correct. An earlier aspect-ratio
# heuristic lived here and was knowingly wrong on 2 of those files.
#
# So the classification is an *input*, not something derived: callers match the
# file's path against operator-supplied globs (classify_path below) and pass the
# answer to parse_header. The build records its answer in the fingerprint, so a
# later glob change invalidates the entries it would reclassify.
STACK_Z_VOXEL_ANGSTROM = 10.0

_UNSUPPORTED_MODE_REASONS = {
    3: "complex data",
    4: "complex data",
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


def classify_path(relpath: str, stack_globs, volume_globs=()) -> bool:
    """Whether `relpath` names an image stack, per operator-supplied globs.

    fnmatch patterns, matched against the whole path, so `*` crosses `/` and
    `*/TiltSeries/*` means "anywhere under a TiltSeries directory".

    volume_globs win over stack_globs, because real trees put both kinds in one
    directory: the Janelia corpus has `.../external/s200.mrc` (a 55-tilt stack)
    beside `.../external/s200_ctf.mrc` (a 512x512x55 volume). Broad stack
    directories plus narrow volume exclusions classify all 3648 corpus files
    correctly; an include-only list cannot without per-filename patterns.

    No globs configured means nothing is a stack -- z comes from cella as it
    always did. That is the safe default: it is the pre-existing behaviour, and
    getting it wrong understates a tilt series' z extent rather than silently
    averaging tilts together.
    """
    if any(fnmatch.fnmatch(relpath, g) for g in volume_globs):
        return False
    return any(fnmatch.fnmatch(relpath, g) for g in stack_globs)


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
    is_image_stack: bool
    mode0_signedness_is_ambiguous: bool
    dtype: np.dtype
    data_offset: int
    file_size: int
    mtime_ns: int

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return (self.nz, self.ny, self.nx)

    @property
    def served_dtype(self) -> np.dtype:
        """The dtype Neuroglancer is told about and chunk bodies are encoded in.

        Differs from `dtype` (the on-disk layout, which every byte offset and
        itemsize calculation must keep using) only for mode 12: float16 widens
        to float32. The widening is exact -- every float16 is representable in
        float32 -- and costs 2x on the wire, which is the only way to serve
        these files at all since Neuroglancer cannot render float16.
        """
        return _WIDEN_ON_SERVE.get(self.dtype, self.dtype)


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


def parse_header(fd: int, file_size: int, mtime_ns: int, assume_mode0: str | None = None,
                  is_image_stack: bool = False) -> MrcHeader:
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

    is_stack = is_image_stack

    voxel_size_is_default = False
    if mx == 0 or my == 0 or mz == 0 or all(c == 0.0 for c in cella):
        voxel_size = (1.0, 1.0, 1.0)
        voxel_size_is_default = True
    else:
        if not is_stack and (mx, my, mz) != (nx, ny, nz):
            # Fail closed (sec 0) rather than silently computing a bogus
            # per-axis voxel size: a writer that sets mz=1 while cella_z spans
            # the whole depth would otherwise divide that depth into a single
            # voxel and make the z scale bar nz times too large.
            #
            # Image stacks are exempt, and are the reason this used to reject
            # real files: mz=1-regardless-of-nz is the *convention* there, not
            # a defect (110 of the 2871 image stacks in the deployment corpus
            # write it, and were refused outright). Their z voxel size does
            # not come from cella_z at all -- see below -- so a mismatched mz
            # cannot poison it.
            raise NonStandardGridSizeError(
                f"grid size (mx,my,mz)=({mx},{my},{mz}) does not match sample "
                f"count (nx,ny,nz)=({nx},{ny},{nz}); per-axis voxel size would "
                f"be unreliable"
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
        # Only x/y are read from cella for a stack, so a zero or dummy cella_z
        # there is expected rather than a sign the voxel size was guessed.
        voxel_size_is_default = any(c == 0.0 for c in (cella[:2] if is_stack else cella))

    if is_stack:
        # Whatever the writer stamped on cella_z describes nothing: Relion
        # copies the x/y pixel size onto the tilt axis (making 55 tilts 8.3nm
        # "thick" against a 621nm-wide image, a 75x squash that leaves z
        # unnavigable), IMOD writes a dummy 1.0. One unit per slice instead.
        voxel_size = (voxel_size[0], voxel_size[1], STACK_Z_VOXEL_ANGSTROM)

    return MrcHeader(
        nx=nx, ny=ny, nz=nz, mode=mode,
        nxstart=nxstart, nystart=nystart, nzstart=nzstart,
        mx=mx, my=my, mz=mz, nsymbt=nsymbt,
        mapc=mapc, mapr=mapr, maps=maps,
        voxel_size_angstrom=voxel_size,
        voxel_size_is_default=voxel_size_is_default,
        is_image_stack=is_stack,
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
