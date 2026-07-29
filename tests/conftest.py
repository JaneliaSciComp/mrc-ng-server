import struct

import numpy as np
import pytest

HEADER_SIZE = 1024

MODE_DTYPE = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16, 12: np.float16}


def make_mrc(
    path,
    shape,  # (nx, ny, nz)
    mode=1,
    voxel_size_angstrom=(1.0, 1.0, 1.0),
    nsymbt=0,
    mapc=1, mapr=2, maps=3,
    imod_flags=None,  # None, "signed", or "unsigned" -- only meaningful for mode 0
    fill=None,  # callable(zz, yy, xx) -> value, or None for zeros
    truncate_bytes=0,
    nstart=(0, 0, 0),  # nxstart, nystart, nzstart
    grid_size=None,  # (mx, my, mz); defaults to shape when None
):
    nx, ny, nz = shape
    dtype = MODE_DTYPE[mode]
    mx, my, mz = grid_size if grid_size is not None else (nx, ny, nz)
    cella = tuple(v * m for v, m in zip(voxel_size_angstrom, (mx, my, mz)))

    header = bytearray(HEADER_SIZE)
    struct.pack_into("<3i", header, 0, nx, ny, nz)
    struct.pack_into("<i", header, 12, mode)
    struct.pack_into("<3i", header, 16, *nstart)
    struct.pack_into("<3i", header, 28, mx, my, mz)
    struct.pack_into("<3f", header, 40, *cella)
    struct.pack_into("<3f", header, 52, 90.0, 90.0, 90.0)  # cellb
    struct.pack_into("<3i", header, 64, mapc, mapr, maps)
    struct.pack_into("<i", header, 92, nsymbt)
    struct.pack_into("<4s", header, 104, b"    ")  # exttyp
    struct.pack_into("<i", header, 108, 20140)  # nversion
    if imod_flags is not None:
        struct.pack_into("<i", header, 152, 1146047817)  # imodStamp
        flags = 1 if imod_flags == "signed" else 0
        struct.pack_into("<i", header, 156, flags)
    struct.pack_into("<4s", header, 208, b"MAP ")
    struct.pack_into("<4B", header, 212, 0x44, 0x41, 0x00, 0x00)  # little-endian machst

    ext = b"\x00" * nsymbt
    if fill is None:
        data = np.zeros((nz, ny, nx), dtype=dtype)
    else:
        zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
        data = fill(zz, yy, xx).astype(dtype)

    with open(path, "wb") as f:
        f.write(bytes(header))
        f.write(ext)
        f.write(data.tobytes())
        if truncate_bytes:
            f.truncate(f.tell() - truncate_bytes)

    return path


@pytest.fixture
def make_mrc_file(tmp_path):
    def _make(name="test.mrc", **kwargs):
        return make_mrc(tmp_path / name, **kwargs)
    return _make
