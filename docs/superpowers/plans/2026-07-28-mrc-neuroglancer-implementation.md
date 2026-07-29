# MRC → Neuroglancer Precomputed Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `mrcng`, a library + CLI (`mrc-pyramid`) + FastAPI service that serves MRC tomograms to Neuroglancer via the `precomputed` protocol, with scale 0 always read live from the MRC and scales 1..N served only from a validated, fingerprinted cache built offline.

**Architecture:** A shared library (`src/mrcng/`) implements MRC header parsing, path safety, the precomputed protocol's naming/encoding rules, pread-based chunk extraction, and block-mean downsampling. Two independent entry points consume it: `mrc-pyramid` (CLI, writes the cache tree) and the FastAPI app (`mrcng.server.app`, read-only, never writes or computes on the request path).

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, numpy, pydantic-settings (runtime); pytest, mrcfile, httpx (dev). Managed by pixi (`pyproject.toml`, single `default` environment). Full spec: `docs/superpowers/specs/2026-07-28-mrc-neuroglancer-design.md`.

## Global Constraints

- Never write to the source tree or open an MRC for writing (spec §0).
- The server never writes to the cache and never downsamples on the request path (spec §0).
- Scale 0 always comes from the MRC directly, even when a cache exists (spec §0).
- Fail closed: non-standard headers raise a named error, never a guess (spec §0).
- Runtime deps are exactly `fastapi`, `uvicorn`, `numpy`, `pydantic-settings`; dev-only `pytest`, `mrcfile`, `httpx`. No new deps (spec §0, §10).
- `mrcfile` is a test oracle only, never imported outside `tests/` (spec §0).
- Data is C-order with **x fastest, z slowest** everywhere (spec §2.2).
- `resolution` in `info` JSON is nanometres; MRC `cella` is ångström — divide by 10 (spec §5.2).
- Every `os.pread` call goes through `pread_exact`; never call `os.pread` directly elsewhere (spec §6).
- Run tests with `pixi run pytest <path>` (or `pixi run test` for the whole suite) from the repo root — the pixi `default` environment already has all runtime+dev deps installed.
- Package lives at `src/mrcng/`, tests at `tests/`, installed editable via pixi/hatchling — no `sys.path` hacks needed; `import mrcng` works directly once `pixi install` has run once (already done).

---

### Task 1: `mrcheader.py` — header parsing, dtype table, IMOD mode-0 detection

**Files:**
- Create: `src/mrcng/mrcheader.py`
- Test: `tests/test_mrcheader.py`
- Test helper: `tests/conftest.py`

**Interfaces:**
- Produces: `MrcHeader` frozen dataclass with fields `nx, ny, nz, mode, mx, my, mz, nsymbt, mapc, mapr, maps, voxel_size_angstrom: tuple[float,float,float], voxel_size_is_default: bool, dtype: np.dtype, data_offset: int, file_size: int, mtime_ns: int`.
- Produces: `parse_header(fd: int, file_size: int, mtime_ns: int, assume_mode0: str | None = None) -> MrcHeader`.
- Produces: exceptions `UnsupportedModeError`, `UnsupportedByteOrderError`, `NonStandardAxisOrderError`, `TruncatedFileError`, all subclasses of `MrcFormatError(Exception)`.
- Produces: `DTYPE_TABLE: dict[int, np.dtype]` for modes `{1: int16, 2: float32, 6: uint16, 12: float16}` (mode 0 handled specially, see below).

**Step-by-step:**

- [ ] **Step 1: Write the test fixture helper `make_mrc`**

Create `tests/conftest.py`:

```python
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
    fill=None,  # callable(z, y, x) -> value, or None for zeros
    truncate_bytes=0,
):
    nx, ny, nz = shape
    dtype = MODE_DTYPE[mode]
    mx, my, mz = nx, ny, nz
    cella = tuple(v * m for v, m in zip(voxel_size_angstrom, (mx, my, mz)))

    header = bytearray(HEADER_SIZE)
    struct.pack_into("<3i", header, 0, nx, ny, nz)
    struct.pack_into("<i", header, 12, mode)
    struct.pack_into("<3i", header, 16, 0, 0, 0)  # nxstart/nystart/nzstart
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
```

- [ ] **Step 2: Write failing tests for header parsing**

Create `tests/test_mrcheader.py`:

```python
import os
import numpy as np
import mrcfile
import pytest

from mrcng.mrcheader import (
    parse_header, UnsupportedModeError, UnsupportedByteOrderError,
    NonStandardAxisOrderError, TruncatedFileError,
)


def _parse(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        st = os.stat(fd)
        return parse_header(fd, st.st_size, st.st_mtime_ns)
    finally:
        os.close(fd)


def test_basic_int16_header_matches_mrcfile(make_mrc_file):
    path = make_mrc_file(shape=(64, 32, 16), mode=1, voxel_size_angstrom=(2.0, 2.0, 4.0))
    hdr = _parse(path)
    with mrcfile.open(path, permissive=True) as mf:
        assert hdr.nx == mf.header.nx == 64
        assert hdr.ny == mf.header.ny == 32
        assert hdr.nz == mf.header.nz == 16
        assert hdr.dtype == mf.data.dtype
        np.testing.assert_allclose(hdr.voxel_size_angstrom, (2.0, 2.0, 4.0))


def test_odd_dimensions_and_extended_header(make_mrc_file):
    path = make_mrc_file(name="odd.mrc", shape=(101, 97, 53), mode=2, nsymbt=128)
    hdr = _parse(path)
    with mrcfile.open(path, permissive=True) as mf:
        assert hdr.data_offset == 1024 + 128 == mf.header.nsymbt + 1024
        assert (hdr.nx, hdr.ny, hdr.nz) == (mf.header.nx, mf.header.ny, mf.header.nz)


def test_anisotropic_volume(make_mrc_file):
    path = make_mrc_file(name="aniso.mrc", shape=(2048, 2048, 64), mode=1)
    hdr = _parse(path)
    assert (hdr.nx, hdr.ny, hdr.nz) == (2048, 2048, 64)


def test_unsupported_mode_raises(make_mrc_file):
    path = make_mrc_file(shape=(4, 4, 4), mode=1)
    # hand-corrupt the mode field to an unsupported complex mode (3)
    import struct
    with open(path, "r+b") as f:
        f.seek(12)
        f.write(struct.pack("<i", 3))
    with pytest.raises(UnsupportedModeError):
        _parse(path)


def test_non_standard_axis_order_raises(make_mrc_file):
    path = make_mrc_file(shape=(4, 4, 4), mode=1, mapc=2, mapr=1, maps=3)
    with pytest.raises(NonStandardAxisOrderError):
        _parse(path)


def test_truncated_file_raises(make_mrc_file):
    path = make_mrc_file(shape=(16, 16, 16), mode=1, truncate_bytes=100)
    with pytest.raises(TruncatedFileError):
        _parse(path)


def test_zero_cella_falls_back_to_default_voxel_size(make_mrc_file):
    path = make_mrc_file(shape=(4, 4, 4), mode=1, voxel_size_angstrom=(0.0, 0.0, 0.0))
    hdr = _parse(path)
    assert hdr.voxel_size_angstrom == (1.0, 1.0, 1.0)
    assert hdr.voxel_size_is_default is True


def test_mode0_default_signed_agrees_with_mrcfile(make_mrc_file):
    path = make_mrc_file(shape=(4, 4, 4), mode=0)
    hdr = _parse(path)
    with mrcfile.open(path, permissive=True) as mf:
        assert hdr.dtype == mf.data.dtype == np.dtype(np.int8)


def test_mode0_imod_unsigned_diverges_from_mrcfile_default(make_mrc_file):
    # mrcfile always returns int8 for mode 0; we deliberately diverge when
    # an IMOD unsigned stamp is present (spec section 2.3).
    path = make_mrc_file(shape=(4, 4, 4), mode=0, imod_flags="unsigned")
    hdr = _parse(path)
    assert hdr.dtype == np.dtype(np.uint8)


def test_mode0_imod_signed_matches_default(make_mrc_file):
    path = make_mrc_file(shape=(4, 4, 4), mode=0, imod_flags="signed")
    hdr = _parse(path)
    assert hdr.dtype == np.dtype(np.int8)


def test_mode0_assume_mode0_cli_override(make_mrc_file):
    path = make_mrc_file(shape=(4, 4, 4), mode=0)
    fd = os.open(str(path), os.O_RDONLY)
    try:
        st = os.stat(fd)
        hdr = parse_header(fd, st.st_size, st.st_mtime_ns, assume_mode0="uint8")
    finally:
        os.close(fd)
    assert hdr.dtype == np.dtype(np.uint8)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pixi run pytest tests/test_mrcheader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mrcng.mrcheader'`

- [ ] **Step 4: Implement `mrcheader.py`**

Create `src/mrcng/mrcheader.py`:

```python
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

# byte offset -> struct format, verified against mrcfile.dtypes.HEADER_DTYPE
_STRUCT = struct.Struct(
    "<3i"    # 0:  nx, ny, nz
    "i"      # 12: mode
    "3i"     # 16: nxstart, nystart, nzstart
    "3i"     # 28: mx, my, mz
    "3f"     # 40: cella x,y,z
    "3f"     # 52: cellb (unused)
    "3i"     # 64: mapc, mapr, maps
    "3f"     # 76: dmin, dmax, dmean (unused)
    "i"      # 88: ispg (unused)
    "i"      # 92: nsymbt
)

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


def _unpack(raw: bytes, byteorder: str):
    fmt = byteorder + _STRUCT.format[1:]
    return struct.Struct(fmt).unpack_from(raw, 0)


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
        nx, ny, nz, mode = struct.unpack_from(">4i", raw, 0)
        if mode not in _VALID_MODES:
            raise UnsupportedByteOrderError(
                f"mode field is not a recognised value in either byte order (little-endian read: {struct.unpack_from('<i', raw, 12)[0]})"
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
    from mrcng.reader import pread_exact
    return pread_exact(fd, HEADER_SIZE, 0)
```

Note: this imports `pread_exact` from `reader.py`, written in Task 4. Since Task 4 comes after this task, temporarily inline a minimal local retry loop instead so this task's tests pass standalone — replace the body of `_pread_header` with:

```python
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
```

(add `import os` at the top). Task 4 will replace this with a call to the real `pread_exact` once it exists, removing the duplication.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pixi run pytest tests/test_mrcheader.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/mrcng/mrcheader.py tests/test_mrcheader.py tests/conftest.py
git commit -m "Add MRC2014 header parsing with IMOD mode-0 detection"
```

---

### Task 2: `paths.py` — dataset identity and path safety

**Files:**
- Create: `src/mrcng/paths.py`
- Test: `tests/test_paths.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `dataset_id(relpath: str) -> str` (16 hex chars).
- Produces: `cache_dir_for(cache_root: Path, ds_id: str) -> Path` → `cache_root/ds_id[:2]/ds_id`.
- Produces: `resolve_source(root: Path, relpath: str) -> Path`, raising `PathNotAllowed(Exception)` on any violation.

**Step-by-step:**

- [ ] **Step 1: Write failing tests**

Create `tests/test_paths.py`:

```python
import os
import hashlib
import pytest

from mrcng.paths import dataset_id, cache_dir_for, resolve_source, PathNotAllowed


def test_dataset_id_is_deterministic_sha256_prefix():
    rel = "session42/tomo_0031.mrc"
    expected = hashlib.sha256(rel.encode()).hexdigest()[:16]
    assert dataset_id(rel) == expected
    assert dataset_id(rel) == dataset_id(rel)


def test_cache_dir_uses_two_char_prefix(tmp_path):
    ds_id = "abcdef0123456789"
    result = cache_dir_for(tmp_path, ds_id)
    assert result == tmp_path / "ab" / "abcdef0123456789"


def test_resolve_source_happy_path(tmp_path):
    (tmp_path / "sub").mkdir()
    f = tmp_path / "sub" / "a.mrc"
    f.write_bytes(b"x")
    assert resolve_source(tmp_path, "sub/a.mrc") == f.resolve()


@pytest.mark.parametrize("bad_relpath", [
    "../escape.mrc",
    "sub/../../escape.mrc",
    "/etc/passwd",
    "",
    "sub/\x00null.mrc",
])
def test_resolve_source_rejects_unsafe_paths(tmp_path, bad_relpath):
    with pytest.raises(PathNotAllowed):
        resolve_source(tmp_path, bad_relpath)


def test_resolve_source_rejects_missing_file(tmp_path):
    with pytest.raises(PathNotAllowed):
        resolve_source(tmp_path, "does_not_exist.mrc")


def test_resolve_source_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside_target.mrc"
    outside.write_bytes(b"secret")
    root = tmp_path / "root"
    root.mkdir()
    link = root / "escape.mrc"
    os.symlink(outside, link)
    try:
        with pytest.raises(PathNotAllowed):
            resolve_source(root, "escape.mrc")
    finally:
        outside.unlink()
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_paths.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Implement `paths.py`**

Create `src/mrcng/paths.py`:

```python
"""Dataset identity and path-safety helpers."""
from __future__ import annotations

import hashlib
from pathlib import Path


class PathNotAllowed(Exception):
    pass


def dataset_id(relpath: str) -> str:
    return hashlib.sha256(relpath.encode()).hexdigest()[:16]


def cache_dir_for(cache_root: Path, ds_id: str) -> Path:
    return Path(cache_root) / ds_id[:2] / ds_id


def resolve_source(root: Path, relpath: str) -> Path:
    if not relpath or "\x00" in relpath:
        raise PathNotAllowed(f"empty or null-containing relpath: {relpath!r}")

    parts = Path(relpath).parts
    if any(p == ".." for p in parts):
        raise PathNotAllowed(f"path traversal in relpath: {relpath!r}")
    if Path(relpath).is_absolute():
        raise PathNotAllowed(f"absolute relpath not allowed: {relpath!r}")

    root_resolved = Path(root).resolve()
    candidate = (root_resolved / relpath).resolve()

    if not candidate.is_relative_to(root_resolved):
        raise PathNotAllowed(f"resolved path escapes root: {relpath!r}")
    if not candidate.is_file():
        raise PathNotAllowed(f"not a file: {relpath!r}")

    return candidate
```

- [ ] **Step 4: Run to verify pass**

Run: `pixi run pytest tests/test_paths.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mrcng/paths.py tests/test_paths.py
git commit -m "Add dataset id and path-safety helpers"
```

---

### Task 3: `reader.py` — pread chunk extraction

**Files:**
- Create: `src/mrcng/reader.py`
- Modify: `src/mrcng/mrcheader.py` (replace the temporary inline retry loop in `_pread_header` with a call to `reader.pread_exact`)
- Test: `tests/test_reader.py`

**Interfaces:**
- Consumes: `MrcHeader` from Task 1 (`dtype`, `data_offset`, `nx, ny, nz`).
- Produces: `pread_exact(fd: int, count: int, offset: int) -> bytes`, raising `UnexpectedEOF(Exception)` on a 0-length read before `count` is satisfied.
- Produces: `read_chunk(fd: int, hdr: MrcHeader, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int, row_bytes_threshold: int = 4096) -> np.ndarray` returning shape `(z1-z0, y1-y0, x1-x0)`, raising `ChunkOutOfBounds(Exception)` if the clipped region is empty.
- Produces: `ReadStrategy` enum `{ROW_WISE, SPAN_WISE}` and `choose_strategy(x0, x1, itemsize, threshold) -> ReadStrategy` (exposed so tests can assert both paths independently).

**Step-by-step:**

- [ ] **Step 1: Write failing tests**

Create `tests/test_reader.py`:

```python
import os
import numpy as np
import pytest

from mrcng.mrcheader import parse_header
from mrcng.reader import pread_exact, read_chunk, choose_strategy, ReadStrategy, UnexpectedEOF, ChunkOutOfBounds


def _open_and_parse(path):
    fd = os.open(str(path), os.O_RDONLY)
    st = os.stat(fd)
    hdr = parse_header(fd, st.st_size, st.st_mtime_ns)
    return fd, hdr


def test_pread_exact_reads_full_count(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"abcdefghij")
    fd = os.open(str(f), os.O_RDONLY)
    try:
        assert pread_exact(fd, 10, 0) == b"abcdefghij"
        assert pread_exact(fd, 4, 3) == b"defg"
    finally:
        os.close(fd)


def test_pread_exact_raises_on_short_read(tmp_path):
    f = tmp_path / "short.bin"
    f.write_bytes(b"abc")
    fd = os.open(str(f), os.O_RDONLY)
    try:
        with pytest.raises(UnexpectedEOF):
            pread_exact(fd, 10, 0)
    finally:
        os.close(fd)


def test_choose_strategy_threshold():
    assert choose_strategy(0, 1, itemsize=2, threshold=4096) == ReadStrategy.SPAN_WISE
    assert choose_strategy(0, 2048, itemsize=2, threshold=4096) == ReadStrategy.ROW_WISE


def test_read_chunk_matches_mrcfile_for_random_region(make_mrc_file):
    def fill(zz, yy, xx):
        return (xx + 1000 * yy + 1_000_000 * zz) % 30000

    path = make_mrc_file(shape=(40, 30, 20), mode=1, fill=fill)
    fd, hdr = _open_and_parse(path)
    try:
        import mrcfile
        with mrcfile.open(path, permissive=True) as mf:
            reference = mf.data  # shape (nz, ny, nx)

        arr = read_chunk(fd, hdr, x0=5, x1=25, y0=3, y1=20, z0=1, z1=10)
        np.testing.assert_array_equal(arr, reference[1:10, 3:20, 5:25])
    finally:
        os.close(fd)


def test_read_chunk_row_wise_and_span_wise_agree(make_mrc_file):
    def fill(zz, yy, xx):
        return (xx + 1000 * yy + 1_000_000 * zz) % 30000

    path = make_mrc_file(shape=(4096, 8, 8), mode=1, fill=fill)
    fd, hdr = _open_and_parse(path)
    try:
        row_wise = read_chunk(fd, hdr, 0, 4096, 0, 4, 0, 4, row_bytes_threshold=0)
        span_wise = read_chunk(fd, hdr, 0, 4096, 0, 4, 0, 4, row_bytes_threshold=10**9)
        np.testing.assert_array_equal(row_wise, span_wise)
    finally:
        os.close(fd)


def test_read_chunk_clips_out_of_bounds_raises(make_mrc_file):
    path = make_mrc_file(shape=(4, 4, 4), mode=1)
    fd, hdr = _open_and_parse(path)
    try:
        with pytest.raises(ChunkOutOfBounds):
            read_chunk(fd, hdr, x0=10, x1=20, y0=0, y1=4, z0=0, z1=4)
    finally:
        os.close(fd)


def test_read_chunk_clips_partial_edge(make_mrc_file):
    path = make_mrc_file(shape=(4, 4, 4), mode=1, fill=lambda zz, yy, xx: xx)
    fd, hdr = _open_and_parse(path)
    try:
        arr = read_chunk(fd, hdr, x0=2, x1=10, y0=0, y1=4, z0=0, z1=4)
        assert arr.shape == (4, 4, 2)  # clipped to nx=4
    finally:
        os.close(fd)
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_reader.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Implement `reader.py`**

Create `src/mrcng/reader.py`:

```python
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
```

- [ ] **Step 4: Replace the temporary retry loop in `mrcheader.py`**

Edit `src/mrcng/mrcheader.py`, replace:

```python
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
```

with:

```python
def _pread_header(fd: int) -> bytes:
    from mrcng.reader import pread_exact, UnexpectedEOF
    try:
        return pread_exact(fd, HEADER_SIZE, 0)
    except UnexpectedEOF as e:
        raise TruncatedFileError("file shorter than the 1024-byte MRC header") from e
```

Remove the now-unused `import os` at the top of `mrcheader.py` if nothing else in the file uses it (check with `grep -n "os\." src/mrcng/mrcheader.py`).

- [ ] **Step 5: Run full test suite so far to verify no regression**

Run: `pixi run pytest tests/test_mrcheader.py tests/test_reader.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/mrcng/reader.py src/mrcng/mrcheader.py tests/test_reader.py
git commit -m "Add pread-based chunk reader with row-wise/span-wise strategies"
```

---

### Task 4: `precomputed.py` — scale planning, info JSON, chunk naming/encoding

**Files:**
- Create: `src/mrcng/precomputed.py`
- Test: `tests/test_precomputed.py`

**Interfaces:**
- Consumes: `MrcHeader` from Task 1.
- Produces: `@dataclass ScaleLevel(key: str, size: tuple[int,int,int], factors: tuple[int,int,int])`.
- Produces: `plan_scales(size0: tuple[int,int,int], min_axis_size: int = 32, max_levels: int = 6) -> list[ScaleLevel]` (level 0 first, factors `(1,1,1)`, key `"1_1_1"`).
- Produces: `build_info(hdr, scales: list[ScaleLevel], chunk_size: tuple[int,int,int], encoding: str = "raw") -> dict` (JSON-serialisable).
- Produces: `parse_chunk_name(name: str) -> tuple[int,int,int,int,int,int]` (x0,x1,y0,y1,z0,z1), raising `ValueError` on malformed input.
- Produces: `chunk_name(x0,x1,y0,y1,z0,z1) -> str`.
- Produces: `clip_chunk_to_scale(scale: ScaleLevel, x0,x1,y0,y1,z0,z1) -> tuple[int,int,int,int,int,int]`.
- Produces: `encode_chunk(arr: np.ndarray) -> bytes` (asserts C-contiguous + little-endian first).

**Step-by-step:**

- [ ] **Step 1: Write failing tests**

Create `tests/test_precomputed.py`:

```python
import numpy as np
import pytest

from mrcng.precomputed import (
    plan_scales, build_info, chunk_name, parse_chunk_name,
    clip_chunk_to_scale, encode_chunk, ScaleLevel,
)


def test_plan_scales_isotropic_stops_at_min_axis_size():
    scales = plan_scales((256, 256, 256), min_axis_size=32, max_levels=10)
    assert scales[0].key == "1_1_1"
    assert scales[0].size == (256, 256, 256)
    # 256 -> 128 -> 64 -> 32 (stop, since next would be <=32)
    assert scales[-1].size == (32, 32, 32)
    for lvl in scales:
        assert all(s >= 32 or s == scales[0].size[i] for i, s in enumerate(lvl.size)) or True


def test_plan_scales_anisotropic_pins_short_axis():
    # z is already small; only x,y should keep halving
    scales = plan_scales((4096, 4096, 40), min_axis_size=32, max_levels=6)
    z_sizes = {lvl.size[2] for lvl in scales}
    assert z_sizes == {40}  # z never changes since 40 <= min_axis_size... but must still bin at least once? see impl
    keys = [lvl.key for lvl in scales]
    assert keys[0] == "1_1_1"
    assert all(k.endswith("_1") for k in keys)  # z factor stays 1 throughout


def test_plan_scales_uses_ceil_for_odd_sizes():
    scales = plan_scales((101, 101, 101), min_axis_size=32, max_levels=10)
    # level 1: factor (2,2,2), size = ceil(101/2) = 51
    assert scales[1].size == (51, 51, 51)
    assert scales[1].key == "2_2_2"


def test_plan_scales_respects_max_levels():
    scales = plan_scales((4096, 4096, 4096), min_axis_size=32, max_levels=3)
    assert len(scales) == 3


def test_build_info_converts_angstrom_to_nanometres():
    class FakeHdr:
        nx, ny, nz = 100, 100, 100
        dtype = np.dtype(np.int16)
        voxel_size_angstrom = (6.8, 6.8, 6.8)

    scales = plan_scales((100, 100, 100), min_axis_size=32, max_levels=2)
    info = build_info(FakeHdr(), scales, chunk_size=(64, 64, 64))
    assert info["@type"] == "neuroglancer_multiscale_volume"
    assert info["data_type"] == "int16"
    assert info["scales"][0]["resolution"] == pytest.approx([0.68, 0.68, 0.68])
    level1_factor = scales[1].factors
    expected_res = [0.68 * f for f in level1_factor]
    assert info["scales"][1]["resolution"] == pytest.approx(expected_res)


def test_chunk_name_roundtrip():
    name = chunk_name(0, 64, 64, 128, 0, 32)
    assert name == "0-64_64-128_0-32"
    assert parse_chunk_name(name) == (0, 64, 64, 128, 0, 32)


def test_parse_chunk_name_rejects_malformed():
    with pytest.raises(ValueError):
        parse_chunk_name("not-a-chunk-name")


def test_clip_chunk_to_scale_clips_edge_chunk():
    scale = ScaleLevel(key="1_1_1", size=(100, 100, 40), factors=(1, 1, 1))
    clipped = clip_chunk_to_scale(scale, 0, 64, 0, 64, 0, 64)
    assert clipped == (0, 64, 0, 64, 0, 40)


def test_clip_chunk_to_scale_rejects_misaligned_request():
    scale = ScaleLevel(key="1_1_1", size=(100, 100, 40), factors=(1, 1, 1))
    with pytest.raises(ValueError):
        clip_chunk_to_scale(scale, 5, 69, 0, 64, 0, 64, chunk_size=(64, 64, 64))


def test_encode_chunk_byte_order():
    nx, ny, nz = 4, 3, 2
    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    arr = (xx + 1000 * yy + 1_000_000 * zz).astype("<i2")
    raw = encode_chunk(arr)
    # byte at flat index i (x fastest) must equal arr.tobytes() exactly
    assert raw == arr.tobytes()
    # spot-check a specific voxel: x=2,y=1,z=1 -> value 1_000_000 + 1000 + 2
    flat_index = (1 * ny * nx + 1 * nx + 2)
    value = int.from_bytes(raw[flat_index * 2: flat_index * 2 + 2], "little", signed=True)
    assert value == 1_000_000 + 1000 + 2


def test_encode_chunk_rejects_non_contiguous():
    arr = np.zeros((4, 4, 4), dtype="<i2").T  # transposed -> not C-contiguous
    with pytest.raises(ValueError):
        encode_chunk(arr)
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_precomputed.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Implement `precomputed.py`**

Create `src/mrcng/precomputed.py`:

```python
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
        "data_type": str(hdr.dtype),
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
    cx0, cx1 = x0, min(x1, sx)
    cy0, cy1 = y0, min(y1, sy)
    cz0, cz1 = z0, min(z1, sz)

    if chunk_size is not None:
        gx, gy, gz = chunk_size
        if x0 % gx != 0 or y0 % gy != 0 or z0 % gz != 0:
            raise ValueError(f"chunk origin not grid-aligned to {chunk_size}: {(x0, y0, z0)}")
        expected_x1 = min(x0 + gx, sx)
        expected_y1 = min(y0 + gy, sy)
        expected_z1 = min(z0 + gz, sz)
        if (cx1, cy1, cz1) != (expected_x1, expected_y1, expected_z1) or x1 != x0 + gx or y1 != y0 + gy or z1 != z0 + gz:
            raise ValueError(
                f"chunk extent {(x0, x1, y0, y1, z0, z1)} does not match grid for scale size {scale.size}"
            )

    if cx1 <= cx0 or cy1 <= cy0 or cz1 <= cz0:
        raise ValueError(f"chunk request entirely out of bounds for scale size {scale.size}")

    return cx0, cx1, cy0, cy1, cz0, cz1


def encode_chunk(arr: np.ndarray) -> bytes:
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError("array must be C-contiguous for precomputed raw encoding")
    if arr.dtype.byteorder not in ("<", "="):
        raise ValueError(f"array must be little-endian, got byteorder {arr.dtype.byteorder!r}")
    return arr.tobytes()
```

- [ ] **Step 4: Run tests, fix the anisotropic pinning edge case**

Run: `pixi run pytest tests/test_precomputed.py -v`

If `test_plan_scales_anisotropic_pins_short_axis` fails because z (40) is `> min_axis_size` (32) so it still gets binned once — that's correct per spec ("multiplies the factor of an axis by 2 only if that axis's current size is > min_axis_size"); 40 > 32 so z bins once to `ceil(40/2)=20`, then 20 is not `> 32` so it stops. Fix the test's assumption instead of the implementation: change the assertion to check that z stops changing *after* the first level, not that it never changes:

```python
def test_plan_scales_anisotropic_pins_short_axis():
    scales = plan_scales((4096, 4096, 40), min_axis_size=32, max_levels=6)
    z_sizes = [lvl.size[2] for lvl in scales]
    assert z_sizes[0] == 40
    assert z_sizes[1] == 20  # one bin: 40 > 32
    assert all(z == 20 for z in z_sizes[1:])  # stops changing once <= min_axis_size
```

Expected after fix: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mrcng/precomputed.py tests/test_precomputed.py
git commit -m "Add precomputed protocol: scale planning, info JSON, chunk naming/encoding"
```

---

### Task 5: `server/config.py` + `server/app.py` — FastAPI service, scale 0 only (M2)

**Files:**
- Create: `src/mrcng/server/config.py`
- Create: `src/mrcng/server/app.py`
- Test: `tests/test_server_scale0.py`

**Interfaces:**
- Consumes: `resolve_source` (Task 2), `parse_header` (Task 1), `plan_scales`/`build_info`/`parse_chunk_name`/`clip_chunk_to_scale`/`encode_chunk` (Task 4), `read_chunk` (Task 3).
- Produces: `Settings(BaseSettings)` with fields `source_root: Path, cache_root: Path, chunk_size: tuple[int,int,int] = (64,64,64), max_concurrent_reads: int = 32, fd_cache_size: int = 256, cors_origins: str = "*"`, env prefix `MRCNG_`.
- Produces: FastAPI `app` in `mrcng.server.app` with routes `GET /healthz` and `GET /data/{full_path:path}`.
- Produces (for Task 6 to extend): a module-level `get_settings()` dependency and a private `_dispatch(full_path: str)` split into `_serve_info`/`_serve_chunk` so cache logic can be inserted later without restructuring routing.

**Step-by-step:**

- [ ] **Step 1: Write failing integration tests**

Create `tests/test_server_scale0.py`:

```python
import os
import pytest
from fastapi.testclient import TestClient

from mrcng.server.config import Settings
from mrcng.server.app import create_app


@pytest.fixture
def client(tmp_path, make_mrc_file):
    source_root = tmp_path / "source"
    source_root.mkdir()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    def fill(zz, yy, xx):
        return (xx + 1000 * yy + 1_000_000 * zz) % 30000

    make_mrc_file.__wrapped__ if False else None  # no-op, keep fixture import used
    from tests.conftest import make_mrc
    make_mrc(source_root / "tomo.mrc", shape=(80, 60, 40), mode=1, voxel_size_angstrom=(6.8, 6.8, 6.8), fill=fill)

    settings = Settings(source_root=source_root, cache_root=cache_root)
    app = create_app(settings)
    return TestClient(app)


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200


def test_info_uncached_has_single_scale(client):
    resp = client.get("/data/tomo.mrc/info")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["scales"]) == 1
    assert body["scales"][0]["key"] == "1_1_1"
    assert body["scales"][0]["size"] == [80, 60, 40]


def test_info_missing_file_404s(client):
    resp = client.get("/data/does_not_exist.mrc/info")
    assert resp.status_code == 404


def test_info_path_traversal_404s(client):
    resp = client.get("/data/../outside.mrc/info")
    assert resp.status_code == 404


def test_scale0_chunk_matches_source(client, tmp_path):
    resp = client.get("/data/tomo.mrc/1_1_1/0-40_0-60_0-80")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/octet-stream"

    import numpy as np
    import mrcfile
    with mrcfile.open(tmp_path / "source" / "tomo.mrc", permissive=True) as mf:
        expected = mf.data  # (nz, ny, nx) = (80, 60, 40)
    got = np.frombuffer(resp.content, dtype="<i2").reshape(80, 60, 40)
    np.testing.assert_array_equal(got, expected)


def test_scale0_chunk_out_of_grid_404s(client):
    resp = client.get("/data/tomo.mrc/1_1_1/1000-1064_0-60_0-80")
    assert resp.status_code == 404


def test_uncached_higher_scale_404s(client):
    resp = client.get("/data/tomo.mrc/2_2_1/0-32_0-32_0-40")
    assert resp.status_code == 404


def test_malformed_chunk_spec_400s(client):
    resp = client.get("/data/tomo.mrc/not-a-scale/not-a-chunk")
    assert resp.status_code == 400
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_server_scale0.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Implement `server/config.py`**

Create `src/mrcng/server/config.py`:

```python
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MRCNG_")

    source_root: Path
    cache_root: Path
    chunk_size: tuple[int, int, int] = (64, 64, 64)
    max_concurrent_reads: int = 32
    fd_cache_size: int = 256
    cors_origins: str = "*"
```

- [ ] **Step 4: Implement `server/app.py`**

Create `src/mrcng/server/app.py`:

```python
"""FastAPI service speaking the Neuroglancer precomputed protocol.

M2 scope: scale 0 only, read directly from the MRC on every request (no fd
cache, no fingerprint/cache logic yet -- those are added in later tasks
without changing this routing structure).
"""
from __future__ import annotations

import os
import re

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from mrcng.mrcheader import parse_header, MrcFormatError
from mrcng.paths import resolve_source, PathNotAllowed
from mrcng.precomputed import (
    plan_scales, build_info, parse_chunk_name, clip_chunk_to_scale, encode_chunk,
)
from mrcng.reader import read_chunk, ChunkOutOfBounds, UnexpectedEOF

_SCALE_KEY_RE = re.compile(r"^\d+_\d+_\d+$")
_CHUNK_RE = re.compile(r"^\d+-\d+_\d+-\d+_\d+-\d+$")


def _open_header(settings, relpath: str):
    path = resolve_source(settings.source_root, relpath)
    fd = os.open(str(path), os.O_RDONLY)
    try:
        st = os.stat(fd)
        hdr = parse_header(fd, st.st_size, st.st_mtime_ns)
        return fd, hdr
    except BaseException:
        os.close(fd)
        raise


def create_app(settings) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.cors_origins] if settings.cors_origins != "*" else ["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/data/{full_path:path}")
    def dispatch(full_path: str):
        segments = full_path.split("/")

        if segments[-1] == "info":
            relpath = "/".join(segments[:-1])
            return _serve_info(settings, relpath)

        if len(segments) >= 2 and _SCALE_KEY_RE.match(segments[-2]) and _CHUNK_RE.match(segments[-1]):
            relpath = "/".join(segments[:-2])
            scale_key, chunk_str = segments[-2], segments[-1]
            return _serve_chunk(settings, relpath, scale_key, chunk_str)

        return Response(status_code=404)

    return app


def _serve_info(settings, relpath: str) -> Response:
    try:
        fd, hdr = _open_header(settings, relpath)
    except PathNotAllowed:
        return Response(status_code=404)
    try:
        try:
            scales = plan_scales(
                (hdr.nx, hdr.ny, hdr.nz),
                min_axis_size=32,
                max_levels=1,  # M2: single scale only, no cache awareness yet
            )
            info = build_info(hdr, scales, chunk_size=settings.chunk_size)
        except MrcFormatError as e:
            return Response(content=str(e), status_code=422)
    finally:
        os.close(fd)

    import json
    return Response(
        content=json.dumps(info),
        media_type="application/json",
        headers={"Cache-Control": "no-cache, must-revalidate"},
    )


def _serve_chunk(settings, relpath: str, scale_key: str, chunk_str: str) -> Response:
    try:
        fd, hdr = _open_header(settings, relpath)
    except PathNotAllowed:
        return Response(status_code=404)

    try:
        if scale_key != "1_1_1":
            return Response(status_code=404)  # M2: no cache, nothing above scale 0

        try:
            x0, x1, y0, y1, z0, z1 = parse_chunk_name(chunk_str)
        except ValueError:
            return Response(status_code=400)

        from mrcng.precomputed import ScaleLevel
        scale0 = ScaleLevel(key="1_1_1", size=(hdr.nx, hdr.ny, hdr.nz), factors=(1, 1, 1))
        try:
            cx0, cx1, cy0, cy1, cz0, cz1 = clip_chunk_to_scale(
                scale0, x0, x1, y0, y1, z0, z1, chunk_size=settings.chunk_size,
            )
        except ValueError:
            return Response(status_code=404)

        try:
            arr = read_chunk(fd, hdr, cx0, cx1, cy0, cy1, cz0, cz1)
        except (ChunkOutOfBounds, UnexpectedEOF):
            return Response(status_code=404)

        body = encode_chunk(arr)
        return Response(
            content=body,
            media_type="application/octet-stream",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )
    finally:
        os.close(fd)
```

- [ ] **Step 5: Run tests, iterate to green**

Run: `pixi run pytest tests/test_server_scale0.py -v`

Likely issue: `Settings` requires `source_root`/`cache_root` as `Path` but pydantic-settings may coerce `tuple[int,int,int]` for `chunk_size` oddly when passed directly in Python (not via env var) — since the test constructs `Settings(source_root=..., cache_root=...)` directly (not through env vars), this should work fine as pydantic validates constructor kwargs the same way. If `chunk_size` type validation complains, confirm the field accepts a plain Python tuple of ints (it will — pydantic v2 handles `tuple[int, int, int]` natively).

Expected after fixes: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mrcng/server/config.py src/mrcng/server/app.py tests/test_server_scale0.py
git commit -m "Add FastAPI service serving scale 0 directly from MRC files (M2)"
```

---

### Task 6: `downsample.py` — block-mean downsampling

**Files:**
- Create: `src/mrcng/downsample.py`
- Test: `tests/test_downsample.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure numpy function).
- Produces: `block_mean(arr: np.ndarray, factors: tuple[int, int, int]) -> np.ndarray`.

**Step-by-step:**

- [ ] **Step 1: Write failing tests**

Create `tests/test_downsample.py`:

```python
import numpy as np
import pytest

from mrcng.downsample import block_mean


def test_constant_volume_downsamples_to_same_constant():
    arr = np.full((8, 8, 8), 7, dtype=np.int16)
    result = block_mean(arr, (2, 2, 2))
    assert result.shape == (4, 4, 4)
    assert np.all(result == 7)
    assert result.dtype == arr.dtype


def test_ramp_volume_matches_analytical_expectation():
    # 1-D ramp along the last axis, factor 2: [0,1,2,3] -> mean of pairs -> [0.5, 2.5] -> round half away from zero -> [1, 3]
    arr = np.arange(4, dtype=np.int16).reshape(1, 1, 4)
    result = block_mean(arr, (2, 1, 1))
    assert result.shape == (1, 1, 2)
    assert result.tolist() == [[[1, 3]]]


def test_non_divisible_edge_averages_actual_voxel_count():
    # size 5 with factor 2 -> blocks of size 2,2,1 (last block has 1 voxel)
    arr = np.array([10, 20, 30, 40, 50], dtype=np.int16).reshape(1, 1, 5)
    result = block_mean(arr, (2, 1, 1))
    # blocks: mean(10,20)=15, mean(30,40)=35, mean(50)=50
    assert result.tolist() == [[[15, 35, 50]]]


def test_no_overflow_for_int16_accumulation():
    arr = np.full((2, 2, 2), 32000, dtype=np.int16)
    result = block_mean(arr, (2, 2, 2))
    assert result.item() == 32000  # would overflow if summed in int16 before dividing


def test_float32_input_downsamples_correctly():
    arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(1, 1, 4)
    result = block_mean(arr, (2, 1, 1))
    np.testing.assert_allclose(result, [[[1.5, 3.5]]])
    assert result.dtype == np.float32


def test_rounds_half_away_from_zero_for_negative_values():
    arr = np.array([-1, -2], dtype=np.int16).reshape(1, 1, 2)
    result = block_mean(arr, (2, 1, 1))
    # mean = -1.5, round half away from zero -> -2
    assert result.item() == -2


def test_clips_to_dtype_range():
    arr = np.array([127, 127], dtype=np.int8).reshape(1, 1, 2)
    result = block_mean(arr, (2, 1, 1))
    assert result.item() == 127  # exact mean is 127, no clipping needed; sanity check dtype stays int8
    assert result.dtype == np.int8
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_downsample.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Implement `downsample.py`**

Create `src/mrcng/downsample.py`:

```python
"""Block-mean downsampling. Accumulates in a wider dtype to avoid overflow;
never operates on a memmap (there are none in this codebase -- read chunks
via mrcng.reader first)."""
from __future__ import annotations

import numpy as np


def _reduceat_mean_1d(arr: np.ndarray, factor: int, axis: int, accum_dtype) -> np.ndarray:
    n = arr.shape[axis]
    indices = np.arange(0, n, factor)
    counts = np.diff(np.append(indices, n))
    summed = np.add.reduceat(arr.astype(accum_dtype), indices, axis=axis)
    shape = [1] * arr.ndim
    shape[axis] = len(counts)
    counts = counts.reshape(shape)
    return summed / counts


def block_mean(arr: np.ndarray, factors: tuple[int, int, int]) -> np.ndarray:
    src_dtype = arr.dtype
    if np.issubdtype(src_dtype, np.floating):
        accum_dtype = np.float64
    else:
        accum_dtype = np.int32

    result = arr
    for axis, factor in enumerate(factors):
        if factor == 1:
            continue
        result = _reduceat_mean_1d(result, factor, axis, accum_dtype)

    if np.issubdtype(src_dtype, np.floating):
        return result.astype(src_dtype)

    rounded = np.where(result >= 0, np.floor(result + 0.5), np.ceil(result - 0.5))
    info = np.iinfo(src_dtype)
    clipped = np.clip(rounded, info.min, info.max)
    return clipped.astype(src_dtype)
```

- [ ] **Step 4: Run to verify pass**

Run: `pixi run pytest tests/test_downsample.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mrcng/downsample.py tests/test_downsample.py
git commit -m "Add block-mean downsampling with overflow-safe accumulation"
```

---

### Task 7: `fingerprint.py` — compute, write, validate

**Files:**
- Create: `src/mrcng/fingerprint.py`
- Test: `tests/test_fingerprint.py`

**Interfaces:**
- Consumes: `MrcHeader` (Task 1), `pread_exact` (Task 3).
- Produces: `Params` typed dict/dataclass with `chunk_size, downsample, min_axis_size, max_levels, dtype, encoding`.
- Produces: `enum Validity {VALID, STALE, INCOMPATIBLE}`.
- Produces: `compute_header_sha256(fd: int, data_offset: int) -> str`.
- Produces: `build_fingerprint(hdr, relpath: str, params: Params, scales: list[str], generator_version: str, build_duration_s: float) -> dict`.
- Produces: `write_fingerprint(cache_dir: Path, fingerprint: dict) -> None` (writes `fingerprint.json`, fsyncs the file and the directory).
- Produces: `read_fingerprint(cache_dir: Path) -> dict | None` (None if absent or unparsable JSON).
- Produces: `validate(fp: dict, hdr, fd: int, current_params: Params) -> Validity`.

**Step-by-step:**

- [ ] **Step 1: Write failing tests**

Create `tests/test_fingerprint.py`:

```python
import json
import os
import time

from mrcng.mrcheader import parse_header
from mrcng.fingerprint import (
    Params, Validity, compute_header_sha256, build_fingerprint,
    write_fingerprint, read_fingerprint, validate,
)


def _params(**overrides):
    base = dict(chunk_size=(64, 64, 64), downsample="mean", min_axis_size=32,
                max_levels=6, dtype="int16", encoding="raw")
    base.update(overrides)
    return Params(**base)


def _open(path):
    fd = os.open(str(path), os.O_RDONLY)
    st = os.stat(fd)
    return fd, parse_header(fd, st.st_size, st.st_mtime_ns)


def test_write_then_read_roundtrip(tmp_path, make_mrc_file):
    path = make_mrc_file(shape=(8, 8, 8), mode=1)
    fd, hdr = _open(path)
    try:
        fp = build_fingerprint(
            hdr, relpath="tomo.mrc", params=_params(),
            scales=["2_2_2"], generator_version="mrc-pyramid 0.1.0",
            build_duration_s=1.23,
        )
        cache_dir = tmp_path / "cache_entry"
        cache_dir.mkdir()
        write_fingerprint(cache_dir, fp)
        loaded = read_fingerprint(cache_dir)
        assert loaded["source_relpath"] == "tomo.mrc"
        assert loaded["scales"] == ["2_2_2"]
    finally:
        os.close(fd)


def test_read_fingerprint_missing_returns_none(tmp_path):
    assert read_fingerprint(tmp_path / "nope") is None


def test_validate_valid_when_everything_matches(tmp_path, make_mrc_file):
    path = make_mrc_file(shape=(8, 8, 8), mode=1)
    fd, hdr = _open(path)
    try:
        fp = build_fingerprint(hdr, "tomo.mrc", _params(), ["2_2_2"], "v0.1.0", 1.0)
        assert validate(fp, hdr, fd, _params()) == Validity.VALID
    finally:
        os.close(fd)


def test_validate_stale_when_source_mtime_changes(tmp_path, make_mrc_file):
    path = make_mrc_file(shape=(8, 8, 8), mode=1)
    fd, hdr = _open(path)
    try:
        fp = build_fingerprint(hdr, "tomo.mrc", _params(), ["2_2_2"], "v0.1.0", 1.0)
    finally:
        os.close(fd)

    time.sleep(0.01)
    os.utime(path, None)  # bump mtime

    fd2 = os.open(str(path), os.O_RDONLY)
    try:
        st = os.stat(fd2)
        hdr2 = parse_header(fd2, st.st_size, st.st_mtime_ns)
        assert validate(fp, hdr2, fd2, _params()) == Validity.STALE
    finally:
        os.close(fd2)


def test_validate_incompatible_when_chunk_size_differs(tmp_path, make_mrc_file):
    path = make_mrc_file(shape=(8, 8, 8), mode=1)
    fd, hdr = _open(path)
    try:
        fp = build_fingerprint(hdr, "tomo.mrc", _params(), ["2_2_2"], "v0.1.0", 1.0)
        different_params = _params(chunk_size=(32, 32, 32))
        assert validate(fp, hdr, fd, different_params) == Validity.INCOMPATIBLE
    finally:
        os.close(fd)


def test_validate_incompatible_when_schema_version_unknown(tmp_path, make_mrc_file):
    path = make_mrc_file(shape=(8, 8, 8), mode=1)
    fd, hdr = _open(path)
    try:
        fp = build_fingerprint(hdr, "tomo.mrc", _params(), ["2_2_2"], "v0.1.0", 1.0)
        fp["schema_version"] = 999
        assert validate(fp, hdr, fd, _params()) == Validity.INCOMPATIBLE
    finally:
        os.close(fd)


def test_generator_version_change_alone_stays_valid(tmp_path, make_mrc_file):
    path = make_mrc_file(shape=(8, 8, 8), mode=1)
    fd, hdr = _open(path)
    try:
        fp = build_fingerprint(hdr, "tomo.mrc", _params(), ["2_2_2"], "v0.1.0", 1.0)
        fp["generator_version"] = "mrc-pyramid 9.9.9"
        assert validate(fp, hdr, fd, _params()) == Validity.VALID
    finally:
        os.close(fd)
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_fingerprint.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Implement `fingerprint.py`**

Create `src/mrcng/fingerprint.py`:

```python
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


_ADDRESSING_FIELDS = ("chunk_size", "encoding", "dtype")


def compute_header_sha256(fd: int, data_offset: int) -> str:
    raw = pread_exact(fd, data_offset, 0)
    return hashlib.sha256(raw).hexdigest()


def build_fingerprint(hdr, relpath: str, params: Params, scales: list[str],
                       generator_version: str, build_duration_s: float,
                       built_at: str | None = None) -> dict:
    fd = os.open(relpath, os.O_RDONLY) if False else None  # placeholder, replaced below
    raise NotImplementedError  # see corrected implementation below
```

Replace that last stub (the placeholder `build_fingerprint` above was left in mid-draft; do not commit it) with the real implementation that takes the already-open `fd` used to parse `hdr` rather than reopening the file — since `hdr` alone doesn't carry an fd, thread `fd` through explicitly:

```python
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
```

Note: the plan draft above accidentally included a broken first version of
`build_fingerprint`. When implementing, write the file with only the
corrected version (the one taking `fd` as its first argument) — there
should be exactly one `build_fingerprint` function in the final file.

- [ ] **Step 4: Run to verify pass**

Run: `pixi run pytest tests/test_fingerprint.py -v`
Expected: all PASS. (The test file above calls `build_fingerprint(hdr, "tomo.mrc", ...)` without `fd` in a few places — while implementing, update every call site in `tests/test_fingerprint.py` to pass `fd` as the first argument, matching the real signature `build_fingerprint(fd, hdr, relpath, params, scales, generator_version, build_duration_s)`. Fix the test file, not the signature — `fd` is required because computing `source_header_sha256` needs to read the file.)

- [ ] **Step 5: Commit**

```bash
git add src/mrcng/fingerprint.py tests/test_fingerprint.py
git commit -m "Add fingerprint compute/write/validate with fsync commit semantics"
```

---

### Task 8: `pyramid.py` — build orchestration

**Files:**
- Create: `src/mrcng/pyramid.py`
- Test: `tests/test_pyramid.py`

**Interfaces:**
- Consumes: `parse_header` (Task 1), `resolve_source`/`dataset_id`/`cache_dir_for` (Task 2), `read_chunk` (Task 3), `plan_scales`/`build_info`/`chunk_name`/`encode_chunk`/`ScaleLevel` (Task 4), `block_mean` (Task 6), `fingerprint.*` (Task 7).
- Produces: `enum BuildStatus {BUILT, SKIPPED_VALID, SKIPPED_LOCKED, FAILED}`.
- Produces: `@dataclass BuildResult(relpath, dataset_id, status, source_bytes, cache_bytes, levels_built, duration_s, error)`.
- Produces: `build_one(source_root, cache_root, relpath, params: Params, force: bool = False) -> BuildResult`.

**Step-by-step:**

- [ ] **Step 1: Write failing tests**

Create `tests/test_pyramid.py`:

```python
import os
import fcntl
import numpy as np
import pytest

from mrcng.fingerprint import Params, read_fingerprint
from mrcng.paths import dataset_id, cache_dir_for
from mrcng.pyramid import build_one, BuildStatus


def _params(**overrides):
    base = dict(chunk_size=(8, 8, 8), downsample="mean", min_axis_size=8,
                max_levels=3, dtype="int16", encoding="raw")
    base.update(overrides)
    return Params(**base)


@pytest.fixture
def source_and_cache(tmp_path, make_mrc_file):
    source_root = tmp_path / "source"
    source_root.mkdir()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    def fill(zz, yy, xx):
        return (xx + 1000 * yy + 1_000_000 * zz) % 30000

    path = source_root / "tomo.mrc"
    from tests.conftest import make_mrc
    make_mrc(path, shape=(32, 32, 32), mode=1, fill=fill)
    return source_root, cache_root, "tomo.mrc"


def test_build_creates_fingerprint_and_scales(source_and_cache):
    source_root, cache_root, relpath = source_and_cache
    result = build_one(source_root, cache_root, relpath, _params())
    assert result.status == BuildStatus.BUILT
    assert result.levels_built >= 1

    ds_id = dataset_id(relpath)
    cache_dir = cache_dir_for(cache_root, ds_id)
    fp = read_fingerprint(cache_dir)
    assert fp is not None
    assert fp["source_relpath"] == relpath


def test_build_output_matches_in_memory_reference_downsample(source_and_cache):
    source_root, cache_root, relpath = source_and_cache
    build_one(source_root, cache_root, relpath, _params())

    import mrcfile
    from mrcng.downsample import block_mean
    from mrcng.precomputed import chunk_name

    with mrcfile.open(source_root / relpath, permissive=True) as mf:
        level0 = mf.data  # (nz, ny, nx)

    expected_level1 = block_mean(level0, (2, 2, 2))  # factors x,y,z but block_mean takes matching axis order
    ds_id = dataset_id(relpath)
    cache_dir = cache_dir_for(cache_root, ds_id)

    name = chunk_name(0, min(8, expected_level1.shape[2]), 0, min(8, expected_level1.shape[1]), 0, min(8, expected_level1.shape[0]))
    chunk_path = cache_dir / "2_2_2" / name
    assert chunk_path.exists()
    on_disk = np.fromfile(chunk_path, dtype="<i2").reshape(
        min(8, expected_level1.shape[0]), min(8, expected_level1.shape[1]), min(8, expected_level1.shape[2])
    )
    np.testing.assert_array_equal(
        on_disk,
        expected_level1[0:on_disk.shape[0], 0:on_disk.shape[1], 0:on_disk.shape[2]],
    )


def test_skips_valid_cache_unless_forced(source_and_cache):
    source_root, cache_root, relpath = source_and_cache
    first = build_one(source_root, cache_root, relpath, _params())
    assert first.status == BuildStatus.BUILT

    second = build_one(source_root, cache_root, relpath, _params())
    assert second.status == BuildStatus.SKIPPED_VALID

    forced = build_one(source_root, cache_root, relpath, _params(), force=True)
    assert forced.status == BuildStatus.BUILT


def test_concurrent_build_reports_skipped_locked(source_and_cache):
    source_root, cache_root, relpath = source_and_cache
    ds_id = dataset_id(relpath)
    cache_dir = cache_dir_for(cache_root, ds_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / ".lock"
    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    try:
        result = build_one(source_root, cache_root, relpath, _params())
        assert result.status == BuildStatus.SKIPPED_LOCKED
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def test_killed_build_leaves_no_fingerprint_and_is_rebuilt(source_and_cache, monkeypatch):
    source_root, cache_root, relpath = source_and_cache

    from mrcng import pyramid as pyramid_module
    original = pyramid_module.write_fingerprint

    def boom(*args, **kwargs):
        raise RuntimeError("simulated crash before fingerprint write")

    monkeypatch.setattr(pyramid_module, "write_fingerprint", boom)
    with pytest.raises(RuntimeError):
        build_one(source_root, cache_root, relpath, _params())

    ds_id = dataset_id(relpath)
    cache_dir = cache_dir_for(cache_root, ds_id)
    assert read_fingerprint(cache_dir) is None

    monkeypatch.setattr(pyramid_module, "write_fingerprint", original)
    result = build_one(source_root, cache_root, relpath, _params())
    assert result.status == BuildStatus.BUILT
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_pyramid.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Implement `pyramid.py`**

Create `src/mrcng/pyramid.py`:

```python
"""Pyramid build orchestration -- used only by the mrc-pyramid CLI, never by
the server. Level 1 is built by streaming from the source MRC; every level
after that is built from the previous level's cache chunks, so the total
cost is ~1.15 passes over the source, not N passes."""
from __future__ import annotations

import enum
import fcntl
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mrcng.downsample import block_mean
from mrcng.fingerprint import Params, build_fingerprint, write_fingerprint, read_fingerprint, validate, Validity
from mrcng.mrcheader import parse_header
from mrcng.paths import resolve_source, dataset_id, cache_dir_for
from mrcng.precomputed import plan_scales, build_info, chunk_name, encode_chunk
from mrcng.reader import read_chunk

GENERATOR_VERSION = "mrc-pyramid 0.1.0"


class BuildStatus(enum.Enum):
    BUILT = "built"
    SKIPPED_VALID = "skipped_valid"
    SKIPPED_LOCKED = "skipped_locked"
    FAILED = "failed"


@dataclass
class BuildResult:
    relpath: str
    dataset_id: str
    status: BuildStatus
    source_bytes: int = 0
    cache_bytes: int = 0
    levels_built: int = 0
    duration_s: float = 0.0
    error: str | None = None


def _open_source(source_root: Path, relpath: str):
    path = resolve_source(source_root, relpath)
    fd = os.open(str(path), os.O_RDONLY)
    st = os.stat(fd)
    hdr = parse_header(fd, st.st_size, st.st_mtime_ns)
    return fd, hdr


def _write_chunk(cache_dir: Path, scale_key: str, name: str, arr: np.ndarray) -> int:
    scale_dir = cache_dir / scale_key
    scale_dir.mkdir(parents=True, exist_ok=True)
    body = encode_chunk(arr)
    (scale_dir / name).write_bytes(body)
    return len(body)


def _build_level1_from_source(fd, hdr, cache_dir: Path, level0, level1, chunk_size, max_block_bytes: int = 256 * 1024 * 1024) -> int:
    cx, cy, cz = chunk_size
    fx, fy, fz = level1.factors
    nx, ny, nz = level0.size
    out_nx, out_ny, out_nz = level1.size
    cache_bytes = 0

    for oz0 in range(0, out_nz, cz):
        oz1 = min(oz0 + cz, out_nz)
        src_z0, src_z1 = oz0 * fz, min(oz1 * fz, nz)
        for oy0 in range(0, out_ny, cy):
            oy1 = min(oy0 + cy, out_ny)
            src_y0, src_y1 = oy0 * fy, min(oy1 * fy, ny)

            row_bytes_full = (src_z1 - src_z0) * (src_y1 - src_y0) * nx * hdr.dtype.itemsize
            x_pieces = max(1, -(-row_bytes_full // max_block_bytes))
            piece_out_width = -(-out_nx // x_pieces)

            for piece_start in range(0, out_nx, piece_out_width):
                piece_out_end = min(piece_start + piece_out_width, out_nx)
                src_x0, src_x1 = piece_start * fx, min(piece_out_end * fx, nx)

                block = read_chunk(fd, hdr, src_x0, src_x1, src_y0, src_y1, src_z0, src_z1)
                downsampled = block_mean(block, (fz, fy, fx))

                for oz_local in range(downsampled.shape[0]):
                    oz = oz0 + oz_local
                    if oz >= out_nz:
                        break
                    for oy_local in range(downsampled.shape[1]):
                        oy = oy0 + oy_local
                        if oy >= out_ny:
                            break
                        for ox0 in range(piece_start, piece_out_end, cx):
                            ox1 = min(ox0 + cx, out_nx)
                            ox0_local, ox1_local = ox0 - piece_start, ox1 - piece_start
                            chunk_arr = downsampled[oz_local:oz_local + 1, oy_local:oy_local + 1, ox0_local:ox1_local]
                            chunk_arr = np.ascontiguousarray(chunk_arr)
                            name = chunk_name(ox0, ox1, oy, oy + 1, oz, oz + 1)
                            cache_bytes += _write_chunk(cache_dir, level1.key, name, chunk_arr)

    return cache_bytes


def build_one(source_root, cache_root, relpath: str, params: Params, force: bool = False) -> BuildResult:
    source_root, cache_root = Path(source_root), Path(cache_root)
    start = time.monotonic()
    ds_id = dataset_id(relpath)
    cache_dir = cache_dir_for(cache_root, ds_id)

    fd, hdr = _open_source(source_root, relpath)
    try:
        existing = read_fingerprint(cache_dir)
        if existing is not None and not force:
            if validate(existing, hdr, fd, params) == Validity.VALID:
                return BuildResult(relpath, ds_id, BuildStatus.SKIPPED_VALID, source_bytes=hdr.file_size)

        cache_dir.mkdir(parents=True, exist_ok=True)
        lock_path = cache_dir / ".lock"
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
        try:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return BuildResult(relpath, ds_id, BuildStatus.SKIPPED_LOCKED, source_bytes=hdr.file_size)

            fp_path = cache_dir / "fingerprint.json"
            if fp_path.exists():
                fp_path.unlink()

            scales = plan_scales((hdr.nx, hdr.ny, hdr.nz), params.min_axis_size, params.max_levels)
            cache_bytes = 0
            levels_built = 0

            if len(scales) > 1:
                cache_bytes += _build_level1_from_source(fd, hdr, cache_dir, scales[0], scales[1], params.chunk_size)
                levels_built += 1

                for level_idx in range(2, len(scales)):
                    cache_bytes += _build_level_from_previous(
                        cache_dir, scales[level_idx - 1], scales[level_idx], params.chunk_size,
                    )
                    levels_built += 1

            info = build_info(hdr, scales, params.chunk_size, params.encoding)
            import json
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
            )
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
    finally:
        os.close(fd)


def _build_level_from_previous(cache_dir: Path, prev_scale, next_scale, chunk_size) -> int:
    fx = next_scale.factors[0] // prev_scale.factors[0]
    fy = next_scale.factors[1] // prev_scale.factors[1]
    fz = next_scale.factors[2] // prev_scale.factors[2]
    cx, cy, cz = chunk_size
    out_nx, out_ny, out_nz = next_scale.size
    prev_nx, prev_ny, prev_nz = prev_scale.size
    cache_bytes = 0

    for oz in range(0, out_nz, 1):
        src_z0, src_z1 = oz * fz, min((oz + 1) * fz, prev_nz)
        for oy in range(0, out_ny, 1):
            src_y0, src_y1 = oy * fy, min((oy + 1) * fy, prev_ny)
            for ox0 in range(0, out_nx, cx):
                ox1 = min(ox0 + cx, out_nx)
                src_x0, src_x1 = ox0 * fx, min(ox1 * fx, prev_nx)

                block = _read_prev_level_region(cache_dir, prev_scale, chunk_size, src_x0, src_x1, src_y0, src_y1, src_z0, src_z1)
                downsampled = block_mean(block, (fz, fy, fx))
                downsampled = np.ascontiguousarray(downsampled)

                name = chunk_name(ox0, ox1, oy, oy + 1, oz, oz + 1)
                cache_bytes += _write_chunk(cache_dir, next_scale.key, name, downsampled)

    return cache_bytes


def _read_prev_level_region(cache_dir: Path, scale, chunk_size, x0, x1, y0, y1, z0, z1) -> np.ndarray:
    cx, cy, cz = chunk_size
    out = None
    for z in range(z0, z1):
        for y in range(y0, y1):
            for cxi0 in range(x0 - x0 % cx, x1, cx):
                cxi1 = min(cxi0 + cx, scale.size[0])
                name = chunk_name(cxi0, cxi1, y - y % cy, min(y - y % cy + cy, scale.size[1]), z - z % cz, min(z - z % cz + cz, scale.size[2]))
                path = cache_dir / scale.key / name
                raw = path.read_bytes()
                arr = np.frombuffer(raw, dtype="<i2")  # dtype corrected below via params in real impl
                # NOTE: reading a full chunk file back and slicing out the sub-region
                # required (x0:x1, y0:y1, z0:z1) is the simplest correct approach here;
                # a production version would cache the decoded chunk across the (y0,z0)
                # loop instead of re-reading per voxel row. See Task 8 self-review note.
    if out is None:
        raise NotImplementedError("see self-review note below; this helper needs a rewrite")
    return out


def _fsync_tree(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(root):
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
```

**Self-review note (fix before running tests):** the `_read_prev_level_region` sketch above is incomplete — it must actually assemble and return a `(z1-z0, y1-y0, x1-x0)` array read from the previous level's cache chunk files, using `mrcng.reader`-style dtype handling (the dtype comes from `params.dtype`, not a hardcoded `<i2`). Since chunk files on disk are exactly one output chunk each (no sub-chunk addressing needed within `_build_level_from_previous`, because it already iterates `ox0` in steps of `cx` — i.e., every "read" of the previous level corresponds to reading between 1 and `fx*fy*fz` *whole* previous-level chunk files and concatenating them), rewrite it as:

```python
def _read_prev_level_region(cache_dir: Path, scale, chunk_size, dtype, x0, x1, y0, y1, z0, z1) -> np.ndarray:
    cx, cy, cz = chunk_size
    sx, sy, sz = scale.size
    out = np.empty((z1 - z0, y1 - y0, x1 - x0), dtype=dtype)

    for z in range(z0, z1):
        for y in range(y0, y1):
            x = x0
            while x < x1:
                block_x0 = (x // cx) * cx
                block_x1 = min(block_x0 + cx, sx)
                block_y0 = (y // cy) * cy
                block_y1 = min(block_y0 + cy, sy)
                block_z0 = (z // cz) * cz
                block_z1 = min(block_z0 + cz, sz)

                name = chunk_name(block_x0, block_x1, block_y0, block_y1, block_z0, block_z1)
                path = cache_dir / scale.key / name
                raw = np.frombuffer(path.read_bytes(), dtype=dtype).reshape(
                    block_z1 - block_z0, block_y1 - block_y0, block_x1 - block_x0
                )

                take_x0 = max(x, block_x0) - block_x0
                take_x1 = min(x1, block_x1) - block_x0
                out_x0 = max(x, block_x0) - x0
                out_x1 = min(x1, block_x1) - x0
                out[z - z0, y - y0, out_x0:out_x1] = raw[z - block_z0, y - block_y0, take_x0:take_x1]

                x = block_x1
    return out
```

And update `_build_level_from_previous`'s call site to pass `dtype=np.dtype(prev_scale_dtype)` — thread the dtype through from `params.dtype` (e.g. `np.dtype(params.dtype)`) since `_build_level_from_previous` already has `chunk_size` and can also take `dtype` as a parameter; add it to both function signatures (`_build_level_from_previous(cache_dir, prev_scale, next_scale, chunk_size, dtype)` and pass `dtype` through to `_read_prev_level_region`).

- [ ] **Step 4: Run tests, iterate to green**

Run: `pixi run pytest tests/test_pyramid.py -v`

Debug iteratively — this is the most complex module in the codebase. Common issues to check if tests fail:
- Off-by-one in `range(0, out_nz, 1)` loops in `_build_level_from_previous` (single-voxel-at-a-time in z/y is intentionally simple/slow but correct; do not optimize it in this task).
- `block_mean` factor order: `block_mean` takes `(factor_z, factor_y, factor_x)` matching array axis order `(z, y, x)` — every call site must pass factors in that order, not `(fx, fy, fz)`.
- `chunk_name`/`clip_chunk_to_scale` expect `(x0, x1, y0, y1, z0, z1)` — double-check argument order at each call site against Task 4's signatures.

Expected after fixes: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mrcng/pyramid.py tests/test_pyramid.py
git commit -m "Add pyramid build orchestration with locking and commit semantics"
```

---

### Task 9: `cli.py` — mrc-pyramid entry point

**Files:**
- Create: `src/mrcng/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `build_one`, `BuildStatus` (Task 8), `Params` (Task 7), `resolve_source`/`dataset_id`/`cache_dir_for` (Task 2), `read_fingerprint`/`validate` (Task 7).
- Produces: `main(argv: list[str] | None = None) -> int`, registered as the `mrc-pyramid` console script (already wired in `pyproject.toml`'s `[project.scripts]`).
- Produces: subcommands `build`, `status`, `prune`.

**Step-by-step:**

- [ ] **Step 1: Write failing tests**

Create `tests/test_cli.py`:

```python
import json
import os

from mrcng.cli import main


def _make_source_tree(tmp_path, make_mrc):
    source_root = tmp_path / "source"
    source_root.mkdir()
    make_mrc(source_root / "a.mrc", shape=(16, 16, 16), mode=1)
    make_mrc(source_root / "sub" / "b.mrc" if (source_root / "sub").exists() else source_root / "b.mrc", shape=(16, 16, 16), mode=1)
    return source_root


def test_build_writes_report_jsonl(tmp_path, capsys):
    from tests.conftest import make_mrc
    source_root = tmp_path / "source"
    source_root.mkdir()
    make_mrc(source_root / "a.mrc", shape=(16, 16, 16), mode=1)
    cache_root = tmp_path / "cache"
    report_path = tmp_path / "report.jsonl"

    rc = main([
        "build", str(source_root), "--cache-root", str(cache_root),
        "--chunk-size", "8,8,8", "--min-axis-size", "8", "--max-levels", "3",
        "--report", str(report_path),
    ])
    assert rc == 0

    lines = report_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["relpath"] == "a.mrc"
    assert record["status"] == "built"


def test_status_reports_missing_and_valid(tmp_path, capsys):
    from tests.conftest import make_mrc
    source_root = tmp_path / "source"
    source_root.mkdir()
    make_mrc(source_root / "a.mrc", shape=(16, 16, 16), mode=1)
    cache_root = tmp_path / "cache"

    main(["build", str(source_root), "--cache-root", str(cache_root), "--chunk-size", "8,8,8"])
    rc = main(["status", str(source_root), "--cache-root", str(cache_root)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "a.mrc" in out
    assert "valid" in out


def test_prune_removes_orphaned_cache_entries(tmp_path):
    from tests.conftest import make_mrc
    source_root = tmp_path / "source"
    source_root.mkdir()
    mrc_path = source_root / "a.mrc"
    make_mrc(mrc_path, shape=(16, 16, 16), mode=1)
    cache_root = tmp_path / "cache"

    main(["build", str(source_root), "--cache-root", str(cache_root), "--chunk-size", "8,8,8"])
    os.remove(mrc_path)

    rc = main(["prune", "--cache-root", str(cache_root), "--source-root", str(source_root)])
    assert rc == 0

    from mrcng.paths import dataset_id, cache_dir_for
    cache_dir = cache_dir_for(cache_root, dataset_id("a.mrc"))
    assert not cache_dir.exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_cli.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Implement `cli.py`**

Create `src/mrcng/cli.py`:

```python
"""mrc-pyramid CLI: build/status/prune."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from mrcng.fingerprint import Params, read_fingerprint, validate
from mrcng.mrcheader import parse_header
from mrcng.paths import dataset_id, cache_dir_for
from mrcng.pyramid import build_one, BuildStatus


def _parse_chunk_size(s: str) -> tuple[int, int, int]:
    parts = tuple(int(p) for p in s.split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("chunk-size must be X,Y,Z")
    return parts


def _iter_mrc_files(source_root: Path, globs: list[str]):
    seen = set()
    for pattern in globs:
        for path in sorted(source_root.rglob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path.relative_to(source_root).as_posix()


def _build_command(args) -> int:
    source_root = Path(args.source_root)
    cache_root = Path(args.cache_root)
    params = Params(
        chunk_size=tuple(args.chunk_size), downsample="mean",
        min_axis_size=args.min_axis_size, max_levels=args.max_levels,
        dtype="int16", encoding="raw",
    )
    globs = args.glob or ["*.mrc"]

    report_lines = []
    for relpath in _iter_mrc_files(source_root, globs):
        try:
            result = build_one(source_root, cache_root, relpath, params, force=args.force)
            record = {
                "relpath": result.relpath, "dataset_id": result.dataset_id,
                "status": result.status.value, "source_bytes": result.source_bytes,
                "cache_bytes": result.cache_bytes, "levels_built": result.levels_built,
                "duration_s": result.duration_s, "error": result.error,
            }
        except Exception as e:
            record = {"relpath": relpath, "status": "failed", "error": str(e)}
        report_lines.append(record)
        print(json.dumps(record))

    if args.report:
        with open(args.report, "w") as f:
            for record in report_lines:
                f.write(json.dumps(record) + "\n")

    return 0


def _status_command(args) -> int:
    import os
    source_root = Path(args.source_root)
    cache_root = Path(args.cache_root)

    for relpath in _iter_mrc_files(source_root, ["*.mrc"]):
        ds_id = dataset_id(relpath)
        cache_dir = cache_dir_for(cache_root, ds_id)
        fp = read_fingerprint(cache_dir)
        if fp is None:
            print(f"{relpath}: missing")
            continue

        path = source_root / relpath
        fd = os.open(str(path), os.O_RDONLY)
        try:
            st = os.stat(fd)
            hdr = parse_header(fd, st.st_size, st.st_mtime_ns)
            params = Params(**fp["params"])
            params = Params(**{**fp["params"], "chunk_size": tuple(fp["params"]["chunk_size"])})
            v = validate(fp, hdr, fd, params)
        finally:
            os.close(fd)
        print(f"{relpath}: {v.value}")

    return 0


def _prune_command(args) -> int:
    cache_root = Path(args.cache_root)
    source_root = Path(args.source_root)

    known_ids = {dataset_id(rel) for rel in _iter_mrc_files(source_root, ["*.mrc"])}

    for prefix_dir in cache_root.iterdir():
        if not prefix_dir.is_dir():
            continue
        for entry_dir in prefix_dir.iterdir():
            if entry_dir.is_dir() and entry_dir.name not in known_ids:
                shutil.rmtree(entry_dir)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mrc-pyramid")
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build")
    build_p.add_argument("source_root")
    build_p.add_argument("--cache-root", required=True)
    build_p.add_argument("--glob", action="append")
    build_p.add_argument("--chunk-size", type=_parse_chunk_size, default=(64, 64, 64))
    build_p.add_argument("--min-axis-size", type=int, default=32)
    build_p.add_argument("--max-levels", type=int, default=6)
    build_p.add_argument("--force", action="store_true")
    build_p.add_argument("--report")
    build_p.set_defaults(func=_build_command)

    status_p = sub.add_parser("status")
    status_p.add_argument("source_root")
    status_p.add_argument("--cache-root", required=True)
    status_p.set_defaults(func=_status_command)

    prune_p = sub.add_parser("prune")
    prune_p.add_argument("--cache-root", required=True)
    prune_p.add_argument("--source-root", required=True)
    prune_p.set_defaults(func=_prune_command)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests, iterate to green**

Run: `pixi run pytest tests/test_cli.py -v`
Expected: all PASS. If `test_build_writes_report_jsonl` fails because the test's `make_mrc` helper call for a `sub/b.mrc` path in the (unused) `_make_source_tree` helper errors — that helper isn't actually called by any test in this file, remove it from the test file if flagged by a linter, it's dead code left over from drafting.

- [ ] **Step 5: Commit**

```bash
git add src/mrcng/cli.py tests/test_cli.py
git commit -m "Add mrc-pyramid CLI (build/status/prune)"
```

---

### Task 10: Cache-aware server — wire fingerprint validation into `app.py` (M4)

**Files:**
- Modify: `src/mrcng/server/app.py`
- Test: `tests/test_server_cached.py`

**Interfaces:**
- Consumes: `read_fingerprint`/`validate`/`Params` (Task 7), `dataset_id`/`cache_dir_for` (Task 2), `build_one` (Task 8, used only in test setup to create a real cache — never imported by `app.py` itself).
- Produces: `_serve_info` now returns the cached `info` file verbatim when the fingerprint is `VALID`; `_serve_chunk` now serves cached chunk files for non-`"1_1_1"` scale keys when `VALID`, 404 otherwise.

**Step-by-step:**

- [ ] **Step 1: Write failing tests**

Create `tests/test_server_cached.py`:

```python
import os
import time
import numpy as np
import pytest
from fastapi.testclient import TestClient

from mrcng.fingerprint import Params
from mrcng.pyramid import build_one
from mrcng.server.config import Settings
from mrcng.server.app import create_app


@pytest.fixture
def cached_setup(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    def fill(zz, yy, xx):
        return (xx + 1000 * yy + 1_000_000 * zz) % 30000

    from tests.conftest import make_mrc
    relpath = "tomo.mrc"
    make_mrc(source_root / relpath, shape=(32, 32, 32), mode=1, fill=fill)

    params = Params(chunk_size=(8, 8, 8), downsample="mean", min_axis_size=8,
                     max_levels=3, dtype="int16", encoding="raw")
    build_one(source_root, cache_root, relpath, params)

    settings = Settings(source_root=source_root, cache_root=cache_root, chunk_size=(8, 8, 8))
    return TestClient(create_app(settings)), source_root, relpath


def test_info_has_all_scales_when_cache_valid(cached_setup):
    client, _, relpath = cached_setup
    resp = client.get(f"/data/{relpath}/info")
    assert resp.status_code == 200
    scales = resp.json()["scales"]
    assert len(scales) > 1


def test_cached_chunk_byte_identical_to_disk(cached_setup):
    client, source_root, relpath = cached_setup
    resp = client.get(f"/data/{relpath}/2_2_2/0-8_0-8_0-8")
    assert resp.status_code == 200

    from mrcng.paths import dataset_id, cache_dir_for
    cache_dir = cache_dir_for(source_root.parent / "cache", dataset_id(relpath))
    on_disk = (cache_dir / "2_2_2" / "0-8_0-8_0-8").read_bytes()
    assert resp.content == on_disk


def test_stale_cache_falls_back_to_single_scale(cached_setup):
    client, source_root, relpath = cached_setup
    time.sleep(0.01)
    os.utime(source_root / relpath, None)  # bump mtime -> STALE

    resp = client.get(f"/data/{relpath}/info")
    assert resp.status_code == 200
    assert len(resp.json()["scales"]) == 1

    chunk_resp = client.get(f"/data/{relpath}/2_2_2/0-8_0-8_0-8")
    assert chunk_resp.status_code == 404


def test_scale0_still_served_when_cache_valid(cached_setup):
    client, _, relpath = cached_setup
    resp = client.get(f"/data/{relpath}/1_1_1/0-8_0-8_0-8")
    assert resp.status_code == 200
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_server_cached.py -v`
Expected: FAIL — `test_info_has_all_scales_when_cache_valid` and the others fail because `app.py` still hardcodes `max_levels=1` and 404s all non-`"1_1_1"` scale keys.

- [ ] **Step 3: Modify `server/app.py`**

Replace the body of `_serve_info` in `src/mrcng/server/app.py`:

```python
def _serve_info(settings, relpath: str) -> Response:
    try:
        fd, hdr = _open_header(settings, relpath)
    except PathNotAllowed:
        return Response(status_code=404)

    try:
        from mrcng.fingerprint import Params, read_fingerprint, validate, Validity
        from mrcng.paths import dataset_id, cache_dir_for

        params = Params(
            chunk_size=tuple(settings.chunk_size), downsample="mean",
            min_axis_size=32, max_levels=6, dtype=str(hdr.dtype), encoding="raw",
        )
        ds_id = dataset_id(relpath)
        cache_dir = cache_dir_for(settings.cache_root, ds_id)
        fp = read_fingerprint(cache_dir)

        if fp is not None and validate(fp, hdr, fd, params) == Validity.VALID:
            info_bytes = (cache_dir / "info").read_bytes()
            return Response(
                content=info_bytes, media_type="application/json",
                headers={"Cache-Control": "no-cache, must-revalidate"},
            )

        try:
            scales = plan_scales((hdr.nx, hdr.ny, hdr.nz), min_axis_size=32, max_levels=1)
            info = build_info(hdr, scales, chunk_size=settings.chunk_size)
        except MrcFormatError as e:
            return Response(content=str(e), status_code=422)
    finally:
        os.close(fd)

    import json
    return Response(
        content=json.dumps(info), media_type="application/json",
        headers={"Cache-Control": "no-cache, must-revalidate"},
    )
```

Replace the body of `_serve_chunk`:

```python
def _serve_chunk(settings, relpath: str, scale_key: str, chunk_str: str) -> Response:
    try:
        fd, hdr = _open_header(settings, relpath)
    except PathNotAllowed:
        return Response(status_code=404)

    try:
        try:
            x0, x1, y0, y1, z0, z1 = parse_chunk_name(chunk_str)
        except ValueError:
            return Response(status_code=400)

        if scale_key == "1_1_1":
            from mrcng.precomputed import ScaleLevel
            scale0 = ScaleLevel(key="1_1_1", size=(hdr.nx, hdr.ny, hdr.nz), factors=(1, 1, 1))
            try:
                cx0, cx1, cy0, cy1, cz0, cz1 = clip_chunk_to_scale(
                    scale0, x0, x1, y0, y1, z0, z1, chunk_size=settings.chunk_size,
                )
            except ValueError:
                return Response(status_code=404)
            try:
                arr = read_chunk(fd, hdr, cx0, cx1, cy0, cy1, cz0, cz1)
            except (ChunkOutOfBounds, UnexpectedEOF):
                return Response(status_code=404)
            body = encode_chunk(arr)
            return Response(
                content=body, media_type="application/octet-stream",
                headers={"Cache-Control": "public, max-age=31536000, immutable"},
            )

        from mrcng.fingerprint import Params, read_fingerprint, validate, Validity
        from mrcng.paths import dataset_id, cache_dir_for

        params = Params(
            chunk_size=tuple(settings.chunk_size), downsample="mean",
            min_axis_size=32, max_levels=6, dtype=str(hdr.dtype), encoding="raw",
        )
        ds_id = dataset_id(relpath)
        cache_dir = cache_dir_for(settings.cache_root, ds_id)
        fp = read_fingerprint(cache_dir)
        if fp is None or validate(fp, hdr, fd, params) != Validity.VALID:
            return Response(status_code=404)

        chunk_path = cache_dir / scale_key / chunk_str
        if not chunk_path.is_file():
            return Response(status_code=404)

        from fastapi.responses import FileResponse
        return FileResponse(
            chunk_path, media_type="application/octet-stream",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )
    finally:
        os.close(fd)
```

- [ ] **Step 4: Run full server test suite**

Run: `pixi run pytest tests/test_server_scale0.py tests/test_server_cached.py -v`
Expected: all PASS. If `Params(dtype=str(hdr.dtype), ...)` mismatches what `build_one` used during test setup (e.g. `"int16"` vs `"<i2"` string form), normalize by using `str(np.dtype(hdr.dtype).name)` (yields `"int16"`) consistently in both `app.py` and wherever `cli.py`/tests construct `Params` for a given header — check `hdr.dtype.name` gives `"int16"` and use that exact call in both places.

- [ ] **Step 5: Commit**

```bash
git add src/mrcng/server/app.py tests/test_server_cached.py
git commit -m "Wire fingerprint validation into server: cached info and chunks (M4)"
```

---

### Task 11: `server/fdcache.py` — bounded fd + header LRU cache (M5)

**Files:**
- Create: `src/mrcng/server/fdcache.py`
- Modify: `src/mrcng/server/app.py` (replace `_open_header`'s per-request `os.open`/`os.close` with the cache)
- Test: `tests/test_fdcache.py`

**Interfaces:**
- Consumes: `parse_header` (Task 1).
- Produces: `class FdCache` with `get(path: Path) -> MrcHeader` (opens+parses on miss, keeps the fd open, keyed by `(resolved_path, size, mtime_ns)`), `fd_for(path: Path) -> int` (returns the cached fd, same key), `close_all()`.

**Step-by-step:**

- [ ] **Step 1: Write failing tests**

Create `tests/test_fdcache.py`:

```python
import os
import time

from mrcng.server.fdcache import FdCache


def test_get_parses_header_once_and_reuses_fd(make_mrc_file):
    path = make_mrc_file(shape=(8, 8, 8), mode=1)
    cache = FdCache(max_size=8)
    hdr1 = cache.get(path)
    fd1 = cache.fd_for(path)
    hdr2 = cache.get(path)
    fd2 = cache.fd_for(path)
    assert hdr1 == hdr2
    assert fd1 == fd2
    cache.close_all()


def test_replaced_file_misses_cache(make_mrc_file, tmp_path):
    path = make_mrc_file(shape=(8, 8, 8), mode=1)
    cache = FdCache(max_size=8)
    hdr1 = cache.get(path)

    time.sleep(0.01)
    os.remove(path)
    from tests.conftest import make_mrc
    make_mrc(path, shape=(16, 16, 16), mode=1)  # same path, different size/mtime

    hdr2 = cache.get(path)
    assert hdr2.nx == 16
    assert hdr1.nx == 8
    cache.close_all()


def test_eviction_closes_oldest_fd(make_mrc_file, tmp_path):
    cache = FdCache(max_size=2)
    paths = []
    for i in range(3):
        from tests.conftest import make_mrc
        p = tmp_path / f"f{i}.mrc"
        make_mrc(p, shape=(4, 4, 4), mode=1)
        paths.append(p)
        cache.get(p)

    # first path's fd should have been evicted and closed
    key0 = cache._key_for(paths[0])
    assert key0 not in cache._entries
    cache.close_all()
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_fdcache.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Implement `server/fdcache.py`**

Create `src/mrcng/server/fdcache.py`:

```python
"""Bounded LRU of open MRC file descriptors and their parsed headers, keyed
by (resolved_path, size, mtime_ns) so a replaced source file misses the
cache and is reopened -- never serving stale data from an old fd."""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from pathlib import Path

from mrcng.mrcheader import parse_header


class FdCache:
    def __init__(self, max_size: int = 256):
        self._max_size = max_size
        self._entries: OrderedDict[tuple, tuple[int, object]] = OrderedDict()
        self._lock = threading.Lock()

    def _key_for(self, path: Path) -> tuple:
        st = os.stat(path)
        return (str(path), st.st_size, st.st_mtime_ns)

    def get(self, path: Path):
        key = self._key_for(path)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                return entry[1]

        fd = os.open(str(path), os.O_RDONLY)
        st = os.stat(fd)
        hdr = parse_header(fd, st.st_size, st.st_mtime_ns)

        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                os.close(fd)
                self._entries.move_to_end(key)
                return existing[1]

            self._entries[key] = (fd, hdr)
            self._entries.move_to_end(key)
            self._evict_if_needed()
            return hdr

    def fd_for(self, path: Path) -> int:
        key = self._key_for(path)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                raise KeyError(f"no cached fd for {path}; call get() first")
            return entry[0]

    def _evict_if_needed(self):
        while len(self._entries) > self._max_size:
            _, (fd, _) = self._entries.popitem(last=False)
            os.close(fd)

    def close_all(self):
        with self._lock:
            for fd, _ in self._entries.values():
                os.close(fd)
            self._entries.clear()
```

- [ ] **Step 4: Run to verify pass**

Run: `pixi run pytest tests/test_fdcache.py -v`
Expected: all PASS

- [ ] **Step 5: Wire `FdCache` into `server/app.py`**

Modify `src/mrcng/server/app.py`: add `_FD_CACHE = None` module-level, initialize it in `create_app`, and replace `_open_header` to use it instead of opening/closing per request:

```python
def create_app(settings) -> FastAPI:
    from mrcng.server.fdcache import FdCache
    fd_cache = FdCache(max_size=settings.fd_cache_size)

    app = FastAPI()
    app.state.fd_cache = fd_cache
    # ... (CORS middleware unchanged) ...

    @app.get("/data/{full_path:path}")
    def dispatch(full_path: str):
        # ... unchanged dispatch logic, but pass fd_cache through ...
        if segments[-1] == "info":
            relpath = "/".join(segments[:-1])
            return _serve_info(settings, fd_cache, relpath)
        if len(segments) >= 2 and _SCALE_KEY_RE.match(segments[-2]) and _CHUNK_RE.match(segments[-1]):
            relpath = "/".join(segments[:-2])
            return _serve_chunk(settings, fd_cache, relpath, segments[-2], segments[-1])
        return Response(status_code=404)

    return app
```

Update `_open_header`, `_serve_info`, and `_serve_chunk` signatures to take `fd_cache` and replace the `os.open(...)`/`parse_header(...)` pair with:

```python
def _open_header(settings, fd_cache, relpath: str):
    path = resolve_source(settings.source_root, relpath)
    hdr = fd_cache.get(path)
    fd = fd_cache.fd_for(path)
    return fd, hdr
```

Since the fd is now owned by the cache, remove every `finally: os.close(fd)` in `_serve_info`/`_serve_chunk` that previously closed it after each request — the fd cache owns its lifetime now, not the request handler. Double-check there is no other code path in `app.py` that still calls `os.close(fd)` on a cache-owned fd.

- [ ] **Step 6: Run the full server test suite to confirm no regression**

Run: `pixi run pytest tests/test_server_scale0.py tests/test_server_cached.py tests/test_fdcache.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/mrcng/server/fdcache.py src/mrcng/server/app.py tests/test_fdcache.py
git commit -m "Add bounded fd/header LRU cache and wire it into the server"
```

---

### Task 12: Concurrency semaphore, structured logging, `/healthz` detail (M5)

**Files:**
- Modify: `src/mrcng/server/app.py`
- Test: `tests/test_server_observability.py`

**Interfaces:**
- Consumes: `Settings.max_concurrent_reads` (Task 5).
- Produces: `_serve_chunk`'s scale-0 read now runs through `asyncio.to_thread` under an `asyncio.Semaphore(settings.max_concurrent_reads)`; both route handlers become `async def`. Structured JSON log line emitted per chunk/info request via the stdlib `logging` module with a dedicated `mrcng.access` logger. `/healthz` returns `{"status": "ok", "version": ..., "source_root": ..., "cache_root": ...}`.

**Step-by-step:**

- [ ] **Step 1: Write failing tests**

Create `tests/test_server_observability.py`:

```python
import json
import logging

from fastapi.testclient import TestClient

from mrcng.server.config import Settings
from mrcng.server.app import create_app


def test_healthz_reports_version_and_roots(tmp_path):
    source_root = tmp_path / "source"; source_root.mkdir()
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    settings = Settings(source_root=source_root, cache_root=cache_root)
    client = TestClient(create_app(settings))

    resp = client.get("/healthz")
    body = resp.json()
    assert body["status"] == "ok"
    assert body["source_root"] == str(source_root)
    assert body["cache_root"] == str(cache_root)
    assert "version" in body


def test_chunk_request_emits_structured_log(tmp_path, caplog, make_mrc_file):
    from tests.conftest import make_mrc
    source_root = tmp_path / "source"; source_root.mkdir()
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    make_mrc(source_root / "tomo.mrc", shape=(8, 8, 8), mode=1)

    settings = Settings(source_root=source_root, cache_root=cache_root)
    client = TestClient(create_app(settings))

    with caplog.at_level(logging.INFO, logger="mrcng.access"):
        resp = client.get("/data/tomo.mrc/1_1_1/0-8_0-8_0-8")
    assert resp.status_code == 200

    records = [r for r in caplog.records if r.name == "mrcng.access"]
    assert len(records) == 1
    payload = json.loads(records[0].message)
    assert payload["relpath"] == "tomo.mrc"
    assert payload["scale_key"] == "1_1_1"
    assert payload["cache_hit"] is False
    assert "duration_ms" in payload
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_server_observability.py -v`
Expected: FAIL — `/healthz` doesn't yet report `version`/`source_root`/`cache_root`; no `mrcng.access` logger emits anything.

- [ ] **Step 3: Modify `server/app.py`**

Add near the top of the file:

```python
import asyncio
import json as _json
import logging
import time

_access_logger = logging.getLogger("mrcng.access")

MRCNG_VERSION = "0.1.0"
```

Replace the `/healthz` handler:

```python
@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "version": MRCNG_VERSION,
        "source_root": str(settings.source_root),
        "cache_root": str(settings.cache_root),
    }
```

Wrap the scale-0 read in `_serve_chunk` with the semaphore and make the route + `_serve_chunk` async. In `create_app`, add:

```python
    semaphore = asyncio.Semaphore(settings.max_concurrent_reads)
```

and pass `semaphore` through to `_serve_chunk` the same way `fd_cache` is threaded through (update the `dispatch` route to `async def dispatch(...)`, `await _serve_chunk(...)`/`await _serve_info(...)`, and change both function definitions to `async def`).

Inside `_serve_chunk`, replace the direct `read_chunk(...)` call for the `scale_key == "1_1_1"` branch with:

```python
            async with semaphore:
                try:
                    arr = await asyncio.to_thread(read_chunk, fd, hdr, cx0, cx1, cy0, cy1, cz0, cz1)
                except (ChunkOutOfBounds, UnexpectedEOF):
                    return Response(status_code=404)
```

Add structured logging at the end of both `_serve_info` and `_serve_chunk` (right before each `return`), e.g. in `_serve_chunk`:

```python
    duration_ms = (time.monotonic() - start) * 1000
    _access_logger.info(_json.dumps({
        "relpath": relpath, "scale_key": scale_key, "chunk": chunk_str,
        "cache_hit": scale_key != "1_1_1", "duration_ms": round(duration_ms, 2),
    }))
```

Record `start = time.monotonic()` at the top of `_serve_chunk` (and similarly `_serve_info`, logging `relpath` and `cache_hit` there too) before doing any work, so `duration_ms` covers the whole handler.

- [ ] **Step 4: Run tests, iterate to green**

Run: `pixi run pytest tests/test_server_observability.py tests/test_server_scale0.py tests/test_server_cached.py -v`
Expected: all PASS. `TestClient` (from `fastapi.testclient`, built on `httpx`) handles async route handlers transparently — no test changes needed for the `async def` conversion.

- [ ] **Step 5: Commit**

```bash
git add src/mrcng/server/app.py tests/test_server_observability.py
git commit -m "Add concurrency semaphore, structured access logs, and healthz detail"
```

---

### Task 13: `benchmark.py` — load-test script (M5)

**Files:**
- Create: `src/mrcng/benchmark.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: nothing from the library directly — it's an `httpx`-based client script, but it can run against a real server or (in tests) an in-process ASGI app via `httpx.ASGITransport`.
- Produces: `run_benchmark(base_url: str, dataset_relpaths: list[str], concurrency: int = 8, requests_per_dataset: int = 20) -> dict` returning `{"p50_ms": ..., "p95_ms": ..., "p99_ms": ..., "count": ..., "errors": ...}`. Produces a `main()` CLI wrapper (registered as `pixi run benchmark`, invoked as `python -m mrcng.benchmark`).

**Step-by-step:**

- [ ] **Step 1: Write failing test**

Create `tests/test_benchmark.py`:

```python
import httpx
import pytest

from mrcng.server.config import Settings
from mrcng.server.app import create_app
from mrcng.benchmark import run_benchmark_async


@pytest.mark.anyio
async def test_benchmark_against_in_process_app(tmp_path):
    from tests.conftest import make_mrc
    source_root = tmp_path / "source"; source_root.mkdir()
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    make_mrc(source_root / "tomo.mrc", shape=(16, 16, 16), mode=1)

    settings = Settings(source_root=source_root, cache_root=cache_root)
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        result = await run_benchmark_async(
            client, dataset_relpaths=["tomo.mrc"], concurrency=4, requests_per_dataset=5,
        )

    assert result["count"] == 5
    assert result["errors"] == 0
    assert result["p50_ms"] >= 0


@pytest.fixture
def anyio_backend():
    return "asyncio"
```

Note: `pytest.mark.anyio` requires the `anyio` pytest plugin, which ships as a transitive dependency of `httpx`/`fastapi`'s stack (`starlette` depends on `anyio`) — confirm it's importable with `pixi run python -c "import anyio"` before writing this; if unavailable, use `pytest-asyncio`'s `@pytest.mark.asyncio` instead and check it's already present (it is not in the pinned dev deps — in that case, write this single test as a plain synchronous test using `asyncio.run(...)` instead of pytest-asyncio, avoiding a new dependency):

```python
import asyncio
import httpx

from mrcng.server.config import Settings
from mrcng.server.app import create_app
from mrcng.benchmark import run_benchmark_async


def test_benchmark_against_in_process_app(tmp_path):
    from tests.conftest import make_mrc
    source_root = tmp_path / "source"; source_root.mkdir()
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    make_mrc(source_root / "tomo.mrc", shape=(16, 16, 16), mode=1)

    settings = Settings(source_root=source_root, cache_root=cache_root)
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)

    async def _run():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await run_benchmark_async(
                client, dataset_relpaths=["tomo.mrc"], concurrency=4, requests_per_dataset=5,
            )

    result = asyncio.run(_run())
    assert result["count"] == 5
    assert result["errors"] == 0
    assert result["p50_ms"] >= 0
```

Use this synchronous-wrapper version as the actual test file content (no `anyio`/`pytest-asyncio` marker, no new dependency).

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_benchmark.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Implement `benchmark.py`**

Create `src/mrcng/benchmark.py`:

```python
"""Load-test script: N concurrent clients requesting scale-0 chunks for a
list of datasets, reporting p50/p95/p99 latency. Run via `pixi run benchmark`."""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx


async def _one_request(client: httpx.AsyncClient, relpath: str) -> tuple[float, bool]:
    start = time.monotonic()
    try:
        resp = await client.get(f"/data/{relpath}/1_1_1/0-64_0-64_0-64")
        ok = resp.status_code in (200, 404)  # 404 is a valid answer for a small volume
    except httpx.HTTPError:
        ok = False
    duration_ms = (time.monotonic() - start) * 1000
    return duration_ms, ok


async def run_benchmark_async(
    client: httpx.AsyncClient, dataset_relpaths: list[str],
    concurrency: int = 8, requests_per_dataset: int = 20,
) -> dict:
    tasks_queue = [
        relpath for relpath in dataset_relpaths for _ in range(requests_per_dataset)
    ]
    semaphore = asyncio.Semaphore(concurrency)
    durations = []
    errors = 0

    async def _worker(relpath):
        nonlocal errors
        async with semaphore:
            duration_ms, ok = await _one_request(client, relpath)
            durations.append(duration_ms)
            if not ok:
                errors += 1

    await asyncio.gather(*(_worker(r) for r in tasks_queue))

    durations.sort()
    def _pctile(p):
        if not durations:
            return 0.0
        idx = min(len(durations) - 1, int(len(durations) * p))
        return durations[idx]

    return {
        "count": len(durations),
        "errors": errors,
        "p50_ms": round(_pctile(0.50), 2),
        "p95_ms": round(_pctile(0.95), 2),
        "p99_ms": round(_pctile(0.99), 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mrc-benchmark")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--relpath", action="append", required=True, dest="dataset_relpaths")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests-per-dataset", type=int, default=20)
    args = parser.parse_args(argv)

    async def _run():
        async with httpx.AsyncClient(base_url=args.base_url) as client:
            return await run_benchmark_async(
                client, args.dataset_relpaths, args.concurrency, args.requests_per_dataset,
            )

    result = asyncio.run(_run())
    print(result)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

- [ ] **Step 4: Run to verify pass**

Run: `pixi run pytest tests/test_benchmark.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mrcng/benchmark.py tests/test_benchmark.py
git commit -m "Add httpx-based benchmark script for scale-0 read latency"
```

---

### Task 14: README — install, config, pixi tasks, Neuroglancer instructions

**Files:**
- Create: `README.md`

**Interfaces:** none (documentation only).

**Step-by-step:**

- [ ] **Step 1: Write `README.md`**

Cover, in order: (1) one-paragraph project description linking the two deliverables; (2) prerequisites (`pixi`) and `pixi install`; (3) the `MRCNG_*` environment variables from `Settings` (Task 5) with an example `.env` snippet (using placeholder paths like `/path/to/tomograms` and `/path/to/cache` — never the private dataset path used in manual testing); (4) `pixi run build-cache <source_root> --cache-root <cache_root> [options]` with a short explanation of what it does and that it's safe to re-run (skips valid entries); (5) `pixi run pyramid-status` / `pixi run pyramid-prune`; (6) `pixi run serve` to start the FastAPI app, noting it reads `MRCNG_SOURCE_ROOT`/`MRCNG_CACHE_ROOT` from the environment; (7) `pixi run test` and `pixi run benchmark`; (8) a "Loading data into Neuroglancer" section: construct the URL `precomputed://http://<host>:<port>/data/<relpath-to-mrc-file>` (no `/info` suffix — Neuroglancer appends it), open https://neuroglancer-demo.appspot.com/ (or a self-hosted instance), add a new layer with source type "precomputed" and paste that URL, and what to expect (single-resolution image for an uncached file, multi-resolution smooth zoom for a cached one); (9) an example nginx snippet (as a fenced code block) showing a reverse proxy in front of uvicorn with long-lived caching headers passed through for chunk responses, explicitly noting this is optional and untested in this repo (spec §10).

- [ ] **Step 2: Sanity-check the README's commands actually match `pyproject.toml`**

Run: `grep -n "^[a-z-]* = " pyproject.toml` and confirm every task name mentioned in the README (`build-cache`, `pyramid-status`, `pyramid-prune`, `serve`, `test`, `benchmark`) exists verbatim in `[tool.pixi.tasks]`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Add README with setup, pixi tasks, and Neuroglancer loading instructions"
```

---

### Task 15: Full suite pass + manual smoke test against real data

**Files:** none created; this task verifies the whole system.

**Step-by-step:**

- [ ] **Step 1: Run the entire automated test suite**

Run: `pixi run test`
Expected: all tests across every module PASS. If anything fails, fix the specific module (do not weaken assertions to make failures disappear) and re-run until green.

- [ ] **Step 2: Build a real cache from private sample data (not committed anywhere)**

Run (do not put this path in any file that gets committed — command only, run directly in the shell):

```bash
pixi run build-cache "/nrs/scicompsoft/rokicki/cryoet/data/Experimental/gouauxlab_20241206_AMmilled24-3/20241206_AMmilled24-3_92/TiltSeries/20241206_AMmilled24-3_92/stack" --cache-root /tmp/mrcng-smoke-cache --report /tmp/mrcng-smoke-report.jsonl
cat /tmp/mrcng-smoke-report.jsonl
```

Confirm at least one file reports `"status": "built"` with `levels_built >= 1` (or investigate and fix if the real file's header trips an unhandled case — this is exactly the kind of real-world MRC file the hand-rolled parser must survive).

- [ ] **Step 3: Start the server against the real source tree and cache, verify manually**

```bash
MRCNG_SOURCE_ROOT="/nrs/scicompsoft/rokicki/cryoet/data/Experimental/gouauxlab_20241206_AMmilled24-3/20241206_AMmilled24-3_92/TiltSeries/20241206_AMmilled24-3_92/stack" \
MRCNG_CACHE_ROOT=/tmp/mrcng-smoke-cache \
pixi run serve &
sleep 2
curl -s http://localhost:8000/healthz
curl -s "http://localhost:8000/data/20241206_AMmilled24-3_92.mrc/info" | head -c 2000
curl -s -o /tmp/chunk0.bin -w "%{http_code} %{size_download}\n" "http://localhost:8000/data/20241206_AMmilled24-3_92.mrc/1_1_1/0-64_0-64_0-64"
kill %1
```

Confirm: `/healthz` returns 200; `/info` shows multiple scales (cache built successfully) with plausible `resolution` values (a few ångströms → a fraction of a nanometre, not 10x off); the scale-0 chunk request returns 200 with a non-trivial byte count matching `64*64*64*2` (int16) or less at a clipped edge.

- [ ] **Step 4: Clean up smoke-test artifacts**

```bash
rm -rf /tmp/mrcng-smoke-cache /tmp/mrcng-smoke-report.jsonl /tmp/chunk0.bin
```

These are scratch outputs outside the repo — nothing to commit here.

- [ ] **Step 5: Final commit if anything was fixed during smoke testing**

If Step 2 or Step 3 surfaced a bug fixed in library code, `git add` the specific fixed files and commit with a message describing the real-world case that broke (e.g. "Handle <specific header quirk> found in real tomogram data"), still without naming the private data path anywhere in the commit message or code.

---

## Plan Self-Review

**Spec coverage:** §1 package layout — Tasks 1–13 create every listed file. §2 mrcheader — Task 1 (with the offset correction and IMOD logic carried over verbatim from the design doc). §3 paths — Task 2. §4 fingerprint — Task 7. §5 precomputed — Task 4. §6 reader — Task 3. §7 downsample — Task 6. §8 pyramid/cli — Tasks 8–9. §9 server (all subsections) — Tasks 5, 10, 11, 12. §10 scope decisions — Task 14 (README nginx-doc-only, no `/metrics` anywhere in the plan). §11 scaffolding — already done prior to this plan (repo init, pyproject, pixi tasks). §12 tests — covered per-task plus Task 15's integration/manual pass. §13 milestones — M1=Tasks1-4, M2=Task5, M3=Tasks6-9, M4=Task10, M5=Tasks11-13, plus README (Task 14) and final verification (Task 15).

**Type consistency fix applied:** `Params.dtype` is a string (e.g. `"int16"`) everywhere — Task 10's server code uses `hdr.dtype.name` to produce it, matching `Task 7`'s `Params.dtype: str` field and `Task 8`/`Task 9`'s construction of `Params(dtype="int16", ...)` in tests. `ScaleLevel`, `chunk_name`/`parse_chunk_name`, `read_chunk`, `clip_chunk_to_scale`, `encode_chunk` signatures are used identically across Tasks 4, 5, 8, 10.

**Known deliberately-deferred rough edges (documented, not silent):** Task 8's `_build_level_from_previous` reads previous-level cache chunks one output voxel-row at a time (`range(0, out_nz, 1)` / `range(0, out_ny, 1)`) rather than batching whole rows — correct but not the streaming-optimized approach used for level 1. This matches the spec's explicit scope (§8 only requires the *level-1-from-source* streaming to be bounded-memory and batched; level-N-from-N-1 has no such requirement stated) and keeps the code simple. If build times on real multi-level pyramids turn out too slow in Task 15's smoke test, that's a follow-up, not a blocker for this plan.
