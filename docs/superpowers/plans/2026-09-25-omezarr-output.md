# OME-Zarr 0.5 Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve every MRC as an OME-Zarr 0.5 (Zarr v3) multiscale image at `/omezarr/<relpath>` alongside precomputed at `/precomputed/<relpath>` (and the legacy `/data/`), with the pyramid build writing both layouts to disk.

**Architecture:** A new derivation module `omezarr.py` produces Zarr v3 / OME 0.5 metadata and pads edge chunks. The builder downsamples each level once and writes it to every selected format (`precomputed/` and/or `omezarr/` subdirectories of the cache entry); the fingerprint records which formats exist. The server gains an `/omezarr/` dispatcher that serves cached metadata and chunks verbatim and synthesises level 0 from the MRC, mirroring the precomputed path. No request-time format translation.

**Tech Stack:** Python 3.11, numpy, FastAPI, pydantic-settings, pytest; `zarr>=3` as a test-only dependency (pixi `test` feature).

**Spec:** `docs/superpowers/specs/2026-09-25-omezarr-output-design.md`

## Global Constraints

- OME-Zarr `0.5` on Zarr v3 only; metadata files are named `zarr.json`. No v2 files (`.zattrs`, `.zgroup`, `.zarray`) are ever written or served (404).
- Codec chain is exactly `[{"name": "bytes", "configuration": {"endian": "little"}}]`; no compressor.
- Chunk key encoding is `default` with separator `/`, so chunk files are `omezarr/<level>/c/<kz>/<ky>/<kx>`.
- Array axes are `["z", "y", "x"]`, all `type: "space"`, `unit: "nanometer"`. Scale values are ångström ÷ 10, identical to precomputed `info`.
- Downsampled level translation per axis is `(f - 1) / 2 * base_res_nm`; level 0 has no translation entry.
- Precomputed layout moves under `<entry>/precomputed/`; OME-Zarr under `<entry>/omezarr/`. `fingerprint.json` and `.lock` stay at the entry root.
- `fingerprint.SCHEMA_VERSION = 4`, `fingerprint.DERIVATION_VERSION = 2`. Fingerprint gains `"formats": [...]`.
- Format names are exactly `"precomputed"` and `"omezarr"`. Build default is both. `--formats` is one comma-separated option; `MRCNG_FORMATS` is its environment default.
- Level 0 is never written to disk in either format; the server always reads it from the MRC.
- The server never downsamples or writes on the request path. No new server setting.
- Run tests with `pixi run -e default pytest -q`.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

Inputs the spec implies but that no single requirement spells out. Each has a test pinned to the owning task.

1. `GET /omezarr/zarr.json` (empty relpath) must 404, not 500. → Task 5.
2. A mode-12 (float16) file must serve `data_type: "float32"` and a float32-wide padded level-0 chunk. → Task 6.
3. A chunk index far outside the grid (`c/0/0/999999`) must 404 at level 0 and at cached levels without touching the fingerprint's scale sizes. → Task 6.
4. A volume smaller than one chunk in every axis produces a single fully padded OME-Zarr chunk of exactly `prod(chunk_shape) * itemsize` bytes. → Task 3.
5. An image stack gets z factor 1 at every level, z translation 0, and keeps `nz` in every level's `shape`. → Task 1 (metadata) and Task 3 (build).

---

### Task 1: `omezarr.py` metadata and chunk helpers

**Files:**
- Create: `src/mrcng/omezarr.py`
- Test: `tests/test_omezarr.py`

**Interfaces:**
- Consumes: `mrcng.precomputed.ScaleLevel` (`key`, `size: (x, y, z)`, `factors: (fx, fy, fz)`), `MrcHeader.voxel_size_angstrom`, `.served_dtype`, `.is_image_stack`, `.voxel_size_is_default`.
- Produces:
  - `FORMATS: tuple[str, str] = ("precomputed", "omezarr")`
  - `chunk_rel_path(level: int, kz: int, ky: int, kx: int) -> str` → `"1/c/0/2/3"` (relative to `omezarr/`)
  - `chunk_region(size_xyz, chunk_size_xyz, kz, ky, kx) -> (x0, x1, y0, y1, z0, z1)`; raises `ValueError` when the index origin is outside the level.
  - `pad_chunk(arr: np.ndarray, chunk_shape_zyx) -> np.ndarray`; returns `arr` itself when already full-size.
  - `build_array_json(size_xyz, chunk_size_xyz, dtype: np.dtype) -> dict`
  - `build_group_json(hdr, scales: list[ScaleLevel], name: str) -> dict`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_omezarr.py
import numpy as np
import pytest

from mrcng.mrcheader import parse_header
from mrcng.precomputed import plan_scales
from mrcng import omezarr


def _hdr(path, is_image_stack=False):
    import os
    fd = os.open(str(path), os.O_RDONLY)
    try:
        st = os.stat(fd)
        return parse_header(fd, st.st_size, st.st_mtime_ns, is_image_stack=is_image_stack)
    finally:
        os.close(fd)


def test_chunk_rel_path_is_level_c_z_y_x():
    assert omezarr.chunk_rel_path(1, 0, 2, 3) == "1/c/0/2/3"


def test_chunk_region_clips_to_level_and_rejects_outside():
    # size (x=20, y=10, z=5), chunk (8, 8, 8): x index 2 covers [16, 20)
    assert omezarr.chunk_region((20, 10, 5), (8, 8, 8), 0, 1, 2) == (16, 20, 8, 10, 0, 5)
    with pytest.raises(ValueError):
        omezarr.chunk_region((20, 10, 5), (8, 8, 8), 1, 0, 0)  # z origin 8 >= 5
    with pytest.raises(ValueError):
        omezarr.chunk_region((20, 10, 5), (8, 8, 8), 0, 0, 999999)


def test_pad_chunk_returns_same_object_when_full_and_zero_pads_otherwise():
    full = np.ones((8, 8, 8), dtype="<i2")
    assert omezarr.pad_chunk(full, (8, 8, 8)) is full

    edge = np.ones((5, 8, 3), dtype="<i2")
    out = omezarr.pad_chunk(edge, (8, 8, 8))
    assert out.shape == (8, 8, 8) and out.dtype == edge.dtype and out.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(out[:5, :, :3], 1)
    assert out[5:].sum() == 0 and out[:, :, 3:].sum() == 0


@pytest.mark.parametrize("np_dtype,zarr_name", [
    ("int8", "int8"), ("uint8", "uint8"), ("int16", "int16"), ("uint16", "uint16"), ("float32", "float32"),
])
def test_array_json_reverses_to_zyx_and_names_dtype(np_dtype, zarr_name):
    doc = omezarr.build_array_json((20, 10, 5), (8, 8, 4), np.dtype(np_dtype))
    assert doc["zarr_format"] == 3 and doc["node_type"] == "array"
    assert doc["shape"] == [5, 10, 20]
    assert doc["chunk_grid"] == {"name": "regular", "configuration": {"chunk_shape": [4, 8, 8]}}
    assert doc["chunk_key_encoding"] == {"name": "default", "configuration": {"separator": "/"}}
    assert doc["data_type"] == zarr_name
    assert doc["fill_value"] == 0
    assert doc["codecs"] == [{"name": "bytes", "configuration": {"endian": "little"}}]
    assert doc["dimension_names"] == ["z", "y", "x"]


def test_group_json_scale_and_translation_per_level(make_mrc_file):
    # 20 Å voxels -> 2.0 nm; level 1 factor 2 -> scale 4.0, translation 0.5 * 2.0 = 1.0
    path = make_mrc_file(shape=(64, 64, 64), mode=1, voxel_size_angstrom=(20.0, 20.0, 20.0))
    hdr = _hdr(path)
    scales = plan_scales((64, 64, 64), min_axis_size=16, max_levels=3)
    assert [s.key for s in scales] == ["1_1_1", "2_2_2", "4_4_4"]

    doc = omezarr.build_group_json(hdr, scales, name="tomo.mrc")
    assert doc["zarr_format"] == 3 and doc["node_type"] == "group"
    ome = doc["attributes"]["ome"]
    assert ome["version"] == "0.5"
    ms = ome["multiscales"][0]
    assert ms["name"] == "tomo.mrc"
    assert ms["axes"] == [
        {"name": "z", "type": "space", "unit": "nanometer"},
        {"name": "y", "type": "space", "unit": "nanometer"},
        {"name": "x", "type": "space", "unit": "nanometer"},
    ]
    ds = ms["datasets"]
    assert [d["path"] for d in ds] == ["0", "1", "2"]
    assert ds[0]["coordinateTransformations"] == [{"type": "scale", "scale": [2.0, 2.0, 2.0]}]
    assert ds[1]["coordinateTransformations"] == [
        {"type": "scale", "scale": [4.0, 4.0, 4.0]},
        {"type": "translation", "translation": [1.0, 1.0, 1.0]},
    ]
    assert ds[2]["coordinateTransformations"] == [
        {"type": "scale", "scale": [8.0, 8.0, 8.0]},
        {"type": "translation", "translation": [3.0, 3.0, 3.0]},
    ]
    assert doc["attributes"]["mrcng"] == {"voxel_size_is_default": False, "is_image_stack": False}


def test_group_json_image_stack_keeps_z_unbinned_and_untranslated(make_mrc_file):
    path = make_mrc_file(shape=(64, 64, 8), mode=1, voxel_size_angstrom=(20.0, 20.0, 20.0))
    hdr = _hdr(path, is_image_stack=True)
    scales = plan_scales((64, 64, 8), min_axis_size=16, max_levels=2, downsample_z=False)
    assert scales[1].key == "2_2_1"

    doc = omezarr.build_group_json(hdr, scales, name="stack.mrc")
    ds = doc["attributes"]["ome"]["multiscales"][0]["datasets"]
    # z is 1 nm per slice at level 0, unchanged at level 1, and never shifted
    assert ds[0]["coordinateTransformations"][0]["scale"] == [1.0, 2.0, 2.0]
    assert ds[1]["coordinateTransformations"][0]["scale"] == [1.0, 4.0, 4.0]
    assert ds[1]["coordinateTransformations"][1]["translation"] == [0.0, 1.0, 1.0]
    assert doc["attributes"]["mrcng"]["is_image_stack"] is True
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e default pytest -q tests/test_omezarr.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'mrcng.omezarr'`

- [ ] **Step 3: Implement the module**

```python
# src/mrcng/omezarr.py
"""OME-Zarr 0.5 (Zarr v3) metadata, chunk addressing and edge padding.

Derivation module: everything here lands in cache files, so a change to it
needs a fingerprint.DERIVATION_VERSION bump (see CLAUDE.md).

Array order is (z, y, x) on the wire and on disk, matching precomputed raw
chunks byte for byte in the interior. The one difference is the edge: Zarr
chunks are always chunk_shape-sized, padded with fill_value, where precomputed
clips them. Level 0 is never on disk; omezarr/ is a serving cache, not a
standalone store.
"""
from __future__ import annotations

import numpy as np

FORMATS = ("precomputed", "omezarr")

_CODECS = [{"name": "bytes", "configuration": {"endian": "little"}}]


def chunk_rel_path(level: int, kz: int, ky: int, kx: int) -> str:
    """Path of one chunk relative to the omezarr/ directory ("default" key
    encoding, "/" separator)."""
    return f"{level}/c/{kz}/{ky}/{kx}"


def chunk_region(size_xyz, chunk_size_xyz, kz: int, ky: int, kx: int):
    """(x0, x1, y0, y1, z0, z1) of the *data* a grid index covers, clipped to
    the level. Raises ValueError when the chunk origin is outside the level."""
    sx, sy, sz = size_xyz
    cx, cy, cz = chunk_size_xyz
    x0, y0, z0 = kx * cx, ky * cy, kz * cz
    if x0 >= sx or y0 >= sy or z0 >= sz:
        raise ValueError(f"chunk index {(kz, ky, kx)} outside level of size {tuple(size_xyz)}")
    return x0, min(x0 + cx, sx), y0, min(y0 + cy, sy), z0, min(z0 + cz, sz)


def pad_chunk(arr: np.ndarray, chunk_shape_zyx) -> np.ndarray:
    """Zero-pad a clipped (z, y, x) block up to chunk_shape. Interior chunks
    come back as the same object, no copy."""
    if arr.shape == tuple(chunk_shape_zyx):
        return arr
    out = np.zeros(tuple(chunk_shape_zyx), dtype=arr.dtype)
    out[: arr.shape[0], : arr.shape[1], : arr.shape[2]] = arr
    return out


def build_array_json(size_xyz, chunk_size_xyz, dtype: np.dtype) -> dict:
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [int(s) for s in reversed(tuple(size_xyz))],
        # numpy and Zarr v3 spell int8/uint8/int16/uint16/float32 identically.
        "data_type": np.dtype(dtype).name,
        "chunk_grid": {"name": "regular",
                       "configuration": {"chunk_shape": [int(c) for c in reversed(tuple(chunk_size_xyz))]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": _CODECS,
        "dimension_names": ["z", "y", "x"],
        "attributes": {},
    }


def build_group_json(hdr, scales, name: str) -> dict:
    # Same numbers precomputed info uses (angstrom / 10), so both endpoints
    # show the same scale bar. Index order is (x, y, z) in hdr and scales,
    # (z, y, x) in OME.
    base_nm = [a / 10.0 for a in hdr.voxel_size_angstrom]
    datasets = []
    for i, lvl in enumerate(scales):
        fx, fy, fz = lvl.factors
        transforms = [{"type": "scale", "scale": [base_nm[2] * fz, base_nm[1] * fy, base_nm[0] * fx]}]
        if i > 0:
            # NGFF indexes the *centre* of a sample. A mean over f source voxels
            # is centred (f-1)/2 source voxels in from the origin. Precomputed
            # needs no such offset: Neuroglancer's precomputed model is
            # corner-aligned.
            transforms.append({"type": "translation", "translation": [
                (fz - 1) / 2 * base_nm[2], (fy - 1) / 2 * base_nm[1], (fx - 1) / 2 * base_nm[0],
            ]})
        datasets.append({"path": str(i), "coordinateTransformations": transforms})
    return {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5",
                "multiscales": [{
                    "name": name,
                    "axes": [{"name": ax, "type": "space", "unit": "nanometer"} for ax in ("z", "y", "x")],
                    "datasets": datasets,
                }],
            },
            # "ome" is reserved by the spec; the non-spec flags precomputed puts
            # at the top level of info live under our own key here.
            "mrcng": {"voxel_size_is_default": hdr.voxel_size_is_default,
                      "is_image_stack": hdr.is_image_stack},
        },
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `pixi run -e default pytest -q tests/test_omezarr.py`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mrcng/omezarr.py tests/test_omezarr.py
git commit -m "feat: omezarr module with OME-Zarr 0.5 metadata and chunk helpers

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Fingerprint records formats; schema and derivation bumps

**Files:**
- Modify: `src/mrcng/fingerprint.py` (constants at lines 44 and 71, module-list comment at lines 48-53, `build_fingerprint` at lines 102-131)
- Test: `tests/test_fingerprint.py`

**Interfaces:**
- Produces: `build_fingerprint(fd, hdr, relpath, params, scales, generator_version, build_duration_s, *, formats: tuple[str, ...]) -> dict` with a new `"formats": list[str]` key; `SCHEMA_VERSION == 4`; `DERIVATION_VERSION == 2`.

- [ ] **Step 1: Update existing call sites in the tests and add new tests**

Every `build_fingerprint(...)` call in `tests/test_fingerprint.py` gains `formats=("precomputed",)`:

```bash
sed -i 's/build_fingerprint(fd, hdr, "tomo.mrc", _params(), {"2_2_2": (4, 4, 4)}, "v0.1.0", 1.0)/build_fingerprint(fd, hdr, "tomo.mrc", _params(), {"2_2_2": (4, 4, 4)}, "v0.1.0", 1.0, formats=("precomputed",))/' tests/test_fingerprint.py
sed -i 's/generator_version="test", build_duration_s=0.0)/generator_version="test", build_duration_s=0.0, formats=("precomputed",))/' tests/test_fingerprint.py
sed -i 's/generator_version="mrc-pyramid 0.1.0",/generator_version="mrc-pyramid 0.1.0", formats=("precomputed", "omezarr"),/' tests/test_fingerprint.py
sed -i 's/assert fp\["schema_version"\] == 3/assert fp["schema_version"] == 4/' tests/test_fingerprint.py
```

Then check nothing was missed: `grep -n "build_fingerprint(" tests/test_fingerprint.py` — every call must contain `formats=`. Fix any stragglers by hand.

Append:

```python
def test_fingerprint_records_formats_and_schema_3_is_incompatible(make_mrc_file):
    import os
    from mrcng.fingerprint import (
        Params, Validity, build_fingerprint, validate, SCHEMA_VERSION, DERIVATION_VERSION,
    )
    from mrcng.mrcheader import parse_header

    assert SCHEMA_VERSION == 4 and DERIVATION_VERSION == 2

    path = make_mrc_file(shape=(8, 8, 8), mode=1)
    fd = os.open(str(path), os.O_RDONLY)
    try:
        st = os.stat(fd)
        hdr = parse_header(fd, st.st_size, st.st_mtime_ns)
        params = Params(chunk_size=(8, 8, 8), downsample="mean", min_axis_size=8,
                        max_levels=3, dtype="int16", encoding="raw")
        fp = build_fingerprint(fd, hdr, "t.mrc", params, scales={}, generator_version="test",
                               build_duration_s=0.0, formats=("omezarr",))
        assert fp["formats"] == ["omezarr"]
        assert validate(fp, hdr, fd, params) == Validity.VALID

        # A fingerprint written by the previous layout (no formats key, schema 3)
        # must never validate: its chunks sit at the entry root, not under precomputed/.
        fp["schema_version"] = 3
        del fp["formats"]
        assert validate(fp, hdr, fd, params) == Validity.INCOMPATIBLE
    finally:
        os.close(fd)
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e default pytest -q tests/test_fingerprint.py`
Expected: FAIL with `TypeError: build_fingerprint() got an unexpected keyword argument 'formats'`

- [ ] **Step 3: Implement**

In `src/mrcng/fingerprint.py`:

```python
SCHEMA_VERSION = 4   # v3 -> v4: added "formats"
```

```python
DERIVATION_VERSION = 2   # v1 -> v2: layouts moved under precomputed/ and omezarr/
```

In the module-list comment (the paragraph beginning "It tracks the modules that decide what a build writes"), add `omezarr.py (build_group_json, build_array_json, pad_chunk, chunk_rel_path)` to the list after `precomputed.py (...)`.

`build_fingerprint` signature and body:

```python
def build_fingerprint(fd: int, hdr, relpath: str, params: Params,
                       scales: dict[str, tuple[int, int, int]],
                       generator_version: str, build_duration_s: float,
                       *, formats: tuple[str, ...]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": generator_version,
        "derivation_version": DERIVATION_VERSION,
        "is_image_stack": hdr.is_image_stack,
        # Exactly the layouts this build wrote (precomputed/, omezarr/). A
        # request for a format not listed here serves single-resolution, the
        # same as no cache. Not in Params: it is *what* was built, not *how*.
        "formats": list(formats),
        "source_relpath": relpath,
        ...  # remaining keys unchanged
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `pixi run -e default pytest -q tests/test_fingerprint.py`
Expected: all PASS. (Other test files now fail because `pyramid.py` does not pass `formats`; Task 3 fixes them.)

- [ ] **Step 5: Commit**

```bash
git add src/mrcng/fingerprint.py tests/test_fingerprint.py
git commit -m "feat: fingerprint records formats; schema 4, derivation 2

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Builder writes both layouts

**Files:**
- Modify: `src/mrcng/pyramid.py` (`BuildResult`, `_write_chunk`, `_build_level_from_source`, `_read_prev_level_region`, `_build_level_from_previous`, `build_one`)
- Modify: `src/mrcng/server/app.py:154` (`(cache_dir / "info")`) and `:268` (`chunk_path = cache_dir / scale_key / chunk_str`) — layout move only, so the server keeps working
- Modify: `src/mrcng/cli.py:157-172` (`_build_one_record` record gains `formats`)
- Test: `tests/test_pyramid.py`, `tests/test_server_cached.py:52`, `tests/test_server_observability.py:99`

**Interfaces:**
- Consumes: `omezarr.FORMATS`, `omezarr.chunk_rel_path`, `omezarr.pad_chunk`, `omezarr.build_group_json`, `omezarr.build_array_json`; `build_fingerprint(..., formats=)`.
- Produces: `build_one(source_root, cache_root, relpath, params, force=False, max_block_bytes=..., assume_mode0=None, stack_globs=(), volume_globs=(), formats: tuple[str, ...] = FORMATS) -> BuildResult`; `BuildResult.formats: tuple[str, ...]`. On disk: `<entry>/precomputed/info`, `<entry>/precomputed/<key>/<chunk>`, `<entry>/omezarr/zarr.json`, `<entry>/omezarr/<i>/zarr.json`, `<entry>/omezarr/<i>/c/<kz>/<ky>/<kx>`.

- [ ] **Step 1: Move existing layout references in the tests**

```bash
sed -i 's|cache_dir / "2_2_2" / name|cache_dir / "precomputed" / "2_2_2" / name|; s|(cache_dir / "info")|(cache_dir / "precomputed" / "info")|; s|(cache_dir / "2_2_1" / "0-64_0-64_0-8")|(cache_dir / "precomputed" / "2_2_1" / "0-64_0-64_0-8")|' tests/test_pyramid.py
sed -i 's|(cache_dir / "2_2_2" / "0-8_0-8_0-8")|(cache_dir / "precomputed" / "2_2_2" / "0-8_0-8_0-8")|; s|(cache_dir / "4_4_4")|(cache_dir / "precomputed" / "4_4_4")|g' tests/test_server_cached.py tests/test_server_observability.py
grep -rn 'cache_dir / "2_2\|cache_dir / "info"\|cache_dir / "4_4' tests/ | grep -v precomputed   # must print nothing
```

- [ ] **Step 2: Add the new builder tests**

Append to `tests/test_pyramid.py`:

```python
def _read_zarr_chunk(cache_dir, level, kz, ky, kx, chunk_shape_zyx, dtype="<i2"):
    from mrcng.omezarr import chunk_rel_path
    raw = (cache_dir / "omezarr" / chunk_rel_path(level, kz, ky, kx)).read_bytes()
    return np.frombuffer(raw, dtype=dtype).reshape(chunk_shape_zyx)


def test_default_build_writes_both_layouts_and_records_formats(source_and_cache):
    source_root, cache_root, relpath = source_and_cache
    result = build_one(source_root, cache_root, relpath, _params())
    assert result.status == BuildStatus.BUILT
    assert result.formats == ("precomputed", "omezarr")

    cache_dir = cache_dir_for(cache_root, dataset_id(relpath))
    assert read_fingerprint(cache_dir)["formats"] == ["precomputed", "omezarr"]
    assert (cache_dir / "precomputed" / "info").is_file()
    assert (cache_dir / "omezarr" / "zarr.json").is_file()
    # metadata for every level including 0 (which has no chunks), chunks for 1..N
    assert (cache_dir / "omezarr" / "0" / "zarr.json").is_file()
    assert not (cache_dir / "omezarr" / "0" / "c").exists()
    assert (cache_dir / "omezarr" / "1" / "zarr.json").is_file()
    assert (cache_dir / "omezarr" / "1" / "c" / "0" / "0" / "0").is_file()
    # nothing from the old layout at the entry root
    assert not (cache_dir / "info").exists() and not (cache_dir / "2_2_2").exists()


def test_omezarr_chunks_equal_precomputed_chunks_after_padding(tmp_path, make_mrc_file):
    """Interior chunks are byte-identical; edge chunks equal the clipped
    precomputed chunk zero-padded to chunk_shape."""
    from mrcng.precomputed import chunk_name
    from mrcng.omezarr import pad_chunk

    source_root = tmp_path / "source"; source_root.mkdir()
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    # 36x20x12 -> level 1 is 18x10x6 with chunk 8: edges on every axis
    make_mrc_file(name="source/t.mrc", shape=(36, 20, 12), mode=1,
                  fill=lambda zz, yy, xx: (xx + 1000 * yy + 1_000_000 * zz) % 30000)
    build_one(source_root, cache_root, "t.mrc", _params())
    cache_dir = cache_dir_for(cache_root, dataset_id("t.mrc"))
    fp = read_fingerprint(cache_dir)
    assert fp["scales"]["2_2_2"] == [18, 10, 6]

    checked = 0
    for kz in range(1):
        for ky in range(2):
            for kx in range(3):
                x0, y0, z0 = kx * 8, ky * 8, kz * 8
                x1, y1, z1 = min(x0 + 8, 18), min(y0 + 8, 10), min(z0 + 8, 6)
                pre = np.frombuffer(
                    (cache_dir / "precomputed" / "2_2_2" / chunk_name(x0, x1, y0, y1, z0, z1)).read_bytes(),
                    dtype="<i2").reshape(z1 - z0, y1 - y0, x1 - x0)
                zarr_chunk = _read_zarr_chunk(cache_dir, 1, kz, ky, kx, (8, 8, 8))
                np.testing.assert_array_equal(zarr_chunk, pad_chunk(pre, (8, 8, 8)))
                checked += 1
    assert checked == 6


def test_omezarr_only_build_matches_both_formats_build(tmp_path, make_mrc_file):
    """Level 2 is downsampled from level 1's *cache files*. With only omezarr
    selected those files are padded, so the reader must slice the padding off
    or level 2 averages zeros into its edge voxels."""
    source_root = tmp_path / "source"; source_root.mkdir()
    make_mrc_file(name="source/t.mrc", shape=(36, 20, 12), mode=1,
                  fill=lambda zz, yy, xx: (xx + 1000 * yy + 1_000_000 * zz) % 30000)

    both_root = tmp_path / "both"; both_root.mkdir()
    only_root = tmp_path / "only"; only_root.mkdir()
    build_one(source_root, both_root, "t.mrc", _params())
    result = build_one(source_root, only_root, "t.mrc", _params(), formats=("omezarr",))
    assert result.formats == ("omezarr",)

    both_dir = cache_dir_for(both_root, dataset_id("t.mrc"))
    only_dir = cache_dir_for(only_root, dataset_id("t.mrc"))
    assert not (only_dir / "precomputed").exists()
    assert read_fingerprint(only_dir)["formats"] == ["omezarr"]

    both_tree = {p.relative_to(both_dir / "omezarr").as_posix(): p.read_bytes()
                 for p in sorted((both_dir / "omezarr").rglob("*")) if p.is_file()}
    only_tree = {p.relative_to(only_dir / "omezarr").as_posix(): p.read_bytes()
                 for p in sorted((only_dir / "omezarr").rglob("*")) if p.is_file()}
    assert any(k.startswith("2/c/") for k in both_tree)  # a level-2 actually exists
    assert only_tree == both_tree


def test_rebuild_with_different_formats_replaces_the_entry(source_and_cache):
    source_root, cache_root, relpath = source_and_cache
    build_one(source_root, cache_root, relpath, _params())
    cache_dir = cache_dir_for(cache_root, dataset_id(relpath))

    # same formats -> skipped; a different set is not "valid" for this request
    assert build_one(source_root, cache_root, relpath, _params()).status == BuildStatus.SKIPPED_VALID
    result = build_one(source_root, cache_root, relpath, _params(), formats=("omezarr",))
    assert result.status == BuildStatus.BUILT
    assert not (cache_dir / "precomputed").exists()
    assert read_fingerprint(cache_dir)["formats"] == ["omezarr"]

    result = build_one(source_root, cache_root, relpath, _params(), formats=("precomputed",))
    assert result.status == BuildStatus.BUILT
    assert not (cache_dir / "omezarr").exists()
    assert (cache_dir / "precomputed" / "info").is_file()


def test_sub_chunk_volume_produces_one_fully_padded_chunk(tmp_path, make_mrc_file):
    source_root = tmp_path / "source"; source_root.mkdir()
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    # 10x12x14 at chunk 8 and min_axis_size 4 -> level 1 is 5x6x7, a single all-edge chunk
    make_mrc_file(name="source/small.mrc", shape=(10, 12, 14), mode=1,
                  fill=lambda zz, yy, xx: 7)
    build_one(source_root, cache_root, "small.mrc", _params(min_axis_size=4, max_levels=2))
    cache_dir = cache_dir_for(cache_root, dataset_id("small.mrc"))
    chunk = _read_zarr_chunk(cache_dir, 1, 0, 0, 0, (8, 8, 8))
    assert chunk.nbytes == 8 * 8 * 8 * 2
    np.testing.assert_array_equal(chunk[:7, :6, :5], 7)
    assert chunk[7:].sum() == 0 and chunk[:, 6:].sum() == 0 and chunk[:, :, 5:].sum() == 0


def test_image_stack_omezarr_keeps_every_slice(tmp_path, make_mrc_file):
    import json as _json
    source_root = tmp_path / "source"; source_root.mkdir()
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    make_mrc_file(name="source/stack.mrc", shape=(256, 256, 8), mode=1,
                  fill=lambda zz, yy, xx: (zz + 1) * 100)
    build_one(source_root, cache_root, "stack.mrc",
              _params(chunk_size=(64, 64, 64), min_axis_size=32), stack_globs=("*stack.mrc",))
    cache_dir = cache_dir_for(cache_root, dataset_id("stack.mrc"))

    arr1 = _json.loads((cache_dir / "omezarr" / "1" / "zarr.json").read_text())
    assert arr1["shape"] == [8, 128, 128]
    group = _json.loads((cache_dir / "omezarr" / "zarr.json").read_text())
    ds1 = group["attributes"]["ome"]["multiscales"][0]["datasets"][1]["coordinateTransformations"]
    assert ds1[0]["scale"][0] == 1.0 and ds1[1]["translation"][0] == 0.0

    chunk = _read_zarr_chunk(cache_dir, 1, 0, 0, 0, (64, 64, 64))
    assert [int(chunk[z, 0, 0]) for z in range(8)] == [(z + 1) * 100 for z in range(8)]
    assert chunk[8:].sum() == 0  # padded z
```

- [ ] **Step 3: Run to verify failure**

Run: `pixi run -e default pytest -q tests/test_pyramid.py`
Expected: FAIL. Existing tests fail with `TypeError: build_fingerprint() missing 1 required keyword-only argument: 'formats'`; new tests fail on `formats` / missing `precomputed/`.

- [ ] **Step 4: Implement in `pyramid.py`**

Imports:

```python
from mrcng import omezarr
from mrcng.omezarr import FORMATS
```

`BuildResult` gains a field:

```python
    voxel_size_is_default: bool = False
    formats: tuple[str, ...] = ()
```

Replace `_write_chunk` with:

```python
def _write_chunks(cache_dir: Path, formats, level: int, scale_key: str, chunk_size,
                  x0: int, x1: int, y0: int, y1: int, z0: int, z1: int, arr: np.ndarray) -> int:
    """Write one downsampled block to every selected layout. Downsampling
    happened once, upstream; this is only the write."""
    arr = np.ascontiguousarray(arr)
    written = 0
    if "precomputed" in formats:
        scale_dir = cache_dir / "precomputed" / scale_key
        scale_dir.mkdir(parents=True, exist_ok=True)
        body = encode_chunk(arr)
        (scale_dir / chunk_name(x0, x1, y0, y1, z0, z1)).write_bytes(body)
        written += len(body)
    if "omezarr" in formats:
        cx, cy, cz = chunk_size
        path = cache_dir / "omezarr" / omezarr.chunk_rel_path(level, z0 // cz, y0 // cy, x0 // cx)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = encode_chunk(np.ascontiguousarray(omezarr.pad_chunk(arr, (cz, cy, cx))))
        path.write_bytes(body)
        written += len(body)
    return written
```

`_build_level_from_source` gains `formats` (positional, after `chunk_size`) and its inner write becomes:

```python
def _build_level_from_source(fd, hdr, cache_dir: Path, level0, level1, chunk_size, formats,
                              max_block_bytes: int = DEFAULT_MAX_BLOCK_BYTES) -> int:
    ...
                for x0 in range(px0, px1, cx):
                    x1 = min(x0 + cx, sx)
                    cache_bytes += _write_chunks(
                        cache_dir, formats, 1, level1.key, chunk_size,
                        x0, x1, y0, y1, z0, z1, band[:, :, x0 - px0:x1 - px0],
                    )
```

Add a chunk reader that hides which layout holds the previous level:

```python
def _read_prev_chunk(cache_dir: Path, formats, level: int, scale, chunk_size, dtype,
                     bx0: int, bx1: int, by0: int, by1: int, bz0: int, bz1: int) -> np.ndarray:
    """One whole chunk of an already-built level as a (z, y, x) array of its
    *clipped* extent. Prefers precomputed (already clipped); an omezarr-only
    build reads the padded chunk and slices the padding off."""
    shape = (bz1 - bz0, by1 - by0, bx1 - bx0)
    if "precomputed" in formats:
        raw = (cache_dir / "precomputed" / scale.key / chunk_name(bx0, bx1, by0, by1, bz0, bz1)).read_bytes()
        return np.frombuffer(raw, dtype=dtype).reshape(shape)
    cx, cy, cz = chunk_size
    raw = (cache_dir / "omezarr" / omezarr.chunk_rel_path(level, bz0 // cz, by0 // cy, bx0 // cx)).read_bytes()
    return np.frombuffer(raw, dtype=dtype).reshape(cz, cy, cx)[: shape[0], : shape[1], : shape[2]]
```

`_read_prev_level_region` gains `formats, level` and replaces its two `raw = ...; block = np.frombuffer(...)` lines:

```python
def _read_prev_level_region(cache_dir: Path, formats, level: int, scale, chunk_size, dtype,
                             x0: int, x1: int, y0: int, y1: int, z0: int, z1: int) -> np.ndarray:
    ...
                block = _read_prev_chunk(cache_dir, formats, level, scale, chunk_size, dtype,
                                         block_x0, block_x1, block_y0, block_y1, block_z0, block_z1)
                out[...] = block[...]   # unchanged slicing
```

`_build_level_from_previous` gains `formats, level` (the index of `next_scale`):

```python
def _build_level_from_previous(cache_dir: Path, formats, level: int, prev_scale, next_scale,
                               chunk_size, dtype) -> int:
    ...
        block = _read_prev_level_region(
            cache_dir, formats, level - 1, prev_scale, chunk_size, dtype,
            src_x0, src_x1, src_y0, src_y1, src_z0, src_z1,
        )
        downsampled = block_mean(block, (fz, fy, fx))
        cache_bytes += _write_chunks(cache_dir, formats, level, next_scale.key, chunk_size,
                                     x0, x1, y0, y1, z0, z1, downsampled)
```

`build_one`:

```python
def build_one(source_root, cache_root, relpath: str, params: Params, force: bool = False,
              max_block_bytes: int = DEFAULT_MAX_BLOCK_BYTES,
              assume_mode0: str | None = None,
              stack_globs=(), volume_globs=(),
              formats: tuple[str, ...] = FORMATS) -> BuildResult:
    formats = tuple(formats)
    unknown = set(formats) - set(FORMATS)
    if unknown or not formats:
        raise ValueError(f"formats must be a non-empty subset of {FORMATS}, got {formats!r}")
    ...
        existing = read_fingerprint(cache_dir)
        if (existing is not None and not force
                and validate(existing, hdr, fd, params) == Validity.VALID
                # Formats are replaced, never merged: the entry must hold
                # exactly what this run asks for or it is rebuilt from scratch.
                and set(existing.get("formats", ())) == set(formats)):
            return BuildResult(relpath, ds_id, BuildStatus.SKIPPED_VALID, source_bytes=hdr.file_size,
                               voxel_size_is_default=hdr.voxel_size_is_default,
                               formats=tuple(existing["formats"]))
```

The level loop and metadata writes:

```python
            if len(scales) > 1:
                cache_bytes += _build_level_from_source(
                    fd, hdr, cache_dir, scales[0], scales[1], params.chunk_size, formats, max_block_bytes,
                )
                levels_built += 1
                for i in range(2, len(scales)):
                    cache_bytes += _build_level_from_previous(
                        cache_dir, formats, i, scales[i - 1], scales[i], params.chunk_size, hdr.served_dtype,
                    )
                    levels_built += 1

            if "precomputed" in formats:
                (cache_dir / "precomputed").mkdir(exist_ok=True)
                info = build_info(hdr, scales, params.chunk_size, params.encoding)
                (cache_dir / "precomputed" / "info").write_text(json.dumps(info))
            if "omezarr" in formats:
                zroot = cache_dir / "omezarr"
                zroot.mkdir(exist_ok=True)
                (zroot / "zarr.json").write_text(json.dumps(
                    omezarr.build_group_json(hdr, scales, name=relpath.rsplit("/", 1)[-1])))
                for i, lvl in enumerate(scales):  # level 0 too: metadata only, no chunks
                    (zroot / str(i)).mkdir(exist_ok=True)
                    (zroot / str(i) / "zarr.json").write_text(json.dumps(
                        omezarr.build_array_json(lvl.size, params.chunk_size, hdr.served_dtype)))

            _fsync_tree(cache_dir)

            fp = build_fingerprint(
                fd, hdr, relpath, params,
                scales={s.key: s.size for s in scales[1:]},
                generator_version=GENERATOR_VERSION,
                build_duration_s=time.monotonic() - start,
                formats=formats,
            )
            write_fingerprint(cache_dir, fp)

            return BuildResult(
                relpath, ds_id, BuildStatus.BUILT,
                source_bytes=hdr.file_size, cache_bytes=cache_bytes,
                levels_built=levels_built, duration_s=time.monotonic() - start,
                voxel_size_is_default=hdr.voxel_size_is_default,
                formats=formats,
            )
```

The existing "drop every existing scale dir" loop (`for child in cache_dir.iterdir(): if child.is_dir(): shutil.rmtree(child)`) already removes `precomputed/` and `omezarr/`; update its comment to say it drops every layout so a rebuild with fewer formats leaves nothing behind.

- [ ] **Step 5: Move the two server layout references (no behaviour change yet)**

In `src/mrcng/server/app.py`:

```python
                    body = (cache_dir / "precomputed" / "info").read_bytes()
```

```python
    chunk_path = cache_dir / "precomputed" / scale_key / chunk_str
```

In `src/mrcng/cli.py` `_build_one_record`, add to the returned dict:

```python
            "formats": list(result.formats),
```

- [ ] **Step 6: Run the whole suite**

Run: `pixi run -e default pytest -q`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/mrcng/pyramid.py src/mrcng/server/app.py src/mrcng/cli.py tests/
git commit -m "feat: build writes precomputed/ and omezarr/ layouts per selected formats

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: CLI `--formats` for build and status

**Files:**
- Modify: `src/mrcng/cli.py` (`_build_command`, `_status_command`, `main` parser setup, `_build_one_record` task tuple)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `build_one(..., formats=)`, `omezarr.FORMATS`, `fp["formats"]`.
- Produces: `mrc-pyramid build --formats precomputed,omezarr` (default from `MRCNG_FORMATS`, else both); `mrc-pyramid status --formats ...` printing `<relpath>: <validity> [<f1>,<f2>]`, with `incomplete` replacing `valid` when a requested format is missing.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
def test_build_formats_option_selects_layouts(tmp_path, make_mrc_file):
    source_root = tmp_path / "source"; source_root.mkdir()
    make_mrc_file(name="source/a.mrc", shape=(16, 16, 16), mode=1)
    cache_root = tmp_path / "cache"

    rc = main(["build", "--source-root", str(source_root), "--cache-root", str(cache_root),
               "--chunk-size", "8,8,8", "--formats", "omezarr"])
    assert rc == 0
    from mrcng.paths import dataset_id, cache_dir_for
    cache_dir = cache_dir_for(cache_root, dataset_id("a.mrc"))
    assert (cache_dir / "omezarr" / "zarr.json").is_file()
    assert not (cache_dir / "precomputed").exists()
    assert read_fingerprint(cache_dir)["formats"] == ["omezarr"]


def test_build_formats_default_comes_from_env(tmp_path, make_mrc_file, monkeypatch):
    monkeypatch.setenv("MRCNG_FORMATS", "precomputed")
    source_root = tmp_path / "source"; source_root.mkdir()
    make_mrc_file(name="source/a.mrc", shape=(16, 16, 16), mode=1)
    cache_root = tmp_path / "cache"
    assert main(["build", "--source-root", str(source_root), "--cache-root", str(cache_root),
                 "--chunk-size", "8,8,8"]) == 0
    from mrcng.paths import dataset_id, cache_dir_for
    assert read_fingerprint(cache_dir_for(cache_root, dataset_id("a.mrc")))["formats"] == ["precomputed"]


@pytest.mark.parametrize("bad", ["", "zarr2", "precomputed,,omezarr", "precomputed,n5"])
def test_build_rejects_unknown_or_empty_formats(tmp_path, bad):
    with pytest.raises(SystemExit):
        main(["build", "--source-root", str(tmp_path), "--cache-root", str(tmp_path / "c"),
              "--formats", bad])


def test_status_prints_formats_and_incomplete(tmp_path, make_mrc_file, capsys):
    source_root = tmp_path / "source"; source_root.mkdir()
    make_mrc_file(name="source/a.mrc", shape=(16, 16, 16), mode=1)
    cache_root = tmp_path / "cache"
    main(["build", "--source-root", str(source_root), "--cache-root", str(cache_root),
          "--chunk-size", "8,8,8", "--formats", "precomputed"])

    main(["status", str(source_root), "--cache-root", str(cache_root), "--chunk-size", "8,8,8",
          "--formats", "precomputed"])
    assert "a.mrc: valid [precomputed]" in capsys.readouterr().out

    main(["status", str(source_root), "--cache-root", str(cache_root), "--chunk-size", "8,8,8"])
    assert "a.mrc: incomplete [precomputed]" in capsys.readouterr().out
```

Make sure `tests/test_cli.py` imports `pytest` and `read_fingerprint` at the top (`from mrcng.fingerprint import read_fingerprint`); add them if absent.

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e default pytest -q tests/test_cli.py`
Expected: FAIL with `unrecognized arguments: --formats`

- [ ] **Step 3: Implement**

In `src/mrcng/cli.py`:

```python
from mrcng.omezarr import FORMATS


def _parse_formats(s: str) -> tuple[str, ...]:
    """precomputed,omezarr -> ("precomputed", "omezarr"). One option, not a
    repeatable one, so passing it once *replaces* the default instead of
    appending to it."""
    parts = tuple(p.strip() for p in s.split(","))
    if not parts or any(p not in FORMATS for p in parts):
        raise argparse.ArgumentTypeError(
            f"formats must be a comma-separated subset of {','.join(FORMATS)}, got {s!r}")
    return parts


def _add_formats_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--formats", type=_parse_formats,
                        default=_parse_formats(os.environ.get("MRCNG_FORMATS") or ",".join(FORMATS)),
                        help="comma-separated layouts to build/check: precomputed,omezarr "
                             "(default: $MRCNG_FORMATS, else both)")
```

Note `any(p not in FORMATS ...)` rejects `""` from an empty string or a double comma, because `"" not in FORMATS`.

In `main()`, after `_add_classification_args(build_p)` add `_add_formats_arg(build_p)`; after `_add_classification_args(status_p)` add `_add_formats_arg(status_p)`.

`_build_command`: the task tuple gains `args.formats` as its last element; `_build_one_record` unpacks it and passes `formats=formats` to `build_one`:

```python
    (source_root, cache_root, relpath, params, force, max_block_bytes, assume_mode0,
     stack_globs, volume_globs, formats) = task
    ...
        result = build_one(source_root, cache_root, relpath, params, force=force,
                           max_block_bytes=max_block_bytes, assume_mode0=assume_mode0,
                           stack_globs=stack_globs, volume_globs=volume_globs, formats=formats)
```

and in `_build_command`:

```python
    tasks = [
        (source_root, cache_root, relpath, params, args.force, args.max_block_bytes,
         args.assume_mode0, tuple(args.stack_glob or ()), tuple(args.volume_glob or ()),
         tuple(args.formats))
        for relpath in relpaths
    ]
```

`_status_command`, replace the final `print`:

```python
        built = tuple(fp.get("formats", ()))
        status = result.value
        if result == Validity.VALID and not set(args.formats) <= set(built):
            status = "incomplete"  # valid, but a plain build would still rebuild it
        print(f"{relpath}: {status} [{','.join(built)}]")
```

Add `Validity` to the `from mrcng.fingerprint import ...` line.

- [ ] **Step 4: Run to verify pass**

Run: `pixi run -e default pytest -q tests/test_cli.py`
Expected: all PASS (the existing `assert "valid" in out` still holds).

- [ ] **Step 5: Commit**

```bash
git add src/mrcng/cli.py tests/test_cli.py
git commit -m "feat: mrc-pyramid --formats selects and reports cache layouts

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Server: `/precomputed/` alias and `/omezarr/` metadata

**Files:**
- Modify: `src/mrcng/server/app.py`
- Test: `tests/test_server_omezarr.py` (new)

**Interfaces:**
- Consumes: `omezarr.build_group_json`, `omezarr.build_array_json`; `Handle.validity_for(cache_dir, params)`; `fp["formats"]`, `fp["scales"]`.
- Produces: routes `GET /precomputed/{full_path}` (same handler as `/data/`), `GET /omezarr/<relpath>/zarr.json`, `GET /omezarr/<relpath>/<i>/zarr.json`. Internal helpers used by Task 6: `_valid_fp(handle, cache_dir, params, fmt) -> dict | None`, `_fingerprint_etag(fp) -> str`, `_INT_RE`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_server_omezarr.py
import json
import os
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mrcng.fingerprint import Params
from mrcng.paths import dataset_id, cache_dir_for
from mrcng.pyramid import build_one
from mrcng.server.config import Settings
from mrcng.server.app import create_app


def _fill(zz, yy, xx):
    return (xx + 1000 * yy + 1_000_000 * zz) % 30000


PARAMS = Params(chunk_size=(8, 8, 8), downsample="mean", min_axis_size=8,
                max_levels=3, dtype="int16", encoding="raw")


@pytest.fixture
def roots(tmp_path, make_mrc_file):
    source_root = tmp_path / "source"; source_root.mkdir()
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    # 36x20x12: edge chunks on every axis at chunk 8; level 1 is 18x10x6
    make_mrc_file(name="source/tomo.mrc", shape=(36, 20, 12), mode=1, fill=_fill)
    return source_root, cache_root


def _client(source_root, cache_root):
    return TestClient(create_app(Settings(source_root=source_root, cache_root=cache_root, chunk_size=(8, 8, 8))))


@pytest.fixture
def cached(roots):
    source_root, cache_root = roots
    build_one(source_root, cache_root, "tomo.mrc", PARAMS)
    return _client(source_root, cache_root), source_root, cache_root


@pytest.fixture
def uncached(roots):
    source_root, cache_root = roots
    return _client(source_root, cache_root), source_root, cache_root


def test_precomputed_prefix_is_an_alias_for_data(cached):
    client, _, _ = cached
    a = client.get("/data/tomo.mrc/info")
    b = client.get("/precomputed/tomo.mrc/info")
    assert a.status_code == b.status_code == 200 and a.content == b.content
    a = client.get("/data/tomo.mrc/2_2_2/0-8_0-8_0-6")
    b = client.get("/precomputed/tomo.mrc/2_2_2/0-8_0-8_0-6")
    assert a.status_code == b.status_code == 200 and a.content == b.content


def test_cached_group_json_is_served_verbatim_with_fingerprint_etag(cached):
    client, _, cache_root = cached
    cache_dir = cache_dir_for(cache_root, dataset_id("tomo.mrc"))
    resp = client.get("/omezarr/tomo.mrc/zarr.json")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.content == (cache_dir / "omezarr" / "zarr.json").read_bytes()
    assert resp.headers["cache-control"] == "no-cache, must-revalidate"
    fp = json.loads((cache_dir / "fingerprint.json").read_text())
    assert resp.headers["etag"] == f'"{fp["source_header_sha256"][:16]}-{fp["derivation_version"]}"'
    assert [d["path"] for d in resp.json()["attributes"]["ome"]["multiscales"][0]["datasets"]] == ["0", "1", "2"]


def test_cached_array_json_served_for_each_level_and_404_beyond(cached):
    client, _, cache_root = cached
    cache_dir = cache_dir_for(cache_root, dataset_id("tomo.mrc"))
    for level in (0, 1, 2):
        resp = client.get(f"/omezarr/tomo.mrc/{level}/zarr.json")
        assert resp.status_code == 200, level
        assert resp.content == (cache_dir / "omezarr" / str(level) / "zarr.json").read_bytes()
    assert client.get("/omezarr/tomo.mrc/3/zarr.json").status_code == 404


def test_uncached_group_lists_only_level_0_and_higher_metadata_404s(uncached):
    client, _, _ = uncached
    resp = client.get("/omezarr/tomo.mrc/zarr.json")
    assert resp.status_code == 200
    ms = resp.json()["attributes"]["ome"]["multiscales"][0]
    assert [d["path"] for d in ms["datasets"]] == ["0"]
    assert ms["name"] == "tomo.mrc"
    assert resp.headers["cache-control"] == "no-cache, must-revalidate" and resp.headers["etag"]

    arr0 = client.get("/omezarr/tomo.mrc/0/zarr.json")
    assert arr0.status_code == 200
    assert arr0.json()["shape"] == [12, 20, 36] and arr0.json()["data_type"] == "int16"
    assert client.get("/omezarr/tomo.mrc/1/zarr.json").status_code == 404


def test_stale_cache_falls_back_to_single_level(cached):
    client, source_root, _ = cached
    time.sleep(0.01)
    os.utime(source_root / "tomo.mrc", None)
    resp = client.get("/omezarr/tomo.mrc/zarr.json")
    assert [d["path"] for d in resp.json()["attributes"]["ome"]["multiscales"][0]["datasets"]] == ["0"]


def test_precomputed_only_cache_serves_single_level_on_omezarr(roots):
    source_root, cache_root = roots
    build_one(source_root, cache_root, "tomo.mrc", PARAMS, formats=("precomputed",))
    client = _client(source_root, cache_root)
    assert len(client.get("/precomputed/tomo.mrc/info").json()["scales"]) == 3
    resp = client.get("/omezarr/tomo.mrc/zarr.json")
    assert [d["path"] for d in resp.json()["attributes"]["ome"]["multiscales"][0]["datasets"]] == ["0"]


def test_omezarr_only_cache_serves_single_scale_on_precomputed(roots):
    source_root, cache_root = roots
    build_one(source_root, cache_root, "tomo.mrc", PARAMS, formats=("omezarr",))
    client = _client(source_root, cache_root)
    assert len(client.get("/precomputed/tomo.mrc/info").json()["scales"]) == 1
    assert client.get("/precomputed/tomo.mrc/2_2_2/0-8_0-8_0-6").status_code == 404


def test_zarr_json_disambiguation_for_a_file_named_like_a_level(tmp_path, make_mrc_file):
    source_root = tmp_path / "source"; source_root.mkdir()
    (source_root / "sub").mkdir()
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    make_mrc_file(name="source/sub/1", shape=(8, 8, 8), mode=1)   # a file literally named "1"
    client = _client(source_root, cache_root)
    resp = client.get("/omezarr/sub/1/zarr.json")
    assert resp.status_code == 200 and resp.json()["node_type"] == "group"


@pytest.mark.parametrize("url", [
    "/omezarr/zarr.json",                 # empty relpath
    "/omezarr/tomo.mrc/.zattrs",          # zarr v2 probes
    "/omezarr/tomo.mrc/.zgroup",
    "/omezarr/tomo.mrc/0/.zarray",
    "/omezarr/missing.mrc/zarr.json",
    "/omezarr/..%2Foutside.mrc/zarr.json",
    "/omezarr/tomo.mrc/notalevel/zarr.json",
])
def test_omezarr_non_routes_404(uncached, url):
    client, _, _ = uncached
    assert client.get(url).status_code == 404
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e default pytest -q tests/test_server_omezarr.py`
Expected: FAIL; `/precomputed/...` and `/omezarr/...` return 404 everywhere (the alias and the disambiguation tests fail on status codes, the 404 tests pass vacuously).

- [ ] **Step 3: Implement**

In `src/mrcng/server/app.py`:

Imports:

```python
from mrcng import omezarr
```

Helpers near `_source_etag`:

```python
_INT_RE = re.compile(r"^\d+$")


def _fingerprint_etag(fp: dict) -> str:
    return f'"{fp["source_header_sha256"][:16]}-{fp["derivation_version"]}"'


def _valid_fp(handle, cache_dir: Path, params: Params, fmt: str) -> dict | None:
    """The fingerprint if this entry is VALID *and* the build wrote layout `fmt`,
    else None. Every cache read in the server goes through here, so a format the
    build did not write is indistinguishable from no cache at all."""
    validity, fp = handle.validity_for(cache_dir, params)
    if validity != Validity.VALID or fp is None or fmt not in fp.get("formats", ()):
        return None
    return fp
```

Use `_valid_fp` in the existing precomputed handlers. In `_serve_info` replace

```python
            validity, fp = handle.validity_for(cache_dir, _current_params(settings, hdr))
            cache_hit = validity == Validity.VALID and fp is not None
```

with

```python
            fp = _valid_fp(handle, cache_dir, _current_params(settings, hdr), "precomputed")
            cache_hit = fp is not None
```

and its `etag = f'"{fp[...]}..."'` line with `etag = _fingerprint_etag(fp)`. In `_serve_chunk` replace

```python
            validity, fp = handle.validity_for(cache_dir, _current_params(settings, hdr))
            if fp is None or validity != Validity.VALID:
                return Response(status_code=404)  # no valid cache -> nothing above scale 0
```

with

```python
            fp = _valid_fp(handle, cache_dir, _current_params(settings, hdr), "precomputed")
            if fp is None:
                return Response(status_code=404)  # no valid cache for this layout -> nothing above scale 0
```

Routes in `create_app`:

```python
    @app.get("/data/{full_path:path}")          # legacy alias, kept for saved links
    @app.get("/precomputed/{full_path:path}")
    async def dispatch(full_path: str):
        ...  # body unchanged

    @app.get("/omezarr/{full_path:path}")
    async def dispatch_omezarr(full_path: str):
        seg = full_path.split("/")

        # <relpath>/<level>/c/<kz>/<ky>/<kx> -- unambiguous: relpath must be a
        # file, and a file has no children.
        if (len(seg) >= 6 and seg[-4] == "c"
                and all(_INT_RE.match(s) for s in (seg[-5], seg[-3], seg[-2], seg[-1]))):
            return await _serve_zarr_chunk(
                settings, fd_cache, semaphore, "/".join(seg[:-5]),
                int(seg[-5]), int(seg[-3]), int(seg[-2]), int(seg[-1]),
            )

        if seg[-1] == "zarr.json":
            # "a/b/1/zarr.json" is level 1 of a/b if a/b is a file, else the
            # group of a file literally named a/b/1. Costs one extra resolve.
            if len(seg) >= 3 and _INT_RE.match(seg[-2]):
                relpath = "/".join(seg[:-2])
                try:
                    resolve_source(settings.source_root, relpath)
                except PathNotAllowed:
                    pass
                else:
                    return await _serve_zarr_metadata(settings, fd_cache, relpath, int(seg[-2]))
            return await _serve_zarr_metadata(settings, fd_cache, "/".join(seg[:-1]), None)

        return Response(status_code=404)   # includes .zattrs/.zgroup/.zarray v2 probes
```

`_serve_zarr_chunk` is Task 6; for this task add a stub so the module imports:

```python
async def _serve_zarr_chunk(settings, fd_cache, semaphore, relpath, level, kz, ky, kx) -> Response:
    return Response(status_code=404)
```

Metadata handler:

```python
async def _serve_zarr_metadata(settings, fd_cache: FdCache, relpath: str, level: int | None) -> Response:
    """Group zarr.json (level None) or one level's array zarr.json. Cached and
    valid for omezarr: the build's file, verbatim. Otherwise a single-level
    document from the live header, mirroring the uncached precomputed info."""
    start = time.monotonic()
    try:
        path = resolve_source(settings.source_root, relpath)
    except PathNotAllowed:
        return Response(status_code=404)

    rel = "zarr.json" if level is None else f"{level}/zarr.json"
    body: bytes | None = None
    etag: str | None = None
    try:
        with fd_cache.open(path) as handle:
            hdr = handle.hdr
            cache_dir = _cache_dir_for(settings, relpath)
            fp = _valid_fp(handle, cache_dir, _current_params(settings, hdr), "omezarr")
            cache_hit = fp is not None
            if cache_hit:
                if level is not None and level > len(fp["scales"]):
                    return Response(status_code=404)
                try:
                    body = (cache_dir / "omezarr" / rel).read_bytes()
                    json.loads(body)
                except (OSError, json.JSONDecodeError):
                    _logger.error(
                        "%s: fingerprint is valid but omezarr/%s is unreadable or corrupt; "
                        "falling back to single level", relpath, rel,
                    )
                    cache_hit = False
                else:
                    etag = _fingerprint_etag(fp)
            if not cache_hit:
                if level not in (None, 0):
                    return Response(status_code=404)
                scales = plan_scales((hdr.nx, hdr.ny, hdr.nz), min_axis_size=32, max_levels=1)
                if level is None:
                    doc = omezarr.build_group_json(hdr, scales, name=relpath.rsplit("/", 1)[-1])
                else:
                    doc = omezarr.build_array_json(scales[0].size, settings.chunk_size, hdr.served_dtype)
                body = json.dumps(doc).encode()
                etag = _source_etag(hdr)
    except MrcFormatError as e:
        return _header_error_response(e)

    _log_access(relpath, rel, "", cache_hit, start)
    return Response(
        content=body,
        media_type="application/json",
        headers={"Cache-Control": "no-cache, must-revalidate", "ETag": etag},
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `pixi run -e default pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mrcng/server/app.py tests/test_server_omezarr.py
git commit -m "feat: /precomputed alias and /omezarr metadata routes

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Server: `/omezarr/` chunks

**Files:**
- Modify: `src/mrcng/server/app.py` (replace the `_serve_zarr_chunk` stub; extract `_cached_file_response` from the tail of `_serve_chunk`)
- Test: `tests/test_server_omezarr.py`

**Interfaces:**
- Consumes: `omezarr.chunk_region`, `omezarr.pad_chunk`, `omezarr.chunk_rel_path`, `read_chunk`, `encode_chunk`, `_valid_fp`.
- Produces: `GET /omezarr/<relpath>/<i>/c/<kz>/<ky>/<kx>`; `_cached_file_response(settings, chunk_path) -> Response` shared by both formats.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_server_omezarr.py`:

```python
def test_level0_chunk_interior_equals_precomputed_and_edge_is_padded(uncached):
    client, source_root, _ = uncached
    # interior chunk (kz=0, ky=0, kx=0): identical bytes to precomputed scale 0
    z = client.get("/omezarr/tomo.mrc/0/c/0/0/0")
    p = client.get("/precomputed/tomo.mrc/1_1_1/0-8_0-8_0-8")
    assert z.status_code == p.status_code == 200 and z.content == p.content
    assert z.headers["cache-control"] == "no-cache, must-revalidate" and z.headers["etag"]
    assert "x-mrcng-read-strategy" in z.headers

    # edge chunk kx=4 covers x[32:36), ky=2 covers y[16:20), kz=1 covers z[8:12)
    z = client.get("/omezarr/tomo.mrc/0/c/1/2/4")
    assert z.status_code == 200 and len(z.content) == 8 * 8 * 8 * 2
    arr = np.frombuffer(z.content, dtype="<i2").reshape(8, 8, 8)
    zz, yy, xx = np.meshgrid(np.arange(8, 12), np.arange(16, 20), np.arange(32, 36), indexing="ij")
    np.testing.assert_array_equal(arr[:4, :4, :4], _fill(zz, yy, xx).astype("<i2"))
    assert arr[4:].sum() == 0 and arr[:, 4:].sum() == 0 and arr[:, :, 4:].sum() == 0


@pytest.mark.parametrize("idx", ["0/0/5", "0/3/0", "2/0/0", "0/0/999999"])
def test_level0_chunk_outside_grid_404s(uncached, idx):
    client, _, _ = uncached
    assert client.get(f"/omezarr/tomo.mrc/0/c/{idx}").status_code == 404


def test_cached_chunk_is_served_verbatim_and_immutable(cached):
    client, _, cache_root = cached
    cache_dir = cache_dir_for(cache_root, dataset_id("tomo.mrc"))
    resp = client.get("/omezarr/tomo.mrc/1/c/0/1/2")
    assert resp.status_code == 200
    assert resp.content == (cache_dir / "omezarr" / "1" / "c" / "0" / "1" / "2").read_bytes()
    assert len(resp.content) == 8 * 8 * 8 * 2
    assert resp.headers["cache-control"] == "public, max-age=31536000, immutable"


@pytest.mark.parametrize("url", [
    "/omezarr/tomo.mrc/1/c/0/0/999999",   # outside grid, never a file
    "/omezarr/tomo.mrc/3/c/0/0/0",        # level beyond the pyramid
])
def test_cached_chunk_missing_404s(cached, url):
    client, _, _ = cached
    assert client.get(url).status_code == 404


def test_uncached_higher_level_chunk_404s(uncached):
    client, _, _ = uncached
    assert client.get("/omezarr/tomo.mrc/1/c/0/0/0").status_code == 404


def test_precomputed_only_cache_404s_omezarr_chunks(roots):
    source_root, cache_root = roots
    build_one(source_root, cache_root, "tomo.mrc", PARAMS, formats=("precomputed",))
    client = _client(source_root, cache_root)
    assert client.get("/omezarr/tomo.mrc/1/c/0/0/0").status_code == 404


def test_cached_chunk_x_accel_redirect_when_sendfile_disabled(roots):
    source_root, cache_root = roots
    build_one(source_root, cache_root, "tomo.mrc", PARAMS)
    settings = Settings(source_root=source_root, cache_root=cache_root, chunk_size=(8, 8, 8),
                        serve_cache_via_sendfile=False, cache_internal_location="/__cache__")
    client = TestClient(create_app(settings))
    resp = client.get("/omezarr/tomo.mrc/1/c/0/0/0")
    assert resp.status_code == 200 and resp.content == b""
    cache_dir = cache_dir_for(cache_root, dataset_id("tomo.mrc"))
    rel = (cache_dir / "omezarr" / "1" / "c" / "0" / "0" / "0").relative_to(cache_root).as_posix()
    assert resp.headers["x-accel-redirect"] == f"/__cache__/{rel}"


def test_mode12_float16_serves_float32_metadata_and_padded_chunk(tmp_path, make_mrc_file):
    source_root = tmp_path / "source"; source_root.mkdir()
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    make_mrc_file(name="source/half.mrc", shape=(6, 6, 6), mode=12, fill=lambda zz, yy, xx: xx + 0.5)
    client = _client(source_root, cache_root)
    assert client.get("/omezarr/half.mrc/0/zarr.json").json()["data_type"] == "float32"
    resp = client.get("/omezarr/half.mrc/0/c/0/0/0")
    assert resp.status_code == 200 and len(resp.content) == 8 * 8 * 8 * 4
    arr = np.frombuffer(resp.content, dtype="<f4").reshape(8, 8, 8)
    assert arr[0, 0, 3] == 3.5 and arr[7, 7, 7] == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e default pytest -q tests/test_server_omezarr.py`
Expected: FAIL on the 200 assertions (stub returns 404).

- [ ] **Step 3: Implement**

In `src/mrcng/server/app.py`, extract the tail of `_serve_chunk` (everything from `if settings.serve_cache_via_sendfile:` to the end) into:

```python
def _cached_file_response(settings, chunk_path: Path) -> Response:
    """A built chunk file, in either layout. Immutable: the URL bakes in the
    level and index and the fingerprint guards the content."""
    if settings.serve_cache_via_sendfile:
        return FileResponse(
            chunk_path,
            media_type="application/octet-stream",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )
    # nginx serves the body itself via X-Accel-Redirect; Python only sets
    # headers and is out of the data path entirely.
    rel = chunk_path.relative_to(settings.cache_root).as_posix()
    return Response(
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            "X-Accel-Redirect": f"{settings.cache_internal_location}/{rel}",
        },
    )
```

and end `_serve_chunk` with `return _cached_file_response(settings, chunk_path)`.

Replace the stub:

```python
async def _serve_zarr_chunk(settings, fd_cache: FdCache, semaphore: asyncio.Semaphore,
                            relpath: str, level: int, kz: int, ky: int, kx: int) -> Response:
    start = time.monotonic()
    try:
        path = resolve_source(settings.source_root, relpath)
    except PathNotAllowed:
        return Response(status_code=404)

    cx, cy, cz = settings.chunk_size
    chunk_label = f"{kz}/{ky}/{kx}"
    try:
        with fd_cache.open(path) as handle:
            fd, hdr = handle.fd, handle.hdr
            if level == 0:
                try:
                    x0, x1, y0, y1, z0, z1 = omezarr.chunk_region(
                        (hdr.nx, hdr.ny, hdr.nz), settings.chunk_size, kz, ky, kx)
                except ValueError:
                    return Response(status_code=404)

                threshold = settings.read_row_bytes_threshold
                async with semaphore:
                    try:
                        arr = await asyncio.to_thread(
                            read_chunk, fd, hdr, x0, x1, y0, y1, z0, z1, threshold,
                        )
                    except ChunkOutOfBounds:
                        return Response(status_code=404)
                    except UnexpectedEOF as e:
                        _logger.error(
                            "unexpected EOF reading %s omezarr/0/%s: %s", relpath, chunk_label, e,
                        )
                        return Response(status_code=500)

                # Zarr chunks are always chunk_shape-sized; precomputed clips instead.
                body = encode_chunk(np.ascontiguousarray(omezarr.pad_chunk(arr, (cz, cy, cx))))
                _log_access(relpath, "0", chunk_label, False, start)
                return Response(
                    content=body,
                    media_type="application/octet-stream",
                    headers={
                        # Same reasoning as the precomputed scale-0 path: the source
                        # is mutable at this relpath, so revalidate every time.
                        "Cache-Control": "no-cache, must-revalidate",
                        "ETag": _source_etag(hdr),
                        "X-Mrcng-Read-Strategy": choose_strategy(
                            x0, x1, hdr.dtype.itemsize, threshold).value,
                    },
                )

            cache_dir = _cache_dir_for(settings, relpath)
            fp = _valid_fp(handle, cache_dir, _current_params(settings, hdr), "omezarr")
            if fp is None or level > len(fp["scales"]):
                return Response(status_code=404)
    except MrcFormatError as e:
        return _header_error_response(e)

    # No grid check against the fingerprint here, unlike precomputed: the key
    # is four integers, so there is nothing path-unsafe to validate, and an
    # index outside the grid is simply not a file.
    chunk_path = cache_dir / "omezarr" / omezarr.chunk_rel_path(level, kz, ky, kx)
    if not chunk_path.is_file():
        return Response(status_code=404)

    _log_access(relpath, str(level), chunk_label, True, start)
    return _cached_file_response(settings, chunk_path)
```

Add `import numpy as np` to the module imports.

- [ ] **Step 4: Run to verify pass**

Run: `pixi run -e default pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mrcng/server/app.py tests/test_server_omezarr.py
git commit -m "feat: serve OME-Zarr chunks (level 0 from MRC, higher levels from cache)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Browse UI links for both formats

**Files:**
- Modify: `src/mrcng/server/browse.py` (`build_neuroglancer_link`, `_render_listing`)
- Test: `tests/test_browse.py`

**Interfaces:**
- Produces: `build_neuroglancer_link(scheme, netloc, relpath, fmt="precomputed") -> str`; `fmt` is `"precomputed"` → `precomputed://<scheme>://<netloc>/precomputed/<relpath>`, `"omezarr"` → `zarr3://<scheme>://<netloc>/omezarr/<relpath>`.

- [ ] **Step 1: Update and add tests**

In `tests/test_browse.py` change the two existing expectations from `/data/` to `/precomputed/`:

```bash
sed -i 's|precomputed://https://example.org:8443/data/sub/tomo.mrc|precomputed://https://example.org:8443/precomputed/sub/tomo.mrc|; s|precomputed://http://localhost:8000/data/top.mrc|precomputed://http://localhost:8000/precomputed/top.mrc|' tests/test_browse.py
```

Append:

```python
def test_build_neuroglancer_link_omezarr_uses_zarr3_scheme():
    link = build_neuroglancer_link("https", "example.org:8443", "sub/tomo.mrc", fmt="omezarr")
    state = json.loads(unquote(link[len(NEUROGLANCER_BASE_URL):]))
    assert state["layers"][0]["source"] == "zarr3://https://example.org:8443/omezarr/sub/tomo.mrc"
    assert state["layers"][0]["name"] == "tomo.mrc"


def test_listing_offers_both_formats_per_file(browse_client):
    resp = browse_client.get("/browse")
    assert resp.status_code == 200
    assert "Neuroglancer (precomputed)" in resp.text
    assert "Neuroglancer (OME-Zarr)" in resp.text
    assert "precomputed%3A%2F%2F" in resp.text and "zarr3%3A%2F%2F" in resp.text
    assert "/data/" not in unquote(resp.text)
```

`tests/test_browse.py:54` asserts the literal text `"Open in Neuroglancer"`; change it to `"Neuroglancer (precomputed)"`.

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e default pytest -q tests/test_browse.py`
Expected: FAIL (`/data/` still generated; `fmt` unknown keyword).

- [ ] **Step 3: Implement**

```python
def build_neuroglancer_link(scheme: str, netloc: str, relpath: str, fmt: str = "precomputed") -> str:
    """relpath is POSIX-style, relative to MRCNG_SOURCE_ROOT. fmt selects the
    endpoint: precomputed:// over /precomputed/, or zarr3:// over /omezarr/
    (zarr3 rather than zarr so Neuroglancer does not probe for v2 metadata)."""
    name = relpath.rsplit("/", 1)[-1]
    if fmt == "omezarr":
        source = f"zarr3://{scheme}://{netloc}/omezarr/{relpath}"
    else:
        source = f"precomputed://{scheme}://{netloc}/precomputed/{relpath}"
    state = {"layers": [{"type": "auto", "source": source, "name": name}]}
    encoded = quote(json.dumps(state, separators=(",", ":")), safe="")
    return f"{NEUROGLANCER_BASE_URL}{encoded}"
```

In `_render_listing`, the per-file line becomes:

```python
            pre = build_neuroglancer_link(request.url.scheme, request.url.netloc, file_relpath)
            zarr = build_neuroglancer_link(request.url.scheme, request.url.netloc, file_relpath, fmt="omezarr")
            parts.append(
                f'<li>{html.escape(f.name)} '
                f'&mdash; <a href="{html.escape(pre)}" target="_blank">Neuroglancer (precomputed)</a> '
                f'&middot; <a href="{html.escape(zarr)}" target="_blank">Neuroglancer (OME-Zarr)</a></li>'
            )
```

- [ ] **Step 4: Run to verify pass**

Run: `pixi run -e default pytest -q tests/test_browse.py`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mrcng/server/browse.py tests/test_browse.py
git commit -m "feat: browse UI links both precomputed and OME-Zarr endpoints

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Conformance test with the `zarr` library

**Files:**
- Modify: `pyproject.toml` (`[tool.pixi.feature.test.dependencies]`), `pixi.lock`
- Test: `tests/test_pyramid.py`

**Interfaces:**
- Consumes: the on-disk `omezarr/` directory written by Task 3.

- [ ] **Step 1: Add the test-only dependency**

```bash
pixi add --feature test "zarr>=3"
pixi run -e default python -c "import zarr; print(zarr.__version__)"
```

Expected: prints a 3.x version. `pyproject.toml` now lists `zarr = ">=3"` under the test feature and `pixi.lock` is updated.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_pyramid.py`:

```python
def test_omezarr_layout_opens_with_the_zarr_library(source_and_cache):
    """The one check that our hand-written zarr.json files are what a real
    Zarr v3 implementation expects. Level 0 has no chunks on disk by design,
    so it reads back as fill_value -- asserted here so nobody mistakes the
    cache for a standalone store."""
    import zarr
    from mrcng.downsample import block_mean
    import mrcfile

    source_root, cache_root, relpath = source_and_cache
    build_one(source_root, cache_root, relpath, _params())
    cache_dir = cache_dir_for(cache_root, dataset_id(relpath))

    group = zarr.open_group(str(cache_dir / "omezarr"), mode="r", zarr_format=3)
    ome = group.attrs["ome"]
    assert ome["version"] == "0.5"
    assert [d["path"] for d in ome["multiscales"][0]["datasets"]] == ["0", "1", "2"]

    with mrcfile.open(source_root / relpath, permissive=True) as mf:
        level0 = np.asarray(mf.data)
    level1 = group["1"]
    assert level1.shape == (16, 16, 16) and level1.dtype == np.dtype("int16")
    assert level1.chunks == (8, 8, 8)
    np.testing.assert_array_equal(level1[:], block_mean(level0, (2, 2, 2)))
    level2 = group["2"]
    np.testing.assert_array_equal(level2[:], block_mean(block_mean(level0, (2, 2, 2)), (2, 2, 2)))

    assert group["0"].shape == (32, 32, 32)
    assert group["0"][:].sum() == 0   # serving cache, not a standalone store
```

- [ ] **Step 3: Run to verify it passes (or fails and points at a metadata bug)**

Run: `pixi run -e default pytest -q tests/test_pyramid.py::test_omezarr_layout_opens_with_the_zarr_library`
Expected: PASS. If `zarr` rejects `zarr.json`, the error names the field; fix it in `omezarr.build_array_json` / `build_group_json` (Task 1's unit tests define the intended values, so adjust both together) and note the correction in the spec.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml pixi.lock tests/test_pyramid.py
git commit -m "test: open the omezarr cache layout with the zarr library

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Docs

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `notes/TestPlan.md`

- [ ] **Step 1: README**

Under "Loading data into Neuroglancer", step 2 becomes:

````markdown
2. For a file at `<MRCNG_SOURCE_ROOT>/some/relative/path.mrc`, the
   Neuroglancer data source URL is one of:

   ```
   precomputed://https://your-host:8000/precomputed/some/relative/path.mrc
   zarr3://https://your-host:8000/omezarr/some/relative/path.mrc
   ```

   (no `/info` or `/zarr.json` suffix — Neuroglancer appends those itself).
   `/data/` is a legacy alias for `/precomputed/` and keeps working for saved
   links.
````

Add a section after "Cache invalidation":

````markdown
### Output formats

The server speaks two protocols from the same cache entry:

- **precomputed** at `/precomputed/<relpath>` (alias `/data/`): Neuroglancer's
  native format, clipped edge chunks.
- **OME-Zarr 0.5** (Zarr v3) at `/omezarr/<relpath>`: `zarr.json` group and
  array metadata, raw little-endian chunks at `<level>/c/<z>/<y>/<x>`, edge
  chunks zero-padded to the chunk shape. Downsampled levels carry a
  `translation` of `(f-1)/2` source voxels per axis, the OME-NGFF pixel-centre
  convention.

Level 0 is read from the MRC on request in both formats and is never on disk,
so `omezarr/` is a serving cache, not a standalone OME-Zarr store: opened
directly with a Zarr library, level 0 reads as zeros.

Which layouts a build writes is `--formats` (default `precomputed,omezarr`;
environment default `MRCNG_FORMATS`):

```bash
pixi run build-cache --source-root ... --cache-root ... --formats omezarr
```

Formats are replaced, not merged: an entry is only skipped as valid when it
holds exactly the requested set, otherwise it is rebuilt with exactly that set.
`mrc-pyramid status` prints the formats beside the validity and says
`incomplete` when a valid entry lacks one you asked for. A request for a format
the build did not write serves single-resolution on that endpoint.

Cache entry layout:

```
<cache_root>/<xx>/<dataset_id>/
  fingerprint.json
  precomputed/info
  precomputed/<scale_key>/<x0-x1_y0-y1_z0-z1>
  omezarr/zarr.json
  omezarr/<level>/zarr.json          # level 0 has metadata but no chunks
  omezarr/<level>/c/<kz>/<ky>/<kx>
```

**Caches built before this landed must be rebuilt.** The fingerprint schema
changed (v4) and the precomputed layout moved under `precomputed/`; old entries
read as `incompatible` and a plain `mrc-pyramid build` rebuilds them. Build into
a fresh `--cache-root` and swap to avoid the single-resolution window.
````

In the nginx snippet, change `location /data/ {` to `location ~ ^/(data|precomputed|omezarr)/ {`.

In the "Browsing data in a web browser" section, replace the sentence about the single "Open in Neuroglancer" link with: every `.mrc`/`.rec` file gets two links, "Neuroglancer (precomputed)" and "Neuroglancer (OME-Zarr)".

- [ ] **Step 2: CLAUDE.md**

In the `DERIVATION_VERSION` paragraph, change `Today that means \`mrcheader.py\`, \`precomputed.py\`, \`downsample.py\`, \`pyramid.py\` and \`reader.py\`` to include `omezarr.py`. Add to "Other traps":

```markdown
- Two layouts, one entry. `precomputed/` clips edge chunks; `omezarr/` pads them
  to `chunk_shape` with zeros. Interior chunks are byte-identical. The level-
  from-level cascade reads whichever layout the build selected, and slices the
  padding off when it is `omezarr/` — forgetting that averages zeros into edge
  voxels at level 2+. `fingerprint["formats"]` says which layouts exist; a
  format not listed is served as no cache.
```

- [ ] **Step 3: notes/TestPlan.md**

Append:

```markdown
## OME-Zarr endpoint (manual)

1. Build one tomogram and one tilt series with the default `--formats`.
2. In Neuroglancer add two layers for the same tomogram: `precomputed://…/precomputed/<relpath>` and `zarr3://…/omezarr/<relpath>`.
3. Zoom through every level. Features must coincide at every zoom; a half-voxel drift growing with zoom-out means Neuroglancer's Zarr reader already applies a centre correction and the `translation` in `omezarr.build_group_json` must be dropped (update the spec too).
4. Scale bar reads the same on both; no seams at the volume edges (padded chunks render as zeros only beyond the volume).
5. Tilt series: z steps through every tilt at every level on `/omezarr/`; no tilt averaging.
```

- [ ] **Step 4: Run the full suite one last time and commit**

Run: `pixi run -e default pytest -q`
Expected: all PASS.

```bash
git add README.md CLAUDE.md notes/TestPlan.md
git commit -m "docs: OME-Zarr output, --formats, layout move and rebuild note

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
