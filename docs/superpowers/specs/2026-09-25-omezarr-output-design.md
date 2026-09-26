# Design: OME-Zarr 0.5 output alongside precomputed

## Goal

Serve every MRC under `MRCNG_SOURCE_ROOT` as an OME-Zarr 0.5 multiscale image
(Zarr v3) in addition to the existing Neuroglancer `precomputed` volume. Two new
URL prefixes select the format; the existing `/data/` prefix stays as an alias
for precomputed. The pyramid build writes each format's chunks and metadata to
disk so the server serves them verbatim, exactly as it serves `info` and
precomputed chunks today. Which formats a build writes is an operator option.

## Decisions already made

- **OME-Zarr 0.5 only** (Zarr v3, `zarr.json`). No 0.4 / Zarr v2 support.
- **Raw chunks only.** Codec chain is `bytes` little-endian. No compressor, no
  new runtime dependency. Interior chunk bytes are identical to the precomputed
  chunk covering the same region.
- **Default build writes both formats.** `--formats precomputed,omezarr` is the
  default; `--formats omezarr` writes one.
- **Two independent layouts on disk.** The server never translates one format
  into the other. Chosen over a single canonical store with request-time
  translation because the repo's core rule is that the server serves what a
  build wrote and derives nothing on the request path.
- **Full cache rebuild is accepted.** `SCHEMA_VERSION` bumps, and the precomputed
  layout moves under a `precomputed/` subdirectory beside `omezarr/`.
- **Downsampled levels carry a translation** following the OME-NGFF pixel-centre
  convention (see section 3).

## 1. Non-goals

- Level 0 on disk. It is always read from the MRC on request, in both formats.
  The `omezarr/` directory is therefore a serving cache, not a standalone
  OME-Zarr store: opening it directly with a Zarr library reads level 0 as
  `fill_value`. Documented, not fixed.
- Compression, sharding, Zarr v2, OME-Zarr 0.4, multiple channels or time.
- A server-side format setting. The server serves whichever URL is requested;
  whether cached levels exist for that format is answered by the fingerprint.
- Incremental "add a format to an existing entry" builds. Formats are replaced,
  not merged (section 2).

## 2. On-disk layout and fingerprint

```
<cache_root>/<xx>/<dataset_id>/
  fingerprint.json               # written last; presence == entry complete
  .lock
  precomputed/
    info
    2_2_2/0-64_0-64_0-64 ...     # clipped edge chunks, unchanged bytes
    4_4_4/...
  omezarr/
    zarr.json                    # group: ome 0.5 multiscales, all levels incl. 0
    0/zarr.json                  # array metadata for level 0 (no chunks on disk)
    1/zarr.json
    1/c/<kz>/<ky>/<kx> ...       # padded chunks, C-order (z, y, x)
    2/zarr.json
    2/c/...
```

OME level numbering: `0` is full resolution, `1` the first downsample. Level `i`
corresponds to `plan_scales(...)[i]`; the precomputed key for the same level is
`scales[i].key`.

`fingerprint.json` changes:

| Field | Change |
|---|---|
| `schema_version` | 3 → 4 (a key was added) |
| `derivation_version` | 1 → 2 (artifacts moved; redundant with the schema bump but documents it) |
| `formats` | **new**: list of format names this build wrote, e.g. `["precomputed", "omezarr"]` |
| `scales` | unchanged: `key -> [sx, sy, sz]` for levels ≥ 1, in level order |

Validity for a request in format F is `validate(...) == VALID and F in
fp["formats"]`. Anything else serves single-resolution on that endpoint, the
same fallback precomputed has today.

Build semantics: `build_one` is `SKIPPED_VALID` only if the fingerprint is
`VALID` **and** `set(fp["formats"]) == set(requested formats)`. Otherwise it
rebuilds the entry from scratch: every subdirectory is removed, exactly the
requested formats are written, and `formats` records them. So `--formats
omezarr` against an entry that has both drops `precomputed/`. The fingerprint
always describes what is on disk. `mrc-pyramid status` prints the formats next
to the validity, e.g. `foo.mrc: valid [precomputed,omezarr]`, and reports
`incomplete` when the entry is valid but lacks a requested format.

## 3. OME-Zarr 0.5 metadata

New module `src/mrcng/omezarr.py`. It is a derivation module (its output lands
in cache files), so it joins the `DERIVATION_VERSION` list in
`fingerprint.py` and `CLAUDE.md`.

### Group `omezarr/zarr.json`

```json
{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": {
    "ome": {
      "version": "0.5",
      "multiscales": [{
        "name": "<basename of relpath>",
        "axes": [
          {"name": "z", "type": "space", "unit": "nanometer"},
          {"name": "y", "type": "space", "unit": "nanometer"},
          {"name": "x", "type": "space", "unit": "nanometer"}
        ],
        "datasets": [
          {"path": "0", "coordinateTransformations": [
              {"type": "scale", "scale": [rz, ry, rx]}]},
          {"path": "1", "coordinateTransformations": [
              {"type": "scale", "scale": [rz*fz, ry*fy, rx*fx]},
              {"type": "translation", "translation": [(fz-1)/2*rz, (fy-1)/2*ry, (fx-1)/2*rx]}]}
        ]
      }]
    },
    "mrcng": {"voxel_size_is_default": false, "is_image_stack": false}
  }
}
```

- `rz, ry, rx` are the base voxel size in nanometres (ångström ÷ 10), the same
  numbers precomputed `info` uses. Nanometre rather than ångström so both
  endpoints display the same scale bar.
- `fz, fy, fx` are the level's cumulative factors (precomputed key `fx_fy_fz`).
- **Translation.** A mean over `f` source voxels is centred `(f-1)/2` source
  voxels in from the origin; under the NGFF convention that array index `i`
  addresses the centre of sample `i`, that is the level's offset. Level 0 has
  no translation entry. An axis with factor 1 (z for an image stack) gets 0.
  Precomputed keeps `voxel_offset [0,0,0]`: Neuroglancer's precomputed model is
  corner-aligned and a downsampled voxel already spans exactly its source region
  there. Manual test plan item: load the same file on both endpoints in
  Neuroglancer and confirm the features line up at every zoom level; if
  Neuroglancer's Zarr reader is found to apply its own half-voxel correction,
  the translation is dropped and this section is amended.
- Image stacks: z scale is `1.0` nm per slice and `fz` is always 1, mirroring
  precomputed.
- The non-spec flags live under `attributes.mrcng` because `ome` is reserved.

### Array `omezarr/<i>/zarr.json`

```json
{
  "zarr_format": 3,
  "node_type": "array",
  "shape": [nz, ny, nx],
  "data_type": "int16",
  "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [cz, cy, cx]}},
  "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
  "fill_value": 0,
  "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
  "dimension_names": ["z", "y", "x"],
  "attributes": {}
}
```

`data_type` is `hdr.served_dtype.name` (`int8`, `uint8`, `int16`, `uint16`,
`float32`). Zarr's names coincide with numpy's for these five. `shape` is the
level's size reversed to `(z, y, x)`; `chunk_shape` is `MRCNG_CHUNK_SIZE`
reversed.

### Chunk `omezarr/<i>/c/<kz>/<ky>/<kx>`

The C-order bytes of a `(cz, cy, cx)` array. The region it covers is
`[k*c, min((k+1)*c, size))` per axis; where that is short of `c` the remainder
is zero-filled. `omezarr.pad_chunk(arr, chunk_shape)` does the padding and is a
no-op (no copy) for interior chunks.

## 4. Server routes

`/data/{path}` and `/precomputed/{path}` route to the existing precomputed
dispatcher unchanged except for the cache subdirectory (`cache_dir /
"precomputed"`).

`/omezarr/{full_path}` parses from the right, as the precomputed dispatcher does:

| Suffix pattern | Meaning |
|---|---|
| `<relpath>/zarr.json` | group metadata |
| `<relpath>/<i>/zarr.json` | array metadata for level `i` |
| `<relpath>/<i>/c/<kz>/<ky>/<kx>` | chunk (all four are `\d+`) |
| anything else, including `.zattrs`, `.zgroup`, `.zarray` | 404 |

The chunk pattern is unambiguous: `relpath` must resolve to a file and a file has
no children. `zarr.json` has two candidate parses (`a/b/1/zarr.json` is the
level-1 array of `a/b` or the group of a file literally named `a/b/1`). Try the
array parse first when the penultimate segment is all digits and `a/b` resolves
to a file; otherwise try the group parse. This costs at most one extra
`resolve_source`.

### Metadata

Cached and valid for `omezarr`: serve `omezarr/zarr.json` or
`omezarr/<i>/zarr.json` verbatim, with the same `ETag` and `Cache-Control:
no-cache, must-revalidate` precomputed `info` uses; `404` if `i` is not a level
the fingerprint lists. Unreadable or corrupt file with a valid fingerprint: log
and fall back, as `_serve_info` does.

Not cached: generate a single-level group and the level-0 array from the live
header with `omezarr.build_group_json` / `build_array_json`, exactly the way
`build_info` is used for the uncached precomputed `info`. `ETag` is the source
etag. Any `i > 0` is 404.

### Chunks

Level 0: bounds-check `k` against the level-0 grid, read the clipped region
with `read_chunk` under the semaphore, `pad_chunk`, `encode_chunk`. Same
headers as the precomputed scale-0 path (`no-cache`, source `ETag`,
`X-Mrcng-Read-Strategy`).

Level ≥ 1: require validity `VALID` and `"omezarr" in fp["formats"]`, then
`is_file()` on `omezarr/<i>/c/<kz>/<ky>/<kx>` and serve it via `FileResponse`
or `X-Accel-Redirect`, with the immutable `Cache-Control`, through the same
code path the precomputed cached chunk uses. No grid validation against the
fingerprint: the indices are integers, so there is no path-safety concern, and
an out-of-grid index is simply not a file. (Precomputed validates the extent
first because its names encode sizes that could disagree with the grid; Zarr
keys cannot.)

Access log entries reuse `_log_access` with `scale_key` set to the level
number and `chunk` to `kz/ky/kx`.

## 5. Builder and CLI

### `pyramid.py`

- `build_one` takes `formats: tuple[str, ...]`. Formats are not added to
  `Params`: they record *what* was built, not *how*, so they are their own
  fingerprint key and not one of `_ADDRESSING_FIELDS`.
- `_write_chunk` becomes `_write_chunks(cache_dir, formats, level_index, scale,
  x0..z1, arr)`: writes the clipped array to `precomputed/<key>/<name>` when
  `precomputed` is selected, and the padded array to
  `omezarr/<i>/c/<kz>/<ky>/<kx>` when `omezarr` is selected. Returns bytes
  written. The downsample is computed once per chunk regardless of how many
  formats are written.
- `_read_prev_level_region` reads from `precomputed/` when that format is
  selected, otherwise from `omezarr/` and slices the padding off
  (`block[:bz, :by, :bx]`). One helper `_read_prev_chunk(...)` hides the choice.
- After the levels: write `precomputed/info` if selected; write
  `omezarr/zarr.json` and `omezarr/<i>/zarr.json` for every level including 0
  if selected. Then `_fsync_tree`, then the fingerprint with `formats`.
- `BuildResult` gains `formats`; the JSONL record includes it.

### `cli.py`

- `build` gains `--formats`, comma-separated, default from `MRCNG_FORMATS`,
  then `precomputed,omezarr`. Values validated against `{"precomputed",
  "omezarr"}`; empty is an error. Parsed with the same style as `--chunk-size`.
- `status` gains the same option and prints `<validity> [<formats>]`, with
  `incomplete` when valid but a requested format is missing.
- `prune` is unchanged.

### `config.py`

No new server setting. The server serves both endpoints unconditionally.

## 6. Browse UI

Each file gets two links: "Neuroglancer (precomputed)" using
`precomputed://.../precomputed/<relpath>` and "Neuroglancer (OME-Zarr)" using
`zarr3://.../omezarr/<relpath>`. `zarr3://` rather than `zarr://` so Neuroglancer
does not probe for v2 metadata first. The `/data/` prefix is no longer
generated by the UI; it keeps working for saved links.

## 7. Versions and docs

- `fingerprint.SCHEMA_VERSION = 4`, `DERIVATION_VERSION = 2`.
- `fingerprint.py` module-list comment and `CLAUDE.md` gain `omezarr.py`.
- `README.md`: layout, `--formats`, both Neuroglancer URL forms, the
  "serving cache, not a standalone store" caveat, the rebuild note.
- Existing caches are invalidated wholesale; the README's advice to build into
  a fresh `--cache-root` and swap applies.

## 8. Error handling

Unchanged ground rule: missing, stale, incompatible, outdated, corrupt, or
format-absent all read as "no cache" and the endpoint serves single-resolution.
Header parse errors return the existing 422 body. Level-0 read errors map as
they do for precomputed (`ChunkOutOfBounds` → 404, `UnexpectedEOF` → 500 with
an error log).

## 9. Testing

Unit (`tests/test_omezarr.py`):
- group JSON: axes, per-level scale and translation values, image-stack z
  handling, `mrcng` attributes.
- array JSON: shape/chunk reversal to `(z, y, x)`, `data_type` for each served
  dtype.
- `pad_chunk`: interior chunk returned without copy; edge chunk zero-padded to
  `chunk_shape`.
- chunk path for a grid index.

Builder (`tests/test_pyramid.py` additions):
- default build writes both layouts; `formats` in fingerprint.
- for every interior chunk, `omezarr` bytes equal the `precomputed` bytes of the
  same region; edge chunks equal the precomputed chunk after padding.
- `--formats omezarr` alone produces byte-identical level-2+ chunks to a
  both-formats build (exercises the padded-read path).
- rebuild with a different format set drops the other directory and updates
  `formats`.
- **Conformance:** open `cache_dir / "omezarr"` with `zarr` (Python, v3;
  test-only dependency added to the `test` pixi feature) and read level 1 as an
  array; compare to `block_mean` of the source. This is the one check that our
  hand-written metadata is spec-conformant to a real implementation.

Server (`tests/test_server_omezarr.py`):
- cached group and array `zarr.json` served verbatim with the fingerprint ETag.
- uncached group lists only level 0; level ≥ 1 metadata and chunks 404.
- level-0 edge chunk is padded to full `chunk_shape`; interior chunk equals the
  precomputed scale-0 chunk bytes.
- level ≥ 1 chunk served with immutable `Cache-Control`; out-of-grid index 404;
  `formats` lacking `omezarr` serves single level even when precomputed is valid.
- `zarr.json` disambiguation: a file whose name is a digit string still gets a
  group.
- `/precomputed/` and `/data/` return identical bodies.

CLI (`tests/test_cli.py`): `--formats` parsing, rejection of unknown names,
`status` output with formats and `incomplete`.

Browse (`tests/test_browse.py`): two links per file with the expected prefixes.

Fingerprint (`tests/test_fingerprint.py`): schema-3 fingerprints read as
`INCOMPATIBLE`.

Manual (`notes/TestPlan.md` addition): load one tomogram and one tilt series on
both endpoints in Neuroglancer; check features align across zoom levels
(validates the translation choice), scale bar matches, and no seams at edges.
