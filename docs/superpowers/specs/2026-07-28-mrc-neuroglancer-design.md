# Design: MRC → Neuroglancer Precomputed Service

Source of truth for this project. Adapted from the user-authored plan at
`notes/mrc-neuroglancer-implementation-plan.md` (kept locally, not tracked in
git). This document is the tracked, authoritative version; where it corrects
or extends the notes plan, this document wins.

Two deliverables sharing one library, `mrcng`:

1. **`mrc-pyramid`** — an offline CLI that walks a directory tree of MRC
   tomograms and writes a downsample pyramid into a cache tree outside the
   source directories, with a fingerprint linking each cache entry to its
   source file.
2. **`mrc-server`** (a FastAPI app, run via `uvicorn`) — speaks the
   Neuroglancer `precomputed` protocol. Scale 0 is read directly from the MRC
   file with `pread`. Scales 1..N are served from the cache **if and only if**
   a valid fingerprinted cache exists. If not, the service advertises a
   single-resolution volume and never computes anything on the request path.

## 0. Ground rules

- The service never writes to the cache and never downsamples. Missing/stale/
  corrupt cache → `info` has exactly one scale. No lazy generation, no
  background jobs, no fallback computation.
- The source tree is read-only. Never open an MRC for writing or create files
  inside it.
- Scale 0 always comes from the MRC, even when a cache exists. The cache holds
  scales 1..N only.
- Fail closed on ambiguity: non-standard headers raise a clear error naming
  the field and value. Do not guess.
- Target Python 3.11+. Runtime deps: `fastapi`, `uvicorn`, `numpy`,
  `pydantic-settings`. Dev only: `pytest`, `mrcfile`, `httpx`. No other
  dependencies are added (see §Scope decisions on metrics, below).
- `mrcfile` is a test oracle, not a runtime dependency (it memory-maps; the
  whole point of this design is to avoid that). Header parsing is hand-rolled
  with `struct`; tests assert agreement with `mrcfile` **except** for the one
  case documented in §2.3 below, where we deliberately diverge from it.

## 1. Package layout

```
src/mrcng/
  __init__.py
  mrcheader.py      # MRC header parsing, dtype table, geometry
  paths.py          # dataset id, path safety, cache locations
  fingerprint.py    # fingerprint compute / write / validate
  precomputed.py    # info JSON, scale keys, chunk names, chunk encoding
  reader.py         # pread-based chunk extraction from MRC
  downsample.py     # block-mean downsampling
  pyramid.py        # pyramid build orchestration (used by CLI only)
  cli.py            # mrc-pyramid entry point
  benchmark.py       # load-test script (M5)
  server/
    __init__.py
    app.py          # FastAPI app + routes
    config.py       # pydantic-settings
    fdcache.py      # bounded LRU of open fds + parsed headers
tests/
```

## 2. `mrcheader.py`

Parse the fixed 1024-byte MRC2014 header with `struct.unpack`, read via a
single `os.pread(fd, 1024, 0)`.

### 2.1 Field offsets (verified against `mrcfile.dtypes.HEADER_DTYPE`)

The notes plan's offset table for `exttyp`/`nversion` was wrong (it collided
with `map`/`machst`); the table below is the corrected, `mrcfile`-verified
layout:

| Offset (bytes) | Name | Meaning |
|---|---|---|
| 0 | `nx, ny, nz` | int32 ×3, columns / rows / sections |
| 12 | `mode` | int32, data type code |
| 16 | `nxstart, nystart, nzstart` | int32 ×3, ignored for v1 (record in metadata) |
| 28 | `mx, my, mz` | int32 ×3, grid size |
| 40 | `cella` | float32 ×3, cell dimensions in **ångström** |
| 52 | `cellb` | float32 ×3, cell angles (unused) |
| 64 | `mapc, mapr, maps` | int32 ×3, axis correspondence |
| 76 | `dmin, dmax, dmean` | float32 ×3 (unused) |
| 88 | `ispg` | int32 (unused) |
| 92 | `nsymbt` | int32, extended header length in bytes |
| 96 | `extra1` | 8 bytes, vendor scratch space |
| 104 | `exttyp` | 4-char extended header type |
| 108 | `nversion` | int32 |
| 112 | `extra2` | 84 bytes vendor scratch space; **IMOD stamp lives here** (§2.3) |
| 196 | `origin` | float32 ×3 (unused) |
| 208 | `map` | 4 bytes, should read `"MAP "` |
| 212 | `machst` | 4 bytes, machine/byte-order stamp |
| 220 | `nlabl` | int32 (unused) |
| 224 | `label` | 10×80 chars (unused) |

### 2.2 Derived geometry

```
data_offset = 1024 + nsymbt
itemsize    = DTYPE_TABLE[mode].itemsize
offset(x,y,z) = data_offset + (z*ny*nx + y*nx + x) * itemsize
```

Data is C-order with **x fastest, z slowest**. Put this in the module
docstring.

### 2.3 Dtype table and mode-0 signedness

| mode | numpy dtype | notes |
|---|---|---|
| 0 | `int8` or `uint8` | ambiguous — see below |
| 1 | `int16` | **primary target for v1** |
| 2 | `float32` | |
| 6 | `uint16` | |
| 12 | `float16` | |
| 3, 4 | — | complex; raise `UnsupportedModeError` |

Mode 0 signedness: `mrcfile.dtype_from_mode` always returns `int8` for mode 0
— it does not implement IMOD-aware dispatch. We deliberately go further,
because the ground rule is correctness against real tomography data, not
agreement with `mrcfile` for its own sake:

- Read `imodStamp` = int32 at byte 152. If it equals `1146047817`
  (`0x444F4D49`, ASCII `"IMOD"`), the file sets IMOD conventions.
- Then read `imodFlags` = int32 at byte 156. Bit `0x1` set → bytes are signed
  (`int8`); bit `0x1` clear → unsigned (`uint8`).
- If `imodStamp` is absent or doesn't match, default to `int8` and log a
  warning naming the file (this matches `mrcfile`'s default, so the two agree
  in the common case). Add a `--assume-mode0 {int8,uint8}` CLI override.

This means unit tests assert agreement with `mrcfile` for every mode and
field **except** mode-0 files with an IMOD unsigned-byte stamp, where a
dedicated test asserts our own expected `uint8` output against a synthetic
fixture (not `mrcfile`, since it doesn't model this case).

All values are little-endian in practice. Detect big-endian by
sanity-checking `mode` in `{0,1,2,3,4,6,12}` after little-endian unpack; if it
fails, retry big-endian and raise `UnsupportedByteOrderError` — do not
silently support it in v1.

### 2.4 Validation performed at open

- `mapc, mapr, maps == (1, 2, 3)`. Anything else → `NonStandardAxisOrderError`
  naming the actual values.
- `nx, ny, nz > 0`.
- `nsymbt >= 0` and `data_offset + nx*ny*nz*itemsize <= file_size`. If short,
  raise — never serve truncated data as zeros.
- Voxel size: `voxel_size_angstrom = cella / (mx, my, mz)`, guarding division
  by zero. If `mx/my/mz` are zero or `cella` is zero, fall back to `(1,1,1)`
  Å and set a `voxel_size_is_default` flag surfaced in `info` metadata and
  the CLI report.

Expose a frozen dataclass `MrcHeader` with all of the above plus `file_size`,
`mtime_ns`.

## 3. `paths.py`

### 3.1 Dataset identity

```python
dataset_id(relpath: PurePosixPath) -> str  # sha256(str(relpath).encode()).hexdigest()[:16]
```

Deterministic from the source-relative path — no index file needed. Cache
layout:

```
<cache_root>/<dataset_id[:2]>/<dataset_id>/
    fingerprint.json
    info
    2_2_1/0-64_0-64_0-64
    4_4_2/...
```

`fingerprint.json` records the original relative path so the cache tree stays
debuggable and a reverse index can be rebuilt by scanning.

### 3.2 Path safety (security-critical)

```python
def resolve_source(root: Path, relpath: str) -> Path
```

- Reject any component that is `..`, absolute, empty, or contains a null byte.
- `candidate = (root / relpath).resolve()`; require
  `candidate.is_relative_to(root.resolve())`.
- Require `candidate.is_file()`.
- Symlink escape is covered by the `resolve()` + `is_relative_to` check;
  add an explicit test with a symlink pointing outside the root.

Raise `PathNotAllowed` → HTTP 404 (not 403 — don't confirm existence of
out-of-tree paths).

## 4. `fingerprint.py`

```json
{
  "schema_version": 1,
  "generator_version": "mrc-pyramid 0.1.0",
  "source_relpath": "session42/tomo_0031.mrc",
  "source_size": 8589934592,
  "source_mtime_ns": 1751030400000000000,
  "source_header_sha256": "…",
  "params": {
    "chunk_size": [64, 64, 64],
    "downsample": "mean",
    "min_axis_size": 32,
    "max_levels": 6,
    "dtype": "int16",
    "encoding": "raw"
  },
  "scales": ["2_2_1", "4_4_2", "8_8_4"],
  "built_at": "2026-07-28T10:00:00Z",
  "build_duration_s": 412.3
}
```

`source_header_sha256` is the SHA-256 of the first `1024 + nsymbt` bytes.

`validate(fp, header, current_params) -> Validity` returns `VALID`, `STALE`
(size/mtime/header hash differ from the live file), or `INCOMPATIBLE`
(`schema_version` unknown, or `params` differ in a way that changes chunk
addressing). The server treats all non-`VALID` outcomes identically (no
cache), logging them distinctly. `generator_version` changing alone does
**not** invalidate; only `schema_version` does.

**Commit semantics:** `fingerprint.json` is written last, after every chunk
and `info` are on disk and the directory is fsynced. Its presence is the only
signal a cache entry is complete. A build that dies partway leaves a
directory with no fingerprint — read as "no cache" and overwritten on the
next run.

## 5. `precomputed.py`

### 5.1 Scale keys and sizes

Level 0 has factors `(1,1,1)`, key `"1_1_1"`. Each subsequent level doubles an
axis's factor **only if** that axis's current size is `> min_axis_size`
(default 32) — handles the anisotropic case normal in tomography.

```python
size_{L+1}[i] = ceil(size_L[i] / factor_step[i])
```

Use `ceil` identically in the generator and in `info`; an off-by-one here
produces black seams at the far edge of the volume. Stop when all axes are at
their floor, or `max_levels` is reached. Scale key is `"{fx}_{fy}_{fz}"` of
cumulative factors.

### 5.2 `info` JSON

```json
{
  "@type": "neuroglancer_multiscale_volume",
  "type": "image",
  "data_type": "int16",
  "num_channels": 1,
  "scales": [
    {
      "key": "1_1_1",
      "size": [4096, 4096, 512],
      "resolution": [0.68, 0.68, 0.68],
      "voxel_offset": [0, 0, 0],
      "chunk_sizes": [[64, 64, 64]],
      "encoding": "raw"
    }
  ]
}
```

`resolution` is in **nanometres**; MRC `cella` is in ångström; divide by 10.
Level *L* resolution = level 0 resolution × cumulative factor, per axis.

### 5.3 Chunk naming

```
{key}/{x0}-{x1}_{y0}-{y1}_{z0}-{z1}
```

Bounds are clipped to the scale's `size`; edge chunks are smaller than
`chunk_size`. The server computes the clipped extent from the requested name
and validates it matches the grid exactly — a request for `0-64_...` on a
volume of z-size 40 must be `0-40`, never padded.

### 5.4 Chunk encoding (`raw`)

Little-endian, x fastest, then y, then z, then channel. A numpy array shaped
`(z, y, x)`, C-contiguous, little-endian → `arr.tobytes()` is exactly correct.
Assert `C_CONTIGUOUS` and byte order before writing. Test with a volume where
`value = x + 1000*y + 1000000*z`, verifying the byte at each position.

## 6. `reader.py` — pread chunk extraction

```python
def read_chunk(fd, hdr, x0, x1, y0, y1, z0, z1) -> np.ndarray
```

Returns shape `(z1-z0, y1-y0, x1-x0)`, dtype from the header, C-contiguous.

Two strategies chosen by a threshold: **row-wise** (one `pread` per `(z,y)`,
exact, no over-read) vs **span-wise** (one `pread` per `z`, covering
`x0..x1` within the full row, then sliced — over-reads but far fewer
syscalls).

```python
row_bytes = (x1 - x0) * itemsize
strategy = ROW_WISE if row_bytes >= 4096 else SPAN_WISE
```

Span-wise's over-read isn't wasted: Neuroglancer requests all x-chunks of a
given `(y, z)` tile range together, so the over-read lands in page cache and
satisfies neighbours. Threshold is configurable; record which strategy was
used in a debug header for in-situ benchmarking.

`pread_exact(fd, count, offset) -> bytes` loops until satisfied, raises
`UnexpectedEOF` on a 0-length return. Every read goes through this — never
call `os.pread` directly elsewhere.

Clip the requested region to the volume before reading. Empty clipped region
→ `ChunkOutOfBounds` → 404.

## 7. `downsample.py`

```python
def block_mean(arr: np.ndarray, factors: tuple[int, int, int]) -> np.ndarray
```

- Accumulate in `int32` for integer inputs (`float32` accumulates in
  `float64`) — 8×int16 max overflows int16.
- Non-divisible edges: the trailing partial block averages over however many
  voxels it actually contains. Implement via `np.add.reduceat` plus a
  per-block count array (less code, less error-prone than manual reshaping).
- Round half away from zero, clip to dtype range, cast back.
- Never `astype` a memmap — no memmaps exist here, but state it in the
  docstring.

## 8. `pyramid.py` + `cli.py`

```
mrc-pyramid build SOURCE_ROOT --cache-root CACHE \
    [--glob '*.mrc' --glob '*.rec'] \
    [--jobs 4] [--chunk-size 64,64,64] [--min-axis-size 32] [--max-levels 6] \
    [--max-block-bytes 256M] [--force] [--dry-run] [--assume-mode0 {int8,uint8}] \
    [--report report.jsonl] [--log-level INFO]

mrc-pyramid status SOURCE_ROOT --cache-root CACHE
mrc-pyramid prune --cache-root CACHE --source-root SOURCE
```

Per-file algorithm: open read-only → parse/validate header → compute
`dataset_id`/cache dir/fingerprint; skip (`SKIPPED_VALID`) if valid and not
`--force` → acquire exclusive `flock` on `<cache_dir>/.lock`, skip
(`SKIPPED_LOCKED`) if already held → delete any existing `fingerprint.json`
immediately → compute scale plan → build level 1 from the MRC by streaming →
build level *L* from level *L-1* (never directly from level 0 except level 1;
cascade costs ~1.15 passes over the original, not N) → write `info`
(including scale 0's descriptor) → fsync the directory tree → write+fsync
`fingerprint.json` → release lock → report `BUILT`.

Streaming level 1 iterates the output chunk grid in `(z, y)` order,
processing all x-chunks of a row together, splitting the x range so each
piece stays under `--max-block-bytes`. Peak RSS ≈ `max_block_bytes × jobs`.

Reporting: one JSON object per file per line (relpath, dataset_id, status,
source/cache bytes, levels built, duration, error).

Parallelism: `multiprocessing.Pool` over files (I/O-bound streaming within a
file, parallelises across files). Default `--jobs` = `min(4, cpu_count())` —
storage is usually the bottleneck; oversubscribing NFS hurts.

## 9. `mrc-server` — the FastAPI service

### 9.1 Configuration (`pydantic-settings`, env-prefixed `MRCNG_`)

| Setting | Default | Notes |
|---|---|---|
| `source_root` | — | required |
| `cache_root` | — | required |
| `chunk_size` | `64,64,64` | must match the cache's params or reads as `INCOMPATIBLE` |
| `max_concurrent_reads` | 32 | semaphore around threadpool I/O |
| `fd_cache_size` | 256 | keep well under `ulimit -n` |
| `cors_origins` | `*` | Neuroglancer requires CORS unless same-origin |

### 9.2 Routing

```python
@app.get("/data/{full_path:path}")
async def dispatch(full_path: str): ...
```

Manual dispatch (FastAPI's `:path` converter is greedy, don't declare
overlapping routes): last segment `info` → relpath is everything before it;
last segment matches `^\d+-\d+_\d+-\d+_\d+-\d+$` and the one before matches
`^\d+_\d+_\d+$` → scale key + chunk spec, relpath is everything before them;
else 404. Validate the parsed chunk spec against the scale grid before
touching the filesystem.

### 9.3 `GET /data/{relpath}/info`

Resolve/safety-check path (404 on failure) → get header from fd/header cache
→ compute `dataset_id`, look for `fingerprint.json`. If present and `VALID`:
return the cached `info` bytes verbatim. Else: generate `info` with exactly
one scale (`"1_1_1"`) from the header; log the reason (missing/stale/
incompatible) at INFO, rate-limited per `(path, mtime)`.

`Content-Type: application/json`; `Cache-Control: no-cache, must-revalidate`;
`ETag` from the fingerprint hash or source mtime.

### 9.4 `GET /data/{relpath}/{key}/{chunk}`

`key == "1_1_1"` → read from the MRC (resolve path, get fd+header, validate
chunk extent against the scale-0 grid, `asyncio.to_thread(read_chunk, ...)`
under the concurrency semaphore, `arr.tobytes()`). Any other key → serve
`<cache_dir>/<key>/<chunk>` as a file, only after confirming the fingerprint
is `VALID`; 404 if absent.

Headers: `Content-Type: application/octet-stream`;
`Cache-Control: public, max-age=31536000, immutable`; `ETag`.

### 9.5 fd + header cache (`fdcache.py`)

`OrderedDict` keyed by `(resolved_path, size, mtime_ns)` → `(fd, MrcHeader)`.
On eviction, `os.close(fd)`. Guard with a `threading.Lock`. A replaced source
file misses the cache and is reopened — no stale fd serving old data. Cap at
`fd_cache_size`; log a warning if eviction rate is high.

### 9.6 Concurrency

`asyncio.Semaphore(max_concurrent_reads)` around the threadpool call — not
for CPU (`pread` releases the GIL) but to avoid queueing hundreds of
concurrent reads at the storage layer (NFS timeouts).

### 9.7 Errors and observability

- `PathNotAllowed`, `ChunkOutOfBounds`, missing file → 404, empty body.
- Malformed chunk spec → 400.
- Header validation failure (`NonStandardAxisOrderError`, truncated file,
  unsupported mode) → 422 with a short JSON body naming the problem.
- `UnexpectedEOF` mid-read → 500, logged loudly (file changed under us).
- `/healthz` → 200 with version and config summary.
- Structured logs (JSON): relpath, key, chunk, bytes, strategy, duration,
  cache hit/miss, fd-cache hit rate, semaphore wait time.

## 10. Scope decisions (this document's additions to the notes plan)

- **No `/metrics` (Prometheus) endpoint.** The notes plan lists it as
  optional, but it isn't achievable within the pinned dependency list
  (§0). Structured JSON logs carry the same fields instead. User-confirmed.
- **nginx config** ships as a documented snippet in the README only — no
  nginx available in this environment to test against. Correspondingly, the
  server always serves cached chunks via Starlette's `FileResponse` (which
  already uses `sendfile`); there is no `X-Accel-Redirect` code path or
  config toggle for it — that's a deployment-time nginx optimization the
  README documents, not a branch the app needs to carry and test.
- **Benchmark script** (`mrcng/benchmark.py`) uses stdlib + `httpx` only, run
  via `pixi run benchmark`.
- Everything in the notes plan's "Deliberately out of scope" section (int8/
  uint8 display path, sharded precomputed format, gzip content-encoding,
  inotify watcher, cache LRU eviction, auth, multi-channel/time-series MRC)
  stays out.
- A private real-data path was shared for manual smoke testing only; it is
  never written into README, tests, comments, or commit messages.

## 11. Project scaffolding

- Single `pyproject.toml` with `[tool.pixi.*]` sections — no separate
  `pixi.toml`. `src/mrcng/` layout; `hatchling` packages `["src/mrcng"]`.
- One pixi environment (`default`) containing runtime + dev/test deps, so
  every `pixi run` task works without `--environment` flags.
- Pixi tasks: `build-cache`, `pyramid-status`, `pyramid-prune`, `serve`,
  `test`, `benchmark`.
- Local git repo, no remote. `.gitignore` excludes `/notes/`, `.pixi/`,
  build artifacts, and `*.mrc`. Commit incrementally per milestone.

## 12. Tests

Fixtures: a `make_mrc(path, shape, dtype, voxel_size_angstrom, nsymbt=0,
imod_flags=None, fill=...)` helper writing valid MRC files, covering: a
non-empty extended header, odd dimensions (`101×97×53`), anisotropic
(`2048×2048×64`), truncated, and IMOD-stamped signed/unsigned mode-0.

- Unit: header parse agrees with `mrcfile` on every fixture except the IMOD
  mode-0 case (§2.3); offset arithmetic against `mrcfile`'s indexing for
  random `(x,y,z)`; chunk encoding byte order via the
  `x + 1000y + 1000000z` volume; `block_mean` on constant/ramp volumes
  including non-divisible edges; scale planning `ceil` agreement between
  plan and generator; both read strategies produce identical arrays;
  `pread_exact` loop correctness with a mocked short-returning pread.
- Security: `..` traversal, absolute paths, null bytes, symlink escape all
  yield 404.
- Integration: pyramid build vs. in-memory reference downsample; uncached
  server (`info` has one scale, scale-0 matches source, `2_2_1` 404s);
  cached server (`info` has all scales, cached chunks byte-identical);
  fingerprint invalidation on source mtime bump (**highest-value test** —
  the failure mode that silently serves wrong data); killed build (no
  fingerprint) behaves as uncached; two concurrent `mrc-pyramid` runs don't
  corrupt each other (second reports `SKIPPED_LOCKED`).
- Manual acceptance (not automated): load
  `precomputed://http://localhost:8000/data/<relpath>` in Neuroglancer for
  an uncached and a cached file; verify zoom, no edge seams, correct nm
  scale bar.

## 13. Milestones

1. **M1** — `mrcheader`, `paths`, `reader`, `precomputed`; unit tests green
   against `mrcfile`.
2. **M2** — Server, scale 0 only, no cache logic. Verify in Neuroglancer —
   de-risks the whole design; get here fast.
3. **M3** — `downsample`, `pyramid`, CLI, fingerprints. Verify output
   against an in-memory reference.
4. **M4** — Cache-aware server: fingerprint validation, cached `info`,
   cached chunk serving, staleness tests.
5. **M5** — Hardening: fd cache, semaphore, structured logs, `/healthz`,
   nginx doc snippet, benchmark script.
