# mrc-ng-server

Serves MRC/REC tomograms to [Neuroglancer](https://github.com/google/neuroglancer) via its `precomputed` protocol.

There are two parts:

- **`mrc-pyramid`** — an offline precomputation that walks a directory of MRC files and
  writes a downsample pyramid for each one into a cache, fingerprinted against the
  source file it was built from.
- **The FastAPI server** — serves scale 0 directly from the MRC file on every
  request (`pread`, no memory-mapping, no full-file load). Scales 1..N are
  served from the cache **only if** a valid, matching fingerprint exists;
  otherwise the server advertises a single-resolution volume and never
  downsamples or writes anything on the request path.

See [docs/superpowers/specs/2026-07-28-mrc-neuroglancer-design.md](https://github.com/JaneliaSciComp/mrc-ng-server/blob/main/docs/superpowers/specs/2026-07-28-mrc-neuroglancer-design.md) for the
full design, and [docs/mrc-layout-and-reads.md](https://github.com/JaneliaSciComp/mrc-ng-server/blob/main/docs/mrc-layout-and-reads.md) for why MRC suits this
and how chunk reads actually work (read that before touching `reader.py`).

## Install

Requires [pixi](https://pixi.sh/).

```bash
pixi install
```

This creates a single `default` environment with both runtime and test
dependencies (`fastapi`, `uvicorn`, `numpy`, `pydantic-settings`, `pytest`,
`mrcfile`, `httpx`) and installs the `mrcng` package editable.

## Configuration

The server and CLI read settings from environment variables prefixed
`MRCNG_` (via `pydantic-settings`). At minimum you need:

```bash
export MRCNG_SOURCE_ROOT=/path/to/tomograms   # read-only root of MRC files
export MRCNG_CACHE_ROOT=/path/to/cache        # where mrc-pyramid writes pyramids
```

Other settings (all optional, with defaults):

| Variable | Default | Meaning |
|---|---|---|
| `MRCNG_CHUNK_SIZE` | `64,64,64` | Must match whatever `mrc-pyramid` was run with, or caches read as incompatible |
| `MRCNG_MAX_CONCURRENT_READS` | `32` | Semaphore around threadpool MRC reads |
| `MRCNG_FD_CACHE_SIZE` | `256` | Max open file descriptors kept warm (keep well under `ulimit -n`) |
| `MRCNG_CORS_ORIGINS` | `*` | Neuroglancer needs CORS unless the viewer is served same-origin |
| `MRCNG_STACK_GLOBS` | *(empty)* | Comma-separated fnmatch patterns for files whose z is a slice index. Must match what `mrc-pyramid` built with |
| `MRCNG_VOLUME_GLOBS` | *(empty)* | Comma-separated patterns forcing 3D-volume treatment; wins over `MRCNG_STACK_GLOBS` |

A `.env` file in the repo root also works (pydantic-settings loads it via
the environment).

## Building the cache

```bash
pixi run build-cache --source-root /path/to/tomograms --cache-root /path/to/cache
```

Extra arguments pass straight through to `mrc-pyramid build`:

```bash
pixi run build-cache --source-root /path/to/tomograms --cache-root /path/to/cache \
    --glob '*.mrc' --glob '*.rec' --jobs 4 \
    --chunk-size 64,64,64 --min-axis-size 32 --max-levels 6 \
    --report report.jsonl
```

For local dev, `--under some/subdir` scopes the glob walk to a subdirectory of
`--source-root` (relpaths stay relative to the full `--source-root`, so the
cache resolves the same as a full-tree build) — a quick way to build a handful
of tomograms without listing them one by one via `--from-file`:

```bash
pixi run build-cache --source-root /path/to/tomograms --cache-root /path/to/cache \
    --under Experimental/some_lab/some_sample
```

Safe to re-run: files with a valid, up-to-date cache are skipped
(`SKIPPED_VALID`); pass `--force` to rebuild anyway. Two concurrent runs
over the same tree don't corrupt each other — the second reports
`SKIPPED_LOCKED` for any file the first is already building.

Check cache status per file, or remove cache entries whose source file is
gone:

```bash
pixi run pyramid-status /path/to/tomograms --cache-root /path/to/cache
pixi run pyramid-prune --cache-root /path/to/cache --source-root /path/to/tomograms
```

## Running the server

```bash
pixi run serve
```

Starts uvicorn on `0.0.0.0:8000` over HTTPS, using the shared cert at
`/opt/certs/{cert.crt,cert.key}` — Neuroglancer runs in-browser and won't load
a plain-HTTP data source from a page served over HTTPS (mixed content), so TLS
here isn't optional. Reads `MRCNG_SOURCE_ROOT` / `MRCNG_CACHE_ROOT` (and the
other `MRCNG_*` settings) from the environment.

Check it's up:

```bash
curl -k https://localhost:8000/healthz
```

(`-k` because the shared cert isn't necessarily issued for `localhost`; check
from the real hostname to validate normally.)

## Tests

```bash
pixi run test
```

## Benchmark

Once the server is running:

```bash
pixi run benchmark --base-url https://localhost:8000 \
    --relpath some/relative/path.mrc --concurrency 8 --requests-per-dataset 20
```

Reports p50/p95/p99 latency (ms) for scale-0 chunk requests. Run before and
after tuning `MRCNG_MAX_CONCURRENT_READS` or `MRCNG_FD_CACHE_SIZE`, not
before you have a number to compare against.

## Loading data into Neuroglancer

1. Start the server (above) and note its base URL, e.g. `https://your-host:8000`.
2. For a file at `<MRCNG_SOURCE_ROOT>/some/relative/path.mrc`, the
   Neuroglancer data source URL is one of:

   ```
   precomputed://https://your-host:8000/precomputed/some/relative/path.mrc
   zarr3://https://your-host:8000/omezarr/some/relative/path.mrc
   ```

   (no `/info` or `/zarr.json` suffix — Neuroglancer appends those itself).
   `/data/` is a legacy alias for `/precomputed/` and keeps working for saved
   links.
3. Open a Neuroglancer instance (e.g. https://neuroglancer-demo.appspot.com/,
   or a self-hosted build) and add a new layer, pasting one of the URLs above.
4. What to expect:
   - **No cache built yet**: a single-resolution image layer. Correct at
     full zoom, but there's nothing to zoom out to smoothly — Neuroglancer
     is reading directly from the MRC file.
   - **Cache built and valid** (via `pixi run build-cache`): a
     multi-resolution volume. Zooming out should be smooth with no seams at
     the volume edges, and the scale bar should read in nanometres matching
     the MRC voxel size (ångström ÷ 10).
5. If a cache goes stale (source file rebuilt/modified) or is deleted,
   `/info` automatically drops back to a single scale and requests for
   higher scales 404 — reload the layer in Neuroglancer to pick that up.

### Cache invalidation

`/info` for a cached file is the **verbatim artifact the build wrote**, not a
recomputation — so it always describes exactly the chunk files on disk. A cache
entry is invalidated automatically by either of:

- **the source header changing** — size, mtime, or a sha256 over the header
  (including the extended header) differs → `stale`
- **build-determining code changing** — `fingerprint.DERIVATION_VERSION`, bumped
  by hand when a change alters the voxel size or data_type in `info`, the scale
  plan, the chunk bytes, or the encoding, differs → `outdated`

`mrc-pyramid status` reports both states per file. A plain `mrc-pyramid build`
rebuilds anything that isn't `valid` — `--force` is only needed to rebuild a
still-valid entry.

Two consequences worth knowing:

- **`DERIVATION_VERSION` must be bumped by whoever changes a derivation.** It is
  not computed. Miss it and both the cached `info` and the cached chunks keep
  being served as valid, with no warning from `status` and no failing test —
  which is how a zero-`cella_z` tilt stack once served
  `"resolution": [.., .., 0.0]` for weeks after the fix landed. A full rebuild
  of the 1.38 TiB Janelia tree is ~5.4 core-hours, under 90 minutes at
  `--jobs 4`, so bumping when unsure is much cheaper than not bumping.
- **Invalidation is synchronised.** Every entry expires at once, and the server
  never builds on the request path, so the whole corpus serves
  single-resolution until the rebuild catches up. To avoid that window, build
  into a fresh `--cache-root` and swap it in.

Voxel data is never hashed — that would mean reading the entire corpus on every
validation — so `valid` means "the header is unchanged", not "no byte of the
file changed".

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

### Image stacks (tilt series, gain references)

A tilt series' z axis is a tilt *index*, not a spatial axis, so it gets
handled differently from a tomogram's:

- z resolution is advertised as **1 nm per slice**, ignoring whatever the
  writer stamped on `cella_z`. Honouring it made 55 tilts 8.3 nm "thick"
  against a 621 nm-wide image — a 75× squash that left the tilt axis
  unnavigable. `info` carries a non-spec `"is_image_stack": true` when this
  applies.
- z is **never downsampled**: scale keys go `2_2_1`, `4_4_1`, … so a level
  never averages tilts taken at different angles into one plane.
- the `(mx,my,mz) != (nx,ny,nz)` grid check doesn't apply, since `mz=1`
  regardless of `nz` is a normal convention for these files.

Which files those are is **operator configuration**, not inference — no MRC
header field can answer it, and shape cannot either (measured over 3648 corpus
files, true 2D span `nz/max(nx,ny)` 0.0001–0.2200 and true 3D span 0.1276–1.4120,
so the classes overlap and no threshold is correct).

Set it at build time, and give the server the same values:

```bash
mrc-pyramid build ... \
    --stack-glob '*/TiltSeries/*' --stack-glob '*/Gains/*' \
    --volume-glob '*/Tomograms/*' --volume-glob '*_ctf.mrc'
```

`--volume-glob` wins over `--stack-glob`, because real trees mix both in one
directory: this corpus has `.../external/s200.mrc` (a 55-tilt stack) beside
`.../external/s200_ctf.mrc` (a 512×512×55 volume). Patterns are `fnmatch` against
the relpath, so `*` crosses `/` and `*/TiltSeries/*` means "anywhere under".

With no globs set nothing is a stack and z comes from `cella` as it always did.

The build records its answer in the fingerprint, so **changing the globs
invalidates exactly the entries it reclassifies** (`incompatible`) and a plain
`mrc-pyramid build` rebuilds them. If the server's globs disagree with the
build's, affected entries read as `incompatible` and it serves single-resolution
— safe, but silent, so keep the two in sync.

**Caches built before this landed must be rebuilt** (`--force`): their z-binned
scale keys are no longer advertised for a stack, so those files fall back to
single-resolution until rebuilt.

## Browsing data in a web browser

Instead of constructing Neuroglancer URLs by hand, browse `MRCNG_SOURCE_ROOT`
directly:

```
https://your-host:8000/browse
```

Click through subdirectories; every `.mrc`/`.rec` file gets two links,
"Neuroglancer (precomputed)" and "Neuroglancer (OME-Zarr)", each opening
`https://neuroglancer-demo.appspot.com` with an `"auto"`-type layer already
pointing at that file on the matching endpoint — the same URLs you'd build by
hand per the section above, generated for you.

This is deliberately minimal: no search, filtering, or cache-status
indicators (use `pixi run pyramid-status` for that), no JavaScript, and it
lives in its own module (`src/mrcng/server/browse.py`) separate from the
`precomputed` API. See
`docs/superpowers/specs/2026-07-29-browsing-ui-design.md` for the design.

## Optional: nginx in front of the server

Not required, and untested in this repo (no nginx available here), but for
production deployments a reverse proxy in front of uvicorn can offload
static chunk serving and add CDN-friendly caching:

```nginx
location ~ ^/(data|precomputed|omezarr)/ {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;

    # Chunk responses are already marked immutable by the app; let nginx
    # (or an upstream CDN) cache them accordingly.
    proxy_cache_valid 200 365d;
    add_header Cache-Control "public, max-age=31536000, immutable" always;
}
```
