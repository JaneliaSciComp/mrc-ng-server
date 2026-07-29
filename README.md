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

A `.env` file in the repo root also works (pydantic-settings loads it via
the environment).

## Building the cache

```bash
pixi run build-cache /path/to/tomograms --cache-root /path/to/cache
```

Extra arguments pass straight through to `mrc-pyramid build`:

```bash
pixi run build-cache /path/to/tomograms --cache-root /path/to/cache \
    --glob '*.mrc' --glob '*.rec' --jobs 4 \
    --chunk-size 64,64,64 --min-axis-size 32 --max-levels 6 \
    --report report.jsonl
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
   Neuroglancer data source URL is:

   ```
   precomputed://https://your-host:8000/data/some/relative/path.mrc
   ```

   (no `/info` suffix — Neuroglancer appends that itself).
3. Open a Neuroglancer instance (e.g. https://neuroglancer-demo.appspot.com/,
   or a self-hosted build) and add a new layer with source type
   **precomputed**, pasting the URL above.
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

## Browsing data in a web browser

Instead of constructing Neuroglancer URLs by hand, browse `MRCNG_SOURCE_ROOT`
directly:

```
https://your-host:8000/browse
```

Click through subdirectories; every `.mrc`/`.rec` file gets an "Open in
Neuroglancer" link that opens `https://neuroglancer-demo.appspot.com` with an
`"auto"`-type layer already pointing at that file — the same URL you'd build
by hand per the section above, generated for you.

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
location /data/ {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;

    # Chunk responses are already marked immutable by the app; let nginx
    # (or an upstream CDN) cache them accordingly.
    proxy_cache_valid 200 365d;
    add_header Cache-Control "public, max-age=31536000, immutable" always;
}
```
