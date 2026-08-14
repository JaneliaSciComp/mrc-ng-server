# mrc-ng-server

## Before you finish: does your change need a `DERIVATION_VERSION` bump?

The pyramid cache serves `info` and chunks **verbatim** from what a build wrote.
Nothing detects that the code which produced them has changed — so if you alter
what a build produces, you must bump `DERIVATION_VERSION` in
`src/mrcng/fingerprint.py` or every existing cache entry keeps being served as
valid, indefinitely, with no warning from `mrc-pyramid status` and no failing
test.

Bump it if you changed any of these in `mrcheader.py`, `precomputed.py`,
`downsample.py`, `pyramid.py` or `reader.py`:

- the voxel size, `data_type`, or anything else that appears in `info`
- the scale plan (which levels exist, their sizes or factors)
- the chunk bytes (downsampling maths, encoding, dtype widening)

Do **not** bump for changes to `server/`, `cli.py`, `benchmark.py`, comments,
docstrings, or tests — those cannot change what a build wrote.

When unsure, bump. A full rebuild of the 1.38 TiB Janelia tree is ~5.4
core-hours (under 90 minutes at `--jobs 4`); serving silently-wrong cached
metadata cost weeks last time.

## Other traps

- `hdr.dtype` is the on-disk dtype; `hdr.served_dtype` is the on-the-wire
  dtype. They differ only for MRC mode 12 (float16 widens to float32 because
  Neuroglancer can't render float16). Every byte offset/itemsize calculation
  must use `dtype`; only chunk bodies and `info` use `served_dtype`. Confusing
  them corrupts every read.
- Edge chunks are clipped, never padded. Neuroglancer requests the
  already-clipped extent (e.g. `0-64_0-64_0-40` for a volume with z-size 40).
  See `precomputed.py`'s module docstring.
- `pread` only, never mmap. The server never downsamples or writes on the
  request path — scales >= 1 come from the cache or 404.
- Do not retune `STACK_ASPECT_RATIO` in `mrcheader.py`. True 2D files span
  ratio 0.0001-0.2200 and true 3D files span 0.1276-1.4120 — the classes
  overlap, so no threshold is correct. It is knowingly wrong on 2 of 3648
  corpus files. The comment above `_is_image_stack` explains it.
- `/notes/` is gitignored; committed docs go in `docs/`.
- Read `docs/mrc-layout-and-reads.md` before touching `reader.py`.

## Tests

`pixi run -e default pytest -q`
