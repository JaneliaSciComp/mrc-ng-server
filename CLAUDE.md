# mrc-ng-server

## Before you finish: does your change need a `DERIVATION_VERSION` bump?

The pyramid cache serves `info` and chunks **verbatim** from what a build wrote.
Nothing detects that the code which produced them has changed — so if you alter
what a build produces, you must bump `DERIVATION_VERSION` in
`src/mrcng/fingerprint.py` or every existing cache entry keeps being served as
valid, indefinitely, with no warning from `mrc-pyramid status` and no failing
test.

The test: **can your change alter the values in `info` or the bytes in a chunk
file?** Today that means `mrcheader.py`, `precomputed.py`, `downsample.py`,
`pyramid.py` and `reader.py` — but apply the question, not the list, if you add a
module. Bump if you changed any of:

- the voxel size, `data_type`, or anything else that appears in `info`
- the scale plan (which levels exist, their sizes or factors)
- the chunk bytes (downsampling maths, encoding, dtype widening)

Do **not** bump for changes to `server/`, `cli.py`, `benchmark.py`, comments,
docstrings, or tests — those cannot change what a build wrote.

If you changed the *shape* of `fingerprint.json` itself — a key added, removed,
renamed or retyped — that is `SCHEMA_VERSION`, not `DERIVATION_VERSION`. The two
are contrasted in full at the top of `src/mrcng/fingerprint.py`.

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
- **Image-stack classification is operator config, not inference.** Which files
  have a z axis that is a slice index comes from `--stack-glob` / `--volume-glob`
  (`MRCNG_STACK_GLOBS` / `MRCNG_VOLUME_GLOBS`), matched in
  `mrcheader.classify_path`. Volume globs win over stack globs. Nothing in an MRC
  header can answer this and shape cannot either — an aspect-ratio heuristic used
  to live here and was wrong on real files. The builder records its answer in the
  fingerprint, so the server and `mrc-pyramid` must be given the same globs or
  uncached files classify differently; cached files carry the answer with them.

## Tests

`pixi run -e default pytest -q`
