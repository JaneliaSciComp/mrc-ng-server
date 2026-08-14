# Design: the cached `info` becomes the single source of truth

## Goal

When a cache entry is valid, serve `cache_dir/"info"` verbatim instead of
recomputing it from the live MRC header on every request, and make the
fingerprint strong enough that "valid" actually means it. Two invalidation
triggers, both of which must be automatic:

1. the source file's header changes in any way → that entry rebuilds
2. any code that determines what a build produces changes → every entry rebuilds

The payoff is that the `info` a client reads always describes exactly the chunk
files sitting on disk, because both came out of the same build.

## Why this is worth doing

Today `_serve_info` recomputes `info` from the header and then *intersects* the
recomputed scale plan with the set of scales the fingerprint says were built:

```python
advertised = {"1_1_1", *fp["scales"]}
scales = [s for s in plan_scales(..., fp["params"]["min_axis_size"],
                                fp["params"]["max_levels"],
                                downsample_z=not hdr.is_image_stack)
          if s.key in advertised]
```

That intersection is a silent failure surface. If the server's scale plan ever
disagrees with the builder's — different code version, a differently-configured
`downsample_z`, a future per-path classification override — the disagreeing
levels vanish from `info` with no error, and the dataset quietly drops to fewer
resolutions than it has on disk. The 2026-08-13 image-stack work hit exactly
this: caches built with z-binned keys (`2_2_2`) stop being advertised the
moment the server starts planning `2_2_1`.

Serving the built artifact removes the disagreement by construction. It also
deletes the `min_axis_size`/`max_levels`-from-fingerprint plumbing and the
`advertised` set from `app.py`.

It has a second, forward-looking benefit. The open `MRCNG_STACK_GLOBS` idea —
classifying image stacks by path rather than by shape — is currently unsafe
because the builder and the server would each have to be configured
identically or the scale keys diverge. If the classification is baked into the
cached `info` at build time, the server does not need the globs at all for
cached files, and the two cannot disagree.

## What we are giving up

This reverses commit 46e8a88 ("Rebuild info from the header instead of serving
the cached copy"), which exists because of a real production incident: a fix to
voxel-size derivation landed, nothing invalidated the existing caches, and a
zero-`cella_z` tilt stack went on advertising `"resolution": [.., .., 0.0]` —
rejected outright by Neuroglancer — for as long as its stale `info` sat in the
cache.

**The loss is availability, not correctness.** `OUTDATED` sets
`cache_hit = False`, which falls through to the header-derived single-scale
path — `build_info` against the live header with the *new* code. So a
derivation fix still takes effect on the very next request, and scale 0 is
always read straight from the MRC, never cached. The 46e8a88 incident cannot
recur: the corrected voxel size would be served immediately.

What is actually lost is scales ≥ 1, until the rebuild lands. Zooming out stops
being smooth because Neuroglancer reads scale 0 at every zoom, and a client
holding the previous `info` gets 404s for its `2_2_2` keys until the layer is
reloaded — the same behaviour already documented for a stale cache.

Compared by kind of change, only one row favours the status quo:

| derivation change | today | proposed |
|---|---|---|
| metadata only (voxel size) | correct info, pyramid intact | correct info, pyramid gone until rebuild |
| scale plan (`downsample_z`) | correct info, levels silently missing forever | correct info, levels restored by rebuild |
| chunk content (`block_mean`, `reader`) | serves wrong cached voxels forever | invalidates and rebuilds |

That third row is the strongest argument for this work and has nothing to do
with `info`. Today a bug fix in `downsample.py` or `reader.py` invalidates
nothing — the fingerprint compares chunk *addressing* and source identity, not
the code that produced the voxels — so incorrect cached chunks are served
indefinitely. `derivation_version` is the only thing in this design that
closes that.

All of which holds only if the bump actually happens. `derivation_version` is a
hand-maintained constant, so that is a procedural guarantee, not a mechanical
one -- see the residual-risk note below.

Measured rebuild cost, from the 839 `build_duration_s` values recorded in the
dev cache: 3.3 core-hours for 0.87 TiB, i.e. ~264 GiB/core-hour. The full
3648-file / 1.38 TiB tree is ~5.4 core-hours, under 90 minutes wall-clock at
`--jobs 4`. Hours, not days — which is what makes whole-corpus invalidation
tolerable at all.

## Fingerprint schema v2

`SCHEMA_VERSION` goes 1 → 2. Old fingerprints read as `INCOMPATIBLE` and
rebuild; no migration code.

Two changes.

### `derivation_version`

A hand-maintained integer in `fingerprint.py`, bumped whenever a change alters
what a build produces:

```python
# Bump when a change alters what a build produces: the voxel size or data_type
# in info, the scale plan, the chunk bytes, or the encoding. Bumping invalidates
# every cache entry (Validity.OUTDATED) so they rebuild against the new
# behaviour. NOT bumping leaves the old artifacts served as though nothing
# changed -- see the residual-risk note in
# docs/superpowers/specs/2026-08-14-cached-info-authority-design.md.
DERIVATION_VERSION = 1
```

`validate()` gains, after the `schema_version` check:

```python
if fp.get("derivation_version") != DERIVATION_VERSION:
    return Validity.OUTDATED
```

An earlier draft computed this as a sha256 over the source of every
build-determining module, so that it could never be forgotten. Rejected as
unnecessary complexity: ~25 lines plus four tests, a `glob`-vs-`rglob` trap
(non-recursive globbing silently excludes future subpackages) and a
silent-empty-hash fallback, all to automate a one-line bump — and it invalidated
the whole corpus on comment-only edits.

**Placement is the mechanism.** A constant is only as good as the reminder to
bump it, and the reminder has to reach whoever edits `mrcheader.py`,
`precomputed.py`, `downsample.py`, `pyramid.py` or `reader.py` — not whoever
edits `fingerprint.py`, who already sees it. In practice those editors are
agents, so the reminder goes where an agent reliably reads it: a `CLAUDE.md` at
the repo root, which is loaded into every agent session automatically. The repo
has no `CLAUDE.md` today, so this design adds one. The comment on the constant
is necessary but not sufficient placement.

**Residual risk, stated plainly.** If a derivation change lands without a bump,
stale cached `info` *and* stale cached chunks are served as valid, indefinitely
and with no signal — the 46e8a88 incident. The mitigation is procedural rather
than mechanical: `mrc-pyramid status` cannot warn about it and no test can catch
it. Accepted deliberately, in exchange for not carrying the hashing machinery.

### `scales` becomes a mapping

`"scales": ["2_2_1", "4_4_1"]` becomes `"scales": {"2_2_1": [16, 16, 32],
"4_4_1": [8, 8, 32]}` — key to that level's `[sx, sy, sz]`.

This is what lets `_serve_chunk` validate a requested chunk extent without
recomputing the scale plan, which is the last place the server would otherwise
still have to derive `downsample_z`. Iterating a dict yields its keys, so
`{"1_1_1", *fp["scales"]}` and `scale_key in fp["scales"]` keep working
unchanged.

## `Validity.OUTDATED`

A new member, distinct from `STALE` (source changed) and `INCOMPATIBLE` (chunk
addressing differs), so an operator running `mrc-pyramid status` can tell "the
code moved under this cache" from "someone replaced the MRC". Every existing
caller tests `== Validity.VALID` or `!= Validity.VALID`, so adding a member is
safe, and `_status_command` already prints `result.value` — the new state
surfaces in `status` with no CLI change.

## Serving

### `_serve_info`

- Valid cache → return the bytes of `cache_dir/"info"` as-is.
- No valid cache → unchanged: single-scale `info` derived from the header via
  `plan_scales(..., max_levels=1)` and `build_info`.

`build_info` therefore stays. "One cached truth" holds for cached files only;
the uncached path is the documented single-resolution behaviour and the
"serve scale 0 straight from the MRC" premise, and it is not up for removal.
What we get is one *implementation* (`build_info`) invoked at one *time*
(build) for every cached file, rather than at two times whose outputs can drift.

**Edge case:** fingerprint valid but `info` unreadable or corrupt. The
fingerprint is written last, after `info` is fsynced, so this shouldn't happen;
if it does, fall back to the single-scale header-derived path and log an error.
This follows the existing ground rule that missing, stale, incompatible and
corrupt all read as "no cache".

**ETag.** The current cache-hit ETag is derived from `source_header_sha256`
alone. That becomes wrong here: after a derivation change and rebuild the
source is byte-identical, so the ETag would not change while the `info` body
did, and a client holding a 304 would keep the old metadata forever. The ETag
must therefore be
`f'"{fp["source_header_sha256"][:16]}-{fp["derivation_version"]}"'`.

### `_serve_chunk`

The cache branch keeps its `scale_key in fp["scales"]` membership check and
builds the `ScaleLevel` it needs for `clip_chunk_to_scale` straight from the
fingerprint, with factors parsed from the key:

```python
size = tuple(fp["scales"][scale_key])
factors = tuple(int(f) for f in scale_key.split("_"))
scale = ScaleLevel(key=scale_key, size=size, factors=factors)
```

After this, `app.py` no longer calls `plan_scales` on the cached path at all,
and no longer reads `min_axis_size`/`max_levels` from the fingerprint.

## Test consequences

`test_cached_info_is_rebuilt_from_the_header_not_served_from_disk` asserts the
behaviour this design deliberately reverses, and must be deleted. Deleting a
regression test for a real incident is only acceptable if its protection is
replaced, so the replacement is explicit:
`test_stale_derivation_invalidates_the_cache` writes a wrong `derivation_version`
into an otherwise-valid fingerprint — standing in for any code change that
would alter the derivation — and asserts the server falls back to single-scale
rather than serving the built `info`.

The poisoned-`info` test is replaced by its inverse,
`test_valid_cache_serves_the_built_info_bytes`, which asserts the served body
is byte-identical to `cache_dir/"info"`.

## Out of scope

- Hashing voxel data. "The file changed at all" is detected as
  size + mtime_ns + sha256 of the header (including the extended header), as
  today. Hashing 1.38 TiB of voxels per validation is not viable, so a
  data-only rewrite that preserves size and mtime stays undetected. Unchanged
  by this work, but stated so nobody reads "valid" as stronger than it is.
- `MRCNG_STACK_GLOBS`. Enabled by this change, not part of it.
- Rebuilding into a fresh cache root and swapping, to avoid the whole corpus
  degrading to single-resolution simultaneously. An ops procedure, not code.
- Any change to `pyramid.py`'s build logic beyond what `build_fingerprint`
  records.
