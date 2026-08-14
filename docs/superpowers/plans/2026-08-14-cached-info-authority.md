# Cached `info` Authority Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve `cache_dir/"info"` verbatim when a cache entry is valid, and make the fingerprint strong enough that "valid" guarantees it — automatically invalidating every entry when any build-determining code changes.

**Architecture:** Fingerprint schema v2 gains a hand-maintained `DERIVATION_VERSION` constant, bumped whenever a change alters what a build produces, and turns `scales` from a list of keys into a key→size mapping. `validate()` returns a new `Validity.OUTDATED` on derivation mismatch. `_serve_info` then returns the built `info` bytes as-is, and `_serve_chunk` validates chunk extents from the fingerprint's recorded sizes — after which `app.py` never calls `plan_scales` on the cached path and cannot disagree with the builder.

**Tech Stack:** Python stdlib only (`pathlib`, `json`), FastAPI, pytest. No new dependencies, no new imports in `fingerprint.py`.

**Spec:** `docs/superpowers/specs/2026-08-14-cached-info-authority-design.md`

## Global Constraints

- No new dependencies — stdlib plus already-installed `fastapi`/`numpy`/`pytest`.
- `SCHEMA_VERSION` goes `1` → `2`. No migration code: v1 fingerprints read as `INCOMPATIBLE` and rebuild.
- `DERIVATION_VERSION` is a hand-maintained integer, deliberately not computed. Its comment, and the `CLAUDE.md` rule added in Task 4, are the only things making sure it gets bumped — treat both as load-bearing, not decoration.
- The cache-hit ETag must be `f'"{fp["source_header_sha256"][:16]}-{fp["derivation_version"]}"'`. Header-sha alone is wrong: after a derivation change + rebuild the source is byte-identical, so a client holding a 304 would keep stale `info` forever.
- `build_info` and the header-derived single-scale path stay. Uncached files keep serving single-resolution `info` — a documented feature, not up for removal.
- Missing, stale, incompatible, outdated, and corrupt must all read as "no cache". That includes a valid fingerprint whose `info` file is unreadable.
- Every existing caller tests `== Validity.VALID` / `!= Validity.VALID`, so adding an enum member is safe. Do not change those comparisons to enumerate members.
- Run the full suite with `pixi run -e default pytest -q` before each commit. It is currently 171 passing.

---

### Task 1: Fingerprint schema v2 — `DERIVATION_VERSION`, `scales` mapping, `Validity.OUTDATED`

**Files:**
- Modify: `src/mrcng/fingerprint.py`
- Modify: `src/mrcng/pyramid.py:305-312` (the `build_fingerprint` call site — pass sizes instead of keys)
- Test: `tests/test_fingerprint.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `DERIVATION_VERSION: int` — module-level constant in `fingerprint.py`, starting at `1`.
  - `Validity.OUTDATED` — new enum member, value `"outdated"`.
  - `build_fingerprint(fd, hdr, relpath, params, scales: dict[str, tuple[int, int, int]], generator_version: str, build_duration_s: float) -> dict` — the `scales` parameter changes from `list[str]` to a mapping of key → `(sx, sy, sz)`. Task 3 reads `fp["scales"][key]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fingerprint.py`:

```python
def test_validate_reports_outdated_when_derivation_changed(make_mrc_file):
    import os
    from mrcng.fingerprint import (
        Params, Validity, build_fingerprint, validate,
    )
    from mrcng.mrcheader import parse_header

    path = make_mrc_file(shape=(8, 8, 8), mode=1)
    fd = os.open(str(path), os.O_RDONLY)
    try:
        st = os.stat(fd)
        hdr = parse_header(fd, st.st_size, st.st_mtime_ns)
        params = Params(chunk_size=(8, 8, 8), downsample="mean", min_axis_size=8,
                        max_levels=3, dtype="int16", encoding="raw")
        fp = build_fingerprint(fd, hdr, "t.mrc", params, scales={}, 
                               generator_version="test", build_duration_s=0.0)
        assert validate(fp, hdr, fd, params) == Validity.VALID

        fp["derivation_version"] = 999
        assert validate(fp, hdr, fd, params) == Validity.OUTDATED
    finally:
        os.close(fd)


def test_build_fingerprint_records_scale_sizes(make_mrc_file):
    import os
    from mrcng.fingerprint import Params, build_fingerprint
    from mrcng.mrcheader import parse_header

    path = make_mrc_file(shape=(8, 8, 8), mode=1)
    fd = os.open(str(path), os.O_RDONLY)
    try:
        st = os.stat(fd)
        hdr = parse_header(fd, st.st_size, st.st_mtime_ns)
        params = Params(chunk_size=(8, 8, 8), downsample="mean", min_axis_size=8,
                        max_levels=3, dtype="int16", encoding="raw")
        fp = build_fingerprint(fd, hdr, "t.mrc", params,
                               scales={"2_2_2": (4, 4, 4)},
                               generator_version="test", build_duration_s=0.0)
    finally:
        os.close(fd)
    assert fp["schema_version"] == 2
    assert fp["scales"] == {"2_2_2": [4, 4, 4]}
    # iterating the mapping still yields keys, which app.py relies on
    assert list(fp["scales"]) == ["2_2_2"]
    assert "2_2_2" in fp["scales"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run -e default pytest tests/test_fingerprint.py -q`
Expected: FAIL — `AttributeError: OUTDATED` on `Validity`, and `test_build_fingerprint_records_scale_sizes` fails on `schema_version == 2`.

- [ ] **Step 3: Implement schema v2 in `fingerprint.py`**

Change `SCHEMA_VERSION` and add the imports and derivation machinery near the top, after `_ADDRESSING_FIELDS`:

```python
SCHEMA_VERSION = 2

_ADDRESSING_FIELDS = ("chunk_size", "encoding", "dtype")

# Bump when a change alters what a build produces: the voxel size or data_type in
# info, the scale plan, the chunk bytes, or the encoding. Bumping invalidates
# every cache entry (Validity.OUTDATED) so they rebuild against the new
# behaviour. NOT bumping leaves the old artifacts served as though nothing
# changed, with no signal anywhere -- that is how a zero-cella-z tilt stack once
# served "resolution": [.., .., 0.0] for weeks after the fix landed (46e8a88).
# The modules this tracks are mrcheader.py, precomputed.py, downsample.py,
# pyramid.py and reader.py.
DERIVATION_VERSION = 1
```

Add the enum member:

```python
class Validity(enum.Enum):
    VALID = "valid"
    STALE = "stale"
    INCOMPATIBLE = "incompatible"
    OUTDATED = "outdated"
```

Change `build_fingerprint`'s signature and body:

```python
def build_fingerprint(fd: int, hdr, relpath: str, params: Params,
                       scales: dict[str, tuple[int, int, int]],
                       generator_version: str, build_duration_s: float) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": generator_version,
        "derivation_version": DERIVATION_VERSION,
        "source_relpath": relpath,
        "source_size": hdr.file_size,
        "source_mtime_ns": hdr.mtime_ns,
        "source_header_sha256": compute_header_sha256(fd, hdr.data_offset),
        "params": asdict(params),
        # key -> [sx, sy, sz]. The sizes let the server validate a requested
        # chunk extent without recomputing the scale plan, which is the last
        # place it would otherwise have to re-derive downsample_z and risk
        # disagreeing with the build that wrote these chunks.
        "scales": {k: list(v) for k, v in scales.items()},
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_duration_s": build_duration_s,
    }
```

Add the derivation check to `validate()`, immediately after the `schema_version` check:

```python
    if fp.get("derivation_version") != DERIVATION_VERSION:
        return Validity.OUTDATED
```

- [ ] **Step 4: Update the `build_fingerprint` call site in `pyramid.py`**

In `build_one`, replace:

```python
            fp = build_fingerprint(
                fd, hdr, relpath, params,
                scales=[s.key for s in scales[1:]],
                generator_version=GENERATOR_VERSION,
                build_duration_s=time.monotonic() - start,
            )
```

with:

```python
            fp = build_fingerprint(
                fd, hdr, relpath, params,
                scales={s.key: s.size for s in scales[1:]},
                generator_version=GENERATOR_VERSION,
                build_duration_s=time.monotonic() - start,
            )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pixi run -e default pytest tests/test_fingerprint.py tests/test_pyramid.py -q`
Expected: PASS.

- [ ] **Step 6: Run the full suite and fix fallout**

Run: `pixi run -e default pytest -q`

Expected failures to fix, all from `scales` changing shape or the schema bump:
- Any test asserting `fp["scales"] == ["2_2_1", "4_4_1"]` becomes `list(fp["scales"]) == ["2_2_1", "4_4_1"]`, or assert against the mapping.
- Any test constructing a fingerprint dict by hand needs `"derivation_version": DERIVATION_VERSION` and `"schema_version": 2`.
- `tests/test_pyramid.py::test_image_stack_pyramid_never_bins_z` asserts `fp["scales"] == ["2_2_1", "4_4_1"]` — change to `list(fp["scales"]) == ["2_2_1", "4_4_1"]`.

- [ ] **Step 7: Commit**

```bash
git add src/mrcng/fingerprint.py src/mrcng/pyramid.py tests/
git commit -m "Add DERIVATION_VERSION and per-scale sizes to the fingerprint

Schema v2. DERIVATION_VERSION is bumped by hand whenever a change alters what a
build produces, which invalidates every cache entry so it rebuilds against the
new behaviour. New Validity.OUTDATED distinguishes 'the code moved' from 'the
source changed' in mrc-pyramid status.

scales becomes a key -> [sx,sy,sz] mapping so the server can validate a chunk
extent without recomputing the scale plan."
```

---

### Task 2: `_serve_info` returns the built `info` bytes

**Files:**
- Modify: `src/mrcng/server/app.py:141-192` (`_serve_info`)
- Test: `tests/test_server_cached.py`

**Interfaces:**
- Consumes: `Validity.OUTDATED` and `fp["derivation_version"]` from Task 1.
- Produces: no new symbols. Removes `_serve_info`'s dependence on `fp["params"]["min_axis_size"]` and `fp["params"]["max_levels"]`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_server_cached.py`, **delete** `test_cached_info_is_rebuilt_from_the_header_not_served_from_disk` in full (lines 45-74, including its comment block). It asserts the behaviour this task deliberately reverses. Its protection is replaced by the third test below — do not delete it without adding that one.

Append these three tests:

```python
def test_valid_cache_serves_the_built_info_bytes(cached_setup):
    # The inverse of the old rebuilt-from-header guard: the body must now be
    # exactly what the build wrote, so info can never disagree with the chunks
    # sitting next to it on disk.
    client, _, cache_root, relpath = cached_setup
    from mrcng.paths import dataset_id, cache_dir_for
    cache_dir = cache_dir_for(cache_root, dataset_id(relpath))

    resp = client.get(f"/data/{relpath}/info")
    assert resp.status_code == 200
    assert resp.content == (cache_dir / "info").read_bytes()


def test_stale_derivation_invalidates_the_cache(cached_setup):
    # Replaces test_cached_info_is_rebuilt_from_the_header_not_served_from_disk.
    # That test existed because a derivation fix used to leave every cached info
    # stale and still served (46e8a88: a zero-cella-z tilt stack advertising
    # "resolution": [.., .., 0.0] long after the header fix landed). We now
    # serve the stored bytes, so the protection has to come from invalidation
    # instead: a wrong derivation_version stands in for any such code change, and
    # must drop the entry to the single-scale fallback rather than serve it.
    client, _, cache_root, relpath = cached_setup
    from mrcng.paths import dataset_id, cache_dir_for
    cache_dir = cache_dir_for(cache_root, dataset_id(relpath))

    fp_path = cache_dir / "fingerprint.json"
    fp = json.loads(fp_path.read_text())
    assert len(fp["scales"]) > 0, "fixture must have built at least one extra level"
    fp["derivation_version"] = 999
    fp_path.write_text(json.dumps(fp))

    resp = client.get(f"/data/{relpath}/info")
    assert resp.status_code == 200
    assert [s["key"] for s in resp.json()["scales"]] == ["1_1_1"]


def test_cached_info_etag_covers_the_derivation_version(cached_setup):
    # After a derivation change and rebuild the source is byte-identical, so an
    # ETag built from source_header_sha256 alone would not change while the info
    # body did -- a client holding a 304 would keep stale metadata forever.
    client, _, cache_root, relpath = cached_setup
    from mrcng.paths import dataset_id, cache_dir_for
    cache_dir = cache_dir_for(cache_root, dataset_id(relpath))
    fp = json.loads((cache_dir / "fingerprint.json").read_text())

    resp = client.get(f"/data/{relpath}/info")
    assert str(fp["derivation_version"]) in resp.headers["etag"]
    assert fp["source_header_sha256"][:16] in resp.headers["etag"]


def test_valid_fingerprint_with_unreadable_info_falls_back(cached_setup):
    # The fingerprint is written last, after info is fsynced, so this should be
    # unreachable -- but "missing, stale, incompatible or corrupt all read as no
    # cache" is the codebase's ground rule, and a half-deleted cache entry must
    # degrade rather than 500.
    client, _, cache_root, relpath = cached_setup
    from mrcng.paths import dataset_id, cache_dir_for
    cache_dir = cache_dir_for(cache_root, dataset_id(relpath))
    (cache_dir / "info").unlink()

    resp = client.get(f"/data/{relpath}/info")
    assert resp.status_code == 200
    assert [s["key"] for s in resp.json()["scales"]] == ["1_1_1"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run -e default pytest tests/test_server_cached.py -q`
Expected: FAIL — `test_valid_cache_serves_the_built_info_bytes` fails because the body is recomputed JSON with different key order/whitespace; the ETag test fails because the ETag has no derivation id; the unreadable-info test passes vacuously for now (it currently never reads that file).

- [ ] **Step 3: Rewrite `_serve_info`'s cache branch**

Replace the whole body of `_serve_info` between `try:` and the `except MrcFormatError` with:

```python
    body: bytes | None = None
    try:
        with fd_cache.open(path) as handle:
            hdr = handle.hdr
            cache_dir = _cache_dir_for(settings, relpath)
            validity, fp = handle.validity_for(cache_dir, _current_params(settings, hdr))
            cache_hit = validity == Validity.VALID and fp is not None
            if cache_hit:
                # The built artifact is authoritative. It and the chunk files
                # next to it came out of the same build, so info can never
                # advertise a level the cache does not have -- which is what
                # recomputing the scale plan here and intersecting it with
                # fp["scales"] used to risk, silently, whenever the server's
                # plan disagreed with the builder's.
                #
                # Safe only because validate() returns OUTDATED when
                # fingerprint.DERIVATION_VERSION moves, so a derivation change
                # invalidates every entry instead of leaving stale bytes served
                # forever (46e8a88) -- which means that constant MUST be bumped
                # whenever a derivation changes.
                try:
                    body = (cache_dir / "info").read_bytes()
                except OSError:
                    # Unreachable in a complete entry: fingerprint.json is
                    # written last, after info is fsynced. Degrade rather than
                    # 500, per the missing/stale/corrupt-all-read-as-no-cache
                    # ground rule.
                    _logger.error(
                        "%s: fingerprint is valid but cache info is unreadable; "
                        "falling back to single scale", relpath,
                    )
                    cache_hit = False
                else:
                    etag = f'"{fp["source_header_sha256"][:16]}-{fp["derivation_version"]}"'
            if not cache_hit:
                scales = plan_scales((hdr.nx, hdr.ny, hdr.nz), min_axis_size=32, max_levels=1)
                body = json.dumps(build_info(hdr, scales, chunk_size=settings.chunk_size)).encode()
                etag = _source_etag(hdr)
    except MrcFormatError as e:
        return _header_error_response(e)

    _log_access(relpath, "info", "", cache_hit, start)
    return Response(
        content=body,
        media_type="application/json",
        headers={"Cache-Control": "no-cache, must-revalidate", "ETag": etag},
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pixi run -e default pytest tests/test_server_cached.py -q`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `pixi run -e default pytest -q`

`tests/test_server_cached.py::test_cached_info_has_etag_derived_from_fingerprint` may assert the exact old ETag string — update it to accept the new `<sha>-<derivation>` form, or delete it as superseded by `test_cached_info_etag_covers_the_derivation_version`.

- [ ] **Step 6: Commit**

```bash
git add src/mrcng/server/app.py tests/test_server_cached.py
git commit -m "Serve the built info bytes when the cache is valid

Reverses 46e8a88 now that validate() invalidates on derivation change. The
served info and the chunks beside it come from the same build, so info can no
longer advertise a level the cache lacks -- which recomputing the scale plan
and intersecting it with fp[scales] risked silently.

ETag now covers derivation_version: after a derivation change and rebuild the
source is byte-identical, so a source-hash-only ETag would let a 304 pin stale
info."
```

---

### Task 3: `_serve_chunk` validates from the fingerprint, not `plan_scales`

**Files:**
- Modify: `src/mrcng/server/app.py:252-284` (the cached branch of `_serve_chunk`) and the `mrcng.precomputed` import at `app.py:26-28`
- Test: `tests/test_server_cached.py`

**Interfaces:**
- Consumes: `fp["scales"]` as a key → `[sx, sy, sz]` mapping (Task 1).
- Produces: no new symbols. After this task `plan_scales` is used in `app.py` only for the uncached single-scale `info` path.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_server_cached.py`:

The discriminating test is the poisoned-size one: it is the only assertion that
distinguishes "validated from the fingerprint" from "validated from a recomputed
plan". The `cached_setup` fixture builds a 32³ volume with `chunk_size=(8,8,8)`,
`min_axis_size=8`, `max_levels=3`, so the fingerprint records
`{"2_2_2": [16,16,16], "4_4_4": [8,8,8]}`.

```python
def test_cached_chunk_extent_comes_from_the_fingerprint_not_a_recomputed_plan(cached_setup):
    # The one assertion that tells the two implementations apart. Poison the
    # recorded size for a level; a server that recomputes plan_scales gets the
    # real (16,16,16) back and serves 200, while one that trusts the fingerprint
    # clips against (4,4,4) and 404s. This is the whole point of the change: the
    # build that wrote the chunks decides how they are addressed.
    client, _, cache_root, relpath = cached_setup
    from mrcng.paths import dataset_id, cache_dir_for
    cache_dir = cache_dir_for(cache_root, dataset_id(relpath))
    fp_path = cache_dir / "fingerprint.json"
    fp = json.loads(fp_path.read_text())

    scale_key = "2_2_2"
    assert fp["scales"][scale_key] == [16, 16, 16], "fixture shape changed"
    # The chunk is legitimately served before poisoning.
    assert client.get(f"/data/{relpath}/{scale_key}/0-8_0-8_0-8").status_code == 200

    fp["scales"][scale_key] = [4, 4, 4]
    fp_path.write_text(json.dumps(fp))

    resp = client.get(f"/data/{relpath}/{scale_key}/0-8_0-8_0-8")
    assert resp.status_code == 404


def test_cached_chunk_404s_when_scale_key_absent_from_fingerprint(cached_setup):
    # Unchanged guard, re-asserted against the mapping so the list -> dict change
    # is covered: a level dir on disk that the fingerprint does not list is a
    # leftover from an earlier build of a different source, and its bytes are
    # stale. Passes before and after this task; it is here as a regression net,
    # not as the proof.
    client, _, cache_root, relpath = cached_setup
    from mrcng.paths import dataset_id, cache_dir_for
    cache_dir = cache_dir_for(cache_root, dataset_id(relpath))
    fp_path = cache_dir / "fingerprint.json"
    fp = json.loads(fp_path.read_text())
    removed = "4_4_4"
    del fp["scales"][removed]
    fp_path.write_text(json.dumps(fp))

    assert (cache_dir / removed).is_dir(), "level must still exist on disk"
    resp = client.get(f"/data/{relpath}/{removed}/0-8_0-8_0-8")
    assert resp.status_code == 404
```

- [ ] **Step 2: Run the tests to verify the discriminating one fails**

Run: `pixi run -e default pytest tests/test_server_cached.py -k "extent_comes_from_the_fingerprint or absent_from_fingerprint" -q`

Expected: `test_cached_chunk_extent_comes_from_the_fingerprint_not_a_recomputed_plan`
FAILS with `assert 200 == 404` — the current code recomputes the real size and
ignores the poisoned one. `test_cached_chunk_404s_when_scale_key_absent_from_fingerprint`
PASSES already; that is expected and fine, it guards the list→dict change rather
than the new behaviour.

- [ ] **Step 3: Replace the `plan_scales` recomputation**

In `_serve_chunk`, replace this block:

```python
            # Recompute the scale plan from this build's own params (a pure
            # calculation, no extra I/O -- min_axis_size/max_levels come from
            # the fingerprint, not the server's current settings, since they
            # don't invalidate the cache and an older build may have used
            # different values). Validates the chunk spec against the grid
            # before touching the filesystem, per sec 9.
            scales = plan_scales(
                (hdr.nx, hdr.ny, hdr.nz),
                fp["params"]["min_axis_size"], fp["params"]["max_levels"],
                downsample_z=not hdr.is_image_stack,
            )
            scale = next((s for s in scales if s.key == scale_key), None)
            if scale is None:
                return Response(status_code=404)
```

with:

```python
            # The level's size comes from the build that wrote these chunks, so
            # validation cannot drift from them -- recomputing the scale plan
            # here meant the server had to re-derive downsample_z and agree with
            # the builder, or 404 levels that exist. Validates the chunk spec
            # against the grid before touching the filesystem, per sec 9.
            scale = ScaleLevel(
                key=scale_key,
                size=tuple(fp["scales"][scale_key]),
                factors=tuple(int(f) for f in scale_key.split("_")),
            )
```

The preceding membership check (`if scale_key not in fp.get("scales", ())`) already
guarantees the key is present, so the lookup cannot raise.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pixi run -e default pytest tests/test_server_cached.py -q`
Expected: PASS.

- [ ] **Step 5: Confirm `app.py` no longer derives the cached scale plan**

Run: `grep -n "plan_scales\|is_image_stack\|min_axis_size" src/mrcng/server/app.py`

Expected: exactly one `plan_scales` call (the uncached single-scale path in `_serve_info`), one `min_axis_size` (in `_current_params`), and **no** `is_image_stack`. If `is_image_stack` still appears in `app.py`, a derivation path was missed.

- [ ] **Step 6: Run the full suite**

Run: `pixi run -e default pytest -q`

- [ ] **Step 7: Commit**

```bash
git add src/mrcng/server/app.py tests/test_server_cached.py
git commit -m "Validate cached chunk extents from the fingerprint's scale sizes

Removes the last place the server re-derived the scale plan, so it can no
longer disagree with the build that wrote the chunks. app.py no longer needs
is_image_stack or the fingerprint's min_axis_size/max_levels."
```

---

### Task 4: The `CLAUDE.md` bump rule, plus docs

`DERIVATION_VERSION` is hand-maintained, so the reminder to bump it *is* the
safety mechanism — and it has to reach whoever edits `mrcheader.py` or
`downsample.py`, not whoever edits `fingerprint.py`. This task is therefore not
cosmetic: without Step 1 the design has no enforcement at all.

**Files:**
- Create: `CLAUDE.md` (repo root — none exists today)
- Modify: `README.md` (the "Loading data into Neuroglancer" area, after the existing cache/stale bullet at step 5)
- Modify: `docs/superpowers/specs/2026-07-28-mrc-neuroglancer-design.md` (note that the info-serving rule changed, and point at the new spec)
- Test: none — documentation only.

**Interfaces:**
- Consumes: nothing.
- Produces: nothing.

- [ ] **Step 1: Create `CLAUDE.md` with the bump rule**

The repo has no `CLAUDE.md`. Create it with exactly this content — agent sessions
load it automatically, which is the only placement that reaches the person
editing a derivation module:

```markdown
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

## Tests

`pixi run -e default pytest -q`
```

- [ ] **Step 2: Add a README section**

Insert after the numbered list in "Loading data into Neuroglancer" (after the step 5 bullet about stale caches):

```markdown
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
```

- [ ] **Step 3: Note the reversal in the original design spec**

Append to `docs/superpowers/specs/2026-07-28-mrc-neuroglancer-design.md`:

```markdown
## Amendment 2026-08-14: cached `info` is authoritative again

The rule that `info` is always recomputed from the live header (commit
46e8a88) is reversed. A valid cache entry's `info` is now served verbatim,
which is safe because the fingerprint gained a `DERIVATION_VERSION` that
invalidates every entry when it is bumped. See
`docs/superpowers/specs/2026-08-14-cached-info-authority-design.md`.
```

- [ ] **Step 4: Verify the suite is still green and commit**

Run: `pixi run -e default pytest -q`
Expected: PASS.

```bash
git add CLAUDE.md README.md docs/superpowers/specs/2026-07-28-mrc-neuroglancer-design.md
git commit -m "Add the DERIVATION_VERSION bump rule and document cache invalidation

The cache now serves info and chunks verbatim, and nothing detects that the code
which produced them changed -- so the bump rule is the safety mechanism. It lives
in CLAUDE.md because the person who needs it is editing mrcheader.py, not
fingerprint.py."
```

---

## Self-review notes

**Spec coverage.** Schema v2 / `DERIVATION_VERSION` / `OUTDATED` → Task 1. `scales`
mapping → Task 1 (recorded) + Task 3 (consumed). `_serve_info` verbatim + ETag +
unreadable-`info` fallback → Task 2. `_serve_chunk` validation → Task 3. Test
replacement of the 46e8a88 guard → Task 2 Step 1. Docs → Task 4. Out-of-scope
items (voxel hashing, `MRCNG_STACK_GLOBS`, cache-root swapping) have no tasks by
design.

**Ordering constraint.** Task 2 must not land before Task 1. Serving stored
`info` without the derivation gate is exactly the 46e8a88 regression.

**Known gap, deliberate.** No task adds a `--summary` or version-specific flag to
`mrc-pyramid status`; `_status_command` already prints `result.value`, so
`outdated` appears with no CLI change.

**Unmechanised risk, accepted.** `DERIVATION_VERSION` is hand-maintained. A
derivation change that lands without a bump serves stale `info` *and* stale
chunks as valid, with no signal. Task 4 Step 1 is the whole mitigation, which is
why it is a required step rather than a documentation nicety.
