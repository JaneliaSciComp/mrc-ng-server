# Folder-Browsing UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a person browse the folder tree under `MRCNG_SOURCE_ROOT` in a browser and get a one-click Neuroglancer link for every `.mrc`/`.rec` file found.

**Architecture:** A new, self-contained module `src/mrcng/server/browse.py` (routes, listing logic, HTML rendering, Neuroglancer link construction), mounted into the existing FastAPI app via `include_router`. It shares exactly one thing with the rest of the codebase: a new `resolve_dir()` path-safety helper added to `paths.py`, mirroring the existing `resolve_source()`. Never touches `app.py`'s dispatch/`_serve_info`/`_serve_chunk`, `fdcache.py`, `fingerprint.py`, `precomputed.py`, or `reader.py`.

**Tech Stack:** FastAPI (already a dependency), Python stdlib only (`html`, `json`, `urllib.parse`, `pathlib`). No new dependencies, no JavaScript, no templating library.

## Global Constraints

- No new dependencies — stdlib plus the already-installed `fastapi`/`starlette` only.
- No JavaScript or client-side rendering — plain server-rendered HTML strings.
- No authentication — matches the rest of the server.
- `browse.py` must never import from `app.py`, `fdcache.py`, `fingerprint.py`, `precomputed.py`, or `reader.py`. Its only dependency inside `mrcng` is `mrcng.paths` (and `settings.source_root`).
- Neuroglancer links point at `https://neuroglancer-demo.appspot.com/#!<percent-encoded JSON>`, with layer `"type": "auto"`.
- Only files matching `*.mrc` or `*.rec` (case-insensitive) get listed/linked; every other file is omitted from the listing entirely.
- No cache/build-status indicators, no pagination, no search/filter/sort controls — out of scope per spec.
- `<scheme>`/`<host>` for the generated Neuroglancer source URL come from the incoming request (`request.url.scheme` / `request.url.netloc`), never from a config setting.

Spec: `docs/superpowers/specs/2026-07-29-browsing-ui-design.md`

---

### Task 1: `resolve_dir` path-safety helper

**Files:**
- Modify: `src/mrcng/paths.py`
- Test: `tests/test_paths.py`

**Interfaces:**
- Consumes: nothing new (uses the existing `PathNotAllowed` exception already defined in `paths.py`).
- Produces: `resolve_dir(root: Path, relpath: str) -> Path`, used by Task 3.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_paths.py` (the file already has `from mrcng.paths import dataset_id, cache_dir_for, resolve_source, PathNotAllowed` at the top — add `resolve_dir` to that import):

```python
from mrcng.paths import dataset_id, cache_dir_for, resolve_source, resolve_dir, PathNotAllowed
```

Append these tests to the end of the file:

```python
def test_resolve_dir_root_with_empty_relpath(tmp_path):
    assert resolve_dir(tmp_path, "") == tmp_path.resolve()


def test_resolve_dir_subdirectory(tmp_path):
    (tmp_path / "sub").mkdir()
    assert resolve_dir(tmp_path, "sub") == (tmp_path / "sub").resolve()


@pytest.mark.parametrize("bad_relpath", [
    "../escape",
    "sub/../../escape",
    "/etc",
    "sub/\x00null",
])
def test_resolve_dir_rejects_unsafe_paths(tmp_path, bad_relpath):
    with pytest.raises(PathNotAllowed):
        resolve_dir(tmp_path, bad_relpath)


def test_resolve_dir_rejects_missing_directory(tmp_path):
    with pytest.raises(PathNotAllowed):
        resolve_dir(tmp_path, "does_not_exist")


def test_resolve_dir_rejects_a_file_path(tmp_path):
    f = tmp_path / "a.mrc"
    f.write_bytes(b"x")
    with pytest.raises(PathNotAllowed):
        resolve_dir(tmp_path, "a.mrc")


def test_resolve_dir_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside_dir_target"
    outside.mkdir(exist_ok=True)
    root = tmp_path / "root"
    root.mkdir()
    link = root / "escape"
    os.symlink(outside, link)
    try:
        with pytest.raises(PathNotAllowed):
            resolve_dir(root, "escape")
    finally:
        link.unlink()
        outside.rmdir()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_paths.py -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_dir'`

- [ ] **Step 3: Implement `resolve_dir`**

Append to `src/mrcng/paths.py` (after the existing `resolve_source` function):

```python
def resolve_dir(root: Path, relpath: str) -> Path:
    """Like resolve_source, but for directories: relpath == "" means the
    root itself (used for /browse with no subpath), and the resolved
    candidate must be a directory, not a file."""
    if "\x00" in relpath:
        raise PathNotAllowed(f"null byte in relpath: {relpath!r}")

    if relpath:
        parts = Path(relpath).parts
        if any(p == ".." for p in parts):
            raise PathNotAllowed(f"path traversal in relpath: {relpath!r}")
        if Path(relpath).is_absolute():
            raise PathNotAllowed(f"absolute relpath not allowed: {relpath!r}")

    root_resolved = Path(root).resolve()
    candidate = (root_resolved / relpath).resolve() if relpath else root_resolved

    if not candidate.is_relative_to(root_resolved):
        raise PathNotAllowed(f"resolved path escapes root: {relpath!r}")
    if not candidate.is_dir():
        raise PathNotAllowed(f"not a directory: {relpath!r}")

    return candidate
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest tests/test_paths.py -v`
Expected: PASS — all tests in the file green.

- [ ] **Step 5: Commit**

```bash
git add src/mrcng/paths.py tests/test_paths.py
git commit -m "Add resolve_dir path-safety helper for directory browsing"
```

---

### Task 2: Neuroglancer link builder (pure function)

**Files:**
- Create: `src/mrcng/server/browse.py`
- Test: Create `tests/test_browse.py`

**Interfaces:**
- Consumes: nothing (stdlib only: `json`, `urllib.parse.quote`).
- Produces: `NEUROGLANCER_BASE_URL: str`, `build_neuroglancer_link(scheme: str, netloc: str, relpath: str) -> str` — used by Task 3.

- [ ] **Step 1: Write the failing test**

Create `tests/test_browse.py`:

```python
import json
from urllib.parse import unquote

from mrcng.server.browse import build_neuroglancer_link, NEUROGLANCER_BASE_URL


def test_build_neuroglancer_link_contains_auto_layer_and_source():
    link = build_neuroglancer_link("https", "example.org:8443", "sub/tomo.mrc")
    assert link.startswith(NEUROGLANCER_BASE_URL)

    encoded = link[len(NEUROGLANCER_BASE_URL):]
    state = json.loads(unquote(encoded))
    layer = state["layers"][0]
    assert layer["type"] == "auto"
    assert layer["source"] == "precomputed://https://example.org:8443/data/sub/tomo.mrc"
    assert layer["name"] == "tomo.mrc"


def test_build_neuroglancer_link_name_is_the_basename_at_root():
    link = build_neuroglancer_link("http", "localhost:8000", "top.mrc")
    encoded = link[len(NEUROGLANCER_BASE_URL):]
    state = json.loads(unquote(encoded))
    assert state["layers"][0]["name"] == "top.mrc"
    assert state["layers"][0]["source"] == "precomputed://http://localhost:8000/data/top.mrc"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_browse.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mrcng.server.browse'`

- [ ] **Step 3: Write minimal implementation**

Create `src/mrcng/server/browse.py`:

```python
"""Folder-browsing UI: lists MRCNG_SOURCE_ROOT and builds one-click
Neuroglancer links for MRC/REC files.

Deliberately separate from the precomputed API (app.py) -- this module only
depends on Settings.source_root and mrcng.paths, never on
fdcache/fingerprint/precomputed/reader.
"""
from __future__ import annotations

import json
from urllib.parse import quote

NEUROGLANCER_BASE_URL = "https://neuroglancer-demo.appspot.com/#!"


def build_neuroglancer_link(scheme: str, netloc: str, relpath: str) -> str:
    """relpath is POSIX-style, relative to MRCNG_SOURCE_ROOT."""
    name = relpath.rsplit("/", 1)[-1]
    source = f"precomputed://{scheme}://{netloc}/data/{relpath}"
    state = {"layers": [{"type": "auto", "source": source, "name": name}]}
    encoded = quote(json.dumps(state, separators=(",", ":")), safe="")
    return f"{NEUROGLANCER_BASE_URL}{encoded}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/test_browse.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mrcng/server/browse.py tests/test_browse.py
git commit -m "Add Neuroglancer link builder for the browsing UI"
```

---

### Task 3: Browse router — listing, rendering, mounting

**Files:**
- Modify: `src/mrcng/server/browse.py`
- Modify: `src/mrcng/server/app.py`
- Test: `tests/test_browse.py`

**Interfaces:**
- Consumes: `resolve_dir(root, relpath) -> Path` and `PathNotAllowed` from `mrcng.paths` (Task 1); `build_neuroglancer_link(scheme, netloc, relpath) -> str` and `NEUROGLANCER_BASE_URL` from `mrcng.server.browse` (Task 2).
- Produces: `create_browse_router(settings) -> APIRouter`, mounted from `app.py`'s `create_app()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_browse.py`:

```python
import re

import pytest
from fastapi.testclient import TestClient

from mrcng.server.config import Settings
from mrcng.server.app import create_app


@pytest.fixture
def browse_client(tmp_path, make_mrc_file):
    source_root = tmp_path / "source"
    source_root.mkdir()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    (source_root / "sub").mkdir()
    make_mrc_file(name="source/top.mrc", shape=(4, 4, 4), mode=1)
    make_mrc_file(name="source/sub/nested.rec", shape=(4, 4, 4), mode=1)
    (source_root / "sub" / "notes.txt").write_text("not a tomogram")

    settings = Settings(source_root=source_root, cache_root=cache_root)
    return TestClient(create_app(settings))


def test_root_listing_shows_subdirs_and_mrc_files(browse_client):
    resp = browse_client.get("/browse")
    assert resp.status_code == 200
    assert "sub/" in resp.text
    assert "top.mrc" in resp.text
    assert "Open in Neuroglancer" in resp.text


def test_subdirectory_listing_shows_rec_file_and_parent_link(browse_client):
    resp = browse_client.get("/browse/sub")
    assert resp.status_code == 200
    assert "nested.rec" in resp.text
    assert "parent directory" in resp.text
    assert "notes.txt" not in resp.text  # not .mrc/.rec -> not listed


@pytest.mark.parametrize("bad_path", ["../escape", "sub/../../escape"])
def test_browse_rejects_traversal(browse_client, bad_path):
    resp = browse_client.get(f"/browse/{bad_path}")
    assert resp.status_code == 404


def test_browse_missing_directory_404s(browse_client):
    resp = browse_client.get("/browse/does_not_exist")
    assert resp.status_code == 404


def test_neuroglancer_link_uses_the_requests_own_host(browse_client):
    resp = browse_client.get("/browse")
    match = re.search(r'href="(https://neuroglancer-demo\.appspot\.com/#![^"]+)"', resp.text)
    assert match is not None

    import json
    from urllib.parse import unquote
    link = match.group(1)
    encoded = link.split("#!", 1)[1]
    state = json.loads(unquote(encoded))
    assert state["layers"][0]["type"] == "auto"
    assert state["layers"][0]["source"] == "precomputed://http://testserver/data/top.mrc"


def test_empty_directory_renders_without_erroring(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    settings = Settings(source_root=source_root, cache_root=cache_root)
    client = TestClient(create_app(settings))

    resp = client.get("/browse")
    assert resp.status_code == 200
    assert "no subdirectories" in resp.text
    assert "no .mrc/.rec files" in resp.text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_browse.py -v`
Expected: FAIL — `404 Not Found` for `/browse` (route doesn't exist yet), or similar failures on every new test.

- [ ] **Step 3: Implement the router in `browse.py`**

Replace the entire contents of `src/mrcng/server/browse.py` (which Task 2 created with just `build_neuroglancer_link`) with this complete file:

```python
"""Folder-browsing UI: lists MRCNG_SOURCE_ROOT and builds one-click
Neuroglancer links for MRC/REC files.

Deliberately separate from the precomputed API (app.py) -- this module only
depends on Settings.source_root and mrcng.paths, never on
fdcache/fingerprint/precomputed/reader.
"""
from __future__ import annotations

import html
import json
from pathlib import PurePosixPath
from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from mrcng.paths import resolve_dir, PathNotAllowed

NEUROGLANCER_BASE_URL = "https://neuroglancer-demo.appspot.com/#!"
_MRC_SUFFIXES = (".mrc", ".rec")


def build_neuroglancer_link(scheme: str, netloc: str, relpath: str) -> str:
    """relpath is POSIX-style, relative to MRCNG_SOURCE_ROOT."""
    name = relpath.rsplit("/", 1)[-1]
    source = f"precomputed://{scheme}://{netloc}/data/{relpath}"
    state = {"layers": [{"type": "auto", "source": source, "name": name}]}
    encoded = quote(json.dumps(state, separators=(",", ":")), safe="")
    return f"{NEUROGLANCER_BASE_URL}{encoded}"


def create_browse_router(settings) -> APIRouter:
    router = APIRouter()

    @router.get("/browse", response_class=HTMLResponse)
    async def browse_root(request: Request):
        return _render_listing(settings, request, "")

    @router.get("/browse/{path:path}", response_class=HTMLResponse)
    async def browse_path(request: Request, path: str):
        return _render_listing(settings, request, path)

    return router


def _render_listing(settings, request: Request, relpath: str) -> HTMLResponse:
    try:
        directory = resolve_dir(settings.source_root, relpath)
    except PathNotAllowed:
        return HTMLResponse(status_code=404, content="")

    entries = sorted(directory.iterdir(), key=lambda p: p.name.lower())
    subdirs = [e for e in entries if e.is_dir() and not e.name.startswith(".")]
    files = [
        e for e in entries
        if e.is_file() and not e.name.startswith(".") and e.suffix.lower() in _MRC_SUFFIXES
    ]

    base_relpath = PurePosixPath(relpath).as_posix() if relpath else ""
    parts = [f"<h1>{html.escape('/' + base_relpath)}</h1>"]

    if relpath:
        parent = str(PurePosixPath(relpath).parent)
        parent = "" if parent == "." else parent
        parent_href = f"/browse/{quote(parent)}" if parent else "/browse"
        parts.append(f'<p><a href="{parent_href}">.. (parent directory)</a></p>')

    if subdirs:
        parts.append("<ul>")
        for d in subdirs:
            child_relpath = f"{base_relpath}/{d.name}" if base_relpath else d.name
            parts.append(
                f'<li><a href="/browse/{quote(child_relpath)}">{html.escape(d.name)}/</a></li>'
            )
        parts.append("</ul>")
    else:
        parts.append("<p>(no subdirectories)</p>")

    if files:
        parts.append("<ul>")
        for f in files:
            file_relpath = f"{base_relpath}/{f.name}" if base_relpath else f.name
            link = build_neuroglancer_link(request.url.scheme, request.url.netloc, file_relpath)
            parts.append(
                f'<li>{html.escape(f.name)} '
                f'&mdash; <a href="{html.escape(link)}" target="_blank">Open in Neuroglancer</a></li>'
            )
        parts.append("</ul>")
    else:
        parts.append("<p>(no .mrc/.rec files)</p>")

    return HTMLResponse("\n".join(parts))
```

Note this supersedes Task 2's version of the file wholesale — `build_neuroglancer_link` is unchanged from Task 2, just carried over verbatim alongside the new router code, with all imports consolidated at the top.

- [ ] **Step 4: Mount the router in `app.py`**

In `src/mrcng/server/app.py`, add the import alongside the existing `mrcng.server.fdcache` import:

```python
from mrcng.server.browse import create_browse_router
from mrcng.server.fdcache import FdCache
```

Then in `create_app(settings)`, add one line after the CORS middleware is configured and before the `@app.get("/healthz")` route:

```python
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    app.include_router(create_browse_router(settings))

    @app.get("/healthz")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pixi run pytest tests/test_browse.py -v`
Expected: PASS — all tests in the file green.

- [ ] **Step 6: Run the full test suite**

Run: `pixi run pytest -q`
Expected: PASS — all tests pass, no regressions in `test_server_scale0.py`, `test_server_cached.py`, etc. (confirms `include_router` didn't shadow or collide with the existing `/data/{full_path:path}` catch-all route).

- [ ] **Step 7: Commit**

```bash
git add src/mrcng/server/browse.py src/mrcng/server/app.py tests/test_browse.py
git commit -m "Add /browse routes: folder listing with Neuroglancer links"
```

---

## Manual Verification

After Task 3, confirm in a real browser against a running server:

1. `pixi run serve`
2. Visit `https://<host>:8000/browse` — see subdirectories and any top-level `.mrc`/`.rec` files.
3. Click into a subdirectory a few levels deep (e.g. down to a `TiltSeries/raw/stack` folder) — confirm the ".. (parent directory)" link and nested navigation work.
4. Click "Open in Neuroglancer" on a real file — confirm it opens `neuroglancer-demo.appspot.com` with an `"auto"`-type layer already loaded pointing at that file.
