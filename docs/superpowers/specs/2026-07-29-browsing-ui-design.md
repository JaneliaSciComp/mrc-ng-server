# Design: Basic folder-browsing UI for Neuroglancer links

## Goal

Let a person browse the folder tree under `MRCNG_SOURCE_ROOT` in a web browser
and, for each MRC/REC file found, get a one-click link that opens it in
Neuroglancer (`https://neuroglancer-demo.appspot.com`) with a layer of type
`"auto"` pointing at this server's `precomputed://` source for that file.

Explicitly "very basic": no search, no filtering, no cache-status indicators,
no pagination, no JavaScript, no new dependencies. A todo list of things
*not* to build lives in "Out of scope" below.

## Non-goal: keep this out of the precomputed API's way

This must not become entangled with the precomputed-protocol code
(`app.py`'s dispatch/`_serve_info`/`_serve_chunk`, `fdcache.py`,
`fingerprint.py`, `precomputed.py`, `reader.py`). The browsing feature is a
separate concern (human-facing navigation) from the API (machine-facing
data serving), and should be readable, testable, and removable independently
of it.

## Architecture

New module: `src/mrcng/server/browse.py`. Contains everything specific to
this feature:

- The two routes (below)
- Directory-listing logic
- HTML rendering (plain string building, no templating library)
- Neuroglancer link construction

It exposes one function, `create_browse_router(settings) -> APIRouter`,
mounted from `create_app()` in `app.py`:

```python
from mrcng.server.browse import create_browse_router
...
app.include_router(create_browse_router(settings))
```

That import and that one line are the entire integration surface.
`browse.py` never imports from `app.py`, and the only piece of `mrcng` it
depends on beyond `settings.source_root` is a new shared path-safety helper
in `paths.py` (see below) — it does not touch fd caching, fingerprints, or
chunk serving.

## Shared dependency: `paths.py` gets `resolve_dir`

`paths.py` already has `resolve_source(root, relpath) -> Path`, which
enforces path-traversal/absolute-path/null-byte/symlink-escape protection
and requires the resolved path to be a file. Directory browsing needs the
same protections but for directories, including the root itself (empty
relpath).

Add:

```python
def resolve_dir(root: Path, relpath: str) -> Path:
    ...
```

Mirrors `resolve_source`'s checks (reject `..` components, reject absolute
paths, reject null bytes, reject symlink escape via
`resolve()` + `is_relative_to()`), with two differences:

- `relpath == ""` is valid and means "the root itself" (`resolve_source`
  rejects empty relpath outright, since scale-0/chunk requests always name a
  real file).
- Requires `candidate.is_dir()` instead of `candidate.is_file()`.

Raises the existing `PathNotAllowed` on any failure, same as
`resolve_source`, so both features report path problems identically.

This is the one intentional piece of shared code, and it's a generic safety
utility rather than API logic — duplicating traversal/symlink-escape
protection instead of sharing it would be a correctness risk, not a
separation win.

## Routes

- `GET /browse` — lists the root of `MRCNG_SOURCE_ROOT`.
- `GET /browse/{path:path}` — lists a subdirectory.

Both are handled by the same internal listing function, parameterized by the
resolved directory and its relpath (used to build "up one level" and child
links, and file relpaths for Neuroglancer sources).

## Listing contents

- Directories are always shown, sorted, each linking to
  `/browse/<child-relpath>`.
- A ".." link back to the parent is shown whenever the current path isn't
  the root.
- Files are shown **only** if their name matches `*.mrc` or `*.rec`
  (case-insensitive glob), sorted. Every other file (gain references, maps,
  logs, whatever else lives alongside tilt series) is omitted entirely —
  not listed, not linked. This keeps the page readable in directories that
  mix tomogram files with unrelated ones.
- No cache/build-status indicator, no pagination, no file size/mtime
  columns. `mrc-pyramid status` already answers "is this cached"; this page
  only answers "where are the files and how do I open one."

All filenames and path segments are HTML-escaped before being written into
the response — they come from the filesystem, and this becomes a public
HTTP response.

## Neuroglancer link construction

For a file at relpath `R` (POSIX-style, relative to `MRCNG_SOURCE_ROOT`),
requested via a browser request whose URL has scheme `S` and host (+ port)
`H`:

```
source = f"precomputed://{S}://{H}/data/{R}"
state  = {"layers": [{"type": "auto", "source": source, "name": <basename of R>}]}
link   = "https://neuroglancer-demo.appspot.com/#!" + urllib.parse.quote(json.dumps(state, separators=(",", ":")), safe="")
```

`S`/`H` are read from the incoming request (`request.url.scheme`,
`request.url.netloc`), not from a config setting — so the generated links
are always correct for whatever hostname/port/scheme was actually used to
reach the browsing page (matches the HTTPS setup already in place, and keeps
working if the port or hostname changes later without touching this code).

`urllib.parse.quote(..., safe="")` percent-encodes everything except RFC
3986 unreserved characters, which is a strict superset of what
JavaScript's `encodeURIComponent` (what Neuroglancer's own UI uses to build
these links) would encode — safe to consume either way.

## Error handling

- `resolve_dir` failure (traversal, escape, not a directory, doesn't exist)
  → 404, no body — same convention `_serve_info`/`_serve_chunk` already use
  for `PathNotAllowed`.
- Empty directory (no subdirectories, no matching files) → renders normally
  with an inline "(empty)" note in place of the missing list.

## Testing

New file: `tests/test_browse.py`, kept separate from the existing server
test files, matching the module split.

- Root listing shows subdirectories and `.mrc`/`.rec` files with links.
- Nested subdirectory listing works, and includes a working ".." link.
- `..`, absolute paths, and null bytes in the path all 404 (mirrors the
  existing `resolve_source` security tests in `test_paths.py`).
- A file that doesn't match `*.mrc`/`*.rec` is never listed and never
  linked.
- The generated Neuroglancer link, when decoded, contains
  `"type": "auto"`, a `precomputed://` source built from the *test client's*
  own request host, and the correct relpath.

## Out of scope

- Search, filtering, sorting controls.
- Cache/build-status badges per file.
- Pagination for very large directories.
- Any JavaScript or client-side rendering.
- Authentication (matches the rest of the server, which has none).
- A separate process/port for this feature (rejected during design in favor
  of a same-process, separate-module split).
