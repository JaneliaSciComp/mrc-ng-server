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
