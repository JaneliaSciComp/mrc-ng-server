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
