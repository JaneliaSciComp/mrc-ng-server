"""Dataset identity and path-safety helpers."""
from __future__ import annotations

import hashlib
from pathlib import Path


class PathNotAllowed(Exception):
    pass


def dataset_id(relpath: str) -> str:
    return hashlib.sha256(relpath.encode()).hexdigest()[:16]


def cache_dir_for(cache_root: Path, ds_id: str) -> Path:
    return Path(cache_root) / ds_id[:2] / ds_id


def resolve_source(root: Path, relpath: str) -> Path:
    if not relpath or "\x00" in relpath:
        raise PathNotAllowed(f"empty or null-containing relpath: {relpath!r}")

    parts = Path(relpath).parts
    if any(p == ".." for p in parts):
        raise PathNotAllowed(f"path traversal in relpath: {relpath!r}")
    if Path(relpath).is_absolute():
        raise PathNotAllowed(f"absolute relpath not allowed: {relpath!r}")

    root_resolved = Path(root).resolve()
    candidate = (root_resolved / relpath).resolve()

    if not candidate.is_relative_to(root_resolved):
        raise PathNotAllowed(f"resolved path escapes root: {relpath!r}")
    if not candidate.is_file():
        raise PathNotAllowed(f"not a file: {relpath!r}")

    return candidate
