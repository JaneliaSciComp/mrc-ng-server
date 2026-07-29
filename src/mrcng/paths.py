"""Dataset identity and path-safety helpers."""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath


class PathNotAllowed(Exception):
    pass


def dataset_id(relpath: str) -> str:
    # Normalize before hashing so spellings that resolve to the same file --
    # "sub//t.mrc", "./sub/t.mrc", "sub/./t.mrc" -- collapse to the same id.
    # Without this, a client or proxy that joins URL segments naively lands on
    # a different (empty) cache entry and the whole pyramid silently
    # disappears, even though resolve_source happily serves scale 0 from the
    # same file.
    normalized = PurePosixPath(relpath).as_posix()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


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
