from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MRCNG_")

    source_root: Path
    cache_root: Path
    chunk_size: tuple[int, int, int] = (64, 64, 64)
    # Mode-0 signedness is ambiguous without an IMOD stamp (sec 2). Must match
    # whatever a pyramid build for the same tree used, or every mode-0 file's
    # cache reads as INCOMPATIBLE (params.dtype mismatch) -- fail-safe, but
    # silently so.
    assume_mode0: Literal["int8", "uint8"] | None = None
    # Below this many bytes per chunk row, read_chunk switches from one pread per
    # (z, y) row to one pread per z-plane: 64x fewer syscalls, at an nx/(x1-x0)
    # over-read. Which side wins is a property of the storage, so this is a knob
    # and not a constant -- benchmark it against the real backend. Responses
    # carry X-Mrcng-Read-Strategy so a sweep can attribute latency.
    read_row_bytes_threshold: int = 4096
    max_concurrent_reads: int = 32
    fd_cache_size: int = 256
    cors_origins: str = "*"  # "*" or a comma-separated list of origins
    serve_cache_via_sendfile: bool = True
    # Used only when serve_cache_via_sendfile is False: the nginx `location`
    # that internal_redirect-serves files from cache_root, so nginx does the
    # sendfile and Python is out of the path entirely.
    cache_internal_location: str = "/__mrcng_cache__"
