"""FastAPI service speaking the Neuroglancer precomputed protocol.

Scale 0 is always read directly from the MRC. Scales 1..N are served from
the cache only when a fingerprint validates -- otherwise info advertises a
single scale and chunk requests above scale 0 404, never computing anything
on the request path.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from mrcng.fingerprint import Params, Validity, read_fingerprint, validate as validate_fingerprint
from mrcng.mrcheader import MrcFormatError
from mrcng.paths import resolve_source, PathNotAllowed, dataset_id, cache_dir_for
from mrcng.precomputed import (
    plan_scales, build_info, parse_chunk_name, clip_chunk_to_scale, encode_chunk, ScaleLevel,
)
from mrcng.reader import read_chunk, ChunkOutOfBounds, UnexpectedEOF
from mrcng.server.fdcache import FdCache

MRCNG_VERSION = "0.1.0"

_SCALE_KEY_RE = re.compile(r"^\d+_\d+_\d+$")
_CHUNK_RE = re.compile(r"^\d+-\d+_\d+-\d+_\d+-\d+$")

_access_logger = logging.getLogger("mrcng.access")


def _current_params(settings, hdr) -> Params:
    return Params(
        chunk_size=tuple(settings.chunk_size), downsample="mean",
        min_axis_size=32, max_levels=6, dtype=hdr.dtype.name, encoding="raw",
    )


def _cache_dir_and_fingerprint(settings, hdr, relpath: str):
    cache_dir = cache_dir_for(settings.cache_root, dataset_id(relpath))
    return cache_dir, read_fingerprint(cache_dir)


def _log_access(relpath: str, scale_key: str, chunk: str, cache_hit: bool, start: float) -> None:
    _access_logger.info(json.dumps({
        "relpath": relpath, "scale_key": scale_key, "chunk": chunk,
        "cache_hit": cache_hit, "duration_ms": round((time.monotonic() - start) * 1000, 2),
    }))


def get_app() -> FastAPI:
    """uvicorn factory entry point: `uvicorn mrcng.server.app:get_app --factory`.
    Reads Settings from the MRCNG_* environment at call time (not import
    time), so importing this module for tests never requires source_root/
    cache_root to be set."""
    from mrcng.server.config import Settings
    return create_app(Settings())


def create_app(settings) -> FastAPI:
    fd_cache = FdCache(max_size=settings.fd_cache_size)
    semaphore = asyncio.Semaphore(settings.max_concurrent_reads)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        fd_cache.close_all()

    app = FastAPI(lifespan=lifespan)
    app.state.fd_cache = fd_cache
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.cors_origins] if settings.cors_origins != "*" else ["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    def healthz():
        return {
            "status": "ok",
            "version": MRCNG_VERSION,
            "source_root": str(settings.source_root),
            "cache_root": str(settings.cache_root),
        }

    @app.get("/data/{full_path:path}")
    async def dispatch(full_path: str):
        segments = full_path.split("/")

        if segments[-1] == "info":
            relpath = "/".join(segments[:-1])
            return await _serve_info(settings, fd_cache, relpath)

        if len(segments) >= 2 and _SCALE_KEY_RE.match(segments[-2]) and _CHUNK_RE.match(segments[-1]):
            relpath = "/".join(segments[:-2])
            return await _serve_chunk(settings, fd_cache, semaphore, relpath, segments[-2], segments[-1])

        return Response(status_code=404)

    return app


async def _serve_info(settings, fd_cache: FdCache, relpath: str) -> Response:
    start = time.monotonic()
    try:
        path = resolve_source(settings.source_root, relpath)
    except PathNotAllowed:
        return Response(status_code=404)

    with fd_cache.open(path) as (fd, hdr):
        cache_dir, fp = _cache_dir_and_fingerprint(settings, hdr, relpath)
        if fp is not None and validate_fingerprint(fp, hdr, fd, _current_params(settings, hdr)) == Validity.VALID:
            _log_access(relpath, "info", "", True, start)
            return Response(
                content=(cache_dir / "info").read_bytes(),
                media_type="application/json",
                headers={"Cache-Control": "no-cache, must-revalidate"},
            )

        try:
            scales = plan_scales((hdr.nx, hdr.ny, hdr.nz), min_axis_size=32, max_levels=1)
            info = build_info(hdr, scales, chunk_size=settings.chunk_size)
        except MrcFormatError as e:
            return Response(content=str(e), status_code=422)

    _log_access(relpath, "info", "", False, start)
    return Response(
        content=json.dumps(info),
        media_type="application/json",
        headers={"Cache-Control": "no-cache, must-revalidate"},
    )


async def _serve_chunk(settings, fd_cache: FdCache, semaphore: asyncio.Semaphore,
                        relpath: str, scale_key: str, chunk_str: str) -> Response:
    start = time.monotonic()
    try:
        path = resolve_source(settings.source_root, relpath)
    except PathNotAllowed:
        return Response(status_code=404)

    x0, x1, y0, y1, z0, z1 = parse_chunk_name(chunk_str)
    if x1 <= x0 or y1 <= y0 or z1 <= z0:
        return Response(status_code=400)

    with fd_cache.open(path) as (fd, hdr):
        if scale_key == "1_1_1":
            scale0 = ScaleLevel(key="1_1_1", size=(hdr.nx, hdr.ny, hdr.nz), factors=(1, 1, 1))
            try:
                cx0, cx1, cy0, cy1, cz0, cz1 = clip_chunk_to_scale(
                    scale0, x0, x1, y0, y1, z0, z1, chunk_size=settings.chunk_size,
                )
            except ValueError:
                return Response(status_code=404)

            async with semaphore:
                try:
                    arr = await asyncio.to_thread(read_chunk, fd, hdr, cx0, cx1, cy0, cy1, cz0, cz1)
                except (ChunkOutOfBounds, UnexpectedEOF):
                    return Response(status_code=404)

            body = encode_chunk(arr)
            _log_access(relpath, scale_key, chunk_str, False, start)
            return Response(
                content=body,
                media_type="application/octet-stream",
                headers={"Cache-Control": "public, max-age=31536000, immutable"},
            )

        cache_dir, fp = _cache_dir_and_fingerprint(settings, hdr, relpath)
        if fp is None or validate_fingerprint(fp, hdr, fd, _current_params(settings, hdr)) != Validity.VALID:
            return Response(status_code=404)  # no valid cache -> nothing above scale 0

    chunk_path = cache_dir / scale_key / chunk_str
    if not chunk_path.is_file():
        return Response(status_code=404)

    _log_access(relpath, scale_key, chunk_str, True, start)
    return FileResponse(
        chunk_path,
        media_type="application/octet-stream",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )
