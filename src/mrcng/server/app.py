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
from importlib.metadata import version
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from mrcng import omezarr
from mrcng.fingerprint import Params, Validity
from mrcng.mrcheader import MrcFormatError
from mrcng.paths import resolve_source, PathNotAllowed, dataset_id, cache_dir_for
from mrcng.precomputed import (
    plan_scales, build_info, parse_chunk_name, clip_chunk_to_scale, encode_chunk, ScaleLevel,
)
from mrcng.reader import read_chunk, choose_strategy, ChunkOutOfBounds, UnexpectedEOF
from mrcng.server.browse import create_browse_router
from mrcng.server.config import parse_globs
from mrcng.server.fdcache import FdCache

MRCNG_VERSION = version("mrc-ng-server")

_SCALE_KEY_RE = re.compile(r"^\d+_\d+_\d+$")
_CHUNK_RE = re.compile(r"^\d+-\d+_\d+-\d+_\d+-\d+$")
_INT_RE = re.compile(r"^\d+$")

_access_logger = logging.getLogger("mrcng.access")
_logger = logging.getLogger("mrcng.server")


def _source_etag(hdr) -> str:
    # Scale-0 content is a pure function of the source file's identity, which
    # is exactly what the fd cache already keys on -- reuse it here.
    return f'"{hdr.mtime_ns:x}-{hdr.file_size:x}"'


def _fingerprint_etag(fp: dict) -> str:
    return f'"{fp["source_header_sha256"][:16]}-{fp["derivation_version"]}"'


def _valid_fp(handle, cache_dir: Path, params: Params, fmt: str) -> dict | None:
    """The fingerprint if this entry is VALID *and* the build wrote layout `fmt`,
    else None. Every cache read in the server goes through here, so a format the
    build did not write is indistinguishable from no cache at all."""
    validity, fp = handle.validity_for(cache_dir, params)
    if validity != Validity.VALID or fp is None or fmt not in fp.get("formats", ()):
        return None
    return fp


def _header_error_response(e: MrcFormatError) -> Response:
    # A data problem an operator needs to see, per spec sec 9 -- name the
    # exception type and its message rather than returning a bare 500.
    return Response(
        content=json.dumps({"error": type(e).__name__, "detail": str(e)}),
        media_type="application/json",
        status_code=422,
    )


def _current_params(settings, hdr) -> Params:
    return Params(
        chunk_size=tuple(settings.chunk_size), downsample="mean",
        min_axis_size=32, max_levels=6, dtype=hdr.served_dtype.name, encoding="raw",
    )


def _cache_dir_for(settings, relpath: str) -> Path:
    return cache_dir_for(settings.cache_root, dataset_id(relpath))


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
    fd_cache = FdCache(max_size=settings.fd_cache_size, assume_mode0=settings.assume_mode0,
                       stack_globs=parse_globs(settings.stack_globs),
                       volume_globs=parse_globs(settings.volume_globs),
                       source_root=settings.source_root)
    semaphore = asyncio.Semaphore(settings.max_concurrent_reads)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        fd_cache.close_all()

    app = FastAPI(lifespan=lifespan)
    app.state.fd_cache = fd_cache
    if settings.cors_origins == "*":
        origins = ["*"]
    else:
        origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    app.include_router(create_browse_router(settings))

    @app.get("/healthz")
    def healthz():
        return {
            "status": "ok",
            "version": MRCNG_VERSION,
            "source_root": str(settings.source_root),
            "cache_root": str(settings.cache_root),
        }

    @app.get("/data/{full_path:path}")          # legacy alias, kept for saved links
    @app.get("/precomputed/{full_path:path}")
    async def dispatch(full_path: str):
        segments = full_path.split("/")

        if segments[-1] == "info":
            relpath = "/".join(segments[:-1])
            return await _serve_info(settings, fd_cache, relpath)

        if len(segments) >= 2 and _SCALE_KEY_RE.match(segments[-2]) and _CHUNK_RE.match(segments[-1]):
            relpath = "/".join(segments[:-2])
            return await _serve_chunk(settings, fd_cache, semaphore, relpath, segments[-2], segments[-1])

        return Response(status_code=404)

    @app.get("/omezarr/{full_path:path}")
    async def dispatch_omezarr(full_path: str):
        seg = full_path.split("/")

        # <relpath>/<level>/c/<kz>/<ky>/<kx> -- unambiguous: relpath must be a
        # file, and a file has no children.
        if (len(seg) >= 6 and seg[-4] == "c"
                and all(_INT_RE.match(s) for s in (seg[-5], seg[-3], seg[-2], seg[-1]))):
            return await _serve_zarr_chunk(
                settings, fd_cache, semaphore, "/".join(seg[:-5]),
                int(seg[-5]), int(seg[-3]), int(seg[-2]), int(seg[-1]),
            )

        if seg[-1] == "zarr.json":
            # "a/b/1/zarr.json" is level 1 of a/b if a/b is a file, else the
            # group of a file literally named a/b/1. Costs one extra resolve.
            if len(seg) >= 3 and _INT_RE.match(seg[-2]):
                relpath = "/".join(seg[:-2])
                try:
                    resolve_source(settings.source_root, relpath)
                except PathNotAllowed:
                    pass
                else:
                    return await _serve_zarr_metadata(settings, fd_cache, relpath, int(seg[-2]))
            return await _serve_zarr_metadata(settings, fd_cache, "/".join(seg[:-1]), None)

        return Response(status_code=404)   # includes .zattrs/.zgroup/.zarray v2 probes

    return app


async def _serve_info(settings, fd_cache: FdCache, relpath: str) -> Response:
    start = time.monotonic()
    try:
        path = resolve_source(settings.source_root, relpath)
    except PathNotAllowed:
        return Response(status_code=404)

    body: bytes | None = None
    etag: str | None = None
    try:
        with fd_cache.open(path) as handle:
            hdr = handle.hdr
            cache_dir = _cache_dir_for(settings, relpath)
            fp = _valid_fp(handle, cache_dir, _current_params(settings, hdr), "precomputed")
            cache_hit = fp is not None
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
                    body = (cache_dir / "precomputed" / "info").read_bytes()
                    json.loads(body)
                except (OSError, json.JSONDecodeError):
                    # Unreachable in a complete entry: fingerprint.json is
                    # written last, after info is fsynced. Degrade rather than
                    # 500, per the missing/stale/corrupt-all-read-as-no-cache
                    # ground rule.
                    _logger.error(
                        "%s: fingerprint is valid but cache info is unreadable or "
                        "corrupt; falling back to single scale", relpath,
                    )
                    cache_hit = False
                else:
                    etag = _fingerprint_etag(fp)
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

    try:
        with fd_cache.open(path) as handle:
            fd, hdr = handle.fd, handle.hdr
            if scale_key == "1_1_1":
                scale0 = ScaleLevel(key="1_1_1", size=(hdr.nx, hdr.ny, hdr.nz), factors=(1, 1, 1))
                try:
                    cx0, cx1, cy0, cy1, cz0, cz1 = clip_chunk_to_scale(
                        scale0, x0, x1, y0, y1, z0, z1, chunk_size=settings.chunk_size,
                    )
                except ValueError:
                    return Response(status_code=404)

                threshold = settings.read_row_bytes_threshold
                async with semaphore:
                    try:
                        arr = await asyncio.to_thread(
                            read_chunk, fd, hdr, cx0, cx1, cy0, cy1, cz0, cz1, threshold,
                        )
                    except ChunkOutOfBounds:
                        return Response(status_code=404)
                    except UnexpectedEOF as e:
                        # The file shrank under us between header parse and read --
                        # a data problem, not an absent tile. Log loudly (sec 9).
                        _logger.error(
                            "unexpected EOF reading %s %s/%s: %s", relpath, scale_key, chunk_str, e,
                        )
                        return Response(status_code=500)

                body = encode_chunk(arr)
                _log_access(relpath, scale_key, chunk_str, False, start)
                return Response(
                    content=body,
                    media_type="application/octet-stream",
                    headers={
                        # Unlike a cached (scale>=1) chunk, this URL bakes in no
                        # fingerprint or mtime -- the source is mutable at the
                        # same relpath, so "immutable, 1yr" would have a CDN
                        # serve year-old voxels after a replace with no way to
                        # invalidate. Revalidate on every request instead; the
                        # ETag makes that cheap (304 when unchanged).
                        "Cache-Control": "no-cache, must-revalidate",
                        "ETag": _source_etag(hdr),
                        "X-Mrcng-Read-Strategy": choose_strategy(
                            cx0, cx1, hdr.dtype.itemsize, threshold).value,
                    },
                )

            cache_dir = _cache_dir_for(settings, relpath)
            fp = _valid_fp(handle, cache_dir, _current_params(settings, hdr), "precomputed")
            if fp is None:
                return Response(status_code=404)  # no valid cache for this layout -> nothing above scale 0

            # The fingerprint is authoritative about which scales this build wrote.
            # A key that is on disk but not in the list is a leftover from an
            # earlier build of a different source, and its bytes are stale.
            if scale_key not in fp.get("scales", ()):
                return Response(status_code=404)

            # The level's size comes from the build that wrote these chunks, so
            # validation cannot drift from them -- recomputing the scale plan
            # here meant the server had to re-derive downsample_z and agree with
            # the builder, or 404 levels that exist. Validates the chunk spec
            # against the grid before touching the filesystem, per sec 9.
            try:
                scale = ScaleLevel(
                    key=scale_key,
                    size=tuple(fp["scales"][scale_key]),
                    factors=tuple(int(f) for f in scale_key.split("_")),
                )
                clip_chunk_to_scale(scale, x0, x1, y0, y1, z0, z1, chunk_size=settings.chunk_size)
            except (TypeError, ValueError):
                return Response(status_code=404)
    except MrcFormatError as e:
        return _header_error_response(e)

    chunk_path = cache_dir / "precomputed" / scale_key / chunk_str
    if not chunk_path.is_file():
        return Response(status_code=404)

    _log_access(relpath, scale_key, chunk_str, True, start)
    if settings.serve_cache_via_sendfile:
        return FileResponse(
            chunk_path,
            media_type="application/octet-stream",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    # nginx serves the body itself via X-Accel-Redirect; Python only sets
    # headers and is out of the data path entirely.
    rel = chunk_path.relative_to(settings.cache_root).as_posix()
    return Response(
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            "X-Accel-Redirect": f"{settings.cache_internal_location}/{rel}",
        },
    )


async def _serve_zarr_metadata(settings, fd_cache: FdCache, relpath: str, level: int | None) -> Response:
    """Group zarr.json (level None) or one level's array zarr.json. Cached and
    valid for omezarr: the build's file, verbatim. Otherwise a single-level
    document from the live header, mirroring the uncached precomputed info."""
    start = time.monotonic()
    try:
        path = resolve_source(settings.source_root, relpath)
    except PathNotAllowed:
        return Response(status_code=404)

    rel = "zarr.json" if level is None else f"{level}/zarr.json"
    body: bytes | None = None
    etag: str | None = None
    try:
        with fd_cache.open(path) as handle:
            hdr = handle.hdr
            cache_dir = _cache_dir_for(settings, relpath)
            fp = _valid_fp(handle, cache_dir, _current_params(settings, hdr), "omezarr")
            cache_hit = fp is not None
            if cache_hit:
                if level is not None and level > len(fp["scales"]):
                    return Response(status_code=404)
                try:
                    body = (cache_dir / "omezarr" / rel).read_bytes()
                    json.loads(body)
                except (OSError, json.JSONDecodeError):
                    _logger.error(
                        "%s: fingerprint is valid but omezarr/%s is unreadable or corrupt; "
                        "falling back to single level", relpath, rel,
                    )
                    cache_hit = False
                else:
                    etag = _fingerprint_etag(fp)
            if not cache_hit:
                if level not in (None, 0):
                    return Response(status_code=404)
                scales = plan_scales((hdr.nx, hdr.ny, hdr.nz), min_axis_size=32, max_levels=1)
                if level is None:
                    doc = omezarr.build_group_json(hdr, scales, name=relpath.rsplit("/", 1)[-1])
                else:
                    doc = omezarr.build_array_json(scales[0].size, settings.chunk_size, hdr.served_dtype)
                body = json.dumps(doc).encode()
                etag = _source_etag(hdr)
    except MrcFormatError as e:
        return _header_error_response(e)

    _log_access(relpath, rel, "", cache_hit, start)
    return Response(
        content=body,
        media_type="application/json",
        headers={"Cache-Control": "no-cache, must-revalidate", "ETag": etag},
    )


async def _serve_zarr_chunk(settings, fd_cache, semaphore, relpath, level, kz, ky, kx) -> Response:
    return Response(status_code=404)  # replaced in the chunk task
