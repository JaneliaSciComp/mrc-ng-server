"""mrc-pyramid CLI: build/status/prune."""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import shutil
import sys
from pathlib import Path

from mrcng.fingerprint import Params, read_fingerprint, validate
from mrcng.mrcheader import parse_header
from mrcng.paths import dataset_id, cache_dir_for
from mrcng.pyramid import build_one, BuildStatus, DEFAULT_MAX_BLOCK_BYTES

_logger = logging.getLogger("mrcng.pyramid")


def _parse_chunk_size(s: str) -> tuple[int, int, int]:
    parts = tuple(int(p) for p in s.split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("chunk-size must be X,Y,Z")
    return parts


def _parse_size(s: str) -> int:
    """256M / 2G / 1048576 -> bytes."""
    units = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    s = s.strip().upper().rstrip("B")
    if s and s[-1] in units:
        return int(float(s[:-1]) * units[s[-1]])
    return int(s)


def _add_source_root_arg(parser: argparse.ArgumentParser, *, flag: bool = False) -> None:
    name = "--source-root" if flag else "source_root"
    parser.add_argument(name, nargs=None if flag else "?",
                        default=os.environ.get("MRCNG_SOURCE_ROOT"),
                        help="root of MRC files (default: $MRCNG_SOURCE_ROOT)")


def _add_cache_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-root", default=os.environ.get("MRCNG_CACHE_ROOT"),
                        help="cache tree root (default: $MRCNG_CACHE_ROOT)")


def _iter_mrc_files(source_root: Path, globs: list[str], walk_root: Path | None = None):
    walk_root = walk_root or source_root
    seen = set()
    for pattern in globs:
        for path in sorted(walk_root.rglob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path.relative_to(source_root).as_posix()


def _relpaths_from_file(list_path: str, source_root: Path) -> list[str]:
    """Read newline-separated relpaths (relative to source_root) to build.

    Blank lines and ``#`` comments are skipped. An entry that isn't a file under
    source_root is skipped with a warning -- the same is-file / in-tree contract
    the glob walk enforces, so a bad or hostile list can't build outside the
    tree. Lets a caller (e.g. a catalog scanner) build exactly its known set of
    volumes instead of globbing every stray ``.mrc``.
    """
    root = source_root.resolve()
    out: list[str] = []
    with open(list_path) as f:
        for line in f:
            rel = line.strip()
            if not rel or rel.startswith("#"):
                continue
            abs_path = (root / rel).resolve()
            if not abs_path.is_relative_to(root):
                _logger.warning("skipping %r: resolves outside source root", rel)
                continue
            if not abs_path.is_file():
                _logger.warning("skipping %r: not a file under source root", rel)
                continue
            out.append(abs_path.relative_to(root).as_posix())
    return out


def _select_relpaths(
    source_root: Path, globs: list[str] | None, from_file: str | None,
    walk_root: Path | None = None,
) -> list[str]:
    """Relpaths to build, from --glob and/or --from-file (deduped, order-preserving).

    With neither, default to ``*.mrc`` (glob the whole tree) -- the general-purpose
    behaviour. With either or both, build exactly their union and add no implicit
    ``*.mrc``, so a --from-file run builds only the listed volumes. ``walk_root``
    (from --under) scopes the glob walk to a subdirectory while relpaths stay
    relative to ``source_root``, so the cache is addressable the same way a
    full-tree build would produce -- --from-file paths are unaffected since
    they're already explicit.
    """
    seen: set[str] = set()
    out: list[str] = []

    def add(rel: str) -> None:
        if rel not in seen:
            seen.add(rel)
            out.append(rel)

    if globs:
        for rel in _iter_mrc_files(source_root, globs, walk_root):
            add(rel)
    if from_file:
        for rel in _relpaths_from_file(from_file, source_root):
            add(rel)
    if not globs and not from_file:
        for rel in _iter_mrc_files(source_root, ["*.mrc"], walk_root):
            add(rel)
    return out


def _build_one_record(task: tuple) -> dict:
    """Top-level (picklable) so multiprocessing.Pool can call it directly."""
    source_root, cache_root, relpath, params, force, max_block_bytes, assume_mode0 = task
    try:
        result = build_one(source_root, cache_root, relpath, params, force=force,
                           max_block_bytes=max_block_bytes, assume_mode0=assume_mode0)
        return {
            "relpath": result.relpath, "dataset_id": result.dataset_id,
            "status": result.status.value, "source_bytes": result.source_bytes,
            "cache_bytes": result.cache_bytes, "levels_built": result.levels_built,
            "duration_s": result.duration_s,
            "voxel_size_is_default": result.voxel_size_is_default, "error": None,
        }
    except Exception as e:
        return {"relpath": relpath, "status": "failed", "error": str(e)}


def _build_command(args) -> int:
    logging.basicConfig(level=args.log_level)
    source_root = Path(args.source_root)
    cache_root = Path(args.cache_root)
    walk_root = source_root
    if args.under:
        walk_root = (source_root / args.under).resolve()
        if not walk_root.is_relative_to(source_root.resolve()) or not walk_root.is_dir():
            raise SystemExit(f"--under {args.under!r} is not a directory under source_root")
    params = Params(
        chunk_size=tuple(args.chunk_size), downsample="mean",
        min_axis_size=args.min_axis_size, max_levels=args.max_levels,
        dtype="unset",  # build_one derives the real per-file dtype from each header
        encoding="raw",
    )
    relpaths = _select_relpaths(source_root, args.glob, args.from_file, walk_root)
    tasks = [
        (source_root, cache_root, relpath, params, args.force, args.max_block_bytes, args.assume_mode0)
        for relpath in relpaths
    ]

    records = []
    # Within a file the work is I/O-bound streaming; across files it
    # parallelises cleanly (sec 8). Oversubscribing NFS with more workers than
    # files or than requested helps nothing.
    if args.jobs <= 1 or len(tasks) <= 1:
        for record in map(_build_one_record, tasks):
            records.append(record)
            print(json.dumps(record))
    else:
        with multiprocessing.Pool(min(args.jobs, len(tasks))) as pool:
            for record in pool.imap(_build_one_record, tasks):
                records.append(record)
                print(json.dumps(record))

    if args.report:
        with open(args.report, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    return 1 if any(r["status"] == "failed" for r in records) else 0


def _status_command(args) -> int:
    source_root = Path(args.source_root)
    cache_root = Path(args.cache_root)
    globs = args.glob or ["*.mrc"]

    for relpath in _iter_mrc_files(source_root, globs):
        ds_id = dataset_id(relpath)
        cache_dir = cache_dir_for(cache_root, ds_id)
        fp = read_fingerprint(cache_dir)
        if fp is None:
            print(f"{relpath}: missing")
            continue

        fd = os.open(str(source_root / relpath), os.O_RDONLY)
        try:
            st = os.stat(fd)
            hdr = parse_header(fd, st.st_size, st.st_mtime_ns, assume_mode0=args.assume_mode0)
            if hdr.mode0_signedness_is_ambiguous:
                _logger.warning(
                    "%s: mode-0 signedness is ambiguous (no IMOD stamp), defaulting to "
                    "int8; pass --assume-mode0 to override", relpath,
                )
            # Validate against the *configured* params, not the fingerprint's
            # own -- comparing fp["params"] to itself can never report
            # incompatible.
            params = Params(
                chunk_size=tuple(args.chunk_size), downsample="mean",
                min_axis_size=args.min_axis_size, max_levels=args.max_levels,
                dtype=hdr.served_dtype.name, encoding="raw",
            )
            result = validate(fp, hdr, fd, params)
        finally:
            os.close(fd)
        print(f"{relpath}: {result.value}")

    return 0


def _prune_command(args) -> int:
    cache_root = Path(args.cache_root)
    source_root = Path(args.source_root)
    globs = args.glob or ["*.mrc"]

    known_ids = {dataset_id(rel) for rel in _iter_mrc_files(source_root, globs)}

    if cache_root.exists():
        for prefix_dir in cache_root.iterdir():
            if not prefix_dir.is_dir():
                continue
            for entry_dir in prefix_dir.iterdir():
                if entry_dir.is_dir() and entry_dir.name not in known_ids:
                    shutil.rmtree(entry_dir)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mrc-pyramid")
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build")
    _add_source_root_arg(build_p, flag=True)
    _add_cache_root_arg(build_p)
    build_p.add_argument("--glob", action="append")
    build_p.add_argument(
        "--from-file",
        help="file of newline-separated relpaths (relative to source_root) to "
             "build; combined with any --glob. With neither, the whole tree is "
             "globbed for *.mrc",
    )
    build_p.add_argument(
        "--under",
        help="scope the --glob walk to this subdirectory of source_root (relpaths "
             "are still computed relative to source_root, so the cache matches what "
             "a full-tree build would produce). Dev convenience for building a "
             "small subset without hand-listing files via --from-file.",
    )
    build_p.add_argument("--chunk-size", type=_parse_chunk_size, default=(64, 64, 64))
    build_p.add_argument("--min-axis-size", type=int, default=32)
    build_p.add_argument("--max-levels", type=int, default=6)
    build_p.add_argument("--max-block-bytes", type=_parse_size, default=DEFAULT_MAX_BLOCK_BYTES,
                         help="cap on one streamed source block (e.g. 256M); peak RSS "
                              "per worker is roughly 3x this")
    build_p.add_argument("--assume-mode0", choices=("int8", "uint8"),
                         help="mode-0 signedness when no IMOD stamp is present")
    build_p.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1),
                         help="multiprocessing.Pool over files; the bottleneck is usually "
                              "storage, so more than a few oversubscribes NFS")
    build_p.add_argument("--log-level", default="INFO")
    build_p.add_argument("--force", action="store_true")
    build_p.add_argument("--report")
    build_p.set_defaults(func=_build_command)

    status_p = sub.add_parser("status")
    _add_source_root_arg(status_p)
    _add_cache_root_arg(status_p)
    status_p.add_argument("--glob", action="append")
    status_p.add_argument("--chunk-size", type=_parse_chunk_size, default=(64, 64, 64))
    status_p.add_argument("--min-axis-size", type=int, default=32)
    status_p.add_argument("--max-levels", type=int, default=6)
    status_p.add_argument("--assume-mode0", choices=("int8", "uint8"))
    status_p.set_defaults(func=_status_command)

    prune_p = sub.add_parser("prune")
    _add_cache_root_arg(prune_p)
    _add_source_root_arg(prune_p, flag=True)
    prune_p.add_argument("--glob", action="append")
    prune_p.set_defaults(func=_prune_command)

    args = parser.parse_args(argv)
    if args.source_root is None:
        parser.error("source_root is required (arg or $MRCNG_SOURCE_ROOT)")
    if args.cache_root is None:
        parser.error("--cache-root is required (arg or $MRCNG_CACHE_ROOT)")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
