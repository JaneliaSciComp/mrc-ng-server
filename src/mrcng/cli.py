"""mrc-pyramid CLI: build/status/prune."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from mrcng.fingerprint import Params, read_fingerprint, validate
from mrcng.mrcheader import parse_header
from mrcng.paths import dataset_id, cache_dir_for
from mrcng.pyramid import build_one, BuildStatus


def _parse_chunk_size(s: str) -> tuple[int, int, int]:
    parts = tuple(int(p) for p in s.split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("chunk-size must be X,Y,Z")
    return parts


def _iter_mrc_files(source_root: Path, globs: list[str]):
    seen = set()
    for pattern in globs:
        for path in sorted(source_root.rglob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path.relative_to(source_root).as_posix()


def _build_command(args) -> int:
    source_root = Path(args.source_root)
    cache_root = Path(args.cache_root)
    params = Params(
        chunk_size=tuple(args.chunk_size), downsample="mean",
        min_axis_size=args.min_axis_size, max_levels=args.max_levels,
        dtype="int16", encoding="raw",
    )
    globs = args.glob or ["*.mrc"]

    records = []
    for relpath in _iter_mrc_files(source_root, globs):
        try:
            result = build_one(source_root, cache_root, relpath, params, force=args.force)
            record = {
                "relpath": result.relpath, "dataset_id": result.dataset_id,
                "status": result.status.value, "source_bytes": result.source_bytes,
                "cache_bytes": result.cache_bytes, "levels_built": result.levels_built,
                "duration_s": result.duration_s, "error": None,
            }
        except Exception as e:
            record = {"relpath": relpath, "status": "failed", "error": str(e)}
        records.append(record)
        print(json.dumps(record))

    if args.report:
        with open(args.report, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    return 0


def _status_command(args) -> int:
    source_root = Path(args.source_root)
    cache_root = Path(args.cache_root)

    for relpath in _iter_mrc_files(source_root, ["*.mrc"]):
        ds_id = dataset_id(relpath)
        cache_dir = cache_dir_for(cache_root, ds_id)
        fp = read_fingerprint(cache_dir)
        if fp is None:
            print(f"{relpath}: missing")
            continue

        fd = os.open(str(source_root / relpath), os.O_RDONLY)
        try:
            st = os.stat(fd)
            hdr = parse_header(fd, st.st_size, st.st_mtime_ns)
            params = Params(**{**fp["params"], "chunk_size": tuple(fp["params"]["chunk_size"])})
            result = validate(fp, hdr, fd, params)
        finally:
            os.close(fd)
        print(f"{relpath}: {result.value}")

    return 0


def _prune_command(args) -> int:
    cache_root = Path(args.cache_root)
    source_root = Path(args.source_root)

    known_ids = {dataset_id(rel) for rel in _iter_mrc_files(source_root, ["*.mrc"])}

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
    build_p.add_argument("source_root")
    build_p.add_argument("--cache-root", required=True)
    build_p.add_argument("--glob", action="append")
    build_p.add_argument("--chunk-size", type=_parse_chunk_size, default=(64, 64, 64))
    build_p.add_argument("--min-axis-size", type=int, default=32)
    build_p.add_argument("--max-levels", type=int, default=6)
    build_p.add_argument("--force", action="store_true")
    build_p.add_argument("--report")
    build_p.set_defaults(func=_build_command)

    status_p = sub.add_parser("status")
    status_p.add_argument("source_root")
    status_p.add_argument("--cache-root", required=True)
    status_p.set_defaults(func=_status_command)

    prune_p = sub.add_parser("prune")
    prune_p.add_argument("--cache-root", required=True)
    prune_p.add_argument("--source-root", required=True)
    prune_p.set_defaults(func=_prune_command)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
