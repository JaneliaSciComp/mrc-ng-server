"""Load-test script: N concurrent clients requesting each dataset's actual
first chunk at one or more scales, reporting p50/p95/p99 latency broken down
per scale key. Scale key "1_1_1" is always an uncached direct-MRC read; any
other key is a cached fingerprinted read -- reported separately, per sec 11.
Run via `pixi run benchmark`."""
from __future__ import annotations

import argparse
import asyncio
import time

import httpx

from mrcng.precomputed import ScaleLevel, chunk_name, clip_chunk_to_scale


async def _first_chunk_path(client: httpx.AsyncClient, relpath: str, scale_key: str) -> str | None:
    """The real, correctly-clipped first chunk for (relpath, scale_key), looked
    up from /info rather than guessed -- so a 404 during the run always means
    a real problem, never an out-of-bounds guess. None if this dataset has no
    such scale (e.g. a cached scale key requested against an uncached
    dataset)."""
    resp = await client.get(f"/data/{relpath}/info")
    resp.raise_for_status()
    entry = next((s for s in resp.json()["scales"] if s["key"] == scale_key), None)
    if entry is None:
        return None
    cx, cy, cz = entry["chunk_sizes"][0]
    # factors are irrelevant to clip_chunk_to_scale, which only reads .size
    scale = ScaleLevel(key=scale_key, size=tuple(entry["size"]), factors=(1, 1, 1))
    # No chunk_size kwarg here -- that branch *validates* an already-clipped
    # extent (what the server does with a client's request); omitting it is
    # the branch that *computes* the clip for us, which is what we want for
    # the volume's first chunk.
    x0, x1, y0, y1, z0, z1 = clip_chunk_to_scale(scale, 0, cx, 0, cy, 0, cz)
    return f"/data/{relpath}/{scale_key}/{chunk_name(x0, x1, y0, y1, z0, z1)}"


async def _one_request(client: httpx.AsyncClient, path: str) -> tuple[float, bool]:
    start = time.monotonic()
    try:
        resp = await client.get(path)
        ok = resp.status_code == 200
    except httpx.HTTPError:
        ok = False
    duration_ms = (time.monotonic() - start) * 1000
    return duration_ms, ok


def _percentiles(durations: list[float]) -> dict:
    durations = sorted(durations)

    def _pctile(p):
        if not durations:
            return 0.0
        idx = min(len(durations) - 1, int(len(durations) * p))
        return durations[idx]

    return {
        "count": len(durations),
        "p50_ms": round(_pctile(0.50), 2),
        "p95_ms": round(_pctile(0.95), 2),
        "p99_ms": round(_pctile(0.99), 2),
    }


async def run_benchmark_async(
    client: httpx.AsyncClient, dataset_relpaths: list[str],
    concurrency: int = 8, requests_per_dataset: int = 20,
    scale_keys: list[str] = ("1_1_1",),
) -> dict:
    # One /info lookup per distinct (dataset, scale) up front; not part of the
    # measured load.
    paths = {
        (relpath, scale_key): await _first_chunk_path(client, relpath, scale_key)
        for relpath in dataset_relpaths for scale_key in scale_keys
    }

    semaphore = asyncio.Semaphore(concurrency)
    by_scale: dict[str, list[float]] = {k: [] for k in scale_keys}
    errors = 0

    async def _worker(relpath, scale_key):
        nonlocal errors
        path = paths[(relpath, scale_key)]
        async with semaphore:
            if path is None:
                errors += 1
                return
            duration_ms, ok = await _one_request(client, path)
            by_scale[scale_key].append(duration_ms)
            if not ok:
                errors += 1

    jobs = [
        (relpath, scale_key)
        for relpath in dataset_relpaths for scale_key in scale_keys
        for _ in range(requests_per_dataset)
    ]
    await asyncio.gather(*(_worker(r, s) for r, s in jobs))

    by_scale_result = {
        scale_key: {**_percentiles(durations), "cached": scale_key != "1_1_1"}
        for scale_key, durations in by_scale.items()
    }
    return {
        "errors": errors,
        "count": sum(v["count"] for v in by_scale_result.values()),
        "by_scale": by_scale_result,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mrc-benchmark")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--relpath", action="append", required=True, dest="dataset_relpaths")
    parser.add_argument("--scale-key", action="append", dest="scale_keys",
                        help="repeatable; defaults to scale-0 (1_1_1) only")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests-per-dataset", type=int, default=20)
    args = parser.parse_args(argv)

    async def _run():
        async with httpx.AsyncClient(base_url=args.base_url) as client:
            return await run_benchmark_async(
                client, args.dataset_relpaths, args.concurrency, args.requests_per_dataset,
                scale_keys=args.scale_keys or ["1_1_1"],
            )

    result = asyncio.run(_run())
    print(result)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
