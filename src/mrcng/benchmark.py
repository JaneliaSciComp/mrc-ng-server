"""Load-test script: N concurrent clients requesting scale-0 chunks for a
list of datasets, reporting p50/p95/p99 latency. Run via `pixi run benchmark`."""
from __future__ import annotations

import argparse
import asyncio
import time

import httpx


async def _one_request(client: httpx.AsyncClient, relpath: str) -> tuple[float, bool]:
    start = time.monotonic()
    try:
        resp = await client.get(f"/data/{relpath}/1_1_1/0-64_0-64_0-64")
        ok = resp.status_code in (200, 404)  # 404 is a valid answer for a small volume
    except httpx.HTTPError:
        ok = False
    duration_ms = (time.monotonic() - start) * 1000
    return duration_ms, ok


async def run_benchmark_async(
    client: httpx.AsyncClient, dataset_relpaths: list[str],
    concurrency: int = 8, requests_per_dataset: int = 20,
) -> dict:
    tasks_queue = [relpath for relpath in dataset_relpaths for _ in range(requests_per_dataset)]
    semaphore = asyncio.Semaphore(concurrency)
    durations = []
    errors = 0

    async def _worker(relpath):
        nonlocal errors
        async with semaphore:
            duration_ms, ok = await _one_request(client, relpath)
            durations.append(duration_ms)
            if not ok:
                errors += 1

    await asyncio.gather(*(_worker(r) for r in tasks_queue))

    durations.sort()

    def _pctile(p):
        if not durations:
            return 0.0
        idx = min(len(durations) - 1, int(len(durations) * p))
        return durations[idx]

    return {
        "count": len(durations),
        "errors": errors,
        "p50_ms": round(_pctile(0.50), 2),
        "p95_ms": round(_pctile(0.95), 2),
        "p99_ms": round(_pctile(0.99), 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mrc-benchmark")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--relpath", action="append", required=True, dest="dataset_relpaths")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests-per-dataset", type=int, default=20)
    args = parser.parse_args(argv)

    async def _run():
        async with httpx.AsyncClient(base_url=args.base_url) as client:
            return await run_benchmark_async(
                client, args.dataset_relpaths, args.concurrency, args.requests_per_dataset,
            )

    result = asyncio.run(_run())
    print(result)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
