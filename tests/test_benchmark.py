import asyncio

import httpx

from mrcng.fingerprint import Params
from mrcng.pyramid import build_one
from mrcng.server.config import Settings
from mrcng.server.app import create_app
from mrcng.benchmark import run_benchmark_async


def test_benchmark_against_in_process_app(tmp_path, make_mrc_file):
    # Regression: the old benchmark guessed a hardcoded chunk name
    # ("0-64_0-64_0-64") and counted 404 as success, so against this 16^3
    # volume it silently timed pure 404s and still reported errors == 0.
    # Looking the real chunk up via /info makes every request a genuine hit.
    source_root = tmp_path / "source"; source_root.mkdir()
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    make_mrc_file(name="source/tomo.mrc", shape=(16, 16, 16), mode=1)

    settings = Settings(source_root=source_root, cache_root=cache_root)
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)

    async def _run():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await run_benchmark_async(
                client, dataset_relpaths=["tomo.mrc"], concurrency=4, requests_per_dataset=5,
            )

    result = asyncio.run(_run())
    assert result["count"] == 5
    assert result["errors"] == 0
    scale0 = result["by_scale"]["1_1_1"]
    assert scale0["count"] == 5
    assert scale0["p50_ms"] >= 0
    assert scale0["cached"] is False


def test_benchmark_breaks_down_cached_vs_uncached_by_scale(tmp_path, make_mrc_file):
    source_root = tmp_path / "source"; source_root.mkdir()
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    make_mrc_file(name="source/tomo.mrc", shape=(32, 32, 32), mode=1)

    params = Params(chunk_size=(8, 8, 8), downsample="mean", min_axis_size=8,
                    max_levels=3, dtype="int16", encoding="raw")
    build_one(source_root, cache_root, "tomo.mrc", params)

    settings = Settings(source_root=source_root, cache_root=cache_root, chunk_size=(8, 8, 8))
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)

    async def _run():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await run_benchmark_async(
                client, dataset_relpaths=["tomo.mrc"], concurrency=4, requests_per_dataset=3,
                scale_keys=["1_1_1", "2_2_2"],
            )

    result = asyncio.run(_run())
    assert result["errors"] == 0
    assert result["by_scale"]["1_1_1"]["cached"] is False
    assert result["by_scale"]["2_2_2"]["cached"] is True
    assert result["by_scale"]["1_1_1"]["count"] == 3
    assert result["by_scale"]["2_2_2"]["count"] == 3


def test_benchmark_reports_error_for_scale_the_dataset_does_not_have(tmp_path, make_mrc_file):
    source_root = tmp_path / "source"; source_root.mkdir()
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    make_mrc_file(name="source/tomo.mrc", shape=(16, 16, 16), mode=1)  # no cache built

    settings = Settings(source_root=source_root, cache_root=cache_root)
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)

    async def _run():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await run_benchmark_async(
                client, dataset_relpaths=["tomo.mrc"], concurrency=2, requests_per_dataset=2,
                scale_keys=["2_2_2"],  # doesn't exist -- no cache was built
            )

    result = asyncio.run(_run())
    assert result["errors"] == 2
    assert result["by_scale"]["2_2_2"]["count"] == 0
