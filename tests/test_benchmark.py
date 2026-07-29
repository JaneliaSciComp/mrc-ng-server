import asyncio

import httpx

from mrcng.server.config import Settings
from mrcng.server.app import create_app
from mrcng.benchmark import run_benchmark_async


def test_benchmark_against_in_process_app(tmp_path, make_mrc_file):
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
    assert result["p50_ms"] >= 0
