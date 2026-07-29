import os
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mrcng.fingerprint import Params
from mrcng.pyramid import build_one
from mrcng.server.config import Settings
from mrcng.server.app import create_app


@pytest.fixture
def cached_setup(tmp_path, make_mrc_file):
    source_root = tmp_path / "source"
    source_root.mkdir()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    def fill(zz, yy, xx):
        return (xx + 1000 * yy + 1_000_000 * zz) % 30000

    relpath = "tomo.mrc"
    make_mrc_file(name=f"source/{relpath}", shape=(32, 32, 32), mode=1, fill=fill)

    params = Params(chunk_size=(8, 8, 8), downsample="mean", min_axis_size=8,
                     max_levels=3, dtype="int16", encoding="raw")
    build_one(source_root, cache_root, relpath, params)

    settings = Settings(source_root=source_root, cache_root=cache_root, chunk_size=(8, 8, 8))
    client = TestClient(create_app(settings))
    return client, source_root, cache_root, relpath


def test_info_has_all_scales_when_cache_valid(cached_setup):
    client, _, _, relpath = cached_setup
    resp = client.get(f"/data/{relpath}/info")
    assert resp.status_code == 200
    scales = resp.json()["scales"]
    assert len(scales) > 1


def test_cached_chunk_byte_identical_to_disk(cached_setup):
    client, _, cache_root, relpath = cached_setup
    resp = client.get(f"/data/{relpath}/2_2_2/0-8_0-8_0-8")
    assert resp.status_code == 200

    from mrcng.paths import dataset_id, cache_dir_for
    cache_dir = cache_dir_for(cache_root, dataset_id(relpath))
    on_disk = (cache_dir / "2_2_2" / "0-8_0-8_0-8").read_bytes()
    assert resp.content == on_disk


def test_stale_cache_falls_back_to_single_scale(cached_setup):
    client, source_root, _, relpath = cached_setup
    time.sleep(0.01)
    os.utime(source_root / relpath, None)  # bump mtime -> STALE

    resp = client.get(f"/data/{relpath}/info")
    assert resp.status_code == 200
    assert len(resp.json()["scales"]) == 1

    chunk_resp = client.get(f"/data/{relpath}/2_2_2/0-8_0-8_0-8")
    assert chunk_resp.status_code == 404


def test_scale0_still_served_when_cache_valid(cached_setup):
    client, _, _, relpath = cached_setup
    resp = client.get(f"/data/{relpath}/1_1_1/0-8_0-8_0-8")
    assert resp.status_code == 200
    assert len(resp.content) == 8 * 8 * 8 * 2


def test_cached_info_has_etag_derived_from_fingerprint(cached_setup):
    client, _, _, relpath = cached_setup
    resp = client.get(f"/data/{relpath}/info")
    assert resp.status_code == 200
    assert resp.headers["etag"]


def test_cached_chunk_keeps_long_immutable_cache_control(cached_setup):
    # Cached (scale>=1) chunks are content-immutable for a given fingerprint,
    # unlike scale-0 -- this header is intentional and unchanged by the
    # scale-0 fix.
    client, _, _, relpath = cached_setup
    resp = client.get(f"/data/{relpath}/2_2_2/0-8_0-8_0-8")
    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "public, max-age=31536000, immutable"


def test_rebuild_over_a_replaced_source_leaves_no_readable_orphan_scale(cached_setup, tmp_path):
    """Regression: the deeper levels of a previous build survived a rebuild of a
    smaller source and stayed readable under the *new* valid fingerprint, so a
    request for a key no longer in info returned the old file's voxels, 200 OK."""
    client, source_root, cache_root, relpath = cached_setup
    from mrcng.paths import dataset_id, cache_dir_for
    from tests.conftest import make_mrc

    cache_dir = cache_dir_for(cache_root, dataset_id(relpath))
    assert (cache_dir / "4_4_4").is_dir()  # built from the 32^3 source

    time.sleep(0.01)
    os.remove(source_root / relpath)
    make_mrc(source_root / relpath, shape=(16, 16, 16), mode=1,
             fill=lambda z, y, x: np.full_like(x, 222))
    params = Params(chunk_size=(8, 8, 8), downsample="mean", min_axis_size=8,
                    max_levels=3, dtype="int16", encoding="raw")
    build_one(source_root, cache_root, relpath, params)

    assert not (cache_dir / "4_4_4").exists()  # removed by the rebuild
    keys = [s["key"] for s in client.get(f"/data/{relpath}/info").json()["scales"]]
    assert keys == ["1_1_1", "2_2_2"]
    assert client.get(f"/data/{relpath}/4_4_4/0-8_0-8_0-8").status_code == 404


def test_corrupt_fingerprint_reads_as_no_cache(cached_setup):
    # Regression: valid JSON that isn't a dict made validate()'s fp.get(...)
    # raise AttributeError -> 500, instead of reading as "no cache" like a
    # missing or stale fingerprint.
    client, _, cache_root, relpath = cached_setup
    from mrcng.paths import dataset_id, cache_dir_for

    cache_dir = cache_dir_for(cache_root, dataset_id(relpath))
    (cache_dir / "fingerprint.json").write_text("[]")

    resp = client.get(f"/data/{relpath}/info")
    assert resp.status_code == 200
    assert len(resp.json()["scales"]) == 1

    assert client.get(f"/data/{relpath}/2_2_2/0-8_0-8_0-8").status_code == 404


def test_scale_key_not_in_fingerprint_404s(cached_setup):
    """Second half of the same guard: covers caches built before the rebuild
    fix, where the orphan dir is already on disk."""
    client, _, cache_root, relpath = cached_setup
    from mrcng.paths import dataset_id, cache_dir_for

    cache_dir = cache_dir_for(cache_root, dataset_id(relpath))
    orphan = cache_dir / "64_64_64"
    orphan.mkdir()
    (orphan / "0-8_0-8_0-8").write_bytes(b"\x00" * (8 * 8 * 8 * 2))

    assert client.get(f"/data/{relpath}/64_64_64/0-8_0-8_0-8").status_code == 404
