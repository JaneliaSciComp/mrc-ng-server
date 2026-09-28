import json
import os
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mrcng.fingerprint import Params
from mrcng.paths import dataset_id, cache_dir_for
from mrcng.pyramid import build_one
from mrcng.server.config import Settings
from mrcng.server.app import create_app


def _fill(zz, yy, xx):
    return (xx + 1000 * yy + 1_000_000 * zz) % 30000


PARAMS = Params(chunk_size=(8, 8, 8), downsample="mean", min_axis_size=8,
                max_levels=3, dtype="int16", encoding="raw")


@pytest.fixture
def roots(tmp_path, make_mrc_file):
    source_root = tmp_path / "source"; source_root.mkdir()
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    # 36x20x12: edge chunks on every axis at chunk 8; level 1 is 18x10x6
    make_mrc_file(name="source/tomo.mrc", shape=(36, 20, 12), mode=1, fill=_fill)
    return source_root, cache_root


def _client(source_root, cache_root):
    return TestClient(create_app(Settings(source_root=source_root, cache_root=cache_root, chunk_size=(8, 8, 8))))


@pytest.fixture
def cached(roots):
    source_root, cache_root = roots
    build_one(source_root, cache_root, "tomo.mrc", PARAMS)
    return _client(source_root, cache_root), source_root, cache_root


@pytest.fixture
def uncached(roots):
    source_root, cache_root = roots
    return _client(source_root, cache_root), source_root, cache_root


def test_precomputed_prefix_is_an_alias_for_data(cached):
    client, _, _ = cached
    a = client.get("/data/tomo.mrc/info")
    b = client.get("/precomputed/tomo.mrc/info")
    assert a.status_code == b.status_code == 200 and a.content == b.content
    a = client.get("/data/tomo.mrc/2_2_2/0-8_0-8_0-6")
    b = client.get("/precomputed/tomo.mrc/2_2_2/0-8_0-8_0-6")
    assert a.status_code == b.status_code == 200 and a.content == b.content


def test_cached_group_json_is_served_verbatim_with_fingerprint_etag(cached):
    client, _, cache_root = cached
    cache_dir = cache_dir_for(cache_root, dataset_id("tomo.mrc"))
    resp = client.get("/omezarr/tomo.mrc/zarr.json")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.content == (cache_dir / "omezarr" / "zarr.json").read_bytes()
    assert resp.headers["cache-control"] == "no-cache, must-revalidate"
    fp = json.loads((cache_dir / "fingerprint.json").read_text())
    assert resp.headers["etag"] == f'"{fp["source_header_sha256"][:16]}-{fp["derivation_version"]}"'
    assert [d["path"] for d in resp.json()["attributes"]["ome"]["multiscales"][0]["datasets"]] == ["0", "1", "2"]


def test_cached_array_json_served_for_each_level_and_404_beyond(cached):
    client, _, cache_root = cached
    cache_dir = cache_dir_for(cache_root, dataset_id("tomo.mrc"))
    for level in (0, 1, 2):
        resp = client.get(f"/omezarr/tomo.mrc/{level}/zarr.json")
        assert resp.status_code == 200, level
        assert resp.content == (cache_dir / "omezarr" / str(level) / "zarr.json").read_bytes()
    assert client.get("/omezarr/tomo.mrc/3/zarr.json").status_code == 404


def test_uncached_group_lists_only_level_0_and_higher_metadata_404s(uncached):
    client, _, _ = uncached
    resp = client.get("/omezarr/tomo.mrc/zarr.json")
    assert resp.status_code == 200
    ms = resp.json()["attributes"]["ome"]["multiscales"][0]
    assert [d["path"] for d in ms["datasets"]] == ["0"]
    assert ms["name"] == "tomo.mrc"
    assert resp.headers["cache-control"] == "no-cache, must-revalidate" and resp.headers["etag"]

    arr0 = client.get("/omezarr/tomo.mrc/0/zarr.json")
    assert arr0.status_code == 200
    assert arr0.json()["shape"] == [12, 20, 36] and arr0.json()["data_type"] == "int16"
    assert client.get("/omezarr/tomo.mrc/1/zarr.json").status_code == 404


def test_stale_cache_falls_back_to_single_level(cached):
    client, source_root, _ = cached
    time.sleep(0.01)
    os.utime(source_root / "tomo.mrc", None)
    resp = client.get("/omezarr/tomo.mrc/zarr.json")
    assert [d["path"] for d in resp.json()["attributes"]["ome"]["multiscales"][0]["datasets"]] == ["0"]


def test_precomputed_only_cache_serves_single_level_on_omezarr(roots):
    source_root, cache_root = roots
    build_one(source_root, cache_root, "tomo.mrc", PARAMS, formats=("precomputed",))
    client = _client(source_root, cache_root)
    assert len(client.get("/precomputed/tomo.mrc/info").json()["scales"]) == 3
    resp = client.get("/omezarr/tomo.mrc/zarr.json")
    assert [d["path"] for d in resp.json()["attributes"]["ome"]["multiscales"][0]["datasets"]] == ["0"]


def test_omezarr_only_cache_serves_single_scale_on_precomputed(roots):
    source_root, cache_root = roots
    build_one(source_root, cache_root, "tomo.mrc", PARAMS, formats=("omezarr",))
    client = _client(source_root, cache_root)
    assert len(client.get("/precomputed/tomo.mrc/info").json()["scales"]) == 1
    assert client.get("/precomputed/tomo.mrc/2_2_2/0-8_0-8_0-6").status_code == 404


def test_zarr_json_disambiguation_for_a_file_named_like_a_level(tmp_path, make_mrc_file):
    source_root = tmp_path / "source"; source_root.mkdir()
    (source_root / "sub").mkdir()
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    make_mrc_file(name="source/sub/1", shape=(8, 8, 8), mode=1)   # a file literally named "1"
    client = _client(source_root, cache_root)
    resp = client.get("/omezarr/sub/1/zarr.json")
    assert resp.status_code == 200 and resp.json()["node_type"] == "group"


@pytest.mark.parametrize("url", [
    "/omezarr/zarr.json",                 # empty relpath
    "/omezarr/tomo.mrc/.zattrs",          # zarr v2 probes
    "/omezarr/tomo.mrc/.zgroup",
    "/omezarr/tomo.mrc/0/.zarray",
    "/omezarr/missing.mrc/zarr.json",
    "/omezarr/..%2Foutside.mrc/zarr.json",
    "/omezarr/tomo.mrc/notalevel/zarr.json",
])
def test_omezarr_non_routes_404(uncached, url):
    client, _, _ = uncached
    assert client.get(url).status_code == 404


def test_level0_chunk_interior_equals_precomputed_and_edge_is_padded(uncached):
    client, source_root, _ = uncached
    # interior chunk (kz=0, ky=0, kx=0): identical bytes to precomputed scale 0
    z = client.get("/omezarr/tomo.mrc/0/c/0/0/0")
    p = client.get("/precomputed/tomo.mrc/1_1_1/0-8_0-8_0-8")
    assert z.status_code == p.status_code == 200 and z.content == p.content
    assert z.headers["cache-control"] == "no-cache, must-revalidate" and z.headers["etag"]
    assert "x-mrcng-read-strategy" in z.headers

    # edge chunk kx=4 covers x[32:36), ky=2 covers y[16:20), kz=1 covers z[8:12)
    z = client.get("/omezarr/tomo.mrc/0/c/1/2/4")
    assert z.status_code == 200 and len(z.content) == 8 * 8 * 8 * 2
    arr = np.frombuffer(z.content, dtype="<i2").reshape(8, 8, 8)
    zz, yy, xx = np.meshgrid(np.arange(8, 12), np.arange(16, 20), np.arange(32, 36), indexing="ij")
    np.testing.assert_array_equal(arr[:4, :4, :4], _fill(zz, yy, xx).astype("<i2"))
    assert arr[4:].sum() == 0 and arr[:, 4:].sum() == 0 and arr[:, :, 4:].sum() == 0


@pytest.mark.parametrize("idx", ["0/0/5", "0/3/0", "2/0/0", "0/0/999999"])
def test_level0_chunk_outside_grid_404s(uncached, idx):
    client, _, _ = uncached
    assert client.get(f"/omezarr/tomo.mrc/0/c/{idx}").status_code == 404


def test_cached_chunk_is_served_verbatim_and_immutable(cached):
    client, _, cache_root = cached
    cache_dir = cache_dir_for(cache_root, dataset_id("tomo.mrc"))
    resp = client.get("/omezarr/tomo.mrc/1/c/0/1/2")
    assert resp.status_code == 200
    assert resp.content == (cache_dir / "omezarr" / "1" / "c" / "0" / "1" / "2").read_bytes()
    assert len(resp.content) == 8 * 8 * 8 * 2
    assert resp.headers["cache-control"] == "public, max-age=31536000, immutable"


@pytest.mark.parametrize("url", [
    "/omezarr/tomo.mrc/1/c/0/0/999999",   # outside grid, never a file
    "/omezarr/tomo.mrc/3/c/0/0/0",        # level beyond the pyramid
])
def test_cached_chunk_missing_404s(cached, url):
    client, _, _ = cached
    assert client.get(url).status_code == 404


def test_uncached_higher_level_chunk_404s(uncached):
    client, _, _ = uncached
    assert client.get("/omezarr/tomo.mrc/1/c/0/0/0").status_code == 404


def test_precomputed_only_cache_404s_omezarr_chunks(roots):
    source_root, cache_root = roots
    build_one(source_root, cache_root, "tomo.mrc", PARAMS, formats=("precomputed",))
    client = _client(source_root, cache_root)
    assert client.get("/omezarr/tomo.mrc/1/c/0/0/0").status_code == 404


def test_cached_chunk_x_accel_redirect_when_sendfile_disabled(roots):
    source_root, cache_root = roots
    build_one(source_root, cache_root, "tomo.mrc", PARAMS)
    settings = Settings(source_root=source_root, cache_root=cache_root, chunk_size=(8, 8, 8),
                        serve_cache_via_sendfile=False, cache_internal_location="/__cache__")
    client = TestClient(create_app(settings))
    resp = client.get("/omezarr/tomo.mrc/1/c/0/0/0")
    assert resp.status_code == 200 and resp.content == b""
    cache_dir = cache_dir_for(cache_root, dataset_id("tomo.mrc"))
    rel = (cache_dir / "omezarr" / "1" / "c" / "0" / "0" / "0").relative_to(cache_root).as_posix()
    assert resp.headers["x-accel-redirect"] == f"/__cache__/{rel}"


def test_mode12_float16_serves_float32_metadata_and_padded_chunk(tmp_path, make_mrc_file):
    source_root = tmp_path / "source"; source_root.mkdir()
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    make_mrc_file(name="source/half.mrc", shape=(6, 6, 6), mode=12, fill=lambda zz, yy, xx: xx + 0.5)
    client = _client(source_root, cache_root)
    assert client.get("/omezarr/half.mrc/0/zarr.json").json()["data_type"] == "float32"
    resp = client.get("/omezarr/half.mrc/0/c/0/0/0")
    assert resp.status_code == 200 and len(resp.content) == 8 * 8 * 8 * 4
    arr = np.frombuffer(resp.content, dtype="<f4").reshape(8, 8, 8)
    assert arr[0, 0, 3] == 3.5 and arr[7, 7, 7] == 0.0
