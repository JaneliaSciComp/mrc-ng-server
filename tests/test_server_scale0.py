import pytest
from fastapi.testclient import TestClient

from mrcng.server.config import Settings
from mrcng.server.app import create_app


@pytest.fixture
def client(tmp_path, make_mrc_file):
    source_root = tmp_path / "source"
    source_root.mkdir()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    def fill(zz, yy, xx):
        return (xx + 1000 * yy + 1_000_000 * zz) % 30000

    make_mrc_file(name="source/tomo.mrc", shape=(80, 60, 40), mode=1,
                  voxel_size_angstrom=(6.8, 6.8, 6.8), fill=fill)

    settings = Settings(source_root=source_root, cache_root=cache_root)
    app = create_app(settings)
    return TestClient(app)


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200


def test_info_uncached_has_single_scale(client):
    resp = client.get("/data/tomo.mrc/info")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["scales"]) == 1
    assert body["scales"][0]["key"] == "1_1_1"
    assert body["scales"][0]["size"] == [80, 60, 40]


def test_info_missing_file_404s(client):
    resp = client.get("/data/does_not_exist.mrc/info")
    assert resp.status_code == 404


def test_info_path_traversal_404s(client):
    resp = client.get("/data/..%2Foutside.mrc/info")
    assert resp.status_code == 404


def test_scale0_chunk_matches_source(client, tmp_path):
    # volume is (nx, ny, nz) = (80, 60, 40); default chunk_size is (64,64,64),
    # so the first grid cell is clipped to 0-64_0-60_0-40 (y,z clipped to the
    # volume; x is a full non-edge chunk since nx=80 > 64).
    resp = client.get("/data/tomo.mrc/1_1_1/0-64_0-60_0-40")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/octet-stream"

    import numpy as np
    import mrcfile
    with mrcfile.open(tmp_path / "source" / "tomo.mrc", permissive=True) as mf:
        expected = mf.data[0:40, 0:60, 0:64]  # (z, y, x)
    got = np.frombuffer(resp.content, dtype="<i2").reshape(40, 60, 64)
    np.testing.assert_array_equal(got, expected)


def test_scale0_chunk_out_of_grid_404s(client):
    resp = client.get("/data/tomo.mrc/1_1_1/1000-1064_0-60_0-40")
    assert resp.status_code == 404


def test_uncached_higher_scale_404s(client):
    resp = client.get("/data/tomo.mrc/2_2_1/0-32_0-32_0-20")
    assert resp.status_code == 404


def test_unmatched_route_shape_404s(client):
    resp = client.get("/data/tomo.mrc/not-a-scale/not-a-chunk")
    assert resp.status_code == 404


def test_malformed_chunk_spec_400s(client):
    # matches the route shape (scale key + chunk-name regex) but the range is inverted
    resp = client.get("/data/tomo.mrc/1_1_1/80-0_0-60_0-40")
    assert resp.status_code == 400
