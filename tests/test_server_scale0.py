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


def test_read_strategy_is_reported_and_tunable(tmp_path, make_mrc_file):
    # the threshold is the only knob left once span-wise stops being strictly
    # dominated, so it has to be settable and the choice has to be visible
    from mrcng.server.config import Settings
    from mrcng.server.app import create_app
    from fastapi.testclient import TestClient

    source_root = tmp_path / "src2"; source_root.mkdir()
    (tmp_path / "cache2").mkdir()
    make_mrc_file(name="src2/t.mrc", shape=(80, 60, 40), mode=1)

    def strategy_for(threshold):
        settings = Settings(source_root=source_root, cache_root=tmp_path / "cache2",
                            read_row_bytes_threshold=threshold)
        resp = TestClient(create_app(settings)).get("/data/t.mrc/1_1_1/0-64_0-60_0-40")
        assert resp.status_code == 200
        return resp.headers["x-mrcng-read-strategy"]

    assert strategy_for(0) == "row_wise"
    assert strategy_for(10**9) == "span_wise"


def test_scale0_chunk_and_info_have_revalidatable_cache_headers(client):
    # Regression: scale-0 chunks used "public, max-age=31536000, immutable",
    # but the URL bakes in no fingerprint or mtime and the source is mutable
    # at the same relpath -- a CDN would serve year-old voxels after a
    # replace with no way to invalidate.
    resp = client.get("/data/tomo.mrc/1_1_1/0-64_0-60_0-40")
    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "no-cache, must-revalidate"
    assert resp.headers["etag"]

    info_resp = client.get("/data/tomo.mrc/info")
    assert info_resp.status_code == 200
    assert info_resp.headers["etag"]


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


def test_header_validation_error_returns_422_with_json_body(tmp_path, make_mrc_file):
    # Regression: parse_header raises inside fd_cache.open(), outside any
    # handler's try/except, so this used to bubble up as a bare 500.
    source_root = tmp_path / "s422"; source_root.mkdir()
    (tmp_path / "c422").mkdir()
    make_mrc_file(name="s422/permuted.mrc", shape=(8, 8, 8), mode=1, mapc=2, mapr=1, maps=3)
    settings = Settings(source_root=source_root, cache_root=tmp_path / "c422")
    client = TestClient(create_app(settings))

    for url in ("/data/permuted.mrc/info", "/data/permuted.mrc/1_1_1/0-8_0-8_0-8"):
        resp = client.get(url)
        assert resp.status_code == 422
        body = resp.json()
        assert body["error"] == "NonStandardAxisOrderError"
        assert "mapc" in body["detail"]


def test_truncated_file_returns_422(tmp_path, make_mrc_file):
    source_root = tmp_path / "strunc"; source_root.mkdir()
    (tmp_path / "ctrunc").mkdir()
    make_mrc_file(name="strunc/t.mrc", shape=(16, 16, 16), mode=1, truncate_bytes=100)
    settings = Settings(source_root=source_root, cache_root=tmp_path / "ctrunc")
    client = TestClient(create_app(settings))

    resp = client.get("/data/t.mrc/info")
    assert resp.status_code == 422
    assert resp.json()["error"] == "TruncatedFileError"


def test_server_assume_mode0_setting_reaches_the_header_parser(tmp_path, make_mrc_file):
    import numpy as np
    source_root = tmp_path / "smode0"; source_root.mkdir()
    (tmp_path / "cmode0").mkdir()
    make_mrc_file(name="smode0/t.mrc", shape=(8, 8, 8), mode=0)  # no IMOD stamp -> ambiguous

    settings = Settings(source_root=source_root, cache_root=tmp_path / "cmode0",
                        assume_mode0="uint8")
    resp = TestClient(create_app(settings)).get("/data/t.mrc/1_1_1/0-8_0-8_0-8")
    assert resp.status_code == 200
    assert np.frombuffer(resp.content, dtype=np.uint8).max() <= 255  # didn't error as int8/mixed


def test_server_logs_ambiguous_mode0_signedness(tmp_path, make_mrc_file, caplog):
    import logging
    source_root = tmp_path / "swarn"; source_root.mkdir()
    (tmp_path / "cwarn").mkdir()
    make_mrc_file(name="swarn/t.mrc", shape=(8, 8, 8), mode=0)

    settings = Settings(source_root=source_root, cache_root=tmp_path / "cwarn")
    client = TestClient(create_app(settings))
    with caplog.at_level(logging.WARNING, logger="mrcng.server"):
        assert client.get("/data/t.mrc/info").status_code == 200
    assert any("ambiguous" in r.message for r in caplog.records if r.name == "mrcng.server")


def test_unexpected_eof_mid_read_returns_500_and_logs(tmp_path, make_mrc_file, monkeypatch, caplog):
    # Regression: UnexpectedEOF was lumped in with ChunkOutOfBounds -> 404, so a
    # file that shrank under the server looked like an ordinary empty tile.
    import logging
    from mrcng.reader import UnexpectedEOF
    from mrcng.server import app as app_module

    source_root = tmp_path / "seof"; source_root.mkdir()
    (tmp_path / "ceof").mkdir()
    make_mrc_file(name="seof/t.mrc", shape=(8, 8, 8), mode=1)
    settings = Settings(source_root=source_root, cache_root=tmp_path / "ceof")
    client = TestClient(create_app(settings))

    def boom(*args, **kwargs):
        raise UnexpectedEOF("simulated truncation mid-read")

    monkeypatch.setattr(app_module, "read_chunk", boom)

    with caplog.at_level(logging.ERROR, logger="mrcng.server"):
        resp = client.get("/data/t.mrc/1_1_1/0-8_0-8_0-8")
    assert resp.status_code == 500
    assert any(r.name == "mrcng.server" and r.levelno == logging.ERROR for r in caplog.records)
