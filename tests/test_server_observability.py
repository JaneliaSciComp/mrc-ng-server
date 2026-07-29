import json
import logging

from fastapi.testclient import TestClient

from mrcng.server.config import Settings
from mrcng.server.app import create_app


def test_healthz_reports_version_and_roots(tmp_path):
    source_root = tmp_path / "source"; source_root.mkdir()
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    settings = Settings(source_root=source_root, cache_root=cache_root)
    client = TestClient(create_app(settings))

    resp = client.get("/healthz")
    body = resp.json()
    assert body["status"] == "ok"
    assert body["source_root"] == str(source_root)
    assert body["cache_root"] == str(cache_root)
    assert "version" in body


def test_chunk_request_emits_structured_log(tmp_path, caplog, make_mrc_file):
    source_root = tmp_path / "source"; source_root.mkdir()
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    make_mrc_file(name="source/tomo.mrc", shape=(8, 8, 8), mode=1)

    settings = Settings(source_root=source_root, cache_root=cache_root)
    client = TestClient(create_app(settings))

    with caplog.at_level(logging.INFO, logger="mrcng.access"):
        resp = client.get("/data/tomo.mrc/1_1_1/0-8_0-8_0-8")
    assert resp.status_code == 200

    records = [r for r in caplog.records if r.name == "mrcng.access"]
    assert len(records) == 1
    payload = json.loads(records[0].message)
    assert payload["relpath"] == "tomo.mrc"
    assert payload["scale_key"] == "1_1_1"
    assert payload["cache_hit"] is False
    assert "duration_ms" in payload


def test_fd_cache_reused_across_requests(tmp_path, make_mrc_file):
    source_root = tmp_path / "source"; source_root.mkdir()
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    make_mrc_file(name="source/tomo.mrc", shape=(8, 8, 8), mode=1)

    settings = Settings(source_root=source_root, cache_root=cache_root)
    app = create_app(settings)
    client = TestClient(app)

    client.get("/data/tomo.mrc/info")
    client.get("/data/tomo.mrc/1_1_1/0-8_0-8_0-8")

    assert len(app.state.fd_cache._entries) == 1


def test_cors_origins_comma_separated_list(tmp_path):
    # Regression: cors_origins was passed straight through as a single-element
    # list, so a comma-separated setting became one bogus literal origin
    # string instead of the list of origins it names.
    source_root = tmp_path / "source"; source_root.mkdir()
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    settings = Settings(source_root=source_root, cache_root=cache_root,
                        cors_origins="https://a.example, https://b.example")
    client = TestClient(create_app(settings))

    resp = client.get("/healthz", headers={"Origin": "https://b.example"})
    assert resp.headers.get("access-control-allow-origin") == "https://b.example"

    resp2 = client.get("/healthz", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in resp2.headers


def test_x_accel_redirect_used_when_sendfile_disabled(tmp_path, make_mrc_file):
    from mrcng.fingerprint import Params
    from mrcng.paths import dataset_id, cache_dir_for
    from mrcng.pyramid import build_one

    source_root = tmp_path / "source"; source_root.mkdir()
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    make_mrc_file(name="source/tomo.mrc", shape=(32, 32, 32), mode=1)

    params = Params(chunk_size=(8, 8, 8), downsample="mean", min_axis_size=8,
                    max_levels=3, dtype="int16", encoding="raw")
    build_one(source_root, cache_root, "tomo.mrc", params)

    settings = Settings(source_root=source_root, cache_root=cache_root, chunk_size=(8, 8, 8),
                        serve_cache_via_sendfile=False, cache_internal_location="/__cache__")
    client = TestClient(create_app(settings))

    resp = client.get("/data/tomo.mrc/2_2_2/0-8_0-8_0-8")
    assert resp.status_code == 200
    assert resp.content == b""  # nginx replaces the body; Python sends none

    cache_dir = cache_dir_for(cache_root, dataset_id("tomo.mrc"))
    rel = (cache_dir / "2_2_2" / "0-8_0-8_0-8").relative_to(cache_root).as_posix()
    assert resp.headers["x-accel-redirect"] == f"/__cache__/{rel}"
