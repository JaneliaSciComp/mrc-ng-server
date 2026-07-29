import json
import re
from urllib.parse import unquote

import pytest
from fastapi.testclient import TestClient

from mrcng.server.browse import build_neuroglancer_link, NEUROGLANCER_BASE_URL
from mrcng.server.config import Settings
from mrcng.server.app import create_app


def test_build_neuroglancer_link_contains_auto_layer_and_source():
    link = build_neuroglancer_link("https", "example.org:8443", "sub/tomo.mrc")
    assert link.startswith(NEUROGLANCER_BASE_URL)

    encoded = link[len(NEUROGLANCER_BASE_URL):]
    state = json.loads(unquote(encoded))
    layer = state["layers"][0]
    assert layer["type"] == "auto"
    assert layer["source"] == "precomputed://https://example.org:8443/data/sub/tomo.mrc"
    assert layer["name"] == "tomo.mrc"


def test_build_neuroglancer_link_name_is_the_basename_at_root():
    link = build_neuroglancer_link("http", "localhost:8000", "top.mrc")
    encoded = link[len(NEUROGLANCER_BASE_URL):]
    state = json.loads(unquote(encoded))
    assert state["layers"][0]["name"] == "top.mrc"
    assert state["layers"][0]["source"] == "precomputed://http://localhost:8000/data/top.mrc"


@pytest.fixture
def browse_client(tmp_path, make_mrc_file):
    source_root = tmp_path / "source"
    source_root.mkdir()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    (source_root / "sub").mkdir()
    make_mrc_file(name="source/top.mrc", shape=(4, 4, 4), mode=1)
    make_mrc_file(name="source/sub/nested.rec", shape=(4, 4, 4), mode=1)
    (source_root / "sub" / "notes.txt").write_text("not a tomogram")

    settings = Settings(source_root=source_root, cache_root=cache_root)
    return TestClient(create_app(settings))


def test_root_listing_shows_subdirs_and_mrc_files(browse_client):
    resp = browse_client.get("/browse")
    assert resp.status_code == 200
    assert "sub/" in resp.text
    assert "top.mrc" in resp.text
    assert "Open in Neuroglancer" in resp.text


def test_subdirectory_listing_shows_rec_file_and_parent_link(browse_client):
    resp = browse_client.get("/browse/sub")
    assert resp.status_code == 200
    assert "nested.rec" in resp.text
    assert "parent directory" in resp.text
    assert "notes.txt" not in resp.text  # not .mrc/.rec -> not listed


@pytest.mark.parametrize("bad_path", ["../escape", "sub/../../escape"])
def test_browse_rejects_traversal(browse_client, bad_path):
    resp = browse_client.get(f"/browse/{bad_path}")
    assert resp.status_code == 404


def test_browse_missing_directory_404s(browse_client):
    resp = browse_client.get("/browse/does_not_exist")
    assert resp.status_code == 404


def test_neuroglancer_link_uses_the_requests_own_host(browse_client):
    resp = browse_client.get("/browse")
    match = re.search(r'href="(https://neuroglancer-demo\.appspot\.com/#![^"]+)"', resp.text)
    assert match is not None

    link = match.group(1)
    encoded = link.split("#!", 1)[1]
    state = json.loads(unquote(encoded))
    assert state["layers"][0]["type"] == "auto"
    assert state["layers"][0]["source"] == "precomputed://http://testserver/data/top.mrc"


def test_empty_directory_renders_without_erroring(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    settings = Settings(source_root=source_root, cache_root=cache_root)
    client = TestClient(create_app(settings))

    resp = client.get("/browse")
    assert resp.status_code == 200
    assert "no subdirectories" in resp.text
    assert "no .mrc/.rec files" in resp.text
