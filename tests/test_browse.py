import json
from urllib.parse import unquote

from mrcng.server.browse import build_neuroglancer_link, NEUROGLANCER_BASE_URL


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
