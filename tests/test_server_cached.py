import json
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


def test_equivalent_path_spellings_share_the_same_cache(tmp_path, make_mrc_file):
    # Regression: dataset_id used to hash the raw request string, so
    # "sub//t.mrc" (a double slash a naive URL join can produce) got a
    # different id than "sub/t.mrc" and its whole pyramid silently vanished,
    # even though resolve_source serves scale 0 from the same file either way.
    source_root = tmp_path / "source"; source_root.mkdir()
    (source_root / "sub").mkdir()
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    make_mrc_file(name="source/sub/t.mrc", shape=(32, 32, 32), mode=1)

    params = Params(chunk_size=(8, 8, 8), downsample="mean", min_axis_size=8,
                    max_levels=3, dtype="int16", encoding="raw")
    build_one(source_root, cache_root, "sub/t.mrc", params)

    client = TestClient(create_app(Settings(source_root=source_root, cache_root=cache_root,
                                            chunk_size=(8, 8, 8))))
    canonical = client.get("/data/sub/t.mrc/info").json()["scales"]
    double_slash = client.get("/data/sub//t.mrc/info").json()["scales"]
    assert double_slash == canonical
    assert len(canonical) > 1


def test_scale0_still_served_when_cache_valid(cached_setup):
    client, _, _, relpath = cached_setup
    resp = client.get(f"/data/{relpath}/1_1_1/0-8_0-8_0-8")
    assert resp.status_code == 200
    assert len(resp.content) == 8 * 8 * 8 * 2


def test_cached_info_has_etag_derived_from_fingerprint(cached_setup):
    client, _, cache_root, relpath = cached_setup
    from mrcng.paths import dataset_id, cache_dir_for
    cache_dir = cache_dir_for(cache_root, dataset_id(relpath))
    fp = json.loads((cache_dir / "fingerprint.json").read_text())

    resp = client.get(f"/data/{relpath}/info")
    assert resp.status_code == 200
    assert resp.headers["etag"] == f'"{fp["source_header_sha256"][:16]}-{fp["derivation_version"]}"'


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


def test_cached_chunk_grid_misalignment_404s_before_touching_disk(cached_setup):
    # Regression: the cached-chunk branch skipped clip_chunk_to_scale entirely
    # (unlike scale-0), relying only on the file happening not to exist. A
    # non-grid-aligned request for a real, valid scale key should still 404,
    # verified against clip_chunk_to_scale's own validation rather than luck.
    client, _, _, relpath = cached_setup
    # 2_2_2 has chunk_size (8,8,8); origin 3 is not a multiple of 8.
    resp = client.get(f"/data/{relpath}/2_2_2/3-11_0-8_0-8")
    assert resp.status_code == 404


def test_cached_chunk_unknown_scale_key_404s(cached_setup):
    client, _, _, relpath = cached_setup
    # well-formed scale-key shape, but not a level this build ever produced
    resp = client.get(f"/data/{relpath}/99_99_99/0-8_0-8_0-8")
    assert resp.status_code == 404


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


def test_repeated_requests_reuse_the_memoized_validity(cached_setup, monkeypatch):
    # The expensive part -- reading fingerprint.json and hashing the header --
    # should run once per fingerprint version, not once per request.
    client, _, _, relpath = cached_setup
    from mrcng.server import fdcache as fdcache_module

    calls = {"read": 0, "validate": 0}
    real_read, real_validate = fdcache_module.read_fingerprint, fdcache_module.validate

    def counting_read(cache_dir):
        calls["read"] += 1
        return real_read(cache_dir)

    def counting_validate(*args, **kwargs):
        calls["validate"] += 1
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(fdcache_module, "read_fingerprint", counting_read)
    monkeypatch.setattr(fdcache_module, "validate", counting_validate)

    for _ in range(5):
        assert client.get(f"/data/{relpath}/info").status_code == 200
        assert client.get(f"/data/{relpath}/2_2_2/0-8_0-8_0-8").status_code == 200

    assert calls["read"] == 1
    assert calls["validate"] == 1


def test_a_build_completing_is_visible_without_a_new_request_needing_eviction(tmp_path, make_mrc_file):
    # Regression risk if validity were memoized against the *source* key alone:
    # the source file never changes when a build completes, so a naive memo
    # keyed only on (path, size, mtime_ns) would keep serving "no cache"
    # forever, until this fd-cache entry happened to get evicted. Keying on
    # fingerprint.json's own stat picks the build up on the very next request.
    source_root = tmp_path / "source"; source_root.mkdir()
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    relpath = "tomo.mrc"
    make_mrc_file(name=f"source/{relpath}", shape=(32, 32, 32), mode=1)

    settings = Settings(source_root=source_root, cache_root=cache_root, chunk_size=(8, 8, 8))
    client = TestClient(create_app(settings))

    assert len(client.get(f"/data/{relpath}/info").json()["scales"]) == 1

    params = Params(chunk_size=(8, 8, 8), downsample="mean", min_axis_size=8,
                     max_levels=3, dtype="int16", encoding="raw")
    build_one(source_root, cache_root, relpath, params)

    assert len(client.get(f"/data/{relpath}/info").json()["scales"]) > 1


def test_valid_cache_serves_the_built_info_bytes(cached_setup):
    # The inverse of the old rebuilt-from-header guard: the body must now be
    # exactly what the build wrote, so info can never disagree with the chunks
    # sitting next to it on disk. Without the sentinel mutation below,
    # json.dumps(build_info(...)) recomputed from the header can happen to be
    # byte-identical to the file on disk for this fixture, so the assertion
    # couldn't fail even against the old recompute-from-header implementation.
    # Adding a key build_info could never produce forces the comparison to
    # only pass if the response is the on-disk bytes, verbatim.
    client, _, cache_root, relpath = cached_setup
    from mrcng.paths import dataset_id, cache_dir_for
    cache_dir = cache_dir_for(cache_root, dataset_id(relpath))

    info_path = cache_dir / "info"
    mutated = json.loads(info_path.read_text())
    mutated["_sdd_sentinel"] = "served-from-disk"
    mutated_bytes = json.dumps(mutated).encode()
    info_path.write_bytes(mutated_bytes)

    resp = client.get(f"/data/{relpath}/info")
    assert resp.status_code == 200
    assert resp.content == mutated_bytes


def test_stale_derivation_invalidates_the_cache(cached_setup):
    # Replaces test_cached_info_is_rebuilt_from_the_header_not_served_from_disk.
    # That test existed because a derivation fix used to leave every cached info
    # stale and still served (46e8a88: a zero-cella-z tilt stack advertising
    # "resolution": [.., .., 0.0] long after the header fix landed). We now
    # serve the stored bytes, so the protection has to come from invalidation
    # instead: a wrong derivation_version stands in for any such code change, and
    # must drop the entry to the single-scale fallback rather than serve it.
    client, _, cache_root, relpath = cached_setup
    from mrcng.paths import dataset_id, cache_dir_for
    cache_dir = cache_dir_for(cache_root, dataset_id(relpath))

    fp_path = cache_dir / "fingerprint.json"
    fp = json.loads(fp_path.read_text())
    assert len(fp["scales"]) > 0, "fixture must have built at least one extra level"
    fp["derivation_version"] = 999
    fp_path.write_text(json.dumps(fp))

    resp = client.get(f"/data/{relpath}/info")
    assert resp.status_code == 200
    assert [s["key"] for s in resp.json()["scales"]] == ["1_1_1"]


def test_cached_info_etag_covers_the_derivation_version(cached_setup):
    # After a derivation change and rebuild the source is byte-identical, so an
    # ETag built from source_header_sha256 alone would not change while the info
    # body did -- a client holding a 304 would keep stale metadata forever.
    client, _, cache_root, relpath = cached_setup
    from mrcng.paths import dataset_id, cache_dir_for
    cache_dir = cache_dir_for(cache_root, dataset_id(relpath))
    fp = json.loads((cache_dir / "fingerprint.json").read_text())

    resp = client.get(f"/data/{relpath}/info")
    assert str(fp["derivation_version"]) in resp.headers["etag"]
    assert fp["source_header_sha256"][:16] in resp.headers["etag"]


def test_valid_fingerprint_with_unreadable_info_falls_back(cached_setup):
    # The fingerprint is written last, after info is fsynced, so this should be
    # unreachable -- but "missing, stale, incompatible or corrupt all read as no
    # cache" is the codebase's ground rule, and a half-deleted cache entry must
    # degrade rather than 500.
    client, _, cache_root, relpath = cached_setup
    from mrcng.paths import dataset_id, cache_dir_for
    cache_dir = cache_dir_for(cache_root, dataset_id(relpath))
    (cache_dir / "info").unlink()

    resp = client.get(f"/data/{relpath}/info")
    assert resp.status_code == 200
    assert [s["key"] for s in resp.json()["scales"]] == ["1_1_1"]
