import pytest

from mrcng.server.config import Settings, parse_globs


def test_glob_settings_parse_from_a_comma_separated_env_var(tmp_path, monkeypatch):
    # Regression: these were tuple[str, ...], and pydantic-settings JSON-parses
    # complex-typed fields straight from the environment, so the documented
    # MRCNG_STACK_GLOBS='*/TiltSeries/*,*/Gains/*' raised SettingsError at startup
    # before any validator ran -- the server would not boot at all.
    monkeypatch.setenv("MRCNG_SOURCE_ROOT", str(tmp_path))
    monkeypatch.setenv("MRCNG_CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("MRCNG_STACK_GLOBS", "*/TiltSeries/*,*/Gains/*")
    monkeypatch.setenv("MRCNG_VOLUME_GLOBS", "*_ctf.mrc")

    settings = Settings()
    assert parse_globs(settings.stack_globs) == ("*/TiltSeries/*", "*/Gains/*")
    assert parse_globs(settings.volume_globs) == ("*_ctf.mrc",)


def test_glob_settings_default_to_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("MRCNG_SOURCE_ROOT", str(tmp_path))
    monkeypatch.setenv("MRCNG_CACHE_ROOT", str(tmp_path))
    monkeypatch.delenv("MRCNG_STACK_GLOBS", raising=False)
    monkeypatch.delenv("MRCNG_VOLUME_GLOBS", raising=False)

    settings = Settings()
    assert parse_globs(settings.stack_globs) == ()
    assert parse_globs(settings.volume_globs) == ()


@pytest.mark.parametrize("raw,expected", [
    ("", ()),
    ("  ", ()),
    ("a,,b", ("a", "b")),
    (" a , b ", ("a", "b")),
])
def test_parse_globs_ignores_blanks_and_whitespace(raw, expected):
    assert parse_globs(raw) == expected


def test_server_matches_globs_against_the_relpath_like_the_builder(tmp_path, make_mrc_file):
    """Regression: the server matched the absolute path, the builder the relpath.

    `*/TiltSeries/*` happens to work either way, so this only shows up with an
    anchored pattern -- and then the two sides disagree, the fingerprint reads
    INCOMPATIBLE, and the dataset silently drops to single-resolution.
    """
    from mrcng.server.fdcache import FdCache

    source_root = tmp_path / "source"
    (source_root / "Experimental" / "ts").mkdir(parents=True)
    make_mrc_file(name="source/Experimental/ts/s.mrc", shape=(256, 256, 8), mode=1)

    cache = FdCache(stack_globs=("Experimental/*",), source_root=source_root)
    try:
        with cache.open(source_root / "Experimental" / "ts" / "s.mrc") as handle:
            assert handle.hdr.is_image_stack is True
    finally:
        cache.close_all()
