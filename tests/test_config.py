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
