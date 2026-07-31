import json
import os

from mrcng.cli import main


def test_build_writes_report_jsonl(tmp_path, make_mrc_file):
    source_root = tmp_path / "source"
    source_root.mkdir()
    make_mrc_file(name="source/a.mrc", shape=(16, 16, 16), mode=1)
    cache_root = tmp_path / "cache"
    report_path = tmp_path / "report.jsonl"

    rc = main([
        "build", str(source_root), "--cache-root", str(cache_root),
        "--chunk-size", "8,8,8", "--min-axis-size", "8", "--max-levels", "3",
        "--report", str(report_path),
    ])
    assert rc == 0

    lines = report_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["relpath"] == "a.mrc"
    assert record["status"] == "built"
    assert record["voxel_size_is_default"] is False


def test_build_report_flags_default_voxel_size(tmp_path, make_mrc_file):
    # Regression: a zero-cella file built without any indication in the report
    # that its voxel size is a made-up (1,1,1) fallback, not read from the file.
    source_root = tmp_path / "source"
    source_root.mkdir()
    make_mrc_file(name="source/a.mrc", shape=(16, 16, 16), mode=1,
                  voxel_size_angstrom=(0.0, 0.0, 0.0))
    cache_root = tmp_path / "cache"
    report_path = tmp_path / "report.jsonl"

    rc = main([
        "build", str(source_root), "--cache-root", str(cache_root),
        "--chunk-size", "8,8,8", "--min-axis-size", "8", "--max-levels", "3",
        "--report", str(report_path),
    ])
    assert rc == 0
    record = json.loads(report_path.read_text().strip())
    assert record["voxel_size_is_default"] is True


def test_status_reports_missing_and_valid(tmp_path, make_mrc_file, capsys):
    source_root = tmp_path / "source"
    source_root.mkdir()
    make_mrc_file(name="source/a.mrc", shape=(16, 16, 16), mode=1)
    cache_root = tmp_path / "cache"

    main(["build", str(source_root), "--cache-root", str(cache_root), "--chunk-size", "8,8,8"])
    capsys.readouterr()  # discard build's output
    # status must be told the same chunk-size the build used, or it correctly
    # reports incompatible -- see test_status_reports_incompatible_for_real below.
    rc = main(["status", str(source_root), "--cache-root", str(cache_root), "--chunk-size", "8,8,8"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "a.mrc" in out
    assert "valid" in out


def test_status_reports_incompatible_for_real(tmp_path, make_mrc_file, capsys):
    # Regression: status built its comparison Params from the fingerprint's own
    # params dict, so validate() compared params against itself and could never
    # report incompatible no matter what was actually configured.
    source_root = tmp_path / "source"
    source_root.mkdir()
    make_mrc_file(name="source/a.mrc", shape=(16, 16, 16), mode=1)
    cache_root = tmp_path / "cache"

    main(["build", str(source_root), "--cache-root", str(cache_root), "--chunk-size", "8,8,8"])
    capsys.readouterr()
    rc = main(["status", str(source_root), "--cache-root", str(cache_root), "--chunk-size", "32,32,32"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "incompatible" in out


def test_prune_removes_orphaned_cache_entries(tmp_path, make_mrc_file):
    source_root = tmp_path / "source"
    source_root.mkdir()
    mrc_path = make_mrc_file(name="source/a.mrc", shape=(16, 16, 16), mode=1)
    cache_root = tmp_path / "cache"

    main(["build", str(source_root), "--cache-root", str(cache_root), "--chunk-size", "8,8,8"])
    os.remove(mrc_path)

    rc = main(["prune", "--cache-root", str(cache_root), "--source-root", str(source_root)])
    assert rc == 0

    from mrcng.paths import dataset_id, cache_dir_for
    cache_dir = cache_dir_for(cache_root, dataset_id("a.mrc"))
    assert not cache_dir.exists()


def test_prune_respects_glob(tmp_path, make_mrc_file):
    # Regression: prune hardcoded *.mrc, so a tree built with a different glob
    # (e.g. *.rec) had every entry deleted as "orphaned" on the first run.
    source_root = tmp_path / "source"
    source_root.mkdir()
    make_mrc_file(name="source/a.rec", shape=(16, 16, 16), mode=1)
    cache_root = tmp_path / "cache"

    main(["build", str(source_root), "--cache-root", str(cache_root),
          "--glob", "*.rec", "--chunk-size", "8,8,8"])

    from mrcng.paths import dataset_id, cache_dir_for
    cache_dir = cache_dir_for(cache_root, dataset_id("a.rec"))
    assert cache_dir.exists()

    rc = main(["prune", "--cache-root", str(cache_root), "--source-root", str(source_root),
               "--glob", "*.rec"])
    assert rc == 0
    assert cache_dir.exists()  # not orphaned once prune is told about the same glob


def test_status_respects_glob(tmp_path, make_mrc_file, capsys):
    source_root = tmp_path / "source"
    source_root.mkdir()
    make_mrc_file(name="source/a.rec", shape=(16, 16, 16), mode=1)
    cache_root = tmp_path / "cache"

    main(["build", str(source_root), "--cache-root", str(cache_root),
          "--glob", "*.rec", "--chunk-size", "8,8,8"])
    capsys.readouterr()

    rc = main(["status", str(source_root), "--cache-root", str(cache_root),
               "--glob", "*.rec", "--chunk-size", "8,8,8"])
    assert rc == 0
    assert "a.rec: valid" in capsys.readouterr().out


def test_build_returns_nonzero_when_a_file_fails(tmp_path, make_mrc_file):
    # Regression: _build_command always returned 0, even when every file in
    # the report had status "failed".
    source_root = tmp_path / "source"
    source_root.mkdir()
    make_mrc_file(name="source/good.mrc", shape=(16, 16, 16), mode=1)
    # a file matching the glob that isn't a real MRC -> parse_header fails
    (source_root / "bad.mrc").write_bytes(b"not an mrc file")
    cache_root = tmp_path / "cache"

    rc = main(["build", str(source_root), "--cache-root", str(cache_root), "--chunk-size", "8,8,8"])
    assert rc == 1


def test_build_jobs_parallelizes_without_changing_results(tmp_path, make_mrc_file):
    source_root = tmp_path / "source"
    source_root.mkdir()
    for name in ("a.mrc", "b.mrc", "c.mrc"):
        make_mrc_file(name=f"source/{name}", shape=(16, 16, 16), mode=1)
    cache_root = tmp_path / "cache"
    report_path = tmp_path / "report.jsonl"

    rc = main([
        "build", str(source_root), "--cache-root", str(cache_root),
        "--chunk-size", "8,8,8", "--jobs", "3", "--report", str(report_path),
    ])
    assert rc == 0

    records = [json.loads(line) for line in report_path.read_text().strip().splitlines()]
    assert {r["relpath"] for r in records} == {"a.mrc", "b.mrc", "c.mrc"}
    assert all(r["status"] == "built" for r in records)


def test_build_accepts_log_level(tmp_path, make_mrc_file):
    source_root = tmp_path / "source"
    source_root.mkdir()
    make_mrc_file(name="source/a.mrc", shape=(16, 16, 16), mode=1)
    cache_root = tmp_path / "cache"

    rc = main(["build", str(source_root), "--cache-root", str(cache_root),
               "--chunk-size", "8,8,8", "--log-level", "DEBUG"])
    assert rc == 0


def test_build_assume_mode0_flag_reaches_the_header_parser(tmp_path, make_mrc_file):
    source_root = tmp_path / "source"
    source_root.mkdir()
    make_mrc_file(name="source/a.mrc", shape=(8, 8, 8), mode=0)  # no IMOD stamp -> ambiguous
    cache_root = tmp_path / "cache"
    report_path = tmp_path / "report.jsonl"

    rc = main([
        "build", str(source_root), "--cache-root", str(cache_root),
        "--chunk-size", "8,8,8", "--assume-mode0", "uint8", "--report", str(report_path),
    ])
    assert rc == 0
    record = json.loads(report_path.read_text().strip())
    assert record["status"] == "built"

    from mrcng.fingerprint import read_fingerprint
    from mrcng.paths import dataset_id, cache_dir_for
    fp = read_fingerprint(cache_dir_for(cache_root, dataset_id("a.mrc")))
    assert fp["params"]["dtype"] == "uint8"


def test_build_from_file_builds_only_listed(tmp_path, make_mrc_file):
    """--from-file builds exactly the listed relpaths, not every *.mrc in the tree.

    Comments (#) and blank lines are ignored, and no implicit *.mrc glob is
    added — so a stray sibling .mrc (a gain reference, say) is left unbuilt.
    """
    source_root = tmp_path / "source"
    source_root.mkdir()
    make_mrc_file(name="source/keep.mrc", shape=(16, 16, 16), mode=1)
    make_mrc_file(name="source/skip.mrc", shape=(16, 16, 16), mode=1)
    cache_root = tmp_path / "cache"
    list_file = tmp_path / "list.txt"
    list_file.write_text("keep.mrc\n# skip.mrc is deliberately not listed\n\n")

    rc = main(["build", str(source_root), "--cache-root", str(cache_root),
               "--chunk-size", "8,8,8", "--from-file", str(list_file)])
    assert rc == 0

    from mrcng.paths import dataset_id, cache_dir_for
    assert cache_dir_for(cache_root, dataset_id("keep.mrc")).exists()
    assert not cache_dir_for(cache_root, dataset_id("skip.mrc")).exists()


def test_build_from_file_skips_missing_and_out_of_tree(tmp_path, make_mrc_file):
    """Entries that don't resolve to a file under source_root are skipped, not fatal."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    make_mrc_file(name="source/a.mrc", shape=(16, 16, 16), mode=1)
    cache_root = tmp_path / "cache"
    list_file = tmp_path / "list.txt"
    list_file.write_text("a.mrc\nnope.mrc\n../outside.mrc\n")  # real, missing, escape

    rc = main(["build", str(source_root), "--cache-root", str(cache_root),
               "--chunk-size", "8,8,8", "--from-file", str(list_file)])
    assert rc == 0  # the two bad entries are skipped; the one real file builds

    from mrcng.paths import dataset_id, cache_dir_for
    assert cache_dir_for(cache_root, dataset_id("a.mrc")).exists()


def test_status_warns_on_ambiguous_mode0_signedness(tmp_path, make_mrc_file, caplog):
    import logging
    source_root = tmp_path / "source"
    source_root.mkdir()
    make_mrc_file(name="source/a.mrc", shape=(8, 8, 8), mode=0)
    cache_root = tmp_path / "cache"

    main(["build", str(source_root), "--cache-root", str(cache_root), "--chunk-size", "8,8,8"])
    with caplog.at_level(logging.WARNING, logger="mrcng.pyramid"):
        main(["status", str(source_root), "--cache-root", str(cache_root), "--chunk-size", "8,8,8"])

    assert any("ambiguous" in r.message for r in caplog.records if r.name == "mrcng.pyramid")
