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


def test_status_reports_missing_and_valid(tmp_path, make_mrc_file, capsys):
    source_root = tmp_path / "source"
    source_root.mkdir()
    make_mrc_file(name="source/a.mrc", shape=(16, 16, 16), mode=1)
    cache_root = tmp_path / "cache"

    main(["build", str(source_root), "--cache-root", str(cache_root), "--chunk-size", "8,8,8"])
    capsys.readouterr()  # discard build's output
    rc = main(["status", str(source_root), "--cache-root", str(cache_root)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "a.mrc" in out
    assert "valid" in out


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
