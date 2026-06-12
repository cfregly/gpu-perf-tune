"""Unit tests for perf_baseline.

Pure-Python tests; no Slurm / MCP / network dependencies. Designed to run
under ``pytest -q tools/perf_baseline/test_perf_baseline.py`` from the
``server/`` directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.perf_baseline.helpers import (
    append_index,
    sha256_of_path,
    utc_now_iso,
    utc_now_slug,
    write_baseline_json,
    write_source_md,
)
from tools.perf_baseline.perf_baseline_cli import CONTRACT, build_parser, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_sha256_of_file(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("hello\n")
    assert sha256_of_path(f) == "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"


def test_sha256_of_directory_is_deterministic(tmp_path: Path) -> None:
    d = tmp_path / "d"
    d.mkdir()
    (d / "x.txt").write_text("one")
    (d / "y.txt").write_text("two")
    h1 = sha256_of_path(d)
    # Order-independent: re-writing in different order should give same hash.
    (d / "x.txt").unlink()
    (d / "y.txt").unlink()
    (d / "y.txt").write_text("two")
    (d / "x.txt").write_text("one")
    h2 = sha256_of_path(d)
    assert h1 == h2


def test_utc_helpers() -> None:
    iso = utc_now_iso()
    slug = utc_now_slug()
    assert iso.endswith("Z") and "T" in iso
    assert "Z" in slug and len(slug) >= 16 and ":" not in slug


def test_write_baseline_json_round_trips(tmp_path: Path) -> None:
    src = tmp_path / "src.txt"
    src.write_text("payload")
    entry = tmp_path / "entry"
    entry.mkdir()
    write_baseline_json(
        entry,
        family="fam",
        measurement="m",
        value=1.23,
        unit="ms",
        source_path=src,
        source_sha256="abc123",
        schema_path=None,
        registered_at_utc=utc_now_iso(),
        operator_user="op",
        hostname="host",
        uname="uname-text",
        profile_and_optimize_sha="deadbeef",
        notes="hello",
    )
    payload = json.loads((entry / "baseline.json").read_text())
    assert payload["family"] == "fam"
    assert payload["measurement"] == "m"
    assert payload["value"] == 1.23
    assert payload["unit"] == "ms"
    assert payload["source_sha256"] == "abc123"
    assert payload["registered_by"]["team"] == "the MLPerf team"


def test_write_source_md_includes_team_attribution(tmp_path: Path) -> None:
    entry = tmp_path / "entry"
    entry.mkdir()
    write_source_md(
        entry,
        family="fam",
        measurement="m",
        operator_user="op",
        hostname="host",
        registered_at_utc=utc_now_iso(),
        profile_and_optimize_sha="deadbeefcafe",
        source_path=tmp_path / "src.txt",
        source_sha256="abc",
        notes="ok",
    )
    text = (entry / "SOURCE.md").read_text()
    assert "the MLPerf team" in text
    assert "Team Attribution" in text


def test_append_index_creates_header_once(tmp_path: Path) -> None:
    reg = tmp_path / "reg"
    reg.mkdir()
    append_index(reg, slug="20260101T000000Z", registered_at_utc="2026-01-01T00:00:00Z", value=1.0, unit="GB/s", notes="first")
    append_index(reg, slug="20260102T000000Z", registered_at_utc="2026-01-02T00:00:00Z", value=2.0, unit="GB/s", notes="second")
    body = (reg / "INDEX.md").read_text()
    assert body.count("# Perf-baseline registry index") == 1
    assert "20260101T000000Z" in body
    assert "20260102T000000Z" in body


# ---------------------------------------------------------------------------
# CLI end-to-end
# ---------------------------------------------------------------------------


def _seed_repo_root(tmp_path: Path) -> Path:
    root = tmp_path / "fake-repo-root"
    root.mkdir()
    (root / "AGENTS.md").write_text("# fake\n")
    (root / "tools").mkdir()
    return root


def test_contract_shape_for_record_and_diff() -> None:
    for verb in ("record", "diff"):
        spec = CONTRACT[verb]
        assert spec["safety"] == "writes_artifacts"
        assert spec["ack"] is None
        assert spec["json"] is True
        assert "required" in spec and "optional" in spec


def test_record_scalar_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    root = _seed_repo_root(tmp_path)
    monkeypatch.setenv("PROFILE_AND_OPTIMIZE_REPO_ROOT", str(root))
    src = tmp_path / "scalar.txt"
    src.write_text("3.14\n")
    rc = main([
        "record",
        "--family", "demo",
        "--measurement", "throughput",
        "--source", str(src),
        "--value", "3.14",
        "--unit", "tokens/s",
        "--notes", "first registration",
        "--json",
    ])
    assert rc == 0
    captured = capsys.readouterr().out
    envelope = json.loads(captured)
    assert envelope["tool"] == "perf_baseline_record"
    assert envelope["library"] == "perf_baseline"
    entry_dir = Path(envelope["entry_dir"])
    assert entry_dir.is_dir()
    assert (entry_dir / "baseline.json").is_file()
    assert (entry_dir / "SOURCE.md").is_file()
    payload = json.loads((entry_dir / "baseline.json").read_text())
    assert payload["value"] == 3.14
    assert payload["unit"] == "tokens/s"


def test_diff_scalar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    root = _seed_repo_root(tmp_path)
    monkeypatch.setenv("PROFILE_AND_OPTIMIZE_REPO_ROOT", str(root))
    src = tmp_path / "scalar.txt"
    src.write_text("100\n")
    main([
        "record",
        "--family", "demo",
        "--measurement", "throughput",
        "--source", str(src),
        "--value", "100",
        "--unit", "tokens/s",
        "--json",
    ])
    record_envelope = json.loads(capsys.readouterr().out)
    baseline_dir = Path(record_envelope["entry_dir"])

    current = tmp_path / "current.txt"
    current.write_text("106\n")  # 6% regression; default tolerance is 5%
    rc = main([
        "diff",
        "--baseline", str(baseline_dir),
        "--current", str(current),
        "--tolerance-percent", "5",
        "--json",
    ])
    assert rc == 0
    diff_envelope = json.loads(capsys.readouterr().out)
    assert diff_envelope["tool"] == "perf_baseline_diff"
    # 1 dimension over tolerance -> YELLOW (per _classify: <=2 keys => YELLOW;
    # 3+ keys => RED). Scalar diffs always produce exactly 1 dimension, so
    # the worst case for a scalar is YELLOW.
    assert diff_envelope["verdict"] == "YELLOW"
    assert diff_envelope["deltas_count"] == 1


def test_diff_scalar_green_when_within_tolerance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _seed_repo_root(tmp_path)
    monkeypatch.setenv("PROFILE_AND_OPTIMIZE_REPO_ROOT", str(root))
    src = tmp_path / "scalar.txt"
    src.write_text("100\n")
    main([
        "record",
        "--family", "demo",
        "--measurement", "throughput",
        "--source", str(src),
        "--value", "100",
        "--unit", "tokens/s",
        "--json",
    ])
    record_envelope = json.loads(capsys.readouterr().out)
    baseline_dir = Path(record_envelope["entry_dir"])

    current = tmp_path / "current.txt"
    current.write_text("103\n")  # 3% delta; default tolerance is 5%
    rc = main([
        "diff",
        "--baseline", str(baseline_dir),
        "--current", str(current),
        "--tolerance-percent", "5",
        "--json",
    ])
    assert rc == 0
    diff_envelope = json.loads(capsys.readouterr().out)
    assert diff_envelope["verdict"] == "GREEN"


def test_build_parser_help_does_not_crash() -> None:
    parser = build_parser()
    assert parser is not None
    # SystemExit on --help is expected.
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
