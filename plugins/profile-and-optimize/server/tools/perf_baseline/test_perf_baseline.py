"""Unit tests for perf_baseline.

Pure-Python tests; no Slurm / MCP / network dependencies. Designed to run
under ``pytest -q tools/perf_baseline/test_perf_baseline.py`` from the
``server/`` directory.
"""

from __future__ import annotations

import json
import subprocess
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


def test_git_sha_discovery_supports_linked_worktree(tmp_path: Path) -> None:
    from tools.perf_baseline.helpers import discover_profile_and_optimize_sha

    primary = tmp_path / "primary"
    linked = tmp_path / "linked"
    subprocess.run(["git", "init", "-q", str(primary)], check=True)
    (primary / "tracked.txt").write_text("test\n")
    subprocess.run(["git", "-C", str(primary), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(primary),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-q",
            "-m",
            "initial",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(primary), "worktree", "add", "-q", "-b", "linked", str(linked)],
        check=True,
    )
    expected = subprocess.run(
        ["git", "-C", str(linked), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert discover_profile_and_optimize_sha(linked) == expected


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
    assert payload["registered_by"] == {"operator_user": "op"}


def test_write_source_md_includes_operator_attribution(tmp_path: Path) -> None:
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
    assert "**Registered by:** `op` on `host`" in text
    assert "## Attribution" in text


def test_append_index_creates_header_once(tmp_path: Path) -> None:
    reg = tmp_path / "reg"
    reg.mkdir()
    append_index(
        reg, slug="20260101T000000Z", registered_at_utc="2026-01-01T00:00:00Z", value=1.0, unit="GB/s", notes="first"
    )
    append_index(
        reg, slug="20260102T000000Z", registered_at_utc="2026-01-02T00:00:00Z", value=2.0, unit="GB/s", notes="second"
    )
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
    (root / "pyproject.toml").write_text("[project]\nname = 'fake'\nversion = '0'\n")
    (root / "mcp_surface.py").write_text("# marker\n")
    (root / "tools").mkdir()
    return root


def test_contract_shape_for_record_and_diff() -> None:
    for verb in ("record", "diff"):
        spec = CONTRACT[verb]
        assert spec["safety"] == "writes_artifacts"
        assert spec["ack"] is None
        assert spec["json"] is True
        assert "required" in spec and "optional" in spec


def test_record_scalar_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _seed_repo_root(tmp_path)
    monkeypatch.setenv("PROFILE_AND_OPTIMIZE_REPO_ROOT", str(root))
    src = tmp_path / "scalar.txt"
    src.write_text("3.14\n")
    rc = main(
        [
            "record",
            "--family",
            "demo",
            "--measurement",
            "throughput",
            "--source",
            str(src),
            "--direction",
            "higher-is-better",
            "--value",
            "3.14",
            "--unit",
            "tokens/s",
            "--notes",
            "first registration",
            "--json",
        ]
    )
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


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--family", "../outside"),
        ("--family", "/tmp/outside"),
        ("--family", "nested/family"),
        ("--measurement", "../outside"),
        ("--measurement", "nested/measurement"),
        ("--measurement", r"nested\measurement"),
    ],
)
def test_record_rejects_unsafe_artifact_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flag: str,
    value: str,
) -> None:
    root = _seed_repo_root(tmp_path)
    monkeypatch.setenv("PROFILE_AND_OPTIMIZE_REPO_ROOT", str(root))
    source = tmp_path / "source.txt"
    source.write_text("1\n")
    arguments = {
        "--family": "demo",
        "--measurement": "throughput",
    }
    arguments[flag] = value

    rc = main(
        [
            "record",
            "--family",
            arguments["--family"],
            "--measurement",
            arguments["--measurement"],
            "--source",
            str(source),
            "--direction",
            "higher-is-better",
            "--value",
            "1",
            "--json",
        ]
    )

    assert rc == 2
    assert "safe path segment" in capsys.readouterr().err
    assert not (root / "experiments").exists()


def test_record_rejects_symlink_escape_from_registry_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _seed_repo_root(tmp_path)
    monkeypatch.setenv("PROFILE_AND_OPTIMIZE_REPO_ROOT", str(root))
    source = tmp_path / "source.txt"
    source.write_text("1\n")
    registry = root / "experiments" / "artifacts" / "perf-baselines"
    registry.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (registry / "demo").symlink_to(outside, target_is_directory=True)

    rc = main(
        [
            "record",
            "--family",
            "demo",
            "--measurement",
            "throughput",
            "--source",
            str(source),
            "--direction",
            "higher-is-better",
            "--value",
            "1",
        ]
    )

    assert rc == 2
    assert "escapes its registry root" in capsys.readouterr().err
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_record_rejects_nonfinite_scalar_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    value: str,
) -> None:
    root = _seed_repo_root(tmp_path)
    monkeypatch.setenv("PROFILE_AND_OPTIMIZE_REPO_ROOT", str(root))
    source = tmp_path / "source.txt"
    source.write_text("1\n")

    rc = main(
        [
            "record",
            "--family",
            "demo",
            "--measurement",
            "throughput",
            "--source",
            str(source),
            "--direction",
            "higher-is-better",
            f"--value={value}",
        ]
    )

    assert rc == 2
    assert "finite" in capsys.readouterr().err
    assert not (root / "experiments").exists()


def test_record_validates_json_source_against_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _seed_repo_root(tmp_path)
    monkeypatch.setenv("PROFILE_AND_OPTIMIZE_REPO_ROOT", str(root))
    source = tmp_path / "source.json"
    source.write_text('{"latency_ms": "slow"}\n')
    schema = tmp_path / "schema.json"
    schema.write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {"latency_ms": {"type": "number"}},
                "required": ["latency_ms"],
                "additionalProperties": False,
            }
        )
    )

    rc = main(
        [
            "record",
            "--family",
            "demo",
            "--measurement",
            "latency",
            "--source",
            str(source),
            "--direction",
            "lower-is-better",
            "--schema",
            str(schema),
            "--json",
        ]
    )

    assert rc == 2
    assert "fails JSON Schema" in capsys.readouterr().err
    assert not (root / "experiments").exists()

    source.write_text('{"latency_ms": 12.5}\n')
    assert (
        main(
            [
                "record",
                "--family",
                "demo",
                "--measurement",
                "latency",
                "--source",
                str(source),
                "--direction",
                "lower-is-better",
                "--schema",
                str(schema),
                "--json",
            ]
        )
        == 0
    )


def test_diff_scalar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    root = _seed_repo_root(tmp_path)
    monkeypatch.setenv("PROFILE_AND_OPTIMIZE_REPO_ROOT", str(root))
    src = tmp_path / "scalar.txt"
    src.write_text("100\n")
    main(
        [
            "record",
            "--family",
            "demo",
            "--measurement",
            "throughput",
            "--source",
            str(src),
            "--direction",
            "two-sided",
            "--value",
            "100",
            "--unit",
            "tokens/s",
            "--json",
        ]
    )
    record_envelope = json.loads(capsys.readouterr().out)
    baseline_dir = Path(record_envelope["entry_dir"])

    current = tmp_path / "current.txt"
    current.write_text("106\n")  # 6% two-sided deviation; default tolerance is 5%
    rc = main(
        [
            "diff",
            "--baseline",
            str(baseline_dir),
            "--current",
            str(current),
            "--tolerance-percent",
            "5",
            "--json",
        ]
    )
    assert rc == 0
    diff_envelope = json.loads(capsys.readouterr().out)
    assert diff_envelope["tool"] == "perf_baseline_diff"
    # A scalar is the headline dimension, so an out-of-tolerance delta is RED.
    assert diff_envelope["verdict"] == "RED"
    assert diff_envelope["deltas_count"] == 1


@pytest.mark.parametrize(
    ("direction", "current_value", "expected"),
    [
        ("higher-is-better", 110.0, "GREEN"),
        ("higher-is-better", 90.0, "RED"),
        ("lower-is-better", 90.0, "GREEN"),
        ("lower-is-better", 110.0, "RED"),
        ("two-sided", 110.0, "RED"),
    ],
)
def test_diff_scalar_honors_metric_direction_and_writes_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    direction: str,
    current_value: float,
    expected: str,
) -> None:
    root = _seed_repo_root(tmp_path)
    monkeypatch.setenv("PROFILE_AND_OPTIMIZE_REPO_ROOT", str(root))
    source = tmp_path / "baseline.txt"
    source.write_text("100\n")
    assert (
        main(
            [
                "record",
                "--family",
                "demo",
                "--measurement",
                "headline",
                "--source",
                str(source),
                "--value",
                "100",
                "--unit",
                "units",
                "--direction",
                direction,
                "--json",
            ]
        )
        == 0
    )
    baseline_dir = Path(json.loads(capsys.readouterr().out)["entry_dir"])
    current = tmp_path / "current.txt"
    current.write_text(f"{current_value}\n")

    assert (
        main(
            [
                "diff",
                "--baseline",
                str(baseline_dir),
                "--current",
                str(current),
                "--tolerance-percent",
                "5",
                "--json",
            ]
        )
        == 0
    )

    envelope = json.loads(capsys.readouterr().out)
    diff_dir = Path(envelope["diff_dir"])
    assert envelope["verdict"] == expected
    assert envelope["direction"] == direction
    assert (diff_dir / "diff.md").is_file()
    assert (diff_dir / "current-snapshot.txt").read_text() == f"{current_value}\n"
    detail = json.loads((diff_dir / "diff.json").read_text())
    assert detail["direction"] == direction


def test_diff_scalar_green_when_within_tolerance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _seed_repo_root(tmp_path)
    monkeypatch.setenv("PROFILE_AND_OPTIMIZE_REPO_ROOT", str(root))
    src = tmp_path / "scalar.txt"
    src.write_text("100\n")
    main(
        [
            "record",
            "--family",
            "demo",
            "--measurement",
            "throughput",
            "--source",
            str(src),
            "--direction",
            "two-sided",
            "--value",
            "100",
            "--unit",
            "tokens/s",
            "--json",
        ]
    )
    record_envelope = json.loads(capsys.readouterr().out)
    baseline_dir = Path(record_envelope["entry_dir"])

    current = tmp_path / "current.txt"
    current.write_text("103\n")  # 3% delta; default tolerance is 5%
    rc = main(
        [
            "diff",
            "--baseline",
            str(baseline_dir),
            "--current",
            str(current),
            "--tolerance-percent",
            "5",
            "--json",
        ]
    )
    assert rc == 0
    diff_envelope = json.loads(capsys.readouterr().out)
    assert diff_envelope["verdict"] == "GREEN"


def _write_baseline_entry(
    tmp_path: Path,
    *,
    value: float | None,
    snapshot: dict[str, object] | None = None,
) -> Path:
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    (baseline_dir / "baseline.json").write_text(
        json.dumps(
            {
                "family": "demo",
                "measurement": "metric",
                "value": value,
                "source_sha256": "test",
            }
        )
    )
    if snapshot is not None:
        (baseline_dir / "source-snapshot.json").write_text(json.dumps(snapshot))
    return baseline_dir


def test_diff_zero_baseline_to_nonzero_is_not_green(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _seed_repo_root(tmp_path)
    monkeypatch.setenv("PROFILE_AND_OPTIMIZE_REPO_ROOT", str(root))
    baseline_dir = _write_baseline_entry(tmp_path, value=0.0)
    current = tmp_path / "current.txt"
    current.write_text("1\n")

    assert main(["diff", "--baseline", str(baseline_dir), "--current", str(current), "--json"]) == 0

    envelope = json.loads(capsys.readouterr().out)
    assert envelope["verdict"] == "RED"
    detail = json.loads((Path(envelope["diff_dir"]) / "diff.json").read_text())
    assert detail["deltas_top20"][0]["delta_pct"] is None


def test_structured_diff_skips_metadata_before_sorting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _seed_repo_root(tmp_path)
    monkeypatch.setenv("PROFILE_AND_OPTIMIZE_REPO_ROOT", str(root))
    baseline_dir = _write_baseline_entry(
        tmp_path,
        value=None,
        snapshot={"metric": 100.0, "label": "baseline"},
    )
    current = tmp_path / "current.json"
    current.write_text('{"metric": 110.0, "label": "candidate"}\n')

    assert main(["diff", "--baseline", str(baseline_dir), "--current", str(current), "--json"]) == 0

    envelope = json.loads(capsys.readouterr().out)
    assert envelope["verdict"] == "YELLOW"
    assert envelope["deltas_count"] == 1


def test_structured_diff_honors_absolute_tolerance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _seed_repo_root(tmp_path)
    monkeypatch.setenv("PROFILE_AND_OPTIMIZE_REPO_ROOT", str(root))
    baseline_dir = _write_baseline_entry(tmp_path, value=None, snapshot={"metric": 1000.0})
    current = tmp_path / "current.json"
    current.write_text('{"metric": 1006.0}\n')

    assert (
        main(
            [
                "diff",
                "--baseline",
                str(baseline_dir),
                "--current",
                str(current),
                "--tolerance-percent",
                "5",
                "--tolerance-absolute",
                "5",
                "--json",
            ]
        )
        == 0
    )

    envelope = json.loads(capsys.readouterr().out)
    assert envelope["verdict"] == "YELLOW"
    assert envelope["tolerance_absolute"] == 5.0


@pytest.mark.parametrize(
    ("family", "measurement"),
    [
        ("../../outside", "throughput"),
        ("demo", "../../outside"),
        ("nested/family", "throughput"),
    ],
)
def test_diff_rejects_unsafe_metadata_from_baseline_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    family: str,
    measurement: str,
) -> None:
    root = _seed_repo_root(tmp_path)
    monkeypatch.setenv("PROFILE_AND_OPTIMIZE_REPO_ROOT", str(root))
    baseline_dir = tmp_path / "malicious-baseline"
    baseline_dir.mkdir()
    (baseline_dir / "baseline.json").write_text(
        json.dumps(
            {
                "family": family,
                "measurement": measurement,
                "value": 1.0,
                "source_sha256": "test",
            }
        )
    )
    current = tmp_path / "current.txt"
    current.write_text("1\n")

    rc = main(
        [
            "diff",
            "--baseline",
            str(baseline_dir),
            "--current",
            str(current),
        ]
    )

    assert rc == 2
    assert "invalid baseline metadata" in capsys.readouterr().err
    assert not (root / "experiments").exists()


def test_build_parser_help_does_not_crash() -> None:
    parser = build_parser()
    assert parser is not None
    # SystemExit on --help is expected.
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
