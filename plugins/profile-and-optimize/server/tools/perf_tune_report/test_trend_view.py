"""Tests for trend_view: longitudinal (model, variant_key) perf/quality trend."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.perf_tune_report.perf_tune_report_cli import main
from tools.perf_tune_report.schema import AtlasCell, write_jsonl
from tools.perf_tune_report.trend_view import build_trends, read_lake_rows, render_markdown


def _row(captured_at: str, tps: float, **kw) -> AtlasCell:
    base = dict(
        cell_id="c", model="GLM-5.1-NVFP4", hardware="GB300", quant="NVFP4",
        tensor_parallel=4, parallel_strategy="TP", mtp=False, max_num_batched_tokens=8192,
        concurrency=64, status="full", output_tps_per_gpu=tps, tpot_median_ms=20.0,
        ttft_avg_ms=100.0, request_throughput_avg=1.0, cache_mode="cold",
        dataset="random", cudagraph_mode="full", kv_cache_dtype="fp8_e4m3",
        image="infr/vllm:v2.12.3", captured_at=captured_at, backend="vllm-sweep",
    )
    base.update(kw)
    return AtlasCell(**base)


def test_build_trends_groups_by_variant_and_flags_regression():
    # Same variant (identical descriptor), two time points: throughput dropped 25% -> regression.
    rows = [
        _row("2026-06-01T00:00:00Z", 1000.0, image="infr/vllm:v2.12.3"),
        _row("2026-06-07T00:00:00Z", 750.0, image="infr/vllm:v2.13.0"),
    ]
    view = build_trends(rows, metric="output_tps_per_gpu", regression_pct=10.0)
    assert view["n_trends"] == 1
    t = view["trends"][0]
    assert t["n_points"] == 2
    assert t["delta_pct"] == -25.0
    assert t["regression"] is True  # tok/s dropped > 10%
    assert t["model"] == "GLM-5.1"
    assert "v2.13.0" in " ".join(t["images"]) or "v2.13.0" in str(t["images"])
    assert view["n_regressions"] == 1


def test_distinct_variants_are_separate_trends():
    # Different quant -> different variant_key -> two trend lines, not one.
    rows = [
        _row("2026-06-01T00:00:00Z", 1000.0, quant="NVFP4"),
        _row("2026-06-07T00:00:00Z", 1100.0, quant="FP8"),
    ]
    view = build_trends(rows, metric="output_tps_per_gpu")
    assert view["n_trends"] == 2
    assert all(t["n_points"] == 1 for t in view["trends"])  # one point each


def test_lower_better_metric_regression_direction():
    # TPOT (lower better): an INCREASE is the regression.
    rows = [
        _row("2026-06-01T00:00:00Z", 1000.0, tpot_median_ms=20.0),
        _row("2026-06-07T00:00:00Z", 1000.0, tpot_median_ms=26.0),
    ]
    view = build_trends(rows, metric="tpot_median_ms", regression_pct=10.0)
    assert view["lower_better"] is True
    assert view["trends"][0]["regression"] is True  # +30% TPOT


def test_cli_trend_view(tmp_path: Path, capsys):
    camps = tmp_path / "campaigns"
    for name, ts, tps in [("a-20260601T000000Z", "2026-06-01T00:00:00Z", 1000.0),
                          ("b-20260607T000000Z", "2026-06-07T00:00:00Z", 700.0)]:
        d = camps / name
        d.mkdir(parents=True)
        write_jsonl([_row(ts, tps)], d / "atlas.jsonl")
    rc = main(["trend_view", "--campaigns-dir", str(camps), "--metric", "output_tps_per_gpu",
               "--concurrency", "64", "--out", str(tmp_path / "TREND.md"), "--json"])
    assert rc == 0
    env = json.loads(capsys.readouterr().out)
    assert env["n_trends"] == 1 and env["n_regressions"] == 1
    assert env["source"] == "local-campaigns"
    md = (tmp_path / "TREND.md").read_text()
    assert "REGRESSION" in md and "trend over time" in md.lower()


def test_read_lake_rows_joins_vllm_commit_and_trends(tmp_path: Path):
    """A pulled-lake snapshot (atlas_v1 + campaign_v1 parquet): the two same-variant rows
    group into ONE trend, and the engine-version axis shows the joined vllm_commit, not the
    image tag (the A3 outcome)."""
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa  # noqa: WPS433 - test-only

    from tools.perf_tune_report.lake_writer import build_atlas_table

    # Two atlas rows, identical descriptor except image (-> same image-independent
    # variant_key), two captured_at points: throughput drops 1000 -> 750 == a regression.
    for camp, ts, tps, commit in [
        ("camp-a", "2026-06-01T00:00:00Z", 1000.0, "aaaaaaa1"),
        ("camp-b", "2026-06-07T00:00:00Z", 750.0, "bbbbbbb2"),
    ]:
        adir = tmp_path / f"perflake/perf-report/atlas_v1/dt=2026-06-07/campaign={camp}"
        adir.mkdir(parents=True)
        pq.write_table(build_atlas_table([_row(ts, tps)], camp), adir / "part-0.parquet")
        cdir = tmp_path / f"perflake/perf-report/campaign_v1/dt=2026-06-07/campaign={camp}"
        cdir.mkdir(parents=True)
        pq.write_table(pa.table({"campaign_id": [camp], "vllm_commit": [commit]}),
                       cdir / "part-0.parquet")

    rows = read_lake_rows(tmp_path)
    assert len(rows) == 2
    view = build_trends(rows, metric="output_tps_per_gpu", regression_pct=10.0)
    assert view["n_trends"] == 1          # same variant_key -> ONE trend across engine versions
    t = view["trends"][0]
    assert t["n_points"] == 2
    assert t["regression"] is True        # 1000 -> 750 tok/s
    assert set(t["images"]) == {"aaaaaaa1", "bbbbbbb2"}  # vllm_commit, not image tag


def test_cli_trend_view_lake_dir(tmp_path: Path, capsys):
    pq = pytest.importorskip("pyarrow.parquet")
    from tools.perf_tune_report.lake_writer import build_atlas_table

    adir = tmp_path / "perflake/perf-report/atlas_v1/dt=2026-06-07/campaign=camp-a"
    adir.mkdir(parents=True)
    pq.write_table(build_atlas_table([_row("2026-06-01T00:00:00Z", 1000.0)], "camp-a"),
                   adir / "part-0.parquet")
    rc = main(["trend_view", "--lake-dir", str(tmp_path), "--json"])
    assert rc == 0
    env = json.loads(capsys.readouterr().out)
    assert env["source"] == "published-lake"
    assert env["n_trends"] == 1
