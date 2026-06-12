"""Unit + smoke tests for the perf_tune_report library.

These run as part of the default ``pytest`` discovery surface (the
``tools/perf_tune_report`` path is listed in ``server/pyproject.toml``
``[tool.pytest.ini_options].testpaths``).

Coverage:

- ``test_schema_*``         -- AtlasCell invariants + JSONL round-trip
- ``test_coverage_*``       -- header line + note line from the synthetic fixture
- ``test_renderer_*``       -- end-to-end smoke render against the fixture
- ``test_contract_*``       -- CONTRACT shape + ack-gating refuses without ack
- ``test_aggregator_*``     -- aggregator handles full/partial/failed cells

The fixture is the same one the ``perf_tune_report_report_smoke`` MCP tool uses;
keeping the tests fixture-driven means any drift between the schema and the
fixture surfaces here first.
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.perf_tune_report.aggregator import aggregate
from tools.perf_tune_report.capture_signature import variant_key_for
from tools.perf_tune_report.coverage import summarize
from tools.perf_tune_report.fixtures._build_synthetic_atlas import build_rows
from tools.perf_tune_report.perf_tune_report_cli import CONTRACT, main
from tools.perf_tune_report.runners.aiperf_bench import (
    normalize_outputs as normalize_outputs_aiperf,
)
from tools.perf_tune_report.runners.common import (
    CellConfig,
    write_backend_file,
    write_normalized_json,
    write_status_file,
)
from tools.perf_tune_report.runners.vllm_sweep import (
    normalize_outputs as normalize_outputs_vllm_sweep,
)
from tools.perf_tune_report.schema import (
    STATUS_EVICTED,
    STATUS_FAILED,
    STATUS_FULL,
    STATUS_PARTIAL,
    AtlasCell,
    read_jsonl,
    write_jsonl,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_atlas.jsonl"


# ---------------------------------------------------------------------------
# schema.py
# ---------------------------------------------------------------------------

def test_schema_rejects_unknown_status():
    with pytest.raises(ValueError, match="status must be one of"):
        AtlasCell(
            cell_id="x", model="m", hardware="H100", quant="FP8",
            tensor_parallel=8, parallel_strategy="EP", mtp=False,
            max_num_batched_tokens=1024, concurrency=1, status="bogus",
        )


def test_schema_rejects_unknown_parallel_strategy():
    with pytest.raises(ValueError, match="parallel_strategy"):
        AtlasCell(
            cell_id="x", model="m", hardware="H100", quant="FP8",
            tensor_parallel=8, parallel_strategy="PP", mtp=False,
            max_num_batched_tokens=1024, concurrency=1, status=STATUS_FULL,
        )


def test_schema_has_metrics_property():
    cell = AtlasCell(
        cell_id="x", model="m", hardware="H100", quant="FP8",
        tensor_parallel=8, parallel_strategy="EP", mtp=False,
        max_num_batched_tokens=1024, concurrency=1, status=STATUS_FULL,
        ttft_avg_ms=100.0, request_throughput_avg=0.1,
    )
    assert cell.has_metrics is True
    cell_no_metrics = AtlasCell(
        cell_id="x", model="m", hardware="H100", quant="FP8",
        tensor_parallel=8, parallel_strategy="EP", mtp=False,
        max_num_batched_tokens=1024, concurrency=0, status=STATUS_FAILED,
    )
    assert cell_no_metrics.has_metrics is False


def test_schema_jsonl_roundtrip(tmp_path):
    rows_in = build_rows()
    p = tmp_path / "atlas.jsonl"
    write_jsonl(rows_in, p)
    rows_out = read_jsonl(p)
    assert len(rows_in) == len(rows_out)
    assert rows_in[0].to_dict() == rows_out[0].to_dict()


def test_schema_jsonl_skips_blank_and_comment_lines():
    fragment = (
        "# comment line ignored\n"
        "\n"
        '{"cell_id":"x","model":"m","hardware":"H100","quant":"FP8","tensor_parallel":8,'
        '"parallel_strategy":"EP","mtp":false,"max_num_batched_tokens":1024,'
        '"concurrency":1,"status":"full","ttft_avg_ms":1.0,'
        '"request_throughput_avg":0.1,"output_tps_per_user":1.0,"output_tps_per_gpu":1.0,'
        '"backend":"vllm-sweep","raw_path":"","captured_at":"","notes":"","extra":{}}\n'
    )
    rows = read_jsonl(io.StringIO(fragment))
    assert len(rows) == 1
    assert rows[0].cell_id == "x"


# ---------------------------------------------------------------------------
# coverage.py
# ---------------------------------------------------------------------------

def test_coverage_matches_pdf_header():
    rows = read_jsonl(FIXTURE_PATH)
    summary = summarize(rows)
    assert summary.atlas_cells == 40
    assert summary.full_sweeps == 38
    assert summary.partial_sweeps == 1
    assert summary.failed_cells == 1
    assert summary.plot_ready_points == 232
    assert summary.evicted_cells == 20


def test_coverage_header_line_format():
    rows = read_jsonl(FIXTURE_PATH)
    summary = summarize(rows)
    assert summary.header_line() == (
        "Coverage: 40 atlas cells | 38 full sweeps | 1 partial sweeps | "
        "1 failed cells | 232 plot-ready concurrency points"
    )


def test_coverage_note_line_omitted_when_no_evicted():
    rows = [
        AtlasCell(
            cell_id="x", model="m", hardware="H100", quant="FP8",
            tensor_parallel=8, parallel_strategy="EP", mtp=False,
            max_num_batched_tokens=1024, concurrency=1, status=STATUS_FULL,
            ttft_avg_ms=1.0, request_throughput_avg=0.1,
            output_tps_per_user=1.0, output_tps_per_gpu=1.0,
        ),
    ]
    summary = summarize(rows)
    assert summary.note_line() is None


def _full_cell(cell_id: str, *, plottable: bool) -> AtlasCell:
    return AtlasCell(
        cell_id=cell_id, model="m", hardware="B200", quant="NVFP4",
        tensor_parallel=8, parallel_strategy="TP", mtp=False,
        max_num_batched_tokens=1024, concurrency=1, status=STATUS_FULL,
        ttft_avg_ms=1.0 if plottable else None,
        request_throughput_avg=0.1 if plottable else None,
        output_tps_per_user=1.0, output_tps_per_gpu=1.0,
    )


def test_coverage_counts_full_but_unplottable():
    rows = [_full_cell("ok", plottable=True), _full_cell("bad", plottable=False)]
    summary = summarize(rows)
    assert summary.full_sweeps == 2
    assert summary.plot_ready_points == 1
    assert summary.non_plot_ready_full_cells == 1
    assert "1 full-but-unplottable cells" in summary.header_line()


# ---------------------------------------------------------------------------
# Renderer: loud completeness (no silent skips / blank charts)
# ---------------------------------------------------------------------------


def _write_atlas(campaign_dir: Path, rows: list[AtlasCell]) -> Path:
    campaign_dir.mkdir(parents=True, exist_ok=True)
    atlas_path = campaign_dir / "atlas.jsonl"
    with atlas_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")
    return atlas_path


def test_renderer_writes_report_status_and_completeness_page(tmp_path):
    pytest.importorskip("matplotlib")
    from tools.perf_tune_report.renderer.render_report import render_report

    campaign = tmp_path / "campaign"
    atlas = _write_atlas(campaign, [_full_cell("c1", plottable=True)])
    out_pdf = tmp_path / "out.pdf"
    status = render_report(atlas, out_pdf, title="completeness test")

    # No kernels/ncu/dcgm -> SoL family omitted, not silently dropped.
    assert status.sol_complete is False
    omitted = {o["page"] for o in status.omitted_pages}
    assert any("Speed-of-Light" in p for p in omitted)
    # Every omission carries a non-empty why + how-to-fix.
    for o in status.omitted_pages:
        assert o["why"] and o["how_to_fix"]

    status_json = json.loads((campaign / "report_status.json").read_text())
    assert status_json["sol_complete"] is False
    assert status_json["omitted_pages"]


def test_renderer_zero_plot_ready_records_empty_scatter(tmp_path):
    pytest.importorskip("matplotlib")
    from tools.perf_tune_report.renderer.render_report import render_report

    campaign = tmp_path / "campaign"
    atlas = _write_atlas(campaign, [_full_cell("c1", plottable=False)])
    out_pdf = tmp_path / "out.pdf"
    status = render_report(atlas, out_pdf, title="empty scatter test")

    assert status.plot_ready_points == 0
    omitted = {o["page"] for o in status.omitted_pages}
    assert any("page 1" in p for p in omitted)
    assert out_pdf.is_file()


# ---------------------------------------------------------------------------
# Renderer: UTC provenance stamp (run-id + rendered-UTC + bench-window)
# ---------------------------------------------------------------------------


def _row_with_capture(captured_at: str) -> AtlasCell:
    return AtlasCell(
        cell_id="ctrl-nomtp", model="GLM-5.1-NVFP4", hardware="B200", quant="NVFP4",
        tensor_parallel=8, parallel_strategy="EP", mtp=False,
        max_num_batched_tokens=512, concurrency=1, status=STATUS_FULL,
        ttft_avg_ms=1.0, request_throughput_avg=0.1,
        output_tps_per_user=1.0, output_tps_per_gpu=1.0,
        captured_at=captured_at,
    )


def test_pdf_provenance_embeds_utc_runid_and_window():
    from datetime import datetime, timezone

    from tools.perf_tune_report.renderer.render_report import build_pdf_provenance

    # A local-clock instant a calendar day BEHIND the UTC instant (22:41 PDT on
    # 05-30 == 05:41Z on 05-31) is exactly the off-by-a-day confusion this
    # provenance stamp removes -- the stamp is computed in UTC.
    rendered_at = datetime(2026, 5, 31, 5, 41, 0, tzinfo=timezone.utc)
    rid = "20260531T081514Z-glm51-deepep-shapes-5k1k"
    prov = build_pdf_provenance(rid, [_row_with_capture("2026-05-31T05:43:09Z")], rendered_at)

    assert prov["rendered_iso"] == "2026-05-31T05:41:00Z"
    assert prov["bench_window"] == "2026-05-31T05:43:09Z"
    # CreationDate is tz-aware UTC (offset 0), not naive/local wall-clock.
    cd = prov["infodict"]["CreationDate"]
    assert cd.tzinfo is not None and cd.utcoffset().total_seconds() == 0
    # Footer + Keywords carry the run-id and flag mtime as non-authoritative.
    assert rid in prov["footer"]
    assert "not authoritative" in prov["footer"]
    assert f"campaign={rid}" in prov["infodict"]["Keywords"]
    assert "rendered_utc=2026-05-31T05:41:00Z" in prov["infodict"]["Keywords"]


def test_pdf_provenance_brackets_multiple_capture_windows():
    from datetime import datetime, timezone

    from tools.perf_tune_report.renderer.render_report import build_pdf_provenance

    rows = [
        _row_with_capture("2026-05-31T05:43:09Z"),
        _row_with_capture("2026-05-31T08:12:00Z"),
        _row_with_capture("2026-05-31T05:43:09Z"),  # duplicate collapses
    ]
    prov = build_pdf_provenance("rid", rows, datetime.now(timezone.utc))
    assert prov["bench_window"] == "2026-05-31T05:43:09Z .. 2026-05-31T08:12:00Z"


def test_pdf_provenance_handles_missing_capture():
    from datetime import datetime, timezone

    from tools.perf_tune_report.renderer.render_report import build_pdf_provenance

    prov = build_pdf_provenance("rid", [_row_with_capture("")], datetime.now(timezone.utc))
    assert prov["bench_window"] == "unknown"


# ---------------------------------------------------------------------------
# CONTRACT shape
# ---------------------------------------------------------------------------

def test_contract_verb_set():
    """The perf_tune_report CONTRACT verb set (champion_select + import_variant_ab
    joined in v1.66.0)."""
    assert set(CONTRACT) == {
        "campaign_init", "cell_run", "atlas_aggregate",
        "report_render", "report_smoke", "publish_to_lake",
        "import_perf_bench", "campaign_run",
        "kernel_profile", "graph_diff",
        "raw_bench_compare", "import_ncu", "import_nsys", "dcgm_correlate",
        "experiments_index", "tpm_summary", "value_view",
        "import_roofline_sweep", "fleet_leaderboard",
        "champion_select", "import_variant_ab",
        "capture_plan", "materialize_capture_reuse",
        "experiment_inventory", "portability_view", "import_model_eval", "trend_view",
        "kernel_reproducer_scaffold", "import_workloads",
    }


def test_contract_import_ncu_writes_artifacts():
    """import_ncu reads an ncu bundle, writes ncu_kernels.json; no ack."""
    entry = CONTRACT["import_ncu"]
    assert entry["safety"] == "writes_artifacts"
    assert entry["ack"] is None
    for flag in ("--campaign", "--cell-id", "--bundle"):
        assert flag in entry["required"]


def test_contract_kernel_profile_is_ack_gated():
    """kernel_profile attaches an ephemeral container; safety=submits_jobs."""
    entry = CONTRACT["kernel_profile"]
    assert entry["safety"] == "submits_jobs"
    assert entry["ack"] == "--i-understand-this-submits-jobs"
    for flag in ("--namespace", "--pod", "--target-container", "--output-dir"):
        assert flag in entry["required"]


def test_contract_graph_diff_writes_artifacts():
    """graph_diff parses local logs; safety=writes_artifacts, no ack."""
    entry = CONTRACT["graph_diff"]
    assert entry["safety"] == "writes_artifacts"
    assert entry["ack"] is None
    for flag in ("--side-a-log", "--side-b-log", "--output-dir"):
        assert flag in entry["required"]


def test_contract_import_perf_bench_writes_artifacts():
    """Importer is not ack-gated (reads bundle, writes campaign cell)."""
    assert CONTRACT["import_perf_bench"]["safety"] == "writes_artifacts"
    assert CONTRACT["import_perf_bench"]["ack"] is None
    assert "--campaign" in CONTRACT["import_perf_bench"]["required"]
    assert "--bundle" in CONTRACT["import_perf_bench"]["required"]


def test_contract_capture_plan_writes_artifacts():
    """capture_plan can persist a plan via --out, so it is not read-only."""
    assert CONTRACT["capture_plan"]["safety"] == "writes_artifacts"
    assert CONTRACT["capture_plan"]["ack"] is None
    assert "--out" in CONTRACT["capture_plan"]["optional"]


def test_contract_cell_run_is_ack_gated():
    entry = CONTRACT["cell_run"]
    assert entry["safety"] == "submits_jobs"
    assert entry["ack"] == "--i-understand-this-submits-jobs"


def test_contract_report_smoke_is_read_only():
    assert CONTRACT["report_smoke"]["safety"] == "read_only"
    assert CONTRACT["report_smoke"]["ack"] is None


def test_contract_writes_artifacts_verbs():
    for verb in (
        "campaign_init", "atlas_aggregate", "report_render", "publish_to_lake",
        "import_perf_bench", "graph_diff",
    ):
        assert CONTRACT[verb]["safety"] == "writes_artifacts"
        assert CONTRACT[verb]["ack"] is None


# ---------------------------------------------------------------------------
# CLI ack-gating
# ---------------------------------------------------------------------------

def _make_campaign(tmp_path: Path, monkeypatch) -> Path:
    import yaml
    monkeypatch.setenv("PERFREPORT_CAMPAIGNS_DIR", str(tmp_path))
    cfg = {
        "name": "test",
        "cells": [{
            "cell_id": "cell1",
            "model": "X", "hardware": "H100", "quant": "FP8",
            "tensor_parallel": 8, "parallel_strategy": "EP", "mtp": False,
            "max_num_batched_tokens": 1024, "concurrencies": [1, 8],
            "vllm_sweep": {"serve_cmd": "vllm serve X", "bench_cmd": "vllm bench serve --model X"},
        }],
    }
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    rc = main(["campaign_init", "--config", str(cfg_path), "--slug", "test", "--json"])
    assert rc == 0
    return next(tmp_path.glob("*-test"))


def test_cli_cell_run_refuses_without_ack(tmp_path, monkeypatch, capsys):
    _make_campaign(tmp_path, monkeypatch)
    rc = main([
        "cell_run", "--campaign", "test", "--cell", "cell1",
        "--backend", "vllm-sweep", "--json",
    ])
    captured = capsys.readouterr()
    assert rc == 2
    assert "ack-gated" in captured.err


def test_cli_cell_run_dry_run_does_not_require_ack(tmp_path, monkeypatch):
    _make_campaign(tmp_path, monkeypatch)
    rc = main([
        "cell_run", "--campaign", "test", "--cell", "cell1",
        "--backend", "vllm-sweep", "--dry-run", "--json",
    ])
    assert rc == 0


def test_cli_report_smoke_renders_pdf(tmp_path, monkeypatch):
    out = tmp_path / "smoke.pdf"
    rc = main(["report_smoke", "--out", str(out), "--json"])
    assert rc == 0
    assert out.is_file()
    # matplotlib PdfPages -> 2-page PDF
    header = out.read_bytes()[:8]
    assert header.startswith(b"%PDF-")


def test_campaign_init_copies_provenance_from_bundle(tmp_path, monkeypatch):
    """campaign_init lifts the evidence bundle's ```provenance``` block into the
    campaign: provenance.json sidecar + config.yaml provenance: + SOURCE.md
    source bullets (so it flows to the lake's campaign_v1 source columns)."""
    import yaml
    monkeypatch.setenv("PERFREPORT_CAMPAIGNS_DIR", str(tmp_path / "campaigns"))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "SOURCE.md").write_text(
        "# SOURCE\n\nhuman prose\n\n"
        "```provenance\n"
        "schema: experiment_provenance_v1\n"
        "identity:\n"
        "  run_id: glm51-x-20260604T103817Z\n"
        "  title: NVFP4 A/B\n"
        "  status: verified\n"
        "source:\n"
        "  - repo: example/vllm\n"
        "    branch: feature/foo\n"
        "    commit: deadbeef\n"
        "    delivery: overlay\n"
        "```\n"
    )
    cfg_path = tmp_path / "x.yaml"
    cfg_path.write_text(yaml.safe_dump({"name": "x", "cells": []}))
    rc = main([
        "campaign_init", "--config", str(cfg_path),
        "--experiment-id", "glm51-x-20260604T103817Z",
        "--evidence-bundle", str(bundle), "--json",
    ])
    assert rc == 0
    camp = tmp_path / "campaigns" / "glm51-x-20260604T103817Z"
    assert (camp / "provenance.json").is_file()
    src = (camp / "SOURCE.md").read_text()
    assert "- vllm_commit: deadbeef" in src
    assert "- delivery: overlay" in src
    assert "- experiment_status: verified" in src
    assert "provenance:" in (camp / "config.yaml").read_text()


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def _seed_campaign_from_fixture(tmp_path: Path) -> Path:
    """Build a campaign dir from the bundled synthetic fixture."""
    campaign_dir = tmp_path / "seeded"
    campaign_dir.mkdir()
    (campaign_dir / "cells").mkdir()
    rows = read_jsonl(FIXTURE_PATH)
    by_cell: dict[str, list[AtlasCell]] = defaultdict(list)
    for r in rows:
        by_cell[r.cell_id].append(r)
    for cell_id, cell_rows in by_cell.items():
        cell_dir = campaign_dir / "cells" / cell_id
        write_backend_file(cell_dir, cell_rows[0].backend)
        write_normalized_json(cell_dir, cell_rows)
        write_status_file(cell_dir, cell_rows[0].status)
    return campaign_dir


def test_aggregator_round_trip_matches_fixture(tmp_path):
    campaign_dir = _seed_campaign_from_fixture(tmp_path)
    result = aggregate(campaign_dir)
    assert result.atlas_path.is_file()
    assert result.row_count == 253
    assert result.cell_count == 60
    assert result.coverage.atlas_cells == 40
    assert result.coverage.full_sweeps == 38
    assert result.coverage.partial_sweeps == 1
    assert result.coverage.failed_cells == 1
    assert result.coverage.plot_ready_points == 232
    assert result.coverage.evicted_cells == 20


def test_aggregator_raises_on_schema_drift(tmp_path):
    campaign_dir = tmp_path / "drift"
    cell_dir = campaign_dir / "cells" / "bad"
    cell_dir.mkdir(parents=True)
    (cell_dir / "normalized.json").write_text(
        json.dumps([{"cell_id": "bad", "status": "weird-status"}])
    )
    with pytest.raises(ValueError, match="schema"):
        aggregate(campaign_dir)


# ---------------------------------------------------------------------------
# Runner parser fixture tests (gap 5 -- field-name mappings against
# representative backend output rather than only the dry-run command path).
# ---------------------------------------------------------------------------

def _make_test_cell(concurrencies=(1, 8, 32)) -> CellConfig:
    return CellConfig(
        cell_id="test-cell",
        model="TestModel",
        hardware="H100",
        quant="FP8",
        tensor_parallel=8,
        parallel_strategy="EP",
        mtp=False,
        max_num_batched_tokens=1024,
        concurrencies=concurrencies,
    )


def test_runner_vllm_sweep_normalize_full_status_and_derived_metrics(tmp_path):
    """vllm bench sweep per-run JSON -> AtlasCell with correct derived metrics.

    Asserts the 4 field-name mappings the runner relies on
    (`max_concurrency`, `median_ttft_ms`, `request_throughput`,
    `output_throughput`) plus the two derived metrics
    (`output_tps_per_gpu` = output_throughput / TP and
    `output_tps_per_user` = output_throughput / concurrency).
    """
    cell = _make_test_cell()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for c in cell.concurrencies:
        (raw_dir / f"c{c}.json").write_text(json.dumps({
            "max_concurrency": c,
            "median_ttft_ms": 100.0 * c,
            "request_throughput": 0.1 * c,
            "output_throughput": 800.0 * c,
        }))

    cell_dir = tmp_path / "cells" / cell.cell_id
    cell_dir.mkdir(parents=True)
    rows, status = normalize_outputs_vllm_sweep(cell, raw_dir, cell_dir)

    assert status == STATUS_FULL
    assert len(rows) == 3
    assert {r.concurrency for r in rows} == set(cell.concurrencies)
    by_c = {r.concurrency: r for r in rows}
    # C=1 has output_throughput=800; TP=8 so tps/gpu=100, tps/user=800.
    assert by_c[1].output_tps_per_gpu == pytest.approx(100.0)
    assert by_c[1].output_tps_per_user == pytest.approx(800.0)
    assert by_c[1].ttft_avg_ms == pytest.approx(100.0)
    assert by_c[1].request_throughput_avg == pytest.approx(0.1)
    # C=32: output_throughput=25600; tps/gpu=3200, tps/user=800.
    assert by_c[32].output_tps_per_gpu == pytest.approx(3200.0)
    assert by_c[32].output_tps_per_user == pytest.approx(800.0)


def _spec_cell(k: int, *, extras_overrides: dict | None = None) -> CellConfig:
    """A cell whose extras carry the typed variant knobs (MTP-K + serving knobs)."""
    extras = {
        "num_speculative_tokens": k,
        "async_scheduling": True,
        "max_num_seqs": 256,
        "prefix_cache": True,
        "bench_backend": "vllm",
        "image": "infr/vllm:v2.12.3",
    }
    extras.update(extras_overrides or {})
    return CellConfig(
        cell_id=f"spec-K{k}",
        model="GLM-5.1-NVFP4",
        hardware="GB300",
        quant="NVFP4",
        tensor_parallel=4,
        parallel_strategy="TP",
        mtp=True,
        max_num_batched_tokens=2048,
        concurrencies=(1,),
        extras=extras,
    )


def _normalize_one(cell: CellConfig, tmp_path):
    # Unique work dir per call via a monotonic counter (the same cell_id can be normalized
    # twice, e.g. K=3 at two image tags; id() is unsafe -- it is reused after GC).
    _normalize_one.n = getattr(_normalize_one, "n", 0) + 1
    work = tmp_path / f"work-{cell.cell_id}-{_normalize_one.n}"
    raw_dir = work / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "c1.json").write_text(json.dumps({
        "max_concurrency": 1,
        "median_ttft_ms": 50.0,
        "request_throughput": 1.0,
        "output_throughput": 60.0,
    }))
    cell_dir = work / "cells" / cell.cell_id
    cell_dir.mkdir(parents=True)
    rows, _ = normalize_outputs_vllm_sweep(cell, raw_dir, cell_dir)
    return rows[0]


def test_runner_vllm_sweep_populates_typed_variant_fields(tmp_path):
    """The runner promotes MTP-K + async/max_num_seqs/prefix/backend to the typed
    AtlasCell fields (not just notes/serve-config) so variant_key can distinguish them."""
    row = _normalize_one(_spec_cell(3), tmp_path)
    assert row.num_speculative_tokens == 3
    assert row.async_scheduling is True
    assert row.max_num_seqs == 256
    assert row.enable_prefix_caching is True
    assert row.bench_backend == "vllm"


def test_runner_vllm_sweep_variant_key_distinguishes_mtp_k(tmp_path):
    """Two cells differing only in num_speculative_tokens (K=2 vs K=3) produce
    different image-independent variant_key hashes -- the material A1 outcome."""
    row_k2 = _normalize_one(_spec_cell(2), tmp_path)
    row_k3 = _normalize_one(_spec_cell(3), tmp_path)
    assert variant_key_for(row_k2) != variant_key_for(row_k3)
    # And it is image-INDEPENDENT: same K + same knobs, different image == same key.
    row_k3_newimg = _normalize_one(
        _spec_cell(3, extras_overrides={"image": "infr/vllm:v9.9.9"}), tmp_path
    )
    assert variant_key_for(row_k3) == variant_key_for(row_k3_newimg)


def test_runner_aiperf_normalize_extracts_nested_metrics(tmp_path):
    """AIPerf reports vary by version -- exercise all 3 candidate
    nested locations (top level, `results`, `summary`) plus the
    secondary field aliases (`mean_ttft_ms`, `output_throughput_tps`,
    `output_tps`, `request_throughput_avg`).
    """
    cell = _make_test_cell(concurrencies=(1, 8, 32))
    raw_dir = tmp_path / "raw"

    # c1: metrics at the TOP level using primary field names.
    (raw_dir / "c1").mkdir(parents=True)
    (raw_dir / "c1" / "profile_export_aiperf.json").write_text(json.dumps({
        "median_ttft_ms": 50.0,
        "request_throughput": 0.5,
        "output_throughput": 1000.0,
    }))
    # c8: metrics nested under `results` using SECONDARY field aliases.
    (raw_dir / "c8").mkdir(parents=True)
    (raw_dir / "c8" / "profile_export_aiperf.json").write_text(json.dumps({
        "results": {
            "mean_ttft_ms": 150.0,
            "request_throughput_avg": 1.5,
            "output_throughput_tps": 8000.0,
        }
    }))
    # c32: metrics nested under `summary` using TERTIARY field aliases.
    (raw_dir / "c32").mkdir(parents=True)
    (raw_dir / "c32" / "profile_export_aiperf.json").write_text(json.dumps({
        "summary": {
            "ttft_avg_ms": 300.0,
            "request_throughput": 3.0,
            "output_tps": 24000.0,
        }
    }))

    cell_dir = tmp_path / "cells" / cell.cell_id
    cell_dir.mkdir(parents=True)
    rows, status = normalize_outputs_aiperf(cell, raw_dir, cell_dir)

    assert status == STATUS_FULL
    assert len(rows) == 3
    by_c = {r.concurrency: r for r in rows}
    assert by_c[1].ttft_avg_ms == pytest.approx(50.0)
    assert by_c[1].output_tps_per_gpu == pytest.approx(1000.0 / 8)  # TP=8
    assert by_c[8].ttft_avg_ms == pytest.approx(150.0)
    assert by_c[8].output_tps_per_user == pytest.approx(8000.0 / 8)
    assert by_c[32].ttft_avg_ms == pytest.approx(300.0)
    assert by_c[32].request_throughput_avg == pytest.approx(3.0)


def test_runner_normalize_no_data_yields_failed(tmp_path):
    """Both runners must mark a cell `failed` when no parseable data exists."""
    cell = _make_test_cell(concurrencies=(1, 8))
    cell_dir = tmp_path / "cells" / cell.cell_id
    cell_dir.mkdir(parents=True)

    # vllm-sweep: empty raw_dir.
    empty_raw = tmp_path / "empty-vllm-raw"
    empty_raw.mkdir()
    rows_vllm, status_vllm = normalize_outputs_vllm_sweep(cell, empty_raw, cell_dir)
    assert status_vllm == STATUS_FAILED
    assert rows_vllm == []

    # aiperf: raw_dir with a JSON file missing all metric fields.
    aiperf_raw = tmp_path / "aiperf-raw"
    (aiperf_raw / "c1").mkdir(parents=True)
    (aiperf_raw / "c1" / "profile_export_aiperf.json").write_text(
        json.dumps({"unrelated": "data"})
    )
    rows_aiperf, status_aiperf = normalize_outputs_aiperf(cell, aiperf_raw, cell_dir)
    assert status_aiperf == STATUS_FAILED
    assert rows_aiperf == []
