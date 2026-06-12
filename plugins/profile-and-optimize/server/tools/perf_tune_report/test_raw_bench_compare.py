"""Tests for tools.perf_tune_report.raw_bench_compare (v1.24.0).

Promoted from the GLM-LWS workshop renderers
(./campaigns workspacescripts/render_lws_baseline_report.py +
render_phase_a_report.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml as _yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.perf_tune_report.raw_bench_compare import (
    BundleSpec,
    RawBenchCompareManifestMalformed,
    _parse_sweep_file,
    load_manifest,
    render_comparison,
)


_SWEEP_TEXT_TEMPLATE = """
============ Serving Benchmark Result ============
Successful requests:                     {successful}
Maximum request concurrency:             {c}
Benchmark duration (s):                  {duration_s}
Total input tokens:                      102400
Total generated tokens:                  41384
Request throughput (req/s):              {req_per_s}
Output token throughput (tok/s):         {output_tps}
Total token throughput (tok/s):          {total_tps}
---------------Time to First Token----------------
Mean TTFT (ms):                          {ttft_mean}
Median TTFT (ms):                        {ttft_median}
P99 TTFT (ms):                           {ttft_p99}
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          {tpot_mean}
Median TPOT (ms):                        {tpot_median}
P99 TPOT (ms):                           {tpot_p99}
==================================================
"""


def _write_sweep(bundle_dir: Path, c: int, *, output_tps: float, tpot_median: float, ttft_median: float = 200.0) -> Path:
    """Write a synthetic sweep-c<N>.txt with the vllm bench serve format."""
    raw = bundle_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / f"sweep-c{c}.txt"
    path.write_text(_SWEEP_TEXT_TEMPLATE.format(
        c=c,
        successful=c * 32,
        duration_s=300.0,
        req_per_s=output_tps / 100.0,
        output_tps=output_tps,
        total_tps=output_tps * 1.5,
        ttft_mean=ttft_median * 1.2,
        ttft_median=ttft_median,
        ttft_p99=ttft_median * 1.8,
        tpot_mean=tpot_median * 1.05,
        tpot_median=tpot_median,
        tpot_p99=tpot_median * 1.3,
    ))
    return path


def _build_2_bundle_layout(tmp_path: Path) -> Path:
    """Create bundles_root with 2 bundles (baseline + champion). Returns root."""
    root = tmp_path / "bundles"
    root.mkdir()
    baseline = root / "model-baseline-20260527T000000Z"
    baseline.mkdir()
    _write_sweep(baseline, 1, output_tps=100.0, tpot_median=20.0)
    _write_sweep(baseline, 2, output_tps=180.0, tpot_median=22.0)
    _write_sweep(baseline, 4, output_tps=320.0, tpot_median=24.0)

    champion = root / "model-champion-20260527T010000Z"
    champion.mkdir()
    _write_sweep(champion, 1, output_tps=120.0, tpot_median=18.0)
    _write_sweep(champion, 2, output_tps=220.0, tpot_median=19.0)
    _write_sweep(champion, 4, output_tps=400.0, tpot_median=20.0)
    return root


def _write_manifest(tmp_path: Path, bundles_root: Path, *, extra_bundles: list[dict] | None = None) -> Path:
    """Write a minimal manifest pointing at the 2-bundle layout."""
    bundles = [
        {
            "glob": "model-baseline-*",
            "label": "Baseline",
            "short": "baseline",
            "knob": "default",
            "color": "#444444",
            "marker": "s",
            "is_baseline": True,
        },
        {
            "glob": "model-champion-*",
            "label": "Champion",
            "short": "champion",
            "knob": "tuned",
            "color": "#1f77b4",
            "marker": "o",
        },
    ]
    if extra_bundles:
        bundles.extend(extra_bundles)
    manifest = {
        "schema_version": 1,
        "campaign_name": "test campaign",
        "hardware": "B200",
        "model": "test-model",
        "bundles_root": str(bundles_root),
        "bundles": bundles,
    }
    p = tmp_path / "manifest.yaml"
    p.write_text(_yaml.dump(manifest))
    return p


# ---------------------------------------------------------------------------
# _parse_sweep_file
# ---------------------------------------------------------------------------


def test_parse_sweep_file_extracts_metrics(tmp_path: Path):
    sweep = _write_sweep(tmp_path, 16, output_tps=1500.0, tpot_median=12.5)
    row = _parse_sweep_file(sweep)
    assert row is not None
    assert row["c"] == 16
    assert row["output_tps"] == pytest.approx(1500.0)
    assert row["tpot_med_ms"] == pytest.approx(12.5)
    # Workshop-friendly aliases
    assert row["ttft_median_ms"] == row["ttft_med_ms"]
    assert row["tpot_median_ms"] == row["tpot_med_ms"]


def test_parse_sweep_file_returns_none_on_wrong_filename(tmp_path: Path):
    f = tmp_path / "not-a-sweep.txt"
    f.write_text("")
    assert _parse_sweep_file(f) is None


def test_parse_sweep_file_returns_none_on_no_metrics(tmp_path: Path):
    f = tmp_path / "sweep-c4.txt"
    f.write_text("crash; no Output token throughput here")
    assert _parse_sweep_file(f) is None


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------


def test_load_manifest_happy_path(tmp_path: Path):
    bundles_root = _build_2_bundle_layout(tmp_path)
    manifest = _write_manifest(tmp_path, bundles_root)
    meta, specs = load_manifest(manifest)
    assert meta["campaign_name"] == "test campaign"
    assert len(specs) == 2
    assert specs[0].is_baseline
    assert specs[0].bundle_path is not None
    assert len(specs[0].rows) == 3
    assert specs[1].label == "Champion"


def test_load_manifest_raises_on_missing_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "missing.yaml")


def test_load_manifest_raises_on_malformed_yaml(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("{not: [valid")
    with pytest.raises(RawBenchCompareManifestMalformed, match="YAML parse error"):
        load_manifest(bad)


def test_load_manifest_raises_on_missing_bundles_key(tmp_path: Path):
    p = tmp_path / "no_bundles.yaml"
    p.write_text(_yaml.dump({"schema_version": 1, "campaign_name": "x"}))
    with pytest.raises(RawBenchCompareManifestMalformed, match="bundles\\[\\] missing or empty"):
        load_manifest(p)


def test_load_manifest_raises_on_missing_required_bundle_field(tmp_path: Path):
    p = tmp_path / "incomplete.yaml"
    p.write_text(_yaml.dump({
        "schema_version": 1,
        "bundles": [{"glob": "x", "label": "y"}],  # missing 'short'
    }))
    with pytest.raises(RawBenchCompareManifestMalformed, match="missing 'short'"):
        load_manifest(p)


def test_load_manifest_resolves_bundles_root_relative_to_manifest(tmp_path: Path):
    bundles_root = _build_2_bundle_layout(tmp_path)
    # Write manifest at sibling location with bundles_root as relative path.
    relative_root = bundles_root.relative_to(tmp_path)
    p = tmp_path / "relative_manifest.yaml"
    p.write_text(_yaml.dump({
        "schema_version": 1,
        "bundles_root": str(relative_root),  # relative
        "bundles": [
            {"glob": "model-baseline-*", "label": "B", "short": "b"},
            {"glob": "model-champion-*", "label": "C", "short": "c"},
        ],
    }))
    _, specs = load_manifest(p)
    assert all(s.bundle_path is not None for s in specs)


def test_load_manifest_handles_missing_bundle_gracefully(tmp_path: Path):
    """Glob that matches no bundles -> spec.bundle_path stays None."""
    bundles_root = _build_2_bundle_layout(tmp_path)
    extra = [{"glob": "nonexistent-*", "label": "missing", "short": "missing"}]
    manifest = _write_manifest(tmp_path, bundles_root, extra_bundles=extra)
    _, specs = load_manifest(manifest)
    assert specs[2].bundle_path is None
    assert specs[2].rows == []


# ---------------------------------------------------------------------------
# render_comparison end-to-end
# ---------------------------------------------------------------------------


def _count_pdf_pages(path: Path) -> int:
    """Cheap PDF page counter (avoid pypdf dependency in tests)."""
    raw = path.read_bytes()
    return raw.count(b"/Type /Page\n") + raw.count(b"/Type /Page ")


def test_render_comparison_emits_pdf_with_expected_pages(tmp_path: Path):
    bundles_root = _build_2_bundle_layout(tmp_path)
    manifest = _write_manifest(tmp_path, bundles_root)
    out_pdf = tmp_path / "out.pdf"
    result = render_comparison(manifest, out_pdf)
    assert out_pdf.is_file()
    assert result.n_bundles == 2
    assert result.n_bundles_with_data == 2
    assert result.n_rows_total == 6
    assert result.baseline_short == "baseline"
    assert result.baseline_peak_tps == pytest.approx(320.0)
    # cover + throughput + ttft + tpot + peaks + summary = 6 pages
    assert _count_pdf_pages(out_pdf) >= 6


def test_render_comparison_peaks_carry_pct_vs_baseline(tmp_path: Path):
    bundles_root = _build_2_bundle_layout(tmp_path)
    manifest = _write_manifest(tmp_path, bundles_root)
    out_pdf = tmp_path / "out.pdf"
    result = render_comparison(manifest, out_pdf)
    champion = next(p for p in result.peaks if p["short"] == "champion")
    # champion peak is 400; baseline 320; (400/320 - 1) * 100 = 25%
    assert champion["pct_vs_baseline"] == pytest.approx(25.0)
    baseline = next(p for p in result.peaks if p["short"] == "baseline")
    assert baseline["pct_vs_baseline"] == pytest.approx(0.0)


def test_render_comparison_raises_when_no_bundles_have_data(tmp_path: Path):
    """All globs miss -> no rows -> refuse to render an empty PDF."""
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    p = tmp_path / "empty_manifest.yaml"
    p.write_text(_yaml.dump({
        "schema_version": 1,
        "bundles_root": str(empty_root),
        "bundles": [{"glob": "missing-*", "label": "x", "short": "x"}],
    }))
    with pytest.raises(ValueError, match="no bundles .* yielded parseable rows"):
        render_comparison(p, tmp_path / "out.pdf")
