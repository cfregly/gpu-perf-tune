"""Tests for tools.perf_tune_report.importers.lws_summary (v1.23.1).

Covers the GLM-LWS multi-variant ``summary.json`` -> ``atlas.jsonl``
direct-emission path. Sibling to ``test_inference_drive_load.py`` and
``test_inference_perf_bench.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.perf_tune_report.importers import (
    detect_lws_summary,
    import_bundle_auto,
    import_lws_summary_bundle,
)


def _make_summary(tmp_path: Path, *, variants: list[dict] | None = None) -> Path:
    """Build a minimal summary.json under tmp_path. Returns tmp_path."""
    if variants is None:
        variants = [
            {
                "short": "LWS-baseline",
                "label": "LWS-baseline (mns=40, mbt=32768, kv=fp8)",
                "knob": "mns=40, mbt=32768, kv=fp8",
                "metrics_per_concurrency": [
                    {
                        "path": str(tmp_path / "raw" / "sweep-c1.txt"),
                        "concurrency": 1,
                        "duration_s": 30.5,
                        "req_per_s": 0.14,
                        "output_tps": 71.43,
                        "ttft_median_ms": 198.58,
                        "tpot_median_ms": 13.64,
                    },
                    {
                        "path": str(tmp_path / "raw" / "sweep-c2.txt"),
                        "concurrency": 2,
                        "duration_s": 32.0,
                        "req_per_s": 0.29,
                        "output_tps": 149.95,
                        "ttft_median_ms": 126.28,
                        "tpot_median_ms": 12.98,
                    },
                ],
            },
            {
                "short": "champ-A",
                "label": "champ-A (mns=64)",
                "knob": "mns=64, mbt=12288, kv=fp8",
                "metrics_per_concurrency": [
                    {
                        "path": str(tmp_path / "raw" / "sweep-A-c1.txt"),
                        "concurrency": 1,
                        "duration_s": 30.0,
                        "req_per_s": 0.20,
                        "output_tps": 95.0,
                        "ttft_median_ms": 150.0,
                        "tpot_median_ms": 10.5,
                    },
                ],
            },
        ]
    summary = {"variants": variants}
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    return tmp_path


# ---------------------------------------------------------------------------
# detect_lws_summary
# ---------------------------------------------------------------------------


def test_detect_lws_summary_happy_path(tmp_path: Path):
    _make_summary(tmp_path)
    assert detect_lws_summary(tmp_path) is True


def test_detect_lws_summary_no_summary_json(tmp_path: Path):
    assert detect_lws_summary(tmp_path) is False


def test_detect_lws_summary_malformed_json(tmp_path: Path):
    (tmp_path / "summary.json").write_text("{not valid json")
    assert detect_lws_summary(tmp_path) is False


def test_detect_lws_summary_empty_variants(tmp_path: Path):
    (tmp_path / "summary.json").write_text(json.dumps({"variants": []}))
    assert detect_lws_summary(tmp_path) is False


def test_detect_lws_summary_variant_missing_metrics_per_concurrency(tmp_path: Path):
    (tmp_path / "summary.json").write_text(
        json.dumps({"variants": [{"short": "x", "label": "x"}]})
    )
    assert detect_lws_summary(tmp_path) is False


# ---------------------------------------------------------------------------
# import_lws_summary_bundle
# ---------------------------------------------------------------------------


def test_import_lws_summary_writes_atlas_jsonl(tmp_path: Path):
    _make_summary(tmp_path)
    result = import_lws_summary_bundle(bundle=tmp_path, campaign_dir=tmp_path)
    assert result.row_count == 3  # 2 from LWS-baseline + 1 from champ-A
    assert result.variant_count == 2
    assert result.concurrencies == [1, 2]
    assert result.atlas_path.is_file()
    rows = [json.loads(ln) for ln in result.atlas_path.read_text().splitlines() if ln.strip()]
    assert len(rows) == 3
    cell_ids = sorted({r["cell_id"] for r in rows})
    assert cell_ids == ["LWS-baseline", "champ-A"]


def test_import_lws_summary_dry_run_does_not_write(tmp_path: Path):
    _make_summary(tmp_path)
    result = import_lws_summary_bundle(bundle=tmp_path, campaign_dir=tmp_path, dry_run=True)
    assert result.row_count == 3
    assert not result.atlas_path.is_file()


def test_import_lws_summary_overrides_applied(tmp_path: Path):
    _make_summary(tmp_path)
    result = import_lws_summary_bundle(
        bundle=tmp_path,
        campaign_dir=tmp_path,
        overrides={
            "model": "moonshotai/Kimi-K2.6",
            "hardware": "GB300",
            "tensor_parallel": 4,
            "captured_at": "2026-05-28T00:00:00Z",
        },
    )
    rows = [json.loads(ln) for ln in result.atlas_path.read_text().splitlines() if ln.strip()]
    assert all(r["model"] == "moonshotai/Kimi-K2.6" for r in rows)
    assert all(r["hardware"] == "GB300" for r in rows)
    assert all(r["tensor_parallel"] == 4 for r in rows)
    assert all(r["captured_at"] == "2026-05-28T00:00:00Z" for r in rows)


def test_import_lws_summary_output_tps_per_user_derived_from_tpot(tmp_path: Path):
    _make_summary(tmp_path)
    result = import_lws_summary_bundle(bundle=tmp_path, campaign_dir=tmp_path)
    rows = [json.loads(ln) for ln in result.atlas_path.read_text().splitlines() if ln.strip()]
    # tpot_median_ms=13.64 -> output_tps_per_user = 1000 / 13.64 ~= 73.31
    baseline_c1 = next(r for r in rows if r["cell_id"] == "LWS-baseline" and r["concurrency"] == 1)
    assert abs(baseline_c1["output_tps_per_user"] - (1000.0 / 13.64)) < 1e-6


def test_import_lws_summary_max_num_batched_tokens_parsed_from_knob(tmp_path: Path):
    _make_summary(tmp_path)
    result = import_lws_summary_bundle(bundle=tmp_path, campaign_dir=tmp_path)
    rows = [json.loads(ln) for ln in result.atlas_path.read_text().splitlines() if ln.strip()]
    assert any(r["max_num_batched_tokens"] == 32768 for r in rows)
    assert any(r["max_num_batched_tokens"] == 12288 for r in rows)


def test_import_lws_summary_raises_when_summary_missing(tmp_path: Path):
    with pytest.raises(ValueError, match="summary.json not found"):
        import_lws_summary_bundle(bundle=tmp_path, campaign_dir=tmp_path)


def test_import_lws_summary_raises_when_no_rows_emitted(tmp_path: Path):
    (tmp_path / "summary.json").write_text(
        json.dumps({"variants": [{"short": "x", "metrics_per_concurrency": []}]})
    )
    with pytest.raises(ValueError, match="no rows emitted"):
        import_lws_summary_bundle(bundle=tmp_path, campaign_dir=tmp_path)


def test_import_lws_summary_raises_on_malformed_json(tmp_path: Path):
    (tmp_path / "summary.json").write_text("{not valid")
    with pytest.raises(ValueError, match="malformed"):
        import_lws_summary_bundle(bundle=tmp_path, campaign_dir=tmp_path)


# ---------------------------------------------------------------------------
# import_bundle_auto dispatch
# ---------------------------------------------------------------------------


def test_import_bundle_auto_dispatches_to_lws_summary(tmp_path: Path):
    _make_summary(tmp_path)
    result = import_bundle_auto(bundle=tmp_path, campaign_dir=tmp_path)
    assert result.importer == "lws_summary"
    assert result.row_count == 3
