"""Unit tests for the drive_load.py JSONL importer (v1.21.0)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.perf_tune_report.importers import (
    detect_bundle_pattern,
    import_bundle_auto,
)
from tools.perf_tune_report.importers.inference_drive_load import (
    DriveLoadImportResult,
    _aggregate_jsonl,
    _enumerate_jsonl_files,
    import_drive_load_bundle,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_BUNDLE_META = {
    "schema": "inference_perfbench_v1",
    "model": "moonshotai/Kimi-K2.6",
    "tensor_parallel_size": 8,
    "parallel_strategy": "TP",
    "mtp": False,
    "max_num_batched_tokens": 4096,
    "max_num_seqs": 50,
    "kv_cache_dtype": "fp8_e4m3",
    "speculative_decoding": "eagle3",
    "vllm_image": "registry.example.com/infr/vllm:v2.12.3",
}


def _make_jsonl_lines(
    n_ok: int,
    n_fail: int,
    *,
    started_base: float = 1_779_735_000.0,
    duration_per_req: float = 2.0,
    prompt_tokens: int = 1024,
    completion_tokens: int = 128,
    sweep_offset: float = 0.0,
) -> str:
    """Build n_ok OK + n_fail FAIL JSONL lines as a single string.

    The OK requests are spread across ``n_ok * duration_per_req`` seconds so
    the aggregator computes a sensible effective_duration_s.
    """
    lines: list[str] = []
    for i in range(n_ok):
        lines.append(
            json.dumps(
                {
                    "shape": "long-short",
                    "started_at": started_base + sweep_offset + (i * 0.5),
                    "duration_s": duration_per_req,
                    "http_status": 200,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "error": None,
                }
            )
        )
    for i in range(n_fail):
        lines.append(
            json.dumps(
                {
                    "shape": "long-short",
                    "started_at": started_base + sweep_offset + (i * 0.5),
                    "duration_s": 0.1,
                    "http_status": 500,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": 0,
                    "error": "EngineDeadError",
                }
            )
        )
    return "\n".join(lines) + "\n"


def _make_multi_c_bundle(tmp_path: Path, *, with_meta: bool = True) -> Path:
    """Bundle with bench-c<NNN>/raw/load.jsonl subdirs (Kimi multi-c pattern)."""
    bundle = tmp_path / "kimi-mns96-sweep-20260526T000000Z"
    bundle.mkdir()
    for c in (15, 30, 60):
        sub = bundle / f"bench-c{c:03d}" / "raw"
        sub.mkdir(parents=True)
        # ok-rate scales inversely with concurrency to mimic a stress sweep
        n_ok = {15: 60, 30: 40, 60: 30}[c]
        n_fail = {15: 0, 30: 4, 60: 6}[c]
        (sub / "load.jsonl").write_text(
            _make_jsonl_lines(n_ok, n_fail, sweep_offset=c * 100.0)
        )
    if with_meta:
        (bundle / "inference_perfbench_v1.json").write_text(json.dumps(_BUNDLE_META))
    return bundle


def _make_single_c_bundle(tmp_path: Path, *, with_meta: bool = True) -> Path:
    """Bundle with a single raw/load.jsonl (no bench-c<NNN>/ subdirs)."""
    bundle = tmp_path / "kimi-single-c-20260526T010000Z"
    raw = bundle / "raw"
    raw.mkdir(parents=True)
    (raw / "load.jsonl").write_text(_make_jsonl_lines(50, 2))
    if with_meta:
        (bundle / "inference_perfbench_v1.json").write_text(json.dumps(_BUNDLE_META))
    return bundle


def _make_campaign(tmp_path: Path) -> Path:
    campaign = tmp_path / "campaign-test"
    campaign.mkdir()
    (campaign / "cells").mkdir()
    return campaign


# ---------------------------------------------------------------------------
# 1. detect_bundle_pattern — auto-dispatch
# ---------------------------------------------------------------------------


def test_detect_pattern_multi_c_drive_load(tmp_path: Path) -> None:
    bundle = _make_multi_c_bundle(tmp_path)
    assert detect_bundle_pattern(bundle) == "inference_drive_load"


def test_detect_pattern_single_c_drive_load(tmp_path: Path) -> None:
    bundle = _make_single_c_bundle(tmp_path)
    assert detect_bundle_pattern(bundle) == "inference_drive_load"


def test_detect_pattern_perf_bench_sweep(tmp_path: Path) -> None:
    """sweep-c*.txt presence wins over load.jsonl when both exist (rare)."""
    bundle = tmp_path / "mixed"
    (bundle / "raw").mkdir(parents=True)
    (bundle / "raw" / "sweep-c15.txt").write_text("============ Serving Benchmark Result ============\n")
    assert detect_bundle_pattern(bundle) == "inference_perf_bench"


def test_detect_pattern_unknown(tmp_path: Path) -> None:
    bundle = tmp_path / "empty"
    bundle.mkdir()
    (bundle / "raw").mkdir()
    assert detect_bundle_pattern(bundle) == "unknown"


def test_detect_pattern_nonexistent(tmp_path: Path) -> None:
    assert detect_bundle_pattern(tmp_path / "does-not-exist") == "unknown"


# ---------------------------------------------------------------------------
# 2. _enumerate_jsonl_files — bundle layout discovery
# ---------------------------------------------------------------------------


def test_enumerate_multi_c_finds_all_subdirs(tmp_path: Path) -> None:
    bundle = _make_multi_c_bundle(tmp_path)
    files = _enumerate_jsonl_files(bundle)
    assert [f.concurrency for f in files] == [15, 30, 60]
    assert all(f.path.name == "load.jsonl" for f in files)


def test_enumerate_single_c_requires_override(tmp_path: Path) -> None:
    bundle = _make_single_c_bundle(tmp_path)
    with pytest.raises(ValueError, match="single-concurrency bundle"):
        _enumerate_jsonl_files(bundle)


def test_enumerate_single_c_uses_override(tmp_path: Path) -> None:
    bundle = _make_single_c_bundle(tmp_path)
    files = _enumerate_jsonl_files(bundle, concurrency_override=42)
    assert len(files) == 1
    assert files[0].concurrency == 42


# ---------------------------------------------------------------------------
# 3. _aggregate_jsonl — per-c metric extraction
# ---------------------------------------------------------------------------


def test_aggregate_clean_all_ok(tmp_path: Path) -> None:
    jsonl = tmp_path / "load.jsonl"
    jsonl.write_text(_make_jsonl_lines(n_ok=20, n_fail=0))
    metrics = _aggregate_jsonl(jsonl)
    assert metrics is not None
    assert metrics.n_total == 20
    assert metrics.n_ok == 20
    assert metrics.n_fail == 0
    assert metrics.req_per_s > 0
    assert metrics.total_input_tokens == 20 * 1024
    assert metrics.total_output_tokens == 20 * 128


def test_aggregate_partial_failures(tmp_path: Path) -> None:
    jsonl = tmp_path / "load.jsonl"
    jsonl.write_text(_make_jsonl_lines(n_ok=10, n_fail=5))
    metrics = _aggregate_jsonl(jsonl)
    assert metrics is not None
    assert metrics.n_total == 15
    assert metrics.n_ok == 10
    assert metrics.n_fail == 5
    # Only OK requests count toward throughput
    assert metrics.total_output_tokens == 10 * 128


def test_aggregate_all_failed_returns_none(tmp_path: Path) -> None:
    jsonl = tmp_path / "load.jsonl"
    jsonl.write_text(_make_jsonl_lines(n_ok=0, n_fail=10))
    assert _aggregate_jsonl(jsonl) is None


def test_aggregate_malformed_jsonl_skipped(tmp_path: Path) -> None:
    jsonl = tmp_path / "load.jsonl"
    good = _make_jsonl_lines(n_ok=5, n_fail=0).strip()
    jsonl.write_text(good + "\n{not valid json\n" + good + "\n")
    metrics = _aggregate_jsonl(jsonl)
    assert metrics is not None
    assert metrics.n_ok == 10
    # The malformed line counts as a failure
    assert metrics.n_fail == 1


def test_aggregate_ttft_none_when_not_recorded(tmp_path: Path) -> None:
    """Non-streaming runs (no ttft_s) -> ttft_median_ms is None (not derived)."""
    jsonl = tmp_path / "load.jsonl"
    jsonl.write_text(_make_jsonl_lines(n_ok=10, n_fail=0))
    metrics = _aggregate_jsonl(jsonl)
    assert metrics is not None
    assert metrics.ttft_median_ms is None


def _make_streaming_lines(n_ok: int, ttft_s: float = 0.25) -> str:
    """OK lines that carry a per-request ttft_s (drive_load.py --stream-all)."""
    lines = []
    for i in range(n_ok):
        lines.append(
            json.dumps(
                {
                    "shape": "streaming",
                    "started_at": 1_779_735_000.0 + (i * 0.5),
                    "duration_s": 2.0,
                    "http_status": 200,
                    "prompt_tokens": 512,
                    "completion_tokens": 256,
                    "error": None,
                    "ttft_s": ttft_s,
                }
            )
        )
    return "\n".join(lines) + "\n"


def test_aggregate_ttft_median_when_recorded(tmp_path: Path) -> None:
    """--stream-all runs record ttft_s -> aggregated to ttft_median_ms (ms)."""
    jsonl = tmp_path / "load.jsonl"
    jsonl.write_text(_make_streaming_lines(n_ok=10, ttft_s=0.25))
    metrics = _aggregate_jsonl(jsonl)
    assert metrics is not None
    assert metrics.ttft_median_ms == pytest.approx(250.0)


def test_streaming_bundle_row_is_plot_ready(tmp_path: Path) -> None:
    """A drive-load bundle with ttft_s produces a plot-ready AtlasCell row."""
    bundle = tmp_path / "kimi-stream-20260529T000000Z"
    raw = bundle / "raw"
    raw.mkdir(parents=True)
    (raw / "load.jsonl").write_text(_make_streaming_lines(n_ok=30))
    (bundle / "inference_perfbench_v1.json").write_text(json.dumps(_BUNDLE_META))
    campaign = _make_campaign(tmp_path)

    result = import_drive_load_bundle(bundle, campaign, concurrency_override=15)
    rows = json.loads((result.cell_dir / "normalized.json").read_text())
    assert rows[0]["ttft_avg_ms"] == pytest.approx(250.0)
    assert rows[0]["request_throughput_avg"] is not None
    # has_metrics (plot-ready) requires both ttft + request_throughput.
    assert rows[0]["ttft_avg_ms"] is not None and rows[0]["request_throughput_avg"] is not None


# ---------------------------------------------------------------------------
# 4. import_drive_load_bundle — end-to-end (multi-c)
# ---------------------------------------------------------------------------


def test_import_multi_c_writes_normalized(tmp_path: Path) -> None:
    bundle = _make_multi_c_bundle(tmp_path)
    campaign = _make_campaign(tmp_path)
    result = import_drive_load_bundle(bundle, campaign)
    assert isinstance(result, DriveLoadImportResult)
    assert result.row_count == 3
    assert result.concurrencies == [15, 30, 60]
    assert result.normalized_path.is_file()
    rows = json.loads(result.normalized_path.read_text())
    assert len(rows) == 3
    # Verify schema invariants on every row
    for row in rows:
        assert row["model"] == "moonshotai/Kimi-K2.6"
        assert row["tensor_parallel"] == 8
        assert row["backend"] == "vllm-sweep"
        assert row["ttft_avg_ms"] is None
        assert row["extra"]["ttft_unavailable_from_drive_load"] is True
        assert "n_ok" in row["extra"]
        assert "n_fail" in row["extra"]


def test_import_partial_status_when_some_failures(tmp_path: Path) -> None:
    bundle = _make_multi_c_bundle(tmp_path)
    campaign = _make_campaign(tmp_path)
    result = import_drive_load_bundle(bundle, campaign)
    # bench-c30 and bench-c60 have fail_count > 0; status should still be "full"
    # because no JSONL file was skipped (skipped = "had zero OK"). Per-row
    # status is "partial" when fail > 0 (see _row_from_aggregate).
    assert result.status == "full"
    rows = json.loads(result.normalized_path.read_text())
    by_c = {r["concurrency"]: r["status"] for r in rows}
    assert by_c[15] == "full"
    assert by_c[30] == "partial"
    assert by_c[60] == "partial"


def test_import_dry_run_writes_nothing(tmp_path: Path) -> None:
    bundle = _make_multi_c_bundle(tmp_path)
    campaign = _make_campaign(tmp_path)
    result = import_drive_load_bundle(bundle, campaign, dry_run=True)
    assert result.row_count == 3
    assert not result.normalized_path.exists()
    assert not (campaign / "cells" / result.cell_id).exists()


# ---------------------------------------------------------------------------
# 5. import_drive_load_bundle — single-c + overrides
# ---------------------------------------------------------------------------


def test_import_single_c_with_override(tmp_path: Path) -> None:
    bundle = _make_single_c_bundle(tmp_path)
    campaign = _make_campaign(tmp_path)
    result = import_drive_load_bundle(
        bundle, campaign, concurrency_override=15
    )
    assert result.concurrencies == [15]
    rows = json.loads(result.normalized_path.read_text())
    assert len(rows) == 1
    assert rows[0]["concurrency"] == 15


def test_import_drive_load_isl_osl_total_and_cache_mode(tmp_path: Path) -> None:
    """v1.42.0 carry-through: mean ISL/OSL + total_tps_per_gpu + cache_mode."""
    bundle = _make_single_c_bundle(tmp_path)
    campaign = _make_campaign(tmp_path)
    result = import_drive_load_bundle(
        bundle, campaign, concurrency_override=15,
        overrides={"cache_mode": "cold", "tensor_parallel": 8},
    )
    row = json.loads(result.normalized_path.read_text())[0]
    # _make_jsonl_lines uses prompt_tokens=1024, completion_tokens=128.
    assert row["mean_input_tokens"] == pytest.approx(1024.0)
    assert row["mean_output_tokens"] == pytest.approx(128.0)
    assert row["total_tps_per_gpu"] is not None and row["total_tps_per_gpu"] > 0
    assert row["cache_mode"] == "cold"


def test_import_missing_model_raises(tmp_path: Path) -> None:
    bundle = _make_multi_c_bundle(tmp_path, with_meta=False)
    campaign = _make_campaign(tmp_path)
    with pytest.raises(ValueError, match="--model is required"):
        import_drive_load_bundle(bundle, campaign)


# ---------------------------------------------------------------------------
# 6. import_bundle_auto — dispatcher
# ---------------------------------------------------------------------------


def test_auto_dispatches_to_drive_load(tmp_path: Path) -> None:
    bundle = _make_multi_c_bundle(tmp_path)
    campaign = _make_campaign(tmp_path)
    result = import_bundle_auto(bundle, campaign)
    assert isinstance(result, DriveLoadImportResult)
    assert getattr(result, "importer", "") == "inference_drive_load"


def test_auto_dispatches_to_perf_bench(tmp_path: Path) -> None:
    """auto_dispatch picks perf_bench when sweep-c*.txt is present."""
    bundle = tmp_path / "sweep-bundle"
    raw = bundle / "raw"
    raw.mkdir(parents=True)
    (raw / "sweep-c8.txt").write_text(
        "============ Serving Benchmark Result ============\n"
        "Successful requests:                     16\n"
        "Benchmark duration (s):                  10.0\n"
        "Request throughput (req/s):              1.6\n"
        "Output token throughput (tok/s):         256.0\n"
        "Total token throughput (tok/s):          2560.0\n"
        "Median TTFT (ms):                        100.0\n"
        "Median TPOT (ms):                        25.0\n"
    )
    (bundle / "inference_perfbench_v1.json").write_text(json.dumps(_BUNDLE_META))
    campaign = _make_campaign(tmp_path)
    result = import_bundle_auto(bundle, campaign)
    # bench-serve ImportResult has no `importer` attr (predates v1.21.0)
    assert not isinstance(result, DriveLoadImportResult)


def test_auto_raises_on_unknown_pattern(tmp_path: Path) -> None:
    bundle = tmp_path / "empty"
    bundle.mkdir()
    campaign = _make_campaign(tmp_path)
    with pytest.raises(ValueError, match="no recognized importer pattern"):
        import_bundle_auto(bundle, campaign)
