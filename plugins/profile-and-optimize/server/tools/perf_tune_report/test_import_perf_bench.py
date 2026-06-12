"""Unit tests for the perf-bench bundle importer (v1.18.0)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.perf_tune_report.capture_signature import variant_key_for
from tools.perf_tune_report.importers.inference_perf_bench import (
    ImportResult,
    _enumerate_sweep_files,
    _parse_metrics,
    _SWEEP_K,
    _SWEEP_SIMPLE,
    import_perf_bench_bundle,
)
from tools.perf_tune_report.schema import AtlasCell


# --- sample bench-serve output blocks (anonymized; matches V1 sweep format) -

_SWEEP_C8 = """============ Serving Benchmark Result ============
Successful requests:                     64
Benchmark duration (s):                  63.34
Request throughput (req/s):              1.01
Output token throughput (tok/s):         517.31
Total token throughput (tok/s):          4659.51
Median TTFT (ms):                        285.48
Median TPOT (ms):                        14.41
"""

_SWEEP_C32 = """============ Serving Benchmark Result ============
Successful requests:                     128
Benchmark duration (s):                  44.81
Request throughput (req/s):              2.857
Output token throughput (tok/s):         1462.54
Total token throughput (tok/s):          11700.51
Median TTFT (ms):                        591.28
Median TPOT (ms):                        19.29
"""

_SWEEP_MALFORMED = "no benchmark result here\nsome other output\n"


# --- fixture helpers ------------------------------------------------------


def _make_bundle(tmp_path: Path, *, with_meta: bool = True) -> Path:
    """Create a fake inference-perf-bench bundle with 2 sweep files."""
    bundle = tmp_path / "fake-bundle-20260526T000000Z"
    raw = bundle / "raw"
    raw.mkdir(parents=True)
    (raw / "sweep-c8.txt").write_text(_SWEEP_C8)
    (raw / "sweep-c32.txt").write_text(_SWEEP_C32)
    if with_meta:
        (bundle / "inference_perfbench_v1.json").write_text(
            json.dumps(
                {
                    "schema": "inference_perfbench_v1",
                    "model": "zai-org/GLM-5.1",
                    "hardware": "B200 (single node, TP=8)",
                    "tensor_parallel_size": 8,
                    "parallel_strategy": "TP",
                    "mtp": False,
                    "max_num_batched_tokens": 12288,
                    "max_num_seqs": 192,
                    "kv_cache_dtype": "fp8_e4m3",
                    "patched_vllm_enabled": True,
                    "vllm_image": "registry.example.com/infr/vllm:v2.12.3",
                }
            )
        )
    return bundle


def _make_campaign(tmp_path: Path) -> Path:
    campaign = tmp_path / "campaign-test"
    campaign.mkdir()
    (campaign / "cells").mkdir()
    return campaign


# --- filename regex tests --------------------------------------------------


def test_sweep_simple_filename_regex():
    m = _SWEEP_SIMPLE.match("sweep-c32.txt")
    assert m is not None
    assert int(m.group(1)) == 32


def test_sweep_k_filename_regex():
    m = _SWEEP_K.match("sweep-K2-c16.txt")
    assert m is not None
    assert int(m.group(1)) == 2
    assert int(m.group(2)) == 16


def test_sweep_filename_regex_rejects_malformed():
    assert _SWEEP_SIMPLE.match("sweep-cabc.txt") is None
    assert _SWEEP_K.match("sweep-K-c8.txt") is None


# --- _enumerate_sweep_files ------------------------------------------------


def test_enumerate_sweep_files_finds_simple_files(tmp_path):
    bundle = _make_bundle(tmp_path, with_meta=False)
    files = _enumerate_sweep_files(bundle)
    assert len(files) == 2
    assert {f.concurrency for f in files} == {8, 32}
    assert all(f.k == 1 for f in files)


def test_enumerate_sweep_files_sorted_by_k_then_c(tmp_path):
    bundle = tmp_path / "ksweep"
    raw = bundle / "raw"
    raw.mkdir(parents=True)
    for k in (3, 2, 1):
        for c in (4, 1, 16):
            (raw / f"sweep-K{k}-c{c}.txt").write_text(_SWEEP_C8)
    files = _enumerate_sweep_files(bundle)
    assert [(f.k, f.concurrency) for f in files] == [
        (1, 1), (1, 4), (1, 16),
        (2, 1), (2, 4), (2, 16),
        (3, 1), (3, 4), (3, 16),
    ]


def test_enumerate_returns_empty_for_missing_raw_dir(tmp_path):
    assert _enumerate_sweep_files(tmp_path / "nonexistent") == []


# --- _parse_metrics --------------------------------------------------------


def test_parse_metrics_extracts_expected_fields(tmp_path):
    fp = tmp_path / "sweep-c32.txt"
    fp.write_text(_SWEEP_C32)
    m = _parse_metrics(fp)
    assert m is not None
    assert m["n_reqs"] == 128
    assert m["duration_s"] == pytest.approx(44.81)
    assert m["req_per_s"] == pytest.approx(2.857)
    assert m["output_tps"] == pytest.approx(1462.54)
    assert m["ttft_med_ms"] == pytest.approx(591.28)
    assert m["tpot_med_ms"] == pytest.approx(19.29)


def test_parse_metrics_returns_none_for_malformed(tmp_path):
    fp = tmp_path / "broken.txt"
    fp.write_text(_SWEEP_MALFORMED)
    assert _parse_metrics(fp) is None


# --- v1.42.0 carry-through: ISL/OSL, cache_mode, prefix-cache --------------

_SWEEP_WITH_TOKENS = """============ Serving Benchmark Result ============
Successful requests:                     100
Benchmark duration (s):                  50.0
Total input tokens:                      320000
Total generated tokens:                  51200
Request throughput (req/s):              2.0
Output token throughput (tok/s):         1024.0
Total token throughput (tok/s):          7424.0
Median TTFT (ms):                        200.0
Median TPOT (ms):                        15.0
"""


def test_parse_metrics_extracts_token_totals(tmp_path):
    fp = tmp_path / "sweep-c16.txt"
    fp.write_text(_SWEEP_WITH_TOKENS)
    m = _parse_metrics(fp)
    assert m is not None
    assert m["total_input_tokens"] == pytest.approx(320000)
    assert m["total_generated_tokens"] == pytest.approx(51200)


def test_import_derives_mean_isl_osl(tmp_path):
    bundle = tmp_path / "shape-bundle-20260601T000000Z"
    raw = bundle / "raw"
    raw.mkdir(parents=True)
    (raw / "sweep-c16.txt").write_text(_SWEEP_WITH_TOKENS)
    campaign = _make_campaign(tmp_path)
    import_perf_bench_bundle(bundle, campaign, overrides={"model": "m", "cell_id": "c"})
    rows = json.loads((campaign / "cells" / "c" / "normalized.json").read_text())
    # mean ISL = 320000/100 = 3200; mean OSL = 51200/100 = 512.
    assert rows[0]["mean_input_tokens"] == pytest.approx(3200.0)
    assert rows[0]["mean_output_tokens"] == pytest.approx(512.0)


def test_import_cache_mode_override(tmp_path):
    bundle = _make_bundle(tmp_path)
    campaign = _make_campaign(tmp_path)
    import_perf_bench_bundle(
        bundle, campaign, overrides={"cell_id": "c", "cache_mode": "warm"}
    )
    rows = json.loads((campaign / "cells" / "c" / "normalized.json").read_text())
    assert all(r["cache_mode"] == "warm" for r in rows)


def test_import_cache_mode_defaults_unknown(tmp_path):
    bundle = _make_bundle(tmp_path)
    campaign = _make_campaign(tmp_path)
    import_perf_bench_bundle(bundle, campaign, overrides={"cell_id": "c"})
    rows = json.loads((campaign / "cells" / "c" / "normalized.json").read_text())
    assert all(r["cache_mode"] == "unknown" for r in rows)


def test_import_prefix_cache_hit_rate_from_bundle_meta(tmp_path):
    bundle = tmp_path / "pchr-bundle-20260601T000000Z"
    raw = bundle / "raw"
    raw.mkdir(parents=True)
    (raw / "sweep-c8.txt").write_text(_SWEEP_C8)
    (bundle / "inference_perfbench_v1.json").write_text(
        json.dumps({"model": "m", "prefix_cache_hit_rate": 0.83})
    )
    campaign = _make_campaign(tmp_path)
    import_perf_bench_bundle(bundle, campaign, overrides={"cell_id": "c"})
    rows = json.loads((campaign / "cells" / "c" / "normalized.json").read_text())
    assert all(r["prefix_cache_hit_rate"] == pytest.approx(0.83) for r in rows)


def test_parse_metrics_derives_req_per_s_if_missing(tmp_path):
    fp = tmp_path / "no-req-per-s.txt"
    fp.write_text(_SWEEP_C32.replace("Request throughput (req/s):              2.857\n", ""))
    m = _parse_metrics(fp)
    assert m is not None
    # 128 / 44.81 ~= 2.857
    assert m["req_per_s_derived"] == pytest.approx(128 / 44.81, rel=1e-3)


# --- import_perf_bench_bundle: success cases -------------------------------


def test_import_bundle_writes_normalized_json(tmp_path):
    bundle = _make_bundle(tmp_path)
    campaign = _make_campaign(tmp_path)

    result = import_perf_bench_bundle(bundle, campaign)

    assert isinstance(result, ImportResult)
    assert result.row_count == 2
    assert result.concurrencies == [8, 32]
    assert result.k_values == [1]
    assert result.status == "full"
    assert result.cell_id == bundle.name

    normalized = result.normalized_path
    assert normalized.is_file()
    data = json.loads(normalized.read_text())
    assert len(data) == 2
    # Spot-check the c=32 row's derived metrics.
    c32 = next(r for r in data if r["concurrency"] == 32)
    assert c32["model"] == "zai-org/GLM-5.1"
    assert c32["hardware"] == "B200"  # parens stripped from "B200 (single node, TP=8)"
    assert c32["tensor_parallel"] == 8
    assert c32["parallel_strategy"] == "TP"
    assert c32["max_num_batched_tokens"] == 12288
    assert c32["mtp"] is False
    assert c32["ttft_avg_ms"] == pytest.approx(591.28)
    assert c32["request_throughput_avg"] == pytest.approx(2.857)
    assert c32["output_tps_per_user"] == pytest.approx(1000 / 19.29)
    assert c32["output_tps_per_gpu"] == pytest.approx(1462.54 / 8)
    assert c32["backend"] == "vllm-sweep"
    assert c32["extra"]["max_num_seqs"] == 192
    assert c32["extra"]["patched_vllm_enabled"] is True
    # Sidecar files
    assert (result.cell_dir / "status.txt").read_text().strip() == "full"
    assert (result.cell_dir / "backend.txt").read_text().strip() == "vllm-sweep"
    assert (result.cell_dir / "SOURCE.md").is_file()


def _make_spec_bundle(tmp_path: Path, *, k: int, name: str) -> Path:
    """A bundle whose meta carries the typed serving-variant knobs (MTP-K + async +
    prefix-caching + bench backend)."""
    bundle = tmp_path / name
    raw = bundle / "raw"
    raw.mkdir(parents=True)
    (raw / "sweep-c8.txt").write_text(_SWEEP_C8)
    (bundle / "inference_perfbench_v1.json").write_text(
        json.dumps(
            {
                "schema": "inference_perfbench_v1",
                "model": "zai-org/GLM-5.1",
                "hardware": "GB300",
                "tensor_parallel_size": 4,
                "parallel_strategy": "TP",
                "max_num_batched_tokens": 2048,
                "max_num_seqs": 256,
                "kv_cache_dtype": "fp8_e4m3",
                "speculative_decoding": {"method": "mtp", "num_speculative_tokens": k},
                "async_scheduling": True,
                "prefix_caching": True,
                "bench_backend": "vllm",
                "vllm_image": "registry.example.com/infr/vllm:v2.12.3",
            }
        )
    )
    return bundle


def test_import_populates_typed_variant_fields(tmp_path):
    """The importer promotes MTP-K + max_num_seqs/async/prefix/backend to the typed
    AtlasCell fields (not only notes/extra) so variant_key can distinguish them."""
    bundle = _make_spec_bundle(tmp_path, k=3, name="spec-bundle-20260607T000000Z")
    campaign = _make_campaign(tmp_path)
    result = import_perf_bench_bundle(bundle, campaign)
    row = json.loads(result.normalized_path.read_text())[0]
    assert row["num_speculative_tokens"] == 3
    assert row["mtp"] is True  # inferred from speculative_decoding.method == mtp
    assert row["max_num_seqs"] == 256
    assert row["async_scheduling"] is True
    assert row["enable_prefix_caching"] is True
    assert row["bench_backend"] == "vllm"


def test_import_variant_key_distinguishes_mtp_k(tmp_path):
    """Two bundles differing only in num_speculative_tokens (K=2 vs K=3) yield rows
    whose image-independent variant_key differs -- the material A1 outcome."""
    b2 = _make_spec_bundle(tmp_path, k=2, name="spec-k2-20260607T000000Z")
    b3 = _make_spec_bundle(tmp_path, k=3, name="spec-k3-20260607T000000Z")
    c2 = _make_campaign(tmp_path)
    c3 = tmp_path / "campaign-test-k3"
    (c3 / "cells").mkdir(parents=True)
    r2 = json.loads(import_perf_bench_bundle(b2, c2).normalized_path.read_text())[0]
    r3 = json.loads(import_perf_bench_bundle(b3, c3).normalized_path.read_text())[0]
    assert variant_key_for(AtlasCell(**r2)) != variant_key_for(AtlasCell(**r3))


def test_import_dry_run_does_not_write(tmp_path):
    bundle = _make_bundle(tmp_path)
    campaign = _make_campaign(tmp_path)

    result = import_perf_bench_bundle(bundle, campaign, dry_run=True)
    assert result.row_count == 2
    # Nothing should be written.
    assert not result.normalized_path.exists()
    assert not result.cell_dir.exists() or not (result.cell_dir / "normalized.json").exists()


def test_import_k_sweep_emits_k_suffixed_cell_ids(tmp_path):
    bundle = tmp_path / "dsv4-ksweep"
    raw = bundle / "raw"
    raw.mkdir(parents=True)
    for k in (1, 2, 3):
        for c in (1, 4):
            (raw / f"sweep-K{k}-c{c}.txt").write_text(_SWEEP_C8)
    (bundle / "inference_perfbench_v1.json").write_text(
        json.dumps({"model": "deepseek-ai/DeepSeek-V4-Flash", "hardware": "B200",
                    "quant": "NVFP4", "tensor_parallel_size": 8,
                    "parallel_strategy": "TP", "mtp": True,
                    "max_num_batched_tokens": 12288})
    )
    campaign = _make_campaign(tmp_path)

    result = import_perf_bench_bundle(bundle, campaign)
    data = json.loads(result.normalized_path.read_text())

    # 3 K values × 2 concurrencies = 6 rows
    assert len(data) == 6
    assert result.k_values == [1, 2, 3]
    # K=1 keeps the base cell_id; K>1 gets a -K<n> suffix
    k_suffixes = sorted({r["cell_id"] for r in data})
    base = bundle.name
    assert base in k_suffixes
    assert f"{base}-K2" in k_suffixes
    assert f"{base}-K3" in k_suffixes
    # extra.spec_decode_k is set for K>1
    k3_row = next(r for r in data if r["cell_id"].endswith("-K3"))
    assert k3_row["extra"]["spec_decode_k"] == 3


def test_import_overrides_take_precedence_over_bundle_meta(tmp_path):
    bundle = _make_bundle(tmp_path)
    campaign = _make_campaign(tmp_path)

    result = import_perf_bench_bundle(
        bundle, campaign,
        overrides={
            "cell_id": "operator-named-cell",
            "model": "operator/override-model",
            "hardware": "GB300",
            "quant": "FP8",
            "tensor_parallel": 16,
            "notes": "operator-provided note",
        },
    )
    data = json.loads(result.normalized_path.read_text())
    assert data[0]["cell_id"] == "operator-named-cell"
    assert data[0]["model"] == "operator/override-model"
    assert data[0]["hardware"] == "GB300"
    assert data[0]["quant"] == "FP8"
    assert data[0]["tensor_parallel"] == 16
    assert "operator-provided note" in data[0]["notes"]


def test_import_no_metadata_requires_model_override(tmp_path):
    bundle = _make_bundle(tmp_path, with_meta=False)
    campaign = _make_campaign(tmp_path)

    with pytest.raises(ValueError, match="--model is required"):
        import_perf_bench_bundle(bundle, campaign)


def test_import_no_metadata_with_overrides_works(tmp_path):
    bundle = _make_bundle(tmp_path, with_meta=False)
    campaign = _make_campaign(tmp_path)
    result = import_perf_bench_bundle(
        bundle, campaign,
        overrides={
            "model": "moonshotai/Kimi-K2.6",
            "quant": "NVFP4",
            "tensor_parallel": 8,
        },
    )
    data = json.loads(result.normalized_path.read_text())
    assert data[0]["model"] == "moonshotai/Kimi-K2.6"


def test_import_skips_malformed_sweep_files(tmp_path):
    bundle = _make_bundle(tmp_path)
    raw = bundle / "raw"
    (raw / "sweep-c64.txt").write_text(_SWEEP_MALFORMED)  # 3rd file, malformed

    campaign = _make_campaign(tmp_path)
    result = import_perf_bench_bundle(bundle, campaign)
    # Still imports the 2 good files; status becomes partial because 1 was skipped.
    assert result.row_count == 2
    assert result.status == "partial"


def test_import_raises_for_missing_bundle(tmp_path):
    campaign = _make_campaign(tmp_path)
    with pytest.raises(ValueError, match="bundle does not exist"):
        import_perf_bench_bundle(tmp_path / "nonexistent", campaign)


def test_import_raises_for_bundle_with_no_sweep_files(tmp_path):
    bundle = tmp_path / "empty-bundle"
    (bundle / "raw").mkdir(parents=True)
    campaign = _make_campaign(tmp_path)

    with pytest.raises(ValueError, match="no sweep-c\\*\\.txt files"):
        import_perf_bench_bundle(bundle, campaign, overrides={"model": "x"})


_SWEEP_NO_TTFT = """============ Serving Benchmark Result ============
Successful requests:                     64
Benchmark duration (s):                  63.34
Request throughput (req/s):              1.01
Output token throughput (tok/s):         517.31
Total token throughput (tok/s):          4659.51
Median TPOT (ms):                        14.41
"""


def test_import_warns_loudly_on_full_but_unplottable_row(tmp_path, capsys):
    """A STATUS_FULL row missing ttft must emit a loud, actionable warning."""
    bundle = tmp_path / "no-ttft-bundle-20260526T000000Z"
    (bundle / "raw").mkdir(parents=True)
    (bundle / "raw" / "sweep-c8.txt").write_text(_SWEEP_NO_TTFT)
    campaign = _make_campaign(tmp_path)

    import_perf_bench_bundle(bundle, campaign, overrides={"model": "x"})

    err = capsys.readouterr().err
    assert "NOT plot-ready" in err
    assert "ttft_avg_ms" in err
    assert "How to fix" in err


def test_require_plot_ready_hard_fails_on_missing_ttft(tmp_path):
    """--require-plot-ready upgrades the missing-ttft WARNING to a hard error,
    so a strict throughput campaign can't be built from incomplete (grep-dropped)
    capture. Default stays back-compat (warn-only, import succeeds)."""
    bundle = tmp_path / "no-ttft-bundle-20260607T000000Z"
    (bundle / "raw").mkdir(parents=True)
    (bundle / "raw" / "sweep-c8.txt").write_text(_SWEEP_NO_TTFT)
    campaign = _make_campaign(tmp_path)

    # default (back-compat): import succeeds (warn-only)
    res = import_perf_bench_bundle(bundle, campaign, overrides={"model": "x"}, dry_run=True)
    assert res.row_count == 1

    # require_plot_ready: hard-fail at import, with an actionable message
    with pytest.raises(ValueError, match="NOT plot-ready"):
        import_perf_bench_bundle(
            bundle,
            campaign,
            overrides={"model": "x"},
            dry_run=True,
            require_plot_ready=True,
        )
