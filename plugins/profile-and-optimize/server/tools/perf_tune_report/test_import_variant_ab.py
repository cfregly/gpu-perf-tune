"""Tests for the variant-A/B importer (run-variant-ab.sh layout)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.perf_tune_report.importers import import_bundle_auto
from tools.perf_tune_report.importers.variant_ab import (
    detect_variant_ab,
    import_variant_ab_bundle,
)


def _bench_text(*, n, reqps, out_tps, total_tps, ttft, tpot) -> str:
    return (
        "============ Serving Benchmark Result ============\n"
        f"Successful requests:                     {n}\n"
        "Benchmark duration (s):                  22.6\n"
        f"Request throughput (req/s):              {reqps}\n"
        f"Output token throughput (tok/s):         {out_tps}\n"
        f"Total token throughput (tok/s):          {total_tps}\n"
        f"Median TTFT (ms):                        {ttft}\n"
        f"Median TPOT (ms):                        {tpot}\n"
        "Median ITL (ms):                         9.44\n"
    )


def _make_arm(arm_dir: Path, *, tp, warm, by_c_trials, result_extra=None):
    """by_c_trials: {concurrency: [ (n,reqps,out,total,ttft,tpot), ... per trial ]}"""
    arm_dir.mkdir(parents=True)
    for c, trials in by_c_trials.items():
        for t, vals in enumerate(trials, start=1):
            n, reqps, out, total, ttft, tpot = vals
            (arm_dir / f"c{c}-t{t}.txt").write_text(
                _bench_text(n=n, reqps=reqps, out_tps=out, total_tps=total, ttft=ttft, tpot=tpot)
            )
    result = {"arm": arm_dir.name, "tp": tp, "warm": warm}
    if result_extra:
        result.update(result_extra)
    (arm_dir / "result.json").write_text(json.dumps(result))


@pytest.fixture
def ab_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "ab-c32"
    # arm 1: base -- 2 concurrencies x 2 trials (averaged)
    _make_arm(
        bundle / "minimax-m27-v-base", tp=4, warm=True,
        by_c_trials={
            1: [(8, 0.21, 220.0, 1100.0, 130.0, 4.40), (8, 0.21, 226.0, 1120.0, 142.0, 4.42)],
            32: [(96, 2.80, 2890.0, 14400.0, 640.0, 10.20), (96, 2.86, 3100.0, 15600.0, 660.0, 9.40)],
        },
    )
    # arm 2: ep
    _make_arm(
        bundle / "minimax-m27-v-ep", tp=4, warm=True,
        by_c_trials={
            32: [(96, 2.79, 2880.0, 14300.0, 650.0, 10.30), (96, 2.85, 3090.0, 15500.0, 655.0, 9.50)],
        },
    )
    return bundle


def test_detect(ab_bundle):
    assert detect_variant_ab(ab_bundle) is True
    assert detect_variant_ab(ab_bundle.parent) is False  # parent has no c<C>-t<T>.txt


def test_import_emits_plot_ready_cells(ab_bundle, tmp_path):
    campaign = tmp_path / "campaign"
    (campaign / "cells").mkdir(parents=True)
    res = import_variant_ab_bundle(
        ab_bundle, campaign,
        overrides={"model": "nvidia/MiniMax-M2.7-NVFP4", "hardware": "GB300",
                   "quant": "NVFP4", "tensor_parallel": 4, "max_num_batched_tokens": 8192},
        captured_at="2026-06-05T00:00:00Z",
    )
    assert sorted(res.cells) == ["minimax-m27-v-base", "minimax-m27-v-ep"]
    assert res.status == "full"
    assert 32 in res.concurrencies

    base = json.loads((campaign / "cells" / "minimax-m27-v-base" / "normalized.json").read_text())
    by_c = {r["concurrency"]: r for r in base}
    c32 = by_c[32]
    # trial-averaged: out_tps (2890+3100)/2 = 2995 cluster -> /tp(4) = 748.75 per GPU
    assert c32["output_tps_per_gpu"] == pytest.approx(748.75, abs=0.5)
    # PLOT-READY: ttft + request throughput present
    assert c32["ttft_avg_ms"] == pytest.approx(650.0, abs=0.5)
    assert c32["request_throughput_avg"] == pytest.approx(2.83, abs=0.02)
    assert c32["tpot_median_ms"] == pytest.approx(9.80, abs=0.05)
    assert c32["hardware"] == "GB300"
    assert c32["tensor_parallel"] == 4
    assert c32["cache_mode"] == "warm"
    assert c32["model"] == "nvidia/MiniMax-M2.7-NVFP4"


def test_auto_dispatch_routes_to_variant_ab(ab_bundle, tmp_path):
    campaign = tmp_path / "campaign2"
    (campaign / "cells").mkdir(parents=True)
    res = import_bundle_auto(
        ab_bundle, campaign,
        overrides={"model": "nvidia/MiniMax-M2.7-NVFP4", "hardware": "GB300"},
    )
    assert getattr(res, "importer", None) == "variant_ab"
    # duck-typed single-cell fields the CLI prints:
    assert "arms" in res.cell_id
    assert res.k_values == [1]


def test_mtp_inferred_from_arm_name(tmp_path):
    bundle = tmp_path / "ab"
    _make_arm(bundle / "m-v-mtpk3", tp=4, warm=True,
              by_c_trials={32: [(96, 2.8, 2900.0, 14000.0, 640.0, 10.0)]})
    campaign = tmp_path / "c"; (campaign / "cells").mkdir(parents=True)
    import_variant_ab_bundle(bundle, campaign, overrides={"model": "x", "hardware": "GB300"})
    row = json.loads((campaign / "cells" / "m-v-mtpk3" / "normalized.json").read_text())[0]
    assert row["mtp"] is True


def test_arm_result_json_flows_variant_descriptors(tmp_path):
    bundle = tmp_path / "ab"
    _make_arm(
        bundle / "qnext-v-fimoe",
        tp=4,
        warm=False,
        by_c_trials={64: [(96, 2.8, 2900.0, 14000.0, 640.0, 10.0)]},
        result_extra={
            "engine": "vllm",
            "parallel_strategy": "EP",
            "max_num_seqs": 96,
            "dataset": "random",
            "cudagraph_mode": "full",
            "gpu_memory_utilization": 0.91,
            "kv_cache_dtype": "fp8_e4m3",
            "image": "infr/vllm:v2.12.3",
            "env": {"VLLM_USE_FLASHINFER_MOE_FP4": "1"},
            "flags": ["--enable-expert-parallel"],
        },
    )
    campaign = tmp_path / "c"; (campaign / "cells").mkdir(parents=True)
    import_variant_ab_bundle(bundle, campaign, overrides={"model": "x", "hardware": "GB300"})
    row = json.loads((campaign / "cells" / "qnext-v-fimoe" / "normalized.json").read_text())[0]
    assert row["parallel_strategy"] == "EP"
    assert row["cache_mode"] == "cold"
    assert row["dataset"] == "random"
    assert row["cudagraph_mode"] == "full"
    assert row["gpu_memory_utilization"] == 0.91
    assert row["kv_cache_dtype"] == "fp8_e4m3"
    assert row["image"] == "infr/vllm:v2.12.3"
    assert row["extra"]["max_num_seqs"] == 96
    assert row["extra"]["env"] == {"VLLM_USE_FLASHINFER_MOE_FP4": "1"}
    assert row["extra"]["flags"] == ["--enable-expert-parallel"]


# ---------------------------------------------------------------------------
# Per-arm zymtrace SoL ingestion (run-variant-ab.sh inline-capture path).
# Mirrors the inference_perf_bench declared-coverage contract, per arm.
# ---------------------------------------------------------------------------

_ZYM_TSVS = {
    "kernel-class.tsv": "event_kind\tkind\tsamples\ncuda\tnative\t1000\ncuda\tcuda\t500\n",
    "top-gpu-frames.tsv": (
        "kernel\tsamples\n"
        "multimem_all_reduce_kernel<bfloat16>\t558\n"
        "bmm_E2m1E2m1_Fp32_sm100f\t328\n"
    ),
    "per-gpu.tsv": (
        "gpu_name\tgpu_uuid\tsamples\n"
        "NVIDIA GB300\taaaa\t35961\n"
    ),
    "per-category.tsv": "category\tsamples\nBMM-NVFP4\t19212\nFMHA\t9226\n",
    "top-python-during-cuda.tsv": "python_frame\tsamples\nvllm.x\t12345\n",
}


def _add_zymtrace(arm_dir: Path) -> None:
    """Drop a valid capture_sources.json + zymtrace/*.tsv into an arm dir, as the
    hardened run-variant-ab.sh inline SoL capture does before teardown."""
    (arm_dir / "capture_sources.json").write_text(
        json.dumps({"schema_version": 1, "captured_sources": ["zymtrace"],
                    "pod_name": arm_dir.name})
    )
    zd = arm_dir / "zymtrace"
    zd.mkdir()
    for name, body in _ZYM_TSVS.items():
        (zd / name).write_text(body)


def test_zymtrace_kernels_emitted_when_arm_declares_it(tmp_path):
    bundle = tmp_path / "ab"
    arm = bundle / "minimax-m27-v-ep"
    _make_arm(arm, tp=4, warm=True,
              by_c_trials={32: [(96, 2.8, 2900.0, 14000.0, 640.0, 10.0)]})
    _add_zymtrace(arm)
    campaign = tmp_path / "c"; (campaign / "cells").mkdir(parents=True)
    import_variant_ab_bundle(bundle, campaign, overrides={"model": "x", "hardware": "GB300"})
    kj = campaign / "cells" / "minimax-m27-v-ep" / "kernels.json"
    assert kj.is_file(), "kernels.json must be emitted for an arm declaring zymtrace"
    payload = json.loads(kj.read_text())
    assert "per_category" in payload and payload["top_kernels"]
    assert "- kernels.json:" in (campaign / "cells" / "minimax-m27-v-ep" / "SOURCE.md").read_text()


def test_no_kernels_json_when_arm_has_no_manifest(tmp_path):
    # An arm WITHOUT capture_sources.json (single-engine / pre-SoL bundle) must still
    # import cleanly and simply not emit kernels.json (correct no-op).
    bundle = tmp_path / "ab"
    _make_arm(bundle / "m-v-base", tp=4, warm=True,
              by_c_trials={32: [(96, 2.8, 2900.0, 14000.0, 640.0, 10.0)]})
    campaign = tmp_path / "c"; (campaign / "cells").mkdir(parents=True)
    res = import_variant_ab_bundle(bundle, campaign, overrides={"model": "x", "hardware": "GB300"})
    assert res.status == "full"
    assert not (campaign / "cells" / "m-v-base" / "kernels.json").exists()


def test_declared_but_broken_zymtrace_aborts_import(tmp_path):
    # Manifest declares zymtrace but a TSV is 0-byte -> loud failure (no silent degrade).
    bundle = tmp_path / "ab"
    arm = bundle / "m-v-ep"
    _make_arm(arm, tp=4, warm=True,
              by_c_trials={32: [(96, 2.8, 2900.0, 14000.0, 640.0, 10.0)]})
    _add_zymtrace(arm)
    (arm / "zymtrace" / "per-category.tsv").write_text("")  # 0-byte -> ZymtraceTSVMissing
    campaign = tmp_path / "c"; (campaign / "cells").mkdir(parents=True)
    with pytest.raises(Exception):
        import_variant_ab_bundle(bundle, campaign, overrides={"model": "x", "hardware": "GB300"})
