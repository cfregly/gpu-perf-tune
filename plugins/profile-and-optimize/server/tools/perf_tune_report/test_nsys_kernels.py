"""Tests for the nsys cuda_gpu_kern_sum -> kernels.json importer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.perf_tune_report.importers.nsys_kernels import (
    NsysKernSumMalformed,
    NsysKernSumMissing,
    import_nsys_kernels,
)

# A trimmed cuda_gpu_kern_sum sample (the real GQA NVFP4-KV capture shape).
_KERN_SUM = """\
 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)   Name
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ----
     40.9       2949542239      38254   77104.2   78176.0     14560    119392      12780.8  bmm_Bfloat16_Bfloat16Bfloat16_Fp32_t128x8x128u2_s6_et128x8_m128x8x16
      8.4        608746267      37296   16322.0   16544.0      9376     24640       1576.3  fmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvDenseP16VarSeqQ8Kv128PersistentSwapsAbForGen
      4.7        342356086      72048    4751.8    5536.0      3392      7936       1186.1  nvjet_tst_64x32_64x16_2x2_2cta_h_bz_splitK_TNT
      3.5        249047608      37248    6686.2    6688.0      5887      7423        162.4  void moe::dev::routing::routingCustom::routingIndicesClusterKernel
      1.7        123056628      38350    3208.8    3200.0      2752      4192        129.1  void vllm::reshape_and_cache_nvfp4_kernel
      1.4        100043146      38350    2608.7    2592.0      2304      3744        113.5  triton_red_fused_fused_add_rms_norm_moe_forward_0
"""


def _mk_bundle(tmp_path: Path, declare: bool = True, kern_sum: str | None = _KERN_SUM) -> Path:
    b = tmp_path / "bundle"
    (b / "nsys").mkdir(parents=True)
    if declare:
        (b / "capture_sources.json").write_text(json.dumps({"captured_sources": ["nsys"]}))
    if kern_sum is not None:
        (b / "nsys" / "cuda_gpu_kern_sum.txt").write_text(kern_sum)
    return b


def test_import_happy_path(tmp_path: Path):
    b = _mk_bundle(tmp_path)
    cell = tmp_path / "cells" / "nvfp4"
    r = import_nsys_kernels(b, cell)
    assert r.kernels_json_path is not None and r.skipped_reason is None
    assert r.top_kernel_count == 6
    payload = json.loads((cell / "kernels.json").read_text())
    assert payload["captured_sources"] == ["nsys"]
    cats = payload["per_category"]
    # the two bmm/routing rows fold into MoE; the fmha row is FMHA; nvjet is cuBLAS
    assert cats["MoE"] == 2949542239 + 249047608 + 100043146  # bf16 bmm + routing + triton-moe...
    assert "FMHA" in cats and cats["FMHA"] == 608746267
    assert "cuBLAS" in cats and "Other" in cats  # nvjet -> cuBLAS; reshape_and_cache_nvfp4 -> Other
    # MoE is the dominant category (the BF16 expert GEMM)
    assert max(cats, key=cats.get) == "MoE"
    assert payload["top_python_during_cuda"] == []


def test_skip_when_not_declared(tmp_path: Path):
    b = _mk_bundle(tmp_path, declare=False)
    r = import_nsys_kernels(b, tmp_path / "cells" / "c")
    assert r.kernels_json_path is None and "does not declare nsys" in r.skipped_reason


def test_missing_kern_sum_raises(tmp_path: Path):
    b = _mk_bundle(tmp_path, kern_sum=None)
    with pytest.raises(NsysKernSumMissing):
        import_nsys_kernels(b, tmp_path / "cells" / "c")


def test_no_rows_raises(tmp_path: Path):
    b = _mk_bundle(tmp_path, kern_sum=" ** header only, no data rows **\n")
    with pytest.raises(NsysKernSumMalformed):
        import_nsys_kernels(b, tmp_path / "cells" / "c")
