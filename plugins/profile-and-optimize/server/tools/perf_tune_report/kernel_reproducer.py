"""perf_tune_report kernel_reproducer_scaffold (v1.69.0).

Scaffold a standalone CUDA/CUTLASS kernel reproducer (.cu + build script) for white-box
kernel debugging -- Track B of the `inference-kernel-whitebox-debug` skill. Emits a
self-contained harness modeled on the proven GLM-5.1 `linear_sm100_mpk` reproducer
(`repro_linsm100_bf16.cu` + `build_repro_linsm100.sh`), parameterized by the kernel's
GEMM dims + the mirage tree + GPU arch. The harness instantiates the kernel template,
feeds CONTROLLED inputs (all-ones, then an optional real dump) and diffs vs a host GEMM.

The operator transcribes the EXACT template params + TMA descriptor types from the
codegen/registration site (e.g. `task_register.cc`) into the marked block -- the
scaffolder emits the canonical `linear_sm100_mpk`-shaped skeleton + the correct build
flags so the boilerplate is not hand-retyped each time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CU_TEMPLATE = r"""// White-box standalone reproducer (scaffolded by perf_tune_report_kernel_reproducer_scaffold).
// Kernel: {kernel_name}  dims: MMA_M={mma_m} MMA_N={mma_n} BATCH={batch} OUT={out} K={k}
//
// all-ones case: weight=input=1.0 -> every output element MUST equal K (={k}).
//   out == K        => MMA accumulation count CORRECT (defect is NOT the MMA core).
//   out ~= N*K      => the kernel over-accumulates (DEFINITIVE kernel-core defect).
// real case (REAL_INPUT=1): /work/repro_w.bin + /work/repro_in.bin (float32) vs host GEMM.
//
// TODO(operator): transcribe the EXACT template params + tma_2d descriptor types for
// {kernel_name} from the codegen site (e.g. task_register.cc register_*_task). The block
// below is the canonical linear_sm100_mpk shape; adjust to your kernel.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

#include "{header}"
#include "tasks/hopper/tma_2d.cuh"

using namespace cute;
using BF = cute::bfloat16_t;

constexpr int MMA_M = {mma_m}, MMA_N = {mma_n}, BATCH = {batch}, OUT = {out}, K = {k};
constexpr int AB = 8, ACC = 2, NCST = 4, TILE = 64, CP = 64;
constexpr int SREP_COL = (TILE + CP - 1) / CP;

// ---- BEGIN operator-transcribed block (exact instantiation + TMA descriptors) ----
using TMA_A   = kernel::tma::tma_2d<BF, 3, 3, 3, OUT,   K,   MMA_M, CP,    K,   1, 1, SREP_COL, MMA_M * CP, true>;
using TMA_B   = kernel::tma::tma_2d<BF, 3, 3, 3, BATCH, K,   MMA_N, CP,    K,   1, 1, SREP_COL, MMA_N * CP, true>;
using TMA_OUT = kernel::tma::tma_2d<BF, 0, 3, 3, BATCH, OUT, MMA_N, MMA_M, OUT, 1, 1, 1,        MMA_N * MMA_M, true>;

__global__ __launch_bounds__(256, 1) void repro(CUtensorMap *da, CUtensorMap *db, CUtensorMap *dout) {{
  TMA_A tma_a(da); TMA_B tma_b(db); TMA_OUT tma_out(dout);
  auto layout_Bias = make_layout(make_shape(Int<BATCH>{{}}, Int<OUT>{{}}), make_stride(Int<OUT>{{}}, Int<1>{{}}));
  auto mBias = make_tensor(make_gmem_ptr((BF *)nullptr), layout_Bias);
  kernel::{kernel_name}<BF, TMA_A, TMA_B, decltype(mBias), TMA_OUT,
                        MMA_M, MMA_N, BATCH, OUT, K, /*NOBIAS=*/true, /*SplitK=*/false, AB, ACC, NCST>(
      tma_a, tma_b, mBias, tma_out);
}}
// ---- END operator-transcribed block ----

int main() {{
  cudaSetDevice(0);
  const bool real_input = (getenv("REAL_INPUT") && atoi(getenv("REAL_INPUT")) == 1);
  size_t wn = (size_t)OUT * K, in = (size_t)BATCH * K, on = (size_t)BATCH * OUT;
  std::vector<float> wf(wn, 1.0f), inf(in, 1.0f);
  if (real_input) {{
    FILE *fw = fopen("/work/repro_w.bin", "rb"), *fi = fopen("/work/repro_in.bin", "rb");
    if (fw) {{ std::vector<float> t(wn); if (fread(t.data(), 4, wn, fw) == wn) wf = t; fclose(fw); }}
    if (fi) {{ std::vector<float> t(in); if (fread(t.data(), 4, in, fi) == in) inf = t; fclose(fi); }}
  }}
  std::vector<BF> hw(wn), hi(in);
  for (size_t i = 0; i < wn; i++) hw[i] = BF(wf[i]);
  for (size_t i = 0; i < in; i++) hi[i] = BF(inf[i]);
  BF *dw, *di, *dout;
  cudaMalloc(&dw, wn * 2); cudaMalloc(&di, in * 2); cudaMalloc(&dout, on * 2);
  cudaMemcpy(dw, hw.data(), wn * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(di, hi.data(), in * 2, cudaMemcpyHostToDevice);
  cudaMemset(dout, 0, on * 2);
  TMA_A ha((void *)dw); TMA_B hb((void *)di); TMA_OUT ho((void *)dout);
  int ss = 200 * 1024;
  cudaFuncSetAttribute(repro, cudaFuncAttributeMaxDynamicSharedMemorySize, ss);
  repro<<<dim3(1, 1, 1), dim3(256, 1, 1), ss>>>(ha.desc_ptr, hb.desc_ptr, ho.desc_ptr);
  cudaError_t e = cudaDeviceSynchronize();
  printf("launch: %s (smem=%d)\n", cudaGetErrorString(e), ss);
  if (e != cudaSuccess) {{ printf("REPRO_LAUNCH_FAILED\n"); return 1; }}
  std::vector<BF> out(on); cudaMemcpy(out.data(), dout, on * 2, cudaMemcpyDeviceToHost);
  auto ref = [&](int t, int n) {{ double a = 0; for (int k = 0; k < K; k++) a += (double)inf[(size_t)t*K+k]*(double)wf[(size_t)n*K+k]; return a; }};
  double l2n = 0, l2d = 0, sg = 0, sr = 0; float maxabs = 0;
  for (int t = 0; t < BATCH; t++) for (int n = 0; n < OUT; n++) {{
    double r = ref(t, n); float got = (float)out[(size_t)t*OUT+n]; double d = got - r;
    l2n += d*d; l2d += r*r; maxabs = fmaxf(maxabs, fabsf((float)d)); sg += fabs((double)got); sr += fabs(r);
  }}
  double relL2 = (l2d > 0) ? sqrt(l2n/l2d) : -1, ratio = (sr > 0) ? sg/sr : -1;
  printf("[REPRO] mode=%s relL2=%.5f maxabs=%.3f kernel/ref_ratio=%.3f\n", real_input?"real":"all-ones", relL2, maxabs, ratio);
  printf("%s (relL2<0.02 => GEMM correct; ratio~N => over-amplifies)\n", (relL2>=0 && relL2<2e-2)?"REPRO_CORRECT":"REPRO_OVER_OR_WRONG");
  return 0;
}}
"""

_SH_TEMPLATE = r"""#!/bin/bash
# Build + run the standalone {kernel_name} reproducer (scaffolded). GB300 / 1 GPU.
set -uo pipefail
M={mirage_tree}
SRC=${{SRC:-{cu_pod_path}}}
OUT=${{OUT:-{exe_pod_path}}}
NVCC={nvcc}
INC="-I$M/include -I$M/include/mirage/persistent_kernel -I$M/deps/cutlass/include -I$M/deps/cutlass/tools/util/include -I$M/deps/json/include"
DEFS="-DMPK_ENABLE_TMA -DMIRAGE_GRACE_BLACKWELL -DMPK_TARGET_CC=103 -DMIRAGE_BACKEND_USE_CUDA -DMODE_OFFLINE -DMIRAGE_USE_CUTLASS_KERNEL=1 -DMPK_MAX_NUM_BATCHED_REQUESTS=1 -DMPK_MAX_NUM_BATCHED_TOKENS=8 -DMPK_MAX_NUM_PAGES=8 -DMPK_PAGE_SIZE=128 -DMPK_MAX_SEQ_LENGTH=128"
ARCH="-gencode=arch={arch} -std=c++20 -O2 -use_fast_math --expt-relaxed-constexpr -diag-suppress=3012 -lineinfo"
echo "[repro-build] $(date -u) nvcc $SRC -> $OUT"
$NVCC "$SRC" $INC $ARCH $DEFS -lcuda -lcudart -o "$OUT"; bc=$?
echo "BUILD_EXIT=$bc"; [ "$bc" -ne 0 ] && exit "$bc"
echo "[repro-run] $(date -u)"; CUDA_VISIBLE_DEVICES=0 "$OUT"; echo "[repro] done ec=$?"
"""


@dataclass
class ScaffoldResult:
    cu_path: str
    build_path: str
    kernel_name: str
    dims: dict[str, int]
    wrote: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "cu_path": self.cu_path,
            "build_path": self.build_path,
            "kernel_name": self.kernel_name,
            "dims": self.dims,
            "wrote": self.wrote,
        }


def scaffold_reproducer(
    *,
    kernel_name: str,
    header: str,
    out_dir: Path,
    mma_m: int = 128,
    mma_n: int = 16,
    batch: int = 8,
    out: int = 1024,
    k: int = 6144,
    mirage_tree: str = "/work/mirage-perop2",
    arch: str = "compute_103a,code=sm_103a",
    nvcc: str = "/usr/local/cuda-13.2/bin/nvcc",
    cu_pod_path: str = "/work/repro_kernel.cu",
    exe_pod_path: str = "/work/repro_kernel",
    dry_run: bool = False,
) -> ScaffoldResult:
    """Emit a standalone reproducer .cu + build .sh into out_dir.

    The .cu is the canonical linear_sm100_mpk-shaped harness parameterized by the GEMM
    dims; the operator transcribes the exact template params + TMA descriptors for
    `kernel_name` into the marked block. The .sh carries the GB300 mirage build flags.
    """
    out_dir = Path(out_dir).expanduser().resolve()
    cu = out_dir / f"repro_{kernel_name}.cu"
    sh = out_dir / f"build_repro_{kernel_name}.sh"
    dims = {"MMA_M": mma_m, "MMA_N": mma_n, "BATCH": batch, "OUT": out, "K": k}
    cu_text = _CU_TEMPLATE.format(
        kernel_name=kernel_name, header=header,
        mma_m=mma_m, mma_n=mma_n, batch=batch, out=out, k=k,
    )
    sh_text = _SH_TEMPLATE.format(
        kernel_name=kernel_name, mirage_tree=mirage_tree, arch=arch, nvcc=nvcc,
        cu_pod_path=cu_pod_path, exe_pod_path=exe_pod_path,
    )
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        cu.write_text(cu_text)
        sh.write_text(sh_text)
    return ScaffoldResult(
        cu_path=str(cu), build_path=str(sh), kernel_name=kernel_name, dims=dims, wrote=not dry_run,
    )
