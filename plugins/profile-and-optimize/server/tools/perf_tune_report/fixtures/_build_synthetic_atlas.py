"""Deterministic generator for ``synthetic_atlas.jsonl``.

The fixture mirrors the GLM-5.1 PDF coverage exactly:

- 8 variants x 5 max_num_batched_tokens = 40 atlas cells
- 38 full sweeps (6 concurrencies each = 228 plot-ready points)
- 1 partial sweep (GB300 NVFP4 TP=4 TP at mbt=2048, 4 concurrencies = 4 points)
- 1 failed cell (B200 NVFP4 TP=8 TP at mbt=8192, 0 plot-ready points)
- 20 evicted MTP context cells (failure context, not in atlas count)

Total: 232 plot-ready concurrency points + 1 failed status row + 20 evicted
status rows = 253 fixture rows representing 60 unique cell_ids.

Anchor metric values for C in {8, 16, 32} come straight from the PDF's
page-2 heatmap so the smoke render matches the source PDF byte-for-byte
in those cells. Values for C in {1, 2, 4} are scaled deterministically
from the C=8 anchor.

Run from the fixtures dir:
    python _build_synthetic_atlas.py > synthetic_atlas.jsonl
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.perf_tune_report.schema import (  # noqa: E402
    BACKEND_VLLM_SWEEP,
    STATUS_EVICTED,
    STATUS_FAILED,
    STATUS_FULL,
    STATUS_PARTIAL,
    AtlasCell,
)

OSL = 4096  # output seq length; matches the PDF's "OSL 4k"
CONCURRENCIES = (1, 2, 4, 8, 16, 32)
MBT_VALUES = (1024, 2048, 4096, 8192, 16384)
CAPTURED_AT = "2026-05-23T00:00:00+00:00"


@dataclass(frozen=True)
class Variant:
    hardware: str
    quant: str
    tp: int
    strategy: str  # "EP" or "TP"
    mtp: bool


VARIANTS: tuple[Variant, ...] = (
    Variant("H100", "FP8", 16, "EP", False),
    Variant("H100", "FP8", 16, "TP", False),
    Variant("H100", "FP8", 16, "EP", True),  # H100 FP8 MTP TP=16 EP
    Variant("H100", "FP8", 16, "TP", True),  # H100 FP8 MTP TP=16 TP
    Variant("B200", "NVFP4", 8, "EP", False),
    Variant("B200", "NVFP4", 8, "TP", False),
    Variant("GB300", "NVFP4", 4, "EP", False),
    Variant("GB300", "NVFP4", 4, "TP", False),
)

MODEL_BY_QUANT = {"FP8": "GLM-5.1-FP8", "NVFP4": "GLM-5.1-NVFP4"}


# ---------------------------------------------------------------------------
# PDF page-2 anchor values. Each list is [mbt1024, mbt2048, mbt4096, mbt8192, mbt16384].
# Index aligned with VARIANTS above (8 variants). Strings "failed" / "partial"
# mark cells with that status at that (variant, mbt).
# ---------------------------------------------------------------------------

C8_TPS_PER_GPU = [
    [14.9, 15.0, 15.1, 15.3, 14.9],
    [14.1, 14.3, 14.3, 14.3, 14.1],
    [21.7, 22.3, 22.7, 23.3, 21.3],
    [20.1, 19.8, 20.7, 20.4, 19.7],
    [61.3, 62.6, 63.0, 62.9, 62.4],
    [63.0, 64.8, 64.8, "failed", 64.4],
    [111.6, 57.8, 117.5, 116.7, 57.8],
    [115.1, 118.1, 121.5, 60.9, 119.4],
]

C8_TTFT_MS = [
    [1664.1, 1647.6, 1844.4, 2113.6, 2345.2],
    [1621.7, 1615.3, 2034.9, 2263.1, 2451.8],
    [1116.7, 901.0, 765.7, 825.0, 1027.5],
    [1265.8, 932.0, 946.7, 862.5, 1019.2],
    [1433.3, 1069.1, 1024.8, 972.7, 1093.2],
    [1446.2, 1151.1, 1035.2, "failed", 1146.9],
    [2221.1, 1721.3, 1098.0, 1169.7, 1234.3],
    [2430.2, 2609.1, 1093.2, 1276.8, 1307.3],
]

C16_TPS_PER_GPU = [
    [24.6, 25.1, 25.3, 25.7, 25.4],
    [24.3, 24.8, 25.2, 25.3, 25.1],
    [38.9, 37.7, 40.2, 38.8, 36.3],
    [35.2, 33.9, 34.1, 34.6, 34.9],
    [105.0, 107.9, 107.4, 94.2, 107.5],
    [108.1, 110.8, 110.9, "failed", 110.2],
    [183.1, 95.6, 193.9, 192.9, 95.3],
    [180.6, "partial", 199.2, 100.4, 196.1],
]

C16_TTFT_MS = [
    [2696.4, 2420.9, 2481.8, 3270.5, 4096.1],
    [2701.0, 2360.9, 2614.6, 3786.7, 4233.2],
    [1688.6, 1351.9, 1180.4, 1218.9, 1665.8],
    [1947.4, 1440.7, 1369.6, 1344.5, 1455.4],
    [1328.6, 985.6, 1182.0, 1434.6, 1366.3],
    [1286.3, 1097.9, 1241.9, "failed", 1453.6],
    [2472.3, 1546.4, 1274.1, 1541.6, 1434.7],
    [3490.8, "partial", 1343.1, 1446.2, 1933.1],
]

C32_TPS_PER_GPU = [
    [28.3, 35.9, 35.0, 37.9, 25.4],
    [29.9, 29.8, 29.7, 27.5, 25.8],
    [52.7, 51.9, 52.4, 49.2, 37.9],
    [50.4, 50.9, 50.0, 49.3, 42.5],
    [182.0, 187.3, 187.5, 188.2, 186.7],
    [182.5, 188.8, 188.8, "failed", 188.6],
    [304.7, 161.5, 326.2, 326.5, 161.7],
    [311.4, "partial", 332.5, 167.1, 330.1],
]

C32_TTFT_MS = [
    [6148.1, 4552.4, 4546.2, 5319.2, 13772.7],
    [6644.0, 5350.7, 5118.1, 5502.0, 7029.9],
    [4876.2, 4276.9, 4437.8, 6692.3, 46293.7],
    [5809.0, 5426.2, 5349.4, 7523.5, 18588.0],
    [1781.3, 971.4, 1155.2, 1276.8, 1376.0],
    [1825.1, 1082.8, 1166.6, "failed", 1420.8],
    [3015.6, 1549.4, 1499.5, 1602.0, 1521.7],
    [3201.9, "partial", 1590.7, 1511.5, 1717.2],
]

PAGE_2 = {
    8: (C8_TPS_PER_GPU, C8_TTFT_MS),
    16: (C16_TPS_PER_GPU, C16_TTFT_MS),
    32: (C32_TPS_PER_GPU, C32_TTFT_MS),
}


def _scale_anchor(tps_anchor: float, ttft_anchor: float, c_anchor: int, c_target: int) -> tuple[float, float]:
    """Scale C=8 anchor to a different concurrency.

    For c_target < c_anchor: throughput scales sub-linearly with concurrency,
    TTFT scales sub-linearly down (queueing time falls).
    Heuristic deliberately simple so the synthetic curves are monotonic and
    visually distinguishable -- the test cares about layout, not realism.
    """
    ratio = c_target / c_anchor
    # tps/GPU rises with concurrency up to saturation. Below C=8 use ratio**0.85
    # to approximate the diminishing-returns shape vLLM produces.
    tps = tps_anchor * (ratio ** 0.85)
    # TTFT below C=8: divide by ratio**0.6 so going from C=8 to C=1 cuts
    # TTFT by ~3.8x. Realistic for prefill-dominated workloads.
    ttft = ttft_anchor * (ratio ** 0.6) if ratio < 1.0 else ttft_anchor
    return tps, ttft


def _derived_metrics(tps_per_gpu: float, concurrency: int, tp: int) -> tuple[float, float]:
    """Derive request_throughput (req/s) and output_tps_per_user from
    tps_per_gpu + concurrency + tp.

    request_throughput_avg ~= total_output_tps / OSL
                           = tps_per_gpu * tp / OSL
    output_tps_per_user    = tps_per_gpu * tp / concurrency
    """
    request_throughput = tps_per_gpu * tp / OSL
    tps_per_user = tps_per_gpu * tp / concurrency
    return request_throughput, tps_per_user


def cell_id_for(variant: Variant, mbt: int) -> str:
    mtp_suffix = "-mtp" if variant.mtp else ""
    return (
        f"{variant.hardware.lower()}-{variant.quant.lower()}-tp{variant.tp}"
        f"-{variant.strategy.lower()}{mtp_suffix}-mbt{mbt}"
    )


def build_rows() -> list[AtlasCell]:
    rows: list[AtlasCell] = []

    for v_idx, variant in enumerate(VARIANTS):
        for m_idx, mbt in enumerate(MBT_VALUES):
            cell_id = cell_id_for(variant, mbt)

            # Determine cell status from anchor data: a "failed" string at C=8
            # means the cell didn't run at all; "partial" at C=16 means the
            # cell completed C=8 but evicted before C=16 finished.
            c8_tps = C8_TPS_PER_GPU[v_idx][m_idx]
            c16_tps = C16_TPS_PER_GPU[v_idx][m_idx]

            if isinstance(c8_tps, str) and c8_tps == "failed":
                # Failed cell: emit one status row at concurrency=0, no metrics.
                rows.append(
                    AtlasCell(
                        cell_id=cell_id,
                        model=MODEL_BY_QUANT[variant.quant],
                        hardware=variant.hardware,
                        quant=variant.quant,
                        tensor_parallel=variant.tp,
                        parallel_strategy=variant.strategy,
                        mtp=variant.mtp,
                        max_num_batched_tokens=mbt,
                        concurrency=0,
                        status=STATUS_FAILED,
                        backend=BACKEND_VLLM_SWEEP,
                        raw_path=f"cells/{cell_id}/raw/",
                        captured_at=CAPTURED_AT,
                        notes="Engine crashed at load; no concurrency points completed.",
                    )
                )
                continue

            if isinstance(c16_tps, str) and c16_tps == "partial":
                # Partial cell: C=8 completed (use that anchor), C=16+ evicted.
                status = STATUS_PARTIAL
                anchor_concurrencies = (1, 2, 4, 8)
            else:
                status = STATUS_FULL
                anchor_concurrencies = CONCURRENCIES  # all 6

            for c in anchor_concurrencies:
                # Pick the anchor: prefer matching concurrency from PAGE_2, else
                # scale from C=8 anchor.
                if c in PAGE_2:
                    tps_grid, ttft_grid = PAGE_2[c]
                    raw_tps = tps_grid[v_idx][m_idx]
                    raw_ttft = ttft_grid[v_idx][m_idx]
                    if isinstance(raw_tps, str) or isinstance(raw_ttft, str):
                        # status string at this (variant, mbt, c); skip
                        continue
                    tps_per_gpu = float(raw_tps)
                    ttft_ms = float(raw_ttft)
                else:
                    tps_per_gpu, ttft_ms = _scale_anchor(
                        float(c8_tps), float(C8_TTFT_MS[v_idx][m_idx]), 8, c
                    )

                req_throughput, tps_per_user = _derived_metrics(
                    tps_per_gpu, c, variant.tp
                )

                rows.append(
                    AtlasCell(
                        cell_id=cell_id,
                        model=MODEL_BY_QUANT[variant.quant],
                        hardware=variant.hardware,
                        quant=variant.quant,
                        tensor_parallel=variant.tp,
                        parallel_strategy=variant.strategy,
                        mtp=variant.mtp,
                        max_num_batched_tokens=mbt,
                        concurrency=c,
                        status=status,
                        ttft_avg_ms=round(ttft_ms, 1),
                        request_throughput_avg=round(req_throughput, 4),
                        output_tps_per_user=round(tps_per_user, 2),
                        output_tps_per_gpu=round(tps_per_gpu, 1),
                        backend=BACKEND_VLLM_SWEEP,
                        raw_path=f"cells/{cell_id}/raw/",
                        captured_at=CAPTURED_AT,
                    )
                )

    # Add 20 evicted MTP context cells (B200/GB300 NVFP4-MTP across 5 mbt x 2 strategies = 20).
    for hw, tp in (("B200", 8), ("GB300", 4)):
        for strategy in ("EP", "TP"):
            for mbt in MBT_VALUES:
                cell_id = (
                    f"{hw.lower()}-nvfp4-tp{tp}-{strategy.lower()}-mtp-mbt{mbt}"
                )
                rows.append(
                    AtlasCell(
                        cell_id=cell_id,
                        model="GLM-5.1-NVFP4",
                        hardware=hw,
                        quant="NVFP4",
                        tensor_parallel=tp,
                        parallel_strategy=strategy,
                        mtp=True,
                        max_num_batched_tokens=mbt,
                        concurrency=0,
                        status=STATUS_EVICTED,
                        backend=BACKEND_VLLM_SWEEP,
                        raw_path=f"cells/{cell_id}/raw/",
                        captured_at=CAPTURED_AT,
                        notes="Evicted before terminal state; failure context, not plotted.",
                    )
                )

    return rows


def main() -> int:
    rows = build_rows()
    for row in rows:
        sys.stdout.write(json.dumps(row.to_dict(), sort_keys=True))
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
