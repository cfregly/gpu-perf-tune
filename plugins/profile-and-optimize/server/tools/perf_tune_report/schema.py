"""Canonical atlas-cell schema for perf-report.

One JSONL row per ``(cell, concurrency)`` point. The same schema is produced
by both runners (``vllm_sweep`` and ``aiperf_bench``) and consumed by both
renderer pages (scatter-grid + heatmap-tables) plus the coverage block.

Status enum:

- ``full``     -- cell ran every requested concurrency point cleanly
- ``partial``  -- cell ran some concurrencies but not all (e.g. evicted late)
- ``failed``   -- cell could not run any concurrency (engine crash, OOM at load)
- ``evicted``  -- cell was killed mid-sweep before reaching terminal state

For ``failed`` / ``evicted`` cells the metric fields are ``None``; the renderer
treats them specially (gray cells in the heatmap, omitted from scatter).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any, Iterable

STATUS_FULL = "full"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"
STATUS_EVICTED = "evicted"
STATUSES = frozenset({STATUS_FULL, STATUS_PARTIAL, STATUS_FAILED, STATUS_EVICTED})

BACKEND_VLLM_SWEEP = "vllm-sweep"
BACKEND_SGLANG_SWEEP = "sglang-sweep"   # cross-engine A/B (SGLang arm), same client/parser as vllm-sweep
BACKEND_AIPERF = "aiperf"
BACKEND_TRTLLM = "trtllm"   # Added v1.20.0 as stub (not yet implemented)
BACKENDS = frozenset({BACKEND_VLLM_SWEEP, BACKEND_SGLANG_SWEEP, BACKEND_AIPERF, BACKEND_TRTLLM})

# serving_engine is the normalized serving engine the bench exercised. The
# backend already encodes the engine arm, so serving_engine is derived from it:
# vllm-sweep -> vllm, sglang-sweep -> sglang, trtllm -> trtllm. aiperf is a
# load-generator client (front-ends any engine), so it maps to "" (unknown).
BACKEND_TO_ENGINE: dict[str, str] = {
    BACKEND_VLLM_SWEEP: "vllm",
    BACKEND_SGLANG_SWEEP: "sglang",
    BACKEND_TRTLLM: "trtllm",
    BACKEND_AIPERF: "",
}


def engine_for_backend(backend: str) -> str:
    """Normalize a bench ``backend`` to the serving engine it exercised.

    Returns "" for unknown/empty backends and for the aiperf load-gen client.
    """
    return BACKEND_TO_ENGINE.get(backend or "", "")


# Full-context descriptor: string fields that MUST be populated (not "unknown") on a
# MEASURED atlas row, per CLAUDE.md "Every performance number carries its full context
# (no bare numbers)" (rule docs/METHODOLOGY.md). The methodology gate
# (``lake_writer.methodology_problems``) flags any field still at "unknown"; publish/render
# ``--strict`` fails closed. ``gpu_memory_utilization`` (None sentinel) is checked alongside.
REQUIRED_CONTEXT_STR_FIELDS: tuple[str, ...] = (
    "dataset",
    "cudagraph_mode",
    "kv_cache_dtype",
    "image",
)


@dataclass
class AtlasCell:
    """One (cell, concurrency) measurement point.

    Required identity fields (the renderer keys plots and legend on these):
    ``cell_id``, ``model``, ``hardware``, ``quant``, ``tensor_parallel``,
    ``parallel_strategy``, ``mtp``, ``max_num_batched_tokens``, ``concurrency``.

    Required status: ``status`` (one of ``STATUSES``).

    Metric fields are ``Optional[float]`` because cells with status in
    ``{failed, evicted}`` have no measurements at the failed concurrency
    points. The renderer skips ``None`` metrics in scatter plots and shows
    them as ``failed`` / ``partial`` in the heatmap tables.

    Provenance: ``backend``, ``raw_path``, ``captured_at``.
    """

    cell_id: str
    model: str
    hardware: str
    quant: str
    tensor_parallel: int
    parallel_strategy: str
    mtp: bool
    max_num_batched_tokens: int
    concurrency: int
    status: str

    ttft_avg_ms: float | None = None
    request_throughput_avg: float | None = None
    output_tps_per_user: float | None = None
    output_tps_per_gpu: float | None = None
    # Total (input+output) token throughput per GPU (added v1.35.0): the
    # "Total token throughput (tok/s)" bench line divided by tensor_parallel.
    # Persisted so the TPM-across-hardware rollup can report total-TPM (the
    # OpenAI/Azure TPM convention) alongside output-only TPM. None for backends
    # that do not emit a total-token line (aiperf/aa/drive_load) -> total-TPM
    # renders n/a downstream.
    total_tps_per_gpu: float | None = None
    # Decode-latency metrics (added v1.33.0) so a consistent schema is recorded
    # every run regardless of focus. TPOT is the decode-latency headline (the
    # importer already parses it); persisting it makes latency-focused runs
    # first-class in the atlas/lake (not derivable-only via output_tps_per_user).
    tpot_median_ms: float | None = None
    itl_avg_ms: float | None = None
    # Reasoning-model first-token split (added 2026-06-09). AIPerf reports TTFT
    # (``time_to_first_token`` = first token of ANY type, including reasoning_content)
    # AND TTFO (``time_to_first_output_token`` = first non-reasoning / answer token).
    # For a reasoning model the two diverge by the think-phase duration, so capturing
    # ONLY ttft_avg_ms under-reports answer latency (the minimax-aabench miss that hid a
    # ~4 s think phase behind a 0.10 s "TTFT"). None for non-reasoning models / backends
    # that do not expose a separate reasoning_content field (then TTFO == TTFT).
    ttfo_avg_ms: float | None = None
    reasoning_token_count: float | None = None
    # Fraction of measured requests that emitted ANY answer token (AIPerf
    # ttfo.count / ttft.count). ttfo_avg_ms is averaged over ONLY those
    # requests; reasoning_token_count over ALL. On synthetic filler a
    # reasoning model can exhaust the whole output budget thinking, so a
    # low-coverage TTFO is an answered-subset stat, not typical answer latency.
    ttfo_coverage: float | None = None

    # Analysis-carry-through fields (added v1.42.0). All nullable/defaulted so
    # older JSONL still parses.
    # Mean per-request input / output sequence length (ISL/OSL) -- the dominant
    # pricing/capacity analysis dimension. vllm-sweep derives these from the
    # "Total input tokens" / "Total generated tokens" bench lines / successful
    # requests; aiperf/drive_load leave them None when unavailable.
    mean_input_tokens: float | None = None
    mean_output_tokens: float | None = None
    # Prefix-cache hit rate (0..1), sourced best-effort from the bundle's
    # inference_perfbench_v1.json; None when the bundle did not record it.
    prefix_cache_hit_rate: float | None = None
    # Warm-vs-cold methodology label: "warm" (cache-primed / sweep-tail) |
    # "cold" (fresh / single-shot) | "unknown". A DECLARED label (no bench
    # signal); set via the importer --cache-mode override or bundle metadata.
    cache_mode: str = "unknown"

    # Full-context descriptor fields (added 2026-06-07, CLAUDE.md "Every performance
    # number carries its full context (no bare numbers)" / rule docs/METHODOLOGY.md).
    # All defaulted so older JSONL still parses; the methodology gate flags any
    # "unknown"/None on a MEASURED row (fail-closed under publish/render --strict).
    dataset: str = "unknown"            # random | sharegpt | sonnet | aa | code | ...
    cudagraph_mode: str = "unknown"     # full | piecewise | none | eager (the eager/cudagraph trap)
    gpu_memory_utilization: float | None = None
    kv_cache_dtype: str = "unknown"     # fp8_e4m3 | bf16 | nvfp4 | ...
    image: str = "unknown"              # serving image tag / vllm commit (the stack the number is from)
    data_parallel: int = 1              # DP replica count (1 = single instance)
    pipeline_parallel: int = 1          # PP size (1 = none)

    # Serving-variant descriptor (added 2026-06-07): the per-variant knobs needed to answer
    # "which variant + why" and to build a stable cross-campaign variant_key. All
    # nullable/optional so older JSONL still parses AND so they never fail-close the
    # methodology gate (they are not always known/applicable on every run).
    num_speculative_tokens: int | None = None   # MTP / EAGLE K (the ``mtp`` bool is on/off; this is the K value)
    async_scheduling: bool | None = None        # --async-scheduling (host/decode overlap)
    max_num_seqs: int | None = None             # --max-num-seqs (decode batch cap; was extra-only)
    enable_prefix_caching: bool | None = None   # prefix-caching on/off (distinct from prefix_cache_hit_rate)
    bench_backend: str = ""                      # bench CLIENT (vllm | openai), distinct from `backend` (the sweep runner)
    variant_key: str = ""                        # stable serving-variant hash (capture_signature); populated at publish/aggregate
    # Ledger-to-atlas data-capture gaps (added 2026-06-07; nullable/defaulted so older
    # JSONL parses). First-class capture for value-findings dimensions the atlas otherwise
    # only carried in ``extra`` -- see perf-tune-report/UPSTREAM-REQUEST-atlas-ledger-
    # datacapture-gaps.md. None/"" on rows where the lever does not apply.
    # Routing / load-balancer A/B descriptor (the KV-router findings).
    router_policy: str = ""             # round-robin | prefix-affinity | cache_aware | hybrid | adaptive | ""
    prefix_reuse: float | None = None   # prefix-reuse level swept in a routing A/B
    per_replica_cache_hit: float | None = None  # per-replica prefix-cache hit (0..1)
    # Spec-decode acceptance length (the RESULT; the K lever is num_speculative_tokens).
    acceptance_length: float | None = None
    # Spec-decode per-position acceptance rate = accepted / draft_tokens over the same
    # window (AL = 1 + accepted/drafts). Distinct axes: AL is k-monotone, accept rate
    # is not, so both are needed to compare draft heads trained at different k.
    # None when spec decode is off / not scraped.
    spec_accept_rate: float | None = None
    # Engine-init KV-cache token capacity (the nvfp4-kv "more KV tokens" headline).
    kv_cache_tokens: int | None = None
    # DeepEP / expert-parallel mode (sub-discriminator of parallel_strategy=EP).
    ep_mode: str = ""                   # deepep-ll | deepep-ht | none | ""
    # Per-(cell,concurrency) DCGM utilization (0..1), promoted from campaign-level
    # dcgm_correlation.json so utilization-vs-concurrency is a flat atlas read.
    dcgm_sm_active: float | None = None
    dcgm_dram_active: float | None = None
    dcgm_tensor_active: float | None = None

    backend: str = ""
    # Serving engine, normalized from backend (vllm / sglang / trtllm; "" for the
    # aiperf load-gen client or an unset backend). Auto-derived from backend in
    # __post_init__ when not explicitly set.
    serving_engine: str = ""
    raw_path: str = ""
    captured_at: str = ""

    # Optional extras the renderer ignores but operators sometimes want.
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(
                f"AtlasCell.status must be one of {sorted(STATUSES)}, got {self.status!r}"
            )
        if self.parallel_strategy not in ("EP", "TP"):
            raise ValueError(
                f"AtlasCell.parallel_strategy must be 'EP' or 'TP', got {self.parallel_strategy!r}"
            )
        if self.backend and self.backend not in BACKENDS:
            raise ValueError(
                f"AtlasCell.backend must be one of {sorted(BACKENDS)} or empty, got {self.backend!r}"
            )
        # Derive serving_engine from backend when not explicitly provided.
        if not self.serving_engine:
            self.serving_engine = engine_for_backend(self.backend)
        if self.cache_mode not in ("warm", "cold", "unknown"):
            raise ValueError(
                f"AtlasCell.cache_mode must be 'warm', 'cold', or 'unknown', got {self.cache_mode!r}"
            )

    @property
    def has_metrics(self) -> bool:
        """True iff this row carries a plot-ready (throughput-scatter) measurement."""
        return self.ttft_avg_ms is not None and self.request_throughput_avg is not None

    @property
    def has_latency_metrics(self) -> bool:
        """True iff this row carries a decode-latency measurement (TPOT or ITL
        or TTFT). A focus=latency run (e.g. c=1 decode / kernel probe) is a
        first-class published result even without request throughput, so the
        publish path counts these via ``plot_ready_latency_points``."""
        return (
            self.tpot_median_ms is not None
            or self.itl_avg_ms is not None
            or self.ttft_avg_ms is not None
        )

    @property
    def legend_key(self) -> tuple[str, str, int, str, bool]:
        """The grouping key the renderer uses to assign one curve / one
        legend entry per ``(hw, quant, TP, EP/TP, MTP)`` combo."""
        return (
            self.hardware,
            self.quant,
            self.tensor_parallel,
            self.parallel_strategy,
            self.mtp,
        )

    @property
    def legend_label(self) -> str:
        """Human-readable label used in the matplotlib legend."""
        mtp_suffix = " MTP" if self.mtp else ""
        return (
            f"{self.hardware} {self.quant}{mtp_suffix} "
            f"TP={self.tensor_parallel} {self.parallel_strategy}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_jsonl(rows: Iterable[AtlasCell], out: IO[str] | Path) -> int:
    """Write rows as one-JSON-object-per-line. Returns the row count."""
    count = 0
    if isinstance(out, Path):
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row.to_dict(), sort_keys=True))
                f.write("\n")
                count += 1
    else:
        for row in rows:
            out.write(json.dumps(row.to_dict(), sort_keys=True))
            out.write("\n")
            count += 1
    return count


def read_jsonl(src: IO[str] | Path) -> list[AtlasCell]:
    """Read a JSONL file into AtlasCell rows. Blank lines and ``#``-comment
    lines are skipped so fixtures can be lightly annotated."""

    def _iter_lines(handle: IO[str]) -> Iterable[str]:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            yield line

    rows: list[AtlasCell] = []
    if isinstance(src, Path):
        with src.open("r", encoding="utf-8") as f:
            lines = list(_iter_lines(f))
    else:
        lines = list(_iter_lines(src))

    for line in lines:
        data = json.loads(line)
        rows.append(AtlasCell(**data))
    return rows
