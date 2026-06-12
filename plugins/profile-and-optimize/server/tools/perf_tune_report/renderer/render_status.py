"""Single source of truth for the renderer's "why + how to fix" messaging.

The renderer never silently drops a conditional page or draws a blank
chart. When a Speed-of-Light / kernel / DCGM page cannot be drawn (its
input data is absent), the renderer draws a visible OMITTED placeholder
page carrying the ``why`` (what input was missing) and ``how_to_fix``
(the exact next step) strings defined here. The same strings power the
loud stderr warnings and the machine-readable ``report_status.json``.

Keeping the strings in one module means the PDF placeholder, the CLI
warning, and the JSON envelope can never drift apart.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class OmissionReason:
    """Why a page was omitted (or a chart is empty) + how to populate it."""

    page: str
    why: str
    how_to_fix: str

    def as_dict(self) -> dict[str, str]:
        return {"page": self.page, "why": self.why, "how_to_fix": self.how_to_fix}


# Canonical (why, how_to_fix) per conditional page. Keyed by a stable page
# id used in report_status.json + the placeholder page title.
OMISSION_REASONS: dict[str, OmissionReason] = {
    "kernel_breakdown": OmissionReason(
        page="GPU kernel breakdown (page 3)",
        why=(
            "No kernels.json found under cells/*/ -- this campaign ran "
            "without zymtrace GPU profiling, so the per-kernel GPU-time "
            "breakdown cannot be computed."
        ),
        how_to_fix=(
            "Re-run the cell with zymtrace capture (analyze-zymtrace-workload "
            "skill), import it with the zymtrace_kernels importer so each "
            "cell carries a kernels.json, then re-render."
        ),
    ),
    "sol_roofline": OmissionReason(
        page="Speed-of-Light roofline (page 4)",
        why=(
            "No kernels.json under cells/*/ (or no matching sol-ceilings.yaml "
            "for this atlas's hardware) -- the per-category Speed-of-Light "
            "roofline layers on zymtrace per_category data that is absent."
        ),
        how_to_fix=(
            "Capture zymtrace per_category data for at least one cell and "
            "ensure configs/sol-ceilings.yaml has a hardware key "
            "matching this atlas, then re-render."
        ),
    ),
    "ncu_scatter": OmissionReason(
        page="Byte-grounded per-kernel SoL scatter (page 5)",
        why=(
            "No ncu_kernels.json under cells/*/ -- this campaign did not "
            "capture an Nsight-Compute per-kernel profile, so the "
            "arithmetic-intensity-vs-roofline scatter cannot be drawn."
        ),
        how_to_fix=(
            "Run the inference-kernel-ncu-profile skill to emit ncu_kernels.json "
            "for a cell (note the TP=8 NVFP4 kernel-replay blocker documented "
            "in that skill), then re-render."
        ),
    ),
    "dcgm_sol": OmissionReason(
        page="DCGM workload-level SoL (page 6)",
        why=(
            "No dcgm_correlation.json under cells/*/ -- DCGM byte/FLOP "
            "workload-level cross-attribution was not captured."
        ),
        how_to_fix=(
            "Run dcgm_correlate (perf_tune_report.dcgm_correlate) against the "
            "workload's DCGM byte-traffic + the atlas to emit "
            "dcgm_correlation.json, then re-render."
        ),
    ),
    "dcgm_xattr": OmissionReason(
        page="zymtrace x DCGM cross-attribution (page 6b)",
        why=(
            "dcgm_correlation.json is present but carries no "
            "per_category_attribution block -- the Level-2 cross-attribution "
            "needs correlate() to be invoked with a kernels_json_path."
        ),
        how_to_fix=(
            "Re-run dcgm_correlate with --kernels-json pointing at the cell's "
            "zymtrace kernels.json so the per_category_attribution block is "
            "populated, then re-render."
        ),
    ),
    "tpm_table": OmissionReason(
        page="TPM supported across hardware types",
        why=(
            "No atlas row carries output_tps_per_gpu, so per-hardware "
            "tokens-per-minute capacity cannot be rolled up."
        ),
        how_to_fix=(
            "Ensure the bench output includes 'Output token throughput "
            "(tok/s)', re-import the bundle, then re-aggregate + re-render."
        ),
    ),
    "prefill_decode_roofline": OmissionReason(
        page="Prefill/Decode roofline (page 7)",
        why=(
            "No roofline_sweep.json under cells/*/ (or no matching "
            "sol-ceilings.yaml for this atlas's hardware) -- the phase-separated "
            "prefill/decode roofline + per-(c,ISL) DCGM utilization was not captured."
        ),
        how_to_fix=(
            "Run *-deploy/profiling/roofline-sweep.sh against the deploy "
            "(decode concurrency + prefill ISL sweep with in-pod dcgmi), import "
            "it with perf_tune_report import_roofline_sweep so each config carries a "
            "roofline_sweep.json, then re-render."
        ),
    ),
    "champion_select": OmissionReason(
        page="Champion selection (page 8)",
        why=(
            "No champion_select.json in the campaign dir -- the baseline-vs-top-X "
            "production-choice synthesis (cross-engine ranking + 4-layer SoL + "
            "overlaid roofline + DRAFT/VERDICT recommendation) was not computed."
        ),
        how_to_fix=(
            "Run `perftunereport champion_select --campaign <id>` (after import + "
            "atlas_aggregate + dcgm_correlate/import_roofline_sweep) to emit "
            "champion_select.json + CHAMPION.md, then re-render."
        ),
    ),
    "scatter_empty": OmissionReason(
        page="Latency-vs-throughput scatter (page 1)",
        why=(
            "Every atlas row is missing ttft_avg_ms and/or "
            "request_throughput_avg, so no concurrency point is plottable "
            "(0 plot-ready points)."
        ),
        how_to_fix=(
            "Ensure the bench output includes both 'Median TTFT (ms)' and "
            "'Request throughput (req/s)' lines, re-import the bundle, then "
            "re-aggregate + re-render."
        ),
    ),
}


# Canonical (why, how_to_fix) per PARTIAL page. A partial page DID render, but
# carries less than a full measurement -- the report must not be read as
# complete. Keyed by a stable id used in report_status.json + the completeness
# page (mirrors OMISSION_REASONS for the absent-page case).
PARTIAL_REASONS: dict[str, OmissionReason] = {
    "ncu_scatter_solonly": OmissionReason(
        page="Byte-grounded per-kernel SoL scatter (page 5)",
        why=(
            "Page 5 rendered but carries only %SoL-only markers: the ncu "
            "artifact did not provide usable FLOPS / DRAM-byte counters for "
            "arithmetic intensity (common causes: --set=basic, a replay fallback "
            "that captured only kernels with null FLOP counters, or partial "
            "ContextSaveFailed output). The markers' x-position is a placeholder "
            "(ridge AI), not a measurement -- only %SoL (SM throughput) is real. "
            "Do NOT read this as a full roofline."
        ),
        how_to_fix=(
            "Re-capture with --roofline-min (or --set full) per the ncu-sister "
            "REPLAY-MODE-APPLICATION-RUNBOOK.md, re-import with import_ncu, then "
            "re-render for a true arithmetic-intensity roofline."
        ),
    ),
    "ncu_scatter_empty": OmissionReason(
        page="Byte-grounded per-kernel SoL scatter (page 5)",
        why=(
            "Page 5 rendered but has NO roofline-ready points: ncu_kernels.json "
            "carries all-null arithmetic intensity / achieved TFLOPS and no "
            "dcgm_correlation.json with per_category_attribution was found."
        ),
        how_to_fix=(
            "Re-capture ncu with --roofline-min / --set full (see the ncu-sister "
            "REPLAY-MODE-APPLICATION-RUNBOOK.md), or add DCGM per_category "
            "attribution, then re-render."
        ),
    ),
}


@dataclass
class RenderStatus:
    """Machine-readable outcome of one render_report() invocation.

    Serialized to ``<campaign_dir>/report_status.json`` and surfaced in the
    report_render CLI JSON envelope. ``sol_complete`` is the single flag the
    publish_to_lake gate keys on.
    """

    rendered_pages: list[str] = field(default_factory=list)
    omitted_pages: list[dict[str, str]] = field(default_factory=list)
    # Pages that DID render but carry less than a full measurement (e.g. a
    # page-5 %SoL-only scatter with arithmetic intensity unmeasured). Recorded
    # so the loud completeness page + the perf-lake campaign_v1.partial_pages
    # column flag the limitation -- the report is not read as fully complete.
    partial_pages: list[dict[str, str]] = field(default_factory=list)
    plot_ready_points: int = 0
    non_plot_ready_full_cells: int = 0
    sol_complete: bool = True
    # focus (added v1.33.0): the run's intent -- "latency" | "throughput" |
    # "mixed". Recorded so latency-focused runs (c=1 decode, kernel probes) are
    # first-class published results, not "drafts". Set from the campaign
    # config.yaml `focus:` key; defaults "mixed".
    focus: str = "mixed"
    # sol_rigor (added v1.33.0): which Speed-of-Light evidence levels are
    # present, highest first -- "L4" (ncu per-kernel arithmetic intensity),
    # "L3" (DCGM byte/FLOP), "L1" (zymtrace sample-share proxy), or "none".
    # The proxy-vs-tight distinction is now a RECORDED field, never a publish
    # blocker: a campaign always publishes; sol_rigor says how tight its SoL is.
    sol_rigor: str = "none"
    # dcgm_grounded is True only when the DCGM workload-level byte/FLOP SoL
    # page (page 6) rendered -- i.e. at least one cell carries a
    # dcgm_correlation.json. It is the L2/L3 byte-grounding analog of the
    # L1 sol_complete (roofline) flag: a campaign can be sol_complete=True
    # off zymtrace alone yet dcgm_grounded=False (no Prometheus byte
    # cross-attribution), which is exactly the silent gap this flag closes.
    dcgm_grounded: bool = True
    # PER-ARM Speed-of-Light coverage (added v1.68.0). sol_complete above is
    # CAMPAIGN-level ("any SoL page rendered"), so a multi-arm campaign whose
    # baseline carries a roofline but whose variants do NOT still reads
    # complete. These fields make the per-variant gap explicit + machine-
    # readable: the publish_to_lake gate, the teardown hook, and the
    # sol-coverage audit all key on them so "baseline + EACH variant has a
    # roofline" is enforced, not just "some arm does". An arm == an atlas
    # cell (its -decode/-prefill roofline shards collapse to the base arm).
    # ``sol_per_arm_complete`` defaults True so a single-arm or no-arm render
    # is trivially complete; it is set False only when arms_uncovered is
    # non-empty.
    arms_total: int = 0
    arms_with_roofline: int = 0
    arms_uncovered: list[str] = field(default_factory=list)
    sol_per_arm_complete: bool = True

    def omit(self, key: str) -> None:
        """Record an omission by its OMISSION_REASONS key."""
        reason = OMISSION_REASONS[key]
        self.omitted_pages.append(reason.as_dict())

    def mark_partial(self, key: str) -> None:
        """Record a partial (rendered-but-limited) page by its PARTIAL_REASONS key."""
        reason = PARTIAL_REASONS[key]
        self.partial_pages.append(reason.as_dict())

    def to_dict(self) -> dict:
        return asdict(self)
