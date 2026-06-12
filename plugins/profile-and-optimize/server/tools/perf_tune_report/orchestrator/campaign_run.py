"""Implementation of the ``perf_tune_report_campaign_run`` orchestrator (Phase 2b).

The CLI surface accepts a matrix YAML of the form::

    campaign:
      name: <slug>
      target_release: <helm-release-name>   # e.g. basic-inference
      target_namespace: <ns>                # defaults to inference
      chart_dir: <path>                     # defaults to workspace chart fork
      base_values: <path>                   # the deploy bundle's my-values-*.yaml
      drain_nodes: [<node>, ...]       # optional; nodes to drain pre-cell
      comparator_baseline: <path>           # optional; default = previous cell

    cells:
      - id: <cell-id>
        # per-cell helm value overrides (merged on top of base_values; can be
        # a path to an overlay yaml OR an inline dict)
        helm_overrides: <path-or-dict>
        # standard cell knobs (concurrencies, traffic shape, dataset, ...)
        concurrencies: [<int>, ...]
        backend: vllm-sweep | aiperf | aa | trtllm (stub)
        # endpoint-only backends (aiperf / aa) carry their config block here;
        # they target an already-running endpoint URL and skip helm + warmup
        aa: {model, url, shape, mode, request_count, api_key_env, ...}
        aiperf: {namespace, bench_pod, kube_context, endpoint_url, served_model, ...}
        # optional per-cell profile flags
        profile:
          zymtrace: on | off
          nsys: short | long | off          # placeholder for Phase 4 wiring

The orchestrator is intentionally a thin coordinator over the existing
per-verb library functions; each step is a separate library call so the
unit tests can stub them out individually.

Safety contract:

- ``submits_jobs`` (highest tier; the orchestrator submits real benchmark
  jobs to the cluster).
- Ack-gated via ``--i-understand-this-submits-jobs``.
- Always-resume on Ctrl-C / exception:
  * Slurm-on-K8s drains are paired with ``try/finally`` resume
  * Helm upgrades carry NO automatic rollback (operator decides via the
    per-cell verdict log written to the campaign's ``commands/`` dir);
    a non-OK verdict aborts subsequent cells (fail-fast unless
    ``--continue-on-red`` is passed)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

# Backends that target an already-running endpoint URL: they do NOT manage a
# helm release, so run_one_cell skips the helm_upgrade + warmup steps for them.
_ENDPOINT_ONLY_BACKENDS = ("aiperf", "aa")


@dataclass
class CellPlan:
    """One cell in the campaign matrix.

    Holds the merged helm overrides + cell metadata; the orchestrator's
    ``run_campaign`` consumes a sequence of these.
    """

    id: str
    backend: str
    concurrencies: tuple[int, ...]
    helm_overrides: dict[str, Any] = field(default_factory=dict)
    profile: dict[str, str] = field(default_factory=dict)
    notes: str = ""
    # Endpoint-only backend config (the cell YAML's ``aa:`` / ``aiperf:`` block).
    # Carries the per-cell endpoint/shape knobs step_bench passes to the runner.
    backend_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.id.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                f"CellPlan.id must be alphanumeric (+ - or _), got {self.id!r}"
            )
        if self.backend not in ("vllm-sweep", "aiperf", "aa", "trtllm"):
            raise ValueError(
                f"CellPlan.backend must be one of vllm-sweep|aiperf|aa|trtllm; "
                f"got {self.backend!r}"
            )
        if not self.concurrencies:
            raise ValueError(f"CellPlan {self.id}: concurrencies must be non-empty")


@dataclass
class CellStepResult:
    """Outcome of a single cell's 10-step pipeline."""

    cell_id: str
    started_at: str
    ended_at: str
    elapsed_s: float
    drain_ok: bool
    helm_ok: bool
    warmup_ok: bool
    bench_ok: bool
    zymtrace_ok: bool
    import_ok: bool
    aggregate_ok: bool
    render_ok: bool
    baseline_record_ok: bool
    baseline_diff_verdict: str  # "GREEN" | "YELLOW" | "RED" | "NA"
    resume_ok: bool  # Slurm-on-K8s resume; always attempted in finally block
    notes: list[str] = field(default_factory=list)


@dataclass
class CampaignRunResult:
    """Final result of a campaign_run invocation."""

    campaign_dir: Path
    cells_attempted: int
    cells_completed: int
    cells_failed: int
    cells_skipped: int
    overall_verdict: str  # "GREEN" | "YELLOW" | "RED" | "NA"
    per_cell: list[CellStepResult] = field(default_factory=list)
    aborted_at_cell: str | None = None  # set when fail-fast triggered
    pdf_path: Path | None = None
    atlas_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["campaign_dir"] = str(self.campaign_dir)
        d["pdf_path"] = str(self.pdf_path) if self.pdf_path else None
        d["atlas_path"] = str(self.atlas_path) if self.atlas_path else None
        for c in d["per_cell"]:
            pass  # already dict-friendly via asdict
        return d


# -----------------------------------------------------------------------------
# Step-runner callable types — exposed so tests can stub each step.
# -----------------------------------------------------------------------------

DrainFn = Callable[[Sequence[str], str], bool]
HelmUpgradeFn = Callable[[str, str, Path, dict[str, Any]], bool]
WarmupFn = Callable[[str, str, str], bool]
BenchFn = Callable[[Path, CellPlan], bool]
ZymtraceFn = Callable[[Path, str], bool]
ImportFn = Callable[[Path, Path, str], bool]
AggregateFn = Callable[[Path], bool]
RenderFn = Callable[[Path, Path], bool]
BaselineRecordFn = Callable[[Path, str], bool]
BaselineDiffFn = Callable[[Path, str, str], str]  # returns verdict string
ResumeFn = Callable[[Sequence[str], str], bool]
DcgmCorrelateFn = Callable[[Path, str], bool]  # (campaign_dir, cell_id) -> wrote dcgm_correlation.json


@dataclass
class StepFns:
    """Bundle of injectable step functions. The default
    ``run_campaign(...)`` builds the production set; tests pass in stubs."""

    drain: DrainFn
    helm_upgrade: HelmUpgradeFn
    warmup: WarmupFn
    bench: BenchFn
    zymtrace: ZymtraceFn
    import_bundle: ImportFn
    aggregate: AggregateFn
    render: RenderFn
    baseline_record: BaselineRecordFn
    baseline_diff: BaselineDiffFn
    resume: ResumeFn
    # Byte-grounding step (DCGM workload-level SoL -> dcgm_correlation.json).
    # Defaulted so existing StepFns(...) constructions (incl. tests) keep
    # working; when None the pipeline logs a loud skip instead of running it.
    dcgm_correlate: DcgmCorrelateFn | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _verdict_rollup(per_cell: Sequence[CellStepResult]) -> str:
    """Combine per-cell verdicts into an overall campaign verdict."""
    if not per_cell:
        return "NA"
    verdicts = {c.baseline_diff_verdict for c in per_cell}
    if "RED" in verdicts:
        return "RED"
    if "YELLOW" in verdicts:
        return "YELLOW"
    if verdicts == {"GREEN"}:
        return "GREEN"
    if verdicts <= {"GREEN", "NA"}:
        return "GREEN" if "GREEN" in verdicts else "NA"
    return "YELLOW"


def run_one_cell(
    cell: CellPlan,
    *,
    campaign_dir: Path,
    target_namespace: str,
    target_release: str,
    chart_dir: Path,
    base_values: Path,
    drain_nodes: Sequence[str],
    comparator_baseline: str,
    step_fns: StepFns,
    dry_run: bool = False,
) -> CellStepResult:
    """Execute the 10-step pipeline for one cell.

    Always-resume contract enforced via try/finally around the Slurm-on-K8s drain.
    """
    started = time.monotonic()
    started_iso = _utc_now_iso()
    notes: list[str] = []

    drain_ok = False
    helm_ok = warmup_ok = bench_ok = zymtrace_ok = False
    import_ok = aggregate_ok = render_ok = False
    baseline_record_ok = False
    verdict = "NA"
    resume_ok = False

    # Single exit point — all step state collected into locals; we build the
    # CellStepResult ONCE after the try/finally so resume_ok reflects the
    # actual outcome of the finally block (early `return` inside the try
    # would otherwise capture stale resume_ok=False).
    aborted_early = False
    try:
        # Step 1: drain Slurm-on-K8s co-tenants (if any).
        if drain_nodes:
            drain_ok = step_fns.drain(drain_nodes, cell.id)
            if not drain_ok and not dry_run:
                notes.append(
                    f"drain returned False for nodes={list(drain_nodes)}; "
                    "aborting cell to avoid running on contended hardware"
                )
                aborted_early = True
        else:
            drain_ok = True  # no nodes to drain; treat as no-op pass

        endpoint_only = cell.backend in _ENDPOINT_ONLY_BACKENDS

        if not aborted_early:
            # Step 2: helm upgrade with the cell's overrides. Endpoint-only
            # backends (aiperf / aa) target an already-running endpoint URL and
            # do not manage a helm release, so the deploy step is a no-op pass.
            if endpoint_only:
                helm_ok = True
                notes.append(
                    f"backend={cell.backend} targets an existing endpoint; "
                    "skipping helm_upgrade"
                )
            else:
                helm_ok = step_fns.helm_upgrade(
                    target_namespace, target_release, chart_dir, cell.helm_overrides
                )
                if not helm_ok:
                    notes.append("helm_upgrade returned False; aborting cell")
                    aborted_early = True

        if not aborted_early:
            # Step 3: warmup probe (skipped for endpoint-only backends).
            if endpoint_only:
                warmup_ok = True
            else:
                warmup_ok = step_fns.warmup(target_namespace, target_release, cell.id)
                if not warmup_ok:
                    notes.append("warmup returned False; continuing to bench anyway")

            # Step 4: cell_run (bench).
            bench_ok = step_fns.bench(campaign_dir, cell)
            if not bench_ok:
                notes.append("bench returned False; aborting cell")
                aborted_early = True

        if not aborted_early:
            # Step 5: zymtrace anchored query (best-effort).
            zymtrace_enabled = cell.profile.get("zymtrace", "on") == "on"
            if zymtrace_enabled:
                zymtrace_ok = step_fns.zymtrace(campaign_dir, cell.id)
                if not zymtrace_ok:
                    notes.append("zymtrace returned False; not blocking cell")

            # Step 6: import_perf_bench (idempotent — converts raw to normalized).
            import_ok = step_fns.import_bundle(
                campaign_dir / "cells" / cell.id, campaign_dir, cell.id
            )

            # Step 7: atlas_aggregate (campaign-level rollup).
            aggregate_ok = step_fns.aggregate(campaign_dir)

            # Step 7b: dcgm_correlate (byte-grounding -> dcgm_correlation.json,
            # renderer pages 6/6b). MUST run before render so the PDF + lake row
            # carry dcgm_grounded=true. Loud-skip (note) when no DCGM input.
            if step_fns.dcgm_correlate is not None:
                dcgm_ok = step_fns.dcgm_correlate(campaign_dir, cell.id)
                if not dcgm_ok:
                    notes.append(
                        "dcgm_correlate produced no dcgm_correlation.json for "
                        f"cell {cell.id} (no cells/{cell.id}/dcgm-frozen.yaml or "
                        "ceilings not found) -> campaign will be dcgm_grounded=FALSE. "
                        "Drop a dcgm_frozen_v1 YAML in the cell dir (capture DCGM "
                        "over the bench window) and re-run for byte-grounding."
                    )
            else:
                notes.append(
                    "dcgm_correlate step not wired (StepFns.dcgm_correlate=None) "
                    "-> campaign will be dcgm_grounded=FALSE unless grounded out-of-band."
                )

            # Step 8: report_render (re-render PDF after each cell — important
            # for the safe-Ctrl-C contract).
            render_ok = step_fns.render(
                campaign_dir, campaign_dir / f"{cell.id}-report.pdf"
            )

            # Step 9: baseline_record.
            baseline_record_ok = step_fns.baseline_record(campaign_dir, cell.id)

            # Step 10: baseline_diff (returns verdict string).
            verdict = step_fns.baseline_diff(
                campaign_dir, cell.id, comparator_baseline
            )

    finally:
        # CRITICAL: always-resume on Slurm-on-K8s drain. This runs even on Ctrl-C
        # via SIGINT (KeyboardInterrupt is caught by the implicit Python
        # bytecode-level try/finally machinery).
        if drain_nodes:
            try:
                resume_ok = step_fns.resume(drain_nodes, cell.id)
                if not resume_ok:
                    notes.append(
                        f"resume returned False for nodes={list(drain_nodes)}; "
                        "Slurm-on-K8s partition may be left in DRAIN state — operator must "
                        "manually `scontrol update state=RESUME`!"
                    )
            except Exception as exc:
                notes.append(f"resume raised: {type(exc).__name__}: {exc}")
                resume_ok = False
        else:
            resume_ok = True

    return _make_cell_result(
        cell, started, started_iso, drain_ok, helm_ok,
        warmup_ok, bench_ok, zymtrace_ok, import_ok,
        aggregate_ok, render_ok, baseline_record_ok,
        verdict, resume_ok, notes,
    )


def _make_cell_result(
    cell, started, started_iso, drain_ok, helm_ok, warmup_ok, bench_ok,
    zymtrace_ok, import_ok, aggregate_ok, render_ok, baseline_record_ok,
    verdict, resume_ok, notes,
) -> CellStepResult:
    return CellStepResult(
        cell_id=cell.id,
        started_at=started_iso,
        ended_at=_utc_now_iso(),
        elapsed_s=round(time.monotonic() - started, 2),
        drain_ok=drain_ok,
        helm_ok=helm_ok,
        warmup_ok=warmup_ok,
        bench_ok=bench_ok,
        zymtrace_ok=zymtrace_ok,
        import_ok=import_ok,
        aggregate_ok=aggregate_ok,
        render_ok=render_ok,
        baseline_record_ok=baseline_record_ok,
        baseline_diff_verdict=verdict,
        resume_ok=resume_ok,
        notes=list(notes),
    )


def run_campaign(
    cells: Sequence[CellPlan],
    *,
    campaign_dir: Path,
    target_namespace: str = "inference",
    target_release: str,
    chart_dir: Path,
    base_values: Path,
    drain_nodes: Sequence[str] = (),
    comparator_baseline: str = "",
    step_fns: StepFns,
    continue_on_red: bool = False,
    dry_run: bool = False,
) -> CampaignRunResult:
    """Run a campaign: iterate cells, executing the 10-step pipeline each.

    Fail-fast contract: a RED verdict aborts subsequent cells unless
    ``continue_on_red=True``. Always-resume on Ctrl-C / exception via the
    per-cell ``try/finally`` block in ``run_one_cell``.

    Writes per-step JSON receipts under ``<campaign_dir>/commands/`` for
    operator review.
    """
    campaign_dir = Path(campaign_dir).expanduser().resolve()
    if not campaign_dir.is_dir():
        raise ValueError(f"campaign_dir does not exist: {campaign_dir}")
    commands_dir = campaign_dir / "commands"
    commands_dir.mkdir(exist_ok=True)

    per_cell: list[CellStepResult] = []
    aborted_at: str | None = None

    for cell in cells:
        result = run_one_cell(
            cell,
            campaign_dir=campaign_dir,
            target_namespace=target_namespace,
            target_release=target_release,
            chart_dir=chart_dir,
            base_values=base_values,
            drain_nodes=drain_nodes,
            comparator_baseline=comparator_baseline,
            step_fns=step_fns,
            dry_run=dry_run,
        )
        per_cell.append(result)

        # Persist per-cell receipt
        receipt = commands_dir / f"cell-{cell.id}-receipt.json"
        receipt.write_text(json.dumps(asdict(result), indent=2))

        if result.baseline_diff_verdict == "RED" and not continue_on_red:
            aborted_at = cell.id
            break

    cells_completed = sum(
        1 for r in per_cell if r.bench_ok and r.import_ok and r.aggregate_ok
    )
    cells_failed = sum(
        1 for r in per_cell if not r.bench_ok or not r.import_ok
    )

    return CampaignRunResult(
        campaign_dir=campaign_dir,
        cells_attempted=len(per_cell),
        cells_completed=cells_completed,
        cells_failed=cells_failed,
        cells_skipped=len(cells) - len(per_cell),
        overall_verdict=_verdict_rollup(per_cell),
        per_cell=per_cell,
        aborted_at_cell=aborted_at,
        pdf_path=campaign_dir / "report.pdf",
        atlas_path=campaign_dir / "atlas.jsonl",
    )
