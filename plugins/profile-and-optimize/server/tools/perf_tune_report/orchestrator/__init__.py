"""Campaign-level orchestrator (Phase 2b).

Wraps the per-cell verbs into a single ``perf_tune_report_campaign_run`` call
that loops over a matrix YAML and, for each cell sequentially:

  1. drain-quiet-window  (when Slurm-on-K8s co-tenant nodes are present)
  2. helm upgrade        (per-variant values overlay)
  3. warmup              (1-shot small sweep to trigger cudagraph capture)
  4. cell_run            (vllm_sweep or aiperf)
  5. zymtrace anchored   (per-kernel breakdown via the v1.19.0 importer)
  6. import_perf_bench   (bridges raw sweep into normalized.json)
  7. atlas_aggregate     (per-campaign rollup)
  8. report_render       (re-rendered PDF after each cell — so the campaign
                          can be safely Ctrl-C'd at any point and the
                          most-recent PDF reflects the cells completed)
  9. baseline_record     (per-cell perf-baseline registry entry)
  10. baseline_diff      (vs prior-cell or operator-named comparator)

Always-resume contract: every cell's helm upgrade is paired with a
``try/finally`` block that ensures the Slurm-on-K8s drain is RESUMED even on
Ctrl-C / exception / non-zero inner-cmd exit. This is the same contract
the bundled ``slurm_quiet_window`` MCP verb honors (v1.17.0).

Added in v1.20.0. Backs the ``perf_tune_report_campaign_run`` MCP verb.
"""

from tools.perf_tune_report.orchestrator.campaign_run import (
    CampaignRunResult,
    CellPlan,
    CellStepResult,
    StepFns,
    run_campaign,
    run_one_cell,
)
from tools.perf_tune_report.orchestrator.production_steps import (
    production_step_fns,
)

__all__ = [
    "CampaignRunResult",
    "CellPlan",
    "CellStepResult",
    "StepFns",
    "production_step_fns",
    "run_campaign",
    "run_one_cell",
]
