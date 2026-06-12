"""Schema constants used by ``validate_artifacts.py``.

Per the Reviewability Overhaul plan the per-benchmark column list and
required summary-field list live in their own small module so reviewers
can see the validator's contract without scrolling through ~1,900 lines
of validation logic. ``validate_artifacts.py`` re-exports the same
names for stable public API compatibility; new code should import from this
module directly.

Quality targets (final log perplexity + secondary FID/CLIP/eval_loss
targets) live in :mod:`tools.shared.mlperf_targets`. They are loaded
from [`tuning/mlperf_rules_v6_0.json`](../../tuning/mlperf_rules_v6_0.json)
so a ruleset bump only touches the JSON, not Python.
"""

from __future__ import annotations

#: Canonical column ordering for the per-benchmark summary CSV/JSON.
#: Adding a new benchmark requires bumping
#: [`tuning/mlperf_rules_v6_0.json`](../../tuning/mlperf_rules_v6_0.json)
#: in the same diff.
BENCHMARK_COLUMNS: tuple[str, ...] = (
    "llama31_8b",
    "dlrm_dcnv2",
    "flux1",
    "llama2_70b_lora",
    "llama31_405b",
    "gpt_oss_20b",
    "deepseekv3_671b",
)

#: MLPerf submission summary.json must carry every one of these fields
#: with a non-empty value. The validator emits a per-field error when one
#: is missing so reviewers can see exactly which compliance bullet
#: failed.
REQUIRED_SUMMARY_FIELDS: tuple[str, ...] = (
    "division",
    "availability",
    "submitter",
    "system",
    "number_of_nodes",
    "accelerator_model_name",
    "accelerators_count",
    "framework",
)


__all__ = (
    "BENCHMARK_COLUMNS",
    "REQUIRED_SUMMARY_FIELDS",
)
