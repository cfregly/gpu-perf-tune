"""Schema constants used by the offline artifact validator.

Per the Reviewability Overhaul plan the per-benchmark column list and
required summary-field list live in their own small module so reviewers
can see the validator's contract without scrolling through ~1,900 lines
of validation logic. ``tools.ai_tuning.artifact_validation`` re-exports the
same names. New code should import from this module directly.

Quality targets belong in the caller-supplied tuning-space manifest. The
validator checks structure and local evidence without asserting an official
submission verdict.
"""

from __future__ import annotations

#: Canonical column ordering for the per-benchmark summary CSV/JSON.
#: Add a new benchmark here when the offline summary schema supports it.
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
