#!/usr/bin/env python3
# PROFILE_AND_OPTIMIZE_OPT_OUT: internal module; tools/ai_tuning/ai_tuning.py remains the canonical CLI.
"""Local helpers for LLM-assisted MLPerf tuning.

The commands in this file are intentionally offline: they inspect artifacts,
validate proposals, and materialize derived files, but they do not submit Slurm
jobs or mutate cluster state.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
from tools.shared.jsonutil import load_json

DEFAULT_SPACE = REPO_ROOT / "tuning" / "tuning-space.b200-llama31-8b.json"

# Safety constants live in a small sibling module so reviewers can audit
# the forbidden-pattern table and ledger-status enum without paging
# through the full tuner CLI. Re-exports preserve backwards compat for
# callers that still import these names from this module.
from tools.ai_tuning.helpers import (
    build_agent_session_report,
    build_objective_catalog,
    build_trial_analysis,
    choose_objective,
    effective_quality_target,
    effective_required_runs,
    error_code,
    run_validator,
    summarize_optional_dirs,
    summarize_raw_dir,
    utc_timestamp,
    write_json,
)
from tools.ai_tuning.safety import (
    REPORT_SCHEMA_VERSION,
)


def command_report(args: argparse.Namespace) -> int:
    space = load_json(Path(args.space))
    target = space.get("target", {})
    selected_objective = choose_objective(space, args.objective)
    required_runs = effective_required_runs(args.raw_benchmark, args.min_runs, selected_objective)
    validator_args = argparse.Namespace(**vars(args))
    validator_args.min_runs = required_runs
    validation_errors = run_validator(validator_args)
    raw_summaries = [
        summarize_raw_dir(Path(raw_dir).resolve(), args.raw_benchmark)
        for raw_dir in args.raw_results_dir
    ]
    successful_runs = sum(
        1
        for raw in raw_summaries
        for run in raw["runs"]
        if run.get("status") == "success" and not run.get("errors")
    )
    quality_target = effective_quality_target(target, selected_objective)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": utc_timestamp(),
        "tuning_space": {
            "path": str(Path(args.space)),
            "schema_version": space.get("schema_version"),
            "id": space.get("id"),
            "target": target,
            "parameter_count": len(space.get("parameters", [])),
        },
        "objectives": build_objective_catalog(space),
        "config_mutation_contracts": space.get("config_mutation_contracts", []),
        "validation": {
            "passed": not validation_errors,
            "error_count": len(validation_errors),
            "errors": [
                {"code": error_code(message), "message": message}
                for message in validation_errors[: args.error_limit]
            ],
            "truncated": len(validation_errors) > args.error_limit,
        },
        "readiness": {
            "benchmark": args.raw_benchmark,
            "objective": selected_objective.get("name") if isinstance(selected_objective, dict) else None,
            "successful_run_count": successful_runs,
            "required_successful_runs": required_runs,
            "quality_target": quality_target,
            "submission_ready": not validation_errors and successful_runs >= required_runs,
        },
        "trial_analysis": build_trial_analysis(
            raw_summaries,
            target,
            selected_objective,
            successful_runs,
            required_runs,
            validation_errors,
        ),
        "raw_results": raw_summaries,
        "gb300_fabric": summarize_optional_dirs(
            args.gb300_fabric_dir, ["evidence.json", "run-context.json"]
        ),
        "gb300_node_selection": summarize_optional_dirs(
            args.gb300_node_selection_dir, ["selector-config.json", "node-scores.json"]
        ),
        "gb300_fabric_localization": summarize_optional_dirs(
            args.gb300_fabric_localization_dir,
            ["fabric-run-manifest.json", "fabric-localization.json"],
        ),
        "operator_actions": space.get("operator_actions", []),
        "agent_session": build_agent_session_report(
            space,
            args.ledger,
            args.remaining_limit,
            args.template_hint_file,
            args.template_hint_limit,
        ),
        "next_actions": [
            "Validate any LLM proposal with proposal validate before editing or submitting.",
            "Use template-patch validate for copied config_*.sh, run.sub, .sbatch, or .hyp edits.",
            "Preview experiment submit commands, then execute only with explicit operator approval.",
        ],
    }
    write_json(report, args.output)
    return 0 if not validation_errors else 1
