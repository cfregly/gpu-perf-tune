#!/usr/bin/env python3
# PROFILE_AND_OPTIMIZE_OPT_OUT: internal module; tools/ai_tuning/ai_tuning.py remains the canonical CLI.
"""Local helpers for LLM-assisted MLPerf tuning.

The commands in this file are intentionally offline: they inspect artifacts,
validate proposals, and materialize derived files, but they do not submit Slurm
jobs or mutate cluster state.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SPACE = REPO_ROOT / "tuning" / "tuning-space.b200-llama31-8b.json"

# Safety constants live in a small sibling module so reviewers can audit
# the forbidden-pattern table and ledger-status enum without paging
# through the full tuner CLI. Re-exports preserve backwards compat for
# callers that still import these names from this module.

from tools.ai_tuning.helpers import discover_finalize_logs, finalize_copy_candidates, validate_finalize_run, write_json


def command_finalize(args: argparse.Namespace) -> int:
    log_dir = args.log_dir.resolve()
    workdir = args.workdir.resolve()
    results_dir = args.results_dir.resolve()
    if not log_dir.is_dir():
        raise SystemExit(f"missing log dir: {log_dir}")
    if not workdir.is_dir():
        raise SystemExit(f"missing workdir: {workdir}")
    runs = [
        validate_finalize_run(log_dir, log_path, args.benchmark)
        for log_path in discover_finalize_logs(log_dir, args.run_id)
    ]
    valid_runs = [run for run in runs if run["valid"]]
    if len(valid_runs) < args.required_runs:
        report = {
            "schema_version": 1,
            "dry_run": args.dry_run,
            "valid": False,
            "error": f"found {len(valid_runs)} compliant runs, need {args.required_runs}",
            "runs": runs,
        }
        write_json(report, args.output)
        return 1
    valid_runs = sorted(valid_runs, key=lambda run: str(run["run_id"]))[: args.required_runs]
    copies = finalize_copy_candidates(log_dir, workdir, results_dir, valid_runs, args.launcher_file)
    copied_files = []
    if not args.dry_run:
        results_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in copies:
        if not src.is_file():
            raise SystemExit(f"missing finalize source: {src}")
        if not args.dry_run:
            shutil.copy2(src, dst)
        copied_files.append(str(dst))
    report = {
        "schema_version": 1,
        "dry_run": args.dry_run,
        "valid": True,
        "benchmark": args.benchmark,
        "required_runs": args.required_runs,
        "selected_runs": valid_runs,
        "results_dir": str(results_dir),
        "copied_file_count": len(copied_files),
        "copied_files": copied_files,
    }
    write_json(report, args.output)
    return 0
