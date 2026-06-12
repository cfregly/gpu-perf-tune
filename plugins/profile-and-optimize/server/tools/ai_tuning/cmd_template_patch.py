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

from tools.ai_tuning.helpers import default_derived_path, evaluate_template_patch_request, write_json


def command_template_patch_validate(args: argparse.Namespace) -> int:
    request = load_json(Path(args.patch))
    target = Path(request.get("target_file", ""))
    if not target.is_absolute():
        target = (REPO_ROOT / target).resolve()
    output = Path(args.output_file or request.get("output_file") or default_derived_path(target))
    if not output.is_absolute():
        output = (REPO_ROOT / output).resolve()
    report = evaluate_template_patch_request(
        request,
        target=target,
        output=output,
        apply=args.apply,
        audit_dir=Path(args.audit_dir) if args.audit_dir else None,
    )
    write_json(report, args.output)
    return 0 if report["safe"] else 1
