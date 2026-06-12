#!/usr/bin/env python3
# PROFILE_AND_OPTIMIZE_OPT_OUT: internal module; tools/ai_tuning/ai_tuning.py remains the canonical CLI.
"""Local helpers for LLM-assisted MLPerf tuning.

The commands in this file are intentionally offline: they inspect artifacts,
validate proposals, and materialize derived files, but they do not submit Slurm
jobs or mutate cluster state.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
from tools.shared.jsonutil import load_json

DEFAULT_SPACE = REPO_ROOT / "tuning" / "tuning-space.b200-llama31-8b.json"

# Safety constants live in a small sibling module so reviewers can audit
# the forbidden-pattern table and ledger-status enum without paging
# through the full tuner CLI. Re-exports preserve backwards compat for
# callers that still import these names from this module.
from tools.ai_tuning.helpers import choose_objective, parameter_index, write_json
from tools.ai_tuning.safety import (
    PROPOSAL_SCHEMA_VERSION,
)


def command_space(args: argparse.Namespace) -> int:
    space = load_json(Path(args.space))
    if args.names_only:
        payload = {
            "schema_version": space.get("schema_version"),
            "id": space.get("id"),
            "parameters": [
                {
                    "name": param.get("name"),
                    "kind": param.get("kind"),
                    "category": param.get("category"),
                    "values": param.get("values"),
                    "minimum": param.get("minimum"),
                    "maximum": param.get("maximum"),
                }
                for param in space.get("parameters", [])
            ],
        }
    else:
        payload = space
    write_json(payload, args.output)
    return 0

def command_matrix(args: argparse.Namespace) -> int:
    space = load_json(Path(args.space))
    selected_objective = choose_objective(space)
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    params = parameter_index(space)
    selected: list[dict[str, Any]] = []
    for name in args.parameter:
        param = params.get(name)
        if param is None:
            raise SystemExit(f"unknown parameter for matrix: {name}")
        values = [str(value) for value in param.get("values", [])]
        if not values:
            raise SystemExit(f"parameter has no finite values for matrix: {name}")
        selected.append({"name": name, "values": values})

    total_candidates = 1
    for item in selected:
        total_candidates *= len(item["values"])
    candidates = []
    for values in itertools.product(*(item["values"] for item in selected)):
        if len(candidates) >= args.limit:
            break
        parameters = {
            selected[index]["name"]: values[index]
            for index in range(len(selected))
        }
        candidates.append(
            {
                "parameters": parameters,
                "objective": selected_objective.get("name") if isinstance(selected_objective, dict) else None,
                "rationale": "Generated from a bounded manifest-defined matrix.",
                "risk": "Matrix candidates still require proposal validation and operator review before cluster submission.",
            }
        )

    payload = {
        "schema_version": PROPOSAL_SCHEMA_VERSION,
        "tuning_space_id": space.get("id"),
        "generated_by": "tools/ai_tuning/ai_tuning.py matrix",
        "total_possible_candidates": total_candidates,
        "truncated": total_candidates > len(candidates),
        "candidates": candidates,
    }
    write_json(payload, args.output)
    return 0
