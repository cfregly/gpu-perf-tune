#!/usr/bin/env python3
# PROFILE_AND_OPTIMIZE_OPT_OUT: internal module; tools/ai_tuning/ai_tuning.py remains the canonical CLI.
"""Local helpers for LLM-assisted MLPerf tuning.

The commands in this file are intentionally offline: they inspect artifacts,
validate proposals, and materialize derived files, but they do not submit Slurm
jobs or mutate cluster state.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
from tools.shared.jsonutil import load_json

DEFAULT_SPACE = REPO_ROOT / "tuning" / "tuning-space.b200-llama31-8b.json"

# Safety constants live in a small sibling module so reviewers can audit
# the forbidden-pattern table and ledger-status enum without paging
# through the full tuner CLI. Re-exports preserve backwards compat for
# callers that still import these names from this module.
from tools.ai_tuning.helpers import (
    _proposal_diff_markdown,
    _proposal_diff_payload,
    append_jsonl,
    candidate_key,
    config_contract_index,
    finite_parameter_domains,
    load_history_keys,
    load_ledger_keys,
    normalize_candidate_parameters,
    normalize_config_patches,
    normalize_priority,
    objective_index,
    parameter_index,
    patch_target_name,
    resolve_path_with_bases,
    stable_hash,
    utc_timestamp,
    value_allowed,
    write_json,
)
from tools.ai_tuning.safety import (
    PROPOSAL_SCHEMA_VERSION,
)


def command_proposal_diff(args: argparse.Namespace) -> int:
    diff = _proposal_diff_payload(
        before_path=Path(args.before),
        after_path=Path(args.after),
    )
    if args.format == "json":
        write_json(diff, args.output)
        return 0
    rendered = _proposal_diff_markdown(diff)
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0

def command_proposal_validate(args: argparse.Namespace) -> int:
    space = load_json(Path(args.space))
    proposal = load_json(Path(args.proposal))
    params = parameter_index(space)
    objectives = objective_index(space)
    config_contracts = config_contract_index(space)
    expected_finite_names = {domain["name"] for domain in finite_parameter_domains(space)}
    seen_keys: set[str] = set()
    history_keys = load_history_keys(args.history) | load_ledger_keys(args.ledger)
    results = []

    schema_version = proposal.get("schema_version", 1)
    candidates = proposal.get("candidates", [])
    if (
        schema_version != PROPOSAL_SCHEMA_VERSION
        or not isinstance(candidates, list)
        or not candidates
    ):
        report = {
            "schema_version": PROPOSAL_SCHEMA_VERSION,
            "valid_count": 0,
            "invalid_count": len(candidates) if isinstance(candidates, list) else 1,
            "results": [
                {
                    "index": 0,
                    "valid": False,
                    "error_codes": ["invalid_proposal_schema"],
                    "errors": [
                        "proposal must use schema_version 1 and candidates must be a non-empty list"
                    ],
                    "warning_codes": [],
                    "warnings": [],
                }
            ],
        }
        write_json(report, args.output)
        return 1

    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            results.append(
                {
                    "index": index,
                    "valid": False,
                    "error_codes": ["invalid_candidate"],
                    "errors": ["candidate must be an object"],
                    "warning_codes": [],
                    "warnings": [],
                    "parameters": {},
                    "rationale": None,
                    "risk": None,
                    "objective": None,
                    "config_patches": [],
                }
            )
            continue
        result = {
            "index": index,
            "valid": True,
            "error_codes": [],
            "errors": [],
            "warning_codes": [],
            "warnings": [],
            "parameters": normalize_candidate_parameters(candidate),
            "rationale": candidate.get("rationale"),
            "risk": candidate.get("risk"),
            "priority": None,
            "objective": candidate.get("objective"),
            "config_patches": candidate.get("config_patches", []),
            "contract_validation": {
                "objective_known": True,
                "required_config_patches": [],
                "mlperf_legality_gates": [],
            },
        }
        objective = candidate.get("objective")
        if objective is None and objectives:
            result["valid"] = False
            result["error_codes"].append("missing_objective")
            result["errors"].append("candidate must declare an objective when the tuning space defines objectives")
            result["contract_validation"]["objective_known"] = False
        elif objective is not None and objectives and str(objective) not in objectives:
            result["valid"] = False
            result["error_codes"].append("unknown_objective")
            result["errors"].append(f"unknown objective {objective}")
            result["contract_validation"]["objective_known"] = False
        if candidate.get("priority") is not None:
            try:
                result["priority"] = normalize_priority(candidate, index + 1)
            except SystemExit as exc:
                result["valid"] = False
                result["error_codes"].append("invalid_priority")
                result["errors"].append(str(exc))
        if not result["parameters"]:
            result["valid"] = False
            result["error_codes"].append("no_parameters")
            result["errors"].append("candidate must include parameters")
        provided_names = set(result["parameters"])
        missing_finite = sorted(expected_finite_names - provided_names)
        if missing_finite:
            message = "candidate does not cover all finite search-space parameters: " + ", ".join(missing_finite)
            if args.require_complete:
                result["valid"] = False
                result["error_codes"].append("missing_finite_parameters")
                result["errors"].append(message)
            else:
                result["warning_codes"].append("partial_candidate")
                result["warnings"].append(message)
        for name, value in result["parameters"].items():
            param = params.get(name)
            if param is None:
                result["valid"] = False
                result["error_codes"].append("unknown_parameter")
                result["errors"].append(f"unknown parameter {name}")
                continue
            allowed, message = value_allowed(param, value)
            if not allowed:
                result["valid"] = False
                result["error_codes"].append("invalid_parameter_value")
                result["errors"].append(f"{name}: {message}")
            if param.get("requires_operator_approval"):
                result["warning_codes"].append("operator_approval_required")
                result["warnings"].append(f"{name} changes require operator approval before submission")
            if param.get("final_run_caution"):
                result["warning_codes"].append("final_run_caution")
                result["warnings"].append(f"{name}: {param['final_run_caution']}")
            legal_values = [str(item) for item in param.get("mlperf_legal_values", [])]
            if legal_values:
                gate = {
                    "parameter": name,
                    "value": value,
                    "legal_values": legal_values,
                    "passed": value in legal_values,
                }
                result["contract_validation"]["mlperf_legality_gates"].append(gate)
                if not gate["passed"]:
                    result["valid"] = False
                    result["error_codes"].append("mlperf_illegal_value")
                    result["errors"].append(f"{name}: value {value!r} is not MLPerf-legal for this manifest")

        config_patches = normalize_config_patches(candidate)
        for name in sorted(result["parameters"]):
            param = params.get(name)
            if param is None:
                continue
            requires_patch = bool(param.get("requires_config_patch")) or name in config_contracts
            if not requires_patch:
                continue
            contract = config_contracts.get(name, {})
            param_contract = param.get("mutation_contract", {})
            if not isinstance(param_contract, dict):
                param_contract = {}
            patch = config_patches.get(name)
            required = {
                "parameter": name,
                "method": contract.get("method") or param_contract.get("method") or "template_patch",
                "target_file": contract.get("target_file") or param_contract.get("target_file"),
                "provided": patch is not None,
            }
            result["contract_validation"]["required_config_patches"].append(required)
            if patch is None:
                result["valid"] = False
                result["error_codes"].append("missing_config_patch")
                result["errors"].append(f"{name}: config mutation requires a config_patches entry")
            elif required["method"] == "template_patch" and not (
                patch.get("patch_file") or patch.get("changes")
            ):
                result["valid"] = False
                result["error_codes"].append("invalid_config_patch_contract")
                result["errors"].append(f"{name}: template_patch contract requires patch_file or inline changes")
            else:
                patch_method = str(patch.get("method", required["method"]))
                if patch_method != required["method"]:
                    result["valid"] = False
                    result["error_codes"].append("invalid_config_patch_method")
                    result["errors"].append(
                        f"{name}: config patch method {patch_method!r} does not match required method {required['method']!r}"
                    )
                patch_target = patch.get("target_file")
                if patch_target is None and patch.get("patch_file"):
                    patch_path = resolve_path_with_bases(str(patch["patch_file"]), [REPO_ROOT, Path.cwd()])
                    if not patch_path.is_file():
                        result["valid"] = False
                        result["error_codes"].append("missing_patch_file")
                        result["errors"].append(f"{name}: patch_file does not exist: {patch_path}")
                    else:
                        patch_request = load_json(patch_path)
                        if isinstance(patch_request, dict):
                            patch_target = patch_request.get("target_file")
                if required["target_file"]:
                    if patch_target is None:
                        result["valid"] = False
                        result["error_codes"].append("missing_config_patch_target")
                        result["errors"].append(f"{name}: config patch must declare target_file")
                    elif patch_target_name(str(patch_target)) != patch_target_name(str(required["target_file"])):
                        result["valid"] = False
                        result["error_codes"].append("invalid_config_patch_target")
                        result["errors"].append(
                            f"{name}: config patch target {patch_target!r} does not match required target {required['target_file']!r}"
                        )

        key = candidate_key(result["parameters"])
        if key in seen_keys:
            result["valid"] = False
            result["error_codes"].append("duplicate_candidate")
            result["errors"].append("candidate duplicates another proposal candidate")
        if key in history_keys:
            result["valid"] = False
            result["error_codes"].append("already_tried")
            result["errors"].append("candidate duplicates a historical trial")
        seen_keys.add(key)
        results.append(result)

    valid_count = sum(1 for result in results if result["valid"])
    report = {
        "schema_version": PROPOSAL_SCHEMA_VERSION,
        "tuning_space_id": space.get("id"),
        "proposal_hash": stable_hash(proposal),
        "valid_count": valid_count,
        "invalid_count": len(results) - valid_count,
        "results": results,
    }
    if args.audit_dir:
        append_jsonl(
            Path(args.audit_dir) / "proposal_audit.jsonl",
            {
                "schema_version": PROPOSAL_SCHEMA_VERSION,
                "timestamp": utc_timestamp(),
                "proposal_hash": report["proposal_hash"],
                "validation_report": report,
            },
        )
    write_json(report, args.output)
    return 0 if valid_count == len(results) else 1
