#!/usr/bin/env python3
"""Shared helpers for the MLPerf AI tuner."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.util
import io
import itertools
import json
import random
import re
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Any

from tools.ai_tuning.optimizer import hyperband as hb_engine
from tools.ai_tuning.optimizer import space as space_module
from tools.ai_tuning.optimizer import tpe as tpe_engine

REPO_ROOT = Path(__file__).resolve().parents[2]
from tools.shared.jsonutil import load_json

DEFAULT_SPACE = REPO_ROOT / "tuning" / "tuning-space.b200-llama31-8b.json"

# Safety constants live in a small sibling module so reviewers can audit
# the forbidden-pattern table and ledger-status enum without paging
# through the full tuner CLI. Re-exports preserve backwards compat for
# callers that still import these names from this module.
from tools.ai_tuning.safety import (
    EXPERIMENT_LEDGER_SCHEMA_VERSION,
    EXPERIMENT_STATUSES,
    FORBIDDEN_PATCH_PATTERNS,
    TEMPLATE_PATCH_SCHEMA_VERSION,
)


def load_validate_artifacts() -> Any:
    path = REPO_ROOT / "validate_artifacts.py"
    spec = importlib.util.spec_from_file_location("validate_artifacts", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def write_json(payload: Any, output: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")

def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")

def utc_timestamp() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def short_hash(payload: Any) -> str:
    return stable_hash(payload)[:12]

def error_code(message: str) -> str:
    lowered = message.lower()
    if "missing raw results directory" in lowered:
        return "missing_raw_results_dir"
    if "expected at least" in lowered and "raw" in lowered:
        return "insufficient_raw_runs"
    if "missing config_" in lowered:
        return "missing_config"
    if "missing launcher" in lowered:
        return "missing_launcher"
    if "missing container-env" in lowered:
        return "missing_env_log"
    if "run_stop status" in lowered:
        return "run_stop_not_success"
    if "final eval_accuracy/log_ppl" in lowered and "exceeds target" in lowered:
        return "metric_target_missed"
    if "missing checker output" in lowered:
        return "missing_checker_output"
    if "checker output" in lowered:
        return "checker_failed"
    if "missing jsonl" in lowered:
        return "missing_jsonl"
    return re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")[:80] or "validation_error"

def parameter_index(space: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(param["name"]): param
        for param in space.get("parameters", [])
        if isinstance(param, dict) and "name" in param
    }

def objective_index(space: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(objective["name"]): objective
        for objective in space.get("objectives", [])
        if isinstance(objective, dict) and "name" in objective
    }

def objective_catalog(space: dict[str, Any]) -> list[dict[str, Any]]:
    return [objective for objective in space.get("objectives", []) if isinstance(objective, dict)]

def choose_objective(space: dict[str, Any], requested: str | None = None) -> dict[str, Any] | None:
    objectives = objective_catalog(space)
    if not objectives:
        return None
    if requested is None:
        return objectives[0]
    selected = objective_index(space).get(str(requested))
    if selected is None:
        raise SystemExit(f"unknown objective {requested!r}")
    return selected

def config_contract_index(space: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(contract["parameter"]): contract
        for contract in space.get("config_mutation_contracts", [])
        if isinstance(contract, dict) and "parameter" in contract
    }

def normalize_config_patches(candidate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    patches = candidate.get("config_patches", [])
    if not isinstance(patches, list):
        return {}
    normalized = {}
    for patch in patches:
        if isinstance(patch, dict) and patch.get("parameter"):
            normalized[str(patch["parameter"])] = patch
    return normalized

def normalize_candidate_parameters(candidate: dict[str, Any]) -> dict[str, str]:
    raw = candidate.get("parameters", {})
    if isinstance(raw, dict):
        return {str(name): str(value) for name, value in raw.items()}
    if isinstance(raw, list):
        normalized: dict[str, str] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            if "name" in item and "value" in item:
                normalized[str(item["name"])] = str(item["value"])
        return normalized
    return {}

def value_allowed(param: dict[str, Any], value: str) -> tuple[bool, str | None]:
    kind = param.get("kind", "string")
    if "values" in param:
        allowed = [str(item) for item in param["values"]]
        if value not in allowed:
            return False, f"value {value!r} is not in allowed values {allowed}"
        return True, None
    if kind == "integer":
        try:
            parsed = int(value)
        except ValueError:
            return False, f"value {value!r} is not an integer"
        if "minimum" in param and parsed < int(param["minimum"]):
            return False, f"value {value!r} is below minimum {param['minimum']}"
        if "maximum" in param and parsed > int(param["maximum"]):
            return False, f"value {value!r} is above maximum {param['maximum']}"
    elif kind == "number":
        try:
            parsed_float = float(value)
        except ValueError:
            return False, f"value {value!r} is not a number"
        if "minimum" in param and parsed_float < float(param["minimum"]):
            return False, f"value {value!r} is below minimum {param['minimum']}"
        if "maximum" in param and parsed_float > float(param["maximum"]):
            return False, f"value {value!r} is above maximum {param['maximum']}"
    elif kind == "boolean" and value.lower() not in {"0", "1", "true", "false"}:
        return False, f"value {value!r} is not a boolean"
    return True, None

def run_validator(args: argparse.Namespace) -> list[str]:
    validator = load_validate_artifacts()
    errors: list[str] = []
    require_nccl_runtime = bool(getattr(args, "require_nccl_runtime", False))
    with contextlib.redirect_stdout(io.StringIO()):
        for raw_dir in args.raw_results_dir:
            validator.validate_raw_results_dir(
                errors,
                Path(raw_dir).resolve(),
                args.raw_benchmark,
                args.min_runs,
                require_nccl_runtime=require_nccl_runtime,
            )
        for fabric_dir in args.gb300_fabric_dir:
            validator.validate_fabric_evidence(
                errors, Path(fabric_dir).resolve(), args.gb300_fabric_require_clean
            )
        for selection_dir in args.gb300_node_selection_dir:
            validator.validate_node_selection_dir(errors, Path(selection_dir).resolve())
        for localization_dir in args.gb300_fabric_localization_dir:
            validator.validate_fabric_localization_dir(errors, Path(localization_dir).resolve())
    return errors

def summarize_raw_dir(raw_dir: Path, benchmark: str) -> dict[str, Any]:
    validator = load_validate_artifacts()
    run_summaries = []
    for run_log in sorted(raw_dir.glob("*_1.log")):
        per_run_errors: list[str] = []
        with contextlib.redirect_stdout(io.StringIO()):
            status, elapsed_minutes, final_metric = validator.validate_result_file(
                per_run_errors, benchmark, run_log
            )
        run_id = validator.run_id_from_log_path(run_log)
        compliance_path = raw_dir / f"compliance_{run_id}.out"
        audit_path = raw_dir / f"audit_{run_id}.out"
        run_summaries.append(
            {
                "run_id": run_id,
                "log": str(run_log),
                "status": status,
                "elapsed_minutes": elapsed_minutes,
                "final_log_ppl": final_metric,
                "compliance_output": str(compliance_path),
                "audit_output": str(audit_path),
                "errors": [
                    {"code": error_code(message), "message": message} for message in per_run_errors
                ],
            }
        )
    return {
        "path": str(raw_dir),
        "benchmark": benchmark,
        "run_log_count": len(run_summaries),
        "config_count": len(list(raw_dir.glob("config_*.sh"))),
        "launcher_count": len(list(raw_dir.glob("*.hyp"))) + int((raw_dir / "run.sub").is_file()),
        "env_log_count": len(list(raw_dir.glob("container-env-*.log")))
        + len(list(raw_dir.glob("*_env.log"))),
        "runs": run_summaries,
    }

def mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None

def stdev_or_none(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None

def effective_quality_target(target: dict[str, Any], objective: dict[str, Any] | None) -> float | int | None:
    if objective and isinstance(objective.get("quality_target"), int | float):
        return objective["quality_target"]
    if isinstance(target.get("quality_target"), int | float):
        return target["quality_target"]
    return None

def effective_required_runs(
    raw_benchmark: str,
    min_runs: int | None,
    objective: dict[str, Any] | None,
) -> int:
    if min_runs is not None:
        return min_runs
    if objective and isinstance(objective.get("minimum_successful_runs"), int):
        return int(objective["minimum_successful_runs"])
    if raw_benchmark == "llama31_8b":
        return 5
    return 1

def build_trial_analysis(
    raw_summaries: list[dict[str, Any]],
    target: dict[str, Any],
    objective: dict[str, Any] | None,
    successful_runs: int,
    required_runs: int,
    validation_errors: list[str],
) -> dict[str, Any]:
    runs = [run for raw in raw_summaries for run in raw.get("runs", [])]
    elapsed_values = [
        float(run["elapsed_minutes"])
        for run in runs
        if isinstance(run.get("elapsed_minutes"), int | float)
    ]
    metric_values = [
        float(run["final_log_ppl"])
        for run in runs
        if isinstance(run.get("final_log_ppl"), int | float)
    ]
    quality_target = effective_quality_target(target, objective)
    if isinstance(quality_target, int | float):
        quality_pass_count = sum(value <= float(quality_target) for value in metric_values)
        quality_gate_passed = bool(metric_values) and quality_pass_count == len(metric_values)
    else:
        quality_pass_count = None
        quality_gate_passed = None
    objective_name = (
        str(objective.get("name"))
        if isinstance(objective, dict) and objective.get("name") is not None
        else "mlperf_submission_readiness"
    )
    return {
        "minimum_bar": [
            "objective_scoring",
            "structured_result_ingestion",
            "config_mutation_contracts",
            "repeatability_noise_handling",
            "mlperf_legality_gates",
            "optimizer_state",
        ],
        "objective_scoring": {
            "primary_objective": objective_name,
            "primary_metric": objective.get("primary_metric") if isinstance(objective, dict) else "submission_ready",
            "direction": objective.get("direction") if isinstance(objective, dict) else "maximize",
            "objective_progress_ratio": min(successful_runs / required_runs, 1.0)
            if required_runs
            else None,
            "submission_readiness_score": min(successful_runs / required_runs, 1.0)
            if required_runs
            else None,
            "mean_elapsed_minutes": mean_or_none(elapsed_values),
            "mean_final_log_ppl": mean_or_none(metric_values),
            "quality_target": quality_target,
        },
        "structured_result_ingestion": {
            "run_count": len(runs),
            "fields": ["status", "elapsed_minutes", "final_log_ppl", "compliance_output", "audit_output", "errors"],
        },
        "repeatability_noise_handling": {
            "sample_count": len(runs),
            "successful_run_count": successful_runs,
            "elapsed_stdev_minutes": stdev_or_none(elapsed_values),
            "final_log_ppl_stdev": stdev_or_none(metric_values),
        },
        "mlperf_legality_gates": [
            {
                "name": "artifact_and_compliance_validation",
                "passed": not validation_errors,
                "error_count": len(validation_errors),
            },
            {
                "name": "required_successful_runs",
                "passed": successful_runs >= required_runs,
                "successful_run_count": successful_runs,
                "required_successful_runs": required_runs,
            },
            {
                "name": "quality_target",
                "passed": quality_gate_passed,
                "passing_run_count": quality_pass_count,
                "quality_target": quality_target,
            },
        ],
        "config_mutation_contracts": {
            "external_config_requires_template_patch": True,
            "preserve_original_config": True,
        },
    }

def build_objective_catalog(space: dict[str, Any]) -> list[dict[str, Any]]:
    objectives = objective_catalog(space)
    if objectives:
        return objectives
    target = space.get("target", {})
    return [
        {
            "name": "mlperf_submission_readiness",
            "direction": "maximize",
            "primary_metric": "submission_ready",
            "minimum_successful_runs": 5,
            "quality_metric": target.get("quality_metric"),
            "quality_target": target.get("quality_target"),
        }
    ]

def run_id_from_log(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_1"):
        return stem[:-2]
    if stem.endswith("_01"):
        return stem[:-3]
    return stem

def discover_finalize_logs(log_dir: Path, run_ids: list[str]) -> list[Path]:
    if run_ids:
        logs = [log_dir / f"{run_id}_1.log" for run_id in run_ids]
    else:
        logs = sorted(log_dir.glob("*_1.log"))
    missing = [str(path) for path in logs if not path.is_file()]
    if missing:
        raise SystemExit(f"missing run log(s): {', '.join(missing)}")
    return logs

def validate_finalize_run(log_dir: Path, log_path: Path, benchmark: str) -> dict[str, Any]:
    validator = load_validate_artifacts()
    errors: list[str] = []
    with contextlib.redirect_stdout(io.StringIO()):
        status, elapsed_minutes, final_metric = validator.validate_result_file(errors, benchmark, log_path)
    run_id = run_id_from_log(log_path)
    compliance = log_dir / f"compliance_{run_id}.out"
    audit = log_dir / f"audit_{run_id}.out"
    validator.validate_checker_output(errors, compliance)
    validator.validate_checker_output(errors, audit)
    return {
        "run_id": run_id,
        "log": str(log_path),
        "status": status,
        "elapsed_minutes": elapsed_minutes,
        "final_log_ppl": final_metric,
        "compliance_output": str(compliance),
        "audit_output": str(audit),
        "valid": not errors,
        "errors": [{"code": error_code(message), "message": message} for message in errors],
    }

def finalize_copy_candidates(
    log_dir: Path,
    workdir: Path,
    results_dir: Path,
    valid_runs: list[dict[str, Any]],
    launcher_file: Path | None,
) -> list[tuple[Path, Path]]:
    copies: list[tuple[Path, Path]] = []
    for run in valid_runs:
        for key in ("log", "compliance_output", "audit_output"):
            src = Path(str(run[key]))
            copies.append((src, results_dir / src.name))
    launcher_candidates = []
    if (workdir / "run.sub").is_file():
        launcher_candidates.append(workdir / "run.sub")
    launcher_candidates.extend(sorted(workdir.glob("*.hyp")))
    if launcher_file is not None:
        launcher_candidates.append(launcher_file)
    if not launcher_candidates:
        raise SystemExit(f"missing launcher in {workdir}: expected run.sub or *.hyp")
    for src in launcher_candidates:
        if src.is_file():
            copies.append((src, results_dir / src.name))
    configs = sorted(workdir.glob("config_*.sh"))
    if not configs:
        raise SystemExit(f"missing config_*.sh in {workdir}")
    for src in configs:
        copies.append((src, results_dir / src.name))
    env_logs = sorted(log_dir.glob("container-env-*.log")) + sorted(log_dir.glob("*_env.log"))
    if not env_logs:
        raise SystemExit(f"missing container-env-*.log or *_env.log in {log_dir}")
    for src in env_logs:
        copies.append((src, results_dir / src.name))
    deduped = []
    seen = set()
    for src, dst in copies:
        key = (src.resolve(), dst)
        if key not in seen:
            seen.add(key)
            deduped.append((src, dst))
    return deduped

def summarize_json_file(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return load_json(path)
    except json.JSONDecodeError as exc:
        return {"error": f"malformed JSON: {exc}"}

def summarize_optional_dirs(paths: list[str], filenames: list[str]) -> list[dict[str, Any]]:
    summaries = []
    for raw_path in paths:
        directory = Path(raw_path).resolve()
        summary = {"path": str(directory), "exists": directory.is_dir(), "files": {}}
        for filename in filenames:
            summary["files"][filename] = summarize_json_file(directory / filename)
        summaries.append(summary)
    return summaries

def finite_parameter_domains(space: dict[str, Any]) -> list[dict[str, Any]]:
    domains = []
    for param in space.get("parameters", []):
        if not isinstance(param, dict) or "name" not in param:
            continue
        values = [str(value) for value in param.get("values", [])]
        if not values and param.get("kind") == "boolean":
            values = ["0", "1"]
        if not values:
            continue
        domains.append(
            {
                "name": str(param["name"]),
                "kind": str(param.get("kind", "string")),
                "values": values,
                "category": param.get("category"),
                "wire": param.get("wire"),
            }
        )
    domains.sort(key=lambda item: item["name"])
    return domains

def build_parameter_coverage(
    domains: list[dict[str, Any]], states: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    coverage = []
    for domain in domains:
        observed_counts = {value: 0 for value in domain["values"]}
        for record in states.values():
            parameters = record.get("parameters", {})
            if not isinstance(parameters, dict):
                continue
            value = parameters.get(domain["name"])
            if value is not None:
                observed_counts[str(value)] = observed_counts.get(str(value), 0) + 1
        observed_values = sum(1 for count in observed_counts.values() if count > 0)
        expected_values = len(domain["values"])
        coverage.append(
            {
                "name": domain["name"],
                "category": domain.get("category"),
                "expected_values": expected_values,
                "observed_values": observed_values,
                "coverage_ratio": observed_values / expected_values if expected_values else 0.0,
                "observed_counts": observed_counts,
            }
        )
    return coverage

def build_remaining_candidates(
    domains: list[dict[str, Any]],
    states: dict[str, dict[str, Any]],
    limit: int,
) -> dict[str, Any]:
    if limit < 0:
        raise SystemExit("--remaining-limit must be non-negative")
    if not domains:
        return {"total": 0, "included": 0, "truncated": False, "candidates": []}
    tried_keys = set()
    for record in states.values():
        parameters = record.get("parameters", {})
        if isinstance(parameters, dict):
            tried_keys.add(candidate_key({str(k): str(v) for k, v in parameters.items()}))
    total_possible = 1
    for domain in domains:
        total_possible *= len(domain["values"])
    estimated_remaining = max(total_possible - len(tried_keys), 0)
    candidates = []
    for values in itertools.product(*(domain["values"] for domain in domains)):
        if len(candidates) >= limit:
            break
        parameters = {
            domains[index]["name"]: values[index]
            for index in range(len(domains))
        }
        key = candidate_key(parameters)
        if key in tried_keys:
            continue
        candidates.append(
            {
                "id": f"cand-{short_hash(parameters)}",
                "parameters": parameters,
            }
        )
    return {
        "total": estimated_remaining,
        "total_is_estimate": bool(tried_keys),
        "included": len(candidates),
        "truncated": estimated_remaining > len(candidates),
        "candidates": candidates,
    }

def build_circuit_breakers(
    counts: dict[str, int], coverage: list[dict[str, Any]], remaining: dict[str, Any]
) -> list[dict[str, str]]:
    breakers = []
    attempted = counts.get("succeeded", 0) + counts.get("failed", 0)
    failure_rate = counts.get("failed", 0) / attempted if attempted else 0.0
    if attempted >= 4 and failure_rate >= 0.5:
        breakers.append(
            {
                "kind": "high_failure_rate",
                "severity": "warning",
                "message": f"Failure rate is {failure_rate * 100:.1f}% across completed experiments.",
            }
        )
    min_coverage = min(
        (item["coverage_ratio"] for item in coverage),
        default=None,
    )
    tracked = sum(counts.values())
    if min_coverage is not None and tracked >= 4 and min_coverage < 0.30:
        breakers.append(
            {
                "kind": "narrow_parameter_coverage",
                "severity": "warning",
                "message": "Observed experiments cover less than 30% of at least one finite parameter.",
            }
        )
    if remaining.get("total") == 0 and tracked > 0:
        breakers.append(
            {
                "kind": "search_space_exhausted",
                "severity": "info",
                "message": "No finite remaining candidates are left for the current tuning space.",
            }
        )
    if counts.get("blocked", 0) >= 3:
        breakers.append(
            {
                "kind": "blocked_experiments",
                "severity": "warning",
                "message": "Several experiments are blocked; inspect scheduler, cluster-access, or artifact prerequisites before adding more.",
            }
        )
    return breakers

def split_name_value(value: str) -> tuple[str | None, str | None]:
    if "=" in value:
        name, parsed = value.split("=", 1)
        return name.strip() or None, parsed.strip().strip('"') or None
    parts = value.split(None, 1)
    if len(parts) == 2:
        return parts[0].strip() or None, parts[1].strip().strip('"') or None
    return value.strip() or None, None

def parse_template_hint_line(line: str, line_number: int) -> dict[str, Any] | None:
    trimmed = line.strip()
    if not trimmed or "hypertune." in trimmed:
        return None
    if trimmed.startswith("#SBATCH"):
        name, value = split_name_value(trimmed.removeprefix("#SBATCH").strip())
        return {"kind": "sbatch", "name": name, "value": value, "line_number": line_number, "raw": line}
    if trimmed.startswith("export "):
        name, value = split_name_value(trimmed.removeprefix("export").strip())
        return {"kind": "env", "name": name, "value": value, "line_number": line_number, "raw": line}
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=\"?\$\{[^:}]+:-([^}]+)\}\"?$", trimmed)
    if match:
        return {
            "kind": "env-default",
            "name": match.group(1),
            "value": match.group(2).strip('"'),
            "line_number": line_number,
            "raw": line,
        }
    if trimmed.startswith("--"):
        name, value = split_name_value(trimmed.rstrip("\\").strip())
        return {"kind": "argument", "name": name, "value": value, "line_number": line_number, "raw": line}
    return None

def build_template_hints(paths: list[str], limit: int) -> dict[str, Any]:
    if limit < 0:
        raise SystemExit("--template-hint-limit must be non-negative")
    hints = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            hints.append(
                {
                    "kind": "missing-file",
                    "name": None,
                    "value": None,
                    "line_number": None,
                    "raw": str(path),
                    "path": str(path),
                }
            )
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            hint = parse_template_hint_line(line, line_number)
            if hint is not None:
                hint["path"] = str(path)
                hints.append(hint)
    return {
        "total": len(hints),
        "included": min(len(hints), limit),
        "truncated": len(hints) > limit,
        "hints": hints[:limit],
    }

def build_agent_session_report(
    space: dict[str, Any],
    ledger: Path | None,
    remaining_limit: int,
    template_hint_paths: list[str],
    template_hint_limit: int,
) -> dict[str, Any]:
    records = read_ledger_records(ledger) if ledger else []
    states = latest_experiment_states(records)
    counts: dict[str, int] = {}
    for record in states.values():
        status = str(record.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    domains = finite_parameter_domains(space)
    coverage = build_parameter_coverage(domains, states)
    remaining = build_remaining_candidates(domains, states, remaining_limit)
    attempted = counts.get("succeeded", 0) + counts.get("failed", 0)
    metrics = {
        "failure_rate": counts.get("failed", 0) / attempted if attempted else 0.0,
        "min_parameter_coverage_ratio": min(
            (item["coverage_ratio"] for item in coverage),
            default=None,
        ),
        "parameter_coverage": coverage,
    }
    return {
        "ledger": str(ledger) if ledger else None,
        "counts": {
            "tracked": len(states),
            "remaining_untested": remaining["total"],
            "planned": counts.get("planned", 0),
            "staged": counts.get("staged", 0),
            "submitted": counts.get("submitted", 0),
            "running": counts.get("running", 0),
            "succeeded": counts.get("succeeded", 0),
            "failed": counts.get("failed", 0),
            "blocked": counts.get("blocked", 0),
            "cancelled": counts.get("cancelled", 0),
        },
        "metrics": metrics,
        "circuit_breakers": build_circuit_breakers(counts, coverage, remaining),
        "search_space": {
            "parameters": domains,
            "feasibility_notes": [
                "Remaining candidates include finite manifest dimensions only.",
                "String-valued parameters without explicit values are excluded from Cartesian expansion.",
            ],
        },
        "template_hints": build_template_hints(template_hint_paths, template_hint_limit),
        "remaining_candidates": remaining,
        "optimizer_state": {
            "strategy": space.get("optimizer", {}).get("default_strategy", "deterministic_matrix")
            if isinstance(space.get("optimizer"), dict)
            else "deterministic_matrix",
            "tracked_experiments": len(states),
            "remaining_finite_candidates": remaining["total"],
            "supported_strategies": space.get("optimizer", {}).get("supported_strategies", ["deterministic_matrix"])
            if isinstance(space.get("optimizer"), dict)
            else ["deterministic_matrix"],
            "notes": "Optimizer state is explicit so future random, Bayesian, or multi-fidelity search can replace deterministic matrices without relying on chat memory.",
        },
    }

def candidate_seen(parameters: dict[str, str], states: dict[str, dict[str, Any]]) -> bool:
    key = candidate_key(parameters)
    for record in states.values():
        record_parameters = record.get("parameters", {})
        if isinstance(record_parameters, dict):
            if candidate_key({str(k): str(v) for k, v in record_parameters.items()}) == key:
                return True
    return False

def decode_combination(index: int, domains: list[dict[str, Any]]) -> dict[str, str]:
    values: list[str] = []
    remaining = index
    for domain in reversed(domains):
        domain_values = domain["values"]
        remaining, offset = divmod(remaining, len(domain_values))
        values.append(domain_values[offset])
    values.reverse()
    return {
        domains[position]["name"]: values[position]
        for position in range(len(domains))
    }

def iter_candidate_parameters(
    domains: list[dict[str, Any]],
    strategy: str,
    rng: random.Random,
) -> tuple[int, Any]:
    total_possible = 1
    for domain in domains:
        total_possible *= len(domain["values"])
    if strategy == "random":
        def iterator() -> Any:
            seen_indices: set[int] = set()
            while len(seen_indices) < total_possible:
                index = rng.randrange(total_possible)
                if index in seen_indices:
                    continue
                seen_indices.add(index)
                yield decode_combination(index, domains)

        return total_possible, iterator()

    def iterator() -> Any:
        for values in itertools.product(*(domain["values"] for domain in domains)):
            yield {
                domains[index]["name"]: values[index]
                for index in range(len(domains))
            }

    return total_possible, iterator()

def _build_optimizer_space(
    space: dict[str, Any],
    selected_names: list[str] | None,
) -> space_module.Space:
    return space_module.Space.from_manifest(space, parameter_names=selected_names)

def _observations_from_states(
    optimizer_space: space_module.Space,
    states: dict[str, dict[str, Any]],
) -> list[tpe_engine.Observation]:
    observations: list[tpe_engine.Observation] = []
    for record in states.values():
        params = record.get("parameters", {})
        if not isinstance(params, dict):
            continue
        try:
            vector = optimizer_space.encode({k: str(v) for k, v in params.items()})
        except ValueError:
            continue
        result = record.get("result_value")
        if result is None:
            continue
        try:
            value = float(result)
        except (TypeError, ValueError):
            continue
        observations.append(tpe_engine.Observation(vector=vector, value=value))
    return observations

def _objective_direction(space: dict[str, Any], objective_name: str) -> str:
    for objective in space.get("objectives", []):
        if isinstance(objective, dict) and str(objective.get("name")) == objective_name:
            direction = str(objective.get("direction", "maximize"))
            if direction in {"minimize", "maximize"}:
                return direction
    return "maximize"

def _resolve_hyperband_config(space: dict[str, Any], variant: str, args: argparse.Namespace) -> hb_engine.HyperbandConfig:
    optimizer_block = space.get("optimizer", {}) if isinstance(space.get("optimizer"), dict) else {}
    block = optimizer_block.get(variant) if isinstance(optimizer_block.get(variant), dict) else {}
    eta = int(args.eta) if args.eta is not None else int(block.get("eta", 3))
    min_budget = float(args.min_budget) if args.min_budget is not None else float(block.get("min_budget", 1.0))
    max_budget = float(args.max_budget) if args.max_budget is not None else float(block.get("max_budget", min_budget * (eta ** 3)))
    return hb_engine.HyperbandConfig(eta=eta, min_budget=min_budget, max_budget=max_budget)

def _experiment_state_value(record: dict[str, Any]) -> float | None:
    raw = record.get("result_value")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None

def load_history_keys(paths: list[str]) -> set[str]:
    keys: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            parameters = record.get("parameters") or record.get("candidate", {}).get("parameters")
            if isinstance(parameters, dict):
                keys.add(candidate_key({str(k): str(v) for k, v in parameters.items()}))
    return keys

def load_ledger_keys(path: Path | None) -> set[str]:
    if path is None:
        return set()
    keys = set()
    for record in latest_experiment_states(read_ledger_records(path)).values():
        parameters = record.get("parameters", {})
        if isinstance(parameters, dict):
            keys.add(candidate_key({str(k): str(v) for k, v in parameters.items()}))
    return keys

def candidate_key(parameters: dict[str, str]) -> str:
    return "|".join(f"{name}={parameters[name]}" for name in sorted(parameters))

def experiment_id(space_id: str, index: int, parameters: dict[str, str]) -> str:
    return f"exp-{short_hash({'space': space_id, 'index': index, 'parameters': parameters})}"

def normalize_priority(candidate: dict[str, Any], default: int) -> int:
    raw_priority = candidate.get("priority", default)
    if isinstance(raw_priority, bool):
        raise SystemExit("candidate priority must be an integer")
    try:
        priority = int(raw_priority)
    except (TypeError, ValueError) as exc:
        raise SystemExit("candidate priority must be an integer") from exc
    if priority < 1:
        raise SystemExit("candidate priority must be at least 1")
    return priority

def infer_experiment_shape(parameters: dict[str, str], space: dict[str, Any]) -> dict[str, Any]:
    target = space.get("target", {})
    nodes = (
        parameters.get("EXPERIMENT_NODES")
        or parameters.get("GATE_NODES")
        or parameters.get("SELECTOR_TARGET_COUNT")
        or str(target.get("default_nodes") or target.get("default_gate_nodes") or "")
    )
    stage = parameters.get("RUN_MODE") or parameters.get("RUN_STAGE") or "planned"
    return {
        "stage": stage,
        "nodes": nodes,
        "partition": parameters.get("SLURM_PARTITION") or target.get("partition"),
        "benchmark": target.get("benchmark"),
    }

def read_ledger_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_number}: malformed ledger JSONL: {exc}") from exc
        if not isinstance(record, dict):
            raise SystemExit(f"{path}:{line_number}: ledger record must be an object")
        records.append(record)
    return records

def latest_experiment_states(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for record in records:
        exp_id = record.get("experiment_id")
        if not exp_id:
            continue
        current = states.setdefault(str(exp_id), {})
        if record.get("event") == "created":
            current.update(record)
        else:
            for key, value in record.items():
                if value is not None:
                    current[key] = value
    return states

def append_experiment_update(
    ledger: Path,
    experiment_id_value: str,
    status: str,
    *,
    slurm_job_id: str | None = None,
    artifact_dir: str | None = None,
    notes: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in EXPERIMENT_STATUSES:
        raise SystemExit(f"unknown status {status!r}; expected one of {sorted(EXPERIMENT_STATUSES)}")
    record = {
        "schema_version": EXPERIMENT_LEDGER_SCHEMA_VERSION,
        "event": "updated",
        "timestamp": utc_timestamp(),
        "experiment_id": experiment_id_value,
        "status": status,
        "slurm_job_id": slurm_job_id,
        "artifact_dir": artifact_dir,
        "notes": notes,
    }
    if extra:
        record.update(extra)
    append_jsonl(ledger, record)
    return record

def planned_submit_command(record: dict[str, Any], script: Path, extra_args: list[str]) -> list[str]:
    shape = record.get("shape", {}) if isinstance(record.get("shape"), dict) else {}
    command = ["sbatch", "--parsable"]
    partition = shape.get("partition")
    nodes = shape.get("nodes")
    if partition:
        command.append(f"--partition={partition}")
    if nodes:
        command.append(f"--nodes={nodes}")
    command.extend(extra_args)
    command.append(str(script))
    return command

def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"

def extract_sbatch_directives(script: Path) -> list[str]:
    directives = []
    for line in script.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("#SBATCH"):
            directives.append(line)
    return directives

def patch_target_name(raw_path: str | None) -> str | None:
    if not raw_path:
        return None
    return Path(str(raw_path)).name

def load_patch_request_from_descriptor(patch: dict[str, Any], script: Path) -> dict[str, Any]:
    patch_file = patch.get("patch_file")
    if patch_file:
        patch_path = resolve_path_with_bases(str(patch_file), [REPO_ROOT, script.parent])
        if not patch_path.is_file():
            raise SystemExit(f"missing patch file: {patch_path}")
        request = load_json(patch_path)
        if not isinstance(request, dict):
            raise SystemExit(f"patch file must contain an object: {patch_path}")
    else:
        request = {
            "schema_version": patch.get("schema_version", TEMPLATE_PATCH_SCHEMA_VERSION),
            "target_file": patch.get("target_file"),
            "output_file": patch.get("output_file"),
            "changes": patch.get("changes", []),
        }
    descriptor_target = patch_target_name(patch.get("target_file"))
    request_target = patch_target_name(request.get("target_file"))
    if descriptor_target and request_target and descriptor_target != request_target:
        raise SystemExit("config patch target_file does not match patch file target")
    if descriptor_target and not request.get("target_file"):
        request["target_file"] = patch.get("target_file")
    return request

def materialize_record_config_patches(
    record: dict[str, Any],
    script: Path,
    wrapper_dir: Path,
    overwrite: bool,
    remote_script: str | None,
) -> tuple[list[str], Path | None, list[str]]:
    source_lines: list[str] = []
    derived_files: list[str] = []
    target_script_override: Path | None = None
    raw_patches = record.get("config_patches", [])
    if not raw_patches:
        return source_lines, target_script_override, derived_files
    if not isinstance(raw_patches, list):
        raise SystemExit("record config_patches must be a list")
    for raw_patch in raw_patches:
        if not isinstance(raw_patch, dict):
            raise SystemExit("config_patches entries must be objects")
        request = load_patch_request_from_descriptor(raw_patch, script)
        target_hint = request.get("target_file")
        if not target_hint:
            raise SystemExit("config patch request must define target_file")
        target = resolve_path_with_bases(str(target_hint), [script.parent, REPO_ROOT])
        output = wrapper_dir / target.name
        if target.resolve() == script.resolve():
            if remote_script:
                raise SystemExit("config patch targeting the submit script is not supported with --remote-script")
            output = wrapper_dir / script.name
        report = evaluate_template_patch_request(
            request,
            target=target,
            output=output,
            apply=True,
        )
        if not report["safe"] or not report["applied"]:
            raise SystemExit(f"failed to materialize config patch for {target.name}")
        derived_files.append(str(output))
        if target.resolve() == script.resolve():
            target_script_override = output
        else:
            source_lines.append(f"source {shell_quote(str(output))}")
    return source_lines, target_script_override, derived_files

def materialize_submit_wrapper(
    record: dict[str, Any],
    script: Path,
    wrapper_dir: Path,
    overwrite: bool,
    remote_script: str | None = None,
) -> Path:
    exp_id = str(record["experiment_id"])
    path = wrapper_dir / f"{exp_id}.sbatch"
    if path.exists() and not overwrite:
        raise SystemExit(f"wrapper already exists: {path}")
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    parameters = record.get("parameters", {})
    if not isinstance(parameters, dict):
        parameters = {}
    source_lines, patched_script, derived_patch_files = materialize_record_config_patches(
        record, script, wrapper_dir, overwrite, remote_script
    )
    passthrough_names = {
        "NCCL_NET_PLUGIN",
        "NCCL_RAIL_PLANE",
        "NCCL_IB_NET_LATENCY",
        "NCCL_IB_ADAPTIVE_ROUTING",
        "NCCL_IB_TC",
        "NCCL_MIN_CTAS",
        "NCCL_SOCKET_IFNAME",
        "NCCL_P2P_DISABLE",
        "NCCL_SHM_DISABLE",
        "NCCL_CUMEM_ENABLE",
        "NVIDIA_IMEX_CHANNELS",
        "UCX_TLS",
        "UCX_NET_DEVICES",
        "OMPI_MCA_coll_hcoll_enable",
        "PMIX_MCA_gds",
        "NCCL_TEST_ITERS",
        "NCCL_TEST_MIN_BYTES",
        "NCCL_TEST_MAX_BYTES",
        "MLPERF_IMAGE",
        "NCCL_IB_HCA_OVERRIDE",
    }
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by tools/ai_tuning/ai_tuning.py experiment submit.",
        f"# experiment_id={exp_id}",
    ]
    lines.extend(extract_sbatch_directives(script))
    lines.append("set -euo pipefail")
    if record.get("objective"):
        lines.append(f"# objective={record['objective']}")
    for name in sorted(passthrough_names & set(parameters)):
        lines.append(f"export {name}={shell_quote(str(parameters[name]))}")
    if "NCCL_IB_HCA_OVERRIDE" in parameters and not str(parameters["NCCL_IB_HCA_OVERRIDE"]):
        lines = [line for line in lines if not line.startswith("export NCCL_IB_HCA_OVERRIDE=")]
    lines.extend(source_lines)
    if patched_script is not None:
        target_script = str(patched_script)
    elif remote_script:
        target_script = remote_script
    else:
        local_script = wrapper_dir / script.name
        if local_script.resolve() != script.resolve():
            if local_script.exists() and local_script.read_bytes() != script.read_bytes() and not overwrite:
                raise SystemExit(f"wrapper script copy already exists with different contents: {local_script}")
            shutil.copy2(script, local_script)
        target_script = f"$(cd -- \"$(dirname -- \"$0\")\" && pwd)/{script.name}"
    for derived_path in derived_patch_files:
        lines.append(f"# derived_patch={derived_path}")
    lines.append(f"exec bash {shell_quote(target_script)} \"$@\"")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)
    return path

def select_submit_candidates(
    states: dict[str, dict[str, Any]],
    statuses: set[str],
    capacity: int,
    experiment_ids: list[str],
) -> list[dict[str, Any]]:
    if capacity <= 0 and not experiment_ids:
        return []
    if experiment_ids:
        missing = [exp_id for exp_id in experiment_ids if exp_id not in states]
        if missing:
            raise SystemExit(f"unknown experiment id(s): {', '.join(missing)}")
        return [states[exp_id] for exp_id in experiment_ids]
    candidates = [
        record
        for record in states.values()
        if str(record.get("status", "")) in statuses
    ]
    candidates.sort(
        key=lambda record: (
            int(record.get("priority", 1_000_000)),
            str(record.get("experiment_id")),
        )
    )
    return candidates[:capacity]

def slurm_state_to_status(state: str) -> str:
    normalized = state.upper().split()[0]
    if normalized in {"PENDING", "CONFIGURING", "COMPLETING"}:
        return "submitted"
    if normalized in {"RUNNING", "RESIZING", "SUSPENDED"}:
        return "running"
    if normalized in {"COMPLETED"}:
        return "succeeded"
    if normalized in {"CANCELLED"}:
        return "cancelled"
    if normalized in {"FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL"}:
        return "failed"
    return "blocked"

def load_poll_statuses(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("jobs"), list):
        rows = payload["jobs"]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise SystemExit("poll status file must be a list or an object with a jobs list")
    statuses = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("slurm_job_id"):
            continue
        statuses[str(row["slurm_job_id"])] = row
    return statuses

def query_slurm_statuses(job_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not job_ids:
        return {}
    command = [
        "sacct",
        "-j",
        ",".join(job_ids),
        "--format=JobIDRaw,State,ExitCode,Elapsed,NodeList",
        "-P",
        "-n",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"sacct failed: {completed.stderr.strip()}")
    statuses: dict[str, dict[str, Any]] = {}
    for line in completed.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 5:
            continue
        job_id, state, exit_code, elapsed, nodes = parts[:5]
        if "." in job_id:
            continue
        statuses[job_id] = {
            "slurm_job_id": job_id,
            "slurm_state": state,
            "exit_code": exit_code,
            "elapsed": elapsed,
            "nodes": nodes,
        }
    return statuses

def copy_artifacts(source: Path, destination: Path, overwrite: bool) -> None:
    if not source.is_dir():
        raise SystemExit(f"missing artifact source directory: {source}")
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise SystemExit(f"artifact destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=overwrite)
        else:
            if target.exists() and not overwrite:
                raise SystemExit(f"artifact destination file exists: {target}")
            shutil.copy2(child, target)

def validate_collected_artifacts(destination: Path, benchmark: str, min_runs: int) -> list[str]:
    validator = load_validate_artifacts()
    errors: list[str] = []
    with contextlib.redirect_stdout(io.StringIO()):
        # require_nccl_runtime=False matches the warn-only default in
        # validate_artifacts.py; ai_tuning's offline collector never gates
        # the user on it. Flip to True only when the operator opts in via
        # `--require-nccl-runtime` on the validator CLI.
        validator.validate_raw_results_dir(
            errors,
            destination.resolve(),
            benchmark,
            min_runs,
            require_nccl_runtime=False,
        )
    return errors

FAILURE_PATTERNS: tuple[tuple[str, str, str], ...] = (
    ("failure", "nccl_ib_create_ah_no_device", "ibv_create_ah failed with error No such device"),
    ("failure", "nccl_ib_modify_qp_no_device", "ibv_modify_qp failed with 19 No such device"),
    ("failure", "nccl_ib_devx_rtr_qp", "ncclIbDevxRtrQp"),
    ("failure", "nccl_connection_closed", "Connection closed"),
    ("failure", "nccl_connection_refused", "Connection refused"),
    ("failure", "missing_probe_binary", "NCCL4RANK_ERROR missing_probe_binary"),
    ("failure", "slurm_more_processors_requested", "More processors requested than permitted"),
    ("failure", "hca_port_missing", "HCA_PORT_ERROR"),
    ("failure", "hca_inventory_all_zero_gids", r"\bnonzero=0\b"),
    ("failure", "hca_inventory_full_all_zero_gids", r"\bnonzero_gids=0\b"),
    ("info", "hca_inventory_usable_gid", "HCA_GID_USABLE"),
    ("warning", "hca_inventory_zero_gid", "HCA_GID_ZERO"),
    ("failure", "slurm_timeout", "TIMEOUT"),
    ("failure", "slurm_node_fail", "NODE_FAIL"),
)

def classify_failure_text(text: str) -> list[dict[str, Any]]:
    findings = []
    for severity, code, pattern in FAILURE_PATTERNS:
        count = len(re.findall(pattern, text, flags=re.IGNORECASE))
        if count:
            findings.append({"severity": severity, "code": code, "pattern": pattern, "count": count})
    return findings

def classify_artifacts(path: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if path.is_file():
        paths = [path]
    elif path.is_dir():
        paths = sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix in {".out", ".log", ".txt"}
        )
    else:
        return findings
    for item in paths:
        text = item.read_text(encoding="utf-8", errors="replace")
        for finding in classify_failure_text(text):
            finding = dict(finding)
            finding["path"] = str(item)
            findings.append(finding)
    return findings

def _proposal_candidate_index(proposal: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """v3.7 W10: index candidates by experiment_id_prefix for diffing."""
    out: dict[str, dict[str, Any]] = {}
    candidates = proposal.get("candidates")
    if not isinstance(candidates, list):
        return out
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        key = str(
            candidate.get("experiment_id_prefix")
            or candidate.get("experiment_id")
            or candidate.get("id")
            or ""
        ).strip()
        if key:
            out[key] = candidate
    return out

def _proposal_diff_payload(
    *,
    before_path: Path,
    after_path: Path,
) -> dict[str, Any]:
    """v3.7 W10: machine-readable diff between two proposal.json files."""
    import hashlib

    before = load_json(before_path)
    after = load_json(after_path)
    before_hash = hashlib.sha256(
        json.dumps(before, sort_keys=True).encode("utf-8")
    ).hexdigest()
    after_hash = hashlib.sha256(
        json.dumps(after, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if before_hash == after_hash:
        return {
            "schema_version": 1,
            "before": str(before_path),
            "after": str(after_path),
            "before_hash": before_hash,
            "after_hash": after_hash,
            "schema_version_match": True,
            "added_candidates": [],
            "removed_candidates": [],
            "changed_candidates": [],
            "unchanged": True,
        }
    before_idx = _proposal_candidate_index(before)
    after_idx = _proposal_candidate_index(after)
    added = sorted(set(after_idx) - set(before_idx))
    removed = sorted(set(before_idx) - set(after_idx))
    changed: list[dict[str, Any]] = []
    for key in sorted(set(before_idx) & set(after_idx)):
        b_params = dict(before_idx[key].get("parameters") or {})
        a_params = dict(after_idx[key].get("parameters") or {})
        if b_params == a_params:
            continue
        added_params = sorted(set(a_params) - set(b_params))
        removed_params = sorted(set(b_params) - set(a_params))
        changed_params = sorted(
            k
            for k in set(a_params) & set(b_params)
            if str(a_params[k]) != str(b_params[k])
        )
        changed.append(
            {
                "experiment_id_prefix": key,
                "added_parameters": [{"name": k, "value": a_params[k]} for k in added_params],
                "removed_parameters": [{"name": k, "value": b_params[k]} for k in removed_params],
                "changed_parameters": [
                    {"name": k, "from": b_params[k], "to": a_params[k]}
                    for k in changed_params
                ],
            }
        )
    return {
        "schema_version": 1,
        "before": str(before_path),
        "after": str(after_path),
        "before_hash": before_hash,
        "after_hash": after_hash,
        "schema_version_match": (
            before.get("schema_version") == after.get("schema_version")
        ),
        "added_candidates": added,
        "removed_candidates": removed,
        "changed_candidates": changed,
        "unchanged": False,
    }

def _proposal_diff_markdown(diff: dict[str, Any]) -> str:
    lines = []
    lines.append("# proposal diff")
    lines.append("")
    lines.append(f"- before : `{diff['before']}` ({diff['before_hash'][:12]})")
    lines.append(f"- after  : `{diff['after']}` ({diff['after_hash'][:12]})")
    lines.append(f"- schema_version_match : {diff['schema_version_match']}")
    lines.append("")
    if diff["unchanged"]:
        lines.append("Proposals are byte-identical.")
        return "\n".join(lines) + "\n"
    if diff["added_candidates"]:
        lines.append("## Added candidates")
        for k in diff["added_candidates"]:
            lines.append(f"- `{k}`")
        lines.append("")
    if diff["removed_candidates"]:
        lines.append("## Removed candidates")
        for k in diff["removed_candidates"]:
            lines.append(f"- `{k}`")
        lines.append("")
    if diff["changed_candidates"]:
        lines.append("## Changed candidates")
        for entry in diff["changed_candidates"]:
            lines.append(f"### {entry['experiment_id_prefix']}")
            if entry["added_parameters"]:
                added = ", ".join(
                    f"{p['name']}={p['value']}" for p in entry["added_parameters"]
                )
                lines.append(f"- added: {added}")
            if entry["removed_parameters"]:
                removed = ", ".join(
                    f"{p['name']}={p['value']}" for p in entry["removed_parameters"]
                )
                lines.append(f"- removed: {removed}")
            if entry["changed_parameters"]:
                changed = ", ".join(
                    f"{p['name']}: {p['from']}->{p['to']}" for p in entry["changed_parameters"]
                )
                lines.append(f"- changed: {changed}")
            lines.append("")
    return "\n".join(lines) + "\n"

def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1

def default_derived_path(target: Path) -> Path:
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return target.with_name(f"{target.stem}.cursor-{timestamp}{target.suffix}")

def resolve_path_with_bases(raw_path: str, bases: list[Path]) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    for base in bases:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (bases[0] / path).resolve()

def validate_patch_safety(original: str, patched: str) -> list[dict[str, Any]]:
    errors = []
    for code, pattern in FORBIDDEN_PATCH_PATTERNS:
        original_matches = len(re.findall(pattern, original, flags=re.IGNORECASE | re.MULTILINE))
        patched_matches = len(re.findall(pattern, patched, flags=re.IGNORECASE | re.MULTILINE))
        if patched_matches > original_matches:
            errors.append(
                {
                    "code": code,
                    "message": "patch introduces a command that requires manual operator handling",
                    "change_index": None,
                }
            )
    return errors

def validate_patched_template_structure(original: str, patched: str) -> list[dict[str, Any]]:
    errors = []
    structural_guards = (
        ("removed_sbatch_nodes", "#SBATCH --nodes", "#SBATCH --nodes declaration"),
        ("removed_sbatch_partition", "#SBATCH --partition", "#SBATCH --partition declaration"),
        ("removed_sbatch_gpus", "#SBATCH --gpus-per-node", "#SBATCH --gpus-per-node declaration"),
        ("removed_sbatch_gres", "#SBATCH --gres", "#SBATCH --gres declaration"),
        ("removed_clean_env_launch", "env -i", "clean env -i launch guard"),
        ("removed_run_stop_quality_target", "log_ppl <= 3.3", "documented LLaMA 3.1 8B quality target"),
    )
    for code, needle, label in structural_guards:
        if needle in original and needle not in patched:
            errors.append(
                {
                    "code": code,
                    "message": f"patch removes {label}",
                    "change_index": None,
                }
            )
    return errors

def evaluate_template_patch_request(
    request: dict[str, Any],
    *,
    target: Path,
    output: Path,
    apply: bool,
    audit_dir: Path | None = None,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    change_reports = []
    if request.get("schema_version", 1) != TEMPLATE_PATCH_SCHEMA_VERSION:
        errors.append(
            {
                "code": "unsupported_schema_version",
                "message": "template patch schema_version must be 1",
                "change_index": None,
            }
        )
    if ".git" in target.parts or ".git" in output.parts:
        errors.append(
            {
                "code": "git_metadata_target",
                "message": "template patch targets must not be inside .git",
                "change_index": None,
            }
        )
    if target == output:
        errors.append(
            {
                "code": "output_overwrites_target",
                "message": "derived output file must not be the same as the target file",
                "change_index": None,
            }
        )
    if not target.is_file():
        errors.append(
            {
                "code": "missing_target",
                "message": f"target file does not exist: {target}",
                "change_index": None,
            }
        )
        original = ""
    else:
        original = target.read_text(encoding="utf-8")

    patched = original
    changes = request.get("changes", [])
    if not isinstance(changes, list) or not changes:
        errors.append(
            {"code": "no_changes", "message": "patch must include at least one change", "change_index": None}
        )
        changes = []
    for index, change in enumerate(changes):
        match_context = str(change.get("match_context", ""))
        replacement = str(change.get("replacement", ""))
        report = {
            "index": index,
            "safe": True,
            "matched": False,
            "line_start": None,
            "line_end": None,
            "errors": [],
            "warnings": [],
            "rationale": change.get("rationale"),
            "expected_impact": change.get("expected_impact"),
            "risk": change.get("risk"),
        }
        if len(match_context.strip()) < 20 or "\n" not in match_context:
            report["safe"] = False
            report["errors"].append(
                {
                    "code": "insufficient_context",
                    "message": "match_context must include enough exact surrounding text",
                    "change_index": index,
                }
            )
        if not replacement.strip():
            report["safe"] = False
            report["errors"].append(
                {"code": "empty_replacement", "message": "replacement must not be empty", "change_index": index}
            )
        matches = list(re.finditer(re.escape(match_context), patched))
        if len(matches) == 1:
            match = matches[0]
            report["matched"] = True
            report["line_start"] = line_number(patched, match.start())
            report["line_end"] = line_number(patched, match.end())
            patched = patched[: match.start()] + replacement + patched[match.end() :]
        elif not matches:
            report["safe"] = False
            report["errors"].append(
                {"code": "no_match", "message": "match_context was not found", "change_index": index}
            )
        else:
            report["safe"] = False
            report["errors"].append(
                {
                    "code": "ambiguous_match",
                    "message": "match_context matched more than once",
                    "change_index": index,
                }
            )
        errors.extend(report["errors"])
        warnings.extend(report["warnings"])
        change_reports.append(report)

    errors.extend(validate_patch_safety(original, patched))
    errors.extend(validate_patched_template_structure(original, patched))
    safe = not errors and all(change["safe"] for change in change_reports)
    diff_preview: list[str] = []
    for index, change in enumerate(changes):
        diff_preview.append(f"--- change {index} before")
        diff_preview.extend(f"-{line}" for line in str(change.get("match_context", "")).splitlines())
        diff_preview.append(f"+++ change {index} after")
        diff_preview.extend(f"+{line}" for line in str(change.get("replacement", "")).splitlines())

    applied = False
    if apply:
        if not safe:
            warnings.append(
                {
                    "code": "apply_skipped",
                    "message": "unsafe or invalid patch was not applied",
                    "change_index": None,
                }
            )
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(patched, encoding="utf-8")
            applied = True

    report = {
        "schema_version": TEMPLATE_PATCH_SCHEMA_VERSION,
        "target_file": str(target),
        "derived_file": str(output),
        "safe": safe,
        "applied": applied,
        "errors": errors,
        "warnings": warnings,
        "changes": change_reports,
        "diff_preview": diff_preview,
    }
    if audit_dir:
        append_jsonl(
            audit_dir / "template_patch_audit.jsonl",
            {
                "schema_version": TEMPLATE_PATCH_SCHEMA_VERSION,
                "timestamp": utc_timestamp(),
                "patch_hash": stable_hash(request),
                "validation_report": report,
            },
        )
    return report

__all__ = [name for name in globals() if not name.startswith('__')]
