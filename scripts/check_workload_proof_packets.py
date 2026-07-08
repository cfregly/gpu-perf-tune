#!/usr/bin/env python3
"""Validate neocloud workload proof packets.

The gate is intentionally dependency-free. The JSON Schema in schemas/ documents
the public shape, while this script enforces the completeness rules that matter
before a packet is shared with a skeptical buyer.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "workload_proof_packet_v1"
WORKFLOW_HANDOFF_SCHEMA_VERSION = "workflow_handoff_v1"
WORKFLOW_ATTACHMENT_ROLE = "workload_level_evidence"
ACCESS_STAGES = {"offline", "shadow", "supervised_pilot", "default"}
DEFAULT_PACKET_GLOBS = (
    "examples/**/workload-proof-packet.json",
    "workload-proof-packets/**/*.json",
    "experiments/artifacts/**/workload-proof-packet.json",
)
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}
SENTINELS = {
    "",
    "__fill__",
    "__fill",
    "fill__",
    "__todo__",
    "todo",
    "tbd",
    "unknown",
    "n/a",
    "na",
    "none",
    "null",
    "replace-me",
    "fill-me",
}
PATH_FIELDS = (
    "stdout_path",
    "stderr_path",
    "exit_path",
    "sanitized_env_path",
    "normalized_summary_path",
    "evidence_path",
)
PATH_LIST_FIELDS = ("raw_outputs", "logs", "profiler_traces")


@dataclass(frozen=True)
class Issue:
    path: Path
    pointer: str
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.pointer}: {self.message}"


def _load_json(path: Path) -> tuple[dict[str, Any] | None, list[Issue]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return None, [Issue(path, "$", f"failed to read JSON: {exc}")]
    if not isinstance(payload, dict):
        return None, [Issue(path, "$", "packet must be a JSON object")]
    return payload, []


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in SENTINELS
    if isinstance(value, list):
        return not value
    if isinstance(value, dict):
        return not value
    return False


def _at(payload: dict[str, Any], dotted: str) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _require_value(issues: list[Issue], packet_path: Path, payload: dict[str, Any], dotted: str) -> None:
    value = _at(payload, dotted)
    if _is_missing(value):
        issues.append(Issue(packet_path, dotted, "missing required value"))


def _require_non_empty_string(
    issues: list[Issue],
    packet_path: Path,
    payload: dict[str, Any],
    dotted: str,
) -> None:
    value = _at(payload, dotted)
    if _is_missing(value) or not isinstance(value, str):
        issues.append(Issue(packet_path, dotted, "must be a non-empty string"))


def _require_number(
    issues: list[Issue],
    packet_path: Path,
    payload: dict[str, Any],
    dotted: str,
    *,
    min_value: float = 0.0,
) -> None:
    value = _at(payload, dotted)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        issues.append(Issue(packet_path, dotted, "must be a number"))
        return
    if value < min_value:
        issues.append(Issue(packet_path, dotted, f"must be >= {min_value:g}"))


def _require_bool(issues: list[Issue], packet_path: Path, payload: dict[str, Any], dotted: str) -> None:
    value = _at(payload, dotted)
    if not isinstance(value, bool):
        issues.append(Issue(packet_path, dotted, "must be true or false"))


def _require_enum(
    issues: list[Issue],
    packet_path: Path,
    payload: dict[str, Any],
    dotted: str,
    allowed: set[str],
) -> None:
    value = _at(payload, dotted)
    if not isinstance(value, str) or value not in allowed:
        issues.append(Issue(packet_path, dotted, f"must be one of {sorted(allowed)}"))


def _require_list_of_objects(
    issues: list[Issue],
    packet_path: Path,
    payload: dict[str, Any],
    dotted: str,
) -> list[dict[str, Any]]:
    value = _at(payload, dotted)
    if not isinstance(value, list) or not value:
        issues.append(Issue(packet_path, dotted, "must be a non-empty list"))
        return []
    out: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            issues.append(Issue(packet_path, f"{dotted}[{index}]", "must be an object"))
            continue
        out.append(item)
    return out


def _require_non_empty_string_list(
    issues: list[Issue],
    packet_path: Path,
    payload: dict[str, Any],
    dotted: str,
) -> None:
    value = _at(payload, dotted)
    if not isinstance(value, list) or not value:
        issues.append(Issue(packet_path, dotted, "must be a non-empty list"))
        return
    for index, item in enumerate(value):
        if _is_missing(item) or not isinstance(item, str):
            issues.append(Issue(packet_path, f"{dotted}[{index}]", "must be a non-empty string"))


def _path_exists(base: Path, value: str) -> bool:
    if value.startswith(("http://", "https://", "s3://", "gs://")):
        return True
    return (base / value).exists()


def _validate_path(
    issues: list[Issue],
    packet_path: Path,
    base: Path,
    pointer: str,
    value: Any,
    *,
    require_existing_paths: bool,
) -> None:
    if _is_missing(value) or not isinstance(value, str):
        issues.append(Issue(packet_path, pointer, "missing required artifact path"))
        return
    if require_existing_paths and not _path_exists(base, value):
        issues.append(Issue(packet_path, pointer, f"artifact path does not exist: {value}"))


def _validate_path_list(
    issues: list[Issue],
    packet_path: Path,
    base: Path,
    pointer: str,
    values: Any,
    *,
    require_existing_paths: bool,
) -> None:
    if not isinstance(values, list) or not values:
        issues.append(Issue(packet_path, pointer, "must be a non-empty artifact path list"))
        return
    for index, value in enumerate(values):
        _validate_path(
            issues,
            packet_path,
            base,
            f"{pointer}[{index}]",
            value,
            require_existing_paths=require_existing_paths,
        )


def _validate_nested_artifact_paths(
    issues: list[Issue],
    packet_path: Path,
    base: Path,
    value: Any,
    pointer: str,
    *,
    require_existing_paths: bool,
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_pointer = f"{pointer}.{key}" if pointer else key
            if key in PATH_FIELDS:
                _validate_path(
                    issues,
                    packet_path,
                    base,
                    child_pointer,
                    child,
                    require_existing_paths=require_existing_paths,
                )
            elif key in PATH_LIST_FIELDS:
                _validate_path_list(
                    issues,
                    packet_path,
                    base,
                    child_pointer,
                    child,
                    require_existing_paths=require_existing_paths,
                )
            else:
                _validate_nested_artifact_paths(
                    issues,
                    packet_path,
                    base,
                    child,
                    child_pointer,
                    require_existing_paths=require_existing_paths,
                )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_nested_artifact_paths(
                issues,
                packet_path,
                base,
                child,
                f"{pointer}[{index}]",
                require_existing_paths=require_existing_paths,
            )


def _validate_command_tuple(
    issues: list[Issue],
    packet_path: Path,
    command: dict[str, Any],
    index: int,
) -> None:
    prefix = f"run.commands[{index}]"
    for key in ("id", "cwd", "command", "stdout_path", "stderr_path", "exit_path"):
        if _is_missing(command.get(key)):
            issues.append(Issue(packet_path, f"{prefix}.{key}", "missing required command tuple field"))
    exit_code = command.get("exit_code")
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        issues.append(Issue(packet_path, f"{prefix}.exit_code", "must be an integer"))


def _validate_gates(
    issues: list[Issue],
    packet_path: Path,
    payload: dict[str, Any],
    *,
    require_verdict: bool,
) -> None:
    gates = _require_list_of_objects(issues, packet_path, payload, "gates")
    required_gate_seen = False
    for index, gate in enumerate(gates):
        prefix = f"gates[{index}]"
        for key in ("name", "evidence_path"):
            if _is_missing(gate.get(key)):
                issues.append(Issue(packet_path, f"{prefix}.{key}", "missing required gate field"))
        for key in ("passed", "required_for_verdict"):
            if not isinstance(gate.get(key), bool):
                issues.append(Issue(packet_path, f"{prefix}.{key}", "must be true or false"))
        if gate.get("required_for_verdict") is True:
            required_gate_seen = True
            if require_verdict and gate.get("passed") is not True:
                issues.append(Issue(packet_path, f"{prefix}.passed", "required verdict gate did not pass"))
    if not required_gate_seen:
        issues.append(Issue(packet_path, "gates", "at least one gate must be required_for_verdict"))


def _validate_workflow_handoff(
    issues: list[Issue],
    packet_path: Path,
    payload: dict[str, Any],
    *,
    require_handoff: bool,
) -> None:
    handoff = payload.get("workflow_handoff")
    if handoff is None:
        if require_handoff:
            issues.append(Issue(packet_path, "workflow_handoff", "missing required workflow handoff"))
        return
    if not isinstance(handoff, dict) or not handoff:
        issues.append(Issue(packet_path, "workflow_handoff", "must be an object"))
        return
    allowed_keys = {
        "schema_version",
        "attachment_role",
        "integration_id",
        "workflow_name",
        "current_access_stage",
        "target_access_stage",
        "packet_proves",
        "workflow_system_proves",
        "not_proven",
        "handoff_notes",
    }
    for key in sorted(set(handoff) - allowed_keys):
        issues.append(Issue(packet_path, f"workflow_handoff.{key}", "unexpected handoff field"))
    for dotted in (
        "workflow_handoff.integration_id",
        "workflow_handoff.workflow_name",
        "workflow_handoff.handoff_notes",
    ):
        _require_non_empty_string(issues, packet_path, payload, dotted)
    _require_enum(
        issues,
        packet_path,
        payload,
        "workflow_handoff.schema_version",
        {WORKFLOW_HANDOFF_SCHEMA_VERSION},
    )
    _require_enum(
        issues,
        packet_path,
        payload,
        "workflow_handoff.attachment_role",
        {WORKFLOW_ATTACHMENT_ROLE},
    )
    _require_enum(issues, packet_path, payload, "workflow_handoff.current_access_stage", ACCESS_STAGES)
    _require_enum(issues, packet_path, payload, "workflow_handoff.target_access_stage", ACCESS_STAGES)
    for dotted in (
        "workflow_handoff.packet_proves",
        "workflow_handoff.workflow_system_proves",
        "workflow_handoff.not_proven",
    ):
        _require_non_empty_string_list(issues, packet_path, payload, dotted)


def _validate_required_shape(
    issues: list[Issue],
    packet_path: Path,
    payload: dict[str, Any],
) -> None:
    for dotted in (
        "packet_id",
        "created_at_utc",
        "owner",
        "audience",
        "workload.name",
        "workload.model_id",
        "workload.model_source",
        "workload.dataset",
        "workload.success_criteria",
        "target.neocloud",
        "target.region",
        "target.zone",
        "target.gpu_sku",
        "target.topology",
        "target.interconnect",
        "target.availability_source",
        "stack.container_image",
        "stack.serving_engine",
        "stack.engine_version",
        "stack.cuda_version",
        "stack.driver_version",
        "stack.nccl_version",
        "run.environment.sanitized_env_path",
        "run.environment.secrets_policy",
        "baseline.name",
        "baseline.kind",
        "baseline.source",
        "evidence.normalized_summary_path",
        "evidence.source_repo",
        "evidence.source_commit",
        "verdict.claim",
        "verdict.proof_scope",
        "verdict.next_lever",
    ):
        _require_value(issues, packet_path, payload, dotted)

    for dotted in (
        "workload.input_tokens",
        "workload.output_tokens",
        "workload.concurrency",
        "workload.request_count",
        "target.gpu_count",
        "measurements.latency.ms_p50",
        "measurements.latency.ms_p95",
        "measurements.latency.tpot_ms",
        "measurements.throughput.output_tokens_per_second",
        "measurements.throughput.total_tokens_per_second",
        "measurements.throughput.tokens_per_second_per_gpu",
        "measurements.utilization.gpu_sm_pct",
        "measurements.utilization.gpu_memory_pct",
        "measurements.cost.gpu_hour_usd",
        "measurements.cost.dollars_per_million_tokens",
        "measurements.reliability.success_rate",
        "baseline.measurements.latency_ms_p95",
        "baseline.measurements.throughput_tokens_per_second",
        "baseline.measurements.cost_dollars_per_million_tokens",
    ):
        _require_number(issues, packet_path, payload, dotted)

    for dotted in (
        "measurements.utilization.power_watts",
        "measurements.cost.tokens_per_dollar",
        "measurements.reliability.error_count",
    ):
        _require_number(issues, packet_path, payload, dotted, min_value=0.0)

    _require_enum(issues, packet_path, payload, "schema_version", {SCHEMA_VERSION})
    _require_enum(issues, packet_path, payload, "status", {"draft", "verdict"})
    _require_bool(issues, packet_path, payload, "baseline.comparable")
    _require_bool(issues, packet_path, payload, "evidence.source_dirty")

    launch_flags = _at(payload, "stack.launch_flags")
    if not isinstance(launch_flags, list) or not launch_flags:
        issues.append(Issue(packet_path, "stack.launch_flags", "must be a non-empty list"))

    commands = _require_list_of_objects(issues, packet_path, payload, "run.commands")
    for index, command in enumerate(commands):
        _validate_command_tuple(issues, packet_path, command, index)

    for dotted in (
        "evidence.raw_outputs",
        "evidence.logs",
        "evidence.profiler_traces",
        "verdict.not_proven",
        "verdict.caveats",
    ):
        value = _at(payload, dotted)
        if not isinstance(value, list) or not value:
            issues.append(Issue(packet_path, dotted, "must be a non-empty list"))


def validate_packet(
    packet_path: Path,
    *,
    require_verdict: bool = False,
    require_workflow_handoff: bool = False,
    require_existing_paths: bool = True,
) -> list[Issue]:
    payload, issues = _load_json(packet_path)
    if payload is None:
        return issues
    _validate_required_shape(issues, packet_path, payload)
    _validate_nested_artifact_paths(
        issues,
        packet_path,
        packet_path.parent,
        payload,
        "",
        require_existing_paths=require_existing_paths,
    )
    _validate_workflow_handoff(
        issues,
        packet_path,
        payload,
        require_handoff=require_workflow_handoff,
    )
    status = payload.get("status")
    if require_verdict and status != "verdict":
        issues.append(Issue(packet_path, "status", "--require-verdict requires status to be 'verdict'"))
    strict_verdict = require_verdict or status == "verdict"
    _validate_gates(issues, packet_path, payload, require_verdict=strict_verdict)
    if strict_verdict:
        if payload.get("evidence", {}).get("source_dirty") is not False:
            issues.append(Issue(packet_path, "evidence.source_dirty", "verdict packets require a clean source"))
        if payload.get("baseline", {}).get("comparable") is not True:
            issues.append(Issue(packet_path, "baseline.comparable", "verdict packets require a comparable baseline"))
    return issues


def _default_packets() -> list[Path]:
    packets: list[Path] = []
    for pattern in DEFAULT_PACKET_GLOBS:
        for path in ROOT.glob(pattern):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.is_file():
                packets.append(path)
    return sorted(set(packets))


def _self_test() -> list[str]:
    findings: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for rel in (
            "commands/00-run.cmd",
            "commands/00-run.stdout",
            "commands/00-run.stderr",
            "commands/00-run.exit",
            "env/sanitized.json",
            "raw/out.json",
            "logs/server.log",
            "profiles/trace.json",
            "summary/normalized.json",
            "gates/completeness.json",
        ):
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")
        packet = {
            "schema_version": SCHEMA_VERSION,
            "packet_id": "self-test",
            "status": "verdict",
            "created_at_utc": "2026-07-06T00:00:00Z",
            "owner": "test",
            "audience": "neocloud buyer",
            "workload": {
                "name": "aa-1k",
                "model_id": "example/model",
                "model_source": "hf",
                "dataset": "aa",
                "input_tokens": 1000,
                "output_tokens": 256,
                "concurrency": 1,
                "request_count": 8,
                "success_criteria": "complete all requests",
            },
            "target": {
                "neocloud": "example",
                "region": "test",
                "zone": "test-a",
                "gpu_sku": "B200",
                "gpu_count": 1,
                "topology": "1x",
                "interconnect": "single-node",
                "availability_source": "operator observed",
            },
            "stack": {
                "container_image": "example:latest",
                "serving_engine": "vllm",
                "engine_version": "0.0.0",
                "cuda_version": "13.0",
                "driver_version": "580.0",
                "nccl_version": "2.28.0",
                "launch_flags": ["--tensor-parallel-size=1"],
            },
            "run": {
                "commands": [
                    {
                        "id": "00-run",
                        "cwd": "/tmp",
                        "command": "python bench.py",
                        "stdout_path": "commands/00-run.stdout",
                        "stderr_path": "commands/00-run.stderr",
                        "exit_path": "commands/00-run.exit",
                        "exit_code": 0,
                    }
                ],
                "environment": {
                    "sanitized_env_path": "env/sanitized.json",
                    "secrets_policy": "redacted before capture",
                },
            },
            "measurements": {
                "latency": {"ms_p50": 1.0, "ms_p95": 2.0, "tpot_ms": 3.0},
                "throughput": {
                    "output_tokens_per_second": 4.0,
                    "total_tokens_per_second": 5.0,
                    "tokens_per_second_per_gpu": 4.0,
                },
                "utilization": {"gpu_sm_pct": 50.0, "gpu_memory_pct": 40.0, "power_watts": 500.0},
                "cost": {
                    "gpu_hour_usd": 4.0,
                    "tokens_per_dollar": 1000.0,
                    "dollars_per_million_tokens": 20.0,
                },
                "reliability": {"success_rate": 1.0, "error_count": 0},
            },
            "baseline": {
                "name": "same-stack-baseline",
                "kind": "same-cloud-previous",
                "source": "packet",
                "comparable": True,
                "measurements": {
                    "latency_ms_p95": 3.0,
                    "throughput_tokens_per_second": 3.0,
                    "cost_dollars_per_million_tokens": 30.0,
                },
            },
            "evidence": {
                "raw_outputs": ["raw/out.json"],
                "logs": ["logs/server.log"],
                "profiler_traces": ["profiles/trace.json"],
                "normalized_summary_path": "summary/normalized.json",
                "source_repo": "example/repo",
                "source_commit": "0123456789abcdef",
                "source_dirty": False,
            },
            "gates": [
                {
                    "name": "completeness",
                    "passed": True,
                    "required_for_verdict": True,
                    "evidence_path": "gates/completeness.json",
                }
            ],
            "verdict": {
                "claim": "synthetic packet validates",
                "proof_scope": "self-test only",
                "not_proven": ["real hardware performance"],
                "caveats": ["synthetic data"],
                "next_lever": "validate a real packet",
            },
            "workflow_handoff": {
                "schema_version": WORKFLOW_HANDOFF_SCHEMA_VERSION,
                "attachment_role": WORKFLOW_ATTACHMENT_ROLE,
                "integration_id": "workflow-self-test",
                "workflow_name": "Synthetic inference workload proof",
                "current_access_stage": "offline",
                "target_access_stage": "shadow",
                "packet_proves": ["synthetic target, stack, command, measurement, and baseline shape"],
                "workflow_system_proves": ["workflow authority, replay, gates, and promotion when attached"],
                "not_proven": ["real B200 performance"],
                "handoff_notes": "Synthetic handoff block for validator coverage.",
            },
        }
        packet_path = root / "workload-proof-packet.json"
        packet_path.write_text(json.dumps(packet), encoding="utf-8")
        valid_issues = validate_packet(
            packet_path,
            require_verdict=True,
            require_workflow_handoff=True,
        )
        if valid_issues:
            findings.append("self-test valid packet failed: " + " | ".join(i.render() for i in valid_issues))
        packet["workload"]["model_id"] = "unknown"
        packet_path.write_text(json.dumps(packet), encoding="utf-8")
        invalid_issues = validate_packet(
            packet_path,
            require_verdict=True,
            require_workflow_handoff=True,
        )
        if not any(issue.pointer == "workload.model_id" for issue in invalid_issues):
            findings.append("self-test invalid packet did not fail workload.model_id")
        packet["workload"]["model_id"] = "example/model"
        packet.pop("workflow_handoff")
        packet_path.write_text(json.dumps(packet), encoding="utf-8")
        missing_handoff_issues = validate_packet(packet_path, require_workflow_handoff=True)
        if not any(issue.pointer == "workflow_handoff" for issue in missing_handoff_issues):
            findings.append("self-test missing workflow handoff did not fail")
        packet["workflow_handoff"] = {
            "schema_version": WORKFLOW_HANDOFF_SCHEMA_VERSION,
            "attachment_role": WORKFLOW_ATTACHMENT_ROLE,
            "integration_id": "workflow-self-test",
            "workflow_name": "Synthetic inference workload proof",
            "current_access_stage": "offline",
            "target_access_stage": "pilot",
            "packet_proves": [],
            "workflow_system_proves": ["workflow authority"],
            "not_proven": ["real hardware performance"],
            "handoff_notes": "Invalid handoff block for self-test coverage.",
        }
        packet_path.write_text(json.dumps(packet), encoding="utf-8")
        invalid_handoff_issues = validate_packet(packet_path, require_workflow_handoff=True)
        if not any(issue.pointer == "workflow_handoff.target_access_stage" for issue in invalid_handoff_issues):
            findings.append("self-test invalid workflow handoff stage did not fail")
        if not any(issue.pointer == "workflow_handoff.packet_proves" for issue in invalid_handoff_issues):
            findings.append("self-test invalid workflow handoff list did not fail")
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packets", nargs="*", type=Path, help="Packets to validate")
    parser.add_argument(
        "--require-verdict",
        action="store_true",
        help="Fail unless packets are verdict-ready and all required verdict gates pass",
    )
    parser.add_argument(
        "--require-workflow-handoff",
        action="store_true",
        help="Fail unless packets include valid workload-level workflow handoff metadata",
    )
    parser.add_argument(
        "--no-path-check",
        action="store_true",
        help="Check packet values but do not require referenced artifact paths to exist",
    )
    parser.add_argument("--self-test", action="store_true", help="Run validator self-tests first")
    args = parser.parse_args(argv)

    findings: list[str] = []
    if args.self_test:
        findings.extend(_self_test())

    packets = [p.resolve() for p in args.packets] if args.packets else _default_packets()
    if not packets:
        findings.append("no workload proof packets found")
    for packet in packets:
        issues = validate_packet(
            packet,
            require_verdict=args.require_verdict,
            require_workflow_handoff=args.require_workflow_handoff,
            require_existing_paths=not args.no_path_check,
        )
        findings.extend(issue.render() for issue in issues)

    if findings:
        print(f"[FAIL] workload proof packet gate found {len(findings)} issue(s):", file=sys.stderr)
        for finding in findings:
            print(f"  {finding}", file=sys.stderr)
        return 1
    print(f"[ok] workload proof packet gate passed ({len(packets)} packet(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
