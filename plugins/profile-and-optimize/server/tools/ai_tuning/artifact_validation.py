"""Offline artifact checks used by the AI tuning CLI.

The validator checks the local bundle shape and the MLLOG fields needed by the
report and finalize commands. It does not claim submission compliance. Official
benchmark checkers remain the authority for a submission verdict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.shared.validation_schema import BENCHMARK_COLUMNS, REQUIRED_SUMMARY_FIELDS


def _mllog_records(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        errors.append(f"cannot read result log {path}: {error}")
        return records
    for line_number, line in enumerate(lines, 1):
        if ":::MLLOG" not in line:
            continue
        payload_start = line.find("{")
        if payload_start < 0:
            errors.append(f"{path}:{line_number}: malformed MLLOG record")
            continue
        try:
            payload = json.loads(line[payload_start:])
        except json.JSONDecodeError as error:
            errors.append(f"{path}:{line_number}: malformed MLLOG JSON: {error}")
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _last_record(records: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    return next((record for record in reversed(records) if record.get("key") == key), None)


def run_id_from_log_path(path: Path) -> str:
    """Return the run identifier from a conventional ``*_1.log`` path."""

    stem = path.stem
    for suffix in ("_01", "_1"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def validate_result_file(
    errors: list[str],
    benchmark: str,
    path: Path,
) -> tuple[str | None, float | None, float | None]:
    """Validate one MLLOG file and return status, elapsed minutes, and metric."""

    if not path.is_file():
        errors.append(f"missing raw result log: {path}")
        return None, None, None

    records = _mllog_records(path, errors)
    benchmark_record = _last_record(records, "submission_benchmark")
    observed_benchmark = benchmark_record.get("value") if benchmark_record else None
    if observed_benchmark != benchmark:
        errors.append(
            f"{path}: submission_benchmark {observed_benchmark!r} does not match {benchmark!r}"
        )

    start = _last_record(records, "run_start")
    stop = _last_record(records, "run_stop")
    if start is None:
        errors.append(f"{path}: missing run_start")
    if stop is None:
        errors.append(f"{path}: missing run_stop")

    status: str | None = None
    if stop is not None:
        metadata = stop.get("metadata")
        if isinstance(metadata, dict) and metadata.get("status") is not None:
            status = str(metadata["status"])
        elif stop.get("value") is not None:
            status = str(stop["value"])
        if status != "success":
            errors.append(f"{path}: run_stop status is {status!r}, expected 'success'")

    elapsed_minutes: float | None = None
    if start is not None and stop is not None:
        try:
            elapsed_minutes = (float(stop["time_ms"]) - float(start["time_ms"])) / 60_000
        except (KeyError, TypeError, ValueError):
            errors.append(f"{path}: run_start and run_stop require numeric time_ms values")

    metric: float | None = None
    metric_record = next(
        (
            record
            for record in reversed(records)
            if record.get("key")
            in {"eval_accuracy", "eval_loss", "final_loss", "log_ppl"}
        ),
        None,
    )
    if metric_record is None:
        errors.append(f"{path}: missing final eval_accuracy/log_ppl metric")
    else:
        try:
            metric = float(metric_record["value"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{path}: final eval_accuracy/log_ppl metric is not numeric")

    return status, elapsed_minutes, metric


def validate_checker_output(errors: list[str], path: Path) -> None:
    """Require one local checker output with an explicit success marker."""

    if not path.is_file():
        errors.append(f"missing checker output: {path}")
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    if "SUCCESS" not in text.upper():
        errors.append(f"checker output does not report success: {path}")


def validate_raw_results_dir(
    errors: list[str],
    raw_dir: Path,
    benchmark: str,
    min_runs: int,
    *,
    require_nccl_runtime: bool = False,
) -> None:
    """Validate the local files consumed by report and collect."""

    if not raw_dir.is_dir():
        errors.append(f"missing raw results directory: {raw_dir}")
        return
    run_logs = sorted(raw_dir.glob("*_1.log"))
    if len(run_logs) < min_runs:
        errors.append(
            f"expected at least {min_runs} raw run logs in {raw_dir}, found {len(run_logs)}"
        )
    if not list(raw_dir.glob("config_*.sh")):
        errors.append(f"missing config_*.sh in raw results directory: {raw_dir}")
    if not (raw_dir / "run.sub").is_file() and not list(raw_dir.glob("*.hyp")):
        errors.append(f"missing launcher run.sub or *.hyp in raw results directory: {raw_dir}")
    if not list(raw_dir.glob("container-env-*.log")) and not list(raw_dir.glob("*_env.log")):
        errors.append(f"missing container-env log in raw results directory: {raw_dir}")

    for run_log in run_logs:
        validate_result_file(errors, benchmark, run_log)
        run_id = run_id_from_log_path(run_log)
        validate_checker_output(errors, raw_dir / f"compliance_{run_id}.out")
        validate_checker_output(errors, raw_dir / f"audit_{run_id}.out")

    if require_nccl_runtime:
        runtime_text = "\n".join(
            path.read_text(encoding="utf-8", errors="replace") for path in run_logs
        )
        if "NCCL" not in runtime_text:
            errors.append(f"raw results directory has no NCCL runtime evidence: {raw_dir}")


def _validate_evidence_dir(errors: list[str], path: Path, label: str) -> None:
    if not path.is_dir():
        errors.append(f"missing {label} directory: {path}")
    elif not any(item.is_file() for item in path.rglob("*")):
        errors.append(f"{label} directory has no files: {path}")


def validate_fabric_evidence(errors: list[str], path: Path, require_clean: bool) -> None:
    """Check that an optional fabric evidence directory is present and populated."""

    _validate_evidence_dir(errors, path, "fabric evidence")
    if require_clean and path.is_dir():
        text = "\n".join(
            item.read_text(encoding="utf-8", errors="replace")
            for item in path.rglob("*")
            if item.is_file()
        )
        if any(marker in text.upper() for marker in ("FAIL", "ERROR")):
            errors.append(f"fabric evidence contains failure markers: {path}")


def validate_node_selection_dir(errors: list[str], path: Path) -> None:
    """Check that an optional node-selection evidence directory is populated."""

    _validate_evidence_dir(errors, path, "node selection evidence")


def validate_fabric_localization_dir(errors: list[str], path: Path) -> None:
    """Check that an optional fabric-localization evidence directory is populated."""

    _validate_evidence_dir(errors, path, "fabric localization evidence")


__all__ = (
    "BENCHMARK_COLUMNS",
    "REQUIRED_SUMMARY_FIELDS",
    "run_id_from_log_path",
    "validate_checker_output",
    "validate_fabric_evidence",
    "validate_fabric_localization_dir",
    "validate_node_selection_dir",
    "validate_raw_results_dir",
    "validate_result_file",
)
