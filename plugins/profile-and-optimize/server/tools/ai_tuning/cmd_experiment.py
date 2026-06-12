#!/usr/bin/env python3
# PROFILE_AND_OPTIMIZE_OPT_OUT: internal module; tools/ai_tuning/ai_tuning.py remains the canonical CLI.
"""Local helpers for LLM-assisted MLPerf tuning.

The commands in this file are intentionally offline: they inspect artifacts,
validate proposals, and materialize derived files, but they do not submit Slurm
jobs or mutate cluster state.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
from tools.shared.jsonutil import load_json

DEFAULT_SPACE = REPO_ROOT / "tuning" / "tuning-space.b200-llama31-8b.json"

# Safety constants live in a small sibling module so reviewers can audit
# the forbidden-pattern table and ledger-status enum without paging
# through the full tuner CLI. Re-exports preserve backwards compat for
# callers that still import these names from this module.
from tools.ai_tuning.helpers import (
    append_experiment_update,
    append_jsonl,
    choose_objective,
    classify_artifacts,
    copy_artifacts,
    error_code,
    experiment_id,
    infer_experiment_shape,
    latest_experiment_states,
    load_poll_statuses,
    materialize_submit_wrapper,
    normalize_candidate_parameters,
    normalize_priority,
    planned_submit_command,
    query_slurm_statuses,
    read_ledger_records,
    select_submit_candidates,
    slurm_state_to_status,
    utc_timestamp,
    validate_collected_artifacts,
    write_json,
)
from tools.ai_tuning.safety import (
    EXPERIMENT_LEDGER_SCHEMA_VERSION,
    EXPERIMENT_STATUSES,
)


def command_experiment_create(args: argparse.Namespace) -> int:
    space = load_json(Path(args.space))
    proposal = load_json(Path(args.proposal))
    selected_objective = choose_objective(space)
    candidates = proposal.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise SystemExit("proposal must contain a non-empty candidates list")

    existing_ids = {
        str(record.get("experiment_id"))
        for record in read_ledger_records(args.ledger)
        if record.get("experiment_id")
    }
    records = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        parameters = normalize_candidate_parameters(candidate)
        if not parameters:
            continue
        exp_id = experiment_id(str(space.get("id", "unknown")), index, parameters)
        if exp_id in existing_ids:
            continue
        record = {
            "schema_version": EXPERIMENT_LEDGER_SCHEMA_VERSION,
            "event": "created",
            "timestamp": utc_timestamp(),
            "experiment_id": exp_id,
            "status": "planned",
            "tuning_space_id": space.get("id"),
            "shape": infer_experiment_shape(parameters, space),
            "parameters": parameters,
            "rationale": candidate.get("rationale"),
            "risk": candidate.get("risk"),
            "priority": normalize_priority(candidate, index + 1),
            "objective": candidate.get("objective")
            or (selected_objective.get("name") if isinstance(selected_objective, dict) else None),
            "config_patches": candidate.get("config_patches", []),
            "optimizer_state": proposal.get("optimizer_state"),
            "owner": args.owner,
            "artifact_dir": args.artifact_root / exp_id if args.artifact_root else None,
            "slurm_job_id": None,
            "notes": args.notes,
        }
        if record["artifact_dir"] is not None:
            record["artifact_dir"] = str(record["artifact_dir"])
        append_jsonl(args.ledger, record)
        records.append(record)
    report = {
        "schema_version": EXPERIMENT_LEDGER_SCHEMA_VERSION,
        "ledger": str(args.ledger),
        "created_count": len(records),
        "skipped_existing_count": len(candidates) - len(records),
        "experiments": records,
    }
    write_json(report, args.output)
    return 0

def command_experiment_update(args: argparse.Namespace) -> int:
    if args.status not in EXPERIMENT_STATUSES:
        raise SystemExit(f"unknown status {args.status!r}; expected one of {sorted(EXPERIMENT_STATUSES)}")
    record = {
        "schema_version": EXPERIMENT_LEDGER_SCHEMA_VERSION,
        "event": "updated",
        "timestamp": utc_timestamp(),
        "experiment_id": args.experiment_id,
        "status": args.status,
        "slurm_job_id": args.slurm_job_id,
        "artifact_dir": str(args.artifact_dir) if args.artifact_dir else None,
        "notes": args.notes,
    }
    append_jsonl(args.ledger, record)
    write_json(record, args.output)
    return 0

def command_experiment_summary(args: argparse.Namespace) -> int:
    records = read_ledger_records(args.ledger)
    states = latest_experiment_states(records)
    experiments = sorted(
        states.values(),
        key=lambda record: (
            int(record.get("priority", 1_000_000)),
            str(record.get("experiment_id")),
        ),
    )
    counts: dict[str, int] = {}
    for record in experiments:
        status = str(record.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    if args.status:
        experiments = [record for record in experiments if record.get("status") == args.status]
    report = {
        "schema_version": EXPERIMENT_LEDGER_SCHEMA_VERSION,
        "ledger": str(args.ledger),
        "record_count": len(records),
        "experiment_count": len(states),
        "status_counts": counts,
        "experiments": experiments[: args.limit],
        "truncated": len(experiments) > args.limit,
    }
    write_json(report, args.output)
    return 0

def command_experiment_submit(args: argparse.Namespace) -> int:
    records = read_ledger_records(args.ledger)
    states = latest_experiment_states(records)
    active = sum(
        1
        for record in states.values()
        if record.get("status") in {"submitted", "running"}
    )
    capacity = max(args.max_concurrent - active, 0)
    statuses = set(args.status)
    candidates = select_submit_candidates(states, statuses, capacity, args.experiment_id)
    script = args.script.resolve()
    if not script.is_file():
        raise SystemExit(f"missing submit script: {script}")
    should_execute = args.execute and args.i_understand_this_submits_jobs
    if args.execute and not args.i_understand_this_submits_jobs:
        raise SystemExit("--execute requires --i-understand-this-submits-jobs")

    submissions = []
    updates = []
    for record in candidates:
        exp_id = str(record["experiment_id"])
        if record.get("config_patches") and not args.materialize_wrappers:
            raise SystemExit("experiments with config_patches require --materialize-wrappers")
        submit_script = script
        wrapper = None
        if args.materialize_wrappers:
            wrapper = materialize_submit_wrapper(
                record,
                script,
                args.wrapper_dir,
                args.overwrite_wrappers,
                args.remote_script,
            )
            submit_script = wrapper
        command = planned_submit_command(record, submit_script, args.sbatch_arg)
        result: dict[str, Any] = {
            "experiment_id": exp_id,
            "command": command,
            "executed": should_execute,
            "slurm_job_id": None,
            "wrapper": str(wrapper) if wrapper else None,
            "objective": record.get("objective"),
            "config_patches": record.get("config_patches"),
        }
        if should_execute:
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            stdout = completed.stdout.strip()
            stderr = completed.stderr.strip()
            result.update(
                {
                    "return_code": completed.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                }
            )
            if completed.returncode == 0 and stdout:
                job_id = stdout.splitlines()[0].split(";", 1)[0].strip()
                result["slurm_job_id"] = job_id
                updates.append(
                    append_experiment_update(
                        args.ledger,
                        exp_id,
                        "submitted",
                        slurm_job_id=job_id,
                        artifact_dir=record.get("artifact_dir"),
                        notes=args.notes,
                        extra={"submit_command": command, "wrapper": str(wrapper) if wrapper else None},
                    )
                )
            else:
                updates.append(
                    append_experiment_update(
                        args.ledger,
                        exp_id,
                        "blocked",
                        artifact_dir=record.get("artifact_dir"),
                        notes="sbatch failed during gated submit",
                        extra={"submit_command": command, "wrapper": str(wrapper) if wrapper else None, "submit_result": result},
                    )
                )
        submissions.append(result)

    report = {
        "schema_version": EXPERIMENT_LEDGER_SCHEMA_VERSION,
        "ledger": str(args.ledger),
        "script": str(script),
        "execute": should_execute,
        "active_count": active,
        "max_concurrent": args.max_concurrent,
        "selected_count": len(candidates),
        "updates_written": len(updates),
        "materialized_wrappers": args.materialize_wrappers,
        "submissions": submissions,
    }
    write_json(report, args.output)
    return 0

def command_experiment_poll(args: argparse.Namespace) -> int:
    records = read_ledger_records(args.ledger)
    states = latest_experiment_states(records)
    tracked = [
        record
        for record in states.values()
        if record.get("slurm_job_id")
        and (not args.status or record.get("status") in set(args.status))
    ]
    job_ids = sorted({str(record["slurm_job_id"]) for record in tracked})
    slurm_statuses = (
        load_poll_statuses(args.status_file)
        if args.status_file
        else query_slurm_statuses(job_ids)
    )
    updates = []
    observations = []
    for record in tracked:
        exp_id = str(record["experiment_id"])
        job_id = str(record["slurm_job_id"])
        slurm = slurm_statuses.get(job_id)
        if slurm is None:
            observations.append(
                {"experiment_id": exp_id, "slurm_job_id": job_id, "found": False}
            )
            continue
        new_status = str(slurm.get("status") or slurm_state_to_status(str(slurm.get("slurm_state", ""))))
        observation = {
            "experiment_id": exp_id,
            "slurm_job_id": job_id,
            "found": True,
            "previous_status": record.get("status"),
            "new_status": new_status,
            "slurm": slurm,
        }
        observations.append(observation)
        if new_status != record.get("status") and not args.no_update:
            updates.append(
                append_experiment_update(
                    args.ledger,
                    exp_id,
                    new_status,
                    slurm_job_id=job_id,
                    artifact_dir=record.get("artifact_dir"),
                    notes=args.notes,
                    extra={"slurm": slurm},
                )
            )
    report = {
        "schema_version": EXPERIMENT_LEDGER_SCHEMA_VERSION,
        "ledger": str(args.ledger),
        "tracked_job_count": len(tracked),
        "updates_written": len(updates),
        "observations": observations,
    }
    write_json(report, args.output)
    return 0

def command_experiment_collect(args: argparse.Namespace) -> int:
    records = read_ledger_records(args.ledger)
    states = latest_experiment_states(records)
    record = states.get(args.experiment_id)
    if record is None:
        raise SystemExit(f"unknown experiment id: {args.experiment_id}")
    destination = args.destination or (
        Path(str(record["artifact_dir"])) if record.get("artifact_dir") else None
    )
    if destination is None:
        raise SystemExit("collect requires --destination or an artifact_dir in the ledger")
    copy_artifacts(args.source, destination, args.overwrite)
    errors = []
    if args.validate_raw:
        errors = validate_collected_artifacts(destination, args.raw_benchmark, args.min_runs)
    failure_classifications = classify_artifacts(destination)
    status = "succeeded" if not errors else "failed"
    if any(item.get("severity") == "failure" for item in failure_classifications):
        status = "failed"
    update = append_experiment_update(
        args.ledger,
        args.experiment_id,
        status,
        slurm_job_id=record.get("slurm_job_id"),
        artifact_dir=str(destination),
        notes=args.notes,
        extra={
            "collection": {
                "source": str(args.source),
                "destination": str(destination),
                "raw_benchmark": args.raw_benchmark,
                "validation_errors": [
                    {"code": error_code(message), "message": message}
                    for message in errors
                ],
                "failure_classifications": failure_classifications,
            }
        },
    )
    report = {
        "schema_version": EXPERIMENT_LEDGER_SCHEMA_VERSION,
        "ledger": str(args.ledger),
        "experiment_id": args.experiment_id,
        "artifact_dir": str(destination),
        "validation_passed": not errors,
        "failure_classifications": failure_classifications,
        "validation_errors": [
            {"code": error_code(message), "message": message}
            for message in errors
        ],
        "update": update,
    }
    write_json(report, args.output)
    return 0 if not errors else 1
