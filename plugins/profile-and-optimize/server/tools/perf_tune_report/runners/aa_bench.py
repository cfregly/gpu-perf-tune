"""aa backend: drives the Artificial Analysis (AA) workload shapes via AIPerf.

One AA shape (``aa-1k`` / ``aa-10k`` / ``aa-100k``) maps to one perf-report
cell, swept across the cell's concurrencies. The command builder, dataset
generator, and report normalizer all live in
[`aa_workload.py`](aa_workload.py); this runner is the thin orchestration
layer that mirrors [`aiperf_bench.py`](aiperf_bench.py):

- builds one ``aiperf profile`` invocation per concurrency for the shape,
- runs them locally (default; the AA use case targets a provider endpoint
  URL) or wrapped in ``kubectl exec`` for in-cluster parity,
- parses the per-concurrency ``profile_export_aiperf.json`` into AtlasCell
  rows under ``<campaign>/cells/<cell-id>/normalized.json``.

Status mapping mirrors the other runners (full / partial / failed).
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from tools.perf_tune_report.runners.aa_workload import (
    AA_SHAPES,
    API_KEY_ENV,
    DEFAULT_CUSTOM_DATASET_TYPE,
    MODE_DATASET_REPLAY,
    MODE_SYNTHETIC,
    REDACTED_API_KEY,
    build_aiperf_auth_config,
    build_aiperf_command,
    generate_aa_dataset,
    normalize_outputs,
    redact_aiperf_command,
    validate_aiperf_command,
    write_aiperf_auth_config,
)
from tools.perf_tune_report.runners.common import (
    CellConfig,
    write_backend_file,
    write_normalized_json,
    write_status_file,
)
from tools.perf_tune_report.runners.settle import prewarm as settle_prewarm
from tools.perf_tune_report.runners.spec_metrics import (
    SpecMetricsCapture,
    attach_windows_to_rows,
)
from tools.perf_tune_report.schema import BACKEND_AIPERF, STATUS_FAILED


@dataclass(frozen=True)
class AaResult:
    cell_dir: Path
    status: str
    row_count: int
    commands: list[list[str]]
    dry_run: bool
    shape: str
    mode: str
    dataset_info: dict | None = None
    # Per-concurrency spec-decode windows ({concurrency: {al, accept_rate, ...}});
    # empty when spec decode is off / not scraped (local mode, dry-run, opt-out).
    spec_windows: dict[int, dict] | None = None


def _wrap_kube(inner: list[str], kube: dict) -> list[str]:
    return [
        "kubectl",
        "exec",
        "-n",
        kube["namespace"],
        "--context",
        kube["kube_context"],
        kube["bench_pod"],
        "--",
        *inner,
    ]


def _wrap_kube_with_stdin_auth_config(inner: list[str], kube: dict) -> list[str]:
    """Run in the pod with the key and secret-free config read from stdin."""
    return [
        "kubectl",
        "exec",
        "-i",
        "-n",
        kube["namespace"],
        "--context",
        kube["kube_context"],
        kube["bench_pod"],
        "--",
        "sh",
        "-c",
        (
            'umask 077; cfg=$(mktemp "${TMPDIR:-/tmp}/aiperf-auth.XXXXXX.json") || exit 1; '
            "trap 'rm -f \"$cfg\"' EXIT; "
            f"IFS= read -r {API_KEY_ENV} || exit 2; export {API_KEY_ENV}; "
            'cat > "$cfg" || exit 1; "$@" --config "$cfg"'
        ),
        "sh",
        *inner,
    ]


def run_cell(
    cell: CellConfig,
    campaign_dir: Path,
    *,
    shape_name: str,
    model: str,
    url: str,
    endpoint: str = "/v1/chat/completions",
    endpoint_type: str = "chat",
    api_key: str | None = None,
    tokenizer: str | None = None,
    tokenizer_trust_remote_code: bool = True,
    mode: str = MODE_SYNTHETIC,
    request_count: int = 10,
    custom_dataset_type: str = DEFAULT_CUSTOM_DATASET_TYPE,
    extra_output_controls: bool = True,
    input_file: str | None = None,
    dataset_count: int | None = None,
    aiperf_cmd: list[str] | None = None,
    kube: dict | None = None,
    spec_scrape: bool = True,
    prewarm_shapes: list[str] | None = None,
    burn_in: bool = False,
    settle_s: int = 30,
    dry_run: bool = False,
    subprocess_runner=subprocess.run,
    sleeper=time.sleep,
) -> AaResult:
    """Run one AA-shape cell across the cell's concurrencies."""
    if shape_name not in AA_SHAPES:
        raise ValueError(f"unknown AA shape {shape_name!r}; expected one of {sorted(AA_SHAPES)}")
    shape = AA_SHAPES[shape_name]
    aiperf_cmd = list(aiperf_cmd) if aiperf_cmd else ["aiperf"]
    validate_aiperf_command(aiperf_cmd)
    if api_key and ("\n" in api_key or "\r" in api_key):
        raise ValueError("API keys must not contain newline characters")

    cell_dir = campaign_dir / "cells" / cell.cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    write_backend_file(cell_dir, BACKEND_AIPERF)
    raw_dir = cell_dir / "raw"

    # In dataset-replay mode, generate the JSONL once unless the operator
    # pre-staged one (required for kube mode, where the path is in-pod).
    dataset_info: dict | None = None
    resolved_input_file = input_file
    if mode == MODE_DATASET_REPLAY and not resolved_input_file:
        if kube is not None:
            raise ValueError("dataset-replay + kube requires a pre-staged in-pod --input-file")
        dataset_path = cell_dir / "dataset" / f"{shape.name}.jsonl"
        if not dry_run:
            dataset_info = generate_aa_dataset(shape, dataset_count or request_count, dataset_path)
        resolved_input_file = str(dataset_path)

    commands_dir = cell_dir / "commands"
    commands_dir.mkdir(exist_ok=True)
    runtime_commands: list[list[str]] = []
    auth_configs: list[str | None] = []
    for c in cell.concurrencies:
        if kube is not None:
            artifact_dir = f"/tmp/aa-{cell.cell_id}-c{c}"
        else:
            artifact_dir = str(raw_dir / f"c{c}")
        inner = build_aiperf_command(
            shape,
            aiperf_cmd=aiperf_cmd,
            model=model,
            url=url,
            output_artifact_dir=artifact_dir,
            endpoint=endpoint,
            endpoint_type=endpoint_type,
            concurrency=c,
            request_count=request_count,
            tokenizer=tokenizer,
            tokenizer_trust_remote_code=tokenizer_trust_remote_code,
            mode=mode,
            input_file=resolved_input_file,
            custom_dataset_type=custom_dataset_type,
            extra_output_controls=extra_output_controls,
        )
        auth_config_text: str | None = None
        if api_key:
            auth_config_path = commands_dir / f"aa-c{c}.aiperf.json"
            auth_config_text = write_aiperf_auth_config(
                auth_config_path,
                build_aiperf_auth_config(
                    shape,
                    model=model,
                    url=url,
                    output_artifact_dir=artifact_dir,
                    endpoint=endpoint,
                    endpoint_type=endpoint_type,
                    concurrency=c,
                    request_count=request_count,
                    mode=mode,
                    input_file=resolved_input_file,
                    custom_dataset_type=custom_dataset_type,
                    extra_output_controls=extra_output_controls,
                ),
            )

        if kube is None:
            runtime_commands.append([*inner, "--config", str(auth_config_path)] if api_key else inner)
        elif api_key:
            runtime_commands.append(_wrap_kube_with_stdin_auth_config(inner, kube))
        else:
            runtime_commands.append(_wrap_kube(inner, kube))
        auth_configs.append(auth_config_text)

    # Credentials stay out of argv. Evidence files and result payloads receive
    # independent, redacted copies as a second boundary.
    evidence_commands = [redact_aiperf_command(command) for command in runtime_commands]

    (commands_dir / "aa-sweep.cmd").write_text("\n".join(shlex.join(command) for command in evidence_commands) + "\n")

    if dry_run:
        write_status_file(cell_dir, STATUS_FAILED)
        return AaResult(
            cell_dir=cell_dir,
            status="dry-run",
            row_count=0,
            commands=evidence_commands,
            dry_run=True,
            shape=shape.name,
            mode=mode,
            dataset_info=dataset_info,
        )

    raw_dir.mkdir(exist_ok=True)

    def run_aiperf(command: list[str], auth_config: str | None):
        kwargs: dict[str, object] = {
            "capture_output": True,
            "text": True,
            "check": False,
        }
        if api_key:
            if kube is None:
                child_env = os.environ.copy()
                child_env[API_KEY_ENV] = api_key
                kwargs["env"] = child_env
            else:
                if auth_config is None:
                    raise RuntimeError("missing AIPerf auth config for Kubernetes run")
                kwargs["input"] = f"{api_key}\n{auth_config}"
        return subprocess_runner(command, **kwargs)

    def redact_output(text: str | None) -> str:
        output = text or ""
        return output.replace(api_key, REDACTED_API_KEY) if api_key else output

    # Spec-decode AL capture (kube mode only): bracket each per-concurrency
    # run with a /metrics scrape from the bench pod so AL / accept-rate land
    # at the (cell, concurrency) atlas row grain, first-class instead of via
    # ad-hoc shell scrapes around the whole cell window.
    spec_capture: SpecMetricsCapture | None = None
    if kube is not None and spec_scrape:
        spec_capture = SpecMetricsCapture(
            cell_dir,
            url,
            kube_wrap=lambda inner: _wrap_kube(inner, kube),
            subprocess_runner=subprocess_runner,
        )

    # Settle discipline (kube mode, opt-in): deploy-first measurements run
    # 6-37% low without shape-matched prewarm + a discarded burn-in pass
    # (the GB300 settle audit). Best-effort; logged, never fatal.
    settle_log: list[str] = []
    if kube is not None and prewarm_shapes:
        ok = settle_prewarm(
            url,
            model,
            prewarm_shapes,
            kube_wrap=lambda inner: _wrap_kube(inner, kube),
            subprocess_runner=subprocess_runner,
        )
        settle_log.append(f"prewarm shapes={prewarm_shapes} {'ok' if ok else 'FAILED'}")
        if ok and settle_s > 0:
            sleeper(settle_s)
    if burn_in and runtime_commands:
        # Run-and-discard one pass of the first concurrency point. In kube
        # mode the in-pod artifact dir is simply overwritten by the recorded
        # run; locally the burn-in writes to raw/burnin, which the
        # normalizer skips (its dir name carries no concurrency suffix).
        if kube is None:
            first_dir = str(raw_dir / f"c{cell.concurrencies[0]}")
            burn_cmd = [token.replace(first_dir, str(raw_dir / "burnin")) for token in runtime_commands[0]]
        else:
            burn_cmd = runtime_commands[0]
        proc = run_aiperf(burn_cmd, auth_configs[0])
        settle_log.append(f"burn-in c={cell.concurrencies[0]} exit={proc.returncode} (DISCARDED)")
        if settle_s > 0:
            sleeper(settle_s)
    if settle_log:
        (cell_dir / "commands").mkdir(exist_ok=True)
        (cell_dir / "commands" / "settle.log").write_text("\n".join(settle_log) + "\n")

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    exits: list[int] = []
    for i, (cmd, c, auth_config) in enumerate(zip(runtime_commands, cell.concurrencies, auth_configs, strict=True)):
        if i > 0 and settle_s > 0 and (prewarm_shapes or burn_in):
            sleeper(settle_s)  # inter-point settle (only when discipline is on)
        if spec_capture is not None:
            spec_capture.scrape(f"pre-c{c}")
        proc = run_aiperf(cmd, auth_config)
        exits.append(proc.returncode)
        stdout_chunks.append(f"# concurrency={c} exit={proc.returncode}\n{redact_output(proc.stdout)}")
        stderr_chunks.append(f"# concurrency={c} exit={proc.returncode}\n{redact_output(proc.stderr)}")
        if spec_capture is not None:
            spec_capture.scrape(f"post-c{c}")
            spec_capture.window(c)
        if kube is not None:
            # Pull the in-pod AIPerf artifact dir back into raw/.
            kubectl_cp = [
                "kubectl",
                "cp",
                "--context",
                kube["kube_context"],
                f"{kube['namespace']}/{kube['bench_pod']}:/tmp/aa-{cell.cell_id}-c{c}",
                str(raw_dir / f"c{c}"),
            ]
            subprocess_runner(kubectl_cp, capture_output=True, text=True, check=False)

    (cell_dir / "commands" / "aa-sweep.stdout").write_text("\n".join(stdout_chunks))
    (cell_dir / "commands" / "aa-sweep.stderr").write_text("\n".join(stderr_chunks))
    (cell_dir / "commands" / "aa-sweep.exit").write_text("\n".join(str(e) for e in exits) + "\n")

    if spec_capture is not None:
        spec_capture.finalize()

    rows, status = normalize_outputs(cell, raw_dir, cell_dir, shape=shape, mode=mode)
    if spec_capture is not None and spec_capture.windows:
        rows = attach_windows_to_rows(rows, spec_capture.windows)
    write_normalized_json(cell_dir, rows)
    write_status_file(cell_dir, status)

    return AaResult(
        cell_dir=cell_dir,
        status=status,
        row_count=len(rows),
        commands=evidence_commands,
        dry_run=False,
        shape=shape.name,
        mode=mode,
        dataset_info=dataset_info,
        spec_windows=spec_capture.windows if spec_capture is not None else None,
    )
