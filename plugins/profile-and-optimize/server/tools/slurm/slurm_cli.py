"""Slurm CLI: workload-agnostic Slurm operations for Slurm-on-K8s clusters.

Four verbs:

- ``triage`` classifies failures without changing cluster state.
- ``drain`` uses the `substitutes_nodes` safety class and drains a comma-list of Slurm node names
                      with an operator-supplied Reason. Ack-gated.
- ``resume`` uses the `substitutes_nodes` safety class and returns those nodes to IDLE.
                      Ack-gated.
- ``quiet_window`` uses the `substitutes_nodes` safety class. It drains, runs an
                      operator-supplied inner command, and attempts a resume
                      from a finally block after inner-command failure or interruption.
                      Ack-gated.

The drain/resume/quiet_window verbs provide pre-bench Slurm-state hygiene for
the [``inference-perf-bench``](../../../skills/inference-perf-bench/SKILL.md) workflow
on Slurm-on-K8s clusters where slurmd worker pods co-host inference vLLM pods on the
same GPU nodes.

The triage verb handles generic Slurm failures. Cluster-level Slurm-on-K8s
incidents hand off to the current `k8s-troubleshooting` skill.

Evidence-bundle layout (drain / resume / quiet_window):

::

    <bundle>/commands/
      01-slurm-pre-state.stdout
      02-drain.{cmd,stdout,stderr,exit,reason}
      03-slurm-post-drain.stdout
      04-bench.{cmd,stdout,stderr,exit}        # quiet_window only
      04-bench-{start,end}.txt                 # quiet_window only
      05-resume.{cmd,stdout,stderr,exit}
      06-slurm-post-resume.stdout

The numeric step IDs keep drain, benchmark, resume, and inspection receipts in
execution order so operators can compare evidence bundles directly.

"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shlex
import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from tools.slurm.signature_patterns import Signature, match_signatures

CONTRACT: dict[str, dict[str, Any]] = {
    "triage": {
        "safety": "read_only",
        "required": ("--jobid",),
        "optional": ("--logdir", "--repo-root", "--json"),
        "json": True,
        "ack": None,
        "description": "Triage a failed Slurm job by parsing sacct + slurm-<jobid>.out and matching against signature patterns.",
    },
    "drain": {
        "safety": "substitutes_nodes",
        "required": ("--nodes", "--ns"),
        "optional": ("--reason", "--ctl", "--ctl-container", "--bundle", "--json"),
        "json": True,
        "ack": "--i-understand-this-substitutes-nodes",
        "description": "Drain a comma-list of Slurm node names (scontrol update State=DRAIN) on a Slurm-on-K8s cluster, with full evidence-bundle capture.",
    },
    "resume": {
        "safety": "substitutes_nodes",
        "required": ("--nodes", "--ns"),
        "optional": ("--reason", "--ctl", "--ctl-container", "--bundle", "--json"),
        "json": True,
        "ack": "--i-understand-this-substitutes-nodes",
        "description": "Resume a comma-list of Slurm node names (scontrol update State=RESUME) on a Slurm-on-K8s cluster, with full evidence-bundle capture.",
    },
    "quiet_window": {
        "safety": "substitutes_nodes",
        "required": ("--nodes", "--ns", "--cmd"),
        "optional": ("--reason", "--ctl", "--ctl-container", "--bundle", "--json"),
        "json": True,
        "ack": "--i-understand-this-substitutes-nodes",
        "description": "Drain Slurm nodes, run an operator-supplied inner command, and attempt a resume from a finally block after failure or interruption.",
    },
}


# --------------------------------------------------------------------------- #
# triage helpers
# --------------------------------------------------------------------------- #


def _operator_path(value: str, *, label: str) -> Path:
    """Normalize a complete path chosen by the local CLI operator."""
    if not value or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty local path without NUL bytes")

    # codeql[py/path-injection]
    return Path(value).expanduser().resolve()


def _resolve_repo_root(arg: str | None) -> Path:
    if arg:
        return _operator_path(arg, label="--repo-root")
    env = os.environ.get("PROFILE_AND_OPTIMIZE_REPO_ROOT")
    if env:
        return _operator_path(env, label="PROFILE_AND_OPTIMIZE_REPO_ROOT")
    current = Path.cwd().resolve()
    while current != current.parent:
        if (
            (current / "pyproject.toml").is_file()
            and (current / "mcp_surface.py").is_file()
            and (current / "tools").is_dir()
        ):
            return current
        current = current.parent
    raise SystemExit("FATAL: cannot resolve repo root; pass --repo-root or set PROFILE_AND_OPTIMIZE_REPO_ROOT")


def _confidence(hit_count: int) -> str:
    if hit_count >= 3:
        return "HIGH"
    if hit_count >= 1:
        return "MEDIUM"
    return "LOW"


def _run_sacct(jobid: str) -> dict[str, str] | None:
    if shutil.which("sacct") is None:
        return None
    cmd = [
        "sacct",
        "-j",
        jobid,
        "-X",
        "-n",
        "-P",
        "--format=JobID,State,ExitCode,Elapsed,DerivedExitCode,Reason,NodeList,WorkDir,Submit,Start,End",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    fields = proc.stdout.splitlines()[0].split("|")
    keys = [
        "JobID",
        "State",
        "ExitCode",
        "Elapsed",
        "DerivedExitCode",
        "Reason",
        "NodeList",
        "WorkDir",
        "Submit",
        "Start",
        "End",
    ]
    return dict(zip(keys, fields, strict=False))


def _find_slurm_out(jobid: str, search_root: Path) -> Path | None:
    candidates: list[Path] = []
    candidates.append(search_root / f"slurm-{jobid}.out")
    candidates.extend(search_root.rglob(f"slurm-{jobid}.out"))
    for c in candidates:
        if c.is_file():
            return c
    return None


def _classify(signature_hits: list[tuple[Signature, int]]) -> dict[str, Any]:
    if not signature_hits:
        return {
            "class": "unknown",
            "confidence": "LOW",
            "evidence": [],
            "next_probe": "no signature matched. Capture the last 500 lines and ask the operator",
        }
    top, count = signature_hits[0]
    return {
        "class": top.klass,
        "confidence": _confidence(count),
        "evidence": [s.klass for s, _ in signature_hits[:5]],
        "description": top.description,
        "next_probe": top.next_probe,
    }


def _cluster_handoff(klass: str, sacct: dict[str, str] | None) -> str | None:
    if sacct is None:
        return None
    workdir = sacct.get("WorkDir", "")
    if "experiments/artifacts/campaign" in workdir or "mlperf" in workdir.lower():
        return "k8s-troubleshooting"
    if klass in {"gpu_xid", "nccl_hang", "nccl_setup", "fabric"}:
        return "k8s-troubleshooting"
    return None


def cmd_triage(args: argparse.Namespace) -> int:
    jobid = args.jobid
    logdir = _operator_path(args.logdir, label="--logdir") if args.logdir else None
    repo_root = None if logdir else _resolve_repo_root(args.repo_root)
    search_root = logdir or repo_root
    assert search_root is not None

    sacct = _run_sacct(jobid)
    slurm_out = _find_slurm_out(jobid, search_root)
    text = slurm_out.read_text(errors="replace") if slurm_out else ""
    if text.count("\n") > 5000:
        text = "\n".join(text.splitlines()[-5000:])

    hits = match_signatures(text)
    classification = _classify(hits)
    handoff = _cluster_handoff(classification["class"], sacct)

    payload: dict[str, Any] = {
        "tool": "slurm_triage",
        "library": "slurm",
        "verb": "triage",
        "safety": CONTRACT["triage"]["safety"],
        "jobid": jobid,
        "logdir": str(logdir) if logdir else None,
        "repo_root": str(repo_root) if repo_root else None,
        "search_root": str(search_root),
        "slurm_out_path": str(slurm_out) if slurm_out else None,
        "sacct": sacct,
        "classification": classification,
        "handoff_skill": handoff,
        "signature_hit_count": sum(c for _, c in hits),
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        klass = classification["class"]
        conf = classification["confidence"]
        print(f"jobid:           {jobid}")
        print(f"slurm_out:       {payload['slurm_out_path']}")
        print(f"sacct.State:     {(sacct or {}).get('State', '(unknown)')}")
        print(f"sacct.ExitCode:  {(sacct or {}).get('ExitCode', '(unknown)')}")
        print(f"classification:  {klass} ({conf})")
        print(f"description:     {classification.get('description', '')}")
        print(f"next_probe:      {classification['next_probe']}")
        if handoff:
            print(f"handoff:         {handoff}")
    return 0


# --------------------------------------------------------------------------- #
# drain / resume / quiet_window helpers
# --------------------------------------------------------------------------- #


def _utc_stamp() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ctl_exec_argv(
    ns: str,
    ctl: str,
    container: str,
    inner: Sequence[str],
) -> list[str]:
    """Build the kubectl-exec argv that runs ``inner`` in the slurm controller."""
    return [
        "kubectl",
        "-n",
        ns,
        "exec",
        ctl,
        "-c",
        container,
        "--",
        *inner,
    ]


# Indirection point for unit tests: a thin wrapper around subprocess.run that
# tests can monkey-patch to inject a fake kubectl.
def _run(
    argv: Sequence[str],
    *,
    capture: bool = True,
    timeout: int | None = 60,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        capture_output=capture,
        text=True,
        timeout=timeout,
        check=False,
    )


# Public alias so tests can monkey-patch ``slurm_cli.RUN`` easily.
RUN: Callable[..., subprocess.CompletedProcess[str]] = _run


def _ctl_run(
    ns: str,
    ctl: str,
    container: str,
    inner: Sequence[str],
    *,
    timeout: int | None = 60,
) -> subprocess.CompletedProcess[str]:
    argv = _ctl_exec_argv(ns, ctl, container, inner)
    return RUN(argv, capture=True, timeout=timeout)


def _sinfo_snapshot(ns: str, ctl: str, container: str) -> str:
    proc = _ctl_run(
        ns,
        ctl,
        container,
        ["sinfo", "-h", "-N", "-o", "%n %T %R"],
        timeout=30,
    )
    if proc.returncode != 0:
        # Capture stderr in the snapshot so the operator can see why; never
        # raise from a sinfo snapshot, since it is best-effort context.
        return f"# sinfo failed (rc={proc.returncode}): {proc.stderr.strip()}\n"
    return proc.stdout


def _resolve_bundle(bundle: str | None, *, family: str) -> Path:
    if bundle:
        return _operator_path(bundle, label="--bundle")
    stamp = _utc_stamp()
    env = os.environ.get("PROFILE_AND_OPTIMIZE_BUNDLE_ROOT")
    base = (
        _operator_path(env, label="PROFILE_AND_OPTIMIZE_BUNDLE_ROOT")
        if env
        else Path.cwd().resolve() / "experiments" / "artifacts" / "inference-perf-bench"
    )
    return base / f"{family}-{stamp}"


def _write_atomic(path: Path, data: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data)
    else:
        path.write_bytes(data)


def _shell_quote_argv(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)


def _scontrol_update_argv(nodes: str, state: str, reason: str) -> list[str]:
    """Build the inner scontrol-update argv list (pre-kubectl-exec wrap)."""
    return [
        "scontrol",
        "update",
        f"NodeName={nodes}",
        f"State={state}",
        f"Reason={reason}",
    ]


def _capture_step(
    bundle: Path,
    step: str,
    *,
    cmd_argv: Sequence[str] | None = None,
    proc: subprocess.CompletedProcess[str] | None = None,
    extra_files: dict[str, str] | None = None,
) -> None:
    """Write the four-file ``.cmd / .stdout / .stderr / .exit`` tuple for one step.

    ``step`` is the file-stem prefix (e.g., ``"02-drain"``). When ``proc`` is
    None this writes only the .cmd file plus any ``extra_files`` entries
    (used for the bench-start/-end UTC stamps that have no proc to attach to).
    """
    commands_dir = bundle / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)
    if cmd_argv is not None:
        _write_atomic(commands_dir / f"{step}.cmd", _shell_quote_argv(cmd_argv) + "\n")
    if proc is not None:
        _write_atomic(commands_dir / f"{step}.stdout", proc.stdout or "")
        _write_atomic(commands_dir / f"{step}.stderr", proc.stderr or "")
        _write_atomic(commands_dir / f"{step}.exit", f"{proc.returncode}\n")
    for suffix, body in (extra_files or {}).items():
        _write_atomic(commands_dir / f"{step}{suffix}", body)


def _capture_step_best_effort(
    bundle: Path,
    step: str,
    *,
    cmd_argv: Sequence[str] | None = None,
    proc: subprocess.CompletedProcess[str] | None = None,
    extra_files: dict[str, str] | None = None,
) -> BaseException | None:
    """Capture a receipt without allowing an artifact failure to block cleanup."""
    try:
        _capture_step(
            bundle,
            step,
            cmd_argv=cmd_argv,
            proc=proc,
            extra_files=extra_files,
        )
    except BaseException as exc:  # noqa: BLE001
        return exc
    return None


def _capture_snapshot_best_effort(
    args: argparse.Namespace,
    bundle: Path,
    step: str,
) -> list[BaseException]:
    """Capture a Slurm snapshot and return every failure to the caller."""
    errors: list[BaseException] = []
    try:
        snapshot = _sinfo_snapshot(args.ns, args.ctl, args.ctl_container)
    except BaseException as exc:  # noqa: BLE001
        errors.append(exc)
        return errors
    if snapshot.startswith("# sinfo failed"):
        errors.append(RuntimeError(snapshot.strip()))
    capture_error = _capture_step_best_effort(
        bundle,
        step,
        extra_files={".stdout": snapshot},
    )
    if capture_error is not None:
        errors.append(capture_error)
    return errors


def _envelope(
    *,
    verb: str,
    bundle: Path,
    nodes: str,
    reason: str,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = CONTRACT[verb]
    payload: dict[str, Any] = {
        "tool": f"slurm_{verb}",
        "library": "slurm",
        "verb": verb,
        "safety": spec["safety"],
        "ack_field": spec["ack"],
        "bundle": str(bundle),
        "nodes": nodes,
        "reason": reason,
    }
    if extras:
        payload.update(extras)
    return payload


def _emit(args: argparse.Namespace, payload: dict[str, Any], *, human_lines: list[str]) -> None:
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for line in human_lines:
            print(line)


def _ack_required_or_die(args: argparse.Namespace, verb: str) -> None:
    if not getattr(args, "i_understand_this_substitutes_nodes", False):
        ack = CONTRACT[verb]["ack"]
        raise SystemExit(
            f"FATAL: {verb} is ack-gated (safety=substitutes_nodes). Pass {ack} to confirm operator approval."
        )


def _attempt_resume(
    args: argparse.Namespace,
    bundle: Path,
    nodes: str,
) -> tuple[int, str, str | None]:
    """Attempt a node resume twice and capture a complete step-05 receipt."""
    resume_inner = _scontrol_update_argv(nodes, "RESUME", "cleared")
    resume_argv = _ctl_exec_argv(args.ns, args.ctl, args.ctl_container, resume_inner)
    exceptions: list[str] = []
    resume_proc: subprocess.CompletedProcess[str] | None = None
    for attempt in (1, 2):
        try:
            resume_proc = RUN(resume_argv, capture=True, timeout=60)
            break
        except BaseException as exc:  # noqa: BLE001
            exceptions.append(f"resume attempt {attempt} raised: {exc!r}")

    if resume_proc is None:
        message = "\n".join(exceptions) + "\n"
        capture_error = _capture_step_best_effort(
            bundle,
            "05-resume",
            cmd_argv=resume_argv,
            extra_files={".stderr": message, ".exit": "255\n"},
        )
        if capture_error is not None:
            exceptions.append(f"resume receipt raised: {capture_error!r}")
            message = "\n".join(exceptions) + "\n"
        return 255, message, "; ".join(exceptions)

    if exceptions:
        combined_stderr = "\n".join(exceptions) + "\n" + (resume_proc.stderr or "")
        resume_proc = subprocess.CompletedProcess(
            args=resume_proc.args,
            returncode=resume_proc.returncode,
            stdout=resume_proc.stdout,
            stderr=combined_stderr,
        )

    capture_error = _capture_step_best_effort(
        bundle,
        "05-resume",
        cmd_argv=resume_argv,
        proc=resume_proc,
    )
    if capture_error is not None:
        exceptions.append(f"resume receipt raised: {capture_error!r}")
    exception_summary = "; ".join(exceptions) or None
    return resume_proc.returncode, resume_proc.stderr or "", exception_summary


def _do_drain_or_resume(
    *,
    verb: str,
    args: argparse.Namespace,
    state: str,
    pre_step: str,
    main_step: str,
    post_step: str,
    default_reason_prefix: str,
) -> tuple[int, dict[str, Any]]:
    """Shared core for the ``drain`` and ``resume`` verbs.

    Returns ``(exit_code, envelope_payload)``.
    """
    _ack_required_or_die(args, verb)
    nodes: str = args.nodes
    reason: str = args.reason or f"{default_reason_prefix} {_utc_stamp()}"
    bundle = _resolve_bundle(args.bundle, family=f"slurm-{verb}-window")

    pre = _sinfo_snapshot(args.ns, args.ctl, args.ctl_container)
    _capture_step(bundle, pre_step, extra_files={".stdout": pre})

    inner = _scontrol_update_argv(nodes, state, reason)
    argv = _ctl_exec_argv(args.ns, args.ctl, args.ctl_container, inner)
    proc = RUN(argv, capture=True, timeout=60)
    _capture_step(
        bundle,
        main_step,
        cmd_argv=argv,
        proc=proc,
        extra_files={".reason": reason + "\n"} if verb == "drain" else None,
    )

    post = _sinfo_snapshot(args.ns, args.ctl, args.ctl_container)
    _capture_step(bundle, post_step, extra_files={".stdout": post})

    payload = _envelope(
        verb=verb,
        bundle=bundle,
        nodes=nodes,
        reason=reason,
        extras={
            "scontrol_exit": proc.returncode,
            "scontrol_stderr": proc.stderr,
        },
    )
    rc = 0 if proc.returncode == 0 else proc.returncode
    return rc, payload


def cmd_drain(args: argparse.Namespace) -> int:
    rc, payload = _do_drain_or_resume(
        verb="drain",
        args=args,
        state="DRAIN",
        pre_step="01-slurm-pre-state",
        main_step="02-drain",
        post_step="03-slurm-post-drain",
        default_reason_prefix="perf bench window",
    )
    _emit(
        args,
        payload,
        human_lines=[
            "verb:    drain",
            f"nodes:   {payload['nodes']}",
            f"reason:  {payload['reason']}",
            f"bundle:  {payload['bundle']}",
            f"sc_exit: {payload['scontrol_exit']}",
        ],
    )
    return rc


def cmd_resume(args: argparse.Namespace) -> int:
    rc, payload = _do_drain_or_resume(
        verb="resume",
        args=args,
        state="RESUME",
        pre_step="01-slurm-pre-state",
        main_step="05-resume",
        post_step="06-slurm-post-resume",
        default_reason_prefix="cleared",
    )
    _emit(
        args,
        payload,
        human_lines=[
            "verb:    resume",
            f"nodes:   {payload['nodes']}",
            f"reason:  {payload['reason']}",
            f"bundle:  {payload['bundle']}",
            f"sc_exit: {payload['scontrol_exit']}",
        ],
    )
    return rc


def cmd_quiet_window(args: argparse.Namespace) -> int:
    """Drain, run an inner command, then attempt to resume in ``finally``.

    Resume runs in a try/finally so an exception in the inner command, a
    KeyboardInterrupt, or a SystemExit cannot leave the partition stuck in
    ``drained``. The inner command's return code is propagated as the
    verb's exit code when it is nonzero. A resume failure after a successful
    inner command forces a nonzero result.
    """
    _ack_required_or_die(args, "quiet_window")
    nodes: str = args.nodes
    reason: str = args.reason or f"perf bench window {_utc_stamp()}"
    bundle = _resolve_bundle(args.bundle, family="slurm-quiet-window")
    inner_cmd: str = args.cmd
    inner_argv = shlex.split(inner_cmd)
    if not inner_argv:
        raise SystemExit("FATAL: --cmd resolved to an empty argv after shlex.split")

    pre = _sinfo_snapshot(args.ns, args.ctl, args.ctl_container)
    _capture_step(bundle, "01-slurm-pre-state", extra_files={".stdout": pre})

    drain_inner = _scontrol_update_argv(nodes, "DRAIN", reason)
    drain_argv = _ctl_exec_argv(args.ns, args.ctl, args.ctl_container, drain_inner)
    drain_proc: subprocess.CompletedProcess[str] | None = None
    drain_exception: BaseException | None = None
    inner_rc: int | None = None
    inner_proc: subprocess.CompletedProcess[str] | None = None
    inner_exception: BaseException | None = None
    pending_exception: BaseException | None = None
    diagnostic_errors: list[BaseException] = []
    resume_exit = 255
    resume_stderr = ""
    resume_exception: str | None = None

    # From the first drain attempt onward, every path reaches this finally and
    # attempts a resume. A command timeout can occur after a partial mutation.
    try:
        try:
            drain_proc = RUN(drain_argv, capture=True, timeout=60)
        except BaseException as exc:  # noqa: BLE001 - resume must run after interrupts.
            drain_exception = exc
            if not isinstance(exc, Exception):
                pending_exception = exc

        if drain_proc is not None:
            capture_error = _capture_step_best_effort(
                bundle,
                "02-drain",
                cmd_argv=drain_argv,
                proc=drain_proc,
                extra_files={".reason": reason + "\n"},
            )
        else:
            message = f"drain invocation raised: {drain_exception!r}\n"
            capture_error = _capture_step_best_effort(
                bundle,
                "02-drain",
                cmd_argv=drain_argv,
                extra_files={
                    ".stderr": message,
                    ".exit": "255\n",
                    ".reason": reason + "\n",
                },
            )
        if capture_error is not None:
            diagnostic_errors.append(capture_error)
            if not isinstance(capture_error, Exception):
                pending_exception = pending_exception or capture_error

        diagnostic_errors.extend(_capture_snapshot_best_effort(args, bundle, "03-slurm-post-drain"))
        for error in diagnostic_errors:
            if not isinstance(error, Exception):
                pending_exception = pending_exception or error

        drain_succeeded = (
            drain_proc is not None and drain_proc.returncode == 0 and drain_exception is None and not diagnostic_errors
        )
        if drain_succeeded:
            capture_error = _capture_step_best_effort(
                bundle,
                "04-bench-start",
                extra_files={".txt": _utc_iso() + "\n"},
            )
            if capture_error is not None:
                diagnostic_errors.append(capture_error)
            else:
                try:
                    inner_proc = RUN(inner_argv, capture=True, timeout=None)
                    inner_rc = inner_proc.returncode
                except BaseException as exc:  # noqa: BLE001 - resume must run after interrupts.
                    inner_exception = exc
                    inner_rc = 1
                    if not isinstance(exc, Exception):
                        pending_exception = pending_exception or exc

                if inner_proc is not None:
                    capture_error = _capture_step_best_effort(
                        bundle,
                        "04-bench",
                        cmd_argv=inner_argv,
                        proc=inner_proc,
                    )
                else:
                    capture_error = _capture_step_best_effort(
                        bundle,
                        "04-bench",
                        cmd_argv=inner_argv,
                        extra_files={
                            ".stderr": f"inner invocation raised: {inner_exception!r}\n",
                            ".exit": "255\n",
                        },
                    )
                if capture_error is not None:
                    diagnostic_errors.append(capture_error)
                end_error = _capture_step_best_effort(
                    bundle,
                    "04-bench-end",
                    extra_files={".txt": _utc_iso() + "\n"},
                )
                if end_error is not None:
                    diagnostic_errors.append(end_error)
    finally:
        # Every attempted drain reaches compensation, even when diagnostics or
        # evidence writes fail after the mutation.
        resume_exit, resume_stderr, resume_exception = _attempt_resume(
            args,
            bundle,
            nodes,
        )
        diagnostic_errors.extend(_capture_snapshot_best_effort(args, bundle, "06-slurm-post-resume"))

    drain_exit = drain_proc.returncode if drain_proc is not None else 255
    drain_stderr = drain_proc.stderr if drain_proc is not None else ""
    inner_skipped = inner_rc is None
    compensation = drain_exit != 0 or drain_exception is not None
    payload = _envelope(
        verb="quiet_window",
        bundle=bundle,
        nodes=nodes,
        reason=reason,
        extras={
            "drain_exit": drain_exit,
            "drain_stderr": drain_stderr,
            "drain_exception": repr(drain_exception) if drain_exception else None,
            "inner_cmd": inner_cmd,
            "inner_exit": inner_rc,
            "inner_exception": repr(inner_exception) if inner_exception else None,
            "inner_skipped": inner_skipped,
            "resume_exit": resume_exit,
            "resume_stderr": resume_stderr,
            "resume_exception": resume_exception,
            "resume_skipped": False,
            "compensation_attempted": True,
            "compensation_exit": resume_exit if compensation else None,
            "compensation_stderr": resume_stderr if compensation else None,
            "compensation_exception": resume_exception if compensation else None,
            "diagnostic_errors": [repr(error) for error in diagnostic_errors],
        },
    )
    _emit(
        args,
        payload,
        human_lines=[
            "verb:        quiet_window",
            f"bundle:      {bundle}",
            f"nodes:       {nodes}",
            f"reason:      {reason}",
            f"drain_exit:  {drain_exit}",
            f"inner_exit:  {inner_rc if inner_rc is not None else 'skipped'}",
            f"resume_exit: {resume_exit}",
        ],
    )
    if pending_exception is not None:
        raise pending_exception
    if drain_exit != 0 or drain_exception is not None:
        return drain_exit or 255
    if diagnostic_errors:
        return 255
    if inner_rc is not None and inner_rc != 0:
        return inner_rc
    if resume_exception is not None:
        return 255
    return resume_exit if resume_exit > 0 else int(resume_exit != 0)


# --------------------------------------------------------------------------- #
# argparse plumbing
# --------------------------------------------------------------------------- #


def _add_common_substitutes_nodes_args(p: argparse.ArgumentParser) -> None:
    """Shared argparse plumbing for the 3 substitutes_nodes verbs."""
    p.add_argument(
        "--nodes",
        required=True,
        help="Comma-separated list of Slurm node names to target, for example 'gpu-node-01,gpu-node-02'.",
    )
    p.add_argument(
        "--reason",
        default=None,
        help="Slurm Reason string. Defaults to 'perf bench window <UTC>' for drain/quiet_window, 'cleared <UTC>' for resume.",
    )
    p.add_argument(
        "--ns",
        required=True,
        help="Kubernetes namespace hosting the Slurm controller.",
    )
    p.add_argument(
        "--ctl",
        default="deploy/slurm-controller",
        help="kubectl resource selector for the slurm controller (default: deploy/slurm-controller).",
    )
    p.add_argument(
        "--ctl-container",
        default="slurmctld",
        help="Container name inside the controller pod that hosts scontrol/sinfo (default: slurmctld).",
    )
    p.add_argument(
        "--bundle",
        default=None,
        help="Override the evidence-bundle directory. Default: $PROFILE_AND_OPTIMIZE_BUNDLE_ROOT or ./experiments/artifacts/inference-perf-bench/<family>-<UTC>/.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON envelope on stdout.",
    )
    p.add_argument(
        "--i-understand-this-substitutes-nodes",
        dest="i_understand_this_substitutes_nodes",
        action="store_true",
        help="Operator approval. Without this flag the verb refuses to mutate Slurm node state.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Workload-agnostic Slurm operations for Slurm-on-K8s clusters: triage / drain / resume / quiet_window."
    )
    sub = parser.add_subparsers(dest="verb", required=True)

    triage = sub.add_parser("triage", description=CONTRACT["triage"]["description"])
    triage.add_argument("--jobid", required=True, help="Slurm job ID")
    triage.add_argument("--logdir", default=None, help="Directory containing slurm-<jobid>.out")
    triage.add_argument("--repo-root", default=None, help="Override PROFILE_AND_OPTIMIZE_REPO_ROOT")
    triage.add_argument("--json", action="store_true", help="Emit JSON envelope")
    triage.set_defaults(func=cmd_triage)

    drain = sub.add_parser("drain", description=CONTRACT["drain"]["description"])
    _add_common_substitutes_nodes_args(drain)
    drain.set_defaults(func=cmd_drain)

    resume = sub.add_parser("resume", description=CONTRACT["resume"]["description"])
    _add_common_substitutes_nodes_args(resume)
    resume.set_defaults(func=cmd_resume)

    quiet = sub.add_parser("quiet_window", description=CONTRACT["quiet_window"]["description"])
    _add_common_substitutes_nodes_args(quiet)
    quiet.add_argument(
        "--cmd",
        required=True,
        help="Inner command to run between drain and resume. Parsed via shlex.split, so quote it: --cmd 'bash run-bench.sh E8'.",
    )
    quiet.set_defaults(func=cmd_quiet_window)

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
