"""perf_tune_report_kernel_profile — kernel-level profiling MCP verb.

Implements the ``inference-kernel-profile`` skill's recipe as a single
callable with operator-friendly steps + always-ack-gated execution.

Architecture
------------

Six discrete steps, each isolated in a ``KernelProfileStepFns`` method so
unit tests can inject a fake step-fn implementation that does no real
``kubectl`` calls:

1. ``validate_pod``       — confirm the namespace + pod + target-container exist
2. ``attach_sidecar``     — ``kubectl debug --share-processes`` to attach the nsys sidecar
3. ``wait_for_sidecar``   — poll until the ephemeral container is Running
4. ``find_vllm_pid``      — ``kubectl exec`` + pgrep to find the engine PID
5. ``run_nsys_profile``   — ``kubectl exec`` to invoke ``nsys profile``
6. ``extract_artifacts``  — ``kubectl cp`` to pull .nsys-rep + summary CSV
                            files out to ``--output-dir``
7. (optional) ``patch_bundle_metadata`` — write the
   ``kernel_profile`` block into the bundle's
   ``inference_perfbench_v1.json`` if ``--bundle`` was provided

Steps 2-6 are no-ops in ``--dry-run`` (the command is still constructed +
logged, but never sent to the kubernetes API).

Safety
------

The verb's contract entry is ``safety="submits_jobs"`` and requires the
``--i-understand-this-submits-jobs`` ack flag. Dry-run bypasses the ack.

Output
------

Writes:

- ``<output-dir>/<capture-name>.nsys-rep``  (binary nsys profile)
- ``<output-dir>/<capture-name>_gpu_kern_sum.csv``
- ``<output-dir>/<capture-name>_cuda_api_sum.csv``
- ``<output-dir>/kernel_profile.json``  (metadata + paths to the above)

And (when ``--bundle`` is provided) updates the bundle's
``inference_perfbench_v1.json`` ``kernel_profile`` block.

CLI envelope returned by the verb:

.. code-block:: json

    {
      "tool": "perf_tune_report_kernel_profile",
      "verb": "kernel_profile",
      "safety": "submits_jobs",
      "ack_required": true,
      "ack_field": "i_understand_this_submits_jobs",
      "captured_at": "<UTC ISO-8601>",
      "namespace": "inference",
      "pod": "basic-inference-xxx",
      "target_container": "basic-inference",
      "sidecar_container": "nsys-debug-<epoch>",
      "sidecar_image": "<sidecar image ref>",
      "vllm_pid": <int>,
      "duration_s": 120,
      "nsys_rep_path": "<abs path>",
      "summary_csv_paths": ["<abs path>", ...],
      "kernel_profile_json_path": "<abs path>",
      "bundle_patched": true|false,
      "dry_run": true|false,
      "step_commands": [<dict per step>, ...]
    }
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_DEFAULT_SIDECAR_IMAGE = "ghcr.io/cfregly/nsys-sidecar:0.1.0"
_DEFAULT_DURATION_S = 120
_DEFAULT_VLLM_PID_PATTERN = "vllm serve"
_DEFAULT_SAMPLE = "cpu"
_DEFAULT_TRACE = "cuda,nvtx,osrt"
_DEFAULT_SAMPLING_FREQUENCY = 1000
# Maximum time we'll wait for the ephemeral container to come up.
_DEFAULT_SIDECAR_READY_TIMEOUT_S = 60
# Per-attempt sleep between sidecar readiness polls.
_SIDECAR_POLL_INTERVAL_S = 2


@dataclass
class KernelProfileStepFns:
    """Step-function container so tests can swap out kubectl calls.

    All callables default to live ``subprocess.run`` invocations. Tests
    construct a fake KernelProfileStepFns with mock callables that return
    canned data + record what was called.
    """

    validate_pod: Any = None
    attach_sidecar: Any = None
    wait_for_sidecar: Any = None
    find_vllm_pid: Any = None
    run_nsys_profile: Any = None
    extract_artifacts: Any = None

    @classmethod
    def production(cls) -> "KernelProfileStepFns":
        return cls(
            validate_pod=_step_validate_pod,
            attach_sidecar=_step_attach_sidecar,
            wait_for_sidecar=_step_wait_for_sidecar,
            find_vllm_pid=_step_find_vllm_pid,
            run_nsys_profile=_step_run_nsys_profile,
            extract_artifacts=_step_extract_artifacts,
        )


@dataclass
class KernelProfileResult:
    """Summary returned by ``capture_kernel_profile``."""

    captured_at: str
    namespace: str
    pod: str
    target_container: str
    sidecar_container: str
    sidecar_image: str
    vllm_pid: int | None
    duration_s: int
    nsys_rep_path: Path
    summary_csv_paths: list[Path]
    kernel_profile_json_path: Path
    bundle_patched: bool
    dry_run: bool
    step_commands: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["nsys_rep_path"] = str(self.nsys_rep_path)
        d["summary_csv_paths"] = [str(p) for p in self.summary_csv_paths]
        d["kernel_profile_json_path"] = str(self.kernel_profile_json_path)
        return d


# ---------------------------------------------------------------------------
# Production step implementations (real subprocess calls)
# ---------------------------------------------------------------------------


def _run_kubectl(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Thin wrapper around subprocess.run with consistent text capture."""
    return subprocess.run(
        ["kubectl", *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _step_validate_pod(*, namespace: str, pod: str, target_container: str) -> dict[str, Any]:
    """Confirm pod + target container exist via ``kubectl get pod``."""
    result = _run_kubectl(
        ["-n", namespace, "get", "pod", pod, "-o", "json"],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"validate_pod: pod {namespace}/{pod} not found: {result.stderr.strip()}"
        )
    pod_obj = json.loads(result.stdout)
    containers = [c["name"] for c in pod_obj.get("spec", {}).get("containers", [])]
    if target_container not in containers:
        raise RuntimeError(
            f"validate_pod: target container {target_container!r} not in pod "
            f"{namespace}/{pod} (containers: {containers})"
        )
    return {
        "cmd": f"kubectl -n {namespace} get pod {pod} -o json",
        "pod_phase": pod_obj.get("status", {}).get("phase", "Unknown"),
        "containers": containers,
    }


def _step_attach_sidecar(
    *,
    namespace: str,
    pod: str,
    target_container: str,
    sidecar_container: str,
    sidecar_image: str,
) -> dict[str, Any]:
    """``kubectl debug --share-processes`` to attach the nsys sidecar."""
    args = [
        "-n", namespace,
        "debug", pod,
        "--image", sidecar_image,
        "--container", sidecar_container,
        "--share-processes",
        f"--target={target_container}",
        "--",
        "sleep", "3600",
    ]
    _run_kubectl(args)
    return {
        "cmd": "kubectl " + " ".join(shlex.quote(a) for a in args),
        "sidecar_container": sidecar_container,
    }


def _step_wait_for_sidecar(
    *,
    namespace: str,
    pod: str,
    sidecar_container: str,
    timeout_s: int = _DEFAULT_SIDECAR_READY_TIMEOUT_S,
) -> dict[str, Any]:
    """Poll until the ephemeral container reports Running state."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = _run_kubectl(
            ["-n", namespace, "get", "pod", pod, "-o", "json"],
            check=False,
        )
        if result.returncode != 0:
            time.sleep(_SIDECAR_POLL_INTERVAL_S)
            continue
        pod_obj = json.loads(result.stdout)
        for ec in pod_obj.get("status", {}).get("ephemeralContainerStatuses", []):
            if ec.get("name") == sidecar_container:
                state = ec.get("state", {})
                if "running" in state:
                    return {
                        "cmd": f"kubectl -n {namespace} get pod {pod} -o json (polled)",
                        "state": "running",
                        "ready_after_s": int(timeout_s - (deadline - time.monotonic())),
                    }
        time.sleep(_SIDECAR_POLL_INTERVAL_S)
    raise RuntimeError(
        f"wait_for_sidecar: ephemeral container {sidecar_container} did not "
        f"become Running within {timeout_s}s"
    )


def _step_find_vllm_pid(
    *,
    namespace: str,
    pod: str,
    sidecar_container: str,
    vllm_pid_pattern: str,
) -> dict[str, Any]:
    """``kubectl exec`` + pgrep to find the engine PID."""
    args = [
        "-n", namespace,
        "exec", pod, "-c", sidecar_container,
        "--", "bash", "-c",
        f"pgrep -f {shlex.quote(vllm_pid_pattern)} | head -1",
    ]
    result = _run_kubectl(args)
    pid_str = result.stdout.strip()
    if not pid_str:
        raise RuntimeError(
            f"find_vllm_pid: no process matching {vllm_pid_pattern!r} in pod "
            f"{namespace}/{pod}"
        )
    return {
        "cmd": "kubectl " + " ".join(shlex.quote(a) for a in args),
        "vllm_pid": int(pid_str),
    }


def _step_run_nsys_profile(
    *,
    namespace: str,
    pod: str,
    sidecar_container: str,
    vllm_pid: int,
    duration_s: int,
    sample: str,
    trace: str,
    sampling_frequency: int,
    capture_name: str,
) -> dict[str, Any]:
    """``kubectl exec`` to invoke ``nsys profile`` against the engine PID."""
    nsys_cmd = (
        f"nsys profile "
        f"--output=/profiling/{shlex.quote(capture_name)} "
        f"--capture-range=cudaProfilerApi "
        f"--duration={duration_s} "
        f"--force-overwrite=true "
        f"--attach-pid={vllm_pid} "
        f"--sample={shlex.quote(sample)} "
        f"--trace={shlex.quote(trace)} "
        f"--sampling-frequency={sampling_frequency} "
        f"&& nsys stats --report cuda_api_sum,gpu_kern_sum "
        f"/profiling/{shlex.quote(capture_name)}.nsys-rep "
        f"--format=csv "
        f"--output=/profiling/{shlex.quote(capture_name)}"
    )
    args = [
        "-n", namespace,
        "exec", pod, "-c", sidecar_container,
        "--", "bash", "-c", nsys_cmd,
    ]
    _run_kubectl(args)
    return {
        "cmd": "kubectl " + " ".join(shlex.quote(a) for a in args),
        "duration_s": duration_s,
        "vllm_pid": vllm_pid,
    }


def _step_extract_artifacts(
    *,
    namespace: str,
    pod: str,
    sidecar_container: str,
    capture_name: str,
    output_dir: Path,
) -> dict[str, Any]:
    """``kubectl cp`` to extract .nsys-rep + summary CSVs into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    in_pod_to_local = {
        f"/profiling/{capture_name}.nsys-rep":
            output_dir / f"{capture_name}.nsys-rep",
        f"/profiling/{capture_name}_gpu_kern_sum.csv":
            output_dir / f"{capture_name}_gpu_kern_sum.csv",
        f"/profiling/{capture_name}_cuda_api_sum.csv":
            output_dir / f"{capture_name}_cuda_api_sum.csv",
    }
    cmds: list[str] = []
    for in_pod_path, local_path in in_pod_to_local.items():
        args = [
            "-n", namespace,
            "cp", f"{pod}:{in_pod_path}", str(local_path),
            "-c", sidecar_container,
        ]
        _run_kubectl(args)
        cmds.append("kubectl " + " ".join(shlex.quote(a) for a in args))
    return {
        "cmds": cmds,
        "local_paths": [str(p) for p in in_pod_to_local.values()],
    }


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def capture_kernel_profile(
    *,
    namespace: str,
    pod: str,
    target_container: str,
    output_dir: Path,
    sidecar_image: str = _DEFAULT_SIDECAR_IMAGE,
    duration_s: int = _DEFAULT_DURATION_S,
    sample: str = _DEFAULT_SAMPLE,
    trace: str = _DEFAULT_TRACE,
    sampling_frequency: int = _DEFAULT_SAMPLING_FREQUENCY,
    vllm_pid_pattern: str = _DEFAULT_VLLM_PID_PATTERN,
    bundle: Path | None = None,
    dry_run: bool = False,
    step_fns: KernelProfileStepFns | None = None,
    captured_at: str | None = None,
    sidecar_container: str | None = None,
    capture_name: str | None = None,
) -> KernelProfileResult:
    """Run the 6-step (+ optional patch) kernel-profile capture pipeline.

    Args:
        namespace: kubernetes namespace of the target pod
        pod: pod name (use ``kubectl get pod -l app=...`` to discover)
        target_container: name of the container whose PID namespace we attach to
        output_dir: where to write the local .nsys-rep + CSV + JSON files
        sidecar_image: nsys sidecar image (defaults to the public ghcr.io image)
        duration_s: nsys profile duration (default 120s)
        sample: nsys --sample value (default ``cpu``)
        trace: nsys --trace value (default ``cuda,nvtx,osrt``)
        sampling_frequency: nsys --sampling-frequency value (default 1000Hz)
        vllm_pid_pattern: pgrep pattern for the engine process
            (default ``"vllm serve"``)
        bundle: optional inference-perf-bench bundle directory; if set, the
            bundle's ``inference_perfbench_v1.json`` ``kernel_profile`` block
            is patched in-place
        dry_run: print + log step commands but skip all kubectl calls
        step_fns: dependency-injected step implementations (production
            defaults via ``KernelProfileStepFns.production()`` if omitted)
        captured_at: ISO-8601 timestamp; defaults to now()
        sidecar_container: name override for the ephemeral container
        capture_name: filename stem for the .nsys-rep / .csv files

    Returns:
        ``KernelProfileResult`` with paths + metadata.

    Raises:
        RuntimeError: any step failed (pod not found, sidecar didn't come up,
            PID not found, nsys exit error)
    """
    if captured_at is None:
        captured_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if sidecar_container is None:
        sidecar_container = f"nsys-debug-{int(time.time())}"
    if capture_name is None:
        capture_name = f"capture-{captured_at.replace(':', '').replace('-', '')}"
    if step_fns is None:
        step_fns = KernelProfileStepFns.production()

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    step_commands: list[dict[str, Any]] = []

    # Step 1: validate pod (always runs, even in dry-run, so we fail fast on
    # bad inputs).
    validate_out = step_fns.validate_pod(
        namespace=namespace, pod=pod, target_container=target_container
    )
    step_commands.append({"step": "validate_pod", **validate_out})

    if dry_run:
        # Stub out steps 2-6 with their command-only output (no side effects).
        step_commands.extend(
            [
                {
                    "step": "attach_sidecar",
                    "cmd": (
                        f"kubectl -n {namespace} debug {pod} "
                        f"--image {sidecar_image} "
                        f"--container {sidecar_container} "
                        f"--share-processes --target={target_container} "
                        f"-- sleep 3600"
                    ),
                    "dry_run": True,
                },
                {"step": "wait_for_sidecar", "dry_run": True},
                {
                    "step": "find_vllm_pid",
                    "cmd": (
                        f"kubectl -n {namespace} exec {pod} -c {sidecar_container} "
                        f"-- bash -c 'pgrep -f \"{vllm_pid_pattern}\" | head -1'"
                    ),
                    "dry_run": True,
                },
                {
                    "step": "run_nsys_profile",
                    "duration_s": duration_s,
                    "dry_run": True,
                },
                {
                    "step": "extract_artifacts",
                    "output_dir": str(output_dir),
                    "dry_run": True,
                },
            ]
        )
        # Pre-compute expected output paths for the result envelope.
        nsys_rep_path = output_dir / f"{capture_name}.nsys-rep"
        csv_paths = [
            output_dir / f"{capture_name}_gpu_kern_sum.csv",
            output_dir / f"{capture_name}_cuda_api_sum.csv",
        ]
        kernel_profile_json_path = output_dir / "kernel_profile.json"
        return KernelProfileResult(
            captured_at=captured_at,
            namespace=namespace,
            pod=pod,
            target_container=target_container,
            sidecar_container=sidecar_container,
            sidecar_image=sidecar_image,
            vllm_pid=None,
            duration_s=duration_s,
            nsys_rep_path=nsys_rep_path,
            summary_csv_paths=csv_paths,
            kernel_profile_json_path=kernel_profile_json_path,
            bundle_patched=False,
            dry_run=True,
            step_commands=step_commands,
        )

    # Live execution: run steps 2-6 sequentially.
    attach_out = step_fns.attach_sidecar(
        namespace=namespace,
        pod=pod,
        target_container=target_container,
        sidecar_container=sidecar_container,
        sidecar_image=sidecar_image,
    )
    step_commands.append({"step": "attach_sidecar", **attach_out})

    wait_out = step_fns.wait_for_sidecar(
        namespace=namespace,
        pod=pod,
        sidecar_container=sidecar_container,
    )
    step_commands.append({"step": "wait_for_sidecar", **wait_out})

    pid_out = step_fns.find_vllm_pid(
        namespace=namespace,
        pod=pod,
        sidecar_container=sidecar_container,
        vllm_pid_pattern=vllm_pid_pattern,
    )
    vllm_pid = int(pid_out["vllm_pid"])
    step_commands.append({"step": "find_vllm_pid", **pid_out})

    profile_out = step_fns.run_nsys_profile(
        namespace=namespace,
        pod=pod,
        sidecar_container=sidecar_container,
        vllm_pid=vllm_pid,
        duration_s=duration_s,
        sample=sample,
        trace=trace,
        sampling_frequency=sampling_frequency,
        capture_name=capture_name,
    )
    step_commands.append({"step": "run_nsys_profile", **profile_out})

    extract_out = step_fns.extract_artifacts(
        namespace=namespace,
        pod=pod,
        sidecar_container=sidecar_container,
        capture_name=capture_name,
        output_dir=output_dir,
    )
    step_commands.append({"step": "extract_artifacts", **extract_out})

    nsys_rep_path = output_dir / f"{capture_name}.nsys-rep"
    csv_paths = [
        output_dir / f"{capture_name}_gpu_kern_sum.csv",
        output_dir / f"{capture_name}_cuda_api_sum.csv",
    ]

    # Write kernel_profile.json with all metadata (always written; the
    # bundle-side patch is optional and additive).
    kernel_profile_json = {
        "captured_at": captured_at,
        "namespace": namespace,
        "pod": pod,
        "target_container": target_container,
        "sidecar_container": sidecar_container,
        "sidecar_image": sidecar_image,
        "vllm_pid": vllm_pid,
        "duration_s": duration_s,
        "sample": sample,
        "trace": trace,
        "sampling_frequency": sampling_frequency,
        "vllm_pid_pattern": vllm_pid_pattern,
        "nsys_rep_path": str(nsys_rep_path),
        "summary_csv_paths": [str(p) for p in csv_paths],
        "method": "kubectl-debug-share-processes",
    }
    kernel_profile_json_path = output_dir / "kernel_profile.json"
    kernel_profile_json_path.write_text(
        json.dumps(kernel_profile_json, indent=2, sort_keys=True)
    )

    # Step 7 (optional): patch the bundle's inference_perfbench_v1.json so
    # the renderer / aggregator picks up the kernel-profile artifact.
    bundle_patched = False
    if bundle is not None:
        bundle = bundle.expanduser().resolve()
        ipb_path = bundle / "inference_perfbench_v1.json"
        if ipb_path.is_file():
            try:
                ipb = json.loads(ipb_path.read_text())
            except json.JSONDecodeError:
                ipb = {}
            ipb["kernel_profile"] = kernel_profile_json
            ipb_path.write_text(json.dumps(ipb, indent=2, sort_keys=True))
            bundle_patched = True

    return KernelProfileResult(
        captured_at=captured_at,
        namespace=namespace,
        pod=pod,
        target_container=target_container,
        sidecar_container=sidecar_container,
        sidecar_image=sidecar_image,
        vllm_pid=vllm_pid,
        duration_s=duration_s,
        nsys_rep_path=nsys_rep_path,
        summary_csv_paths=csv_paths,
        kernel_profile_json_path=kernel_profile_json_path,
        bundle_patched=bundle_patched,
        dry_run=False,
        step_commands=step_commands,
    )
