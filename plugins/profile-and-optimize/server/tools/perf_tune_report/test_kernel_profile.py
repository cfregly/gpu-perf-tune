"""Unit tests for the kernel_profile verb (v1.21.0)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tools.perf_tune_report.kernel_profile import (
    KernelProfileResult,
    KernelProfileStepFns,
    capture_kernel_profile,
)


# ---------------------------------------------------------------------------
# Fake step-fn implementation for testing — records calls + returns canned data
# ---------------------------------------------------------------------------


@dataclass
class _FakeSteps:
    """Records every step's kwargs + returns operator-supplied canned data."""

    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    pid_to_return: int = 12345
    fail_step: str | None = None  # set to a step name to make that step raise

    def _maybe_fail(self, step_name: str) -> None:
        if self.fail_step == step_name:
            raise RuntimeError(f"_FakeSteps.{step_name}: deliberate test failure")

    def validate_pod(self, *, namespace: str, pod: str, target_container: str) -> dict[str, Any]:
        self.calls.append(("validate_pod", locals().copy()))
        self._maybe_fail("validate_pod")
        return {
            "cmd": f"FAKE kubectl get pod {pod}",
            "pod_phase": "Running",
            "containers": [target_container, "sidecar-other"],
        }

    def attach_sidecar(
        self,
        *,
        namespace: str,
        pod: str,
        target_container: str,
        sidecar_container: str,
        sidecar_image: str,
    ) -> dict[str, Any]:
        self.calls.append(("attach_sidecar", locals().copy()))
        self._maybe_fail("attach_sidecar")
        return {"cmd": f"FAKE kubectl debug {pod}", "sidecar_container": sidecar_container}

    def wait_for_sidecar(
        self, *, namespace: str, pod: str, sidecar_container: str
    ) -> dict[str, Any]:
        self.calls.append(("wait_for_sidecar", locals().copy()))
        self._maybe_fail("wait_for_sidecar")
        return {"cmd": "FAKE poll", "state": "running", "ready_after_s": 2}

    def find_vllm_pid(
        self,
        *,
        namespace: str,
        pod: str,
        sidecar_container: str,
        vllm_pid_pattern: str,
    ) -> dict[str, Any]:
        self.calls.append(("find_vllm_pid", locals().copy()))
        self._maybe_fail("find_vllm_pid")
        return {"cmd": "FAKE pgrep", "vllm_pid": self.pid_to_return}

    def run_nsys_profile(
        self,
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
        self.calls.append(("run_nsys_profile", locals().copy()))
        self._maybe_fail("run_nsys_profile")
        return {"cmd": "FAKE nsys profile", "duration_s": duration_s, "vllm_pid": vllm_pid}

    def extract_artifacts(
        self,
        *,
        namespace: str,
        pod: str,
        sidecar_container: str,
        capture_name: str,
        output_dir: Path,
    ) -> dict[str, Any]:
        self.calls.append(("extract_artifacts", locals().copy()))
        self._maybe_fail("extract_artifacts")
        # Touch the files so the test can assert they exist after extract.
        (output_dir / f"{capture_name}.nsys-rep").write_bytes(b"fake-nsys-rep")
        (output_dir / f"{capture_name}_gpu_kern_sum.csv").write_text("fake,csv,1\n")
        (output_dir / f"{capture_name}_cuda_api_sum.csv").write_text("fake,csv,2\n")
        return {
            "cmds": ["FAKE kubectl cp"],
            "local_paths": [
                str(output_dir / f"{capture_name}.nsys-rep"),
                str(output_dir / f"{capture_name}_gpu_kern_sum.csv"),
                str(output_dir / f"{capture_name}_cuda_api_sum.csv"),
            ],
        }


def _make_step_fns(fake: _FakeSteps) -> KernelProfileStepFns:
    return KernelProfileStepFns(
        validate_pod=fake.validate_pod,
        attach_sidecar=fake.attach_sidecar,
        wait_for_sidecar=fake.wait_for_sidecar,
        find_vllm_pid=fake.find_vllm_pid,
        run_nsys_profile=fake.run_nsys_profile,
        extract_artifacts=fake.extract_artifacts,
    )


# ---------------------------------------------------------------------------
# 1. End-to-end: live execution writes all artifacts + records every step
# ---------------------------------------------------------------------------


def test_capture_writes_artifacts_and_invokes_each_step(tmp_path: Path) -> None:
    fake = _FakeSteps()
    out = tmp_path / "out"
    result = capture_kernel_profile(
        namespace="inference",
        pod="basic-inference-abc",
        target_container="basic-inference",
        output_dir=out,
        step_fns=_make_step_fns(fake),
    )
    # Every step ran exactly once, in the documented order.
    assert [c[0] for c in fake.calls] == [
        "validate_pod",
        "attach_sidecar",
        "wait_for_sidecar",
        "find_vllm_pid",
        "run_nsys_profile",
        "extract_artifacts",
    ]
    # The result envelope captured what we expected.
    assert isinstance(result, KernelProfileResult)
    assert result.namespace == "inference"
    assert result.pod == "basic-inference-abc"
    assert result.vllm_pid == 12345
    assert not result.dry_run
    # Files exist + kernel_profile.json was written with full metadata.
    assert result.nsys_rep_path.is_file()
    for p in result.summary_csv_paths:
        assert p.is_file()
    assert result.kernel_profile_json_path.is_file()
    kp_json = json.loads(result.kernel_profile_json_path.read_text())
    assert kp_json["namespace"] == "inference"
    assert kp_json["vllm_pid"] == 12345
    assert kp_json["method"] == "kubectl-debug-share-processes"


# ---------------------------------------------------------------------------
# 2. Dry-run: zero side effects, all 6 steps logged with dry_run flag
# ---------------------------------------------------------------------------


def test_dry_run_skips_steps_2_through_6(tmp_path: Path) -> None:
    fake = _FakeSteps()
    out = tmp_path / "out-dry"
    result = capture_kernel_profile(
        namespace="inference",
        pod="basic-inference-abc",
        target_container="basic-inference",
        output_dir=out,
        step_fns=_make_step_fns(fake),
        dry_run=True,
    )
    # Only validate_pod runs in dry-run (we always check inputs).
    assert [c[0] for c in fake.calls] == ["validate_pod"]
    assert result.dry_run is True
    assert result.vllm_pid is None
    # No nsys-rep file was created (dry-run does not invoke nsys).
    assert not result.nsys_rep_path.exists()
    # step_commands still records all 6 entries so the operator sees the recipe.
    step_names = [s["step"] for s in result.step_commands]
    assert step_names == [
        "validate_pod",
        "attach_sidecar",
        "wait_for_sidecar",
        "find_vllm_pid",
        "run_nsys_profile",
        "extract_artifacts",
    ]
    # The 5 dry-run steps are flagged.
    for step in result.step_commands[1:]:
        assert step.get("dry_run") is True


# ---------------------------------------------------------------------------
# 3. Failure propagation: a failing step aborts the pipeline
# ---------------------------------------------------------------------------


def test_attach_sidecar_failure_propagates(tmp_path: Path) -> None:
    fake = _FakeSteps(fail_step="attach_sidecar")
    with pytest.raises(RuntimeError, match="attach_sidecar"):
        capture_kernel_profile(
            namespace="inference",
            pod="basic-inference-abc",
            target_container="basic-inference",
            output_dir=tmp_path / "out",
            step_fns=_make_step_fns(fake),
        )
    # validate_pod ran, attach_sidecar attempted (raised) -> later steps did not.
    step_names = [c[0] for c in fake.calls]
    assert step_names == ["validate_pod", "attach_sidecar"]


def test_find_pid_failure_propagates(tmp_path: Path) -> None:
    fake = _FakeSteps(fail_step="find_vllm_pid")
    with pytest.raises(RuntimeError, match="find_vllm_pid"):
        capture_kernel_profile(
            namespace="inference",
            pod="basic-inference-abc",
            target_container="basic-inference",
            output_dir=tmp_path / "out",
            step_fns=_make_step_fns(fake),
        )
    step_names = [c[0] for c in fake.calls]
    assert step_names == [
        "validate_pod",
        "attach_sidecar",
        "wait_for_sidecar",
        "find_vllm_pid",
    ]


# ---------------------------------------------------------------------------
# 4. Bundle patching (optional Step 7)
# ---------------------------------------------------------------------------


def test_bundle_patch_writes_kernel_profile_block(tmp_path: Path) -> None:
    fake = _FakeSteps()
    bundle = tmp_path / "fake-bundle"
    bundle.mkdir()
    (bundle / "inference_perfbench_v1.json").write_text(
        json.dumps({"schema": "inference_perfbench_v1", "model": "kimi"})
    )
    out = tmp_path / "out"
    result = capture_kernel_profile(
        namespace="inference",
        pod="basic-inference-abc",
        target_container="basic-inference",
        output_dir=out,
        bundle=bundle,
        step_fns=_make_step_fns(fake),
    )
    assert result.bundle_patched is True
    patched = json.loads((bundle / "inference_perfbench_v1.json").read_text())
    assert "kernel_profile" in patched
    assert patched["kernel_profile"]["vllm_pid"] == 12345
    # Existing fields are preserved.
    assert patched["model"] == "kimi"


def test_no_bundle_means_no_patch(tmp_path: Path) -> None:
    fake = _FakeSteps()
    out = tmp_path / "out"
    result = capture_kernel_profile(
        namespace="inference",
        pod="basic-inference-abc",
        target_container="basic-inference",
        output_dir=out,
        step_fns=_make_step_fns(fake),
    )
    assert result.bundle_patched is False


def test_bundle_without_inference_perfbench_json_is_skipped(tmp_path: Path) -> None:
    fake = _FakeSteps()
    bundle = tmp_path / "no-meta-bundle"
    bundle.mkdir()
    out = tmp_path / "out"
    result = capture_kernel_profile(
        namespace="inference",
        pod="basic-inference-abc",
        target_container="basic-inference",
        output_dir=out,
        bundle=bundle,
        step_fns=_make_step_fns(fake),
    )
    # No inference_perfbench_v1.json -> patch is a no-op (not an error).
    assert result.bundle_patched is False


# ---------------------------------------------------------------------------
# 5. Argument plumbing: custom duration, sample, trace pass through to nsys
# ---------------------------------------------------------------------------


def test_custom_nsys_args_passed_to_step(tmp_path: Path) -> None:
    fake = _FakeSteps()
    out = tmp_path / "out"
    capture_kernel_profile(
        namespace="inference",
        pod="basic-inference-abc",
        target_container="basic-inference",
        output_dir=out,
        duration_s=300,
        sample="none",
        trace="cuda,nvtx",
        sampling_frequency=2000,
        vllm_pid_pattern="python -m vllm",
        step_fns=_make_step_fns(fake),
    )
    nsys_call = next(c for c in fake.calls if c[0] == "run_nsys_profile")
    kwargs = nsys_call[1]
    assert kwargs["duration_s"] == 300
    assert kwargs["sample"] == "none"
    assert kwargs["trace"] == "cuda,nvtx"
    assert kwargs["sampling_frequency"] == 2000
    pid_call = next(c for c in fake.calls if c[0] == "find_vllm_pid")
    assert pid_call[1]["vllm_pid_pattern"] == "python -m vllm"


# ---------------------------------------------------------------------------
# 6. CLI surface (CLI invocation: --dry-run does not require ack)
# ---------------------------------------------------------------------------


def test_cli_kernel_profile_refuses_without_ack(tmp_path: Path, capsys) -> None:
    """Live invocation without ack flag returns FATAL (does not run kubectl)."""
    import argparse
    from tools.perf_tune_report.perf_tune_report_cli import cmd_kernel_profile

    ns = argparse.Namespace(
        namespace="inference",
        pod="basic-inference-abc",
        target_container="basic-inference",
        output_dir=str(tmp_path / "out"),
        sidecar_image="ghcr.io/cfregly/nsys-sidecar:0.1.0",
        duration_seconds=120,
        sample="cpu",
        trace="cuda,nvtx,osrt",
        sampling_frequency=1000,
        vllm_pid_pattern="vllm serve",
        bundle=None,
        i_understand_this_submits_jobs=False,
        dry_run=False,
        json=False,
    )
    rc = cmd_kernel_profile(ns)
    assert rc == 2
    captured = capsys.readouterr()
    assert "ack-gated" in captured.err
