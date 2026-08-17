"""Public workflow skills must use checked-in helpers and MCP tools."""

from __future__ import annotations

from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = PLUGIN_ROOT / "skills"


def _skill_text(name: str) -> str:
    return (SKILLS_ROOT / name / "SKILL.md").read_text(encoding="utf-8")


def test_optimization_skills_do_not_require_absent_private_helpers() -> None:
    text = "".join(
        _skill_text(name)
        for name in (
            "inference-tune-sweep",
            "inference-model-optimize",
            "inference-perf-tune-report",
            "inference-spec-decode-service",
            "inference-spec-decode-tune",
        )
    )
    absent_helpers = {
        "bench-all-workloads.sh",
        "bench-with-sol.sh",
        "capture-run-env.sh",
        "capture-sol-window.sh",
        "known-good-config-gate.sh",
        "overlay-patchset.sh",
        "pin-node.sh",
        "roofline-sweep.sh",
        "run-controlled-ab.sh",
        "run-variant-ab.sh",
        "scaffold-model-bringup.sh",
        "specdec-decide.py",
        "specdec-loop.sh",
        "specdec-presets.sh",
        "canary-arm.yaml",
        "e3bs-arm.yaml",
        "eagle3-accept-eval.yaml",
        "read-acc0.sh",
        "tune-driver.py",
        "tune-trial.sh",
        "verify-grind-closure.sh",
    }

    assert not {name for name in absent_helpers if name in text}


def test_public_runtime_docs_do_not_claim_absent_capture_helpers() -> None:
    paths = (
        PLUGIN_ROOT / "server" / "docs" / "learnings" / "evidence-shape.md",
        PLUGIN_ROOT / "server" / "docs" / "zymtrace-query-hygiene.md",
        PLUGIN_ROOT / "server" / "tools" / "loader_advisor.py",
        PLUGIN_ROOT / "server" / "tools" / "perf_tune_report" / "importers" / "inference_perf_bench.py",
        PLUGIN_ROOT / "server" / "tools" / "perf_tune_report" / "importers" / "zymtrace_kernels.py",
        PLUGIN_ROOT / "server" / "tools" / "perf_tune_report" / "ROOFLINE-METHODOLOGY.md",
        PLUGIN_ROOT / "server" / "tools" / "perf_tune_report" / "lake_writer.py",
        PLUGIN_ROOT / "server" / "tools" / "perf_tune_report" / "perf_tune_report_cli.py",
        PLUGIN_ROOT / "server" / "tools" / "perf_tune_report" / "importers" / "roofline_sweep.py",
        PLUGIN_ROOT / "server" / "tools" / "perf_tune_report" / "importers" / "variant_ab.py",
        PLUGIN_ROOT / "server" / "tools" / "perf_tune_report" / "importers" / "workloads.py",
        PLUGIN_ROOT / "server" / "tools" / "perf_tune_report" / "renderer" / "champion_select.py",
        PLUGIN_ROOT / "server" / "tools" / "perf_tune_report" / "renderer" / "prefill_decode_roofline.py",
        PLUGIN_ROOT / "server" / "tools" / "perf_tune_report" / "renderer" / "render_report.py",
        PLUGIN_ROOT / "server" / "tools" / "perf_tune_report" / "renderer" / "render_status.py",
        SKILLS_ROOT / "inference-aa-workload" / "SKILL.md",
        SKILLS_ROOT / "inference-kernel-profile" / "SKILL.md",
        SKILLS_ROOT / "inference-kernel-ncu-profile" / "SKILL.md",
        SKILLS_ROOT / "inference-perf-bench" / "SKILL.md",
        Path(__file__).resolve().parents[4] / "scripts" / "zymtrace-ingest-wait.sh",
    )
    text = "".join(path.read_text(encoding="utf-8") for path in paths)
    absent_helpers = {
        "audit_artifact_paths.py",
        "audit_evidence_bundle.py",
        "bench-all-workloads.sh",
        "capture-run-env.sh",
        "capture-sol-window.sh",
        "`capture.sh`",
        "nsys-ngc.yaml",
        "nsys-sglang.yaml",
        "PROFILING-RUNBOOK.md",
        "profiling/capture-bench.sh",
        "roofline-sweep.sh",
        "run-variant-ab.sh",
        "scaffold-model-bringup.sh",
    }

    assert not {name for name in absent_helpers if name in text}


def test_evidence_skill_uses_checked_in_capture_helper() -> None:
    text = _skill_text("evidence-bundle-init")
    helper = PLUGIN_ROOT / "server" / "tools" / "shared" / "capture_cmd.sh"

    assert helper.is_file()
    assert "tools/shared/capture_cmd.sh" in text
    assert "git add -f <bundle>" in text
    assert "Local-only by default" in text
