"""Tests for the kernel_reproducer_scaffold verb (perf_tune_report, v1.69.0)."""
from __future__ import annotations

from pathlib import Path

from tools.perf_tune_report.kernel_reproducer import scaffold_reproducer


def test_scaffold_writes_cu_and_build(tmp_path: Path) -> None:
    r = scaffold_reproducer(
        kernel_name="linear_sm100_mpk_task_impl",
        header="tasks/blackwell/linear_sm100_mpk.cuh",
        out_dir=tmp_path,
        k=6144,
        out=1024,
        batch=8,
    )
    assert r.wrote
    cu = Path(r.cu_path)
    sh = Path(r.build_path)
    assert cu.is_file() and sh.is_file()
    cu_text = cu.read_text()
    # exact kernel + header + dims threaded into the harness
    assert "linear_sm100_mpk_task_impl" in cu_text
    assert "tasks/blackwell/linear_sm100_mpk.cuh" in cu_text
    assert "K = 6144" in cu_text and "OUT = 1024" in cu_text and "BATCH = 8" in cu_text
    # the controlled-input host-GEMM verdict is present
    assert "REPRO_CORRECT" in cu_text and "kernel/ref_ratio" in cu_text
    sh_text = sh.read_text()
    # GB300 build flags
    assert "compute_103a" in sh_text and "-std=c++20" in sh_text
    assert "MPK_ENABLE_TMA" in sh_text and "deps/cutlass/include" in sh_text
    d = r.to_dict()
    assert d["kernel_name"] == "linear_sm100_mpk_task_impl"
    assert d["dims"]["K"] == 6144 and d["dims"]["OUT"] == 1024
    assert d["wrote"] is True


def test_scaffold_dry_run_writes_nothing(tmp_path: Path) -> None:
    r = scaffold_reproducer(
        kernel_name="foo_task_impl",
        header="tasks/x.cuh",
        out_dir=tmp_path,
        dry_run=True,
    )
    assert not r.wrote
    assert not Path(r.cu_path).exists()
    assert not Path(r.build_path).exists()


def test_scaffold_custom_arch_and_tree(tmp_path: Path) -> None:
    r = scaffold_reproducer(
        kernel_name="my_kernel",
        header="tasks/blackwell/my_kernel.cuh",
        out_dir=tmp_path,
        mirage_tree="/work/mirage",
        arch="compute_100a,code=sm_100a",
    )
    sh_text = Path(r.build_path).read_text()
    assert "M=/work/mirage" in sh_text
    assert "compute_100a,code=sm_100a" in sh_text


def test_cli_verb_registered() -> None:
    from tools.perf_tune_report.perf_tune_report_cli import CONTRACT, build_parser

    assert "kernel_reproducer_scaffold" in CONTRACT
    assert CONTRACT["kernel_reproducer_scaffold"]["safety"] == "writes_artifacts"
    parser = build_parser()
    ns = parser.parse_args(
        [
            "kernel_reproducer_scaffold",
            "--kernel-name", "linear_sm100_mpk_task_impl",
            "--header", "tasks/blackwell/linear_sm100_mpk.cuh",
            "--output-dir", "/tmp/x",
            "--dry-run",
            "--json",
        ]
    )
    assert ns.func.__name__ == "cmd_kernel_reproducer_scaffold"
