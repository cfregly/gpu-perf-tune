"""Unit tests for the slurm CLI: triage / drain / resume / quiet_window."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.slurm import slurm_cli
from tools.slurm.signature_patterns import match_signatures
from tools.slurm.slurm_cli import CONTRACT, build_parser, main


def _seed_repo_root(tmp_path: Path) -> Path:
    root = tmp_path / "fake-repo-root"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'fake'\nversion = '0'\n")
    (root / "mcp_surface.py").write_text("# marker\n")
    (root / "tools").mkdir()
    return root


def test_contract_has_four_verbs() -> None:
    assert sorted(CONTRACT.keys()) == ["drain", "quiet_window", "resume", "triage"]


def test_contract_shape() -> None:
    spec = CONTRACT["triage"]
    assert spec["safety"] == "read_only"
    assert spec["ack"] is None
    assert "--jobid" in spec["required"]


def test_operator_path_rejects_nul_bytes() -> None:
    with pytest.raises(ValueError, match="without NUL bytes"):
        slurm_cli._operator_path("bundle\x00outside", label="--bundle")


def test_drain_contract_shape() -> None:
    spec = CONTRACT["drain"]
    assert spec["safety"] == "substitutes_nodes"
    assert spec["ack"] == "--i-understand-this-substitutes-nodes"
    assert "--nodes" in spec["required"]
    assert "--ns" in spec["required"]
    assert spec["json"] is True


def test_resume_contract_shape() -> None:
    spec = CONTRACT["resume"]
    assert spec["safety"] == "substitutes_nodes"
    assert spec["ack"] == "--i-understand-this-substitutes-nodes"
    assert "--nodes" in spec["required"]
    assert "--ns" in spec["required"]


def test_quiet_window_contract_shape() -> None:
    spec = CONTRACT["quiet_window"]
    assert spec["safety"] == "substitutes_nodes"
    assert spec["ack"] == "--i-understand-this-substitutes-nodes"
    assert "--nodes" in spec["required"]
    assert "--cmd" in spec["required"]
    assert "--ns" in spec["required"]


def test_match_signatures_oom() -> None:
    text = "Some progress\nout of memory: killed process 12345 by signal 9\nDone\n"
    hits = match_signatures(text)
    assert hits, "expected at least one match"
    top_sig, count = hits[0]
    assert top_sig.klass == "oom"
    assert count >= 1


def test_match_signatures_shm_bloat_oom_2026_05_15_pattern() -> None:
    """Replicates the 2026-05-15 incident slurm-out signature."""
    text = (
        "ds671b training start...\n"
        "rank 0 dataloader worker pid=12345 started\n"
        "slurm-b200-194-005:125282:125580 [0] NCCL INFO Allocated 34210180 "
        "bytes of shared memory in /dev/shm/nccl-rg380Q\n"
        "rank 7 dataloader worker reading /mnt/data/dsv3-c4 shard 18\n"
        "out of memory: killed process 12345 by signal 9\n"
        "oom-killer: pid=12345 child rss=51_000_000_000\n"
    )
    hits = match_signatures(text)
    assert hits, "expected at least one match"
    top_sig, _ = hits[0]
    assert top_sig.klass == "shm_bloat_oom"
    assert "k8s-troubleshooting" in top_sig.next_probe
    assert "slurm_drain" in top_sig.next_probe


def test_match_signatures_shm_bloat_oom_no_space_on_tmpfs() -> None:
    """Captures the secondary symptom: tmpfs `no space left on device` over /dev/shm."""
    text = "rank 4: OSError [Errno 28] No space left on device: '/dev/shm/torch_5_4444'\n"
    hits = match_signatures(text)
    assert hits, "expected a shm_bloat_oom match"
    assert hits[0][0].klass == "shm_bloat_oom"


def test_match_signatures_plain_oom_without_shm_context_still_classifies_as_generic() -> None:
    """Regression: a pure workload-OOM (no /dev/shm evidence) keeps the generic `oom` class,
    so the existing 'bump --mem' next-probe still wires through for those cases."""
    text = (
        "rank 0 forward pass; cuda allocator: failed to allocate 24 GB\n"
        "out of memory: killed process 9999 by signal 9\n"
        "oom-killer: pid=9999 child rss=78_000_000_000\n"
    )
    hits = match_signatures(text)
    assert hits, "expected at least one match"
    assert hits[0][0].klass == "oom", "plain OOM must NOT be reclassified as shm_bloat_oom"


def test_match_signatures_shm_bloat_priority_over_generic_oom() -> None:
    """When both shm_bloat_oom AND generic oom would match, shm_bloat_oom wins
    (regex priority order: more specific first)."""
    text = (
        "slurm-b200:125282 [0] NCCL INFO Allocated 34210180 bytes of shared memory "
        "in /dev/shm/nccl-rg380Q\n"
        "out of memory: killed process 12345 by signal 9\n"
    )
    hits = match_signatures(text)
    # Both signatures fire. The first hit is shm_bloat_oom (higher priority).
    assert hits[0][0].klass == "shm_bloat_oom"
    klasses = [s.klass for s, _ in hits]
    assert "oom" in klasses, "generic `oom` should also be in hits (priority decides ordering)"


def test_match_signatures_nccl_hang() -> None:
    text = "NCCL WARN: timeout on rank 7 waiting for all-reduce\n"
    hits = match_signatures(text)
    assert hits[0][0].klass == "nccl_hang"


def test_match_signatures_priority_order() -> None:
    """When both OOM and walltime signatures fire, OOM wins (higher priority)."""
    text = "out of memory: killed by signal 9\nDUE TO TIME LIMIT\n"
    hits = match_signatures(text)
    assert hits[0][0].klass == "oom"


def test_match_signatures_unknown() -> None:
    hits = match_signatures("everything is fine\n")
    assert hits == []


def test_triage_cli_unknown_jobid_no_slurmout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _seed_repo_root(tmp_path)
    monkeypatch.setenv("PROFILE_AND_OPTIMIZE_REPO_ROOT", str(root))
    rc = main(["triage", "--jobid", "99999", "--logdir", str(tmp_path), "--json"])
    assert rc == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["tool"] == "slurm_triage"
    assert envelope["jobid"] == "99999"
    assert envelope["slurm_out_path"] is None
    assert envelope["classification"]["class"] == "unknown"


def test_triage_cli_classifies_oom_from_synthetic_slurmout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _seed_repo_root(tmp_path)
    monkeypatch.setenv("PROFILE_AND_OPTIMIZE_REPO_ROOT", str(root))
    logdir = tmp_path / "logdir"
    logdir.mkdir()
    (logdir / "slurm-1234.out").write_text(
        "starting workload\n"
        "out of memory: killed process by signal 9\n"
        "oom-killed: cgroup: child_used_15GB\n"
        "exit code 137\n"
    )
    rc = main(["triage", "--jobid", "1234", "--logdir", str(logdir), "--json"])
    assert rc == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["classification"]["class"] == "oom"
    assert envelope["classification"]["confidence"] in {"HIGH", "MEDIUM"}
    assert envelope["signature_hit_count"] >= 1


def test_triage_repo_root_bounds_recursive_log_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _seed_repo_root(tmp_path)
    nested = root / "runs" / "one"
    nested.mkdir(parents=True)
    expected = nested / "slurm-4242.out"
    expected.write_text("out of memory: killed process by signal 9\n")
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "slurm-4242.out").write_text("everything is fine\n")
    monkeypatch.chdir(unrelated)

    assert main(["triage", "--jobid", "4242", "--repo-root", str(root), "--json"]) == 0
    envelope = json.loads(capsys.readouterr().out)

    assert envelope["repo_root"] == str(root)
    assert envelope["search_root"] == str(root)
    assert envelope["slurm_out_path"] == str(expected)
    assert envelope["classification"]["class"] == "oom"


def test_triage_cli_nccl_hang_recommends_ibcheck_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _seed_repo_root(tmp_path)
    monkeypatch.setenv("PROFILE_AND_OPTIMIZE_REPO_ROOT", str(root))
    logdir = tmp_path / "logdir"
    logdir.mkdir()
    (logdir / "slurm-2222.out").write_text("NCCL WARN: timeout on rank 7 waiting for all-reduce\n")
    rc = main(["triage", "--jobid", "2222", "--logdir", str(logdir), "--json"])
    assert rc == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["classification"]["class"] == "nccl_hang"
    # When sacct is not available (no Slurm on the test workstation), handoff is None.
    # When sacct is available, cluster incidents use the maintained public skill.
    if envelope["handoff_skill"] is not None:
        assert envelope["handoff_skill"] == "k8s-troubleshooting"


def test_cluster_handoff_uses_maintained_public_skill() -> None:
    sacct = {"WorkDir": "/work/experiments/artifacts/campaign/test"}
    assert slurm_cli._cluster_handoff("oom", sacct) == "k8s-troubleshooting"
    assert slurm_cli._cluster_handoff("nccl_hang", {"WorkDir": "/work/run"}) == "k8s-troubleshooting"


def test_build_parser_help_does_not_crash() -> None:
    parser = build_parser()
    assert parser is not None
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


# --------------------------------------------------------------------------- #
# drain / resume / quiet_window tests
# --------------------------------------------------------------------------- #


class _FakeRunner:
    """Records every kubectl-exec invocation and returns scripted CompletedProcess
    results. Tests inject one of these by monkey-patching ``slurm_cli.RUN``."""

    def __init__(self, scripted: list[subprocess.CompletedProcess[str]]) -> None:
        self.scripted = list(scripted)
        self.calls: list[list[str]] = []

    def __call__(
        self,
        argv: list[str],
        *,
        capture: bool = True,
        timeout: int | None = 60,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        if not self.scripted:
            return subprocess.CompletedProcess(args=list(argv), returncode=0, stdout="", stderr="")
        return self.scripted.pop(0)


def _ok(stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=stderr)


def _fail(rc: int = 1, stderr: str = "boom") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout="", stderr=stderr)


def _common_drain_args(bundle: Path) -> list[str]:
    return [
        "--nodes",
        "slurm-b200-001,slurm-b200-002",
        "--reason",
        "test reason",
        "--ns",
        "test-namespace",
        "--ctl",
        "deploy/slurm-controller",
        "--ctl-container",
        "slurmctld",
        "--bundle",
        str(bundle),
        "--json",
        "--i-understand-this-substitutes-nodes",
    ]


def test_drain_writes_evidence_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = tmp_path / "bundle"
    runner = _FakeRunner(
        [
            _ok(stdout="slurm-b200-001 idle all\nslurm-b200-002 idle all\n"),  # 01 sinfo
            _ok(stdout=""),  # 02 scontrol drain
            _ok(stdout="slurm-b200-001 drained all\nslurm-b200-002 drained all\n"),  # 03 sinfo
        ]
    )
    monkeypatch.setattr(slurm_cli, "RUN", runner)

    rc = main(["drain", *_common_drain_args(bundle)])
    assert rc == 0

    envelope = json.loads(capsys.readouterr().out)
    assert envelope["tool"] == "slurm_drain"
    assert envelope["safety"] == "substitutes_nodes"
    assert envelope["ack_field"] == "--i-understand-this-substitutes-nodes"
    assert envelope["nodes"] == "slurm-b200-001,slurm-b200-002"
    assert envelope["reason"] == "test reason"
    assert envelope["scontrol_exit"] == 0

    cmds = bundle / "commands"
    assert (cmds / "01-slurm-pre-state.stdout").read_text().startswith("slurm-b200-001 idle")
    assert (cmds / "02-drain.cmd").read_text().startswith("kubectl ")
    assert (cmds / "02-drain.exit").read_text().strip() == "0"
    assert (cmds / "02-drain.reason").read_text().strip() == "test reason"
    assert (cmds / "03-slurm-post-drain.stdout").read_text().startswith("slurm-b200-001 drained")

    assert "State=DRAIN" in " ".join(runner.calls[1])
    assert "Reason=test reason" in " ".join(runner.calls[1])


def test_drain_refuses_without_ack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = tmp_path / "bundle"
    runner = _FakeRunner([])
    monkeypatch.setattr(slurm_cli, "RUN", runner)
    args = [
        "drain",
        "--nodes",
        "x,y",
        "--ns",
        "test-namespace",
        "--bundle",
        str(bundle),
    ]
    with pytest.raises(SystemExit) as excinfo:
        main(args)
    assert "ack-gated" in str(excinfo.value)
    # Refused before any kubectl call.
    assert runner.calls == []


def test_resume_writes_step_files_5_and_6(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = tmp_path / "bundle"
    runner = _FakeRunner(
        [
            _ok(stdout="slurm-b200-001 drained all\n"),  # 01 sinfo
            _ok(stdout=""),  # 05 scontrol resume
            _ok(stdout="slurm-b200-001 idle all\n"),  # 06 sinfo
        ]
    )
    monkeypatch.setattr(slurm_cli, "RUN", runner)

    rc = main(["resume", *_common_drain_args(bundle)])
    assert rc == 0

    cmds = bundle / "commands"
    # Resume uses step IDs 05 + 06 to match the kimi-k26 precedent.
    assert (cmds / "05-resume.cmd").read_text().startswith("kubectl ")
    assert (cmds / "05-resume.exit").read_text().strip() == "0"
    assert (cmds / "06-slurm-post-resume.stdout").read_text().startswith("slurm-b200-001 idle")
    assert "State=RESUME" in " ".join(runner.calls[1])


def test_quiet_window_writes_full_six_step_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = tmp_path / "bundle"
    runner = _FakeRunner(
        [
            _ok(stdout="pre-idle\n"),  # 01
            _ok(stdout=""),  # 02 scontrol drain
            _ok(stdout="drained\n"),  # 03
            # 04 inner cmd is the only call NOT going through kubectl;
            # see the inner_runner indirection below.
            _ok(stdout=""),  # 05 scontrol resume
            _ok(stdout="post-idle\n"),  # 06
        ]
    )
    inner_calls: list[list[str]] = []

    def _wrapped_run(argv: list[str], *, capture: bool = True, timeout: int | None = 60):
        # Inner cmd is the bench: not a kubectl invocation. Detect by argv[0].
        if argv and argv[0] != "kubectl":
            inner_calls.append(list(argv))
            return subprocess.CompletedProcess(args=argv, returncode=0, stdout="bench-ok\n", stderr="")
        return runner(argv, capture=capture, timeout=timeout)

    monkeypatch.setattr(slurm_cli, "RUN", _wrapped_run)

    rc = main(
        [
            "quiet_window",
            *_common_drain_args(bundle),
            "--cmd",
            "true",
        ]
    )
    assert rc == 0
    assert inner_calls == [["true"]]

    cmds = bundle / "commands"
    expected = [
        "01-slurm-pre-state.stdout",
        "02-drain.cmd",
        "02-drain.stdout",
        "02-drain.stderr",
        "02-drain.exit",
        "02-drain.reason",
        "03-slurm-post-drain.stdout",
        "04-bench-start.txt",
        "04-bench.cmd",
        "04-bench.stdout",
        "04-bench.stderr",
        "04-bench.exit",
        "04-bench-end.txt",
        "05-resume.cmd",
        "05-resume.stdout",
        "05-resume.stderr",
        "05-resume.exit",
        "06-slurm-post-resume.stdout",
    ]
    for fn in expected:
        assert (cmds / fn).is_file(), f"missing evidence file: {fn}"
    # Inner exit code propagated.
    assert (cmds / "04-bench.exit").read_text().strip() == "0"
    # Resume happened despite green inner cmd.
    assert (cmds / "05-resume.exit").read_text().strip() == "0"


def test_quiet_window_resumes_when_inner_cmd_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """If the inner bench cmd exits nonzero, resume MUST still run via finally."""
    bundle = tmp_path / "bundle"

    def _runner(argv: list[str], *, capture: bool = True, timeout: int | None = 60):
        if argv and argv[0] != "kubectl":
            # Inner cmd FAILS.
            return subprocess.CompletedProcess(args=argv, returncode=42, stdout="", stderr="bench-failed\n")
        # All kubectl calls (sinfo + scontrol drain + scontrol resume + sinfo) succeed.
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(slurm_cli, "RUN", _runner)

    rc = main(
        [
            "quiet_window",
            *_common_drain_args(bundle),
            "--cmd",
            "false",
        ]
    )
    # Inner rc propagates as the verb's rc.
    assert rc == 42

    cmds = bundle / "commands"
    # Inner cmd exit captured.
    assert (cmds / "04-bench.exit").read_text().strip() == "42"
    # Resume STILL happened.
    assert (cmds / "05-resume.exit").is_file()
    assert (cmds / "05-resume.exit").read_text().strip() == "0"
    # Post-resume sinfo still captured.
    assert (cmds / "06-slurm-post-resume.stdout").is_file()


def test_quiet_window_resumes_when_inner_cmd_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An interrupted inner command still triggers a resume attempt."""
    bundle = tmp_path / "bundle"
    resume_called = {"yes": False}

    def _runner(argv: list[str], *, capture: bool = True, timeout: int | None = 60):
        if argv and argv[0] != "kubectl":
            raise KeyboardInterrupt()
        # Detect the resume call so we can flip the flag.
        if "State=RESUME" in " ".join(argv):
            resume_called["yes"] = True
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(slurm_cli, "RUN", _runner)

    with pytest.raises(KeyboardInterrupt):
        main(
            [
                "quiet_window",
                *_common_drain_args(bundle),
                "--cmd",
                "true",
            ]
        )

    assert resume_called["yes"], "resume must run even when inner cmd raises"


def test_quiet_window_compensates_after_drain_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed drain skips the inner command and attempts a compensating resume."""
    bundle = tmp_path / "bundle"
    inner_called = {"yes": False}
    resume_called = {"yes": False}

    def _runner(argv: list[str], *, capture: bool = True, timeout: int | None = 60):
        joined = " ".join(argv)
        if argv and argv[0] != "kubectl":
            inner_called["yes"] = True
            return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")
        if "State=DRAIN" in joined:
            return subprocess.CompletedProcess(args=argv, returncode=1, stdout="", stderr="permission denied\n")
        if "State=RESUME" in joined:
            resume_called["yes"] = True
            return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(slurm_cli, "RUN", _runner)

    rc = main(
        [
            "quiet_window",
            *_common_drain_args(bundle),
            "--cmd",
            "true",
        ]
    )
    assert rc == 1
    assert inner_called["yes"] is False, "inner cmd must NOT run when drain failed"
    assert resume_called["yes"] is True

    envelope = json.loads(capsys.readouterr().out)
    assert envelope["inner_skipped"] is True
    assert envelope["resume_skipped"] is False
    assert envelope["drain_exit"] == 1
    assert envelope["compensation_attempted"] is True
    assert envelope["compensation_exit"] == 0


@pytest.mark.parametrize("drain_rc", [0, 1])
def test_quiet_window_snapshot_exception_cannot_preempt_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    drain_rc: int,
) -> None:
    bundle = tmp_path / "bundle"
    sinfo_calls = 0
    resume_called = False
    inner_called = False

    def _runner(argv: list[str], *, capture: bool = True, timeout: int | None = 60):
        nonlocal sinfo_calls, resume_called, inner_called
        joined = " ".join(argv)
        if argv and argv[0] != "kubectl":
            inner_called = True
            return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")
        if "State=DRAIN" in joined:
            return subprocess.CompletedProcess(
                args=argv, returncode=drain_rc, stdout="", stderr="drain failed\n" if drain_rc else ""
            )
        if "State=RESUME" in joined:
            resume_called = True
            return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")
        if "sinfo" in joined:
            sinfo_calls += 1
            if sinfo_calls == 2:
                raise subprocess.TimeoutExpired(argv, 30)
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(slurm_cli, "RUN", _runner)
    rc = main(["quiet_window", *_common_drain_args(bundle), "--cmd", "true"])

    assert rc != 0
    assert resume_called is True
    assert inner_called is False
    envelope = json.loads(capsys.readouterr().out)
    assert "TimeoutExpired" in " ".join(envelope["diagnostic_errors"])


def test_quiet_window_drain_receipt_failure_cannot_preempt_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle = tmp_path / "bundle"
    resume_called = False
    original_capture = slurm_cli._capture_step

    def _capture(*args, **kwargs):
        if args[1] == "02-drain":
            raise OSError("artifact filesystem unavailable")
        return original_capture(*args, **kwargs)

    def _runner(argv: list[str], *, capture: bool = True, timeout: int | None = 60):
        nonlocal resume_called
        if "State=RESUME" in " ".join(argv):
            resume_called = True
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(slurm_cli, "_capture_step", _capture)
    monkeypatch.setattr(slurm_cli, "RUN", _runner)
    rc = main(["quiet_window", *_common_drain_args(bundle), "--cmd", "true"])

    assert rc == 255
    assert resume_called is True
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["inner_skipped"] is True
    assert "artifact filesystem unavailable" in " ".join(envelope["diagnostic_errors"])


def test_quiet_window_resume_failure_after_green_inner_forces_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = tmp_path / "bundle"

    def _runner(argv: list[str], *, capture: bool = True, timeout: int | None = 60):
        joined = " ".join(argv)
        if argv and argv[0] != "kubectl":
            return subprocess.CompletedProcess(args=argv, returncode=0, stdout="ok\n", stderr="")
        if "State=RESUME" in joined:
            return subprocess.CompletedProcess(
                args=argv,
                returncode=7,
                stdout="",
                stderr="resume rejected\n",
            )
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(slurm_cli, "RUN", _runner)
    rc = main(["quiet_window", *_common_drain_args(bundle), "--cmd", "true"])

    assert rc == 7
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["inner_exit"] == 0
    assert envelope["resume_exit"] == 7
    assert envelope["resume_stderr"] == "resume rejected\n"
    assert envelope["resume_exception"] is None


def test_quiet_window_resume_exception_after_green_inner_forces_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = tmp_path / "bundle"

    def _runner(argv: list[str], *, capture: bool = True, timeout: int | None = 60):
        joined = " ".join(argv)
        if argv and argv[0] != "kubectl":
            return subprocess.CompletedProcess(args=argv, returncode=0, stdout="ok\n", stderr="")
        if "State=RESUME" in joined:
            raise RuntimeError("controller unavailable")
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(slurm_cli, "RUN", _runner)
    rc = main(["quiet_window", *_common_drain_args(bundle), "--cmd", "true"])

    assert rc == 255
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["inner_exit"] == 0
    assert envelope["resume_exit"] == 255
    assert "controller unavailable" in envelope["resume_exception"]
    assert (bundle / "commands" / "05-resume.exit").read_text().strip() == "255"


def test_quiet_window_retries_interrupted_resume_and_reports_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle = tmp_path / "bundle"
    resume_attempts = 0

    def _runner(argv: list[str], *, capture: bool = True, timeout: int | None = 60):
        nonlocal resume_attempts
        joined = " ".join(argv)
        if argv and argv[0] != "kubectl":
            return subprocess.CompletedProcess(args=argv, returncode=0, stdout="ok\n", stderr="")
        if "State=RESUME" in joined:
            resume_attempts += 1
            if resume_attempts == 1:
                raise KeyboardInterrupt()
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(slurm_cli, "RUN", _runner)
    rc = main(["quiet_window", *_common_drain_args(bundle), "--cmd", "true"])

    assert rc == 255
    assert resume_attempts == 2
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["resume_exit"] == 0
    assert "KeyboardInterrupt" in envelope["resume_exception"]
    assert "attempt 1" in (bundle / "commands" / "05-resume.stderr").read_text()


def test_quiet_window_refuses_without_ack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = tmp_path / "bundle"
    runner = _FakeRunner([])
    monkeypatch.setattr(slurm_cli, "RUN", runner)
    with pytest.raises(SystemExit):
        main(
            [
                "quiet_window",
                "--nodes",
                "x",
                "--ns",
                "test-namespace",
                "--cmd",
                "true",
                "--bundle",
                str(bundle),
            ]
        )
    assert runner.calls == []


def test_quiet_window_refuses_empty_inner_cmd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = tmp_path / "bundle"
    runner = _FakeRunner([])
    monkeypatch.setattr(slurm_cli, "RUN", runner)
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "quiet_window",
                *_common_drain_args(bundle),
                "--cmd",
                "   ",
            ]
        )
    assert "empty argv" in str(excinfo.value)


def test_helpers_build_kubectl_argv_correctly() -> None:
    argv = slurm_cli._ctl_exec_argv(
        "test-namespace",
        "deploy/slurm-controller",
        "slurmctld",
        ["scontrol", "update", "NodeName=x", "State=DRAIN", "Reason=foo"],
    )
    assert argv == [
        "kubectl",
        "-n",
        "test-namespace",
        "exec",
        "deploy/slurm-controller",
        "-c",
        "slurmctld",
        "--",
        "scontrol",
        "update",
        "NodeName=x",
        "State=DRAIN",
        "Reason=foo",
    ]


def test_scontrol_update_argv_basic() -> None:
    assert slurm_cli._scontrol_update_argv("a,b", "DRAIN", "rsn") == [
        "scontrol",
        "update",
        "NodeName=a,b",
        "State=DRAIN",
        "Reason=rsn",
    ]
