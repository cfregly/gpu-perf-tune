"""Unit tests for ai_tuning.py."""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "ai_tuning.py"

SPEC = importlib.util.spec_from_file_location("ai_tuning", SCRIPT)

assert SPEC is not None and SPEC.loader is not None

ai_tuning = importlib.util.module_from_spec(SPEC)

SPEC.loader.exec_module(ai_tuning)

REPO_ROOT = Path(__file__).resolve().parents[2]

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "ai_tuning"

class UtcTimestampTest(unittest.TestCase):
    """Locks Python 3.10 compat for the UTC timestamp helpers.

    `dt.UTC` is Python 3.11+; the cluster Python is 3.10, so historically
    `experiment create` and `default_derived_path` crashed with
    `AttributeError: module 'datetime' has no attribute 'UTC'`. These tests
    exercise both call sites so a future regression to `dt.UTC` would fail
    on a 3.10 import path.
    """

    def test_utc_timestamp_is_z_suffixed_iso(self) -> None:
        ts = ai_tuning.utc_timestamp()
        self.assertTrue(ts.endswith("Z"), msg=f"expected Z-suffix, got {ts!r}")
        self.assertNotIn("+00:00", ts)
        # ISO-8601 second precision: YYYY-MM-DDTHH:MM:SSZ
        self.assertEqual(len(ts), 20, msg=f"expected 20-char ISO timestamp, got {ts!r}")

    def test_default_derived_path_appends_compact_timestamp(self) -> None:
        target = Path("/tmp/foo.hyp")
        derived = ai_tuning.default_derived_path(target)
        self.assertEqual(derived.parent, target.parent)
        self.assertEqual(derived.suffix, target.suffix)
        # filename: <stem>.cursor-<YYYYMMDDTHHMMSSZ><suffix>
        self.assertTrue(derived.stem.startswith("foo.cursor-"), msg=str(derived))
        self.assertTrue(derived.stem.endswith("Z"), msg=str(derived))

class V37ProposalDiffTests(unittest.TestCase):
    """v3.7 W10: ai_tuning proposal diff PROPOSAL1 PROPOSAL2."""

    def _proposal(self, candidates: list[dict]) -> dict:
        return {
            "schema_version": ai_tuning.PROPOSAL_SCHEMA_VERSION,
            "candidates": candidates,
        }

    def test_proposal_diff_detects_added_removed_changed(self) -> None:
        before = self._proposal(
            [
                {
                    "experiment_id_prefix": "stable",
                    "parameters": {"NCCL_MIN_CTAS": "16", "NCCL_RAIL_PLANE": "0"},
                },
                {
                    "experiment_id_prefix": "removed-only",
                    "parameters": {"NCCL_TC": "96"},
                },
            ]
        )
        after = self._proposal(
            [
                {
                    "experiment_id_prefix": "stable",
                    "parameters": {"NCCL_MIN_CTAS": "32", "NCCL_IB_TC": "106"},
                },
                {
                    "experiment_id_prefix": "added-only",
                    "parameters": {"NCCL_RAIL_PLANE": "1"},
                },
            ]
        )
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            before_path = tmp / "before.json"
            after_path = tmp / "after.json"
            before_path.write_text(json.dumps(before), encoding="utf-8")
            after_path.write_text(json.dumps(after), encoding="utf-8")
            json_out = tmp / "diff.json"
            rc = ai_tuning.command_proposal_diff(
                argparse.Namespace(
                    before=before_path,
                    after=after_path,
                    format="json",
                    output=json_out,
                )
            )
            self.assertEqual(rc, 0)
            payload = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(payload["added_candidates"], ["added-only"])
            self.assertEqual(payload["removed_candidates"], ["removed-only"])
            self.assertEqual(len(payload["changed_candidates"]), 1)
            stable = payload["changed_candidates"][0]
            self.assertEqual(stable["experiment_id_prefix"], "stable")
            added_names = {p["name"] for p in stable["added_parameters"]}
            removed_names = {p["name"] for p in stable["removed_parameters"]}
            changed_names = {p["name"] for p in stable["changed_parameters"]}
            self.assertEqual(added_names, {"NCCL_IB_TC"})
            self.assertEqual(removed_names, {"NCCL_RAIL_PLANE"})
            self.assertEqual(changed_names, {"NCCL_MIN_CTAS"})

    def test_proposal_diff_unchanged_for_byte_identical(self) -> None:
        proposal = self._proposal(
            [
                {
                    "experiment_id_prefix": "stable",
                    "parameters": {"NCCL_MIN_CTAS": "16"},
                }
            ]
        )
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            p = tmp / "p.json"
            p.write_text(json.dumps(proposal, sort_keys=True), encoding="utf-8")
            json_out = tmp / "diff.json"
            rc = ai_tuning.command_proposal_diff(
                argparse.Namespace(
                    before=p,
                    after=p,
                    format="json",
                    output=json_out,
                )
            )
            self.assertEqual(rc, 0)
            payload = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertTrue(payload["unchanged"])
            self.assertEqual(payload["added_candidates"], [])
            self.assertEqual(payload["removed_candidates"], [])
            self.assertEqual(payload["changed_candidates"], [])

    def test_proposal_diff_markdown_includes_sections(self) -> None:
        before = self._proposal(
            [
                {
                    "experiment_id_prefix": "stable",
                    "parameters": {"NCCL_MIN_CTAS": "16"},
                }
            ]
        )
        after = self._proposal(
            [
                {
                    "experiment_id_prefix": "stable",
                    "parameters": {"NCCL_MIN_CTAS": "32"},
                },
                {
                    "experiment_id_prefix": "new",
                    "parameters": {"NCCL_RAIL_PLANE": "1"},
                },
            ]
        )
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            before_path = tmp / "before.json"
            after_path = tmp / "after.json"
            before_path.write_text(json.dumps(before), encoding="utf-8")
            after_path.write_text(json.dumps(after), encoding="utf-8")
            md_out = tmp / "diff.md"
            rc = ai_tuning.command_proposal_diff(
                argparse.Namespace(
                    before=before_path,
                    after=after_path,
                    format="markdown",
                    output=md_out,
                )
            )
            self.assertEqual(rc, 0)
            md = md_out.read_text(encoding="utf-8")
            self.assertIn("# proposal diff", md)
            self.assertIn("Added candidates", md)
            self.assertIn("Changed candidates", md)
            self.assertIn("NCCL_MIN_CTAS: 16->32", md)


if __name__ == "__main__":
    unittest.main()

class AiTuningCliTestPart1(unittest.TestCase):
    def make_finalize_fixture(self, root: Path, run_count: int = 5) -> tuple[Path, Path]:
        workdir = root / "workdir"
        log_dir = root / "logs"
        workdir.mkdir()
        log_dir.mkdir()
        (workdir / "config_gb300_128x4.sh").write_text("config\n", encoding="utf-8")
        (workdir / "run.sub").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        (log_dir / "container-env-123.log").write_text("env\n", encoding="utf-8")
        for index in range(1, run_count + 1):
            run_id = f"run{index}"
            (log_dir / f"{run_id}_1.log").write_text(
                ':::MLLOG {"key":"submission_benchmark","value":"llama31_405b","time_ms":0}\n'
                ':::MLLOG {"key":"run_start","value":null,"time_ms":1000}\n'
                ':::MLLOG {"key":"eval_accuracy","value":3.29,"time_ms":2000}\n'
                ':::MLLOG {"key":"run_stop","value":null,"metadata":{"status":"success"},"time_ms":61000}\n',
                encoding="utf-8",
            )
            (log_dir / f"compliance_{run_id}.out").write_text("SUCCESS\n", encoding="utf-8")
            (log_dir / f"audit_{run_id}.out").write_text("SUCCESS\n", encoding="utf-8")
        return workdir, log_dir

    def test_finalize_dry_run_selects_required_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workdir, log_dir = self.make_finalize_fixture(root)
            output = root / "finalize.json"
            rc = ai_tuning.main(
                [
                    "finalize",
                    "--log-dir",
                    str(log_dir),
                    "--workdir",
                    str(workdir),
                    "--results-dir",
                    str(root / "results"),
                    "--required-runs",
                    "5",
                    "--dry-run",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["valid"])
            self.assertTrue(report["dry_run"])
            self.assertEqual(len(report["selected_runs"]), 5)
            self.assertGreaterEqual(report["copied_file_count"], 17)

    def test_finalize_fails_when_insufficient_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workdir, log_dir = self.make_finalize_fixture(root, run_count=1)
            output = root / "finalize.json"
            rc = ai_tuning.main(
                [
                    "finalize",
                    "--log-dir",
                    str(log_dir),
                    "--workdir",
                    str(workdir),
                    "--results-dir",
                    str(root / "results"),
                    "--required-runs",
                    "5",
                    "--dry-run",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(report["valid"])

    def test_matrix_generates_bounded_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "matrix.json"
            rc = ai_tuning.main(
                [
                    "matrix",
                    "--parameter",
                    "RUN_MODE",
                    "--parameter",
                    "EXPERIMENT_NODES",
                    "--limit",
                    "3",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)
            proposal = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(proposal["schema_version"], 1)
            self.assertEqual(len(proposal["candidates"]), 3)
            self.assertTrue(proposal["truncated"])

    def test_optimizer_propose_records_objective_and_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "optimizer-proposal.json"
            rc = ai_tuning.main(
                [
                    "optimizer",
                    "propose",
                    "--space",
                    str(REPO_ROOT / "tuning" / "tuning-space.gb300-ops.json"),
                    "--strategy",
                    "random",
                    "--parameter",
                    "RUN_STAGE",
                    "--parameter",
                    "GATE_NODES",
                    "--limit",
                    "2",
                    "--seed",
                    "7",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)
            proposal = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(proposal["optimizer_state"]["strategy"], "random")
            self.assertEqual(proposal["optimizer_state"]["seed"], 7)
            self.assertEqual(proposal["optimizer_state"]["objective"], "gb300_operational_readiness")
            self.assertEqual(len(proposal["candidates"]), 2)
            self.assertEqual(proposal["candidates"][0]["priority"], 1)

    def test_optimizer_propose_random_handles_huge_search_space(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            space = Path(tmp) / "space.json"
            output = Path(tmp) / "optimizer-proposal.json"
            parameters = []
            for index in range(8):
                parameters.append(
                    {
                        "name": f"P{index}",
                        "category": "test",
                        "kind": "enum",
                        "values": [str(item) for item in range(20)],
                        "wire": "test",
                        "description": "test",
                    }
                )
            space.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "id": "huge-space",
                        "target": {"benchmark": "test"},
                        "objectives": [
                            {
                                "name": "search_progress",
                                "direction": "maximize",
                                "primary_metric": "candidate_count",
                                "minimum_successful_runs": 1,
                            }
                        ],
                        "parameters": parameters,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rc = ai_tuning.main(
                [
                    "optimizer",
                    "propose",
                    "--space",
                    str(space),
                    "--strategy",
                    "random",
                    "--limit",
                    "1",
                    "--seed",
                    "3",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)
            proposal = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(proposal["total_possible_candidates"], 20**8)
            self.assertEqual(len(proposal["candidates"]), 1)

    def test_report_summarizes_valid_raw_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            ledger = Path(tmp) / "ledger.jsonl"
            ai_tuning.append_jsonl(
                ledger,
                {
                    "schema_version": 1,
                    "event": "created",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "experiment_id": "exp-fixture",
                    "status": "failed",
                    "parameters": {"RUN_MODE": "smoke", "EXPERIMENT_NODES": "1"},
                },
            )
            rc = ai_tuning.main(
                [
                    "report",
                    "--raw-results-dir",
                    str(FIXTURES / "raw-run-dir"),
                    "--raw-benchmark",
                    "llama31_8b",
                    "--min-runs",
                    "1",
                    "--ledger",
                    str(ledger),
                    "--template-hint-file",
                    str(FIXTURES / "structural.sbatch"),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["validation"]["passed"])
            self.assertEqual(report["raw_results"][0]["run_log_count"], 1)
            self.assertEqual(report["raw_results"][0]["runs"][0]["final_log_ppl"], 3.2)
            self.assertIn("objective_scoring", report["trial_analysis"])
            self.assertEqual(
                report["trial_analysis"]["objective_scoring"]["submission_readiness_score"],
                1.0,
            )
            self.assertIn(
                "mlperf_legality_gates",
                report["trial_analysis"],
            )
            session = report["agent_session"]
            self.assertGreater(session["remaining_candidates"]["total"], 0)
            self.assertEqual(session["counts"]["failed"], 1)
            self.assertEqual(session["optimizer_state"]["strategy"], "deterministic_matrix")
            self.assertTrue(session["template_hints"]["hints"])
            coverage_names = {item["name"] for item in session["metrics"]["parameter_coverage"]}
            self.assertIn("RUN_MODE", coverage_names)

    def test_report_uses_selected_objective_minimum_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            rc = ai_tuning.main(
                [
                    "report",
                    "--space",
                    str(REPO_ROOT / "tuning" / "tuning-space.b200-llama31-8b.json"),
                    "--objective",
                    "time_to_quality",
                    "--raw-results-dir",
                    str(FIXTURES / "raw-run-dir"),
                    "--raw-benchmark",
                    "llama31_8b",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["readiness"]["objective"], "time_to_quality")
            self.assertEqual(report["readiness"]["required_successful_runs"], 2)
            self.assertEqual(report["trial_analysis"]["objective_scoring"]["primary_objective"], "time_to_quality")
            self.assertEqual(report["trial_analysis"]["objective_scoring"]["objective_progress_ratio"], 0.5)

    def test_report_defaults_to_manifest_objective_for_gb300(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            rc = ai_tuning.main(
                [
                    "report",
                    "--space",
                    str(REPO_ROOT / "tuning" / "tuning-space.gb300-ops.json"),
                    "--raw-benchmark",
                    "llama31_405b",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["readiness"]["objective"], "gb300_operational_readiness")
            self.assertEqual(report["readiness"]["required_successful_runs"], 1)
            self.assertEqual(
                report["trial_analysis"]["objective_scoring"]["primary_objective"],
                "gb300_operational_readiness",
            )

    def test_proposal_validate_accepts_known_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "proposal-report.json"
            audit_dir = Path(tmp) / "audit"
            rc = ai_tuning.main(
                [
                    "proposal",
                    "validate",
                    str(FIXTURES / "proposal-valid.json"),
                    "--audit-dir",
                    str(audit_dir),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["valid_count"], 1)
            self.assertEqual(report["results"][0]["priority"], 1)
            self.assertTrue((audit_dir / "proposal_audit.jsonl").is_file())
            self.assertIn("partial_candidate", report["results"][0]["warning_codes"])

    def test_proposal_validate_rejects_invalid_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proposal = Path(tmp) / "bad-priority.json"
            output = Path(tmp) / "proposal-report.json"
            proposal.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "candidates": [
                            {
                                "parameters": {"NEXP": "1"},
                                "priority": 0,
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rc = ai_tuning.main(
                [
                    "proposal",
                    "validate",
                    str(proposal),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("invalid_priority", report["results"][0]["error_codes"])

    def test_proposal_validate_requires_objective_when_space_defines_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proposal = Path(tmp) / "missing-objective.json"
            output = Path(tmp) / "proposal-report.json"
            proposal.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "candidates": [
                            {
                                "parameters": {"NEXP": "1"},
                                "priority": 1,
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rc = ai_tuning.main(
                [
                    "proposal",
                    "validate",
                    str(proposal),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("missing_objective", report["results"][0]["error_codes"])

    def test_proposal_validate_requires_config_patch_for_trainer_parameter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proposal = Path(tmp) / "trainer-proposal.json"
            output = Path(tmp) / "proposal-report.json"
            proposal.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "candidates": [
                            {
                                "objective": "time_to_quality",
                                "parameters": {"TRAINER_MICRO_BATCH_SIZE": "2"},
                                "priority": 1,
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rc = ai_tuning.main(
                [
                    "proposal",
                    "validate",
                    str(proposal),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            result = report["results"][0]
            self.assertIn("missing_config_patch", result["error_codes"])
            self.assertFalse(result["contract_validation"]["required_config_patches"][0]["provided"])

    def test_proposal_validate_accepts_trainer_parameter_with_config_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proposal = Path(tmp) / "trainer-proposal.json"
            output = Path(tmp) / "proposal-report.json"
            proposal.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "candidates": [
                            {
                                "objective": "time_to_quality",
                                "parameters": {"TRAINER_MICRO_BATCH_SIZE": "2"},
                                "priority": 1,
                                "config_patches": [
                                    {
                                        "parameter": "TRAINER_MICRO_BATCH_SIZE",
                                        "target_file": "config_DGXB200_8x8x1xtp1pp1cp1_8b_fp4.sh",
                                        "changes": [
                                            {
                                                "match_context": "export MICRO_BATCH_SIZE=1\nexport OTHER=1",
                                                "replacement": "export MICRO_BATCH_SIZE=2\nexport OTHER=1",
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rc = ai_tuning.main(
                [
                    "proposal",
                    "validate",
                    str(proposal),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            result = report["results"][0]
            self.assertTrue(result["valid"])
            self.assertTrue(result["contract_validation"]["objective_known"])
            self.assertTrue(result["contract_validation"]["required_config_patches"][0]["provided"])

class AiTuningCliTestPart2(unittest.TestCase):
    def make_finalize_fixture(self, root: Path, run_count: int = 5) -> tuple[Path, Path]:
        workdir = root / "workdir"
        log_dir = root / "logs"
        workdir.mkdir()
        log_dir.mkdir()
        (workdir / "config_gb300_128x4.sh").write_text("config\n", encoding="utf-8")
        (workdir / "run.sub").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        (log_dir / "container-env-123.log").write_text("env\n", encoding="utf-8")
        for index in range(1, run_count + 1):
            run_id = f"run{index}"
            (log_dir / f"{run_id}_1.log").write_text(
                ':::MLLOG {"key":"submission_benchmark","value":"llama31_405b","time_ms":0}\n'
                ':::MLLOG {"key":"run_start","value":null,"time_ms":1000}\n'
                ':::MLLOG {"key":"eval_accuracy","value":3.29,"time_ms":2000}\n'
                ':::MLLOG {"key":"run_stop","value":null,"metadata":{"status":"success"},"time_ms":61000}\n',
                encoding="utf-8",
            )
            (log_dir / f"compliance_{run_id}.out").write_text("SUCCESS\n", encoding="utf-8")
            (log_dir / f"audit_{run_id}.out").write_text("SUCCESS\n", encoding="utf-8")
        return workdir, log_dir

    def test_proposal_validate_rejects_wrong_config_patch_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proposal = Path(tmp) / "trainer-proposal.json"
            output = Path(tmp) / "proposal-report.json"
            proposal.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "candidates": [
                            {
                                "objective": "time_to_quality",
                                "parameters": {"TRAINER_MICRO_BATCH_SIZE": "2"},
                                "priority": 1,
                                "config_patches": [
                                    {
                                        "parameter": "TRAINER_MICRO_BATCH_SIZE",
                                        "target_file": "wrong-config.sh",
                                        "changes": [
                                            {
                                                "match_context": "export MICRO_BATCH_SIZE=1\nexport OTHER=1",
                                                "replacement": "export MICRO_BATCH_SIZE=2\nexport OTHER=1",
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rc = ai_tuning.main(
                [
                    "proposal",
                    "validate",
                    str(proposal),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("invalid_config_patch_target", report["results"][0]["error_codes"])

    def test_proposal_validate_can_require_complete_finite_space(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "proposal-report.json"
            rc = ai_tuning.main(
                [
                    "proposal",
                    "validate",
                    str(FIXTURES / "proposal-valid.json"),
                    "--require-complete",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("missing_finite_parameters", report["results"][0]["error_codes"])

    def test_proposal_validate_rejects_unknown_and_invalid_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "proposal-report.json"
            rc = ai_tuning.main(
                [
                    "proposal",
                    "validate",
                    str(FIXTURES / "proposal-invalid.json"),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            result = report["results"][0]
            self.assertIn("unknown_parameter", result["error_codes"])
            self.assertIn("invalid_parameter_value", result["error_codes"])

    def test_proposal_validate_rejects_empty_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proposal = Path(tmp) / "empty-proposal.json"
            output = Path(tmp) / "proposal-report.json"
            proposal.write_text(
                json.dumps({"schema_version": 1, "candidates": []}) + "\n",
                encoding="utf-8",
            )
            rc = ai_tuning.main(
                [
                    "proposal",
                    "validate",
                    str(proposal),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["results"][0]["error_codes"], ["invalid_proposal_schema"])

    def test_proposal_validate_rejects_non_object_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proposal = Path(tmp) / "bad-proposal.json"
            output = Path(tmp) / "proposal-report.json"
            proposal.write_text(
                json.dumps({"schema_version": 1, "candidates": ["bad"]}) + "\n",
                encoding="utf-8",
            )
            rc = ai_tuning.main(
                [
                    "proposal",
                    "validate",
                    str(proposal),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("invalid_candidate", report["results"][0]["error_codes"])

    def test_template_patch_apply_writes_derived_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "patch-report.json"
            derived = Path(tmp) / "derived-config.sh"
            rc = ai_tuning.main(
                [
                    "template-patch",
                    "validate",
                    str(FIXTURES / "template-patch-safe.json"),
                    "--apply",
                    "--output-file",
                    str(derived),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["safe"])
            self.assertTrue(report["applied"])
            self.assertIn("export NCCL_DEBUG=INFO", derived.read_text(encoding="utf-8"))

    def test_template_patch_rejects_submission_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "patch-report.json"
            derived = Path(tmp) / "derived-config.sh"
            rc = ai_tuning.main(
                [
                    "template-patch",
                    "validate",
                    str(FIXTURES / "template-patch-unsafe.json"),
                    "--apply",
                    "--output-file",
                    str(derived),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(report["safe"])
            self.assertFalse(report["applied"])
            self.assertFalse(derived.exists())
            self.assertIn("submit_slurm_job", [error["code"] for error in report["errors"]])

    def test_template_patch_rejects_indented_submission_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "config.sh"
            patch = Path(tmp) / "patch.json"
            output = Path(tmp) / "patch-report.json"
            target.write_text(
                "export NCCL_DEBUG=WARN\nexport NEXP=1\nexport DATADIR=/mnt/data/llama31-8b/dataset\n",
                encoding="utf-8",
            )
            patch.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "target_file": str(target),
                        "changes": [
                            {
                                "match_context": "export NCCL_DEBUG=WARN\nexport NEXP=1\nexport DATADIR=/mnt/data/llama31-8b/dataset",
                                "replacement": "export NCCL_DEBUG=WARN\n  sbatch run.sub\nexport DATADIR=/mnt/data/llama31-8b/dataset",
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rc = ai_tuning.main(
                [
                    "template-patch",
                    "validate",
                    str(patch),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("submit_slurm_job", [error["code"] for error in report["errors"]])

    def test_template_patch_rejects_output_over_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "patch-report.json"
            rc = ai_tuning.main(
                [
                    "template-patch",
                    "validate",
                    str(FIXTURES / "template-patch-safe.json"),
                    "--apply",
                    "--output-file",
                    str(FIXTURES / "copied-config.sh"),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("output_overwrites_target", [error["code"] for error in report["errors"]])

    def test_template_patch_rejects_removed_structural_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "patch-report.json"
            rc = ai_tuning.main(
                [
                    "template-patch",
                    "validate",
                    str(FIXTURES / "template-patch-removes-structure.json"),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("removed_sbatch_nodes", [error["code"] for error in report["errors"]])

    def test_template_patch_rejects_removed_gres_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "run.sub"
            patch = Path(tmp) / "patch.json"
            output = Path(tmp) / "patch-report.json"
            original = "#SBATCH --nodes=8\n#SBATCH --partition=<partition>\n#SBATCH --gres=gpu:b200:8\nenv -i bash run.sh\n"
            target.write_text(original, encoding="utf-8")
            patch.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "target_file": str(target),
                        "changes": [
                            {
                                "match_context": original,
                                "replacement": "#SBATCH --nodes=8\n#SBATCH --partition=<partition>\nenv -i bash run.sh\n",
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rc = ai_tuning.main(
                [
                    "template-patch",
                    "validate",
                    str(patch),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("removed_sbatch_gres", [error["code"] for error in report["errors"]])

    def test_experiment_create_update_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proposal = Path(tmp) / "proposal.json"
            ledger = Path(tmp) / "ledger.jsonl"
            create_output = Path(tmp) / "create.json"
            update_output = Path(tmp) / "update.json"
            summary_output = Path(tmp) / "summary.json"
            ai_tuning.main(
                [
                    "matrix",
                    "--parameter",
                    "RUN_MODE",
                    "--parameter",
                    "EXPERIMENT_NODES",
                    "--limit",
                    "2",
                    "--output",
                    str(proposal),
                ]
            )
            rc = ai_tuning.main(
                [
                    "experiment",
                    "create",
                    str(proposal),
                    "--ledger",
                    str(ledger),
                    "--artifact-root",
                    str(Path(tmp) / "artifacts"),
                    "--output",
                    str(create_output),
                ]
            )
            self.assertEqual(rc, 0)
            created = json.loads(create_output.read_text(encoding="utf-8"))
            self.assertEqual(created["created_count"], 2)
            exp_id = created["experiments"][0]["experiment_id"]

            rc = ai_tuning.main(
                [
                    "experiment",
                    "update",
                    exp_id,
                    "--ledger",
                    str(ledger),
                    "--status",
                    "submitted",
                    "--slurm-job-id",
                    "12345",
                    "--output",
                    str(update_output),
                ]
            )
            self.assertEqual(rc, 0)
            rc = ai_tuning.main(
                [
                    "experiment",
                    "summary",
                    "--ledger",
                    str(ledger),
                    "--output",
                    str(summary_output),
                ]
            )
            self.assertEqual(rc, 0)
            summary = json.loads(summary_output.read_text(encoding="utf-8"))
            self.assertEqual(summary["experiment_count"], 2)
            submitted = [
                item for item in summary["experiments"] if item["experiment_id"] == exp_id
            ][0]
            self.assertEqual(submitted["status"], "submitted")
            self.assertEqual(submitted["slurm_job_id"], "12345")

    def test_experiment_create_preserves_objective_and_config_patch_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proposal = Path(tmp) / "proposal.json"
            ledger = Path(tmp) / "ledger.jsonl"
            create_output = Path(tmp) / "create.json"
            summary_output = Path(tmp) / "summary.json"
            proposal.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "optimizer_state": {"strategy": "random", "seed": 9},
                        "candidates": [
                            {
                                "objective": "time_to_quality",
                                "parameters": {"TRAINER_MICRO_BATCH_SIZE": "2"},
                                "priority": 1,
                                "config_patches": [
                                    {
                                        "parameter": "TRAINER_MICRO_BATCH_SIZE",
                                        "target_file": "config_DGXB200_8x8x1xtp1pp1cp1_8b_fp4.sh",
                                        "changes": [
                                            {
                                                "match_context": "export MICRO_BATCH_SIZE=1\nexport OTHER=1",
                                                "replacement": "export MICRO_BATCH_SIZE=2\nexport OTHER=1",
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rc = ai_tuning.main(
                [
                    "experiment",
                    "create",
                    str(proposal),
                    "--space",
                    str(REPO_ROOT / "tuning" / "tuning-space.b200-llama31-8b.json"),
                    "--ledger",
                    str(ledger),
                    "--output",
                    str(create_output),
                ]
            )
            self.assertEqual(rc, 0)
            rc = ai_tuning.main(
                [
                    "experiment",
                    "summary",
                    "--ledger",
                    str(ledger),
                    "--output",
                    str(summary_output),
                ]
            )
            self.assertEqual(rc, 0)
            summary = json.loads(summary_output.read_text(encoding="utf-8"))
            record = summary["experiments"][0]
            self.assertEqual(record["objective"], "time_to_quality")
            self.assertEqual(record["optimizer_state"]["strategy"], "random")
            self.assertTrue(record["config_patches"])

    def test_experiment_update_rejects_unknown_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                ai_tuning.main(
                    [
                        "experiment",
                        "update",
                        "exp-bad",
                        "--ledger",
                        str(Path(tmp) / "ledger.jsonl"),
                        "--status",
                        "mystery",
                    ]
                )

class AiTuningCliTestPart3(unittest.TestCase):
    def make_finalize_fixture(self, root: Path, run_count: int = 5) -> tuple[Path, Path]:
        workdir = root / "workdir"
        log_dir = root / "logs"
        workdir.mkdir()
        log_dir.mkdir()
        (workdir / "config_gb300_128x4.sh").write_text("config\n", encoding="utf-8")
        (workdir / "run.sub").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        (log_dir / "container-env-123.log").write_text("env\n", encoding="utf-8")
        for index in range(1, run_count + 1):
            run_id = f"run{index}"
            (log_dir / f"{run_id}_1.log").write_text(
                ':::MLLOG {"key":"submission_benchmark","value":"llama31_405b","time_ms":0}\n'
                ':::MLLOG {"key":"run_start","value":null,"time_ms":1000}\n'
                ':::MLLOG {"key":"eval_accuracy","value":3.29,"time_ms":2000}\n'
                ':::MLLOG {"key":"run_stop","value":null,"metadata":{"status":"success"},"time_ms":61000}\n',
                encoding="utf-8",
            )
            (log_dir / f"compliance_{run_id}.out").write_text("SUCCESS\n", encoding="utf-8")
            (log_dir / f"audit_{run_id}.out").write_text("SUCCESS\n", encoding="utf-8")
        return workdir, log_dir

    def test_experiment_submit_previews_without_executing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proposal = Path(tmp) / "proposal.json"
            ledger = Path(tmp) / "ledger.jsonl"
            create_output = Path(tmp) / "create.json"
            submit_output = Path(tmp) / "submit.json"
            ai_tuning.main(
                [
                    "matrix",
                    "--parameter",
                    "RUN_MODE",
                    "--parameter",
                    "EXPERIMENT_NODES",
                    "--limit",
                    "2",
                    "--output",
                    str(proposal),
                ]
            )
            ai_tuning.main(
                [
                    "experiment",
                    "create",
                    str(proposal),
                    "--ledger",
                    str(ledger),
                    "--output",
                    str(create_output),
                ]
            )
            rc = ai_tuning.main(
                [
                    "experiment",
                    "submit",
                    "--ledger",
                    str(ledger),
                    "--script",
                    str(FIXTURES / "submit.sbatch"),
                    "--max-concurrent",
                    "1",
                    "--output",
                    str(submit_output),
                ]
            )
            self.assertEqual(rc, 0)
            report = json.loads(submit_output.read_text(encoding="utf-8"))
            self.assertFalse(report["execute"])
            self.assertEqual(report["selected_count"], 1)
            self.assertEqual(report["updates_written"], 0)
            self.assertIn("sbatch", report["submissions"][0]["command"][0])

    def test_experiment_submit_requires_wrappers_for_config_patches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            script = Path(tmp) / "run.sub"
            config = Path(tmp) / "config_DGXB200_8x8x1xtp1pp1cp1_8b_fp4.sh"
            script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            config.write_text("export MICRO_BATCH_SIZE=1\nexport OTHER=1\n", encoding="utf-8")
            ai_tuning.append_jsonl(
                ledger,
                {
                    "schema_version": 1,
                    "event": "created",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "experiment_id": "exp-patch",
                    "status": "planned",
                    "parameters": {"TRAINER_MICRO_BATCH_SIZE": "2"},
                    "config_patches": [
                        {
                            "parameter": "TRAINER_MICRO_BATCH_SIZE",
                            "target_file": config.name,
                            "changes": [
                                {
                                    "match_context": "export MICRO_BATCH_SIZE=1\nexport OTHER=1",
                                    "replacement": "export MICRO_BATCH_SIZE=2\nexport OTHER=1",
                                }
                            ],
                        }
                    ],
                },
            )
            with self.assertRaises(SystemExit):
                ai_tuning.main(
                    [
                        "experiment",
                        "submit",
                        "--ledger",
                        str(ledger),
                        "--script",
                        str(script),
                    ]
                )

    def test_experiment_submit_honors_explicit_id_and_materializes_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            wrapper_dir = Path(tmp) / "wrappers"
            output = Path(tmp) / "submit.json"
            ai_tuning.append_jsonl(
                ledger,
                {
                    "schema_version": 1,
                    "event": "created",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "experiment_id": "exp-low",
                    "status": "planned",
                    "priority": 2,
                    "shape": {"nodes": "8", "partition": "gb300"},
                    "parameters": {"NCCL_IB_TC": "96", "UCX_TLS": "self,tcp"},
                },
            )
            ai_tuning.append_jsonl(
                ledger,
                {
                    "schema_version": 1,
                    "event": "created",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "experiment_id": "exp-high",
                    "status": "planned",
                    "priority": 1,
                    "shape": {"nodes": "16", "partition": "gb300"},
                    "parameters": {"NCCL_IB_TC": "106", "UCX_TLS": "self,tcp,sm"},
                },
            )
            rc = ai_tuning.main(
                [
                    "experiment",
                    "submit",
                    "--ledger",
                    str(ledger),
                    "--script",
                    str(FIXTURES / "submit.sbatch"),
                    "--experiment-id",
                    "exp-low",
                    "--materialize-wrappers",
                    "--wrapper-dir",
                    str(wrapper_dir),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["selected_count"], 1)
            self.assertEqual(report["submissions"][0]["experiment_id"], "exp-low")
            wrapper = Path(report["submissions"][0]["wrapper"])
            self.assertTrue(wrapper.is_file())
            wrapper_text = wrapper.read_text(encoding="utf-8")
            self.assertIn("#SBATCH --job-name=ai-tuning-fixture", wrapper_text)
            self.assertIn("export NCCL_IB_TC='96'", wrapper_text)
            self.assertIn("export UCX_TLS='self,tcp'", wrapper_text)
            self.assertIn('$(dirname -- "$0")', wrapper_text)
            self.assertTrue((wrapper_dir / "submit.sbatch").is_file())

    def test_experiment_submit_wrapper_can_use_remote_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            wrapper_dir = Path(tmp) / "wrappers"
            output = Path(tmp) / "submit.json"
            ai_tuning.append_jsonl(
                ledger,
                {
                    "schema_version": 1,
                    "event": "created",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "experiment_id": "exp-remote",
                    "status": "planned",
                    "shape": {"nodes": "8", "partition": "gb300"},
                    "parameters": {"NCCL_IB_TC": "96"},
                },
            )
            rc = ai_tuning.main(
                [
                    "experiment",
                    "submit",
                    "--ledger",
                    str(ledger),
                    "--script",
                    str(FIXTURES / "submit.sbatch"),
                    "--experiment-id",
                    "exp-remote",
                    "--materialize-wrappers",
                    "--wrapper-dir",
                    str(wrapper_dir),
                    "--remote-script",
                    "/mnt/home/$USER/live/submit.sbatch",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)
            wrapper = wrapper_dir / "exp-remote.sbatch"
            self.assertIn(
                "exec bash '/mnt/home/$USER/live/submit.sbatch'",
                wrapper.read_text(encoding="utf-8"),
            )

    def test_experiment_submit_materializes_and_sources_config_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            wrapper_dir = Path(tmp) / "wrappers"
            output = Path(tmp) / "submit.json"
            script = Path(tmp) / "run.sub"
            config = Path(tmp) / "config_DGXB200_8x8x1xtp1pp1cp1_8b_fp4.sh"
            script.write_text("#!/usr/bin/env bash\necho running\n", encoding="utf-8")
            config.write_text("export MICRO_BATCH_SIZE=1\nexport OTHER=1\n", encoding="utf-8")
            ai_tuning.append_jsonl(
                ledger,
                {
                    "schema_version": 1,
                    "event": "created",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "experiment_id": "exp-trainer",
                    "status": "planned",
                    "priority": 1,
                    "objective": "time_to_quality",
                    "shape": {"nodes": "8", "partition": "<partition>"},
                    "parameters": {"TRAINER_MICRO_BATCH_SIZE": "2"},
                    "config_patches": [
                        {
                            "parameter": "TRAINER_MICRO_BATCH_SIZE",
                            "target_file": config.name,
                            "changes": [
                                {
                                    "match_context": "export MICRO_BATCH_SIZE=1\nexport OTHER=1",
                                    "replacement": "export MICRO_BATCH_SIZE=2\nexport OTHER=1",
                                }
                            ],
                        }
                    ],
                },
            )
            rc = ai_tuning.main(
                [
                    "experiment",
                    "submit",
                    "--ledger",
                    str(ledger),
                    "--script",
                    str(script),
                    "--experiment-id",
                    "exp-trainer",
                    "--materialize-wrappers",
                    "--wrapper-dir",
                    str(wrapper_dir),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            wrapper = Path(report["submissions"][0]["wrapper"])
            wrapper_text = wrapper.read_text(encoding="utf-8")
            self.assertIn("# objective=time_to_quality", wrapper_text)
            self.assertIn("source ", wrapper_text)
            derived_config = wrapper_dir / config.name
            self.assertTrue(derived_config.is_file())
            self.assertIn("export MICRO_BATCH_SIZE=2", derived_config.read_text(encoding="utf-8"))

    def test_experiment_submit_requires_double_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            with self.assertRaises(SystemExit):
                ai_tuning.main(
                    [
                        "experiment",
                        "submit",
                        "--ledger",
                        str(ledger),
                        "--script",
                        str(FIXTURES / "submit.sbatch"),
                        "--execute",
                    ]
                )

    def test_experiment_poll_updates_from_status_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            output = Path(tmp) / "poll.json"
            ai_tuning.append_jsonl(
                ledger,
                {
                    "schema_version": 1,
                    "event": "created",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "experiment_id": "exp-fixture",
                    "status": "submitted",
                    "slurm_job_id": "12345",
                    "parameters": {},
                },
            )
            rc = ai_tuning.main(
                [
                    "experiment",
                    "poll",
                    "--ledger",
                    str(ledger),
                    "--status-file",
                    str(FIXTURES / "poll-status.json"),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["updates_written"], 1)
            self.assertEqual(report["observations"][0]["new_status"], "succeeded")

    def test_experiment_collect_copies_and_validates_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            destination = Path(tmp) / "collected"
            output = Path(tmp) / "collect.json"
            ai_tuning.append_jsonl(
                ledger,
                {
                    "schema_version": 1,
                    "event": "created",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "experiment_id": "exp-fixture",
                    "status": "submitted",
                    "slurm_job_id": "12345",
                    "parameters": {},
                },
            )
            rc = ai_tuning.main(
                [
                    "experiment",
                    "collect",
                    "exp-fixture",
                    "--ledger",
                    str(ledger),
                    "--source",
                    str(FIXTURES / "raw-run-dir"),
                    "--destination",
                    str(destination),
                    "--raw-benchmark",
                    "llama31_8b",
                    "--min-runs",
                    "1",
                    "--validate-raw",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["validation_passed"])
            self.assertTrue((destination / "run_a_1.log").is_file())

    def test_experiment_collect_keeps_usable_gid_classification_nonfatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "hca.out").write_text(
                "HCA_GID_USABLE context=container host=gb300 rank=0 device=ibp0p1 port=1 gid_index=3 gid=fe80::1\n",
                encoding="utf-8",
            )
            ledger = root / "ledger.jsonl"
            destination = root / "collected"
            output = root / "collect.json"
            ai_tuning.append_jsonl(
                ledger,
                {
                    "schema_version": 1,
                    "event": "created",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "experiment_id": "exp-gid",
                    "status": "submitted",
                    "parameters": {},
                },
            )
            rc = ai_tuning.main(
                [
                    "experiment",
                    "collect",
                    "exp-gid",
                    "--ledger",
                    str(ledger),
                    "--source",
                    str(source),
                    "--destination",
                    str(destination),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["update"]["status"], "succeeded")
            self.assertEqual(report["failure_classifications"][0]["severity"], "info")

    def test_experiment_collect_refuses_nonempty_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            destination = Path(tmp) / "collected"
            destination.mkdir()
            (destination / "existing.txt").write_text("existing\n", encoding="utf-8")
            ai_tuning.append_jsonl(
                ledger,
                {
                    "schema_version": 1,
                    "event": "created",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "experiment_id": "exp-fixture",
                    "status": "submitted",
                    "parameters": {},
                },
            )
            with self.assertRaises(SystemExit):
                ai_tuning.main(
                    [
                        "experiment",
                        "collect",
                        "exp-fixture",
                        "--ledger",
                        str(ledger),
                        "--source",
                        str(FIXTURES / "raw-run-dir"),
                        "--destination",
                        str(destination),
                        "--raw-benchmark",
                        "llama31_8b",
                        "--min-runs",
                        "1",
                        "--validate-raw",
                    ]
                )

    def test_failure_classifier_detects_nccl_rdma_signature(self) -> None:
        findings = ai_tuning.classify_failure_text(
            "NCCL WARN Call to ibv_create_ah failed with error No such device\n"
            "Call to ncclIbDevxRtrQp failed\n"
            "NCCL WARN Call to ibv_modify_qp failed with 19 No such device"
        )
        codes = {finding["code"] for finding in findings}
        self.assertIn("nccl_ib_create_ah_no_device", codes)
        self.assertIn("nccl_ib_devx_rtr_qp", codes)
        self.assertIn("nccl_ib_modify_qp_no_device", codes)

class AiTuningCliTestPart4(unittest.TestCase):
    def make_finalize_fixture(self, root: Path, run_count: int = 5) -> tuple[Path, Path]:
        workdir = root / "workdir"
        log_dir = root / "logs"
        workdir.mkdir()
        log_dir.mkdir()
        (workdir / "config_gb300_128x4.sh").write_text("config\n", encoding="utf-8")
        (workdir / "run.sub").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        (log_dir / "container-env-123.log").write_text("env\n", encoding="utf-8")
        for index in range(1, run_count + 1):
            run_id = f"run{index}"
            (log_dir / f"{run_id}_1.log").write_text(
                ':::MLLOG {"key":"submission_benchmark","value":"llama31_405b","time_ms":0}\n'
                ':::MLLOG {"key":"run_start","value":null,"time_ms":1000}\n'
                ':::MLLOG {"key":"eval_accuracy","value":3.29,"time_ms":2000}\n'
                ':::MLLOG {"key":"run_stop","value":null,"metadata":{"status":"success"},"time_ms":61000}\n',
                encoding="utf-8",
            )
            (log_dir / f"compliance_{run_id}.out").write_text("SUCCESS\n", encoding="utf-8")
            (log_dir / f"audit_{run_id}.out").write_text("SUCCESS\n", encoding="utf-8")
        return workdir, log_dir

    def test_failure_classifier_detects_slurm_resource_shape_issue(self) -> None:
        findings = ai_tuning.classify_failure_text(
            "srun: error: Unable to create step for job 1171: More processors requested than permitted"
        )
        self.assertEqual(findings[0]["code"], "slurm_more_processors_requested")

    def test_failure_classifier_detects_hca_inventory_summary(self) -> None:
        findings = ai_tuning.classify_failure_text(
            "HCA_GID_SUMMARY context=container host=gb300 rank=0 device=ibp0p1 port=1 total=160 nonzero=0 zero=160\n"
            "HCA_GID_ZERO context=container host=gb300 rank=0 device=ibp0p1 port=1 gid_index=0 gid=0000:0000:0000:0000:0000:0000:0000:0000"
        )
        codes = {finding["code"] for finding in findings}
        self.assertIn("hca_inventory_all_zero_gids", codes)
        self.assertIn("hca_inventory_zero_gid", codes)
        severities = {finding["code"]: finding["severity"] for finding in findings}
        self.assertEqual(severities["hca_inventory_all_zero_gids"], "failure")
        self.assertEqual(severities["hca_inventory_zero_gid"], "warning")

    def test_failure_classifier_detects_full_inventory_all_zero_summary(self) -> None:
        findings = ai_tuning.classify_failure_text(
            "HCA_FULL_PORT_SUMMARY context=container host=gb300 device=ibp0p1 port=1 total_gids=255 nonzero_gids=0 zero_gids=255"
        )
        self.assertEqual(findings[0]["code"], "hca_inventory_full_all_zero_gids")
        self.assertEqual(findings[0]["severity"], "failure")


if __name__ == "__main__":
    unittest.main()
