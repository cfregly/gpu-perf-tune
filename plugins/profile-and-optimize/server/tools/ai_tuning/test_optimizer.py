"""Unit tests for ai_tuning optimizer helpers."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "ai_tuning.py"

PACKAGE_DIR = Path(__file__).resolve().parent

if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

SPEC = importlib.util.spec_from_file_location("ai_tuning", SCRIPT)

assert SPEC is not None and SPEC.loader is not None

ai_tuning = importlib.util.module_from_spec(SPEC)

SPEC.loader.exec_module(ai_tuning)

from optimizer import gp, hyp_format, hyp_session, hyperband, space, tpe  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "ai_tuning"

def _make_space(values_per_dim: int = 5) -> space.Space:
    dims = [
        space.Dimension(
            name="x",
            kind="continuous",
            minimum=0.0,
            maximum=1.0,
        ),
        space.Dimension(
            name="y",
            kind="continuous",
            minimum=0.0,
            maximum=1.0,
        ),
    ]
    return space.Space(dimensions=dims)

def _quadratic(point: dict[str, str]) -> float:
    """Maximized at (0.7, 0.3)."""
    x = float(point["x"])
    y = float(point["y"])
    return -((x - 0.7) ** 2 + (y - 0.3) ** 2)

class SpaceTest(unittest.TestCase):
    def test_categorical_round_trip(self) -> None:
        dim = space.Dimension(name="k", kind="categorical", values=["a", "b", "c"])
        for value in ["a", "b", "c"]:
            self.assertEqual(dim.decode(dim.encode(value)), value)

    def test_integer_round_trip(self) -> None:
        dim = space.Dimension(name="n", kind="integer", minimum=1, maximum=4)
        for value in ["1", "2", "3", "4"]:
            self.assertEqual(dim.decode(dim.encode(value)), value)

    def test_boolean_decode(self) -> None:
        dim = space.Dimension(name="b", kind="boolean", values=["0", "1"])
        self.assertEqual(dim.decode(0.0), "0")
        self.assertEqual(dim.decode(1.0), "1")

    def test_total_finite(self) -> None:
        finite_space = space.Space(
            dimensions=[
                space.Dimension(name="k", kind="categorical", values=["a", "b"]),
                space.Dimension(name="n", kind="integer", minimum=1, maximum=3),
            ]
        )
        self.assertEqual(finite_space.total_finite(), 2 * 3)

if __name__ == "__main__":
    unittest.main()

class TpeTest(unittest.TestCase):
    def test_cold_start_returns_random_until_min_observations(self) -> None:
        rng = random.Random(0)
        candidates, state = tpe.propose(
            _make_space(),
            observations=[],
            direction="maximize",
            rng=rng,
            min_observations=4,
            limit=2,
        )
        self.assertEqual(len(candidates), 2)
        self.assertTrue(state["cold_start"])

    def test_tpe_outperforms_random_on_quadratic(self) -> None:
        rng_random = random.Random(7)
        rng_tpe = random.Random(7)
        opt_space = _make_space()
        random_observations: list[tpe.Observation] = []
        tpe_observations: list[tpe.Observation] = []
        for _ in range(8):
            rand_params = opt_space.random(rng_random)
            random_observations.append(
                tpe.Observation(vector=opt_space.encode(rand_params), value=_quadratic(rand_params))
            )
            tpe_params = opt_space.random(rng_tpe)
            tpe_observations.append(
                tpe.Observation(vector=opt_space.encode(tpe_params), value=_quadratic(tpe_params))
            )

        for _ in range(20):
            tpe_candidates, _ = tpe.propose(
                opt_space,
                tpe_observations,
                direction="maximize",
                rng=rng_tpe,
                min_observations=4,
                limit=1,
            )
            params = tpe_candidates[0]
            tpe_observations.append(
                tpe.Observation(vector=opt_space.encode(params), value=_quadratic(params))
            )
            rand_params = opt_space.random(rng_random)
            random_observations.append(
                tpe.Observation(vector=opt_space.encode(rand_params), value=_quadratic(rand_params))
            )

        best_random = max(o.value for o in random_observations)
        best_tpe = max(o.value for o in tpe_observations)
        self.assertGreaterEqual(best_tpe, best_random)

if __name__ == "__main__":
    unittest.main()

class GpTest(unittest.TestCase):
    def test_predict_recovers_observed_values(self) -> None:
        vectors = [[0.1, 0.2], [0.5, 0.4], [0.9, 0.8]]
        targets = [0.5, 1.5, -0.5]
        model = gp.fit(vectors, targets, length_scale_candidates=(0.3,), noise_variance=1e-6)
        for vector, target in zip(vectors, targets):
            mean, _variance = gp.predict(model, vector)
            self.assertAlmostEqual(mean, target, places=2)

    def test_predict_rejects_dimension_mismatch(self) -> None:
        model = gp.fit([[0.1, 0.2], [0.5, 0.4]], [0.5, 1.5])
        with self.assertRaises(ValueError):
            gp.predict(model, [0.1])

    def test_gp_outperforms_random_on_quadratic(self) -> None:
        rng_random = random.Random(11)
        rng_gp = random.Random(11)
        opt_space = _make_space()
        random_observations: list[tpe.Observation] = []
        gp_observations: list[tpe.Observation] = []
        for _ in range(4):
            rand_params = opt_space.random(rng_random)
            random_observations.append(
                tpe.Observation(vector=opt_space.encode(rand_params), value=_quadratic(rand_params))
            )
            gp_params = opt_space.random(rng_gp)
            gp_observations.append(
                tpe.Observation(vector=opt_space.encode(gp_params), value=_quadratic(gp_params))
            )

        for _ in range(15):
            gp_candidates, _ = gp.propose(
                opt_space,
                gp_observations,
                direction="maximize",
                rng=rng_gp,
                min_observations=2,
                limit=1,
            )
            params = gp_candidates[0]
            gp_observations.append(
                tpe.Observation(vector=opt_space.encode(params), value=_quadratic(params))
            )
            rand_params = opt_space.random(rng_random)
            random_observations.append(
                tpe.Observation(vector=opt_space.encode(rand_params), value=_quadratic(rand_params))
            )

        best_random = max(o.value for o in random_observations)
        best_gp = max(o.value for o in gp_observations)
        self.assertGreater(best_gp, best_random - 0.05)

if __name__ == "__main__":
    unittest.main()

class HyperbandTest(unittest.TestCase):
    def test_plan_produces_brackets(self) -> None:
        config = hyperband.HyperbandConfig(eta=3, min_budget=1, max_budget=27)
        brackets = hyperband.plan(config)
        self.assertGreater(len(brackets), 0)
        first_bracket = brackets[0]
        self.assertGreater(len(first_bracket.rungs), 0)
        first_rung = first_bracket.rungs[0]
        self.assertEqual(first_rung.budget, 1.0)

    def test_propose_emits_initial_bracket(self) -> None:
        config = hyperband.HyperbandConfig(eta=3, min_budget=1, max_budget=9)
        opt_space = _make_space()
        rng = random.Random(0)
        candidates, state = hyperband.propose(
            opt_space,
            observations=[],
            direction="minimize",
            rng=rng,
            config=config,
            variant="hyperband",
        )
        self.assertGreater(len(candidates), 0)
        self.assertEqual(state["current_rung"], 0)
        self.assertIn("emitted_count", state)

    def test_advance_rung_uses_observations(self) -> None:
        config = hyperband.HyperbandConfig(eta=2, min_budget=1, max_budget=4)
        opt_space = _make_space()
        rng = random.Random(0)
        candidates, state = hyperband.propose(
            opt_space,
            observations=[],
            direction="minimize",
            rng=rng,
            config=config,
            variant="hyperband",
        )
        observations = [
            tpe.Observation(
                vector=opt_space.encode(c["parameters"]),
                value=float(index),  # smaller index has smaller value
            )
            for index, c in enumerate(candidates)
        ]
        next_state = hyperband.advance_rung(state)
        next_candidates, next_state = hyperband.propose(
            opt_space,
            observations,
            direction="minimize",
            rng=rng,
            config=config,
            variant="hyperband",
            state_in=next_state,
        )
        self.assertGreater(len(next_candidates), 0)
        self.assertEqual(next_state["current_rung"], 1)
        self.assertGreater(next_state["current_budget"], 1.0)

if __name__ == "__main__":
    unittest.main()

class HypFormatTest(unittest.TestCase):
    def test_parses_real_hyp_fixture(self) -> None:
        fixture = FIXTURES / "hyp" / "scaled_run.tp-comm-overlap-true.hyp"
        self.assertTrue(fixture.is_file())
        params = hyp_format.parse_file(fixture)
        names = {param.name for param in params}
        self.assertIn("MINIBS", names)
        self.assertIn("TENSOR_MODEL_PARALLEL", names)
        manifest = hyp_format.to_manifest_parameters(params)
        kinds = {entry["name"]: entry["kind"] for entry in manifest}
        self.assertEqual(kinds["TENSOR_MODEL_PARALLEL"], "integer")

if __name__ == "__main__":
    unittest.main()

class HypSessionTest(unittest.TestCase):
    def test_imports_synthetic_session(self) -> None:
        session_dir = FIXTURES / "hyp-session"
        session = hyp_session.import_session(
            session_dir,
            parameter_names={
                "TENSOR_MODEL_PARALLEL",
                "PIPELINE_MODEL_PARALLEL",
                "MICRO_BATCH_SIZE",
                "INTERLEAVED_PIPELINE",
                "LR",
            },
        )
        self.assertEqual(len(session.trials), 3)
        scores = sorted(trial.score for trial in session.trials if trial.score is not None)
        self.assertEqual(len(scores), 3)
        self.assertAlmostEqual(scores[0], 10.875, places=3)
        self.assertAlmostEqual(scores[-1], 12.100, places=3)

if __name__ == "__main__":
    unittest.main()

class CliTest(unittest.TestCase):
    def test_optimizer_status_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "status.json"
            rc = ai_tuning.main(
                [
                    "optimizer",
                    "status",
                    "--space",
                    str(REPO_ROOT / "tuning" / "tuning-space.b200-llama31-8b.json"),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("bayesian", payload["available_strategies"])
            self.assertIn("multifidelity", payload["available_strategies"])
            self.assertTrue(payload["contracts_implemented"]["objective_scoring"])
            self.assertTrue(payload["optimizer_capabilities"]["tpe_landed"])
            self.assertTrue(payload["optimizer_capabilities"]["gp_landed"])

    def test_optimizer_propose_bayesian_tpe_cold_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "proposal.json"
            rc = ai_tuning.main(
                [
                    "optimizer",
                    "propose",
                    "--space",
                    str(REPO_ROOT / "tuning" / "tuning-space.gb300-ops.json"),
                    "--strategy",
                    "bayesian",
                    "--variant",
                    "tpe",
                    "--parameter",
                    "GATE_NODES",
                    "--parameter",
                    "NCCL_IB_TC",
                    "--limit",
                    "2",
                    "--seed",
                    "9",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["optimizer_state"]["strategy"], "bayesian")
            self.assertEqual(payload["optimizer_state"]["variant"], "tpe")
            self.assertTrue(payload["optimizer_state"]["cold_start"])
            self.assertEqual(len(payload["candidates"]), 2)

    def test_optimizer_propose_bayesian_gp_cold_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "proposal.json"
            rc = ai_tuning.main(
                [
                    "optimizer",
                    "propose",
                    "--space",
                    str(REPO_ROOT / "tuning" / "tuning-space.gb300-ops.json"),
                    "--strategy",
                    "bayesian",
                    "--variant",
                    "gp",
                    "--parameter",
                    "GATE_NODES",
                    "--parameter",
                    "NCCL_IB_TC",
                    "--limit",
                    "2",
                    "--seed",
                    "9",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["optimizer_state"]["strategy"], "bayesian")
            self.assertEqual(payload["optimizer_state"]["variant"], "gp")
            self.assertTrue(payload["optimizer_state"]["cold_start"])
            self.assertEqual(len(payload["candidates"]), 2)

    def test_optimizer_propose_multifidelity_emits_fidelity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "proposal.json"
            rc = ai_tuning.main(
                [
                    "optimizer",
                    "propose",
                    "--space",
                    str(REPO_ROOT / "tuning" / "tuning-space.gb300-ops.json"),
                    "--strategy",
                    "multifidelity",
                    "--variant",
                    "hyperband",
                    "--parameter",
                    "NCCL_IB_TC",
                    "--parameter",
                    "NCCL_RAIL_PLANE",
                    "--eta",
                    "2",
                    "--min-budget",
                    "1",
                    "--max-budget",
                    "4",
                    "--limit",
                    "4",
                    "--seed",
                    "3",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["optimizer_state"]["strategy"], "multifidelity")
            self.assertEqual(payload["optimizer_state"]["variant"], "hyperband")
            self.assertGreater(len(payload["candidates"]), 0)
            for candidate in payload["candidates"]:
                self.assertIn("fidelity", candidate)

    def test_optimizer_propose_multifidelity_bohb_emits_fidelity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "proposal.json"
            rc = ai_tuning.main(
                [
                    "optimizer",
                    "propose",
                    "--space",
                    str(REPO_ROOT / "tuning" / "tuning-space.gb300-ops.json"),
                    "--strategy",
                    "multifidelity",
                    "--variant",
                    "bohb",
                    "--parameter",
                    "NCCL_IB_TC",
                    "--parameter",
                    "NCCL_RAIL_PLANE",
                    "--eta",
                    "2",
                    "--min-budget",
                    "1",
                    "--max-budget",
                    "4",
                    "--limit",
                    "4",
                    "--seed",
                    "3",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["optimizer_state"]["strategy"], "multifidelity")
            self.assertEqual(payload["optimizer_state"]["variant"], "bohb")
            self.assertGreater(len(payload["candidates"]), 0)
            for candidate in payload["candidates"]:
                self.assertIn("fidelity", candidate)

    def test_optimizer_propose_state_out_round_trip_advances_rung(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            first_output = Path(tmp) / "first.json"
            second_output = Path(tmp) / "second.json"
            base_args = [
                "optimizer",
                "propose",
                "--space",
                str(REPO_ROOT / "tuning" / "tuning-space.gb300-ops.json"),
                "--strategy",
                "multifidelity",
                "--variant",
                "hyperband",
                "--parameter",
                "NCCL_IB_TC",
                "--parameter",
                "NCCL_RAIL_PLANE",
                "--eta",
                "2",
                "--min-budget",
                "1",
                "--max-budget",
                "4",
                "--seed",
                "3",
            ]
            rc = ai_tuning.main(base_args + ["--state-out", str(state_file), "--output", str(first_output)])
            self.assertEqual(rc, 0)
            self.assertTrue(state_file.is_file())
            first = json.loads(first_output.read_text(encoding="utf-8"))
            self.assertEqual(first["optimizer_state"]["current_rung"], 0)
            self.assertTrue(first["optimizer_state"]["bracket_keys"])

            rc = ai_tuning.main(base_args + ["--state-file", str(state_file), "--output", str(second_output)])
            self.assertEqual(rc, 0)
            second = json.loads(second_output.read_text(encoding="utf-8"))
            self.assertEqual(second["optimizer_state"]["current_rung"], 1)
            self.assertGreater(second["optimizer_state"]["current_budget"], first["optimizer_state"]["current_budget"])
            persisted_state = json.loads(state_file.read_text(encoding="utf-8"))
            self.assertEqual(persisted_state["rung_index"], 2)

    def test_optimizer_propose_missing_state_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "proposal.json"
            missing = Path(tmp) / "missing-state.json"
            with self.assertRaises(SystemExit):
                ai_tuning.main(
                    [
                        "optimizer",
                        "propose",
                        "--space",
                        str(REPO_ROOT / "tuning" / "tuning-space.gb300-ops.json"),
                        "--strategy",
                        "multifidelity",
                        "--variant",
                        "hyperband",
                        "--state-file",
                        str(missing),
                        "--output",
                        str(output),
                    ]
                )

    def test_optimizer_compare_returns_winner_by_objective(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            output = Path(tmp) / "compare.json"
            ai_tuning.append_jsonl(
                ledger,
                {
                    "schema_version": 1,
                    "event": "created",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "experiment_id": "exp-a",
                    "status": "succeeded",
                    "parameters": {"NEXP": "1"},
                    "objective": "mlperf_submission_readiness",
                    "result_value": 0.9,
                },
            )
            ai_tuning.append_jsonl(
                ledger,
                {
                    "schema_version": 1,
                    "event": "created",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "experiment_id": "exp-b",
                    "status": "succeeded",
                    "parameters": {"NEXP": "1"},
                    "objective": "mlperf_submission_readiness",
                    "result_value": 0.6,
                },
            )
            rc = ai_tuning.main(
                [
                    "optimizer",
                    "compare",
                    "exp-a",
                    "exp-b",
                    "--ledger",
                    str(ledger),
                    "--space",
                    str(REPO_ROOT / "tuning" / "tuning-space.b200-llama31-8b.json"),
                    "--objective",
                    "mlperf_submission_readiness",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["winner"], "exp-a")
            self.assertEqual(payload["direction"], "maximize")
            self.assertAlmostEqual(payload["metric_deltas"]["result_value_delta"], 0.3)

    def test_optimizer_history_groups_and_sorts_by_objective(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            output = Path(tmp) / "history.json"
            for exp_id, value, variant in [
                ("exp-a", 0.6, "random"),
                ("exp-b", 0.9, "tpe"),
            ]:
                ai_tuning.append_jsonl(
                    ledger,
                    {
                        "schema_version": 1,
                        "event": "created",
                        "timestamp": "2026-01-01T00:00:00Z",
                        "experiment_id": exp_id,
                        "status": "succeeded",
                        "parameters": {"NEXP": "1"},
                        "objective": "mlperf_submission_readiness",
                        "result_value": value,
                        "optimizer_state": {"strategy": "bayesian", "variant": variant},
                    },
                )
            rc = ai_tuning.main(
                [
                    "optimizer",
                    "history",
                    "--ledger",
                    str(ledger),
                    "--space",
                    str(REPO_ROOT / "tuning" / "tuning-space.b200-llama31-8b.json"),
                    "--objective",
                    "mlperf_submission_readiness",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            group = payload["objectives"][0]
            self.assertEqual(group["objective"], "mlperf_submission_readiness")
            self.assertEqual(group["direction"], "maximize")
            self.assertEqual(group["experiments"][0]["experiment_id"], "exp-b")
            self.assertEqual(group["experiments"][0]["optimizer_state"]["variant"], "tpe")

    def test_optimizer_compare_missing_result_has_no_winner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            output = Path(tmp) / "compare.json"
            ai_tuning.append_jsonl(
                ledger,
                {
                    "schema_version": 1,
                    "event": "created",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "experiment_id": "exp-a",
                    "status": "succeeded",
                    "parameters": {"NEXP": "1"},
                    "objective": "mlperf_submission_readiness",
                    "result_value": 0.9,
                },
            )
            ai_tuning.append_jsonl(
                ledger,
                {
                    "schema_version": 1,
                    "event": "created",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "experiment_id": "exp-b",
                    "status": "failed",
                    "parameters": {"NEXP": "1"},
                    "objective": "mlperf_submission_readiness",
                },
            )
            rc = ai_tuning.main(
                [
                    "optimizer",
                    "compare",
                    "exp-a",
                    "exp-b",
                    "--ledger",
                    str(ledger),
                    "--space",
                    str(REPO_ROOT / "tuning" / "tuning-space.b200-llama31-8b.json"),
                    "--objective",
                    "mlperf_submission_readiness",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertIsNone(payload["winner"])
            self.assertIsNone(payload["metric_deltas"]["result_value_delta"])

    def test_optimizer_import_hyp_writes_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            output = Path(tmp) / "import.json"
            rc = ai_tuning.main(
                [
                    "optimizer",
                    "import-hyp",
                    str(FIXTURES / "hyp" / "scaled_run.tp-comm-overlap-true.hyp"),
                    "--session-dir",
                    str(FIXTURES / "hyp-session"),
                    "--ledger",
                    str(ledger),
                    "--target-benchmark",
                    "imported_test",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["session"]["trial_count"], 3)
            self.assertEqual(payload["ledger_records_written"], 3)
            ledger_lines = [
                json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
            self.assertEqual(len(ledger_lines), 3)
            for record in ledger_lines:
                self.assertEqual(record["provenance"], "hyp_import")
                self.assertIn("result_value", record)

    def test_optimizer_import_hyp_markerless_write_space_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            template = Path(tmp) / "plain.hyp"
            template.write_text("#!/bin/bash\necho no markers\n", encoding="utf-8")
            output_space = Path(tmp) / "space.json"
            with self.assertRaises(SystemExit):
                ai_tuning.main(
                    [
                        "optimizer",
                        "import-hyp",
                        str(template),
                        "--write-space",
                        str(output_space),
                    ]
                )

    def test_optimizer_end_to_end_bayesian_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed_ledger = root / "seed-ledger.jsonl"
            proposal = root / "proposal.json"
            create_output = root / "create.json"
            created_ledger = root / "created-ledger.jsonl"
            submit_output = root / "submit.json"
            compare_ledger = root / "compare-ledger.jsonl"
            compare_output = root / "compare.json"

            for exp_id, tc, value in [
                ("seed-a", "96", 0.5),
                ("seed-b", "106", 0.8),
                ("seed-c", "0", 0.2),
                ("seed-d", "96", 0.7),
            ]:
                ai_tuning.append_jsonl(
                    seed_ledger,
                    {
                        "schema_version": 1,
                        "event": "created",
                        "timestamp": "2026-01-01T00:00:00Z",
                        "experiment_id": exp_id,
                        "status": "succeeded",
                        "parameters": {"NCCL_IB_TC": tc, "GATE_NODES": "8"},
                        "objective": "gb300_operational_readiness",
                        "result_value": value,
                    },
                )

            rc = ai_tuning.main(
                [
                    "optimizer",
                    "propose",
                    "--space",
                    str(REPO_ROOT / "tuning" / "tuning-space.gb300-ops.json"),
                    "--strategy",
                    "bayesian",
                    "--variant",
                    "tpe",
                    "--parameter",
                    "NCCL_IB_TC",
                    "--parameter",
                    "GATE_NODES",
                    "--ledger",
                    str(seed_ledger),
                    "--objective",
                    "gb300_operational_readiness",
                    "--limit",
                    "1",
                    "--seed",
                    "4",
                    "--output",
                    str(proposal),
                ]
            )
            self.assertEqual(rc, 0)
            proposed = json.loads(proposal.read_text(encoding="utf-8"))
            self.assertEqual(proposed["optimizer_state"]["variant"], "tpe")

            rc = ai_tuning.main(
                [
                    "experiment",
                    "create",
                    str(proposal),
                    "--space",
                    str(REPO_ROOT / "tuning" / "tuning-space.gb300-ops.json"),
                    "--ledger",
                    str(created_ledger),
                    "--output",
                    str(create_output),
                ]
            )
            self.assertEqual(rc, 0)
            created = json.loads(create_output.read_text(encoding="utf-8"))
            exp_id = created["experiments"][0]["experiment_id"]
            self.assertEqual(created["experiments"][0]["objective"], "gb300_operational_readiness")
            self.assertEqual(created["experiments"][0]["optimizer_state"]["variant"], "tpe")

            rc = ai_tuning.main(
                [
                    "experiment",
                    "submit",
                    "--ledger",
                    str(created_ledger),
                    "--script",
                    str(FIXTURES / "submit.sbatch"),
                    "--experiment-id",
                    exp_id,
                    "--max-concurrent",
                    "1",
                    "--output",
                    str(submit_output),
                ]
            )
            self.assertEqual(rc, 0)
            submit = json.loads(submit_output.read_text(encoding="utf-8"))
            self.assertEqual(submit["selected_count"], 1)
            self.assertFalse(submit["execute"])
            self.assertEqual(submit["submissions"][0]["objective"], "gb300_operational_readiness")

            ai_tuning.append_jsonl(
                compare_ledger,
                {
                    "schema_version": 1,
                    "event": "created",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "experiment_id": "compare-a",
                    "status": "succeeded",
                    "parameters": {"NCCL_IB_TC": "96"},
                    "objective": "gb300_operational_readiness",
                    "result_value": 0.6,
                },
            )
            ai_tuning.append_jsonl(
                compare_ledger,
                {
                    "schema_version": 1,
                    "event": "created",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "experiment_id": "compare-b",
                    "status": "succeeded",
                    "parameters": {"NCCL_IB_TC": "106"},
                    "objective": "gb300_operational_readiness",
                    "result_value": 0.9,
                },
            )
            rc = ai_tuning.main(
                [
                    "optimizer",
                    "compare",
                    "compare-a",
                    "compare-b",
                    "--ledger",
                    str(compare_ledger),
                    "--space",
                    str(REPO_ROOT / "tuning" / "tuning-space.gb300-ops.json"),
                    "--objective",
                    "gb300_operational_readiness",
                    "--output",
                    str(compare_output),
                ]
            )
            self.assertEqual(rc, 0)
            compared = json.loads(compare_output.read_text(encoding="utf-8"))
            self.assertEqual(compared["direction"], "maximize")
            self.assertEqual(compared["winner"], "compare-b")

if __name__ == "__main__":
    unittest.main()
