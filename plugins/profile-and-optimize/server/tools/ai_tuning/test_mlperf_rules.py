from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mlperf_rules import (  # noqa: E402
    DEFAULT_RULES_PATH,
    load_rules,
    validate_candidate,
)
from mlperf_rules import (
    main as rules_main,
)


class TestValidateCandidate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rules = load_rules(DEFAULT_RULES_PATH)

    def test_lr_above_max_blocks_405b(self) -> None:
        violations = validate_candidate(
            self.rules,
            benchmark="llama31_405b",
            parameters={"learning_rate": "0.001"},
        )
        codes = [v.code for v in violations]
        self.assertIn("lr_above_max", codes)

    def test_global_batch_too_large_blocks_405b(self) -> None:
        violations = validate_candidate(
            self.rules,
            benchmark="llama31_405b",
            parameters={"global_batch_size": "20000"},
        )
        codes = [v.code for v in violations]
        self.assertIn("global_batch_outside_legal", codes)

    def test_synthetic_data_in_submission_blocks(self) -> None:
        violations = validate_candidate(
            self.rules,
            benchmark="llama31_405b",
            parameters={},
            nexp=5,
            use_synthetic_data=True,
        )
        codes = [v.code for v in violations]
        self.assertIn("synthetic_data_in_submission", codes)

    def test_warmup_above_proportion_blocks(self) -> None:
        violations = validate_candidate(
            self.rules,
            benchmark="llama31_8b",
            parameters={"warmup_steps": "200", "TRAINER_TRAIN_STEPS": "500"},
        )
        codes = [v.code for v in violations]
        # 200 / 500 = 0.4 > 0.1 limit
        self.assertIn("warmup_above_proportion", codes)

    def test_expert_parallel_illegal_blocks(self) -> None:
        violations = validate_candidate(
            self.rules,
            benchmark="deepseekv3_671b",
            parameters={"EXPERT_PARALLEL": "16"},
        )
        codes = [v.code for v in violations]
        self.assertIn("expert_parallel_illegal", codes)

    def test_aux_loss_coef_outside_blocks_dsv3(self) -> None:
        violations = validate_candidate(
            self.rules,
            benchmark="deepseekv3_671b",
            parameters={"AUX_LOSS_BALANCE_COEF": "10.0"},
        )
        codes = [v.code for v in violations]
        self.assertIn("aux_loss_coef_outside_legal", codes)

    def test_unknown_benchmark_blocks(self) -> None:
        violations = validate_candidate(
            self.rules,
            benchmark="not_a_benchmark",
            parameters={},
        )
        codes = [v.code for v in violations]
        self.assertEqual(codes, ["unknown_benchmark"])

    def test_clean_candidate_passes(self) -> None:
        violations = validate_candidate(
            self.rules,
            benchmark="llama31_405b",
            parameters={
                "learning_rate": "0.0001",
                "global_batch_size": "4096",
                "warmup_steps": "30",
                "TRAINER_TRAIN_STEPS": "500",
                "MLPERF_RULESET": "6.0.0",
            },
        )
        self.assertEqual(violations, [])

    def test_ruleset_mismatch_blocks(self) -> None:
        violations = validate_candidate(
            self.rules,
            benchmark="llama31_405b",
            parameters={"MLPERF_RULESET": "5.1.0"},
        )
        codes = [v.code for v in violations]
        self.assertIn("ruleset_mismatch", codes)


class TestCli(unittest.TestCase):
    def test_validate_proposal_returns_nonzero_on_violation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            proposal = tdp / "proposal.json"
            out = tdp / "out.json"
            proposal.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {"parameters": {"learning_rate": "0.0001"}},
                            {"parameters": {"learning_rate": "1.0"}},
                        ]
                    }
                )
            )
            rc = rules_main(
                [
                    "validate",
                    "--proposal",
                    str(proposal),
                    "--benchmark",
                    "llama31_405b",
                    "--out",
                    str(out),
                ]
            )
            self.assertEqual(rc, 1)
            payload = json.loads(out.read_text())
            self.assertEqual(payload["valid_count"], 1)
            self.assertEqual(payload["invalid_count"], 1)


if __name__ == "__main__":
    unittest.main()
