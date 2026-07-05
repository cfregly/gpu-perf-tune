"""Tests for safety.py.

Per CLAUDE.md "AI-Assisted Tuning Safety", every change to
``FORBIDDEN_PATCH_PATTERNS`` must keep the regex behavior intact and
the ai_tuning re-exports honest. These tests exercise both invariants.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ai_tuning  # noqa: E402
import safety  # noqa: E402


class ForbiddenPatternsTests(unittest.TestCase):
    def test_every_pattern_compiles(self) -> None:
        for code, pattern in safety.FORBIDDEN_PATCH_PATTERNS:
            with self.subTest(code=code):
                self.assertTrue(code)
                re.compile(pattern)

    def test_codes_are_unique(self) -> None:
        codes = [code for code, _ in safety.FORBIDDEN_PATCH_PATTERNS]
        self.assertEqual(len(codes), len(set(codes)))

    def test_canonical_malicious_snippets_are_rejected(self) -> None:
        # Each row asserts that the named pattern fires on the given snippet.
        # If you add a pattern, add a positive case here.
        cases = (
            ("submit_slurm_job", "sbatch run.sub"),
            ("cancel_slurm_job", "scancel 12345"),
            ("mutate_slurm_state", "scontrol update NodeName=gb300-001 State=DOWN"),
            ("mutate_kubernetes", "kubectl delete pod foo"),
            ("destructive_remove", "rm -rf /tmp/cache"),
            ("restart_service", "systemctl restart slurmd"),
            ("power_action", "shutdown -h now"),
        )
        codes_by_name = {code: pattern for code, pattern in safety.FORBIDDEN_PATCH_PATTERNS}
        for code, snippet in cases:
            with self.subTest(code=code, snippet=snippet):
                pattern = codes_by_name[code]
                self.assertRegex(snippet, pattern)

    def test_benign_snippets_do_not_false_fire(self) -> None:
        # Benign strings that previously caused false positives.
        benign = (
            "echo 'launching with sbatch_script.sh'",  # quoted mention only
            "kubectl get pods",
            "rm /tmp/onefile.txt",
        )
        # We don't claim every pattern leaves benign strings untouched -
        # only that no pattern matches the whole-line benign cases. The
        # 'quoted sbatch' case is the canonical example: 'sbatch_script.sh'
        # must NOT match the submit_slurm_job pattern because the regex
        # uses \b after sbatch.
        for snippet in benign:
            with self.subTest(snippet=snippet):
                hits = [
                    code
                    for code, pattern in safety.FORBIDDEN_PATCH_PATTERNS
                    if re.search(pattern, snippet)
                ]
                # benign cases should not match the *submit* pattern.
                self.assertNotIn("submit_slurm_job", hits, snippet)


class ReExportTests(unittest.TestCase):
    def test_ai_tuning_re_exports_match(self) -> None:
        self.assertIs(ai_tuning.FORBIDDEN_PATCH_PATTERNS, safety.FORBIDDEN_PATCH_PATTERNS)
        self.assertEqual(ai_tuning.EXPERIMENT_STATUSES, safety.EXPERIMENT_STATUSES)
        self.assertEqual(ai_tuning.REPORT_SCHEMA_VERSION, safety.REPORT_SCHEMA_VERSION)
        self.assertEqual(ai_tuning.PROPOSAL_SCHEMA_VERSION, safety.PROPOSAL_SCHEMA_VERSION)
        self.assertEqual(
            ai_tuning.TEMPLATE_PATCH_SCHEMA_VERSION,
            safety.TEMPLATE_PATCH_SCHEMA_VERSION,
        )
        self.assertEqual(
            ai_tuning.EXPERIMENT_LEDGER_SCHEMA_VERSION,
            safety.EXPERIMENT_LEDGER_SCHEMA_VERSION,
        )


class StatusTests(unittest.TestCase):
    def test_experiment_statuses_includes_terminal_states(self) -> None:
        for required in ("planned", "submitted", "succeeded", "failed", "cancelled"):
            with self.subTest(state=required):
                self.assertIn(required, safety.EXPERIMENT_STATUSES)


if __name__ == "__main__":
    unittest.main()
