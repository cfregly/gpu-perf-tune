from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import artifact_bundle


class ArtifactBundleShapeTest(unittest.TestCase):
    def _write_bundle(self, root: Path) -> None:
        root.mkdir()
        (root / "summary.md").write_text("# Summary\n\nEvidence summary.\n", encoding="utf-8")
        (root / "SOURCE.md").write_text("# Source\n\nCaptured by test.\n", encoding="utf-8")
        (root / "run-context.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "family": "campaign/llama31_405b",
                    "run_id": "run-1",
                    "created_at_utc": "2026-05-07T00:00:00Z",
                }
            ),
            encoding="utf-8",
        )

    def test_valid_bundle_passes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bundle = Path(td) / "run-1"
            self._write_bundle(bundle)
            self.assertEqual(
                artifact_bundle.validate_bundle_shape(
                    bundle,
                    expected_family="campaign/llama31_405b",
                    expected_run_id="run-1",
                ),
                [],
            )

    def test_missing_run_context_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bundle = Path(td) / "run-1"
            self._write_bundle(bundle)
            (bundle / "run-context.json").unlink()
            issues = artifact_bundle.validate_bundle_shape(bundle)
            self.assertTrue(any("run-context.json" in issue.render() for issue in issues), issues)

    def test_context_family_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bundle = Path(td) / "run-1"
            self._write_bundle(bundle)
            issues = artifact_bundle.validate_bundle_shape(
                bundle,
                expected_family="campaign/deepseekv3_671b",
            )
            self.assertTrue(any("does not match expected" in issue.message for issue in issues), issues)


if __name__ == "__main__":
    unittest.main()
