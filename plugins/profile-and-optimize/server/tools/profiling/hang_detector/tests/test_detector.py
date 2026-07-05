"""Unit tests for the hang-detector orchestrator.

Exercises the offline (fixture-driven) path end-to-end, including
JSONL output, the schema-version validation in the scraper, and the
``--auto-stride`` CLI sweep that runs the detector across a small
set of candidate strides at once.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from tools.profiling.hang_detector.cli import main as cli_main
from tools.profiling.hang_detector.detector import run_detector

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


class RunDetectorTest(unittest.TestCase):
    def test_mod32_hang_fixture_produces_alert(self) -> None:
        result = run_detector(
            fixture_path=FIXTURE_DIR / "gpusd-snapshot-2048n-mod32-hang.json",
            stride=32,
            lag_threshold=1,
            jobid="11523",
        )
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["jobid"], "11523")
        self.assertEqual(result["stride"], 32)
        self.assertEqual(result["lag_threshold"], 1)
        self.assertGreater(result["rank_count"], 0)
        self.assertEqual(len(result["alerts"]), 1)
        alert = result["alerts"][0]
        self.assertEqual(alert["lagging_bucket"], 31)
        self.assertEqual(alert["lag"], 50)
        self.assertEqual(alert["ranks"], [31, 63, 95, 127])

    def test_healthy_fixture_produces_no_alerts(self) -> None:
        result = run_detector(
            fixture_path=FIXTURE_DIR / "gpusd-snapshot-healthy.json",
            stride=32,
            jobid="healthy-baseline",
        )
        self.assertEqual(result["alerts"], [])

    def test_output_path_appends_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = Path(tmpdir) / "timeline.jsonl"
            run_detector(
                fixture_path=FIXTURE_DIR / "gpusd-snapshot-2048n-mod32-hang.json",
                stride=32,
                output_path=outpath,
                jobid="11523",
            )
            run_detector(
                fixture_path=FIXTURE_DIR / "gpusd-snapshot-healthy.json",
                stride=32,
                output_path=outpath,
                jobid="healthy-baseline",
            )
            lines = outpath.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 2)
            parsed = [json.loads(line) for line in lines]
            self.assertEqual(parsed[0]["jobid"], "11523")
            self.assertEqual(parsed[1]["jobid"], "healthy-baseline")
            self.assertEqual(len(parsed[1]["alerts"]), 0)

    def test_missing_output_parent_fails_fast(self) -> None:
        """Per CLAUDE.md 'Fail Fast', refuse to write into a non-existent
        directory rather than auto-mkdir-ing it."""
        outpath = Path("/tmp/this/does/not/exist/timeline.jsonl")
        with self.assertRaises(FileNotFoundError):
            run_detector(
                fixture_path=FIXTURE_DIR / "gpusd-snapshot-healthy.json",
                stride=32,
                output_path=outpath,
            )

    def test_missing_fixture_path_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            run_detector(fixture_path=Path("/tmp/nonexistent-fixture.json"))

    def test_live_cluster_without_nodelist_raises(self) -> None:
        with self.assertRaises(ValueError):
            run_detector(live_cluster=True, nodelist=None)


class AutoStrideCliTest(unittest.TestCase):
    """Exercises the ``--auto-stride`` CLI sweep against the synthetic
    fixtures. The sweep is a pure CLI-layer feature; ``detector.py``,
    ``stride_detector.py``, and ``gpusd_scraper.py`` are unchanged."""

    @staticmethod
    def _run_cli(argv: list[str]) -> tuple[int, dict]:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            rc = cli_main(argv)
        return int(rc), json.loads(buffer.getvalue())

    def test_auto_stride_on_mod32_hang_isolates_stride32_bucket(self) -> None:
        """The 2048N MOD-32 hang fixture only triggers an alert at
        stride=32 (every 32nd rank stagnates). At strides 8, 16, 64
        the lagging ranks are evenly spread across buckets and no
        single bucket lags far enough to trigger an alert at the
        default lag_threshold=1.
        """
        rc, payload = self._run_cli(
            [
                "--fixture",
                str(FIXTURE_DIR / "gpusd-snapshot-2048n-mod32-hang.json"),
                "--auto-stride",
                "--candidate-strides",
                "8,16,32,64",
                "--lag-threshold",
                "1",
                "--json",
            ]
        )
        self.assertEqual(rc, 1, "auto-stride must exit non-zero when any alert fires")
        self.assertEqual(payload["mode"], "auto_stride")
        self.assertEqual(payload["strides_checked"], [8, 16, 32, 64])
        self.assertEqual(len(payload["per_stride"]), 4)
        # Aggregated alerts MUST include the stride=32 bucket=31 entry.
        flagged_strides = {a["stride"] for a in payload["alerts"]}
        self.assertIn(32, flagged_strides)
        mod32_alerts = [a for a in payload["alerts"] if a["stride"] == 32]
        self.assertEqual(len(mod32_alerts), 1)
        self.assertEqual(mod32_alerts[0]["lagging_bucket"], 31)
        self.assertEqual(mod32_alerts[0]["ranks"], [31, 63, 95, 127])

    def test_auto_stride_on_healthy_fixture_emits_no_alerts(self) -> None:
        """Sweeping the healthy fixture across [4, 8] candidate strides
        produces zero alerts and exit code 0."""
        rc, payload = self._run_cli(
            [
                "--fixture",
                str(FIXTURE_DIR / "gpusd-snapshot-healthy.json"),
                "--auto-stride",
                "--candidate-strides",
                "4,8",
                "--json",
            ]
        )
        self.assertEqual(rc, 0)
        self.assertEqual(payload["mode"], "auto_stride")
        self.assertEqual(payload["strides_checked"], [4, 8])
        self.assertEqual(payload["alerts"], [])
        self.assertEqual(len(payload["per_stride"]), 2)
        for single in payload["per_stride"]:
            self.assertEqual(single["alerts"], [])


if __name__ == "__main__":
    unittest.main()
