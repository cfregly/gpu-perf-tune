from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from tools.ai_tuning import ai_tuning, cli, proposals, safety

IO_SPEC = importlib.util.spec_from_file_location("ai_tuning_io", HERE / "io.py")
assert IO_SPEC is not None and IO_SPEC.loader is not None
tuner_io = importlib.util.module_from_spec(IO_SPEC)
IO_SPEC.loader.exec_module(tuner_io)


class AiTuningModuleSplitTest(unittest.TestCase):
    def test_wrapper_and_split_modules_export_core_entrypoints(self) -> None:
        self.assertIs(ai_tuning.main, cli.main)
        self.assertTrue(callable(cli.build_parser))
        self.assertTrue(callable(tuner_io.write_json))
        self.assertTrue(callable(proposals.command_proposal_validate))
        self.assertTrue(safety.FORBIDDEN_PATCH_PATTERNS)

    def test_proposals_direct_import_from_ai_tuning_path(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(HERE)
        with tempfile.TemporaryDirectory() as workdir:
            proc = subprocess.run(
                [sys.executable, "-S", "-c", "import proposals"],
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)


if __name__ == "__main__":
    unittest.main()
