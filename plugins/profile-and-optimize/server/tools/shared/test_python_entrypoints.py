from __future__ import annotations

import unittest

from tools.shared import gb300_mlperf
from tools.shared.slurm_racks import racks


class PythonEntrypointTest(unittest.TestCase):
    def test_gb300_mlperf_parser_keeps_passthrough_args(self) -> None:
        args = gb300_mlperf.build_parser().parse_args(["print-cluster-state", "--json"])
        self.assertEqual(args.args, ["print-cluster-state", "--json"])

    def test_racks_parser_keeps_passthrough_args(self) -> None:
        args = racks.build_parser().parse_args(["rack-summary", "--json"])
        self.assertEqual(args.args, ["rack-summary", "--json"])


if __name__ == "__main__":
    unittest.main()
