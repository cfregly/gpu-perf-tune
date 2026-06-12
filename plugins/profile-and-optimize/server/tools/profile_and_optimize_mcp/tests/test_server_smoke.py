from __future__ import annotations

import contextlib
import importlib.util
import io
import unittest
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from profile_and_optimize_mcp.server import RESOURCE_PATHS, SEARCH_TOOL_SPECS, _search, run_surface_tool, tool_names

REPO = Path(__file__).resolve().parents[3]


class ServerSmokeTest(unittest.TestCase):
    def test_server_can_be_created_when_mcp_is_installed(self) -> None:
        if importlib.util.find_spec("mcp") is None:
            self.skipTest("mcp package is not installed in this interpreter")
        from profile_and_optimize_mcp.server import create_server

        # Pull the canonical count from the single source of truth in
        # mcp_surface.py (61 MLPerf-specific + 34 profile-and-optimize-native across
        # perf_baseline / evidence / slurm / cohort_health /
        # cluster_health_survey / experiments / datasources / findings /
        # fleet_health_baseline / k8s_launch / shm_health / perf_tune_report,
        # plus 2 auxiliary search tools registered separately = 97 total).
        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        import mcp_surface

        server = create_server()
        self.assertIsNotNone(server)
        self.assertEqual(len(tool_names()), mcp_surface._TOTAL_CONTRACT_TOOLS)
        self.assertEqual(
            len(tool_names()) + len(SEARCH_TOOL_SPECS),
            mcp_surface._TOTAL_MCP_TOOLS,
        )
        self.assertGreaterEqual(len(RESOURCE_PATHS), 11)

    def test_canonical_counts_verify(self) -> None:
        """The canonical constants in mcp_surface.py must agree with the
        live derivation. This is the single source of truth that every
        doc / smoke-test / lint script reads."""
        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        import mcp_surface

        live = mcp_surface.verify_canonical_counts()
        self.assertEqual(live["libraries"], mcp_surface._TOTAL_LIBRARIES)
        self.assertEqual(live["contract_tools"], mcp_surface._TOTAL_CONTRACT_TOOLS)
        self.assertEqual(live["aux_tools"], mcp_surface._TOTAL_AUX_TOOLS)
        self.assertEqual(live["total_mcp_tools"], mcp_surface._TOTAL_MCP_TOOLS)

    def test_runtime_surface_is_contract_derived(self) -> None:
        names = set(tool_names())
        for name in (
            "launcher_launch",
            "launcher_dataset",
            "selector_pick",
            "selector_gate_256n",
            "selector_node_lookup",
            "validator_check",
            "validator_schema",
            "contention_snapshot",
            "contention_plan",
            "contention_o11y",
            "leaderboard_build",
            "preflight_best_cohort",
            "ai_tuning_optimizer",
            "ai_tuning_finalize",
            "ai_tuning_experiment",
            "submission_package",
            "submission_prelaunch_compliance",
            "submission_verify_numerics",
            "submission_failure_taxonomy",
            "submission_external_patch_issues",
            "submission_hotspare_pool",
            "submission_mflu_tail",
            "submission_mflu_replay",
            "campaign_validate",
            "campaign_advance",
            "campaign_five_run",
            "campaign_dashboard",
            "campaign_image_inventory",
            "profile_host_overhead",
            "profile_profile_diff",
            # Added in profile-and-optimize v1.13.0 (perf_tune_report library).
            "perf_tune_report_campaign_init",
            "perf_tune_report_cell_run",
            "perf_tune_report_atlas_aggregate",
            "perf_tune_report_report_render",
            "perf_tune_report_report_smoke",
            "perf_tune_report_capture_plan",
            "perf_tune_report_materialize_capture_reuse",
        ):
            with self.subTest(name=name):
                self.assertIn(name, names)
        self.assertEqual(SEARCH_TOOL_SPECS["search_runbooks"], ["runbooks", "docs"])
        self.assertEqual(SEARCH_TOOL_SPECS["search_evidence"], ["experiments/artifacts"])

    def test_runtime_invokes_read_only_tool_with_json_envelope(self) -> None:
        result = run_surface_tool("validator_schema", {"args": ["summary"]})
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["safety"], "read_only")
        self.assertFalse(result["ack_required"])
        self.assertEqual(result["json"]["name"], "summary")

    def test_runtime_forwards_ack_param_as_cli_flag(self) -> None:
        result = run_surface_tool(
            "launcher_launch",
            {
                "args": ["--bench", "llama31_8b", "--nodes", "8", "--dry-run"],
                "i_understand_this_submits_jobs": True,
            },
        )
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["safety"], "submits_jobs")
        self.assertEqual(result["ack_field"], "i_understand_this_submits_jobs")
        self.assertIn("--i-understand-this-submits-jobs", result["args"])
        self.assertTrue(result["json"]["acknowledged"])

    def test_console_entrypoint_list_matches_mcp_surface(self) -> None:
        import mcp_surface

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            rc = mcp_surface.main(["--json", "list"])
        self.assertEqual(rc, 0)
        self.assertEqual(set(tool_names()), {tool["name"] for tool in __import__("json").loads(buffer.getvalue())["tools"]})

    def test_search_tools_return_standard_envelope(self) -> None:
        result = _search("search_runbooks", ["runbooks", "docs"], "MLPerf", limit=3)
        self.assertEqual(result["tool"], "search_runbooks")
        self.assertEqual(result["library"], "mcp_aux")
        self.assertEqual(result["verb"], "search")
        self.assertEqual(result["safety"], "read_only")
        self.assertFalse(result["ack_required"])
        self.assertIsNone(result["ack_field"])
        self.assertIn("MLPerf", result["args"])
        self.assertIn(result["returncode"], (0, 1))
        self.assertIsInstance(result["stdout"], str)
        self.assertIsInstance(result["stderr"], str)
        payload = result["json"]
        self.assertEqual(payload["query"], "MLPerf")
        self.assertEqual(payload["paths"], ["runbooks", "docs"])
        self.assertIsInstance(payload["matches"], list)

    def test_runtime_traps_systemexit_from_argparse_help(self) -> None:
        """Regression test: argparse `--help` raises SystemExit which would
        otherwise propagate through FastMCP's stdio JSON-RPC loop and
        terminate the entire server process. The runtime must catch it,
        normalize to the standard envelope, and let the caller decide via
        `allow_nonzero`."""
        result = run_surface_tool(
            "validator_schema",
            {"args": ["--help"], "allow_nonzero": True},
        )
        self.assertEqual(result["tool"], "validator_schema")
        self.assertEqual(result["safety"], "read_only")
        # argparse `--help` exits with code 0; the envelope must surface
        # that without raising.
        self.assertEqual(result["returncode"], 0)
        # The help text lands in stdout (argparse prints --help to stdout).
        self.assertIn("--help", result["stdout"] + result["stderr"])

    def test_runtime_traps_systemexit_from_unknown_verb(self) -> None:
        """Regression test: an unknown verb triggers argparse's error path
        (SystemExit(2) with a message on stderr). The runtime must catch
        it instead of crashing the server."""
        result = run_surface_tool(
            "validator_schema",
            {"args": ["this-verb-does-not-exist"], "allow_nonzero": True},
        )
        self.assertEqual(result["tool"], "validator_schema")
        self.assertEqual(result["returncode"], 2)
        self.assertTrue(result["stderr"], "argparse error message expected on stderr")

    def test_every_contract_leaf_parser_accepts_json(self) -> None:
        """Regression test: the runtime auto-appends `--json` to argv
        whenever `CONTRACT[verb]["json"]` is True. Every contract-derived
        leaf parser must accept that `--json` token, otherwise argparse
        rejects the call with `unrecognized arguments: --json` at runtime
        and the MCP envelope reports `returncode=2`. This bug was
        discovered on the live MCP for every `ai_tuning_*` tool after
        Phase 2 wiring; keep this test green so we never silently
        regress.

        The check is parser-introspection-only -- it does NOT execute the
        cmd functions (which often shell out to Slurm and fail off-cluster
        with FileNotFoundError). For umbrella verbs whose contract entry
        is `required: ("subverb",)`, every nested subparser is checked
        because the runtime's `--json` lands at the leaf of the subparser
        chain, not on the umbrella.
        """
        import argparse as _ap
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        import mcp_surface

        def collect_leaves(p: _ap.ArgumentParser) -> list[_ap.ArgumentParser]:
            for action in p._actions:  # noqa: SLF001
                if isinstance(action, _ap._SubParsersAction):  # noqa: SLF001
                    leaves: list[_ap.ArgumentParser] = []
                    for child in action.choices.values():
                        leaves.extend(collect_leaves(child))
                    return leaves
            return [p]

        for spec in mcp_surface.derive_tool_specs():
            if not spec.json:
                continue  # Runtime only auto-appends --json when spec.json is True.
            cli = mcp_surface._load_cli_module(spec.library)  # noqa: SLF001
            parser = cli.build_parser()
            verb_parser: _ap.ArgumentParser | None = None
            for action in parser._actions:  # noqa: SLF001
                if isinstance(action, _ap._SubParsersAction):  # noqa: SLF001
                    verb_parser = action.choices.get(spec.verb)
                    break
            self.assertIsNotNone(verb_parser, msg=f"{spec.name}: top-level subparser missing")
            for leaf in collect_leaves(verb_parser):
                option_strings = {
                    opt for action in leaf._actions for opt in action.option_strings  # noqa: SLF001
                }
                with self.subTest(name=spec.name, leaf=leaf.prog):
                    self.assertIn(
                        "--json",
                        option_strings,
                        msg=(
                            f"{spec.name} leaf {leaf.prog!r}: parser does not accept --json. "
                            "Add --json (no-op or real) to this subparser, otherwise the MCP "
                            "runtime will reject the call with `unrecognized arguments: --json`."
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
