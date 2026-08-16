from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from profile_and_optimize_mcp.server import (
    RESOURCE_PATHS,
    SEARCH_TOOL_SPECS,
    _load_mcp_surface,
    _search,
    _search_command,
    run_surface_tool,
    tool_names,
)

REPO = Path(__file__).resolve().parents[3]


class ServerSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._previous_repo_root = os.environ.get("PROFILE_AND_OPTIMIZE_REPO_ROOT")
        os.environ["PROFILE_AND_OPTIMIZE_REPO_ROOT"] = str(REPO)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._previous_repo_root is None:
            os.environ.pop("PROFILE_AND_OPTIMIZE_REPO_ROOT", None)
        else:
            os.environ["PROFILE_AND_OPTIMIZE_REPO_ROOT"] = cls._previous_repo_root
        super().tearDownClass()

    def test_server_can_be_created_when_mcp_is_installed(self) -> None:
        if importlib.util.find_spec("mcp") is None:
            self.skipTest("mcp package is not installed in this interpreter")
        from profile_and_optimize_mcp.server import create_server

        # Pull the canonical counts from mcp_surface.py. The public surface has
        # 51 contract tools across 8 libraries plus 2 auxiliary search tools.
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
        self.assertEqual(len(RESOURCE_PATHS), 6)

    def test_every_registered_resource_exists_and_is_readable(self) -> None:
        for uri, relative_path in RESOURCE_PATHS.items():
            with self.subTest(uri=uri):
                path = REPO / relative_path
                self.assertTrue(path.is_file(), f"missing MCP resource: {path}")
                self.assertTrue(path.read_text(encoding="utf-8").strip())

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
        # Keep one representative from every canonical library here. Exact
        # counts and full derivation equality are checked separately above.
        for name in (
            "ai_tuning_optimizer",
            "profile_host_overhead",
            "perf_baseline_diff",
            "evidence_init",
            "slurm_triage",
            "findings_diff",
            "perf_tune_report_report_smoke",
            "known_good_config_check",
        ):
            with self.subTest(name=name):
                self.assertIn(name, names)
        self.assertEqual(SEARCH_TOOL_SPECS["search_runbooks"], ["runbooks", "docs"])
        self.assertEqual(SEARCH_TOOL_SPECS["search_evidence"], ["experiments/artifacts"])

    def test_runtime_invokes_read_only_tool_with_json_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as logdir:
            result = run_surface_tool(
                "slurm_triage",
                {"args": ["--jobid", "0", "--logdir", logdir]},
            )
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["safety"], "read_only")
        self.assertFalse(result["ack_required"])
        self.assertEqual(result["json"]["tool"], "slurm_triage")
        self.assertEqual(result["json"]["jobid"], "0")

    def test_runtime_forwards_ack_param_as_cli_flag(self) -> None:
        from tools.slurm import slurm_cli

        original_run = slurm_cli.RUN
        slurm_cli.RUN = lambda argv, **kwargs: subprocess.CompletedProcess(
            args=list(argv),
            returncode=0,
            stdout="",
            stderr="",
        )
        try:
            with tempfile.TemporaryDirectory() as bundle:
                result = run_surface_tool(
                    "slurm_drain",
                    {
                        "args": [
                            "--nodes", "test-node",
                            "--ns", "test-namespace",
                            "--bundle", bundle,
                        ],
                        "i_understand_this_substitutes_nodes": True,
                    },
                )
        finally:
            slurm_cli.RUN = original_run
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["safety"], "substitutes_nodes")
        self.assertEqual(result["ack_field"], "i_understand_this_substitutes_nodes")
        self.assertIn("--i-understand-this-substitutes-nodes", result["args"])
        self.assertEqual(result["json"]["tool"], "slurm_drain")
        self.assertEqual(result["json"]["scontrol_exit"], 0)

    def test_runtime_rejects_ack_flag_in_raw_args(self) -> None:
        with self.assertRaisesRegex(ValueError, "params.args"):
            run_surface_tool(
                "slurm_drain",
                {
                    "args": [
                        "--nodes",
                        "test-node",
                        "--bundle",
                        "/tmp/not-used",
                        "--i-understand-this-substitutes-nodes",
                    ]
                },
            )

    def test_runtime_requires_literal_true_for_acknowledgement(self) -> None:
        for value in (False, "false", "true", 1, 0, None):
            with self.subTest(value=value):
                with self.assertRaises(PermissionError):
                    run_surface_tool(
                        "slurm_drain",
                        {
                            "args": [
                                "--nodes",
                                "test-node",
                                "--bundle",
                                "/tmp/not-used",
                            ],
                            "i_understand_this_substitutes_nodes": value,
                        },
                    )

    def test_dynamic_submit_rejects_raw_ack_flag(self) -> None:
        with self.assertRaisesRegex(ValueError, "params.args"):
            run_surface_tool(
                "ai_tuning_experiment",
                {
                    "args": [
                        "submit",
                        "--ledger",
                        "/tmp/not-used.jsonl",
                        "--script",
                        "/tmp/not-used.sbatch",
                        "--execute",
                        "--i-understand-this-submits-jobs",
                    ]
                },
            )

    def test_dynamic_submit_rejects_abbreviated_execute(self) -> None:
        with self.assertRaisesRegex(ValueError, "abbreviated option"):
            run_surface_tool(
                "ai_tuning_experiment",
                {
                    "args": [
                        "submit",
                        "--ledger",
                        "/tmp/not-used.jsonl",
                        "--script",
                        "/tmp/not-used.sbatch",
                        "--exec",
                    ]
                },
            )

    def test_dynamic_submit_rejects_abbreviated_raw_ack(self) -> None:
        with self.assertRaisesRegex(ValueError, "params.args"):
            run_surface_tool(
                "ai_tuning_experiment",
                {
                    "args": [
                        "submit",
                        "--ledger",
                        "/tmp/not-used.jsonl",
                        "--script",
                        "/tmp/not-used.sbatch",
                        "--execute",
                        "--i-understand-this-submits-j",
                    ]
                },
            )

    def test_dynamic_submit_execute_requires_literal_true(self) -> None:
        for value in (False, "false", "true", 1, 0, None):
            with self.subTest(value=value):
                with self.assertRaises(PermissionError):
                    run_surface_tool(
                        "ai_tuning_experiment",
                        {
                            "args": [
                                "submit",
                                "--ledger",
                                "/tmp/not-used.jsonl",
                                "--script",
                                "/tmp/not-used.sbatch",
                                "--execute",
                            ],
                            "i_understand_this_submits_jobs": value,
                        },
                    )

    def test_dynamic_submit_execute_forwards_structured_ack(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            ledger = root / "ledger.jsonl"
            script = root / "submit.sbatch"
            ledger.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "event": "created",
                        "timestamp": "2026-01-01T00:00:00Z",
                        "experiment_id": "exp-test",
                        "status": "planned",
                        "priority": 1,
                        "parameters": {},
                        "shape": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            completed = subprocess.CompletedProcess(
                args=["sbatch"],
                returncode=0,
                stdout="12345;cluster\n",
                stderr="",
            )

            with mock.patch(
                "tools.ai_tuning.cmd_experiment.subprocess.run",
                return_value=completed,
            ) as run:
                result = run_surface_tool(
                    "ai_tuning_experiment",
                    {
                        "args": [
                            "submit",
                            "--ledger",
                            str(ledger),
                            "--script",
                            str(script),
                            "--execute",
                        ],
                        "i_understand_this_submits_jobs": True,
                    },
                )

        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["safety"], "submits_jobs")
        self.assertTrue(result["ack_required"])
        self.assertEqual(result["ack_field"], "i_understand_this_submits_jobs")
        self.assertIn("--i-understand-this-submits-jobs", result["args"])
        run.assert_called_once()

    def test_external_publish_requires_literal_true_acknowledgement(self) -> None:
        for value in (False, "false", "true", 1, 0, None):
            with self.subTest(value=value), self.assertRaises(PermissionError):
                run_surface_tool(
                    "perf_tune_report_publish_to_lake",
                    {
                        "args": ["--campaign", "/tmp/not-used"],
                        "i_understand_this_publishes_externally": value,
                    },
                )

    def test_external_publish_forwards_structured_acknowledgement(self) -> None:
        surface = _load_mcp_surface()
        module = surface._load_cli_module("perf_tune_report")
        received: list[str] = []

        def fake_main(argv: list[str]) -> int:
            received.extend(argv)
            print(json.dumps({"dry_run": False}))
            return 0

        with mock.patch.object(module, "main", side_effect=fake_main):
            result = run_surface_tool(
                "perf_tune_report_publish_to_lake",
                {
                    "args": ["--campaign", "/tmp/not-used"],
                    "i_understand_this_publishes_externally": True,
                },
            )

        self.assertEqual(result["safety"], "publishes_external")
        self.assertTrue(result["ack_required"])
        self.assertEqual(
            result["ack_field"],
            "i_understand_this_publishes_externally",
        )
        self.assertIn("--i-understand-this-publishes-externally", received)

    def test_external_publish_dry_run_needs_no_acknowledgement(self) -> None:
        surface = _load_mcp_surface()
        module = surface._load_cli_module("perf_tune_report")
        received: list[str] = []

        def fake_main(argv: list[str]) -> int:
            received.extend(argv)
            print(json.dumps({"dry_run": True}))
            return 0

        with mock.patch.object(module, "main", side_effect=fake_main):
            result = run_surface_tool(
                "perf_tune_report_publish_to_lake",
                {"args": ["--campaign", "/tmp/not-used", "--dry-run"]},
            )

        self.assertEqual(result["safety"], "publishes_external")
        self.assertFalse(result["ack_required"])
        self.assertNotIn("--i-understand-this-publishes-externally", received)

    def test_cluster_mutation_dry_runs_need_no_acknowledgement(self) -> None:
        surface = _load_mcp_surface()
        module = surface._load_cli_module("perf_tune_report")

        for tool_name, args, ack_flag in (
            (
                "perf_tune_report_cell_run",
                [
                    "--campaign", "/tmp/not-used",
                    "--cell", "cell1",
                    "--backend", "vllm-sweep",
                    "--dry-run",
                ],
                "--i-understand-this-submits-jobs",
            ),
            (
                "perf_tune_report_campaign_run",
                [
                    "--config", "/tmp/not-used.yaml",
                    "--campaign", "/tmp/not-used",
                    "--dry-run",
                ],
                "--i-understand-this-mutates-cluster",
            ),
        ):
            with self.subTest(tool_name=tool_name):
                received: list[str] = []

                def fake_main(argv: list[str], sink: list[str] = received) -> int:
                    sink.extend(argv)
                    print(json.dumps({"dry_run": True}))
                    return 0

                with mock.patch.object(module, "main", side_effect=fake_main):
                    result = run_surface_tool(tool_name, {"args": args})

                self.assertFalse(result["ack_required"])
                self.assertNotIn(ack_flag, received)

    def test_console_entrypoint_list_matches_mcp_surface(self) -> None:
        import mcp_surface

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            rc = mcp_surface.main(["--json", "list"])
        self.assertEqual(rc, 0)
        self.assertEqual(set(tool_names()), {tool["name"] for tool in __import__("json").loads(buffer.getvalue())["tools"]})

    def test_search_tools_return_standard_envelope(self) -> None:
        result = _search("search_runbooks", ["runbooks", "docs"], "performance", limit=3)
        self.assertEqual(result["tool"], "search_runbooks")
        self.assertEqual(result["library"], "mcp_aux")
        self.assertEqual(result["verb"], "search")
        self.assertEqual(result["safety"], "read_only")
        self.assertFalse(result["ack_required"])
        self.assertIsNone(result["ack_field"])
        self.assertIn("performance", result["args"])
        self.assertIn(result["returncode"], (0, 1))
        self.assertIsInstance(result["stdout"], str)
        self.assertIsInstance(result["stderr"], str)
        payload = result["json"]
        self.assertEqual(payload["query"], "performance")
        self.assertEqual(payload["paths"], ["runbooks", "docs"])
        self.assertIsInstance(payload["matches"], list)

    def test_search_places_hostile_query_after_option_terminator(self) -> None:
        query = "--pre=/bin/sh"
        command = _search_command(query, ["runbooks", "docs"], 3)
        terminator = command.index("--")
        self.assertEqual(command[terminator + 1], query)

    def test_search_rejects_invalid_limits(self) -> None:
        for limit in (True, 0, -1, 101, 1.5):
            with self.subTest(limit=limit):
                with self.assertRaises((TypeError, ValueError)):
                    _search("search_runbooks", ["runbooks"], "profile", limit=limit)

    def test_search_caps_results_globally(self) -> None:
        process = mock.Mock()
        process.stdout = io.StringIO("one\ntwo\nthree\n")
        process.poll.return_value = None
        process.wait.return_value = -15

        with mock.patch("profile_and_optimize_mcp.server.subprocess.Popen", return_value=process):
            result = _search("search_runbooks", ["runbooks"], "profile", limit=2)

        self.assertEqual(result["json"]["matches"], ["one", "two"])
        self.assertEqual(result["returncode"], 0)
        process.terminate.assert_called_once_with()

    def test_performance_hints_are_exposed_and_searchable(self) -> None:
        uri = "perftune://repo/docs/performance-hints.md"
        self.assertEqual(RESOURCE_PATHS[uri], "docs/performance-hints.md")
        result = _search(
            "search_runbooks",
            SEARCH_TOOL_SPECS["search_runbooks"],
            "An estimate is a ranking tool",
            limit=5,
        )
        self.assertEqual(result["returncode"], 0)
        self.assertTrue(
            any("docs/performance-hints.md" in match for match in result["json"]["matches"])
        )

    def test_runtime_traps_systemexit_from_argparse_help(self) -> None:
        """Regression test: argparse `--help` raises SystemExit which would
        otherwise propagate through FastMCP's stdio JSON-RPC loop and
        terminate the entire server process. The runtime must catch it,
        normalize to the standard envelope, and let the caller decide via
        `allow_nonzero`."""
        result = run_surface_tool(
            "slurm_triage",
            {"args": ["--help"], "allow_nonzero": True},
        )
        self.assertEqual(result["tool"], "slurm_triage")
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
            "slurm_triage",
            {"args": ["this-verb-does-not-exist"], "allow_nonzero": True},
        )
        self.assertEqual(result["tool"], "slurm_triage")
        self.assertEqual(result["returncode"], 2)
        self.assertTrue(result["stderr"], "argparse error message expected on stderr")

    def test_allow_nonzero_requires_literal_true(self) -> None:
        for value in (False, "false", "true", 1, 0, None):
            with self.subTest(value=value):
                with self.assertRaises(RuntimeError):
                    run_surface_tool(
                        "slurm_triage",
                        {
                            "args": ["this-verb-does-not-exist"],
                            "allow_nonzero": value,
                        },
                    )

    def test_every_contract_leaf_parser_accepts_json(self) -> None:
        """Regression test: the runtime auto-appends `--json` to argv
        whenever `CONTRACT[verb]["json"]` is True. Every contract-derived
        leaf parser must accept that `--json` token, otherwise argparse
        rejects the call with `unrecognized arguments: --json` at runtime
        and the MCP envelope reports `returncode=2`. This bug was
        discovered on the live MCP for every `ai_tuning_*` tool. Keep this
        test green so the runtime cannot silently regress.

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
            for action in p._actions:
                if isinstance(action, _ap._SubParsersAction):
                    leaves: list[_ap.ArgumentParser] = []
                    for child in action.choices.values():
                        leaves.extend(collect_leaves(child))
                    return leaves
            return [p]

        for spec in mcp_surface.derive_tool_specs():
            if not spec.json:
                continue  # Runtime only auto-appends --json when spec.json is True.
            cli = mcp_surface._load_cli_module(spec.library)
            parser = cli.build_parser()
            verb_parser: _ap.ArgumentParser | None = None
            for action in parser._actions:
                if isinstance(action, _ap._SubParsersAction):
                    verb_parser = action.choices.get(spec.verb)
                    break
            self.assertIsNotNone(verb_parser, msg=f"{spec.name}: top-level subparser missing")
            for leaf in collect_leaves(verb_parser):
                option_strings = {
                    opt for action in leaf._actions for opt in action.option_strings
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

    def test_read_only_leaf_parsers_expose_no_write_options(self) -> None:
        import mcp_surface

        write_options = {
            "--apply",
            "--artifact-root",
            "--audit-dir",
            "--execute",
            "--out",
            "--output",
            "--output-dir",
            "--output-file",
            "--write-space",
        }

        def collect_leaves(parser: argparse.ArgumentParser) -> list[argparse.ArgumentParser]:
            for action in parser._actions:
                if isinstance(action, argparse._SubParsersAction):
                    return [
                        leaf
                        for child in action.choices.values()
                        for leaf in collect_leaves(child)
                    ]
            return [parser]

        for spec in mcp_surface.derive_tool_specs():
            if spec.safety != "read_only":
                continue
            module = mcp_surface._load_cli_module(spec.library)
            parser = module.build_parser()
            verb_parser = next(
                (
                    action.choices.get(spec.verb)
                    for action in parser._actions
                    if isinstance(action, argparse._SubParsersAction)
                ),
                None,
            )
            self.assertIsNotNone(verb_parser, spec.name)
            for leaf in collect_leaves(verb_parser):
                options = {
                    option
                    for action in leaf._actions
                    for option in action.option_strings
                }
                with self.subTest(name=spec.name, leaf=leaf.prog):
                    self.assertFalse(
                        options & write_options,
                        f"read_only tool exposes write options: {options & write_options}",
                    )


if __name__ == "__main__":
    unittest.main()
