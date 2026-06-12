#!/usr/bin/env python3
# PROFILE_AND_OPTIMIZE_OPT_OUT: internal module; tools/ai_tuning/ai_tuning.py remains the canonical CLI.
"""Local helpers for LLM-assisted MLPerf tuning.

The commands in this file are intentionally offline: they inspect artifacts,
validate proposals, and materialize derived files, but they do not submit Slurm
jobs or mutate cluster state.
"""

from __future__ import annotations

import argparse
import sys as _sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SPACE = REPO_ROOT / "tuning" / "tuning-space.b200-llama31-8b.json"

# Safety constants live in a small sibling module so reviewers can audit
# the forbidden-pattern table and ledger-status enum without paging
# through the full tuner CLI. Re-exports preserve backwards compat for
# callers that still import these names from this module.
from tools.ai_tuning import helpers as _helpers
from tools.ai_tuning.cmd_experiment import (
    command_experiment_collect,
    command_experiment_create,
    command_experiment_poll,
    command_experiment_submit,
    command_experiment_summary,
    command_experiment_update,
)
from tools.ai_tuning.cmd_finalize import command_finalize
from tools.ai_tuning.cmd_optimizer import (
    command_optimizer_compare,
    command_optimizer_history,
    command_optimizer_import_hyp,
    command_optimizer_propose,
    command_optimizer_status,
)
from tools.ai_tuning.cmd_proposal import command_proposal_diff, command_proposal_validate
from tools.ai_tuning.cmd_report import command_report
from tools.ai_tuning.cmd_space import command_matrix, command_space
from tools.ai_tuning.cmd_template_patch import command_template_patch_validate

try:  # Preserve object identity for direct imports from tools/ai_tuning.
    import safety as _safety  # type: ignore
except ImportError:  # Package import path.
    from tools.ai_tuning import safety as _safety


# Per AGENTS.md, the canonical operator surface for AI-assisted tuning is
# offline: every CLI verb here either prints to stdout, or writes its own
# `--output` / per-verb output path. Slurm submission only happens through
# the `experiment submit` subverb, which carries its own
# `--i-understand-this-submits-jobs` ack flag at the subparser level
# (mirroring `launcher launch`). The MCP top-level CONTRACT entry for
# `experiment` therefore stays at `writes_artifacts` rather than escalating
# the whole umbrella to `submits_jobs` (which would auto-append the ack flag
# to all subverbs and break read-only callers like `experiment summary`).
CONTRACT: dict[str, dict[str, object]] = {
    "space": {
        "safety": "read_only",
        "required": (),
        "optional": ("--space", "--names-only", "--output"),
        "json": True,
        "ack": None,
    },
    "matrix": {
        "safety": "writes_artifacts",
        "required": ("--parameter",),
        "optional": ("--space", "--limit", "--output"),
        "json": True,
        "ack": None,
    },
    "optimizer": {
        "safety": "writes_artifacts",
        "required": ("subverb",),
        # Subverbs (propose | status | history | compare | import-hyp) own
        # their own flag sets; the umbrella parser holds none directly.
        "optional": (),
        "json": True,
        "ack": None,
    },
    "report": {
        "safety": "read_only",
        "required": (),
        "optional": (
            "--space", "--raw-results-dir", "--raw-benchmark", "--min-runs",
            "--objective", "--gb300-fabric-dir", "--gb300-fabric-require-clean",
            "--gb300-node-selection-dir", "--gb300-fabric-localization-dir",
            "--ledger", "--remaining-limit", "--template-hint-file",
            "--template-hint-limit", "--error-limit", "--output",
        ),
        "json": True,
        "ack": None,
    },
    "finalize": {
        "safety": "writes_artifacts",
        "required": ("--log-dir", "--workdir", "--results-dir"),
        "optional": ("--benchmark", "--run-id", "--required-runs", "--launcher-file", "--dry-run", "--output"),
        "json": True,
        "ack": None,
    },
    "proposal": {
        "safety": "read_only",
        "required": ("subverb",),
        "optional": (),
        "json": True,
        "ack": None,
    },
    "template-patch": {
        "safety": "writes_artifacts",
        "required": ("subverb",),
        # `template-patch validate --apply` mutates files; treat the
        # umbrella as writes_artifacts so the MCP envelope's safety field
        # warns the caller before invocation.
        "optional": (),
        "json": True,
        "ack": None,
    },
    "experiment": {
        "safety": "writes_artifacts",
        "required": ("subverb",),
        # `experiment submit` requires --i-understand-this-submits-jobs at
        # the subverb level; pass it through `args` when actually
        # submitting. The umbrella stays writes_artifacts so the MCP
        # runtime does not auto-append the ack flag to read-only subverbs.
        "optional": (),
        "json": True,
        "ack": None,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    space = subparsers.add_parser("space", help="Print a tuning space manifest")
    space.add_argument("--space", type=Path, default=DEFAULT_SPACE)
    space.add_argument("--names-only", action="store_true")
    space.add_argument("--output", type=Path)
    space.set_defaults(func=command_space)

    matrix = subparsers.add_parser("matrix", help="Generate a bounded proposal matrix")
    matrix.add_argument("--space", type=Path, default=DEFAULT_SPACE)
    matrix.add_argument(
        "--parameter",
        action="append",
        required=True,
        help="Parameter name to include in the cartesian matrix. Repeat for multiple dimensions.",
    )
    matrix.add_argument("--limit", type=int, default=32)
    matrix.add_argument("--output", type=Path)
    matrix.set_defaults(func=command_matrix)

    optimizer = subparsers.add_parser("optimizer", help="Optimizer-backed proposal helpers")
    optimizer_subparsers = optimizer.add_subparsers(dest="optimizer_command", required=True)
    optimizer_propose = optimizer_subparsers.add_parser(
        "propose", help="Generate candidates with recorded optimizer state"
    )
    optimizer_propose.add_argument("--space", type=Path, default=DEFAULT_SPACE)
    optimizer_propose.add_argument(
        "--strategy",
        choices=["deterministic_matrix", "random", "bayesian", "multifidelity"],
        default="deterministic_matrix",
    )
    optimizer_propose.add_argument(
        "--variant",
        choices=["tpe", "gp", "hyperband", "bohb"],
        default=None,
    )
    optimizer_propose.add_argument("--parameter", action="append", default=[])
    optimizer_propose.add_argument("--objective")
    optimizer_propose.add_argument("--ledger", type=Path)
    optimizer_propose.add_argument("--limit", type=int, default=32)
    optimizer_propose.add_argument("--seed", type=int, default=0)
    optimizer_propose.add_argument("--state-file", type=Path)
    optimizer_propose.add_argument("--state-out", type=Path)
    optimizer_propose.add_argument("--eta", type=int)
    optimizer_propose.add_argument("--min-budget", type=float)
    optimizer_propose.add_argument("--max-budget", type=float)
    optimizer_propose.add_argument("--output", type=Path)
    optimizer_propose.set_defaults(func=command_optimizer_propose)

    optimizer_status = optimizer_subparsers.add_parser(
        "status", help="Print which serious-tuner contracts and strategies are available for a tuning space"
    )
    optimizer_status.add_argument("--space", type=Path, default=DEFAULT_SPACE)
    optimizer_status.add_argument("--output", type=Path)
    optimizer_status.set_defaults(func=command_optimizer_status)

    optimizer_history = optimizer_subparsers.add_parser(
        "history", help="Summarize observed ledger trials grouped by objective"
    )
    optimizer_history.add_argument("--ledger", type=Path, required=True)
    optimizer_history.add_argument("--space", type=Path)
    optimizer_history.add_argument("--objective")
    optimizer_history.add_argument("--limit", type=int, default=100)
    optimizer_history.add_argument("--output", type=Path)
    optimizer_history.set_defaults(func=command_optimizer_history)

    optimizer_compare = optimizer_subparsers.add_parser(
        "compare", help="Objective-aware delta between two ledger experiments"
    )
    optimizer_compare.add_argument("experiment_a")
    optimizer_compare.add_argument("experiment_b")
    optimizer_compare.add_argument("--ledger", type=Path, required=True)
    optimizer_compare.add_argument("--space", type=Path, required=True)
    optimizer_compare.add_argument("--objective")
    optimizer_compare.add_argument("--output", type=Path)
    optimizer_compare.set_defaults(func=command_optimizer_compare)

    optimizer_import_hyp = optimizer_subparsers.add_parser(
        "import-hyp", help="Import a hypertune .hyp template and optional session directory"
    )
    optimizer_import_hyp.add_argument("template")
    optimizer_import_hyp.add_argument("--session-dir")
    optimizer_import_hyp.add_argument("--ledger", type=Path)
    optimizer_import_hyp.add_argument("--write-space", type=Path)
    optimizer_import_hyp.add_argument("--imported-space-id")
    optimizer_import_hyp.add_argument("--target-benchmark")
    optimizer_import_hyp.add_argument("--default-objective")
    optimizer_import_hyp.add_argument("--output", type=Path)
    optimizer_import_hyp.set_defaults(func=command_optimizer_import_hyp)

    report = subparsers.add_parser("report", help="Build a bounded JSON tuning report")
    report.add_argument("--space", type=Path, default=DEFAULT_SPACE)
    report.add_argument("--raw-results-dir", action="append", default=[])
    report.add_argument("--raw-benchmark", default="llama31_8b")
    report.add_argument("--min-runs", type=int)
    report.add_argument("--objective")
    report.add_argument("--gb300-fabric-dir", action="append", default=[])
    report.add_argument("--gb300-fabric-require-clean", action="store_true")
    report.add_argument("--gb300-node-selection-dir", action="append", default=[])
    report.add_argument("--gb300-fabric-localization-dir", action="append", default=[])
    report.add_argument("--ledger", type=Path)
    report.add_argument("--remaining-limit", type=int, default=20)
    report.add_argument("--template-hint-file", action="append", default=[])
    report.add_argument("--template-hint-limit", type=int, default=40)
    report.add_argument("--error-limit", type=int, default=50)
    report.add_argument("--output", type=Path)
    report.set_defaults(func=command_report)

    finalize = subparsers.add_parser("finalize", help="Assemble or dry-run an MLPerf result bundle")
    finalize.add_argument("--log-dir", type=Path, required=True)
    finalize.add_argument("--workdir", type=Path, required=True)
    finalize.add_argument("--results-dir", type=Path, required=True)
    finalize.add_argument("--benchmark", default="llama31_405b")
    finalize.add_argument("--run-id", action="append", default=[])
    finalize.add_argument("--required-runs", type=int, default=5)
    finalize.add_argument("--launcher-file", type=Path)
    finalize.add_argument("--dry-run", action="store_true")
    finalize.add_argument("--output", type=Path)
    finalize.set_defaults(func=command_finalize)

    proposal = subparsers.add_parser("proposal", help="Proposal helpers")
    proposal_subparsers = proposal.add_subparsers(dest="proposal_command", required=True)
    proposal_validate = proposal_subparsers.add_parser("validate", help="Validate an LLM proposal")
    proposal_validate.add_argument("proposal", type=Path)
    proposal_validate.add_argument("--space", type=Path, default=DEFAULT_SPACE)
    proposal_validate.add_argument("--history", action="append", default=[])
    proposal_validate.add_argument("--ledger", type=Path)
    proposal_validate.add_argument("--require-complete", action="store_true")
    proposal_validate.add_argument("--audit-dir", type=Path)
    proposal_validate.add_argument("--output", type=Path)
    proposal_validate.set_defaults(func=command_proposal_validate)

    proposal_diff = proposal_subparsers.add_parser(
        "diff",
        help=(
            "v3.7 W10: diff two proposal.json files (added/removed/"
            "changed candidates by experiment_id_prefix)."
        ),
    )
    proposal_diff.add_argument("before", type=Path, help="earlier proposal.json")
    proposal_diff.add_argument("after", type=Path, help="later proposal.json")
    proposal_diff.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format. Default markdown.",
    )
    proposal_diff.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to stdout.",
    )
    proposal_diff.set_defaults(func=command_proposal_diff)

    template_patch = subparsers.add_parser("template-patch", help="Template patch helpers")
    template_patch_subparsers = template_patch.add_subparsers(
        dest="template_patch_command", required=True
    )
    patch_validate = template_patch_subparsers.add_parser(
        "validate", help="Validate a context-anchored patch"
    )
    patch_validate.add_argument("patch", type=Path)
    patch_validate.add_argument("--apply", action="store_true")
    patch_validate.add_argument("--output-file", type=Path)
    patch_validate.add_argument("--audit-dir", type=Path)
    patch_validate.add_argument("--output", type=Path)
    patch_validate.set_defaults(func=command_template_patch_validate)

    experiment = subparsers.add_parser("experiment", help="Experiment ledger helpers")
    experiment_subparsers = experiment.add_subparsers(dest="experiment_command", required=True)
    experiment_create = experiment_subparsers.add_parser(
        "create", help="Create planned experiment records from a proposal"
    )
    experiment_create.add_argument("proposal", type=Path)
    experiment_create.add_argument("--space", type=Path, default=DEFAULT_SPACE)
    experiment_create.add_argument("--ledger", type=Path, required=True)
    experiment_create.add_argument("--owner", default="cursor-session")
    experiment_create.add_argument("--artifact-root", type=Path)
    experiment_create.add_argument("--notes")
    experiment_create.add_argument("--output", type=Path)
    experiment_create.set_defaults(func=command_experiment_create)

    experiment_update = experiment_subparsers.add_parser(
        "update", help="Append a status update to an experiment ledger"
    )
    experiment_update.add_argument("experiment_id")
    experiment_update.add_argument("--ledger", type=Path, required=True)
    experiment_update.add_argument("--status", required=True)
    experiment_update.add_argument("--slurm-job-id")
    experiment_update.add_argument("--artifact-dir", type=Path)
    experiment_update.add_argument("--notes")
    experiment_update.add_argument("--output", type=Path)
    experiment_update.set_defaults(func=command_experiment_update)

    experiment_summary = experiment_subparsers.add_parser(
        "summary", help="Summarize latest experiment states from a ledger"
    )
    experiment_summary.add_argument("--ledger", type=Path, required=True)
    experiment_summary.add_argument("--status")
    experiment_summary.add_argument("--limit", type=int, default=100)
    experiment_summary.add_argument("--output", type=Path)
    experiment_summary.set_defaults(func=command_experiment_summary)

    experiment_submit = experiment_subparsers.add_parser(
        "submit", help="Preview or execute gated sbatch submissions from the ledger"
    )
    experiment_submit.add_argument("--ledger", type=Path, required=True)
    experiment_submit.add_argument("--script", type=Path, required=True)
    experiment_submit.add_argument("--experiment-id", action="append", default=[])
    experiment_submit.add_argument("--status", action="append", default=["planned", "staged"])
    experiment_submit.add_argument("--max-concurrent", type=int, default=1)
    experiment_submit.add_argument(
        "--sbatch-arg",
        action="append",
        default=[],
        help="Additional sbatch argument. Repeat for multiple arguments.",
    )
    experiment_submit.add_argument("--execute", action="store_true")
    experiment_submit.add_argument("--i-understand-this-submits-jobs", action="store_true")
    experiment_submit.add_argument("--materialize-wrappers", action="store_true")
    experiment_submit.add_argument("--wrapper-dir", type=Path, default=Path("experiments/generated-sbatch"))
    experiment_submit.add_argument("--overwrite-wrappers", action="store_true")
    experiment_submit.add_argument(
        "--remote-script",
        help="Script path to execute inside the submitted environment. Defaults to a script with the same basename next to the wrapper.",
    )
    experiment_submit.add_argument("--notes")
    experiment_submit.add_argument("--output", type=Path)
    experiment_submit.set_defaults(func=command_experiment_submit)

    experiment_poll = experiment_subparsers.add_parser(
        "poll", help="Poll read-only Slurm status and update the ledger"
    )
    experiment_poll.add_argument("--ledger", type=Path, required=True)
    experiment_poll.add_argument("--status", action="append", default=["submitted", "running"])
    experiment_poll.add_argument("--status-file", type=Path)
    experiment_poll.add_argument("--no-update", action="store_true")
    experiment_poll.add_argument("--notes")
    experiment_poll.add_argument("--output", type=Path)
    experiment_poll.set_defaults(func=command_experiment_poll)

    experiment_collect = experiment_subparsers.add_parser(
        "collect", help="Collect local artifacts, validate them, and update the ledger"
    )
    experiment_collect.add_argument("experiment_id")
    experiment_collect.add_argument("--ledger", type=Path, required=True)
    experiment_collect.add_argument("--source", type=Path, required=True)
    experiment_collect.add_argument("--destination", type=Path)
    experiment_collect.add_argument("--raw-benchmark", default="llama31_8b")
    experiment_collect.add_argument("--min-runs", type=int, default=5)
    experiment_collect.add_argument("--validate-raw", action="store_true")
    experiment_collect.add_argument("--overwrite", action="store_true")
    experiment_collect.add_argument("--notes")
    experiment_collect.add_argument("--output", type=Path)
    experiment_collect.set_defaults(func=command_experiment_collect)

    # Per docs/cli-contract.md "Output mode": every verb must accept --json.
    # The ai_tuning callbacks already emit JSON unconditionally, but the
    # MCP runtime auto-appends `--json` to argv whenever
    # `CONTRACT[verb]["json"]` is True (which it is for all ai_tuning verbs),
    # so every leaf subparser must accept --json or argparse rejects the
    # call with `unrecognized arguments: --json`. Walk the parser tree once
    # and inject a no-op --json on every leaf that doesn't already declare
    # one, rather than scattering 18 boilerplate `add_argument` calls
    # across the parser-construction blocks above.
    _ensure_json_flag_on_leaves(parser)
    return parser


def _ensure_json_flag_on_leaves(parser: argparse.ArgumentParser) -> None:
    """Add a no-op `--json` flag to every leaf subparser that lacks one."""

    def visit(node: argparse.ArgumentParser) -> None:
        nested_action: argparse._SubParsersAction | None = None
        for action in node._actions:  # noqa: SLF001
            if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
                nested_action = action
                break
        if nested_action is not None:
            for child in nested_action.choices.values():
                visit(child)
            return
        existing_options = {
            opt for action in node._actions for opt in action.option_strings  # noqa: SLF001
        }
        if "--json" not in existing_options:
            node.add_argument(
                "--json",
                action="store_true",
                help="No-op; the underlying ai_tuning command emits JSON unconditionally.",
            )

    visit(parser)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def __getattr__(name: str):
    """Compatibility exports while command modules use explicit imports."""
    safety_module = _sys.modules.get("safety", _safety)
    if hasattr(safety_module, name):
        return getattr(safety_module, name)
    if hasattr(_helpers, name):
        return getattr(_helpers, name)
    raise AttributeError(name)

_EXPORT_DENYLIST = {
    "Any",
    "Path",
    "annotations",
    "argparse",
    "contextlib",
    "dt",
    "gp_engine",
    "hashlib",
    "hyp_format",
    "hyp_session",
    "importlib",
    "io",
    "itertools",
    "json",
    "random",
    "re",
    "shutil",
    "space_module",
    "statistics",
    "subprocess",
    "sys",
    "tpe_engine",
}
__all__ = [name for name in globals() if not name.startswith("_") and name not in _EXPORT_DENYLIST]

if __name__ == "__main__":
    raise SystemExit(main())
