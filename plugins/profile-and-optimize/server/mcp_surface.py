"""Contract-derived MCP surface for the bundled `profile_and_optimize` MCP server.

The MCP tool surface is auto-derived from the 8 stub libraries listed in
``LIBRARIES`` below. Each library ships a ``cli.py`` with a ``CONTRACT``
dict whose keys are the CLI verbs and whose values declare safety class,
ack-flag, and JSON-mode for each verb. One MCP tool is registered per
contract verb, named ``<library>_<verb>`` with hyphens converted to
underscores.

The mapping:

- Tool name: ``<library>_<verb_with_underscores>``.
- Safety class: copied from the package CLI's ``CONTRACT[verb]["safety"]``.
- Ack required: ``True`` whenever the CLI verb's contract entry has a
  non-empty ``ack`` flag.
- JSON parsing: ``True`` whenever the CLI exposes ``--json`` for the
  verb, which is true for every contracted verb in v6.0.

The contract test ``tools/profile_and_optimize_mcp/tests/test_server_smoke.py``
asserts:

1. The MCP-derived tool set equals ``_TOTAL_CONTRACT_TOOLS`` tools across
   ``_TOTAL_LIBRARIES`` libraries (see canonical-counts block below).
2. Every tool's safety class is one of the five allowed values and
   matches the contract.
3. Running the derivation twice yields the same tool list and safety
   mapping.
4. Every contract verb with ``json=True`` has a leaf argparse parser
   that accepts ``--json`` (otherwise the runtime auto-append would
   crash with ``unrecognized arguments: --json``).

This server is a small, auto-derived layer. The ``tools/profile_and_optimize_mcp/``
runtime imports this module rather than maintaining a second registry,
and the ``+2`` auxiliary search tools (``search_runbooks``,
``search_evidence``) live in that runtime — counted via
``_TOTAL_AUX_TOOLS`` below.

Canonical counts (single source of truth for every doc that names a
skill / tool / library count): see the constants block following
``LIBRARIES``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
ALLOWED_SAFETIES = {
    "read_only",
    "writes_artifacts",
    "submits_jobs",
    "pulls_data",
    "substitutes_nodes",
}
LIBRARIES = (
    "ai_tuning", "profile",
    "perf_baseline", "evidence",
    "slurm",
    "findings",
    # Backs the inference-perf-tune-report skill.
    "perf_tune_report",
    # Backs the inference-known-good-config skill (per-model required-flag registry + drift check).
    "known_good_config",
)


# ---------------------------------------------------------------------------
# Canonical counts — single source of truth.
#
# Every doc, smoke script, lint script, and CI gate that names a skill /
# tool / library count MUST read these constants (directly via Python
# import, or indirectly via ``scripts/lint-tool-counts.py``) instead of
# hardcoding a number. Drift between these constants and the live
# derivation is asserted at smoke-test time by
# ``verify_canonical_counts()`` below and by
# ``tools/profile_and_optimize_mcp/tests/test_server_smoke.py``.
#
# When a new library or verb is added:
#
#   1. Add the library name to ``LIBRARIES`` above (or add the verb to
#      that library's ``cli.py`` ``CONTRACT`` dict).
#   2. Update ``_TOTAL_CONTRACT_TOOLS`` and ``_TOTAL_MCP_TOOLS`` here.
#   3. ``scripts/lint-tool-counts.py`` will then fail any doc that
#      still says the old number.
#
# ---------------------------------------------------------------------------

#: Number of contract-derived MCP tools the bundled server exposes.
#: Verified by ``verify_canonical_counts()`` and by
#: ``test_server_smoke.test_server_can_be_created_when_mcp_is_installed``.
#:
#: 51 reflects the 8 libraries shipped in this public genericized build;
#: the internal toolchain this repo was rebuilt from carried additional
#: libraries (and their verbs) that were dropped during genericization.
#: When a verb is added or removed, update this constant —
#: ``scripts/lint-tool-counts.py`` fails any doc that still names the
#: old number.
_TOTAL_CONTRACT_TOOLS = 51

#: Number of auxiliary (non-contract-derived) MCP tools registered
#: directly by the FastMCP runtime in ``tools/profile_and_optimize_mcp/server.py``
#: (``search_runbooks`` and ``search_evidence``).
_TOTAL_AUX_TOOLS = 2

#: Total MCP tools exposed by the bundled server.
_TOTAL_MCP_TOOLS = _TOTAL_CONTRACT_TOOLS + _TOTAL_AUX_TOOLS  # 53

#: Number of stub libraries under ``server/`` registered in ``LIBRARIES``.
_TOTAL_LIBRARIES = len(LIBRARIES)  # 8

# Cheap import-time assertion: the ``LIBRARIES`` tuple must match the
# documented library count. The expensive
# ``verify_canonical_counts()`` check (which file-loads every library's
# ``cli.py`` to count contract verbs) is opt-in so importers don't pay
# the cost on every interpreter startup.
assert _TOTAL_LIBRARIES == 8, (
    f"LIBRARIES has {_TOTAL_LIBRARIES} entries but the canonical count is 8. "
    "Update _TOTAL_LIBRARIES + every doc that names this number via "
    "scripts/lint-tool-counts.py."
)


def verify_canonical_counts() -> dict[str, int]:
    """Verify the live MCP surface matches the canonical counts.

    Returns a dict with the live numbers; raises ``AssertionError`` if
    any disagree with the canonical constants above. Called by the
    Makefile ``smoke-test`` target, ``install.sh``, and
    ``test_server_smoke.py`` so a single import of this module is
    sufficient to detect drift everywhere it matters.

    This is intentionally NOT run at import time — file-loading every
    library's ``cli.py`` would slow every importer (including unrelated
    test runs) and would fail in environments where one library's
    optional dependencies (matplotlib, pandas, ...) are not installed.
    """
    contract_tools = len(derive_tool_specs())
    libraries = len(LIBRARIES)
    assert libraries == _TOTAL_LIBRARIES, (
        f"LIBRARIES has {libraries} entries; expected {_TOTAL_LIBRARIES}"
    )
    assert contract_tools == _TOTAL_CONTRACT_TOOLS, (
        f"derive_tool_specs() returned {contract_tools} tools; "
        f"expected {_TOTAL_CONTRACT_TOOLS}. Update _TOTAL_CONTRACT_TOOLS "
        "and every doc that names this number."
    )
    return {
        "libraries": libraries,
        "contract_tools": contract_tools,
        "aux_tools": _TOTAL_AUX_TOOLS,
        "total_mcp_tools": contract_tools + _TOTAL_AUX_TOOLS,
    }


@dataclass(frozen=True)
class ToolSpec:
    name: str
    library: str
    verb: str
    safety: str
    ack_required: bool
    json: bool
    description: str
    invoke: Callable[[list[str]], int] = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "library": self.library,
            "verb": self.verb,
            "safety": self.safety,
            "ack_required": self.ack_required,
            "json": self.json,
            "description": self.description,
        }


def _load_cli_module(library: str):
    """Load the CLI module for a library by file path.

    File-path loading avoids sys.modules collisions when another module
    in the tree shares a library's package name. The module is registered in
    ``sys.modules`` before execution so dataclass annotations resolved by
    ``from __future__ import annotations`` find the live module.
    """
    cache_key = f"mcp_{library}_cli"
    if cache_key in sys.modules:
        return sys.modules[cache_key]
    path = REPO_ROOT / library / "cli.py"
    spec = importlib.util.spec_from_file_location(cache_key, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load CLI module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[cache_key] = module
    spec.loader.exec_module(module)
    return module


def _verb_to_tool_name(library: str, verb: str) -> str:
    return f"{library}_{verb.replace('-', '_')}"


def _verb_description(module: Any, verb: str) -> str:
    """Return a one-line description from the CLI parser if available."""
    parser = module.build_parser()
    for action in parser._actions:  # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            sub = action.choices.get(verb)
            if sub is not None and sub.description:
                first_line = sub.description.strip().splitlines()[0]
                return first_line
    return module.CONTRACT[verb].get("description", verb)


def derive_tool_specs() -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    for library in LIBRARIES:
        module = _load_cli_module(library)
        contract = getattr(module, "CONTRACT", None)
        if not isinstance(contract, dict):
            raise RuntimeError(f"library {library!r} CLI module is missing CONTRACT dict")
        for verb in sorted(contract):
            entry = contract[verb]
            safety = str(entry.get("safety"))
            if safety not in ALLOWED_SAFETIES:
                raise RuntimeError(f"library {library!r} verb {verb!r} has unknown safety class {safety!r}")
            ack_required = bool(entry.get("ack"))
            json_mode = bool(entry.get("json", True))
            description = _verb_description(module, verb)
            tool_name = _verb_to_tool_name(library, verb)
            invoke = _make_invoke(module, verb)
            specs.append(
                ToolSpec(
                    name=tool_name,
                    library=library,
                    verb=verb,
                    safety=safety,
                    ack_required=ack_required,
                    json=json_mode,
                    description=description,
                    invoke=invoke,
                )
            )
    specs.sort(key=lambda spec: spec.name)
    return specs


def _make_invoke(module: Any, verb: str) -> Callable[[list[str]], int]:
    def _invoke(extra_args: list[str]) -> int:
        argv = [verb, *extra_args]
        rc = module.main(argv)
        return int(rc or 0)

    return _invoke


def list_tools() -> list[dict[str, Any]]:
    return [spec.to_dict() for spec in derive_tool_specs()]


def find_spec(name: str) -> ToolSpec:
    for spec in derive_tool_specs():
        if spec.name == name:
            return spec
    raise KeyError(name)


def _format_safety_summary(specs: list[ToolSpec]) -> dict[str, list[str]]:
    summary: dict[str, list[str]] = {safety: [] for safety in sorted(ALLOWED_SAFETIES)}
    for spec in specs:
        summary[spec.safety].append(spec.name)
    for safety in summary:
        summary[safety].sort()
    return summary


def _print_list(specs: list[ToolSpec], *, json_mode: bool) -> int:
    if json_mode:
        payload = {
            "schema_version": 1,
            "library_count": len(LIBRARIES),
            "tool_count": len(specs),
            "safety_summary": {safety: len(names) for safety, names in _format_safety_summary(specs).items()},
            "tools": [spec.to_dict() for spec in specs],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(f"derived {len(specs)} MCP tools across {len(LIBRARIES)} libraries:\n")
    for spec in specs:
        ack = " [ack]" if spec.ack_required else ""
        print(f"  {spec.name:<28} {spec.safety:<18} {spec.library}/{spec.verb}{ack}")
    return 0


def _print_safety(specs: list[ToolSpec], *, json_mode: bool) -> int:
    summary = _format_safety_summary(specs)
    if json_mode:
        print(json.dumps({"schema_version": 1, "safety_summary": summary}, indent=2, sort_keys=True))
        return 0
    for safety in sorted(summary):
        print(f"{safety} ({len(summary[safety])}):")
        for name in summary[safety]:
            print(f"  {name}")
        print()
    return 0


def _invoke_tool(name: str, argv: list[str], *, json_mode: bool) -> int:
    spec = find_spec(name)
    if json_mode and "--json" not in argv:
        argv = [*argv, "--json"]
    return spec.invoke(argv)


def _print_counts(*, json_mode: bool) -> int:
    """Print the canonical-count constants + live verification.

    Used by ``Makefile`` ``smoke-test``, ``install.sh``,
    ``scripts/smoke-mcp-runtime.sh``, and ``scripts/lint-tool-counts.py``
    so every consumer reads the same single source of truth instead of
    hardcoding the numbers.
    """
    live = verify_canonical_counts()
    if json_mode:
        payload = {
            "schema_version": 1,
            "canonical": {
                "libraries": _TOTAL_LIBRARIES,
                "contract_tools": _TOTAL_CONTRACT_TOOLS,
                "aux_tools": _TOTAL_AUX_TOOLS,
                "total_mcp_tools": _TOTAL_MCP_TOOLS,
            },
            "live": live,
            "verified": live["contract_tools"] == _TOTAL_CONTRACT_TOOLS
                       and live["libraries"] == _TOTAL_LIBRARIES,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(f"libraries:        {_TOTAL_LIBRARIES} (live: {live['libraries']})")
    print(f"contract_tools:   {_TOTAL_CONTRACT_TOOLS} (live: {live['contract_tools']})")
    print(f"aux_tools:        {_TOTAL_AUX_TOOLS}")
    print(f"total_mcp_tools:  {_TOTAL_MCP_TOOLS} (live: {live['total_mcp_tools']})")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Contract-derived MCP-tool surface for the bundled profile_and_optimize MCP server.",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable output for list / safety / counts / call")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list every derived MCP tool")
    sub.add_parser("safety", help="print derived tools grouped by safety class")
    sub.add_parser("counts", help="print canonical counts (libraries / contract_tools / aux_tools / total_mcp_tools) and verify they match the live derivation")
    call = sub.add_parser("call", help="invoke a derived MCP tool by name")
    call.add_argument("tool", help="tool name (for example, selector_pick)")
    call.add_argument("args", nargs=argparse.REMAINDER, help="extra args forwarded to the underlying CLI verb")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "counts":
        return _print_counts(json_mode=args.json)
    specs = derive_tool_specs()
    if args.command == "list":
        return _print_list(specs, json_mode=args.json)
    if args.command == "safety":
        return _print_safety(specs, json_mode=args.json)
    if args.command == "call":
        forwarded = list(args.args)
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]
        return _invoke_tool(args.tool, forwarded, json_mode=args.json)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
