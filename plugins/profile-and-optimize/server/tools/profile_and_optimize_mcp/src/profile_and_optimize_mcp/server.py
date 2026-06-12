"""FastMCP runtime for the contract-derived MLPerf tool surface.

The MCP runtime imports ``mcp_surface.py`` from the repo root and
registers exactly one MCP tool per launcher / selector / validator CLI
verb. There is no hand-maintained registry.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .repo import find_repo_root

RESOURCE_PATHS: dict[str, str] = {
    "perftune://repo/docs/cli-contract.md": "docs/cli-contract.md",
    "perftune://repo/docs/operator-commands.md": "docs/operator-commands.md",
    "perftune://repo/docs/mcp-tool-io-contract.md": "docs/mcp-tool-io-contract.md",
    "perftune://repo/docs/mcp-composition.md": "docs/mcp-composition.md",
    "perftune://repo/runbooks/gb300_405b.md": "runbooks/gb300_405b.md",
    "perftune://repo/runbooks/dsv3_671b.md": "runbooks/dsv3_671b.md",
    "perftune://repo/runbooks/b200_8b.md": "runbooks/b200_8b.md",
    "perftune://repo/tools/leaderboard/RELEASE-DAY.md": "tools/leaderboard/RELEASE-DAY.md",
    "perftune://repo/experiments/artifacts/leaderboard/CURRENT.md": "experiments/artifacts/leaderboard/CURRENT.md",
    # Phase 5: operator-runnable shell scripts exposed as static resources so
    # callers can pull the source verbatim. The two scripts are runnable as
    # `bash <path>`; the MCP surface does NOT execute them.
    "perftune://repo/tools/pipeline/submission/capture_evidence_bundle.sh": "tools/pipeline/submission/capture_evidence_bundle.sh",
    "perftune://repo/tools/leaderboard/scripts/build_bundle.sh": "tools/leaderboard/scripts/build_bundle.sh",
}


SEARCH_TOOL_SPECS: dict[str, list[str]] = {
    "search_runbooks": ["runbooks", "docs"],
    "search_evidence": ["experiments/artifacts"],
}


def _search(name: str, paths: list[str], query: str, *, limit: int = 50) -> dict[str, Any]:
    """Wrap `rg` in the same envelope the contract-derived MCP tools use.

    The auxiliary MCP-only tools (`search_runbooks`, `search_evidence`) are
    not CLI verbs, so they have no library / verb / ack semantics from the
    CLI contract. They still return the same envelope shape so MCP clients
    can use one parser path; library is set to ``mcp_aux`` and verb to
    ``search`` to make the auxiliary nature visible.
    """
    repo = find_repo_root()
    argv = [query, "--limit", str(limit), "--paths", *paths]
    cmd = ["rg", "--line-number", "--max-count", str(limit), query, *paths]
    proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, check=False)
    matches = proc.stdout.splitlines() if proc.returncode in (0, 1) else []
    payload = {"query": query, "paths": paths, "matches": matches}
    return {
        "tool": name,
        "library": "mcp_aux",
        "verb": "search",
        "safety": "read_only",
        "ack_required": False,
        "ack_field": None,
        "args": argv,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "json": payload,
    }


def _load_mcp_surface():
    repo_root = str(find_repo_root())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return importlib.import_module("mcp_surface")


def tool_names() -> list[str]:
    surface = _load_mcp_surface()
    return [spec.name for spec in surface.derive_tool_specs()]


def _ack_field(flag: str | None) -> str | None:
    if flag is None:
        return None
    return flag.lstrip("-").replace("-", "_")


def _args_from_params(params: dict[str, Any]) -> list[str]:
    raw = params.get("args", [])
    if isinstance(raw, str):
        return [raw]
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise TypeError("params['args'] must be a string or list of strings")
    return list(raw)


def run_surface_tool(name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    surface = _load_mcp_surface()
    params = dict(params or {})
    spec = surface.find_spec(name)
    module = surface._load_cli_module(spec.library)  # noqa: SLF001 - MCP runtime intentionally mirrors the CLI contract.
    ack_flag = module.CONTRACT[spec.verb].get("ack")
    ack_field = _ack_field(ack_flag)
    argv = _args_from_params(params)
    if ack_flag and ack_field and params.get(ack_field):
        argv.append(ack_flag)
    if spec.json and "--json" not in argv:
        argv.append("--json")

    stdout = io.StringIO()
    stderr = io.StringIO()
    # Argparse's `--help`, an unknown verb, or any other intentional
    # `sys.exit()` inside the underlying CLI raises `SystemExit`. Without a
    # guard, that exception propagates through FastMCP's stdin/stdout JSON-RPC
    # loop and terminates the entire server process, so all subsequent MCP
    # calls hang up with "Connection closed" / "Not connected" until the
    # operator restarts. Catch it here, normalize to the standard envelope's
    # returncode field, and let the caller decide via `allow_nonzero`.
    rc: int
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rc = spec.invoke(argv)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            rc = 0
        elif isinstance(code, int):
            rc = code
        else:
            stderr.write(str(code) + "\n")
            rc = 1

    out = stdout.getvalue()
    err = stderr.getvalue()
    parsed: Any = None
    if out.strip():
        try:
            parsed = json.loads(out)
        except json.JSONDecodeError:
            parsed = None
    result = {
        "tool": name,
        "library": spec.library,
        "verb": spec.verb,
        "safety": spec.safety,
        "ack_required": spec.ack_required,
        "ack_field": ack_field,
        "args": argv,
        "returncode": int(rc),
        "stdout": out,
        "stderr": err,
        "json": parsed,
    }
    if rc != 0 and not params.get("allow_nonzero", False):
        raise RuntimeError(json.dumps(result, sort_keys=True))
    return result


def create_server() -> Any:
    """Create and return the FastMCP server.

    Importing ``mcp`` is intentionally delayed so registry/unit tests can run
    without an MCP runtime installed.
    """

    from mcp.server.fastmcp import FastMCP  # type: ignore

    mcp = FastMCP(
        "profile_and_optimize",
        instructions=(
            "MLPerf Training + inference perf tools derived from launcher "
            "/ selector / validator CLI contracts. Mutating tools mirror the CLI ack "
            "flags. GRIND MANDATE (always-applied): performance work is never 'done' "
            "-- after every measured result, always hunt the next BREAKTHROUGH (the "
            "highest-EV unlock toward Speed-of-Light), not just the next micro-lever; "
            "every finding names its next_lever, and a breakthrough claim stays a DRAFT "
            "until variance-controlled, metric-isolated, fair-baselined, profiled, and "
            "SoL-grounded."
        ),
    )

    surface = _load_mcp_surface()

    def make_tool(spec: Any):
        async def tool(params: dict[str, Any] | None = None) -> dict[str, Any]:
            return run_surface_tool(spec.name, params)

        tool.__name__ = spec.name
        tool.__doc__ = spec.description
        return tool

    for spec in surface.derive_tool_specs():
        mcp.tool()(make_tool(spec))

    def make_search_tool(name: str, paths: list[str]):
        async def search_tool(query: str, limit: int = 50) -> dict[str, Any]:
            return _search(name, paths, query, limit=limit)

        search_tool.__name__ = name
        search_tool.__doc__ = (
            f"Auxiliary MCP-only read-only search over {', '.join(paths)}. "
            "Returns the standard profile_and_optimize envelope with library='mcp_aux' "
            "and verb='search'."
        )
        return search_tool

    for name, paths in SEARCH_TOOL_SPECS.items():
        mcp.tool()(make_search_tool(name, paths))

    def make_resource(rel_path: str):
        def resource() -> str:
            path = find_repo_root() / rel_path
            return path.read_text(encoding="utf-8")

        resource.__name__ = Path(rel_path).stem.replace("-", "_")
        return resource

    for uri, rel_path in RESOURCE_PATHS.items():
        mcp.resource(uri)(make_resource(rel_path))

    return mcp


def run_stdio() -> None:
    create_server().run()
