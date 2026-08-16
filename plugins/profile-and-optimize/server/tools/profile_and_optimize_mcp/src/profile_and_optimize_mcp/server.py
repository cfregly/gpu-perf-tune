"""FastMCP runtime for contract-derived GPU performance tools.

The MCP runtime imports ``mcp_surface.py`` from the repo root and
registers one MCP tool per supported CLI verb. There is no second tool
registry to maintain.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .repo import find_repo_root

RESOURCE_PATHS: dict[str, str] = {
    "perftune://repo/docs/cli-contract.md": "docs/cli-contract.md",
    "perftune://repo/docs/operator-commands.md": "docs/operator-commands.md",
    "perftune://repo/docs/mcp-tool-io-contract.md": "docs/mcp-tool-io-contract.md",
    "perftune://repo/docs/mcp-composition.md": "docs/mcp-composition.md",
    "perftune://repo/docs/performance-hints.md": "docs/performance-hints.md",
    "perftune://repo/runbooks/profile-a-regression.md": "runbooks/profile-a-regression.md",
}


SEARCH_TOOL_SPECS: dict[str, list[str]] = {
    "search_runbooks": ["runbooks", "docs"],
    "search_evidence": ["experiments/artifacts"],
}

_MAX_SEARCH_RESULTS = 100
_MAX_SEARCH_LINE_CHARS = 8192
_MAX_SEARCH_STDERR_CHARS = 8192


def _search_command(query: str, paths: list[str], limit: int) -> list[str]:
    return [
        "rg",
        "--line-number",
        "--max-columns",
        "4096",
        "--max-columns-preview",
        "--max-count",
        str(limit),
        "--",
        query,
        *paths,
    ]


def _search(name: str, paths: list[str], query: str, *, limit: int = 50) -> dict[str, Any]:
    """Wrap `rg` in the same envelope the contract-derived MCP tools use.

    The auxiliary MCP-only tools (`search_runbooks`, `search_evidence`) are
    not CLI verbs, so they have no library / verb / ack semantics from the
    CLI contract. They still return the same envelope shape so MCP clients
    can use one parser path. Library is set to ``mcp_aux`` and verb to
    ``search`` to make the auxiliary nature visible.
    """
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be an integer")
    if not 1 <= limit <= _MAX_SEARCH_RESULTS:
        raise ValueError(f"limit must be between 1 and {_MAX_SEARCH_RESULTS}")

    repo = find_repo_root()
    argv = [query, "--limit", str(limit), "--paths", *paths]
    cmd = _search_command(query, paths, limit)
    matches: list[str] = []
    stderr = ""
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr_file:
        proc = subprocess.Popen(
            cmd,
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
        )
        assert proc.stdout is not None
        while len(matches) < limit:
            line = proc.stdout.readline(_MAX_SEARCH_LINE_CHARS)
            if not line:
                break
            matches.append(line.rstrip("\n"))

        capped = len(matches) == limit and proc.poll() is None
        if capped:
            proc.terminate()
        try:
            returncode = proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            returncode = proc.wait(timeout=2)

        stderr_file.seek(0)
        stderr = stderr_file.read(_MAX_SEARCH_STDERR_CHARS)

    if capped and matches:
        returncode = 0
    stdout = "".join(f"{match}\n" for match in matches)
    if returncode not in (0, 1):
        matches = []
    payload = {"query": query, "paths": paths, "matches": matches}
    return {
        "tool": name,
        "library": "mcp_aux",
        "verb": "search",
        "safety": "read_only",
        "ack_required": False,
        "ack_field": None,
        "args": argv,
        "returncode": int(returncode),
        "stdout": stdout,
        "stderr": stderr,
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


def _is_option_abbreviation(value: str, option: str) -> bool:
    return value.startswith("--") and value != option and option.startswith(value)


def _dynamic_ack(
    module: Any,
    verb: str,
    argv: list[str],
) -> tuple[str | None, bool, str | None]:
    declarations = getattr(module, "MCP_DYNAMIC_ACKS", {})
    verb_declarations = declarations.get(verb, {}) if isinstance(declarations, dict) else {}
    if not argv or not isinstance(verb_declarations, dict):
        return None, False, None
    declaration = verb_declarations.get(argv[0])
    if not isinstance(declaration, dict):
        return None, False, None
    ack_flag = declaration.get("ack")
    required_with = declaration.get("required_with")
    safety = declaration.get("safety")
    if not isinstance(ack_flag, str) or not isinstance(required_with, str):
        raise TypeError(f"invalid dynamic acknowledgement declaration for {verb} {argv[0]}")
    for arg in argv[1:]:
        if _is_option_abbreviation(arg, required_with):
            raise ValueError(
                f"abbreviated option {arg!r} is not allowed; use {required_with!r}"
            )
    return ack_flag, required_with in argv, str(safety) if safety else None


def run_surface_tool(name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    surface = _load_mcp_surface()
    params = dict(params or {})
    spec = surface.find_spec(name)
    module = surface._load_cli_module(spec.library)
    ack_flag = module.CONTRACT[spec.verb].get("ack")
    argv = _args_from_params(params)
    dynamic_ack_flag, dynamic_ack_required, dynamic_safety = _dynamic_ack(
        module,
        spec.verb,
        argv,
    )
    effective_ack_flag = ack_flag or dynamic_ack_flag
    ack_exempt_when = module.CONTRACT[spec.verb].get("ack_exempt_when", ())
    if not isinstance(ack_exempt_when, (tuple, list)) or not all(
        isinstance(flag, str) for flag in ack_exempt_when
    ):
        raise TypeError(f"invalid acknowledgement exemption for {spec.library} {spec.verb}")
    static_ack_required = bool(ack_flag) and not any(flag in argv for flag in ack_exempt_when)
    ack_required = static_ack_required or dynamic_ack_required
    ack_field = _ack_field(effective_ack_flag)
    raw_ack = next(
        (
            arg
            for arg in argv
            if effective_ack_flag
            and (arg == effective_ack_flag or _is_option_abbreviation(arg, effective_ack_flag))
        ),
        None,
    )
    if effective_ack_flag and raw_ack:
        raise ValueError(
            f"pass {ack_field}=true instead of including {raw_ack!r} in params.args"
        )
    if ack_required and ack_field and params.get(ack_field) is not True:
        raise PermissionError(f"{spec.name} requires {ack_field}=true")
    if ack_required and effective_ack_flag and ack_field:
        argv.append(effective_ack_flag)
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
        "safety": dynamic_safety if dynamic_ack_required and dynamic_safety else spec.safety,
        "ack_required": ack_required,
        "ack_field": ack_field,
        "args": argv,
        "returncode": int(rc),
        "stdout": out,
        "stderr": err,
        "json": parsed,
    }
    if rc != 0 and params.get("allow_nonzero") is not True:
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
            "GPU profiling and performance optimization tools derived from 8 CLI "
            "contracts. Ack-gated tools require the matching structured field. "
            "External S3 publish requires current-call approval, while publish dry "
            "runs write local files only and need no acknowledgement. "
            "Keep estimates and single observations labeled DRAFT. Before broad sweeps "
            "or captures, follow perftune://repo/docs/performance-hints.md. Write a cost "
            "ledger, rank ideas by measured contributor share, and run one controlled "
            "experiment. End each measured result with the next candidate lever and the "
            "gate that will prove or refute it."
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
