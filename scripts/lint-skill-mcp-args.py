#!/usr/bin/env python3
"""Cross-check SKILL.md `with:` argument blocks against MCP tool descriptors.

This lint exists because v0.8.0 -> v0.8.1 was a wasted PATCH cycle: a SKILL.md
called `list_prometheus_metric_names with: filter: "dpu"`, but the actual MCP
tool descriptor declared the parameter as `regex`. The the Prometheus MCP MCP
server silently dropped the unknown key and returned the alphabetically-first
page of metrics; the operator interpreted that as missing DPU data.

This lint catches that class of bug at commit time:

1. Parse `plugins/profile-and-optimize/skills/*/SKILL.md` frontmatter `allowed-tools`.
2. Scan each body for `mcp__<server>__<tool> with:` blocks and extract the
   top-level argument names from the indented YAML-ish body.
3. For each (tool, arg-name) pair:
   - External MCP servers: load the descriptor from the Cursor MCP cache
     (`~/.cursor/projects/<workspace>/mcps/<remapped-server>/tools/<tool>.json`)
     and verify the arg appears in `arguments.properties`.
   - Bundled `profile_and_optimize` MCP server: verify the tool name appears in
     `mcp_surface.py list` output; we don't have static arg-name schemas
     for bundled tools (their args are validated by `args:` list, not by
     named keys), so we accept any arg-name that the surface advertises.
   - Servers declared in `.mcp.json` with the `${VAR:-true}` no-op
     placeholder pattern (e.g. `<node-diagnosis-tool>`, `<blast-radius-tool>`, `<vms-tool>`,
     `cursor_ide_browser`) are intentionally optional; frontmatter
     references to them are accepted silently. The skill body's own
     skip-gracefully behavior is the contract; the marketplace
     declaration documents the intent.

Exit codes:
  0 - clean (no unknown args).
  1 - >=1 RED finding (unknown arg in some skill body).
  2 - fatal (descriptor folder missing, etc.).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = REPO_ROOT / "plugins" / "profile-and-optimize" / "skills"
SERVER_DIR = REPO_ROOT / "plugins" / "profile-and-optimize" / "server"
BUNDLED_DESCRIPTORS = REPO_ROOT / "mcp-descriptors"
MCP_CACHE_DEFAULT = Path.home() / ".cursor" / "projects" / "Users-<operator>-<project>" / "mcps"
# Order: env override -> operator's Cursor cache -> bundled descriptors.
# The bundle is the CI-friendly fallback (see mcp-descriptors/README.md).
_env = os.environ.get("PROFILE_AND_OPTIMIZE_MCP_CACHE")
if _env:
    MCP_CACHE = Path(_env)
elif MCP_CACHE_DEFAULT.exists():
    MCP_CACHE = MCP_CACHE_DEFAULT
else:
    MCP_CACHE = BUNDLED_DESCRIPTORS

# SKILL.md `mcp__<server>__*` -> Cursor MCP cache folder name.
# The Cursor MCP installer adds `user-` / `plugin-` prefixes to the server's
# canonical SKILL.md name.
SERVER_REMAP = {
    "prometheus_mcp": "user-prometheus-mcp",
    "profile_and_optimize": "user-profile_and_optimize",
    "zymtrace": "user-zymtrace",
}

def optional_servers() -> set[str]:
    """Servers that frontmatter `allowed-tools` MAY reference even when they
    are NOT declared in `.mcp.json`.

    Two sources:
      1. Servers declared in `.mcp.json` with the legacy `${VAR:-true}` no-op
         placeholder command (Claude Code honors the placeholder; Cursor
         doesn't, hence v1.5.0 trimmed them).
      2. Hardcoded operator-side-optional allowlist:
         - v1.5.1+: the 4 servers that v1.5.0 removed from `.mcp.json` for
           the Cursor-red-dot fix (`<node-diagnosis-tool>`, `<blast-radius-tool>`, `<vms-tool>`,
           `cursor_ide_browser`).
         - v1.9.0+: 6 additional external servers moved out of `.mcp.json`
           for the same Cursor-vs-Claude-Code placeholder asymmetry
           (`prometheus_mcp`, `<enterprise-search-mcp>`, `slack`, `sourcegraph`,
           `atlassian`, `google_workspace`). These have unguarded `${VAR}`
           placeholders with no `:-default` fallback; v1.9.0 docs the
           operator-side wiring snippet for each in the README.
         All operator-side-optional references are still legitimate because
         the skill body skip-gracefully behavior is the contract. Operators
         wire them via their own `~/.cursor/mcp.json` or
         `~/.claude/settings.json` per the "Operator-side optional MCPs"
         section of [`plugins/profile-and-optimize/README.md`](../plugins/profile-and-optimize/README.md).
    """
    # Hardcoded operator-side-optional allowlist (v1.5.1+ + v1.9.0+ + v1.15.0+)
    out: set[str] = {
        "<node-diagnosis-tool>",
        "<blast-radius-tool>",
        "<vms-tool>",
        "cursor_ide_browser",
        "prometheus_mcp",
        "<enterprise-search-mcp>",
        "slack",
        "sourcegraph",
        "atlassian",
        "google_workspace",
        # v1.15.0: added with the analyze-zymtrace-workload import. The
        # zymtrace MCP server is operator-side optional (declared in
        # ~/.cursor/mcp.json or ~/.claude/settings.json, NOT in the
        # plugin's .mcp.json). See plugin README "Operator-side optional
        # MCPs (not in `.mcp.json`)" for the env-var-driven setup.
        "zymtrace",
    }
    # Plus any `${VAR:-true}` placeholders still in .mcp.json (defense in depth)
    mcp_json = REPO_ROOT / "plugins" / "profile-and-optimize" / ".mcp.json"
    try:
        decl = json.loads(mcp_json.read_text())
        for name, spec in (decl.get("mcpServers") or {}).items():
            command = str(spec.get("command", ""))
            if ":-true}" in command:
                out.add(name)
    except (OSError, json.JSONDecodeError):
        pass
    return out


OPTIONAL_SERVERS = optional_servers()


def iter_skill_md_files() -> list[Path]:
    """Return sorted list of SKILL.md files, excluding the `_template/` scaffold.

    The `_template/SKILL.md` under `plugins/profile-and-optimize/skills/_template/` is the
    starter file operators copy when authoring a new skill (per the
    `plugins/profile-and-optimize/README.md` "Skill file shape" section); it is not itself
    a real skill. `bash scripts/validate-skill-prompt.sh` also excludes it,
    so filtering here keeps the two scripts agreeing on the skill count.
    """
    return sorted(
        p for p in SKILLS_DIR.glob("*/SKILL.md")
        if p.parent.name != "_template"
    )


def remap_server(server: str) -> Optional[str]:
    return SERVER_REMAP.get(server)


def list_profile_and_optimize_surface() -> Optional[set[str]]:
    """Use mcp_surface.py to enumerate the bundled profile_and_optimize MCP tool names."""
    surface_py = SERVER_DIR / "mcp_surface.py"
    if not surface_py.exists():
        return None
    venv_py = SERVER_DIR / ".venv" / "bin" / "python"
    py = str(venv_py) if venv_py.exists() else sys.executable
    try:
        out = subprocess.run(
            [py, str(surface_py), "list"],
            capture_output=True, text=True, check=True, cwd=str(SERVER_DIR),
        ).stdout
    except subprocess.CalledProcessError:
        return None
    names: set[str] = set()
    for line in out.splitlines():
        m = re.match(r"^\s+(\w+)\s+(read_only|writes_artifacts|submits_jobs|pulls_data|substitutes_nodes)", line)
        if m:
            names.add(m.group(1))
    # Known aux tools (advertised by the MCP server but not derived from CONTRACT
    # dicts; verified via `validate-mcp-tool-contract.sh` to be 75 total = 73 + 2).
    names.update({"search_runbooks", "search_evidence"})
    return names


def load_descriptor(server: str, tool: str) -> Optional[dict]:
    remapped = remap_server(server)
    if remapped is None:
        return None
    desc_path = MCP_CACHE / remapped / "tools" / f"{tool}.json"
    if not desc_path.exists():
        return None
    try:
        return json.loads(desc_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


WITH_BLOCK_RE = re.compile(
    r"^(?:- |# )?mcp__(?P<server>\w+)__(?P<tool>\w+)\s+with:\s*$",
    re.MULTILINE,
)


def extract_with_blocks(body: str) -> list[tuple[str, str, list[str], int]]:
    """Find every `mcp__<server>__<tool> with:` block; return its arg names + line.

    Block format (taken from current SKILL.md style):
        mcp__prometheus_mcp__query_prometheus with:
          datasourceUid: <uid>
          expr: <PromQL>
          queryType: instant

    Indented lines after the `with:` directive are treated as YAML-style
    `key: value` pairs until a blank line, a less-indented line, a fence,
    or another `mcp__` header.
    """
    out: list[tuple[str, str, list[str], int]] = []
    lines = body.splitlines()
    for idx, line in enumerate(lines):
        m = WITH_BLOCK_RE.match(line)
        if not m:
            continue
        server = m.group("server")
        tool = m.group("tool")
        args: list[str] = []
        for inner in lines[idx + 1 :]:
            stripped = inner.strip()
            if not stripped:
                break
            if stripped.startswith("```"):
                break
            if not inner.startswith((" ", "\t")):
                break
            # YAML key pattern: leading word followed by colon.
            km = re.match(r"^\s+(\w[\w_-]*)\s*:\s*", inner)
            if km:
                args.append(km.group(1))
            # Continuation lines (e.g. multi-line PromQL value) are skipped silently.
        out.append((server, tool, args, idx + 1))
    return out


def parse_frontmatter(text: str) -> dict:
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}
    fm_text = text[4:end]
    fm: dict = {}
    cur_key: Optional[str] = None
    for line in fm_text.splitlines():
        if not line.strip():
            continue
        if line.startswith("  - "):
            if cur_key:
                fm.setdefault(cur_key, []).append(line[4:].strip())
        else:
            m = re.match(r"^(\w[\w_-]*)\s*:\s*(.*)$", line)
            if m:
                cur_key = m.group(1)
                val = m.group(2).strip()
                if val:
                    fm[cur_key] = val
    return fm


def lint_skill(skill_md: Path, profile_and_optimize_surface: Optional[set[str]]) -> list[dict]:
    text = skill_md.read_text()
    fm = parse_frontmatter(text)
    body_start = text.find("\n---\n", 4)
    body = text[body_start + 5 :] if body_start > 0 else text
    skill_name = skill_md.parent.name

    findings: list[dict] = []
    allowed_tools = fm.get("allowed-tools", [])
    if isinstance(allowed_tools, str):
        allowed_tools = [allowed_tools]

    # 1. Validate frontmatter allowed-tools: every `mcp__<server>__<tool>`
    #    must map to a real descriptor or to a declared-optional server.
    for entry in allowed_tools:
        m = re.match(r"^mcp__(\w+)__(\w+)", entry.strip())
        if not m:
            continue
        server, tool = m.group(1), m.group(2)
        if server in OPTIONAL_SERVERS:
            # Server is declared in .mcp.json with the `${*:-true}` no-op
            # placeholder pattern. Frontmatter references are accepted
            # silently; the skill body's skip-gracefully behavior is the
            # contract.
            continue
        if server == "profile_and_optimize":
            if profile_and_optimize_surface is not None and tool not in profile_and_optimize_surface:
                findings.append({"severity": "RED", "skill": skill_name, "msg": f"frontmatter references mcp__profile_and_optimize__{tool}; tool not in bundled MCP surface (run `make mcp-surface` to inspect)"})
            continue
        desc = load_descriptor(server, tool)
        if desc is None:
            findings.append({"severity": "RED", "skill": skill_name, "msg": f"frontmatter references mcp__{server}__{tool}; descriptor JSON not found in {MCP_CACHE / (remap_server(server) or server) / 'tools' / (tool + '.json')}"})

    # 2. Validate body `with:` arg names against the descriptor schema.
    for server, tool, args, lineno in extract_with_blocks(body):
        if server in OPTIONAL_SERVERS:
            continue
        if server == "profile_and_optimize":
            # Bundled tools take an `args: [...]` list (or named kwargs) per CONTRACT.
            # Accept commonly-used keys without static checking; the runtime
            # validator on the server itself catches type errors.
            continue
        desc = load_descriptor(server, tool)
        if desc is None:
            continue  # frontmatter check already emitted a RED above
        props = (desc.get("arguments", {}) or {}).get("properties", {}) or {}
        valid_args = set(props.keys())
        for arg in args:
            if arg not in valid_args:
                findings.append({
                    "severity": "RED",
                    "skill": skill_name,
                    "line": lineno,
                    "msg": f"body line {lineno}: mcp__{server}__{tool} with: '{arg}'; valid args are {sorted(valid_args)}",
                })
    return findings


def main() -> int:
    if not MCP_CACHE.exists():
        print(f"FATAL: MCP descriptor folder not found at {MCP_CACHE}", file=sys.stderr)
        print(f"       Bundled fallback expected at {BUNDLED_DESCRIPTORS} (see mcp-descriptors/README.md).", file=sys.stderr)
        return 2
    if MCP_CACHE == BUNDLED_DESCRIPTORS:
        print(f"[info] using bundled MCP descriptors at {MCP_CACHE} (no Cursor cache found)")

    profile_and_optimize_surface = list_profile_and_optimize_surface()
    if profile_and_optimize_surface is None:
        print(f"WARN: could not load bundled profile_and_optimize MCP surface (run `bash {SERVER_DIR}/install.sh --with-dev`)", file=sys.stderr)

    red_total = 0
    warn_total = 0
    skill_md_files = iter_skill_md_files()
    for skill_md in skill_md_files:
        findings = lint_skill(skill_md, profile_and_optimize_surface)
        for f in findings:
            line = f.get("line", "")
            line_tag = f":{line}" if line else ""
            print(f"  [{f['severity']:4s}] {f['skill']}{line_tag}: {f['msg']}")
            if f["severity"] == "RED":
                red_total += 1
            elif f["severity"] == "WARN":
                warn_total += 1

    print(f"\n[summary] {red_total} RED, {warn_total} WARN across {len(skill_md_files)} SKILL.md files")
    return 1 if red_total > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
