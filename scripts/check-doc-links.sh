#!/usr/bin/env bash
# Operator-facing doc link checker for profile-and-optimize.
#
# Walks the in-scope markdown set (profile-and-optimize-authored docs + vendored
# top-level READMEs), extracts every `[text](url|path)` link via a single
# embedded python pass, classifies each link as relative-path / http-url /
# mailto / anchor-only, and reports a per-file verdict.
#
# - Relative paths: assert the file or directory exists under repo root.
#   (Anchor fragments are stripped before existence checks.)
# - http(s) URLs: cheap HEAD with 10s timeout via `curl`. Skipped silently
#   when curl is missing or --no-network is passed (cluster login nodes
#   often have outbound HTTP blocked).
# - mailto: / `#anchor` / `${VAR}` / `<PLACEHOLDER>` links: skipped.
#
# Exit codes:
#   0 = green (no broken relative paths; HTTP failures are warnings if --no-network)
#   1 = red  (one or more broken relative paths or 4xx/5xx URLs)
#   2 = fatal (script setup error)
#
# Usage:
#   bash scripts/check-doc-links.sh                # full check, network on
#   bash scripts/check-doc-links.sh --no-network   # relative paths only (offline mode)
#   bash scripts/check-doc-links.sh --quiet        # only print red findings + final verdict
#   bash scripts/check-doc-links.sh --files PATTERN
#                                                  # restrict the scan to matching files

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

NO_NETWORK=0
QUIET=0
FILES_PATTERN=""

usage() {
  cat <<'EOF'
Usage: scripts/check-doc-links.sh [options]

Options:
  --no-network         Skip the http(s) HEAD checks; only validate relative paths.
  --quiet              Only print red findings + final verdict (not per-file ok lines).
  --files PATTERN      Restrict scan to filepaths matching the given pattern (rg --glob style).
  -h, --help           Show this help.

Returns exit 0 on green (no broken relative paths). HTTP 4xx/5xx return exit 1.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-network) NO_NETWORK=1; shift ;;
    --quiet) QUIET=1; shift ;;
    --files) FILES_PATTERN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown arg: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! command -v python3 >/dev/null 2>&1; then
  printf 'FATAL: python3 not on PATH\n' >&2
  exit 2
fi

# Run the link extraction + validation in a single embedded python program so
# we don't fork curl/test per link from bash (which is slow on N hundreds of
# links).
REPO_ROOT="${REPO_ROOT}" NO_NETWORK="${NO_NETWORK}" QUIET="${QUIET}" FILES_PATTERN="${FILES_PATTERN}" \
  python3 - <<'PYEOF'
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(os.environ["REPO_ROOT"])
NO_NETWORK = os.environ.get("NO_NETWORK", "0") == "1"
QUIET = os.environ.get("QUIET", "0") == "1"
FILES_PATTERN = os.environ.get("FILES_PATTERN", "")

# Scope: profile-and-optimize-authored docs + the 3 vendored runbooks + selected vendored READMEs.
# Exclude: per-operator artifact bundles (mutable evidence), per-operator learnings (Slack captures),
# .venv, __pycache__.
SCOPE_GLOBS = [
    # Top-level marketplace docs
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "CATALOG.md",
    "REVIEWERS.md",
    "SECURITY.md",
    # Marketplace docs subdir
    "docs/**/*.md",
    # Plugin-level docs
    "plugins/profile-and-optimize/README.md",
    "plugins/profile-and-optimize/server/README.md",
    "plugins/profile-and-optimize/server/AGENTS.md",
    "plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/README.md",
    # Skills
    "plugins/profile-and-optimize/skills/**/*.md",
    # GitHub templates
    ".github/**/*.md",
    # Vendored runbooks (3 files, operator-critical)
    "plugins/profile-and-optimize/server/runbooks/*.md",
    # Vendored top-level READMEs we cite
    "plugins/profile-and-optimize/server/tools/README.md",
    "plugins/profile-and-optimize/server/tuning/best-known/README.md",
    # Vendored docs CITED FROM profile-and-optimize-authored content (per repository convention)
    "plugins/profile-and-optimize/server/docs/mcp-composition.md",
    "plugins/profile-and-optimize/server/docs/mcp-tool-io-contract.md",
    "plugins/profile-and-optimize/server/docs/agent-onboarding.md",
    "plugins/profile-and-optimize/server/docs/grace-blackwell-deltas.md",
]
EXCLUDE_GLOBS = [
    "**/.venv/**",
    "**/__pycache__/**",
    "**/experiments/artifacts/**",   # per-operator mutable evidence
    "**/learnings/slack/**",          # verbatim Slack captures
]

# Legacy vendored content (snapshot of the original-seed surface).
# Broken links inside these files are reported as YELLOW (warn) instead of RED
# (fail) because they predate profile-and-optimize authorship and are tracked as legacy
# tech debt rather than blocking the gate. profile-and-optimize-authored docs are RED on
# the same finding.
VENDORED_GLOBS = [
    "plugins/profile-and-optimize/server/runbooks/**",
    "plugins/profile-and-optimize/server/docs/**",
    "plugins/profile-and-optimize/server/tools/**",
    "plugins/profile-and-optimize/server/tuning/**",
]


def is_vendored(p: Path) -> bool:
    rel = p.relative_to(REPO_ROOT).as_posix()
    for g in VENDORED_GLOBS:
        # Convert simple glob to a Path-match prefix.
        prefix = g.split("**", 1)[0].rstrip("/")
        if rel.startswith(prefix):
            return True
    return False

def collect_files() -> list[Path]:
    files: list[Path] = []
    seen = set()
    for g in SCOPE_GLOBS:
        for p in REPO_ROOT.glob(g):
            if not p.is_file():
                continue
            rel = p.relative_to(REPO_ROOT).as_posix()
            if FILES_PATTERN and FILES_PATTERN not in rel:
                continue
            if any(p.match(eg) for eg in EXCLUDE_GLOBS):
                continue
            if p in seen:
                continue
            seen.add(p)
            files.append(p)
    return sorted(files)

# Match [text](url-or-path). Be liberal with text; the url stops at unescaped ).
# Use a negative-lookbehind to skip backslash-escaped \[...\](...) patterns; CHANGELOG
# entries that quote literal bad-paths-that-were-fixed escape the brackets to opt out.
LINK_RE = re.compile(r"(?<!\\)\[(?P<text>[^\]]+)(?<!\\)\]\((?P<url>[^()\s]+(?:\([^()]*\)[^()\s]*)*)\)")
# Match the CODE-REFERENCE form ```startLine:endLine:filepath\n...```
CODE_REF_RE = re.compile(r"```(?P<start>\d+):(?P<end>\d+):(?P<path>[^\n`]+)\n", re.MULTILINE)
# Match fenced code blocks (```...``` and ~~~...~~~) and inline code spans
# (`code`) so we can strip them before scanning for [text](url) links.
# CHANGELOG entries that quote literal bad-paths-that-were-fixed AND inline
# regex patterns like `t[1-4]([a-z])` shouldn't show up as RED findings.
FENCED_BLOCK_RE = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~", re.MULTILINE)
# Inline code: single backticks (`...`) on one line. Greedy-minimal so we
# match the smallest span and don't accidentally swallow whole paragraphs.
INLINE_CODE_RE = re.compile(r"`[^`\n]+?`")


def strip_code_blocks(text: str) -> str:
    """Replace fenced code blocks AND inline code spans with whitespace so
    their content does not match LINK_RE. Preserve line numbers approximately
    by keeping newlines (fenced blocks span lines; inline code is one-line so
    a flat-replace is fine)."""
    def _drop_fenced(m: re.Match) -> str:
        return "\n" * m.group(0).count("\n")
    text = FENCED_BLOCK_RE.sub(_drop_fenced, text)
    # For inline code: replace with same-length spaces so the LINK_RE regex
    # can't match across the boundary either.
    text = INLINE_CODE_RE.sub(lambda m: " " * len(m.group(0)), text)
    return text

def classify(url: str) -> str:
    if url.startswith(("http://", "https://")):
        return "http"
    if url.startswith("mailto:"):
        return "mailto"
    if url.startswith("#"):
        return "anchor"
    if "${" in url or url.startswith("<") or "<" in url or url.startswith("$("):
        # placeholder / env-var template / template angle-bracket; skip
        return "placeholder"
    return "relative"

def check_relative(url: str, source: Path) -> tuple[bool, str]:
    # Strip anchor + querystring.
    bare = url.split("#", 1)[0].split("?", 1)[0]
    if not bare:
        return True, "anchor-only"
    # Resolve relative to source's parent.
    if bare.startswith("/"):
        target = REPO_ROOT / bare.lstrip("/")
    else:
        target = (source.parent / bare).resolve()
    if target.exists():
        return True, f"-> {target.relative_to(REPO_ROOT) if str(target).startswith(str(REPO_ROOT)) else target}"
    return False, f"not found: {target}"

# Cache HTTP results across the run.
HTTP_CACHE: dict[str, tuple[int, str]] = {}

# GitHub orgs whose repos return 404 to unauthenticated curl HEAD requests even
# though they exist (private visibility; validate independently via `gh api`).
# Links to these orgs are not failed; they're informational-only at HTTP-check
# time. Empty by default -- add your own org prefixes here if needed.
INTERNAL_ORG_PREFIXES: tuple[str, ...] = ()


def _curl_head_inner(url: str) -> tuple[int, str]:
    """Single-URL curl HEAD invocation. Designed to be thread-pool-friendly."""
    try:
        out = subprocess.run(
            ["curl", "-sSL", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "10", url],
            capture_output=True, text=True, timeout=15,
        )
        code_str = out.stdout.strip() or "0"
        code = int(code_str) if code_str.isdigit() else 0
        return code, f"HTTP {code}"
    except subprocess.TimeoutExpired:
        return 0, "TIMEOUT"
    except Exception as exc:  # noqa: BLE001
        return 0, f"err: {exc}"


def check_http(url: str) -> tuple[bool, str]:
    if NO_NETWORK:
        return True, "skipped (--no-network)"
    if url in HTTP_CACHE:
        code, msg = HTTP_CACHE[url]
        return code < 400, msg
    if url.startswith(INTERNAL_ORG_PREFIXES):
        # Internal repo / blob; unauth curl would return 404. Trust that the
        # org/repo was validated separately via `gh api`.
        HTTP_CACHE[url] = (200, "skipped (internal-visibility repo)")
        return True, "skipped (internal-visibility repo)"
    if not shutil_which("curl"):
        return True, "skipped (curl missing)"
    # Cache miss after the pre-pass means this URL wasn't seen in the prefetch
    # set (e.g. discovered later via a link that was inside a code fence at
    # extraction time but outside at validation time). Fall back to per-URL.
    code, msg = _curl_head_inner(url)
    HTTP_CACHE[url] = (code, msg)
    return code < 400, msg

def shutil_which(prog: str) -> str | None:
    import shutil
    return shutil.which(prog)

green_files = 0
red_files = 0       # profile-and-optimize-authored files with broken links
yellow_files = 0    # vendored files with broken links (don't fail the gate)
total_relative = 0
total_http = 0
broken_relative_red = 0
broken_relative_yellow = 0
broken_http = 0

per_file_findings: list[tuple[str, bool, list[str]]] = []   # (rel, vendored, findings)

# Pre-pass 1: collect every unique HTTP URL across all in-scope files so we
# can batch-fetch them in parallel. The per-file loop below then reads from
# HTTP_CACHE in O(1). On a 55-URL corpus this drops the wall-clock from ~7.6s
# (sequential) to ~1.5s (10-worker thread pool); the per-URL 10s timeout is
# unchanged.
files_for_prefetch = collect_files()
prefetch_urls: set[str] = set()
for f in files_for_prefetch:
    body = strip_code_blocks(f.read_text(errors="replace"))
    for m in LINK_RE.finditer(body):
        url = m.group("url").strip()
        if classify(url) == "http" and not url.startswith(INTERNAL_ORG_PREFIXES):
            prefetch_urls.add(url)

if not NO_NETWORK and prefetch_urls and shutil_which("curl"):
    if not QUIET:
        print(f"[prefetch] HEAD-checking {len(prefetch_urls)} unique HTTP URLs in parallel...")
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(_curl_head_inner, url): url for url in sorted(prefetch_urls)}
        for fut in as_completed(futures):
            url = futures[fut]
            try:
                code, msg = fut.result()
            except Exception as exc:  # noqa: BLE001
                code, msg = 0, f"err: {exc}"
            HTTP_CACHE[url] = (code, msg)

for f in collect_files():
    raw_text = f.read_text(errors="replace")
    text = strip_code_blocks(raw_text)
    findings: list[str] = []
    vendored = is_vendored(f)
    severity = "[YELLOW-PATH]" if vendored else "[RED-PATH]"
    http_severity = "[YELLOW-HTTP]" if vendored else "[RED-HTTP]"

    for m in LINK_RE.finditer(text):
        url = m.group("url").strip()
        kind = classify(url)
        if kind == "http":
            total_http += 1
            ok, msg = check_http(url)
            if not ok:
                broken_http += 1
                findings.append(f"  {http_severity}  {url}  ({msg})")
        elif kind == "relative":
            total_relative += 1
            ok, msg = check_relative(url, f)
            if not ok:
                if vendored:
                    broken_relative_yellow += 1
                else:
                    broken_relative_red += 1
                findings.append(f"  {severity}  {url}  ({msg})")
        # anchor / mailto / placeholder: skip silently

    for m in CODE_REF_RE.finditer(raw_text):
        cited = m.group("path").strip()
        total_relative += 1
        target = REPO_ROOT / cited if not cited.startswith("/") else Path(cited)
        if not target.exists():
            alt = (f.parent / cited).resolve()
            if alt.exists():
                continue
            if vendored:
                broken_relative_yellow += 1
            else:
                broken_relative_red += 1
            findings.append(f"  {severity} code-ref {cited}  (not found: {target})")

    rel = f.relative_to(REPO_ROOT).as_posix()
    if findings:
        if vendored:
            yellow_files += 1
        else:
            red_files += 1
        per_file_findings.append((rel, vendored, findings))
    else:
        green_files += 1
        if not QUIET:
            print(f"[ok]  {rel}")

red_only = [(rel, fs) for rel, v, fs in per_file_findings if not v]
yellow_only = [(rel, fs) for rel, v, fs in per_file_findings if v]

if red_only:
    print("\n=== RED findings (profile-and-optimize-authored docs; will fail the gate) ===")
    for rel, fs in red_only:
        print(f"\n{rel}")
        for ff in fs:
            print(ff)

if yellow_only:
    print("\n=== YELLOW findings (legacy vendored docs; pre-existing tech debt; do NOT fail the gate) ===")
    print("  These files predate profile-and-optimize authorship. Fix opportunistically when a related")
    print("  edit lands; bulk rewrites are out of scope for the doc-link gate.")
    for rel, fs in yellow_only:
        print(f"\n{rel}")
        for ff in fs:
            print(ff)

# v0.8.2 extension: cross-validate each SKILL.md `allowed-tools` mcp__<server>__*
# reference against the servers declared in plugins/profile-and-optimize/.mcp.json. Catches
# the case where a skill references an MCP server the marketplace doesn't bundle.
# Optional servers (<node-diagnosis-tool> / <blast-radius-tool> / <vms-tool> / cursor-ide-browser) are
# accepted as WARN since the .mcp.json placeholder-mode keeps them gracefully
# absent from non-equipped operator sessions.
import json as _json
MCP_JSON_PATH = REPO_ROOT / "plugins" / "profile-and-optimize" / ".mcp.json"
mcp_servers_red = 0
mcp_servers_warn = 0
mcp_server_findings: list[tuple[str, list[str]]] = []
if MCP_JSON_PATH.exists():
    try:
        mcp_decl = _json.loads(MCP_JSON_PATH.read_text())
        declared_servers = set(mcp_decl.get("mcpServers", {}).keys())
    except Exception as exc:  # noqa: BLE001
        declared_servers = set()
        print(f"[WARN] could not parse {MCP_JSON_PATH}: {exc}")
    # SKILL.md frontmatter `allowed-tools` -> server name mapping.
    # SKILL.md style: `mcp__<server>__<tool>` (server uses underscores, no `user-`/`plugin-` prefix).
    SKILL_FILES = list((REPO_ROOT / "plugins" / "profile-and-optimize" / "skills").glob("*/SKILL.md"))
    MCP_SERVER_RE = re.compile(r"mcp__([a-z_][a-z0-9_-]*)__")
    # Operator-side optional servers (per v1.5.1 + v1.9.0 README "Operator-side
    # optional MCPs (not in `.mcp.json`)"). Frontmatter `allowed-tools` may
    # reference these even when the server is not declared in `.mcp.json` —
    # operators wire them in their own ~/.cursor/mcp.json or
    # ~/.claude/settings.json. Mirror of the same allowlist in
    # scripts/lint-skill-mcp-args.py optional_servers().
    KNOWN_OPTIONAL = {
        "<node-diagnosis-tool>", "<blast-radius-tool>", "<vms-tool>",
        "cursor_ide_browser", "cursor-ide-browser",
        "prometheus_mcp", "<enterprise-search-mcp>", "slack",
        "sourcegraph", "atlassian", "google_workspace",
        # v1.15.0: added with the analyze-zymtrace-workload import. The
        # zymtrace MCP server is operator-side optional, declared in
        # ~/.cursor/mcp.json or ~/.claude/settings.json (NOT in the
        # plugin's .mcp.json). See plugin README "Operator-side optional
        # MCPs" for the env-var-driven setup.
        "zymtrace",
    }
    for skill_md in SKILL_FILES:
        text = skill_md.read_text()
        servers_referenced = {m.group(1) for m in MCP_SERVER_RE.finditer(text)}
        unknown = sorted(servers_referenced - declared_servers)
        sk_findings: list[str] = []
        for srv in unknown:
            severity = "WARN" if srv in KNOWN_OPTIONAL else "RED"
            if severity == "WARN":
                mcp_servers_warn += 1
                sk_findings.append(f"  [{severity}] mcp__{srv}__*  (server not declared in .mcp.json; accepted as optional)")
            else:
                mcp_servers_red += 1
                sk_findings.append(f"  [{severity}]  mcp__{srv}__*  (server not declared in .mcp.json; either declare or remove from allowed-tools)")
        if sk_findings:
            rel = skill_md.relative_to(REPO_ROOT).as_posix()
            mcp_server_findings.append((rel, sk_findings))

if mcp_server_findings:
    print("\n=== MCP server cross-check (allowed-tools vs .mcp.json) ===")
    for rel, fs in mcp_server_findings:
        print(f"\n{rel}")
        for ff in fs:
            print(ff)

print("\n=== Summary ===")
print(f"  Files scanned:           {green_files + red_files + yellow_files}")
print(f"  Files green:             {green_files}")
print(f"  Files yellow (vendored): {yellow_files}")
print(f"  Files red (authored):    {red_files}")
print(f"  Relative links checked:  {total_relative}")
print(f"  Relative links broken:   {broken_relative_red} red + {broken_relative_yellow} yellow")
print(f"  HTTP links checked:      {total_http}")
print(f"  HTTP links broken:       {broken_http}")
print(f"  MCP-server xref red:     {mcp_servers_red}")
print(f"  MCP-server xref warn:    {mcp_servers_warn}")
if NO_NETWORK:
    print(f"  (HTTP checks SKIPPED via --no-network)")

if broken_relative_red > 0 or broken_http > 0 or mcp_servers_red > 0:
    print(f"\n[FAIL] {broken_relative_red} broken relative paths + {broken_http} broken HTTP URLs + {mcp_servers_red} undeclared MCP servers in profile-and-optimize-authored docs")
    sys.exit(1)
print(f"\n[ok] no broken relative paths in profile-and-optimize-authored docs ({broken_relative_yellow} vendored YELLOWs accepted as known tech debt; {mcp_servers_warn} optional-MCP WARNs accepted)")
sys.exit(0)
PYEOF
