#!/usr/bin/env bash
# Operator-facing doc link checker for profile-and-optimize.
#
# Walks every Markdown file in the repository, extracts every `[text](url|path)` link via a single
# embedded python pass, classifies each link as relative-path / http-url /
# mailto / anchor-only, and reports a per-file verdict.
#
# - Relative paths: assert the file or directory exists from the source file.
# - Markdown anchors: assert the target heading exists, including anchor-only
#   links and fragments on relative Markdown paths.
# - Leading-slash paths: reject them because GitHub resolves them from the
#   website root, not the repository root.
# - Issue and pull request templates: require absolute links because GitHub
#   copies their Markdown into a new issue or pull request body.
# - http(s) URLs: lightweight request with a 10s timeout via `curl`. Skipped
#   when curl is missing or --no-network is passed (cluster login nodes
#   often have outbound HTTP blocked).
# - mailto / `${VAR}` / `<PLACEHOLDER>` links: skipped.
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
  --files PATTERN      Restrict the scan with a shell-style path glob.
  -h, --help           Show this help.

Returns exit 0 on green (no broken relative paths). HTTP 4xx/5xx return exit 1.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-network) NO_NETWORK=1; shift ;;
    --quiet) QUIET=1; shift ;;
    --files)
      if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
        printf 'missing value for --files\n' >&2
        usage >&2
        exit 2
      fi
      FILES_PATTERN="$2"
      shift 2
      ;;
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
import fnmatch
import html
import os
import re
import subprocess
import sys
import unicodedata
from urllib.parse import unquote
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(os.environ["REPO_ROOT"])
NO_NETWORK = os.environ.get("NO_NETWORK", "0") == "1"
QUIET = os.environ.get("QUIET", "0") == "1"
FILES_PATTERN = os.environ.get("FILES_PATTERN", "")

# Exclude generated environments and mutable operator evidence. Test path
# parts relative to the repository because Path.match on an absolute path does
# not reliably match a leading ``**/.venv/**`` pattern.
EXCLUDED_DIR_NAMES = {".git", ".venv", "__pycache__"}
EXCLUDED_PART_SEQUENCES = {
    ("experiments", "artifacts"),
    ("learnings", "slack"),
}


def contains_part_sequence(parts: tuple[str, ...], sequence: tuple[str, ...]) -> bool:
    width = len(sequence)
    return any(parts[index:index + width] == sequence for index in range(len(parts) - width + 1))

def collect_files() -> list[Path]:
    files: list[Path] = []
    for p in REPO_ROOT.rglob("*.md"):
        if not p.is_file():
            continue
        relative_path = p.relative_to(REPO_ROOT)
        rel = relative_path.as_posix()
        if FILES_PATTERN and not fnmatch.fnmatch(rel, FILES_PATTERN):
            continue
        if any(part in EXCLUDED_DIR_NAMES for part in relative_path.parts):
            continue
        if any(contains_part_sequence(relative_path.parts, seq) for seq in EXCLUDED_PART_SEQUENCES):
            continue
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

ANCHOR_CACHE: dict[Path, set[str]] = {}
HEADING_RE = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)
EXPLICIT_ANCHOR_RE = re.compile(
    r"<(?:a|span)[^>]+(?:id|name)=[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)


def github_heading_base(value: str) -> str:
    """Return the GitHub-style base slug for a Markdown heading."""
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    value = html.unescape(value).lower()
    value = re.sub(r"[`*_~]", "", value)
    value = "".join(
        char
        for char in value
        if char in {" ", "-", "_"}
        or unicodedata.category(char)[0] in {"L", "M", "N"}
    )
    return re.sub(r"\s+", "-", value.strip())


def markdown_anchors(target: Path) -> set[str]:
    cached = ANCHOR_CACHE.get(target)
    if cached is not None:
        return cached
    body = target.read_text(errors="replace")
    anchors = set(EXPLICIT_ANCHOR_RE.findall(body))
    duplicates: dict[str, int] = {}
    heading_body = FENCED_BLOCK_RE.sub(
        lambda match: "\n" * match.group(0).count("\n"),
        body,
    )
    for heading in HEADING_RE.findall(heading_body):
        base = github_heading_base(heading)
        suffix = duplicates.get(base, 0)
        anchors.add(base if suffix == 0 else f"{base}-{suffix}")
        duplicates[base] = suffix + 1
    ANCHOR_CACHE[target] = anchors
    return anchors


def check_relative(url: str, source: Path) -> tuple[bool, str]:
    path_part, separator, fragment = url.partition("#")
    bare = path_part.split("?", 1)[0]
    if bare.startswith("/"):
        return False, "leading-slash links resolve from github.com, not the repository"
    source_relative = source.relative_to(REPO_ROOT)
    copied_template = (
        source_relative == Path(".github/PULL_REQUEST_TEMPLATE.md")
        or source_relative.parts[:2] == (".github", "ISSUE_TEMPLATE")
    )
    if copied_template and bare:
        return False, "relative link is copied into an issue or pull request body"
    target = source if not bare else (source.parent / bare).resolve()
    if not target.exists():
        return False, f"not found: {target}"
    if separator and fragment and target.suffix.lower() in {".md", ".markdown"}:
        decoded_fragment = unquote(fragment)
        if decoded_fragment not in markdown_anchors(target):
            relative_target = (
                target.relative_to(REPO_ROOT)
                if str(target).startswith(str(REPO_ROOT))
                else target
            )
            return False, f"heading not found: {relative_target}#{decoded_fragment}"
    relative_target = (
        target.relative_to(REPO_ROOT)
        if str(target).startswith(str(REPO_ROOT))
        else target
    )
    return True, f"-> {relative_target}"

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
        return 200 <= code < 400, msg
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
    return 200 <= code < 400, msg

def shutil_which(prog: str) -> str | None:
    import shutil
    return shutil.which(prog)

green_files = 0
red_files = 0
total_relative = 0
total_http = 0
broken_relative = 0
broken_http = 0

per_file_findings: list[tuple[str, list[str]]] = []

# Pre-pass 1: collect every unique HTTP URL across all in-scope files so we
# can fetch them in parallel. The per-file loop below then reads from
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
        print(f"[prefetch] checking {len(prefetch_urls)} unique HTTP URLs in parallel...")
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
    severity = "[RED-PATH]"
    http_severity = "[RED-HTTP]"

    for m in LINK_RE.finditer(text):
        url = m.group("url").strip()
        kind = classify(url)
        if kind == "http":
            total_http += 1
            ok, msg = check_http(url)
            if not ok:
                broken_http += 1
                findings.append(f"  {http_severity}  {url}  ({msg})")
        elif kind in {"relative", "anchor"}:
            total_relative += 1
            ok, msg = check_relative(url, f)
            if not ok:
                broken_relative += 1
                findings.append(f"  {severity}  {url}  ({msg})")
        # mailto / placeholder: skip silently

    for m in CODE_REF_RE.finditer(raw_text):
        cited = m.group("path").strip()
        total_relative += 1
        target = REPO_ROOT / cited if not cited.startswith("/") else Path(cited)
        if not target.exists():
            alt = (f.parent / cited).resolve()
            if alt.exists():
                continue
            broken_relative += 1
            findings.append(f"  {severity} code-ref {cited}  (not found: {target})")

    rel = f.relative_to(REPO_ROOT).as_posix()
    if findings:
        red_files += 1
        per_file_findings.append((rel, findings))
    else:
        green_files += 1
        if not QUIET:
            print(f"[ok]  {rel}")

if per_file_findings:
    print("\n=== Broken links ===")
    for rel, fs in per_file_findings:
        print(f"\n{rel}")
        for ff in fs:
            print(ff)

# Cross-check each SKILL.md `allowed-tools` MCP reference against the bundled
# server manifest. Operator-provided servers documented in the plugin README are
# accepted as warnings.
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
    # Frontmatter may reference operator-provided servers that are not declared
    # in `.mcp.json`. Mirror the current allowlist in
    # scripts/lint-skill-mcp-args.py optional_servers().
    KNOWN_OPTIONAL = {
        "prometheus_mcp", "zymtrace",
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
print(f"  Files scanned:           {green_files + red_files}")
print(f"  Files green:             {green_files}")
print(f"  Files red:               {red_files}")
print(f"  Relative links checked:  {total_relative}")
print(f"  Relative links broken:   {broken_relative}")
print(f"  HTTP links checked:      {total_http}")
print(f"  HTTP links broken:       {broken_http}")
print(f"  MCP-server xref red:     {mcp_servers_red}")
print(f"  MCP-server xref warn:    {mcp_servers_warn}")
if NO_NETWORK:
    print(f"  (HTTP checks SKIPPED via --no-network)")

if broken_relative > 0 or broken_http > 0 or mcp_servers_red > 0:
    print(f"\n[FAIL] {broken_relative} broken relative paths + {broken_http} broken HTTP URLs + {mcp_servers_red} undeclared MCP servers")
    sys.exit(1)
print(f"\n[ok] no broken links ({mcp_servers_warn} optional-MCP warnings accepted)")
sys.exit(0)
PYEOF
