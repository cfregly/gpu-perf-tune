#!/usr/bin/env python3
"""Assert every doc that names a tool / library / aux-tool count agrees with
the canonical constants in ``plugins/profile-and-optimize/server/mcp_surface.py``.

Background: the v1.13.0 rescan found severe drift between docs that claim
"73 tools" / "75-tool surface" / "84 + 2 = 86" / "90 contract tools" /
"92 total" / "95 contract" / "97 total" while the live derivation
produces 95 + 2 = 97. This lint reads the canonical constants from
``mcp_surface.py`` and fails any future commit that lets a doc drift
away from them.

How it works:

1. Import ``mcp_surface._TOTAL_CONTRACT_TOOLS`` / ``_TOTAL_AUX_TOOLS`` /
   ``_TOTAL_MCP_TOOLS`` / ``_TOTAL_LIBRARIES`` and call
   ``mcp_surface.verify_canonical_counts()`` to confirm they match the
   live derivation. Failure here means a bug in the bundled server,
   not in the docs.
2. For each doc in ``DOCS_TO_LINT``, scan for lines that name a tool
   count, library count, or aux-tool count. Accept the line only if
   the number matches the canonical constant.
3. Lines that name historical counts (intentional CHANGELOG history,
   "75 -> 97" migration notes) are exempt via ``LINE_EXEMPT_SUBSTRINGS``.

Exit codes:
  0 - clean (every doc agrees with the canonical constants).
  1 - >=1 doc disagrees.
  2 - fatal (cannot import mcp_surface, library count mismatch, etc.).

Run from the repo root:

    python3 scripts/lint-tool-counts.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_ROOT / "plugins" / "profile-and-optimize" / "server"

if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

try:
    import mcp_surface  # type: ignore[import-not-found]
except ImportError as exc:
    print(f"FATAL: could not import mcp_surface from {SERVER_DIR}: {exc}", file=sys.stderr)
    sys.exit(2)


DOCS_TO_LINT = (
    "README.md",
    "CONTRIBUTING.md",
    "REVIEWERS.md",
    "SECURITY.md",
    "Makefile",
    ".claude-plugin/marketplace.json",
    "plugins/profile-and-optimize/README.md",
    "plugins/profile-and-optimize/.claude-plugin/plugin.json",
    "plugins/profile-and-optimize/server/README.md",
    "plugins/profile-and-optimize/server/CLAUDE.md",
    "plugins/profile-and-optimize/server/pyproject.toml",
    "plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/README.md",
    "plugins/profile-and-optimize/server/docs/cli-contract.md",
    "plugins/profile-and-optimize/server/docs/mcp-tool-io-contract.md",
)


# Patterns that name a CONTRACT-TOOL count (the 95 number).
CONTRACT_TOOL_PATTERNS = (
    re.compile(r"\b(\d+)\s+contract[\s-]+(?:derived|driven|driving)\b", re.IGNORECASE),
    re.compile(r"\b(\d+)\s+contract\s+tools?\b", re.IGNORECASE),
    re.compile(r"\bderived\s+(\d+)\s+contract\b", re.IGNORECASE),
    re.compile(r"\bderives?\s+(?:the\s+)?(\d+)\s+(?:contract|MCP)\b", re.IGNORECASE),
    re.compile(r"\bderives?\s+(?:the\s+)?(\d+)\s+contract", re.IGNORECASE),
)

# Patterns that name a TOTAL-TOOL count (the 106 number).
TOTAL_TOOL_PATTERNS = (
    re.compile(r"\b(\d+)\s+MCP\s+tools?\s+total\b", re.IGNORECASE),
    re.compile(r"\btotal\s+(\d+)\s+MCP\s+tools?\b", re.IGNORECASE),
    re.compile(r"\b(\d+)\s+total\s+MCP\s+tools?\b", re.IGNORECASE),
    re.compile(r"\b(\d+)\s+MCP\s+verbs?\b", re.IGNORECASE),
    re.compile(r"\b(\d+)-tool\s+surface\b", re.IGNORECASE),
    re.compile(r"\b(\d+)\s+tools?\s+total\b", re.IGNORECASE),
    # `Bundled MCP tools:** 106`
    re.compile(r"\bBundled\s+MCP\s+tools?:\s*\*{0,2}\s*(\d+)\b"),
    # `MCP tools:** 106`
    re.compile(r"\bMCP\s+tools?:\s*\*{0,2}\s*(\d+)\b"),
    # "75 (73 + 2)" / "106 (104 + 2)"
    re.compile(r"\b(\d+)\s*\(\s*\d+\s*\+\s*2\s*\)"),
    # plugin.json description prose
    re.compile(r"\b(\d+)\s+tool[\s-]+surface\b", re.IGNORECASE),
    # "106 tools (104 contract-derived ..." / "106 MCP tools = 104 contract-derived ..."
    # — the dense-prose phrasing in README / plugin README / marketplace.json /
    # plugin.json / pyproject.toml that the old exemptions used to blind.
    re.compile(r"\b(\d+)\s+(?:MCP\s+)?tools?\s*[\(=]\s*\d+\s+contract", re.IGNORECASE),
    # "= 106 total)" (pyproject.toml description tail)
    re.compile(r"=\s*(\d+)\s+total\b", re.IGNORECASE),
)

# Patterns that name a LIBRARY count (the 22 number).
LIBRARY_PATTERNS = (
    re.compile(r"\b(\d+)\s+(?:stub\s+)?libraries\b", re.IGNORECASE),
    re.compile(r"\b(\d+)\s+CLI\s+(?:stub\s+)?(?:libraries|parsers)\b", re.IGNORECASE),
    re.compile(r"\b(\d+)\s+CLI\s+libraries\b", re.IGNORECASE),
    re.compile(r"\bacross\s+(\d+)\s+libraries\b", re.IGNORECASE),
    re.compile(r"\bspans\s+(\d+)\s+libraries\b", re.IGNORECASE),
)


# Exempt ONLY true migration-arrow captions ("75 -> 97", "15 -> 22"). The old
# blanket version-family exemptions ("v0.4" ... "v1.1") were removed in v1.29.0:
# they exempted any dense status/description line that merely *mentioned* a
# version, which is most of them — that is exactly how the 95/97/101/102/105
# count drift survived a passing `make lint-tool-counts`. Narrow exemptions keep
# the lint honest; a genuinely historical count must wear an explicit `N -> M`
# arrow (or live in CHANGELOG.md, which is out of scope) to be skipped.
LINE_EXEMPT_SUBSTRINGS = (
    # Historical migration prose ("75 -> 97").
    "-> 22",
    "-> 95",
    "-> 97",
    " -> 86",  # historical step
    " -> 90",  # historical step
    " -> 92",  # historical step
    "11 stub libraries +",  # historical "11 stub libraries + tools/" intro paragraph; v0.4 era
    # `(15 -> 22)` migration captions.
    "15 -> 22",
    "20 -> 22",
)


def lint_doc(path: Path, *, expected_contract: int, expected_total: int, expected_libs: int) -> list[str]:
    findings: list[str] = []
    if not path.is_file():
        return [f"{path}: NOT FOUND (still in DOCS_TO_LINT — remove or fix path)"]
    text = path.read_text()
    for line_no, line in enumerate(text.splitlines(), start=1):
        if any(sub in line for sub in LINE_EXEMPT_SUBSTRINGS):
            continue

        seen_pairs: set[tuple[str, int]] = set()

        def report(kind: str, num: int, expected: int) -> None:
            if num == expected:
                return
            key = (kind, num)
            if key in seen_pairs:
                return
            seen_pairs.add(key)
            findings.append(
                f"{path}:{line_no}: reports {num} {kind} (expected {expected}): "
                f"{line.strip()[:160]}"
            )

        for pat in CONTRACT_TOOL_PATTERNS:
            for m in pat.finditer(line):
                report("contract-tools", int(m.group(1)), expected_contract)
        for pat in TOTAL_TOOL_PATTERNS:
            for m in pat.finditer(line):
                report("MCP-tools-total", int(m.group(1)), expected_total)
        for pat in LIBRARY_PATTERNS:
            for m in pat.finditer(line):
                report("libraries", int(m.group(1)), expected_libs)

    return findings


def main() -> int:
    live = mcp_surface.verify_canonical_counts()
    expected_contract = mcp_surface._TOTAL_CONTRACT_TOOLS
    expected_total = mcp_surface._TOTAL_MCP_TOOLS
    expected_libs = mcp_surface._TOTAL_LIBRARIES
    print(
        f"[lint-tool-counts] canonical: contract_tools={expected_contract}, "
        f"aux_tools={mcp_surface._TOTAL_AUX_TOOLS}, total_mcp_tools={expected_total}, "
        f"libraries={expected_libs}"
    )
    print(
        f"[lint-tool-counts] live:      contract_tools={live['contract_tools']}, "
        f"total_mcp_tools={live['total_mcp_tools']}, libraries={live['libraries']}"
    )
    all_findings: list[str] = []
    for rel in DOCS_TO_LINT:
        all_findings.extend(
            lint_doc(
                REPO_ROOT / rel,
                expected_contract=expected_contract,
                expected_total=expected_total,
                expected_libs=expected_libs,
            )
        )
    if not all_findings:
        print(
            f"[ok] every doc in scope agrees with the canonical counts "
            f"({expected_contract} contract + {mcp_surface._TOTAL_AUX_TOOLS} aux = "
            f"{expected_total} MCP tools across {expected_libs} libraries)"
        )
        return 0
    print(f"[FAIL] {len(all_findings)} tool-count drift(s) found:", file=sys.stderr)
    for finding in all_findings:
        print(f"  {finding}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
