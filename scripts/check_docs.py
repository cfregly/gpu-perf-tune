#!/usr/bin/env python3
"""Doc-correctness gate: keep audit-caught defects from coming back.

Reads .doccheck.json at the repo root and checks, over the configured docs:

  1. no stale/renamed repo name leaks back into a URL or command
  2. no unfilled placeholder in an install/usage snippet: an angle-bracket
     <your-org> stub, a pre-publication phrase ("once this repo has a remote"),
     or a draft marker (TODO / FIXME / TBD)
  3. every relative markdown link resolves to a real file
  4. (linters) the rule count the README states equals the count in code

It also scans every shipped .md/.txt in the repo for em/en dashes, the one
de-slop rule every repo's writing guide mandates (perf-tune's AGENTS.md and
METHODOLOGY.md: "No em-dashes"), over the whole tree rather than just the
configured docs.

Unlike the deslop prose linter, this gate scans code blocks too: install
commands are exactly where placeholders and stale names hide. A repo that
legitimately documents a marker (the deslop repo defines them) lists it under
"allow" in .doccheck.json. A doc that must show a dash lists itself under
"dash_exclude".

Exit 0 clean, 1 on any finding, 2 on setup error. Run from the repo root:

    python3 scripts/check_docs.py
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / ".doccheck.json"

LINK_RE = re.compile(r"(?<!\\)\[[^\]]+\]\((?P<url>[^)\s]+)\)")
FENCED_RE = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~")
PLACEHOLDER_RE = re.compile(
    r"<[^>\n]{0,60}?\b(your|insert|replace|placeholder|todo|org[-_ ]?name|"
    r"user[-_ ]?name|repo[-_ ]?name|api[-_ ]?key)\b[^>\n]{0,60}?>",
    re.I,
)
STALE_PHRASES = ("once this repo has a remote", "coming soon")
MARKER_RE = re.compile(r"\b(TODO|FIXME|TBD)\b")
DASH_RE = re.compile("[–—]")  # en dash (U+2013), em dash (U+2014)
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__",
             ".pytest_cache", ".mypy_cache", "dist", "build", "site-packages"}


def _line(text: str, idx: int) -> int:
    return text.count("\n", 0, idx) + 1


def load_config() -> dict:
    if not CONFIG.is_file():
        print(f"FATAL: {CONFIG.name} not found at repo root", file=sys.stderr)
        sys.exit(2)
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def check_doc(path: Path, cfg: dict) -> list[str]:
    rel = path.name
    if not path.is_file():
        return [f"{rel}: listed in .doccheck.json but not found"]
    text = path.read_text(encoding="utf-8")
    allow = {a.lower() for a in cfg.get("allow", [])}
    out: list[str] = []

    for old in cfg.get("old_names", []):
        for m in re.finditer(re.escape(old) + r"\b", text):
            out.append(f"{rel}:{_line(text, m.start())}: stale repo name "
                       f"'{old}' (renamed to '{cfg['repo']}')")

    for m in PLACEHOLDER_RE.finditer(text):
        if m.group(0).lower() in allow:
            continue
        out.append(f"{rel}:{_line(text, m.start())}: unfilled placeholder {m.group(0)!r}")
    low = text.lower()
    for phrase in STALE_PHRASES:
        if phrase in allow:
            continue
        idx = low.find(phrase)
        if idx >= 0:
            out.append(f"{rel}:{_line(text, idx)}: pre-publication phrase {phrase!r}")
    for m in MARKER_RE.finditer(text):
        if m.group(0).lower() in allow:
            continue
        out.append(f"{rel}:{_line(text, m.start())}: draft marker {m.group(0)!r}")

    # Relative-link resolution runs on a code-stripped copy so a link shown
    # inside a fenced example is not mistaken for a real link.
    nocode = FENCED_RE.sub(lambda mm: "\n" * mm.group(0).count("\n"), text)
    for m in LINK_RE.finditer(nocode):
        url = m.group("url")
        if url.startswith(("http://", "https://", "mailto:", "#", "<")):
            continue
        bare = url.split("#", 1)[0].split("?", 1)[0]
        if not bare:
            continue
        target = (ROOT / bare.lstrip("/")) if bare.startswith("/") else (path.parent / bare)
        if not target.exists():
            out.append(f"{rel}:{_line(nocode, m.start())}: broken relative link -> {url}")
    return out


def check_dashes(cfg: dict) -> list[str]:
    """No em/en dash may ship in any human-facing .md/.txt. Scans the whole
    tree (not just the configured docs), minus vendored/build dirs and any
    'dash_exclude' glob in .doccheck.json. Set 'dash_scan': false to disable."""
    if not cfg.get("dash_scan", True):
        return []
    excludes = cfg.get("dash_exclude", [])
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in sorted(filenames):
            if not fn.endswith((".md", ".txt")):
                continue
            path = Path(dirpath) / fn
            rel = path.relative_to(ROOT).as_posix()
            if any(fnmatch.fnmatch(rel, pat) for pat in excludes):
                continue
            text = path.read_text(encoding="utf-8")
            for m in DASH_RE.finditer(text):
                out.append(f"{rel}:{_line(text, m.start())}: em/en dash {m.group(0)!r}")
    return out


def check_rule_count(cfg: dict) -> list[str]:
    mod = cfg.get("rule_module")
    if not mod:
        return []
    prefix = cfg["rule_prefix"]
    ids = sorted(set(re.findall(rf"\b{prefix}\d+\b",
                                (ROOT / mod).read_text(encoding="utf-8"))))
    n = len(ids)
    readme = (ROOT / cfg.get("docs", ["README.md"])[0]).read_text(encoding="utf-8")
    out: list[str] = []
    if cfg.get("rule_check", "count") == "presence":
        for rid in ids:
            if rid not in readme:
                out.append(f"README does not document rule {rid} (defined in {mod})")
        return out
    stated: set[int] = set()
    for pat in cfg.get("rule_count_patterns", []):
        stated.update(int(x) for x in re.findall(pat, readme))
    if not stated:
        out.append(f"README states no rule count, code defines {n} ({prefix} rules)")
    for s in sorted(stated):
        if s != n:
            out.append(f"README says {s} rules but {mod} defines {n}: {', '.join(ids)}")
    return out


def main() -> int:
    cfg = load_config()
    findings: list[str] = []
    for d in cfg.get("docs", ["README.md"]):
        findings += check_doc(ROOT / d, cfg)
    findings += check_rule_count(cfg)
    findings += check_dashes(cfg)
    if not findings:
        print(f"[ok] check_docs: {cfg['repo']} docs are consistent")
        return 0
    print(f"[FAIL] check_docs found {len(findings)} issue(s):", file=sys.stderr)
    for f in findings:
        print(f"  {f}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
