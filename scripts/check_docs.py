#!/usr/bin/env python3
"""Doc-correctness gate: keep audit-caught defects from coming back.

Reads .doccheck.json at the repo root and checks, over the configured docs:

  1. no stale/renamed repo name leaks back into a URL or command
  2. no unfilled placeholder in an install/usage snippet: an angle-bracket
     <your-org> stub, a pre-publication phrase ("once this repo has a remote"),
     or a draft marker (TODO / FIXME / TBD)
  3. every relative markdown link resolves to a real file
  4. (linters) the rule count the README states equals the count in code
  5. (scores) every marquee score the README documents as a runnable command is
     re-run and must still print that score. A command of the form
     `python -m <module> <path> ... # ... NN/100` (or a bare `python <script>.py
     ...`, denominator 100 or 10) is executed and its printed score is compared
     to the one in the comment, so a rule change that moves a score cannot ship
     while the README still advertises the old number. Opt in per repo with
     "score_modules" in .doccheck.json (the allowlist of scorers the gate may
     run); absent that key the check is a no-op.

It also scans every shipped .md/.txt in the repo for em/en dashes and prose
semicolons, the de-slop rules every repo's writing guide mandates, over the
whole tree rather than just the configured docs. Code spans, inline code,
links, and entities are exempt from the semicolon scan, so a semicolon in a
shell command or URL is left alone.

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
import shlex
import subprocess
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
# Spans where a semicolon is legitimate (code, inline code, links, entities).
# Blanked before the prose-semicolon scan, newlines kept so line numbers hold.
CODE_SPAN = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`|"
                       r"\]\([^)]*\)|https?://[^\s)]+|&[#a-zA-Z0-9]+;")


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


def check_semicolons(cfg: dict) -> list[str]:
    """No semicolon may ship in prose (an AI-writing tell the user bans). Scans
    every .md/.txt, blanking code spans, inline code, links, and entities first
    so a semicolon in a shell command or URL is left alone. Set
    'semicolon_scan': false to disable, or list a file under 'dash_exclude'."""
    if not cfg.get("semicolon_scan", True):
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
            prose = CODE_SPAN.sub(lambda m: "\n" * m.group(0).count("\n"), text)
            for m in re.finditer(";", prose):
                out.append(f"{rel}:{_line(prose, m.start())}: prose semicolon")
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


def _score_cmd_re(modules: list[str]) -> re.Pattern:
    """Regex for a README command that documents its own score. Only the
    scorers named in 'score_modules' are matchable, so the gate never runs an
    arbitrary documented command. Handles `python -m <module>` and a bare
    `python <script>.py`, with a trailing `# ... NN/100` or `# ... NN/10`."""
    alt = "|".join(re.escape(m) for m in modules)
    return re.compile(
        r"(?P<full>python3? +(?:-m +)?(?:" + alt + r")\b[^\n#]*?)"
        r" *#[^\n]*?(?P<score>\d+) */ *(?P<denom>100|10)\b")


def check_score_claims(cfg: dict) -> list[str]:
    """Re-run every README command that documents its own score and fail if the
    live score drifted from the printed one. This is the gate the naive harness
    '39' silently becoming '23' slipped past: the rule COUNT was checked, the
    example SCORE was not. Driven by 'score_modules' in .doccheck.json (a no-op
    when that key is absent). Set 'score_scan': false to disable."""
    modules = cfg.get("score_modules") or []
    if not modules or not cfg.get("score_scan", True):
        return []
    pattern = _score_cmd_re(modules)
    out: list[str] = []
    for d in cfg.get("docs", ["README.md"]):
        path = ROOT / d
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for m in pattern.finditer(text):
            full = m.group("full")
            claimed, denom = int(m.group("score")), m.group("denom")
            argv = shlex.split(full)
            if not argv:
                continue
            argv = [sys.executable] + argv[1:]
            try:
                proc = subprocess.run(argv, cwd=ROOT, capture_output=True,
                                      text=True, timeout=300)
            except Exception as exc:  # noqa: BLE001
                out.append(f"{d}:{_line(text, m.start())}: '{full}' did not run "
                           f"({type(exc).__name__})")
                continue
            sm = re.search(rf"(\d+) */ *{denom}\b", proc.stdout)
            if not sm:
                err = (proc.stderr.strip().splitlines()[-1:] or [""])[0]
                out.append(f"{d}:{_line(text, m.start())}: '{full}' documents "
                           f"{claimed}/{denom} but printed no score ({err[:80]})")
                continue
            actual = int(sm.group(1))
            if actual != claimed:
                out.append(f"{d}:{_line(text, m.start())}: '{full}' documents "
                           f"{claimed}/{denom} but the command scores {actual}/{denom}")
    return out


def main() -> int:
    cfg = load_config()
    findings: list[str] = []
    for d in cfg.get("docs", ["README.md"]):
        findings += check_doc(ROOT / d, cfg)
    findings += check_rule_count(cfg)
    findings += check_score_claims(cfg)
    findings += check_dashes(cfg)
    findings += check_semicolons(cfg)
    if not findings:
        print(f"[ok] check_docs: {cfg['repo']} docs are consistent")
        return 0
    print(f"[FAIL] check_docs found {len(findings)} issue(s):", file=sys.stderr)
    for f in findings:
        print(f"  {f}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
