"""Findings library: record + render + diff verbs.

A "finding" is one row in a yaml-structured list at `<bundle>/findings.yaml`,
following the schema documented in `docs/findings-schema.md`.

Verbs:
  record: append a finding to a bundle's findings.yaml (writes_artifacts).
  render: read findings.yaml + emit a presentable findings.md (read_only).
  diff:   compare two findings.yaml files + emit a markdown drift report (read_only).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# yaml is required for round-tripping; install via the bundled server's `dev` extras.
try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


VALID_SEVERITIES = ("critical", "high", "medium", "low", "informational")
VALID_STATUSES = ("open", "in_progress", "resolved")


def _require_yaml() -> None:
    if yaml is None:
        print("FATAL: PyYAML not available. Install via `bash plugins/profile-and-optimize/server/install.sh --with-dev`.", file=sys.stderr)
        sys.exit(2)


def _load_findings(path: Path) -> dict[str, Any]:
    _require_yaml()
    if not path.exists():
        return {"findings": []}
    raw = path.read_text()
    if not raw.strip():
        return {"findings": []}
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        data = {"findings": []}
    data.setdefault("findings", [])
    return data


def _dump_findings(path: Path, data: dict[str, Any]) -> None:
    _require_yaml()
    path.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))


def _validate_finding(finding: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = ("id", "severity", "source_skill", "source_query", "headline", "recommended_action")
    for k in required:
        if k not in finding or not finding[k]:
            errors.append(f"missing required field: {k}")
    sev = finding.get("severity")
    if sev not in VALID_SEVERITIES:
        errors.append(f"severity must be one of {VALID_SEVERITIES}; got {sev!r}")
    st = finding.get("status", "open")
    if st not in VALID_STATUSES:
        errors.append(f"status must be one of {VALID_STATUSES}; got {st!r}")
    return errors


def _record(args: argparse.Namespace) -> int:
    _require_yaml()
    path = Path(args.findings_yaml)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _load_findings(path)
    finding: dict[str, Any] = {
        "id": args.id,
        "severity": args.severity,
        "source_skill": args.source_skill,
        "source_query": args.source_query,
        "headline": args.headline,
        "recommended_action": args.recommended_action,
        "status": args.status or "open",
    }
    if args.evidence_path:
        finding["evidence_path"] = args.evidence_path
    if args.affected_entity:
        finding["affected_entities"] = []
        for ent in args.affected_entity:
            kind, value = ent.split("=", 1)
            finding["affected_entities"].append({"kind": kind, "value": value})
    if args.notes:
        finding["notes"] = args.notes
    finding["detected_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    errors = _validate_finding(finding)
    if errors:
        for e in errors:
            print(f"  validation error: {e}", file=sys.stderr)
        return 1

    # Replace if same id; else append.
    existing_ids = {f.get("id") for f in data["findings"]}
    if finding["id"] in existing_ids:
        data["findings"] = [f for f in data["findings"] if f.get("id") != finding["id"]]
    data["findings"].append(finding)
    _dump_findings(path, data)

    result = {
        "library": "findings",
        "verb": "record",
        "safety": "writes_artifacts",
        "findings_yaml": str(path),
        "total_findings": len(data["findings"]),
        "recorded_id": finding["id"],
        "severity": finding["severity"],
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for k, v in result.items():
            print(f"  {k}: {v}")
    return 0


SEVERITY_TO_HEADER = {
    "critical": ("C", "Critical (drop everything)"),
    "high": ("H", "High (significant ops impact)"),
    "medium": ("M", "Medium (worth tracking)"),
    "low": ("L", "Low (informational)"),
    "informational": ("L", "Low (informational)"),
}


def _render(args: argparse.Namespace) -> int:
    data = _load_findings(Path(args.findings_yaml))
    sections: dict[str, list[dict[str, Any]]] = {s: [] for s in ("critical", "high", "medium", "low", "informational")}
    for f in data["findings"]:
        sections.setdefault(f.get("severity", "informational"), []).append(f)

    lines: list[str] = ["# Findings\n"]
    counters: dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}

    def _flush(severity: str) -> None:
        rows = sections.get(severity, [])
        if not rows:
            return
        prefix, label = SEVERITY_TO_HEADER[severity]
        lines.append(f"## {label}\n")
        lines.append("| # | Finding | Source | Action | Status |")
        lines.append("| --- | --- | --- | --- | --- |")
        bucket = "low" if severity == "informational" else severity
        for f in rows:
            counters[bucket] += 1
            num = f"{prefix}{counters[bucket]}"
            src = f"{f.get('source_skill', '?')} / {f.get('source_query', '?')}"
            lines.append(f"| {num} | {f.get('headline', '?')} | {src} | {f.get('recommended_action', '?')} | {f.get('status', 'open')} |")
        lines.append("")

    for sev in ("critical", "high", "medium", "low", "informational"):
        _flush(sev)

    md = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(md)
        if not args.json:
            print(f"  wrote {args.out} ({sum(counters.values())} findings)")
    else:
        if not args.json:
            print(md)
    result = {
        "library": "findings",
        "verb": "render",
        "safety": "read_only",
        "findings_yaml": args.findings_yaml,
        "rendered_to": args.out or "(stdout)",
        "counts": counters,
    }
    if args.json:
        print(json.dumps(result, indent=2))
    return 0


def _diff(args: argparse.Namespace) -> int:
    a_data = _load_findings(Path(args.baseline))
    b_data = _load_findings(Path(args.current))
    by_id_a = {f.get("id"): f for f in a_data["findings"]}
    by_id_b = {f.get("id"): f for f in b_data["findings"]}
    new_ids = sorted(set(by_id_b) - set(by_id_a))
    resolved_ids = sorted(set(by_id_a) - set(by_id_b))
    status_changes: list[tuple[str, str, str]] = []
    for fid in sorted(set(by_id_a) & set(by_id_b)):
        sa = by_id_a[fid].get("status", "open")
        sb = by_id_b[fid].get("status", "open")
        if sa != sb:
            status_changes.append((fid, sa, sb))

    lines = [f"## Findings diff: {args.baseline} -> {args.current}\n"]
    if new_ids:
        lines.append("### New findings (in current, not in baseline)\n")
        for fid in new_ids:
            f = by_id_b[fid]
            lines.append(f"- **{fid}** ({f.get('severity', '?')}): {f.get('headline', '?')}")
        lines.append("")
    if resolved_ids:
        lines.append("### Resolved findings (in baseline, not in current)\n")
        for fid in resolved_ids:
            f = by_id_a[fid]
            lines.append(f"- **{fid}** ({f.get('severity', '?')}): was {f.get('headline', '?')}")
        lines.append("")
    if status_changes:
        lines.append("### Status changes\n")
        for fid, sa, sb in status_changes:
            lines.append(f"- **{fid}**: {sa} -> {sb}")
        lines.append("")
    if not (new_ids or resolved_ids or status_changes):
        lines.append("_No drift between baseline and current findings._\n")

    md = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(md)
    else:
        print(md)
    result = {
        "library": "findings",
        "verb": "diff",
        "safety": "read_only",
        "baseline": args.baseline,
        "current": args.current,
        "new_count": len(new_ids),
        "resolved_count": len(resolved_ids),
        "status_change_count": len(status_changes),
    }
    if args.json:
        print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="findings", description="Structured findings library: record + render + diff.")
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="Append a finding to a bundle's findings.yaml.", description="Append a finding to a bundle's findings.yaml.")
    rec.add_argument("--findings-yaml", required=True, help="Path to the findings.yaml file (will be created if missing)")
    rec.add_argument("--id", required=True)
    rec.add_argument("--severity", required=True, choices=VALID_SEVERITIES)
    rec.add_argument("--source-skill", required=True)
    rec.add_argument("--source-query", required=True)
    rec.add_argument("--headline", required=True)
    rec.add_argument("--recommended-action", required=True)
    rec.add_argument("--status", choices=VALID_STATUSES, default="open")
    rec.add_argument("--evidence-path", default=None)
    rec.add_argument("--affected-entity", action="append", default=None, metavar="KIND=VALUE", help="May be passed multiple times (e.g. --affected-entity zone=<ZONE>)")
    rec.add_argument("--notes", default=None)
    rec.add_argument("--json", action="store_true")
    rec.set_defaults(func=_record)

    rdr = sub.add_parser("render", help="Convert findings.yaml -> findings.md.", description="Convert findings.yaml -> findings.md (table grouped by severity).")
    rdr.add_argument("--findings-yaml", required=True)
    rdr.add_argument("--out", default=None, help="Write markdown to this path; default stdout")
    rdr.add_argument("--json", action="store_true")
    rdr.set_defaults(func=_render)

    dff = sub.add_parser("diff", help="Compare two findings.yaml files + emit drift report.", description="Compare two findings.yaml files and emit a markdown drift report.")
    dff.add_argument("--baseline", required=True)
    dff.add_argument("--current", required=True)
    dff.add_argument("--out", default=None, help="Write markdown to this path; default stdout")
    dff.add_argument("--json", action="store_true")
    dff.set_defaults(func=_diff)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.func(args)


CONTRACT: dict[str, dict] = {
    "record": {
        "safety": "writes_artifacts",
        "required": ("--findings-yaml", "--id", "--severity", "--source-skill", "--source-query", "--headline", "--recommended-action"),
        "optional": ("--status", "--evidence-path", "--affected-entity", "--notes", "--json"),
        "json": True,
        "ack": None,
        "description": "Append a structured finding (one row in findings.yaml) to a bundle.",
    },
    "render": {
        "safety": "read_only",
        "required": ("--findings-yaml",),
        "optional": ("--out", "--json"),
        "json": True,
        "ack": None,
        "description": "Convert findings.yaml to a presentable findings.md table grouped by severity.",
    },
    "diff": {
        "safety": "read_only",
        "required": ("--baseline", "--current"),
        "optional": ("--out", "--json"),
        "json": True,
        "ack": None,
        "description": "Compare two findings.yaml files and emit a markdown drift report (new / resolved / status-changed).",
    },
}


if __name__ == "__main__":
    sys.exit(main())
