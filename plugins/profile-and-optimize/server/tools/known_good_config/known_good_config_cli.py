"""Known-good config registry: record + check.

Per-model REQUIRED serve flags + champion + the bug each flag avoids, so a
hard-won workaround (e.g. Qwen3-Next's ``gdn_prefill_backend=triton``) is NEVER
re-discovered the hard way. These are NOT well-known upstream defaults; several
are operator-discovered workarounds where the upstream ``auto`` path is actively
broken on specific hardware. Treat the registry as private to your org --
entries can encode unpublished bug workarounds, so never post an entry publicly.

Two verbs:

- ``record`` appends a NEW model entry to the registry YAML, comment-preserving
  (the registry is hand-curated; an existing model fails with guidance to edit
  by hand -- PyYAML round-trip would strip the LOUD banner + per-entry prose).
- ``check`` verifies a deploy's serve args contain every ``required_flag``'s
  ``match`` regex for a model and is **fail-closed** (nonzero exit) on a missing
  boot-blocker / crash-high-c / deploy-correctness flag. ``--require-registered``
  makes an unregistered model a failure (used by the grind-closure gate so a
  champion is not "closed" until its known-good config is captured here).

Registry: ``perf-tune-report/configs/known-good-configs.yaml`` (schema
``known_good_config_v1``) in the INFERENCE workspace. Resolved via ``--registry``,
then ``$KNOWN_GOOD_CONFIG_REGISTRY``, then a walk-up search from cwd.

Backs the [`inference-known-good-config`](../../skills/inference-known-good-config/SKILL.md)
skill. Added in profile-and-optimize v1.68.0.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

import yaml

CONTRACT: dict[str, dict[str, Any]] = {
    "record": {
        "safety": "writes_artifacts",
        "required": ("--model",),
        "optional": (
            "--registry", "--slug", "--arch", "--hardware", "--engine",
            "--required-flag", "--champion-config-ref", "--champion-verdict",
            "--champion-campaign", "--fallback", "--grind-frontier", "--notes", "--json",
        ),
        "json": True,
        "ack": None,
        "description": "Append a new model entry to the known-good config registry (comment-preserving).",
    },
    "check": {
        "safety": "read_only",
        "required": ("--model",),
        "optional": (
            "--registry", "--serve-args", "--deploy-file", "--require-registered", "--json",
        ),
        "json": True,
        "ack": None,
        "description": "Check a deploy's serve args contain every required flag for a model (fail-closed on a missing boot-blocker).",
    },
}

#: A missing required flag at one of these severities is a HARD failure
#: (nonzero exit). A missing ``perf`` flag is a warning only.
FAIL_SEVERITIES = {"boot-blocker", "crash-high-c", "deploy-correctness"}

REGISTRY_RELPATH = Path("perf-tune-report") / "configs" / "known-good-configs.yaml"


def _resolve_registry(arg: str | None) -> Path:
    if arg:
        return Path(arg).expanduser().resolve()
    env = os.environ.get("KNOWN_GOOD_CONFIG_REGISTRY")
    if env:
        return Path(env).expanduser().resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        candidate = current / REGISTRY_RELPATH
        if candidate.is_file():
            return candidate.resolve()
        current = current.parent
    raise SystemExit(
        "FATAL: cannot resolve the known-good config registry; pass --registry "
        "or set KNOWN_GOOD_CONFIG_REGISTRY (expected perf-tune-report/configs/known-good-configs.yaml)"
    )


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"FATAL: registry not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict) or "models" not in data:
        raise SystemExit(f"FATAL: not a known_good_config registry (no `models:`): {path}")
    if not isinstance(data.get("models"), list):
        raise SystemExit(f"FATAL: registry `models:` is not a list: {path}")
    return data


def _find_model(registry: dict[str, Any], model: str) -> dict[str, Any] | None:
    for entry in registry["models"]:
        if isinstance(entry, dict) and entry.get("model") == model:
            return entry
    return None


def _emit(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for k, v in sorted(payload.items()):
            print(f"{k}: {v}")


def _parse_required_flag(raw: str) -> dict[str, str]:
    # Pipe-delimited: flag|match|severity|why|affected|evidence (trailing fields optional).
    parts = [p.strip() for p in raw.split("|")]
    keys = ("flag", "match", "severity", "why", "affected", "evidence")
    out: dict[str, str] = {}
    for key, val in zip(keys, parts):
        if val:
            out[key] = val
    if "flag" not in out:
        raise SystemExit(f"FATAL: --required-flag needs at least a flag: {raw!r}")
    out.setdefault("match", re.escape(out["flag"]))
    out.setdefault("severity", "boot-blocker")
    return out


def cmd_record(args: argparse.Namespace) -> int:
    registry_path = _resolve_registry(args.registry)
    registry = _load_registry(registry_path)

    if _find_model(registry, args.model) is not None:
        print(
            f"FATAL: model already registered: {args.model}\n"
            f"  The registry is hand-curated; edit {registry_path} by hand to UPDATE an "
            f"existing entry (a programmatic rewrite would strip the LOUD banner + prose).",
            file=sys.stderr,
        )
        return 2

    entry: dict[str, Any] = {"model": args.model}
    for field_name in ("slug", "arch", "hardware", "engine"):
        val = getattr(args, field_name)
        if val:
            entry[field_name] = val
    required_flags = [_parse_required_flag(r) for r in (args.required_flag or [])]
    entry["required_flags"] = required_flags
    champion: dict[str, str] = {}
    if args.champion_config_ref:
        champion["config_ref"] = args.champion_config_ref
    if args.champion_verdict:
        champion["verdict"] = args.champion_verdict
    if args.champion_campaign:
        champion["campaign"] = args.champion_campaign
    if champion:
        entry["champion"] = champion
    if args.fallback:
        entry["fallback"] = args.fallback
    if args.grind_frontier:
        entry["grind_frontier"] = args.grind_frontier
    if args.notes:
        entry["notes"] = args.notes

    # Comment-preserving append: render the entry as a YAML list block and append
    # at EOF (the `models:` list is the last top-level key in the registry).
    block = yaml.safe_dump([entry], default_flow_style=False, sort_keys=False, allow_unicode=True)
    block = textwrap.indent(block, "  ")
    text = registry_path.read_text()
    if not text.endswith("\n"):
        text += "\n"
    registry_path.write_text(text + block)

    payload = {
        "tool": "known_good_config_record",
        "library": "known_good_config",
        "verb": "record",
        "safety": CONTRACT["record"]["safety"],
        "registry": str(registry_path),
        "model": args.model,
        "required_flags": len(required_flags),
    }
    _emit(payload, as_json=args.json)
    return 0


def _scan_text(args: argparse.Namespace) -> str | None:
    if args.serve_args is not None:
        return args.serve_args
    if args.deploy_file:
        p = Path(args.deploy_file).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"FATAL: --deploy-file not found: {p}")
        return p.read_text()
    return None


def cmd_check(args: argparse.Namespace) -> int:
    registry_path = _resolve_registry(args.registry)
    registry = _load_registry(registry_path)
    entry = _find_model(registry, args.model)

    if entry is None:
        registered = False
        verdict = "fail" if args.require_registered else "pass"
        payload = {
            "tool": "known_good_config_check",
            "library": "known_good_config",
            "verb": "check",
            "safety": CONTRACT["check"]["safety"],
            "registry": str(registry_path),
            "model": args.model,
            "registered": registered,
            "checked_args": False,
            "missing_required": [],
            "verdict": verdict,
            "reason": "model_not_registered",
        }
        _emit(payload, as_json=args.json)
        return 1 if verdict == "fail" else 0

    text = _scan_text(args)
    missing: list[dict[str, str]] = []
    checked_args = text is not None
    if checked_args:
        for rf in entry.get("required_flags") or []:
            match = rf.get("match") or re.escape(rf.get("flag", ""))
            if not match:
                continue
            try:
                found = re.search(match, text) is not None
            except re.error as exc:
                raise SystemExit(f"FATAL: bad `match` regex {match!r} for model {args.model}: {exc}")
            if not found:
                missing.append({
                    "flag": rf.get("flag", match),
                    "severity": rf.get("severity", "boot-blocker"),
                    "why": rf.get("why", ""),
                })

    hard = [m for m in missing if m["severity"] in FAIL_SEVERITIES]
    verdict = "fail" if hard else "pass"
    payload = {
        "tool": "known_good_config_check",
        "library": "known_good_config",
        "verb": "check",
        "safety": CONTRACT["check"]["safety"],
        "registry": str(registry_path),
        "model": args.model,
        "registered": True,
        "checked_args": checked_args,
        "missing_required": missing,
        "verdict": verdict,
    }
    _emit(payload, as_json=args.json)
    return 1 if verdict == "fail" else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record and check per-model known-good serving configs (required flags + champion).",
    )
    sub = parser.add_subparsers(dest="verb", required=True)

    record = sub.add_parser("record", description=CONTRACT["record"]["description"])
    record.add_argument("--model", required=True, help="HF id / served-model-name (the registry key)")
    record.add_argument("--registry", default=None, help="Path to known-good-configs.yaml (else $KNOWN_GOOD_CONFIG_REGISTRY / walk-up)")
    record.add_argument("--slug", default=None, help="Deploy-bundle slug (<slug>-deploy)")
    record.add_argument("--arch", default=None, help="Short arch note (why the quirk exists)")
    record.add_argument("--hardware", default=None, help="Validated hardware (e.g. GB300 TP=4 NVFP4)")
    record.add_argument("--engine", default=None, help="vllm | sglang")
    record.add_argument(
        "--required-flag", action="append", default=None,
        help="Pipe-delimited flag|match|severity|why|affected|evidence (repeatable)",
    )
    record.add_argument("--champion-config-ref", default=None, help="Path to the full champion config (my-values / deploy yaml)")
    record.add_argument("--champion-verdict", default=None, help="DRAFT <n> | VERDICT <n>")
    record.add_argument("--champion-campaign", default=None, help="perf-lake campaign id (evidence)")
    record.add_argument("--fallback", default=None, help="A known-working alternative")
    record.add_argument("--grind-frontier", default=None, help="Cross-ref into value-findings.yaml next_lever")
    record.add_argument("--notes", default=None, help="Free-text notes")
    record.add_argument("--json", action="store_true", help="Emit JSON envelope")
    record.set_defaults(func=cmd_record)

    check = sub.add_parser("check", description=CONTRACT["check"]["description"])
    check.add_argument("--model", required=True, help="HF id / served-model-name to check")
    check.add_argument("--registry", default=None, help="Path to known-good-configs.yaml (else $KNOWN_GOOD_CONFIG_REGISTRY / walk-up)")
    check.add_argument("--serve-args", default=None, help="The joined serve-args string to scan for required flags")
    check.add_argument("--deploy-file", default=None, help="A deploy/values file to scan for required flags")
    check.add_argument(
        "--require-registered", action="store_true",
        help="Treat an unregistered model as a FAILURE (for the grind-closure gate)",
    )
    check.add_argument("--json", action="store_true", help="Emit JSON envelope")
    check.set_defaults(func=cmd_check)

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
