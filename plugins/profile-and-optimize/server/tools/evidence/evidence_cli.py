"""Evidence-bundle scaffolder: ``evidence init``.

Creates a new immutable evidence bundle directory under
``<repo_root>/experiments/artifacts/<family>/<run-id>/`` populated with the
skeleton the project's "Reproducibility-Grade Evidence" rule expects:

- ``SOURCE.md`` with operator + cluster + timestamp + git SHA + intent.
- ``summary.md`` with a verdict skeleton the operator fills in.
- ``commands/`` with a ``README.md`` documenting the four-file tuple
  capture convention and a ``.gitkeep`` for the local directory shape.

Evidence artifacts are ignored by default. Publishing a scrubbed bundle to the
repository requires an explicit review and ``git add -f``.

Works with any experiment family. See the skill
[``evidence-bundle-init``](../../../skills/evidence-bundle-init/SKILL.md) for the
operator-facing workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Reuse the perf-baseline helpers for git-SHA discovery + workstation facts.
from tools.perf_baseline.helpers import (
    discover_profile_and_optimize_sha,
    gather_workstation_facts,
    utc_now_iso,
    utc_now_slug,
)

CONTRACT: dict[str, dict[str, Any]] = {
    "init": {
        "safety": "writes_artifacts",
        "required": ("--family", "--intent"),
        "optional": ("--run-id", "--repo-root", "--json"),
        "json": True,
        "ack": None,
        "description": "Scaffold a new immutable evidence bundle directory.",
    },
}

_SLUG_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_PUBLIC_SKILL_URL = (
    "https://github.com/cfregly/gpu-perf-tune/blob/main/"
    "plugins/profile-and-optimize/skills/evidence-bundle-init/SKILL.md"
)


def _safe_relative_path(value: str, *, label: str, allow_nested: bool) -> Path:
    """Validate an operator slug before using it in an artifact path."""

    if not value or "\\" in value:
        raise ValueError(f"{label} must use letters, numbers, dots, underscores, hyphens, or slashes")
    path = Path(value)
    parts = path.parts
    if (
        path.is_absolute()
        or not parts
        or any(part in {"", ".", ".."} or _SLUG_PART.fullmatch(part) is None for part in parts)
    ):
        raise ValueError(f"{label} must be a safe relative slug")
    if not allow_nested and len(parts) != 1:
        raise ValueError(f"{label} must be one path segment")
    return path


def _resolve_repo_root(arg: str | None) -> Path:
    if arg:
        return Path(arg).expanduser().resolve()
    env = os.environ.get("PROFILE_AND_OPTIMIZE_REPO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (
            (current / "pyproject.toml").is_file()
            and (current / "mcp_surface.py").is_file()
            and (current / "tools").is_dir()
        ):
            return current
        current = current.parent
    raise SystemExit("FATAL: cannot resolve repo root; pass --repo-root or set PROFILE_AND_OPTIMIZE_REPO_ROOT")


SOURCE_MD_TEMPLATE = """# SOURCE

**Family:** `{family}`
**Run-id:** `{run_id}`
**Created at (UTC):** `{utc_iso}`
**Created by:** `{operator_user}` on `{hostname}`
**profile-and-optimize SHA at creation:** `{sha_short}`

## Intent

{intent}

## Provenance

- Workstation kernel: `{uname}`
- Repo: `cfregly/gpu-perf-tune` (skills + bundled MCP server).
- Bundle path: `experiments/artifacts/{family}/{run_id}/`

## Experiment isolation & traceability

The run-id IS the experiment-id: the single join key across this bundle, the
cluster objects, and the perf-lake (`gpu-perf-tune`
[`AGENTS.md`](https://github.com/cfregly/gpu-perf-tune/blob/main/AGENTS.md)
"Experiment Isolation and Traceability").

- experiment_id: {run_id}
- family: {family}
- object label (EVERY cluster object, on metadata AND pod template): `experiment={run_id}`
- cluster resources created (fill in as you create them; every
  Deployment/Pod/Job/PVC/PV/Secret/ConfigMap/Service, experiment-unique-named,
  NEVER a standing/migration name):
  -
- perf-lake campaign: `campaign={run_id}` (run `perftunereport campaign_init
  --experiment-id {run_id} --family {family} --evidence-bundle <this-bundle>`
  so campaign_id == experiment_id; the s3 atlas_v1 + campaign_v1 paths are
  auto-appended below by `publish_to_lake`).
- pre-apply label check: confirm every rendered object and pod template uses
  `experiment={run_id}`. Record the exact validator command and its output in
  `commands/`.

## Source-code attribution (provenance)

Machine-readable link from this run-id to the ACTUAL source under test
(`experiment_provenance_v1`). Fill `source[]` (repo/branch/commit/delivery/image)
before publishing a VERDICT. Record `git rev-parse HEAD` and `git status
--short` from the source checkout in `commands/` so the attribution can be
audited.

```provenance
schema: experiment_provenance_v1
identity:
  run_id: {run_id}
  id_slug: {run_id}
  title: "{family} experiment"
  hypothesis: "__FILL__"
  family: {family}
  tags: []
  status: active            # active|verified|refuted|incomplete|superseded
  supersedes: ""
  superseded_by: ""
source:
  - repo: __FILL__          # e.g. example/vllm
    branch: __FILL__
    commit: __FILL__        # the real SHA under test
    dirty: false
    delivery: __FILL__      # image|overlay|patchedVllm|infr-patch
    image: __FILL__
    image_pip_version: __FILL__
verdict:
  tier: draft               # draft|verdict (a verdict MUST pin a clean source commit)
  claim: ""
  baseline: ""
  metric: ""
```

## Cross-references

- Project [`AGENTS.md`](https://github.com/cfregly/gpu-perf-tune/blob/main/AGENTS.md)
  "Reproducibility-Grade Evidence" +
  "Experiment Isolation & Traceability" rules.
- This file was scaffolded by `mcp__profile_and_optimize__evidence_init` (skill: [`evidence-bundle-init`]({skill_link})).
"""


SUMMARY_MD_TEMPLATE = """# Summary

**Status:** in-progress

## Verdict

_(to be filled in by operator at end of experiment)_

## Findings

-

## Recommendations

-

## Open questions

-
"""


COMMANDS_README_TEMPLATE = """# commands/

Per project [`AGENTS.md`](https://github.com/cfregly/gpu-perf-tune/blob/main/AGENTS.md)
"Reproducibility-Grade Evidence",
every shell command run during this experiment is captured as a four-file tuple:

    00-<step-slug>.cmd       # the exact command
    00-<step-slug>.stdout    # captured stdout
    00-<step-slug>.stderr    # captured stderr
    00-<step-slug>.exit      # exit code

Filenames are zero-padded sequential (00, 01, 02, ...) so natural sort order
matches execution order. From the server root, use the checked-in and tested
capture helper. The `||` branch preserves a nonzero command status even when
the caller enabled `set -e`:

    capture_rc=0
    ART_DIR="{bundle_relative}" \
      bash tools/shared/capture_cmd.sh ls-image -- ls /mnt/data/images/ \
      || capture_rc=$?
    if [ "$capture_rc" -ne 0 ]; then
      exit "$capture_rc"
    fi

The helper shell-quotes the exact argv into `.cmd`, writes stdout and stderr
separately, writes `.exit` before returning, and returns the wrapped command's
status.
"""


def cmd_init(args: argparse.Namespace) -> int:
    repo_root = _resolve_repo_root(args.repo_root)
    try:
        family_path = _safe_relative_path(
            args.family,
            label="family",
            allow_nested=True,
        )
        run_id_path = _safe_relative_path(
            args.run_id or utc_now_slug(),
            label="run-id",
            allow_nested=False,
        )
    except ValueError as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2

    evidence_root = (repo_root / "experiments" / "artifacts").resolve()
    bundle = (evidence_root / family_path / run_id_path).resolve()
    if not bundle.is_relative_to(evidence_root):
        print("FATAL: evidence bundle path escapes experiments/artifacts", file=sys.stderr)
        return 2

    family = family_path.as_posix()
    run_id = run_id_path.as_posix()

    if bundle.exists():
        # Per project AGENTS.md artifact-durability rule: bundles are immutable.
        print(
            f"FATAL: bundle already exists: {bundle}\n"
            "Bundles are immutable per project AGENTS.md. Use a fresh --run-id.",
            file=sys.stderr,
        )
        return 2

    hostname, uname, operator_user = gather_workstation_facts()
    profile_and_optimize_sha = discover_profile_and_optimize_sha(repo_root)
    sha_short = (profile_and_optimize_sha or "(unknown)")[:12]
    utc_iso = utc_now_iso()

    skill_target = repo_root.parent / "skills" / "evidence-bundle-init" / "SKILL.md"
    if skill_target.is_file():
        skill_link = Path(os.path.relpath(skill_target, bundle)).as_posix()
    else:
        skill_link = _PUBLIC_SKILL_URL

    bundle.mkdir(parents=True, exist_ok=False)
    (bundle / "commands").mkdir(parents=True, exist_ok=False)
    (bundle / "commands" / ".gitkeep").write_text("")
    (bundle / "SOURCE.md").write_text(
        SOURCE_MD_TEMPLATE.format(
            family=family,
            run_id=run_id,
            utc_iso=utc_iso,
            operator_user=operator_user,
            hostname=hostname,
            sha_short=sha_short,
            intent=args.intent,
            uname=uname,
            skill_link=skill_link,
        )
    )
    (bundle / "summary.md").write_text(SUMMARY_MD_TEMPLATE)
    bundle_relative = bundle.relative_to(repo_root).as_posix()
    (bundle / "commands" / "README.md").write_text(COMMANDS_README_TEMPLATE.format(bundle_relative=bundle_relative))

    payload = {
        "tool": "evidence_init",
        "library": "evidence",
        "verb": "init",
        "safety": CONTRACT["init"]["safety"],
        "bundle_dir": str(bundle),
        "family": family,
        "run_id": run_id,
        "created_at_utc": utc_iso,
        "profile_and_optimize_sha": profile_and_optimize_sha,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"created bundle: {bundle}")
        print(f"  family:        {family}")
        print(f"  run_id:        {run_id}")
        print(f"  created_utc:   {utc_iso}")
        print(f"  next:          cd {bundle} && start capturing commands/00-<step>.* tuples")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scaffold a new reproducibility-grade evidence bundle directory.")
    sub = parser.add_subparsers(dest="verb", required=True)

    init = sub.add_parser("init", description=CONTRACT["init"]["description"])
    init.add_argument(
        "--family", required=True, help="Family slug, e.g. cluster-health, nccl-tests, gpu-burn, campaign/llama31_8b"
    )
    init.add_argument("--intent", required=True, help="One-line operator intent captured in SOURCE.md")
    init.add_argument("--run-id", default=None, help="Bundle slug; default: <UTC-timestamp>")
    init.add_argument("--repo-root", default=None, help="Override PROFILE_AND_OPTIMIZE_REPO_ROOT")
    init.add_argument("--json", action="store_true", help="Emit JSON envelope")
    init.set_defaults(func=cmd_init)

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
