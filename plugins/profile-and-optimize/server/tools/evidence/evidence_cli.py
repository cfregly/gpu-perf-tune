"""Evidence-bundle scaffolder: ``evidence init``.

Creates a new immutable evidence bundle directory under
``<repo_root>/experiments/artifacts/<family>/<run-id>/`` populated with the
skeleton the workspace's "Reproducibility-Grade Evidence" rule expects:

- ``SOURCE.md`` with operator + cluster + timestamp + git SHA + intent.
- ``summary.md`` with a verdict skeleton the operator fills in.
- ``commands/`` with a ``README.md`` documenting the four-file tuple
  capture convention and a ``.gitkeep`` so the directory survives
  ``git add``.

Workload-agnostic; works for any experiment family. Added in profile-and-optimize v0.4.0.
See the skill [``evidence-bundle-init``](../../skills/evidence-bundle-init/SKILL.md)
for the operator-facing workflow.
"""

from __future__ import annotations

import argparse
import json
import os
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


def _resolve_repo_root(arg: str | None) -> Path:
    if arg:
        return Path(arg).expanduser().resolve()
    env = os.environ.get("PROFILE_AND_OPTIMIZE_REPO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").is_file() and (current / "tools").is_dir():
            return current
        current = current.parent
    raise SystemExit("FATAL: cannot resolve repo root; pass --repo-root or set PROFILE_AND_OPTIMIZE_REPO_ROOT")


SOURCE_MD_TEMPLATE = """# SOURCE

**Family:** `{family}`
**Run-id:** `{run_id}`
**Created at (UTC):** `{utc_iso}`
**Created by:** the MLPerf team (operator: `{operator_user}` on `{hostname}`)
**profile-and-optimize SHA at creation:** `{sha_short}`

## Intent

{intent}

## Provenance

- Workstation kernel: `{uname}`
- Repo: `<your-org>/claude-perf-tune` (skills + bundled MCP server).
- Bundle path: `experiments/artifacts/{family}/{run_id}/`

## Experiment isolation & traceability

The run-id IS the experiment-id: the single join key across this bundle, the
cluster objects, and the perf-lake (workspace `AGENTS.md` "Experiment Isolation
& Traceability").

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
- pre-apply label gate: `perf-tune-glm51/verify-experiment-labels.sh {run_id} <manifests>`.

## Source-code attribution (provenance)

Machine-readable link from this run-id to the ACTUAL source under test
(`experiment_provenance_v1`). Fill `source[]` (repo/branch/commit/delivery/image)
before publishing a VERDICT; in a deploy bundle, auto-capture/refresh it with
`capture-provenance.sh <this-bundle> --write --force`.

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

- Workspace [`AGENTS.md`](../../../AGENTS.md) "Reproducibility-Grade Evidence" +
  "Experiment Isolation & Traceability" rules.
- This file was scaffolded by `mcp__profile_and_optimize__evidence_init` (skill: [`evidence-bundle-init`](../../../plugins/profile-and-optimize/skills/evidence-bundle-init/SKILL.md)).
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

Per workspace [`AGENTS.md`](../../../../AGENTS.md) "Reproducibility-Grade Evidence",
every shell command run during this experiment is captured as a four-file tuple:

    00-<step-slug>.cmd       # the exact command
    00-<step-slug>.stdout    # captured stdout
    00-<step-slug>.stderr    # captured stderr
    00-<step-slug>.exit      # exit code

Filenames are zero-padded sequential (00, 01, 02, ...) so the chronological
order is preserved in `ls`. Use this helper to capture all four atomically:

    run() {
      local n="$1"; shift
      local slug="$1"; shift
      local prefix="$(printf '%02d-%s' "${n}" "${slug}")"
      printf '%s ' "$@" > "${prefix}.cmd"; echo >> "${prefix}.cmd"
      "$@" > "${prefix}.stdout" 2> "${prefix}.stderr"
      echo $? > "${prefix}.exit"
    }
    run 0 ls-image ls /mnt/data/images/
"""


def cmd_init(args: argparse.Namespace) -> int:
    repo_root = _resolve_repo_root(args.repo_root)
    family = args.family
    run_id = args.run_id or utc_now_slug()
    bundle = repo_root / "experiments" / "artifacts" / family / run_id

    if bundle.exists():
        # Per workspace AGENTS.md artifact-durability rule: bundles are immutable.
        print(
            f"FATAL: bundle already exists: {bundle}\n"
            "Bundles are immutable per workspace AGENTS.md. Use a fresh --run-id.",
            file=sys.stderr,
        )
        return 2

    hostname, uname, operator_user = gather_workstation_facts()
    profile_and_optimize_sha = discover_profile_and_optimize_sha(repo_root)
    sha_short = (profile_and_optimize_sha or "(unknown)")[:12]
    utc_iso = utc_now_iso()

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
        )
    )
    (bundle / "summary.md").write_text(SUMMARY_MD_TEMPLATE)
    (bundle / "commands" / "README.md").write_text(COMMANDS_README_TEMPLATE)

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
