# First Hour

Status: Active
Audience: new local users who need to understand the workspace and make safe progress.

This is the shortest safe path for a first session on a laptop or workstation.
It assumes you are new to this workspace, not necessarily new to MLPerf.

> **Note.** Benchmark numbers produced from this tree are
> environment-specific. Do not quote them as external claims.

## Goal

By the end of the first hour, you should know:

- Which directory is the active tree.
- Which docs are authoritative, generated, historical, or ignorable.
- How to run read-only local checks.
- Where to go next for review, engineering, agent setup, or operator work.
- What actions require explicit operator approval before they are safe.

## Minute 0-5: Find The Active Tree

The active tree for this guide is the bundled MCP server at
[`plugins/profile-and-optimize/server/`](/plugins/profile-and-optimize/server). If your workspace also
holds sibling checkouts such as `Megatron-LM/`, `TransformerEngine/`,
`hypertune/`, or `nccl-tests/`, treat them as supporting context unless a
runbook sends you there.

From the repo root:

```bash
cd plugins/profile-and-optimize/server
```

## Minute 5-15: Read The Three Entry Points

Read these in order:

1. [`REVIEWERS.md`](/REVIEWERS.md) - the 30-minute review lane and the
   authoritative/generated/historical/ignorable split.
2. [`../README.md`](/plugins/profile-and-optimize/server/README.md) - the active cockpit: supported targets,
   canonical links, and the minimum operator flow.
3. [`start-here.md`](/plugins/profile-and-optimize/server/docs/start-here.md) - role-based routing after you know the
   basic map.

If a doc disagrees with the runbook under [`../runbooks/`](/plugins/profile-and-optimize/server/runbooks),
the runbook wins for benchmark procedure.

## Minute 15-30: Run Local Read-Only Checks

Install the editable development environment once per checkout:

```bash
bash install.sh --with-dev
```

Then run the repo-layout audit from `server/`:

```bash
python3 tools/shared/audit/audit_repo.py
```

The remaining checks run from the repo root via the Makefile:

```bash
make help
make smoke-test
make pytest
```

If you only have time for one check, run:

```bash
python3 tools/shared/audit/audit_repo.py
```

Keep exact failure text in your handoff or PR. Per [`../CLAUDE.md`](/plugins/profile-and-optimize/server/CLAUDE.md),
do not hide a failing gate behind a broad "known issue" note.

## Minute 30-45: Pick One Reader Path

Pick exactly one next path:

- Reviewing the repo: continue in [`REVIEWERS.md`](/REVIEWERS.md), then
  inspect [`../tools/README.md`](/plugins/profile-and-optimize/server/tools/README.md) and the relevant runbook
  under [`../runbooks/`](/plugins/profile-and-optimize/server/runbooks).
- Changing code or docs: read [`CONTRIBUTING.md`](/CONTRIBUTING.md), then
  use [`../tools/README.md`](/plugins/profile-and-optimize/server/tools/README.md) to find the command surface
  and safety label.
- Setting up agents or MCP: read [`agent-onboarding.md`](/plugins/profile-and-optimize/server/docs/agent-onboarding.md),
  then [`mcp-composition.md`](/plugins/profile-and-optimize/server/docs/mcp-composition.md).
- Preparing cluster work: stop and read [`../CLAUDE.md`](/plugins/profile-and-optimize/server/CLAUDE.md),
  [`../tools/README.md`](/plugins/profile-and-optimize/server/tools/README.md), and the target runbook under
  [`../runbooks/`](/plugins/profile-and-optimize/server/runbooks) before any `sbatch`.

## Minute 45-60: Make One Safe Contribution

Good first progress usually means one of:

- Run `python3 tools/shared/audit/audit_repo.py` and fix a documentation link or status-header issue.
- Run `make pytest` from the repo root and preserve the exact failure if an offline tool regresses.
- Improve a doc link from this guide, [`docs/README.md`](/plugins/profile-and-optimize/server/docs/README.md), or
  [`../tools/README.md`](/plugins/profile-and-optimize/server/tools/README.md) when a useful page is not
  reachable in three clicks.
- Add or update a small fixture-backed unit test for the tool you are touching.

Avoid live cluster actions in the first hour. Do not submit jobs, drain nodes,
pull license-gated data, mutate external systems, or package submission
artifacts unless the operator explicitly asks in the current turn and the
command carries the required acknowledgement flag.

## Glossary

| Term | Meaning |
| --- | --- |
| Active tree | `plugins/profile-and-optimize/server/`, the self-contained MLPerf Training v6.0 cockpit. |
| Sibling repo | A workspace checkout used as reference or supporting context. |
| Runbook | A benchmark procedure under [`../runbooks/`](/plugins/profile-and-optimize/server/runbooks). |
| Campaign | A deliverable-oriented effort tracked as a bundle under your local `./campaigns/` directory. |
| Evidence anchor | Durable generated evidence under [`../experiments/artifacts/`](/plugins/profile-and-optimize/server/experiments/artifacts). |
| Safety label | The read-only, dry-run, submits-jobs, or live-cluster label in [`../tools/README.md`](/plugins/profile-and-optimize/server/tools/README.md). |
| Acknowledgement flag | An explicit `i_understand_this_*` or `--i-understand-this-*` gate required before mutating operations. |
