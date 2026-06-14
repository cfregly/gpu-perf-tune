# Start here

Status: Active
Audience: anyone new to this repo.

> **Note.** Benchmark numbers produced from this tree are
> environment-specific. Do not quote them as official results.

The active surface is small. Pick one path below, read those files, and
ignore the rest until you need it.

| If you are... | Read first |
| --- | --- |
| Brand new to the workspace | [`first-hour.md`](/plugins/profile-and-optimize/server/docs/first-hour.md) |
| Reviewing the repo | [`REVIEWERS.md`](/REVIEWERS.md) |
| Running a benchmark on the cluster | [`../README.md`](/plugins/profile-and-optimize/server/README.md), then the runbook for your target under [`../runbooks/`](/plugins/profile-and-optimize/server/runbooks) |
| Implementing or reviewing a CLI verb | [`cli-contract.md`](/plugins/profile-and-optimize/server/docs/cli-contract.md) plus [`../mcp_surface.py`](/plugins/profile-and-optimize/server/mcp_surface.py) |
| Operating the MCP server | [`mcp-tool-io-contract.md`](/plugins/profile-and-optimize/server/docs/mcp-tool-io-contract.md) and [`../tools/profile_and_optimize_mcp/README.md`](/plugins/profile-and-optimize/server/tools/profile_and_optimize_mcp/README.md) |
| Setting repo policy or CI gates | [`../AGENTS.md`](/plugins/profile-and-optimize/server/AGENTS.md) |
| Looking up campaign state | Your local `./campaigns/` directory |
| Looking up evidence | [`../experiments/artifacts/`](/plugins/profile-and-optimize/server/experiments/artifacts) |

## Runbooks

Operator runbooks live under [`../runbooks/`](/plugins/profile-and-optimize/server/runbooks). Its README
explains what belongs there.

## Mental model in one paragraph

This tree is research infrastructure for running, measuring, and packaging
MLPerf v6.0 training benchmarks on GB300 (and B200) Slurm clusters. The
active operator surface is a set of small CLI libraries (for example
[`selector/`](/plugins/profile-and-optimize/server/selector)) with a stable CLI contract documented in
[`cli-contract.md`](/plugins/profile-and-optimize/server/docs/cli-contract.md) and an auto-derived MCP tool surface in
[`../mcp_surface.py`](/plugins/profile-and-optimize/server/mcp_surface.py). The repo's job is to (1) pick good
cohorts of nodes before launch, (2) run a controlled set of experiments that
optimize for performance, scalability, reproducibility, and consistency,
(3) record results honestly with cited evidence, (4) surface blockers across
every layer, and (5) drive each benchmark to a compliant five-run submission.

## Directory boundary cheat sheet

| Path | What lives here |
| --- | --- |
| [`../README.md`](/plugins/profile-and-optimize/server/README.md) | Live status: goals, blockers, canonical entry points. |
| [`../runbooks/`](/plugins/profile-and-optimize/server/runbooks) | Compact active operator runbooks per target. |
| [`../docs/`](/plugins/profile-and-optimize/server/docs) (this dir) | Engineering reference and operator how-tos. |
| [`../selector/`](/plugins/profile-and-optimize/server/selector) and sibling libraries (`contention/`, `ai_tuning/`, `profile/`, ...) | The operator-facing CLI libraries. |
| [`../tools/`](/plugins/profile-and-optimize/server/tools) | Pipeline-step CLIs, audits, and benchmark scaffolding. |
| [`../experiments/artifacts/`](/plugins/profile-and-optimize/server/experiments/artifacts) | Curated peer-reviewable evidence. |
| [`../tuning/`](/plugins/profile-and-optimize/server/tuning) | Tuning spaces, rules, baselines, best-known knob ledger. |
| [`../AGENTS.md`](/plugins/profile-and-optimize/server/AGENTS.md) | Repo policy: fail-fast and evidence conventions, plus the MCP server's root-discovery marker. |
