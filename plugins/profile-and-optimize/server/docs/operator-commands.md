Status: Active
Audience: operators who need direct commands now that the Makefile is intentionally small.

# Operator Commands

The Makefile keeps a deliberately small target surface. Run `make help` at the
repo root for the current list. Use the direct commands below for operator
workflows that are not Make targets.

| Task | Direct command |
| --- | --- |
| Repo audits | `python3 tools/shared/audit/audit_repo.py` |
| Pick a cohort | `mlperf-selector pick --bench <id> --nodes <N> --json` |
| Explain a pick | `mlperf-selector explain --cohort <pick-artifact> --json` |
| Check the 256N+ pre-launch gate | `mlperf-selector gate-256n --reservation <name> --json` |
| Summarize fabric/NCCL evidence | `mlperf-selector check-fabric --cohort <path> --json` |
| Look up structured drain reasons | `mlperf-selector node-lookup --comment-prefix '<prefix>' --json` |

Every contract-bearing library is also invokable as `python -m <library>`
(for example `python -m contention snapshot --json` or
`python -m evidence --help`). The full verb matrix, safety classes, and
required/optional flags live in
[`docs/cli-contract.md`](/plugins/profile-and-optimize/server/docs/cli-contract.md),
the MCP server derives its tool surface from the same parsers, so a verb
documented there is callable both from the shell and as an MCP tool.
