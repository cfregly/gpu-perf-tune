# docs/ - priority-ranked map

Status: Active

New local users should start with [`first-hour.md`](/plugins/profile-and-optimize/server/docs/first-hour.md). Reviewers should start with [`REVIEWERS.md`](/REVIEWERS.md). The canonical audience map is [`audience-entrypoints.md`](/plugins/profile-and-optimize/server/docs/audience-entrypoints.md). Do not add another orientation page without updating it. This file is only the compact map of the `docs/` surface. Active and reference docs stay at depth 1.

## Active and Reference Docs

| Doc | Status | Purpose |
| --- | --- | --- |
| [`first-hour.md`](/plugins/profile-and-optimize/server/docs/first-hour.md) | Active | First safe local session for new users: workspace map, read-only checks, role routing, and cluster-action stop signs. |
| [`audience-entrypoints.md`](/plugins/profile-and-optimize/server/docs/audience-entrypoints.md) | Active | Canonical audience-to-entrypoint map. Orientation pages link here instead of creating new routes. |
| [`start-here.md`](/plugins/profile-and-optimize/server/docs/start-here.md) | Active | Short orientation paths for operators and reviewers who need a role-specific route. |
| [`api.md`](/plugins/profile-and-optimize/server/docs/api.md) | Active | Stable public Python API and import-surface contract. |
| [`secrets.md`](/plugins/profile-and-optimize/server/docs/secrets.md) | Active | Local credential names, storage expectations, and artifact redaction rules. |
| [`agent-onboarding.md`](/plugins/profile-and-optimize/server/docs/agent-onboarding.md), [`mcp-composition.md`](/plugins/profile-and-optimize/server/docs/mcp-composition.md), [`mcp-tool-io-contract.md`](/plugins/profile-and-optimize/server/docs/mcp-tool-io-contract.md), [`agent-rationale.md`](/plugins/profile-and-optimize/server/docs/agent-rationale.md) | Active/Reference | Agent and MCP onboarding/contracts plus policy rationale. |
| [`perf-lake-contract.md`](/plugins/profile-and-optimize/server/docs/perf-lake-contract.md) | Active/Reference | Supporting engineering references. |
| [`profiling-and-perf-discovery.md`](/plugins/profile-and-optimize/server/docs/profiling-and-perf-discovery.md), [`operator-commands.md`](/plugins/profile-and-optimize/server/docs/operator-commands.md), [`zymtrace-query-hygiene.md`](/plugins/profile-and-optimize/server/docs/zymtrace-query-hygiene.md) | Reference | Profiling workflow, operator command surface, and query-hygiene notes. |
| [`learnings/`](learnings/) | Reference | Distilled lessons: evidence shape. |

## Other Indexes

| Need | Use |
| --- | --- |
| Active operator runbooks | [`../runbooks/`](/plugins/profile-and-optimize/server/runbooks) |
| Stable CLI / MCP contract | [`cli-contract.md`](/plugins/profile-and-optimize/server/docs/cli-contract.md), [`../mcp_surface.py`](/plugins/profile-and-optimize/server/mcp_surface.py) |
| Command router | [`../tools/README.md`](/plugins/profile-and-optimize/server/tools/README.md) |
| Durable evidence families | [`../experiments/artifacts/`](/plugins/profile-and-optimize/server/experiments/artifacts) |
| Repo policy | [`../AGENTS.md`](/plugins/profile-and-optimize/server/AGENTS.md) |
| Review path | [`REVIEWERS.md`](/REVIEWERS.md) |
