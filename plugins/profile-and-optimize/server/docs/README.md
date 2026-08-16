# docs/ - priority-ranked map

Status: Active

New local users should start with [`first-hour.md`](first-hour.md). Reviewers should start with [`REVIEWERS.md`](../../../../REVIEWERS.md). The canonical audience map is [`audience-entrypoints.md`](audience-entrypoints.md). Do not add another orientation page without updating it. This file is only the compact map of the `docs/` surface. Active and reference docs stay at depth 1.

## Active and Reference Docs

| Doc | Status | Purpose |
| --- | --- | --- |
| [`first-hour.md`](first-hour.md) | Active | First safe local session for new users: workspace map, read-only checks, role routing, and cluster-action stop signs. |
| [`audience-entrypoints.md`](audience-entrypoints.md) | Active | Canonical audience-to-entrypoint map. Orientation pages link here instead of creating new routes. |
| [`start-here.md`](start-here.md) | Active | Short orientation paths for operators and reviewers who need a role-specific route. |
| [`api.md`](api.md) | Active | Stable public Python API and import-surface contract. |
| [`secrets.md`](secrets.md) | Active | Local credential names, storage expectations, and artifact redaction rules. |
| [`agent-onboarding.md`](agent-onboarding.md), [`mcp-composition.md`](mcp-composition.md), [`mcp-tool-io-contract.md`](mcp-tool-io-contract.md), [`agent-rationale.md`](agent-rationale.md) | Active/Reference | Agent and MCP onboarding/contracts plus policy rationale. |
| [`perf-lake-contract.md`](perf-lake-contract.md) | Active/Reference | Supporting engineering references. |
| [`profiling-and-perf-discovery.md`](profiling-and-perf-discovery.md), [`operator-commands.md`](operator-commands.md), [`zymtrace-query-hygiene.md`](zymtrace-query-hygiene.md) | Reference | Profiling workflow, operator command surface, and query-hygiene notes. |
| [`performance-hints.md`](performance-hints.md) | Active/Reference | Jeff Dean and Sanjay Ghemawat's performance principles adapted to GPU-inference estimation, measurement, and experiment routing. |
| [`learnings/`](learnings/) | Reference | Distilled lessons: evidence shape. |

## Other Indexes

| Need | Use |
| --- | --- |
| Active operator runbooks | [`../runbooks/`](../runbooks) |
| Stable CLI / MCP contract | [`cli-contract.md`](cli-contract.md), [`../mcp_surface.py`](../mcp_surface.py) |
| Command router | [`../tools/README.md`](../tools/README.md) |
| Durable evidence families | [`../experiments/artifacts/`](../experiments/artifacts) |
| Repo policy | [`../AGENTS.md`](../AGENTS.md) |
| Review path | [`REVIEWERS.md`](../../../../REVIEWERS.md) |
