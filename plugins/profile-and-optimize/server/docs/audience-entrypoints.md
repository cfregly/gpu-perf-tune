# Audience Entrypoints

Status: Active
Audience: maintainers reducing onboarding-page drift.

This is the canonical audience map. Other orientation pages should link here
instead of inventing a new "start here" path.

| Audience | Canonical entrypoint | Why |
| --- | --- | --- |
| New local user | [`first-hour.md`](first-hour.md) | One safe workstation session, read-only checks first. |
| Reviewer | [`REVIEWERS.md`](../../../../REVIEWERS.md) | The 30-minute review path and authoritative/generated/historical split. |
| Operator | [`../README.md`](../README.md), then the target runbook under [`../runbooks/`](../runbooks) | Current target routing and benchmark procedure. |
| Engineer | [`CONTRIBUTING.md`](../../../../CONTRIBUTING.md), then [`../tools/README.md`](../tools/README.md) | Local setup, code-change expectations, and command surfaces. |
| Agent or MCP client | [`agent-onboarding.md`](agent-onboarding.md), then [`mcp-composition.md`](mcp-composition.md) | Tool setup and cross-system routing. |
| Credential setup | [`secrets.md`](secrets.md) | Local secret names and artifact redaction rules. |

Do not add another orientation page without updating this table and the docs
index in the same change.
