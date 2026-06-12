# Audience Entrypoints

Status: Active
Audience: maintainers reducing onboarding-page drift.

This is the canonical audience map. Other orientation pages should link here
instead of inventing a new "start here" path.

| Audience | Canonical entrypoint | Why |
| --- | --- | --- |
| New local user | [`first-hour.md`](/plugins/profile-and-optimize/server/docs/first-hour.md) | One safe workstation session, read-only checks first. |
| Reviewer | [`REVIEWERS.md`](/REVIEWERS.md) | The 30-minute review path and authoritative/generated/historical split. |
| Operator | [`../README.md`](/plugins/profile-and-optimize/server/README.md), then the target runbook under [`../runbooks/`](/plugins/profile-and-optimize/server/runbooks) | Current target routing and benchmark procedure. |
| Engineer | [`CONTRIBUTING.md`](/CONTRIBUTING.md), then [`../tools/README.md`](/plugins/profile-and-optimize/server/tools/README.md) | Local setup, code-change expectations, and command surfaces. |
| Agent or MCP client | [`agent-onboarding.md`](/plugins/profile-and-optimize/server/docs/agent-onboarding.md), then [`mcp-composition.md`](/plugins/profile-and-optimize/server/docs/mcp-composition.md) | Tool setup and cross-system routing. |
| Credential setup | [`secrets.md`](/plugins/profile-and-optimize/server/docs/secrets.md) | Local secret names and artifact redaction rules. |

Do not add another orientation page without updating this table and the docs
index in the same change.
