# Security policy

## Supported versions

`profile-and-optimize` follows semantic versioning. Security fixes ship as PATCH releases against the latest MINOR.

| Version | Supported |
| --- | --- |
| `1.14.x` | Yes (latest. Current release line) |
| `1.13.x` | Yes (previous MINOR. Best-effort PATCH backports) |
| `1.0.x`-`1.12.x` | Best-effort. Please upgrade to `1.14.x` |
| `< 1.0.0` | No |

The release line moves forward as new MINORs ship. See the GitHub releases for the version history.

## Reporting a vulnerability

**Please do NOT open a public GitHub issue for a security vulnerability.** Open a public issue only if the report does not contain any sensitive information (token leaks, credential paths, internal hostnames, etc.).

For a security report:

- Open a private report through GitHub Security Advisories: the repository Security Advisories page (repo **Security** tab → **Advisories** → **Report a vulnerability**).
- Include a minimal reproduction (skill name, exact prompt, what was leaked or what unsafe action was taken).
- Include the plugin version (`cat plugins/profile-and-optimize/.claude-plugin/plugin.json | python3 -c "import json,sys;print(json.load(sys.stdin)['version'])"`).
- Include the operator workstation OS + Claude Code / Cursor version.

You should receive an initial response within 5 business days. If you do not, add a comment to the advisory thread referencing the original report.

## In scope

- The bundled MCP server source at [`plugins/profile-and-optimize/server/`](/plugins/profile-and-optimize/server), including the 8 stub libraries and the `tools/` implementations.
- The skill files at [`plugins/profile-and-optimize/skills/`](/plugins/profile-and-optimize/skills), specifically: any skill that grants `allowed-tools` access beyond what its purpose requires. Any skill that exfiltrates secrets / tokens. Any skill that writes to external chat systems (skills are read-only toward chat).
- The plugin manifests ([`marketplace.json`](/.claude-plugin/marketplace.json), [`plugin.json`](/plugins/profile-and-optimize/.claude-plugin/plugin.json), [`.mcp.json`](/plugins/profile-and-optimize/.mcp.json)) - specifically: tokens / URLs hard-coded instead of `${ENV}` placeholders, ack-flag bypass paths, or any change that escalates a tool's safety class without operator notice.
- The helper scripts in [`scripts/`](/scripts).

## Out of scope

- The 2 external MCP servers declared in [`.mcp.json`](/plugins/profile-and-optimize/.mcp.json) (`grafana`, `github`) and any optional operator-configured MCP servers (e.g. `prometheus_mcp`, `zymtrace`). Vulnerabilities in those should be reported to their respective vendors.
- Cluster-side vulnerabilities (Slurm, container images, NCCL, etc.). Report those to your cluster operator or the relevant upstream project.
- Any vulnerability disclosed in a public GitHub issue (we will close it without comment if it contains sensitive info, and request re-reporting through the private advisory channel above).

## Disclosure timeline

Standard responsible-disclosure: we aim to ship a PATCH release within 30 days of a confirmed vulnerability. If the vulnerability is actively exploited or has a high CVSS, we will expedite.

## Acknowledgements

We credit the reporter in the release notes for the fix release, with the reporter's permission.

## Contact

Security reports: through GitHub Security Advisories.
