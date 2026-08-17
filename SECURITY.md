# Security policy

## Supported versions

Security fixes land on `main` while the next release is in development.

| Version | Status |
| --- | --- |
| `main` | Supported development branch |
| `0.4.x` | Supported release line |
| `0.3.x` and earlier | Unsupported |

The latest published release may not contain fixes that have landed on `main`.
Check the [release history](https://github.com/cfregly/gpu-perf-tune/releases) before deploying.

## Report a vulnerability privately

Do not open a public issue for a suspected vulnerability. Use GitHub's
[private vulnerability report](https://github.com/cfregly/gpu-perf-tune/security/advisories)
form. If that form is unavailable, email
[chris@fregly.com](mailto:chris@fregly.com) with the subject
`gpu-perf-tune security report`.

Include only the detail needed to reproduce and assess the issue:

- The affected commit or plugin version.
- The affected skill, MCP tool, manifest, or script.
- A minimal reproduction using synthetic or redacted values.
- The impact and any known workaround.
- A safe way to contact you for follow-up.

Never post tokens, credentials, internal hostnames, private URLs, customer data,
proprietary code, or unredacted logs in a public issue. If you accidentally post
a secret, revoke or rotate it first, then email the maintainer. Deleting a public
comment does not remove the value from every copy or notification.

You should receive an initial response within five business days. Fix and
disclosure timing depends on severity, exploitability, and release readiness.
The project will coordinate public disclosure with the reporter when practical.

## Scope

Reports are in scope when they affect code or configuration maintained in this
repository, including:

- The bundled MCP server under `plugins/profile-and-optimize/server/`.
- Skills under `plugins/profile-and-optimize/skills/`.
- Plugin manifests, including `.mcp.json`.
- Repository helper scripts and CI workflows.
- Unsafe defaults, permission escalation, secret exposure, or unintended
  external writes caused by this project.

External MCP servers, third-party packages, container images, cluster software,
and operator infrastructure are maintained by their respective projects. Report
upstream vulnerabilities to the relevant vendor. A report is still welcome here
when this repository integrates an external component unsafely.

## Credit

With the reporter's permission, the project will credit the reporter in the
release notes or security advisory for the fix.
