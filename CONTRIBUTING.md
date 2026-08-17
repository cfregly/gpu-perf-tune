# Contributing to `gpu-perf-tune`

`gpu-perf-tune` is a client-neutral GPU performance project. It ships Agent
Skills, an MCP server, safety guards, schemas, and validation tools.
`profile-and-optimize` is the skills package and Claude Code adapter. Claude
Code and Codex are the first-class clients. Other client helpers are best
effort and do not block a release.

## Contributor setup

```bash
git clone https://github.com/cfregly/gpu-perf-tune.git
cd gpu-perf-tune
bash plugins/profile-and-optimize/server/install.sh --with-dev
make all
```

The development install includes the official Agent Skills validator, Ruff,
Pyright, ShellCheck, test tools, and every dependency required by the default
test suite. `make quality` runs the language checks. `make all` adds docs,
skill validation, the MCP smoke test, and pytest. Use `make -j4 all` to run
independent targets in parallel.

Claude Code packaging has a separate optional check:

```bash
make validate-claude-plugin
```

Run it when a change affects the Claude marketplace manifest, plugin manifest,
or Claude hook adapter. The core contributor path does not require Claude Code.

## Add a skill

Start from the maintained template:

```bash
cp -R plugins/profile-and-optimize/templates/skill \
  plugins/profile-and-optimize/skills/example-skill
```

Choose one clear task. Search the existing skills first. A new skill should not
duplicate another workflow or bundle unrelated tasks.

The directory name and frontmatter `name` must match. Names use lowercase
letters, digits, and hyphens, with 64 characters or fewer. The official Agent
Skills specification accepts these frontmatter fields:

- `name`
- `description`
- `license`
- `compatibility`
- `metadata`
- `allowed-tools`

`allowed-tools` is a space-delimited string, not a YAML list. Use the narrowest
set of tools the workflow needs. A minimal header looks like this:

```yaml
---
name: example-skill
description: Profiles a named GPU workload when an operator asks for a bounded kernel analysis.
license: MIT
compatibility: Requires a skills-compatible agent and the tools named in allowed-tools.
metadata:
  last-validated: "2026-08-16"
allowed-tools: "mcp__profile_and_optimize__search_runbooks Read Grep"
---
```

Do not mark adapted third-party work as MIT by default. Preserve its license,
identify the exact source revision, and update `THIRD_PARTY_NOTICES.md`.

Every skill needs:

- A specific `description` that says what it does and when to use it.
- Prerequisites that fail closed when required inputs are missing.
- A numbered workflow with clear report and ask checkpoints.
- Safety rules for external writes, cluster changes, and acknowledgement flags.
- Source references that link to shared docs instead of copying them.
- A measured result format that records workload, baseline, hardware,
  precision, parallelism, engine version, and evidence.

Keep the main skill file focused. Put long reference material in sibling files.

## Validate a skill change

```bash
make validate-agent-skills
make smoke-test
make check-doc-links
```

The first command uses the official reference validator. The smoke target also
checks counts, versions, and MCP argument references. The link check covers all
tracked Markdown files.

If the change affects runtime code, run the full suite:

```bash
make all
```

## MCP tool naming

Skills refer to MCP tools as `mcp__<server-key>__<tool-name>`.

| Server | Key | Tool prefix |
| --- | --- | --- |
| Bundled server | `profile_and_optimize` | `mcp__profile_and_optimize__<tool>` |
| Grafana | `grafana` | `mcp__grafana__<tool>` |
| GitHub | `github` | `mcp__github__<tool>` |
| Prometheus, optional | `prometheus_mcp` | `mcp__prometheus_mcp__<tool>` |
| zymtrace, optional | `zymtrace` | `mcp__zymtrace__<tool>` |

The bundled server exposes 51 contract tools and 2 auxiliary tools across 8
libraries. `plugins/profile-and-optimize/server/mcp_surface.py` is the count
and registration source of truth.

Operator credentials belong in environment variables or private client
configuration. Never commit tokens, internal hostnames, private URLs, customer
data, or unredacted logs.

## Add or change an MCP verb

The server source lives under `plugins/profile-and-optimize/server/`.

1. Add the verb to the relevant library contract and implementation.
2. Add tests for success, invalid input, safety class, and acknowledgement
   behavior where applicable.
3. Update `mcp_surface.py` if the library or public tool surface changes.
4. Update the canonical count constants and any docs that name the count.
5. Run `make pytest` and `make smoke-mcp-runtime`. The pytest target includes
   the MCP and installer tests.
6. Run `make lint-tool-counts` and inspect `make mcp-surface`.

Removing or renaming a public tool is a breaking change. Adding a public tool
is a feature change.

## Versions and changelog

The root `VERSION` file is the release version source of truth. The Claude
adapter manifest, two Python package versions, package README banner, and
latest changelog entry must match it.

| Change | Version bump |
| --- | --- |
| Documentation or behavior-preserving fix | PATCH |
| New skill, tool, or supported workflow | MINOR |
| Breaking public skill, tool, or contract change before 1.0.0 | MINOR plus migration notes |
| Breaking public skill, tool, or contract change from 1.0.0 onward | MAJOR |

Add a dated entry to `CHANGELOG.md`. Run `make lint-versions` and
`make check-version-transition` before opening a pull request. A version may
stay unchanged or increase. It may not decrease.

Merging a new `VERSION` value into `main` triggers the release workflow. It
waits for the full CI matrix on that exact `main` commit, then creates the
annotated tag and GitHub Release from the matching changelog section. Keep the
version bump in the final squash-merged commit. The workflow is idempotent when
that version is already published.

If the exact version-bump SHA fails because of a code defect, fix it under a
new version and changelog entry. A transient CI failure may rerun the same SHA.
Do not move the original version to a later source commit.

`make release` is a manual recovery path for a missing publication. Use it only
from a clean `main` that exactly matches `origin/main`, contains the version
bump, and has passed the required push CI run:

```bash
git fetch origin
git status -sb
git ls-remote --tags origin vX.Y.Z
make release
```

`make release` requires an authenticated `gh` CLI. It verifies the exact CI
run and remote tag before creating any missing tag or stable GitHub Release.
Do not reuse or move a published tag.

The historical `v0.1.0` tag predates this policy and is lightweight. It is the
only accepted exception. Release tags from `v0.2.0` onward are annotated and
immutable after publication.

## Open a pull request

Create a focused branch, commit the smallest coherent change, and open a pull
request. The pull request template lists the required evidence.

Before requesting review:

- Run `make all`.
- Confirm the version and changelog match the change.
- Confirm public docs do not contain private data.
- Explain any skipped check and why it cannot run in your environment.
- Run `make validate-claude-plugin` when Claude packaging changed.

Review guidance lives in [`REVIEWERS.md`](REVIEWERS.md).

## Code of conduct and security

Follow [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Report suspected
vulnerabilities through the private process in [`SECURITY.md`](SECURITY.md).
Do not place sensitive details in a public issue.
