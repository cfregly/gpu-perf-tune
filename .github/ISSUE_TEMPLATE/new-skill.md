---
name: New skill proposal
about: Propose a new skill for the profile-and-optimize skills package.
title: "skill: <proposed-name>: <short description>"
labels: enhancement
---

This issue is public. Use synthetic examples and remove credentials, internal
hostnames, private URLs, customer data, and proprietary code or logs.

## Proposed skill name

`<lowercase-hyphenated-name>` (matches the directory name and the `name:` frontmatter field).

## Task

One paragraph: what task does this skill perform end-to-end?

## When to use

Bulleted list of operator-side trigger scenarios.

## When NOT to use

Bulleted list of overlapping skills + a one-line "use that one instead" pointer for each.

Cross-check against the existing [`profile-and-optimize` skills](https://github.com/cfregly/gpu-perf-tune/tree/main/plugins/profile-and-optimize/skills).

If your task overlaps with any of these, explain why a new skill is warranted.
The proposal must stay tied to GPU performance measurement, diagnosis,
optimization, reporting, or the cluster operations that support those tasks.

## Trigger phrases for the `description` frontmatter

Six or more concrete operator phrases that should auto-route to this skill. Specific verbs, not vague helpers. Example: `"run a perf bench sweep"`, `"profile the decode hot path"`, `"size GPUs for this TPM target"`.

## MCP tools / Bash commands the skill needs

List the `allowed-tools` entries:

- `mcp__<server>__<tool>`
- `Bash(<command>:*)`
- `Read` / `Write` / `Grep`

If a new MCP server is required (not already in [`plugins/profile-and-optimize/.mcp.json`](https://github.com/cfregly/gpu-perf-tune/blob/main/plugins/profile-and-optimize/.mcp.json)), call that out - adding it is a separate PR.

## Safety classification

- Read-only on cluster state? `yes / no`
- Submits Slurm jobs? `yes / no` (if yes: which `i_understand_this_*` ack flag?)
- Mutates external state (Slack write, GitHub create, etc.)? `yes / no` (if yes: STOP and re-read [REVIEWERS.md](https://github.com/cfregly/gpu-perf-tune/blob/main/REVIEWERS.md) -- this is usually a forbidden combination.)

## Workflow sketch

Numbered phases. Each phase: tool call(s) + report-and-ask checkpoint.

1.
2.
3.

## Source-of-truth references

Repo docs the skill would cite (by relative path). Do not duplicate.

-
-

## Are you proposing to author this, or asking the team to?

- [ ] I'll author the PR.
- [ ] I'm asking a maintainer to author it.
