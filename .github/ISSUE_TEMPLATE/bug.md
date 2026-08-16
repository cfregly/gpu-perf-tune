---
name: Bug report
about: A skill misbehaves, an MCP tool returns the wrong thing, the bundled server fails to install, or anything else that should work but doesn't.
title: "bug: <skill-name>: <short symptom>"
labels: bug
---

This issue is public. Remove tokens, credentials, customer data, internal
hostnames, private URLs, private repository names, proprietary prompts, and
unredacted logs. Report security concerns through the private process in
[SECURITY.md](https://github.com/cfregly/gpu-perf-tune/blob/main/SECURITY.md).

## What happened

<!-- One paragraph. -->

## What you expected to happen

<!-- One paragraph. -->

## Reproduction

The exact prompt or command you typed, using synthetic or redacted values:

```
<paste here>
```

The exact response you got (or the place it stalled):

```
<paste here>
```

## Environment

- Exact profile-and-optimize version or commit: `<version or commit>`
- AI client and version: `Claude Code | Codex CLI | Cursor | Gemini CLI | other`
- OS: `<macOS|Linux distro + version>`
- Environment type: `personal laptop | shared workstation | CI | cluster login host`
- Bundled server installed? (`server/.venv/bin/python -m profile_and_optimize_mcp --help` works): `yes / no`
- Any MCP server env vars unset? (e.g. `PROMETHEUS_MCP_URL` empty causes the skill to skip a phase): `list them`

## Bundle / evidence

If the bug produced an artifact under `experiments/artifacts/`, attach only the
smallest sanitized excerpt needed to reproduce the problem. Do not upload a raw
bundle from a private environment.

```
experiments/artifacts/<family>/<run-id>/
```

If it produced a Slurm job, paste a sanitized
`sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,Reason` line.

## Have you checked

- [ ] The skill's `Prerequisites` section in its SKILL.md.
- [ ] The skill's `Safety` section (the bug might be a fail-closed gate firing as designed).
- [ ] [REVIEWERS.md](https://github.com/cfregly/gpu-perf-tune/blob/main/REVIEWERS.md) for whether the symptom matches a known WARN-class lint vs. ERROR-class issue.
