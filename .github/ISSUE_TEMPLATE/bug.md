---
name: Bug report
about: A skill misbehaves, an MCP tool returns the wrong thing, the bundled server fails to install, or anything else that should work but doesn't.
title: "bug: <skill-name>: <short symptom>"
labels: bug
---

## What happened

<!-- One paragraph. -->

## What you expected to happen

<!-- One paragraph. -->

## Reproduction

The exact prompt or slash-command you typed:

```
<paste here>
```

The exact response you got (or the place it stalled):

```
<paste here>
```

## Environment

- profile-and-optimize version (from `plugins/profile-and-optimize/.claude-plugin/plugin.json`): `0.X.Y`
- Claude Code version (`claude --version`): `X.Y.Z`
- Cursor version (if reproduced in Cursor): `X.Y.Z`
- OS: `<macOS|Linux distro + version>`
- Workstation: `<hostname>` (or "personal laptop", "shared workstation", etc.)
- Bundled server installed? (`server/.venv/bin/python -m profile_and_optimize_mcp --help` works): `yes / no`
- Any MCP server env vars unset? (e.g. `PROMETHEUS_MCP_URL` empty causes the skill to skip a phase): `list them`

## Bundle / evidence

If the bug produced an artifact under `experiments/artifacts/`, attach the path:

```
experiments/artifacts/<family>/<run-id>/
```

If it produced a Slurm job, paste the `sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,Reason,NodeList` line.

## Have you checked

- [ ] The skill's `Prerequisites` section in its SKILL.md.
- [ ] The skill's `Safety` section (the bug might be a fail-closed gate firing as designed).
- [ ] [REVIEWERS.md](/REVIEWERS.md) for whether the symptom matches a known WARN-class lint vs. ERROR-class issue.
