<!--
Thanks for the PR. Fill in the sections below. The reviewer will use them
verbatim. See CONTRIBUTING.md and REVIEWERS.md for the full process.
-->

## What this PR does

<!-- One-paragraph summary: what changed and why. Link to the issue if one exists. -->

## Scope of change

- [ ] New skill(s) added (lists below)
- [ ] Existing skill(s) modified
- [ ] MCP server added / changed in `.mcp.json`
- [ ] Bundled server contract or tool surface changed
- [ ] README / CONTRIBUTING / REVIEWERS / docs only
- [ ] Other (describe)

## New / changed skills

<!--
For each new or changed skill, list:
- the skill name (matches directory)
- one-line summary of behavior change
- which MCP tools it adds or removes (compared to what was there before)
-->

-

## Version bump

Current: `0.X.Y` -> Proposed: `0.A.B`

Justification (PATCH / MINOR / MAJOR per [REVIEWERS.md](/REVIEWERS.md#version-bump-scope)):

<!-- Why this bump? -->

## Validation checklist

- [ ] `claude plugin validate plugins/profile-and-optimize` returns PASS.
- [ ] Every new / changed SKILL.md has valid YAML frontmatter (`name` matches directory, `description` <=1024 chars, `allowed-tools` is a list).
- [ ] Skill bodies are <=500 lines (use progressive disclosure for deep reference).
- [ ] No Windows-style paths (`\\`) in any SKILL.md.
- [ ] All file references are one-level-deep and use relative paths.
- [ ] Source-of-truth docs in `server/` are cited, not duplicated.
- [ ] No `slack_send_message` / `slack_schedule_message` / other chat-write tool referenced (skills are read-only toward chat systems).
- [ ] If any new mutating MCP tool is referenced, the corresponding `i_understand_this_*` ack flag is enforced in the workflow.
- [ ] Root `README.md` updated: skill family list, plus the skill-count line if the total changed.
- [ ] If a new MCP server was added: env-var placeholders only. No real tokens / URLs checked in.
- [ ] If the bundled server tool surface changed: `make smoke-test` confirms `mcp_surface.py` derives the expected tool count.

## Optional but appreciated

- [ ] Local install test: `claude plugin update profile-and-optimize@profile-and-optimize-plugins` succeeds and the new / changed skills appear in `claude plugin list`.
- [ ] If a new perf-test skill: ran the bash workflow against a real cohort and captured an evidence bundle in the PR description.

## Notes for reviewers

<!--
Anything reviewers should know before reading: design tradeoffs you chose, alternatives you rejected, follow-ups you deferred to a later PR.
-->
