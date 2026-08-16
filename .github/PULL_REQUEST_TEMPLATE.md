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

Justification (PATCH / MINOR / MAJOR per [REVIEWERS.md](https://github.com/cfregly/gpu-perf-tune/blob/main/REVIEWERS.md#version-review)):

<!-- Why this bump? -->

## Validation checklist

- [ ] `make all` returns PASS.
- [ ] Every new or changed skill passes `make validate-agent-skills`.
- [ ] Every new or changed SKILL.md uses only supported frontmatter fields, and `allowed-tools` is a space-delimited string.
- [ ] Long skills move stable reference material into sibling files when that improves discovery and loading.
- [ ] No Windows-style paths (`\\`) in any SKILL.md.
- [ ] Source-of-truth docs are cited, not copied into the skill.
- [ ] No `slack_send_message` / `slack_schedule_message` / other chat-write tool referenced (skills are read-only toward chat systems).
- [ ] If any new mutating MCP tool is referenced, the corresponding `i_understand_this_*` ack flag is enforced in the workflow.
- [ ] Root `README.md` updated: skill family list, plus the skill-count line if the total changed.
- [ ] If a new MCP server was added: env-var placeholders only. No real tokens / URLs checked in.
- [ ] If the bundled server tool surface changed: `make smoke-test` confirms `mcp_surface.py` derives the expected tool count.

## Adapter checks

- [ ] Not applicable because Claude packaging did not change.
- [ ] `make validate-claude-plugin` passes because Claude packaging changed.
- [ ] A dry run of each changed client installer writes nothing and prints no existing private config.
- [ ] For a new performance skill, the PR states what ran on real hardware and links sanitized evidence, or explains what remains unvalidated.

## Notes for reviewers

<!--
Anything reviewers should know before reading: design tradeoffs you chose, alternatives you rejected, follow-ups you deferred to a later PR.
-->
