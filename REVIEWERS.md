# Reviewing `profile-and-optimize` PRs

A 30-minute reviewer path for `profile-and-optimize` pull requests.

## TL,DR

- **Run** `claude plugin validate plugins/profile-and-optimize` locally. Must pass.
- **Read** the new / changed SKILL.md files end-to-end. Check the frontmatter, the workflow phases, and the safety section.
- **Confirm** the version bump matches the change scope (PATCH vs. MINOR vs. MAJOR).
- **Confirm** the README skill listings were updated.
- **Confirm** no source-of-truth duplication.

If all five are green, approve. The rest of this doc is the why.

## What's authoritative, what's derived, what's historical

| Path | What it is | Reviewer treatment |
| --- | --- | --- |
| `plugins/profile-and-optimize/skills/*/SKILL.md` | The skills themselves. The reviewable unit of the repo. | Read end-to-end. This is most of the PR review. |
| `plugins/profile-and-optimize/.claude-plugin/plugin.json` | Plugin manifest (`name`, `version`, `description`). | Check `version` bump matches the change scope. |
| `plugins/profile-and-optimize/.mcp.json` | MCP server declarations. | Check that any new `mcp__<server>__<tool>` reference in a skill has a corresponding server here. |
| `plugins/profile-and-optimize/README.md` | Plugin-level README. | Check it stays accurate when skills or server libraries change. |
| `plugins/profile-and-optimize/server/` | Bundled MCP server. | All 8 CLI libraries (`ai_tuning`, `profile`, `perf_baseline`, `evidence`, `slurm`, `findings`, `perf_tune_report`, `known_good_config`) are first-class code and get the same review depth as the skills. Canonical counts (8 libraries, 53 tools = 51 contract + 2 aux) live in [`mcp_surface.py`](/plugins/profile-and-optimize/server/mcp_surface.py)'s `_TOTAL_*` constants and are asserted by `make lint-tool-counts`. |
| `.claude-plugin/marketplace.json` | Marketplace registry. | Rarely changes. If it does, confirm the plugin source path is correct. |
| `LICENSE`, `CONTRIBUTING.md`, `REVIEWERS.md` | Static. | No review unless explicitly changed. |
| `.github/` templates | Static. | No review unless explicitly changed. |
| `scripts/` helpers (e.g. `install-skills-into-cursor.sh`, `bootstrap.sh`) | Static operator-facing helpers. | Read once when introduced. Subsequently only re-review when explicitly changed. |

## Version bump scope

The MINOR / MAJOR distinction is the single most-confusing thing for new contributors. Reviewer is the final check.

| Change | Version | Why |
| --- | --- | --- |
| New skill added | MINOR | Skill is a feature. Consumers gain capability. |
| Existing skill's workflow extends (new phase, new MCP tool used) | MINOR | Operator sees changed behavior. Consumers may rely on the new behavior. |
| Existing skill's prose tightens, typo, link fix | PATCH | No behavior change. |
| MCP server added to `.mcp.json` | MINOR | New external dependency surfaces. |
| MCP server removed from `.mcp.json` | MAJOR | Breaking - any skill that referenced it is now broken. |
| Skill removed | MAJOR | Breaking - `/<skill-name>` slash command no longer exists. |
| Skill renamed | MAJOR | Breaking. Even if you add an alias, the old name disappears. |
| MCP tool prefix renamed (e.g. `prometheus` -> `prometheus_mcp`) | MAJOR for cross-version compat, MINOR if it was a typo fix in the same release as the introduction. |
| Bundled server contract / tool surface changed | PATCH if no tool surface change. MINOR if new tools / verbs. MAJOR if removed / renamed tools. |

If the reviewer disagrees with the version bump in the PR, request the bump first. Do not approve.

## What to look for in a SKILL.md review

### Frontmatter

- `name` matches the directory name. Lowercase, hyphens, max 64 chars.
- `description` is third-person, <=1024 chars, includes both WHAT and WHEN trigger phrases. Specific trigger phrases (`"run a perf bench sweep"`, `"profile decode kernels"`), not vague ones (`"helps with perf"`).
- `allowed-tools` is a YAML list. Each entry is either a literal MCP tool (`mcp__<server>__<tool>`) or a Bash selector (`Bash(sbatch:*)`) or a Cursor primitive (`Read`, `Write`, `Grep`).

### Body

- Body <=500 lines. Use progressive disclosure (link to sibling files) for deep reference.
- All file references one-level-deep, relative paths only, no Windows backslashes.
- The Workflow section is numbered phases. Each phase has a clear `report and ask` checkpoint. Never auto-advance past a red gate.
- The Safety section enumerates ack flags + forbidden actions + fail-closed conditions. If the skill could mutate cluster state, the ack flag is explicit.
- The Source-of-truth references section cites repo docs by relative path. The skill should NOT duplicate the cited content (no copy-paste of runbook stanzas).

### Workflow shape

- **Iterative** (pause and ask between phases) is the default. Auto-advance is OK only for fast, fully read-only flows.
- **Fail-closed** on every prerequisite. If `${PROFILE_AND_OPTIMIZE_REPO_ROOT}` is missing, the skill reports and stops. It does not silently `mkdir -p` a fake path.
- **Knowledge-base anchoring first** for any PromQL skill (per [`server/docs/mcp-composition.md`](/plugins/profile-and-optimize/server/docs/mcp-composition.md) "Default Routing").
- **Raw-payload preservation** for any skill that queries external systems (per [`server/docs/perf-lake-contract.md`](/plugins/profile-and-optimize/server/docs/perf-lake-contract.md)).
- **Reproducibility-grade evidence** for any skill that writes artifacts: `SOURCE.md`, `summary.md`, four-file `commands/` tuple capture (see the [`evidence-bundle-init`](/plugins/profile-and-optimize/skills/evidence-bundle-init/SKILL.md) skill).

## Hard safety gates

Reviewer must reject any PR that:

- Adds a soft-pass path around a fail-closed gate.
- Auto-passes the `i_understand_this_submits_jobs` ack flag (or any other `i_understand_this_*` ack) without explicit operator confirmation in the workflow.
- Weakens a verification invocation (dropping strict flags, loosening thresholds) without calling the change out in the PR description.

## What `WARN`-class lint issues are OK to land

- A typo or link drift discovered in CI on the same PR - fix it inline. No separate PR needed.
- A new SKILL.md that exceeds 500 lines by 5-10 - request progressive-disclosure refactor (move detail to a sibling file), but do not block if the skill is otherwise clean.

## What ERROR-class issues block

- `claude plugin validate` fails.
- Frontmatter YAML doesn't parse.
- `name` field doesn't match directory.
- `description` > 1024 chars.
- Windows paths in skill body.
- Repo source-of-truth duplication (skill copy-pastes a runbook stanza instead of linking).
- Missing version bump.
- Chat-write tool referenced (skills are read-only toward chat systems. E.g. no `slack_send_message`).

## Reviewer template comment

When approving:

```
LGTM. Validated:
- claude plugin validate: PASS
- frontmatter: name matches dir, description <=N chars, allowed-tools list of M
- version bump matches scope: <PATCH|MINOR|MAJOR>
- README skill listings updated: yes / no (n/a for non-skill PR)
- No source-of-truth duplication
- Safety section reviewed
```

When requesting changes:

```
Two questions before approve:
1. <specific question with file:line cite>
2. <specific question with file:line cite>

The rest of the PR looks clean; rerun claude plugin validate after the fix and I'll re-review.
```

## Contact

For reviewer-side process questions, open an issue using [`.github/ISSUE_TEMPLATE/question.md`](/.github/ISSUE_TEMPLATE/question.md).
