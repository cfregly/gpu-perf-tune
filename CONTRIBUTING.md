# Contributing to `profile-and-optimize`

Thanks for considering a contribution. This repo ships **agent skills** — markdown files following the open [Agent Skills standard](https://agentskills.io/) — packaged as a Claude Code plugin marketplace. The same skills work in Claude Code and Cursor.

This document covers:

- How to add a new skill.
- How to add or change a bundled MCP server verb.
- Version bump rules.
- Validation checklist.
- What gets reviewed.

## TL;DR new contributor flow

```bash
git clone git@github.com:<your-org>/claude-perf-tune.git
cd profile-and-optimize

# Add a new skill.
mkdir -p plugins/profile-and-optimize/skills/<your-skill-name>
$EDITOR plugins/profile-and-optimize/skills/<your-skill-name>/SKILL.md

# Bump the version (PATCH for fixes, MINOR for new skills, MAJOR for breaks).
$EDITOR plugins/profile-and-optimize/.claude-plugin/plugin.json    # set version
$EDITOR README.md                                       # add the skill to its family list

# Validate.
make all     # or: make -j4 all for the parallel ~1.5s wall-clock run

# Open a PR.
git checkout -b add-<skill-name>
git commit -am "add <skill-name> skill"
gh pr create
```

The PR template prompts you for the rest; run `make -j4 all` locally before every push.

## Local setup (one-time)

The bundled MCP server ships as an editable Python package. To run the test suite or pre-commit hooks locally:

```bash
# Install the bundled server with the `dev` extras (pytest, ruff, pre-commit, pyright, pytest-xdist).
bash plugins/profile-and-optimize/server/install.sh --with-dev

# Wire pre-commit so `git commit` runs ruff + check-doc-links + a few file-hygiene
# checks on staged files. Optional; skip if you'd rather run `make all` manually.
plugins/profile-and-optimize/server/.venv/bin/pip install pre-commit  # if not already on PATH
pre-commit install

# Verify everything's healthy:
make -j4 all   # smoke-test + smoke-mcp-runtime + check-doc-links + pytest (parallel; ~1.5s wall-clock)
```

The pre-commit config (`.pre-commit-config.yaml`) intentionally scopes its ruff hooks to a subset of server subtrees (see the `files:` filters in the config) so it doesn't fight vendored upstream code style.

## Release ritual

After every release (any version bump that lands on `main` + gets a `gh release create`), run the two-command refresh so both the Claude Code marketplace cache and the Cursor symlinks pick up the new version:

```bash
# Refresh the Claude Code marketplace cache (~/.claude/plugins/cache/profile-and-optimize-plugins/profile-and-optimize/<new-version>/).
claude plugin update profile-and-optimize@profile-and-optimize-plugins

# Refresh the per-operator Cursor skill symlinks under ~/.cursor/skills/.
make refresh-symlinks
```

`make refresh-symlinks` wraps [`scripts/install-skills-into-cursor.sh`](/scripts/install-skills-into-cursor.sh); it is idempotent and prints a summary like `Summary: <N> linked, <M> already-linked (skipped), 0 refused`. The Cursor symlinks point at the in-repo `plugins/profile-and-optimize/skills/<skill>/` directories, so re-running after a `git pull` is enough to surface any SKILL.md changes that landed in the new version.

`make refresh-symlinks` is deliberately **not** wired into `make all` — `make all` is the local gate you run before pushing, and it needs to stay portable across workstations where `~/.cursor/` may not exist.

### Cursor upgrade sequence (one shot)

If you run the bundled MCP server from a dev clone (not the Claude Code cache), there is one more thing to refresh after a version bump: the `profile_and_optimize` entry in `~/.cursor/mcp.json` points at the bundled venv, and a moved/rebuilt checkout can leave it stale (the recurring `spawn ... ENOENT` / `No module named profile_and_optimize_mcp` failure — see [`docs/cursor-mcp-troubleshooting.md`](/docs/cursor-mcp-troubleshooting.md)). The full sequence:

```bash
git pull origin main                 # refresh the dev clone
claude plugin update profile-and-optimize@profile-and-optimize-plugins   # refresh the Claude Code cache
make bootstrap                       # re-install the server venv + refresh symlinks + print the mcp.json snippet
                                     # (add FULL=1 for the perf_tune_report renderer deps, DEV=1 for pytest)
make doctor                          # read-only: is the ~/.cursor/mcp.json profile_and_optimize entry still valid?
make doctor FIX=1                    # if STALE: repoint it to this checkout's venv (backs up mcp.json first)
```

`make doctor` only ever inspects/edits the `profile_and_optimize` entry; every other server in `~/.cursor/mcp.json` is left untouched, and `FIX=1` writes a timestamped `~/.cursor/mcp.json.bak-*` before changing anything. Then reload Cursor (toggle the `profile_and_optimize` MCP off/on, or restart) to pick up the repoint.

### Pre-tag safety check

Before bumping `plugin.json` + tagging a new version, always run:

```bash
git fetch origin
git status -sb              # if ahead != 0 OR behind != 0, STOP
git ls-remote --tags origin <target-tag>   # if non-empty, the tag is already taken
```

If `git status -sb` shows `behind > 0` or the target tag exists on origin (someone else pushed it first), DO NOT proceed with the same tag. Either rebase + bump the patch version, OR follow the recovery procedure in [`docs/release-tag-recovery.md`](/docs/release-tag-recovery.md).

## How to add a new skill

### 1. Scope the skill

A good skill is:

- **One task** — a single coherent workflow with a clear start and end.
- **Iterative** — pauses for operator input at every gate; never auto-advances past a red.
- **Trigger-discoverable** — the `description` frontmatter lists the phrases an operator would naturally type to invoke this skill.
- **Read-only by default** — mutating actions require an explicit ack flag (fail fast, no silent fallbacks).

A bad skill is:

- A vague "helper" — three different tasks bundled into one. Split into three skills.
- A wrapper around a single tool with no value-add — just use the tool directly.
- A skill that duplicates incident-triage or general ops tooling — this repo is scoped to inference profiling and optimization.

Before writing: search the [`plugins/profile-and-optimize/skills/`](/plugins/profile-and-optimize/skills) directory for overlap. If your task is incident-triage-shaped rather than perf-shaped, it probably belongs in a different plugin.

### 2. Pick a name

- Lowercase, hyphens, max 64 chars. The directory name and the `name:` frontmatter field MUST match.
- Verb-or-noun phrase, not a gerund. Good: `inference-perf-bench`, `perf-baseline-record`. Bad: `benching-inference-perf`, `recording-perf-baselines`.
- Prefix with the workload family if the skill is family-specific. Good: `inference-perf-bench`. Bad: `perf-bench` (loses the inference-vs-generic discriminator).

### 3. Write `SKILL.md`

Start from the template at [`plugins/profile-and-optimize/skills/_template/SKILL.md`](/plugins/profile-and-optimize/skills/_template/SKILL.md). Every skill must include:

- YAML frontmatter: `name` (matches dir), `description` (<=1024 chars, third-person, includes WHAT + WHEN trigger phrases), `allowed-tools` (YAML list of specific tool selectors).
- `## Purpose` — what the skill does and why, in plain English.
- `## When to use` and `## Example prompts` — trigger discoverability.
- `## Prerequisites` — env vars, repo paths, Slurm reservations, etc. Fails closed.
- `## Interaction style` — iterative pattern (one step, report, ask).
- `## Workflow` — numbered phases. Each phase is a tool call sequence with a `report and ask` checkpoint.
- `## Safety` — ack flags, fail-closed gates, forbidden actions.
- `## Source-of-truth references` — cite repo docs by relative path; never duplicate.

Keep the body under 500 lines. Use progressive disclosure (link to sibling files for deep reference content).

If the skill emits human-facing prose (a report, PR body, or summary), keep the template's "Keep it tight (no AI-slop)" block; the de-slop checklist is canon in [`docs/METHODOLOGY.md`](/docs/METHODOLOGY.md) ("De-slop").

### 4. Update the README

Add the skill to the appropriate family list in the root [`README.md`](/README.md) "What this is" section, and update the skill-count line if the total changed.

### 5. Bump the plugin version

| Change type | Bump | Example |
| --- | --- | --- |
| New skill, new MCP server in `.mcp.json`, expanded workflow in an existing skill | MINOR | `0.3.0` -> `0.4.0` |
| Typo, link fix, clarification in a SKILL.md without behavior change | PATCH | `0.3.0` -> `0.3.1` |
| Renamed skill, removed skill, renamed MCP server key, MCP envelope semantic change, ack-flag semantic change | MAJOR | `0.3.0` -> `1.0.0` |

Edit the `version` field in [`plugins/profile-and-optimize/.claude-plugin/plugin.json`](/plugins/profile-and-optimize/.claude-plugin/plugin.json), then describe the change in your PR description and the release notes.

### 6. Validate locally

```bash
claude plugin validate plugins/profile-and-optimize
```

Must return `Validation passed`. If it fails, the PR will fail review.

Additionally, every SKILL.md in your PR should pass these checks:

```bash
python3 -c "
import yaml, glob
for f in glob.glob('plugins/profile-and-optimize/skills/*/SKILL.md'):
    body = open(f).read()
    assert body.startswith('---\n'), f'{f}: missing frontmatter'
    end = body.find('\n---\n', 4)
    assert end > 0, f'{f}: frontmatter not closed'
    fm = yaml.safe_load(body[4:end])
    assert fm['name'] == f.split('/')[-2], f'{f}: name mismatch'
    assert len(fm['description']) <= 1024, f'{f}: description too long'
    assert isinstance(fm.get('allowed-tools', []), list), f'{f}: allowed-tools must be list'
print('all skills OK')
"
```

And no Windows paths:

```bash
rg -l '\\\\' plugins/  # must be empty
```

### 7. Open a PR

Push your branch and open a PR. The [`PULL_REQUEST_TEMPLATE.md`](/.github/PULL_REQUEST_TEMPLATE.md) checklist walks you through everything reviewers expect.

## MCP tool naming convention

Skills reference MCP tools via `mcp__<server-key>__<tool-name>` in the `allowed-tools` frontmatter. The server keys MUST match what's declared in [`plugins/profile-and-optimize/.mcp.json`](/plugins/profile-and-optimize/.mcp.json):

| Server | Key in `.mcp.json` | Tool prefix in skill frontmatter |
| --- | --- | --- |
| Bundled `profile_and_optimize` MCP (8 libraries; 53 tools total = 51 contract-derived + 2 auxiliary; canonical numbers in [`mcp_surface.py`](/plugins/profile-and-optimize/server/mcp_surface.py)) | `profile_and_optimize` | `mcp__profile_and_optimize__<tool>` (e.g. `mcp__profile_and_optimize__slurm_triage`, `mcp__profile_and_optimize__perf_tune_report_report_render`, `mcp__profile_and_optimize__perf_tune_report_publish_to_lake`) |
| Grafana | `grafana` | `mcp__grafana__<tool>` |
| GitHub | `github` | `mcp__github__<tool>` |
| Prometheus (optional) | `prometheus_mcp` | `mcp__prometheus_mcp__<tool>` |
| zymtrace (optional) | `zymtrace` | `mcp__zymtrace__<tool>` |

Only the first three (`profile_and_optimize`, `grafana`, `github`) are declared in [`plugins/profile-and-optimize/.mcp.json`](/plugins/profile-and-optimize/.mcp.json). The optional servers are configured per-operator — add them to your own `~/.cursor/mcp.json` or `~/.claude/settings.json` if you use them; skills that reference an optional server fall back to the documented bash-tool path when it is absent.

If you need a new MCP server not in this list, add it to `.mcp.json` first (with env-var placeholders for tokens / URLs — never check in real tokens) and update this table.

## How to add or change a bundled MCP verb

The `plugins/profile-and-optimize/server/` directory is the **source of truth** for the bundled MCP server. Adding a new verb is direct:

1. Create a new library directory under `plugins/profile-and-optimize/server/<library-name>/` with `__init__.py`, `__main__.py`, and `cli.py`. The `cli.py` defines a `CONTRACT` dict keyed by verb name (see [`server/perf_baseline/cli.py`](/plugins/profile-and-optimize/server/perf_baseline/cli.py) for a minimal template).
2. Implement the verbs under `plugins/profile-and-optimize/server/tools/<library-name>/` and have the stub `cli.py` import + delegate (matches the pattern used by every existing library).
3. Add the new library to `LIBRARIES` in [`plugins/profile-and-optimize/server/mcp_surface.py`](/plugins/profile-and-optimize/server/mcp_surface.py).
4. Add the new library to `packages.find.include` in [`plugins/profile-and-optimize/server/pyproject.toml`](/plugins/profile-and-optimize/server/pyproject.toml).
5. Add a row to the "What lives here" table in [`plugins/profile-and-optimize/server/AGENTS.md`](/plugins/profile-and-optimize/server/AGENTS.md) and to the corresponding library table in [`plugins/profile-and-optimize/server/README.md`](/plugins/profile-and-optimize/server/README.md).
6. **Update the canonical-counts constants** in [`plugins/profile-and-optimize/server/mcp_surface.py`](/plugins/profile-and-optimize/server/mcp_surface.py) (`_TOTAL_LIBRARIES`, `_TOTAL_CONTRACT_TOOLS`, `_TOTAL_MCP_TOOLS`). The `lint-tool-counts` gate reads these constants and fails the build if any doc names a different number. Run `make lint-tool-counts` locally to see the current expected vs reported numbers.
7. Bump version: PATCH if the verb is operator-internal, MINOR if it's a new tool surface, MAJOR if you removed or renamed an existing verb.
8. Run `python3 plugins/profile-and-optimize/server/mcp_surface.py counts` to confirm the canonical-counts module agrees with the live derivation, and `python3 plugins/profile-and-optimize/server/mcp_surface.py list` to see the new tool name in the surface.


## What gets reviewed

[`REVIEWERS.md`](/REVIEWERS.md) covers the reviewer-side workflow in depth. Short version: reviewers check that

- `claude plugin validate` passes locally;
- the version bump matches the change scope;
- the description triggers are specific and not vague;
- the safety section enumerates ack flags and forbidden actions;
- no source-of-truth duplication;
- the skill is one task, not three.

## Code of conduct

Be respectful and constructive in issues and reviews. Contributions are credited through the git author line; keep skill content impersonal — skills describe workflows, not individual ownership.

## Contact

Open an issue with the [`question.md`](/.github/ISSUE_TEMPLATE/question.md) template.
