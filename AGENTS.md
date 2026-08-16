# AGENTS.md

Canonical guidance for coding agents working in `gpu-perf-tune`.
Client-specific files may point here, but shared policy belongs in this file.

## Run it

```bash
make demo     # print the tool and skill surface, no GPU needed
make help     # list operator targets
make all      # run smoke checks, doc links, skill lint, and pytest
```

The repository layout, client setup, and methodology live in
[`README.md`](README.md) and [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Project names and client boundaries

- `gpu-perf-tune` is the project and repository.
- `profile-and-optimize` is the Agent Skills package. Claude Code consumes it
  as a plugin. Codex installs the same skills from a clone.
- `profile_and_optimize` is the MCP server key.
- Claude Code has first-class Skills, MCP, and opt-in provenance enforcement.
- Codex has first-class Skills and MCP setup. It uses this `AGENTS.md` for
  project guidance and has no packaged client hook.
- Cursor, Gemini CLI, and Google Antigravity helpers are best effort. Do not
  treat those clients as release gates or claim full adapter support.

Keep a Claude-specific instruction in a Claude adapter when the instruction is
truly client-specific. Put project policy here.

## Behavioral guidelines

These rules reduce common agent mistakes. They favor caution over speed. Use
judgment for trivial work.

### 1. Think before coding

Do not hide assumptions or uncertainty.

Before implementing:

- State assumptions that affect the result.
- Present materially different interpretations before choosing one.
- Name a simpler approach when it exists.
- Stop and ask when missing context would change the outcome.

### 2. Keep the solution small

Write the minimum code that solves the requested problem.

- Do not add speculative features.
- Do not build an abstraction for one use.
- Do not add configurability that no caller needs.
- Rewrite a large change when a much smaller one would be clearer.

Ask whether a senior engineer would call the change overcomplicated. If so,
simplify it.

### 3. Make surgical changes

Touch only what the task needs.

- Do not reformat or refactor unrelated code.
- Match the local style.
- Preserve user changes and unrelated worktree state.
- Remove only the imports, variables, and functions made obsolete by your
  change.
- Report unrelated dead code instead of deleting it.

Every changed line should trace to the request.

### 4. Work toward a verifiable result

Translate the task into checks that can pass or fail.

- "Add validation" means test invalid input, then make the test pass.
- "Fix the bug" means reproduce it, then prove the fix.
- "Refactor" means keep the same behavior before and after the change.

For a multi-step task, state a short plan with a verification step.

### 5. Apply the value bar

Measured is not automatically valuable.

Every benchmark, report, skill change, and README claim must be
adversarially-confirmed to add value before promotion. Name the workload,
baseline, skeptical check, receipt, and user-facing value. If that chain is
missing, label the result DRAFT or candidate.

### 6. Estimate, then measure

For performance work, apply the
[`performance hints adaptation`](plugins/profile-and-optimize/server/docs/performance-hints.md):

- Classify the path as setup, per request, per token, or shared library work.
- Write a rough work times unit-cost ledger and state overlap assumptions.
- Rank changes by measured contributor share and maximum possible impact.
- Prefer less work, bulk operations, compact representations, fewer copies,
  reuse, and less synchronization before instruction-level tuning.
- Re-run the production-shaped baseline after each change. Estimates stay
  DRAFT.

## Reproducibility-Grade Evidence

Significant performance work produces an evidence bundle, not loose output.
Record the source revision, operator, environment, exact commands, stdout,
stderr, exit codes, raw measurements, and summary in the bundle. A future
reviewer must be able to identify what ran and decide whether the claim follows
from the captured data.

## Experiment Isolation and Traceability

Use the run ID as the experiment ID and the join key across evidence, cluster
objects, and published results. Give experiment resources unique names and the
matching `experiment=<run-id>` label. Do not reuse evidence across workloads,
source revisions, or delivery methods.

## Working in this repository

- Run `make all` before committing. Use `make -j4 all` to run independent
  targets in parallel.
- Skill, tool, and library counts are canonical in
  [`plugins/profile-and-optimize/server/mcp_surface.py`](plugins/profile-and-optimize/server/mcp_surface.py).
  Never hardcode a replacement count without updating and running the count
  lints.
- Start a new skill from
  [`plugins/profile-and-optimize/templates/skill/SKILL.md`](plugins/profile-and-optimize/templates/skill/SKILL.md).
- Measurement rules live in
  [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).
- MCP request, response, and acknowledgement rules live in
  [`plugins/profile-and-optimize/server/docs/mcp-tool-io-contract.md`](plugins/profile-and-optimize/server/docs/mcp-tool-io-contract.md).
- The root [`VERSION`](VERSION) file is the release version source of truth.
  Keep adapter and package versions plus the changelog aligned. Every release
  uses an annotated `vX.Y.Z` tag. A successful main push CI run triggers the
  release workflow. The version bump must be in the final squash-merged commit.
  Use `make release` only to recover a missing release from the current,
  already-green main version bump.
- Write human-facing text in plain language. Do not use em dashes, en dashes,
  prose semicolons, inflated claims, or filler.
