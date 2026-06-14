# AGENTS.md

Guidance for AI coding agents working in this repository (the canonical copy).
AGENTS.md-aware tools (Codex CLI, Gemini CLI, Cursor, ...) read it natively,
Claude Code loads it via the one-line `@AGENTS.md` import in
[`CLAUDE.md`](CLAUDE.md). Cursor additionally pins it always-on via
[`.cursor/rules/coding-guidelines.mdc`](.cursor/rules/coding-guidelines.mdc).

## Behavioral guidelines

Behavioral guidelines to reduce common LLM coding mistakes.
Source: Public coding-guidelines baseline.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan with a verify step per item. Strong
success criteria let you loop independently. Weak criteria ("make it work")
require constant clarification.

## Working in this repo

- Run `make all` before committing: smoke-test, MCP runtime smoke, doc-link
  check, skill-arg lint, and pytest in one pass (`make -j5 all` runs them in
  parallel).
- Skill / tool / library counts are canonical in
  [`plugins/profile-and-optimize/server/mcp_surface.py`](plugins/profile-and-optimize/server/mcp_surface.py)
  (`_TOTAL_*` constants). The lint gates fail any doc that names a different
  number. Never hardcode a count in a doc.
- Measurement rigor rules live in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)
  (DRAFT-vs-VERDICT labeling, full-context perf numbers, asset validation).
- De-slop writing rules live in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)
  ("De-slop"): every human-facing artifact (report, PR body, ledger row,
  commit message) is written plain. No em-dashes, minimal bold, plain
  punctuation, numbers over adjectives, no marketing language.
