# Claude Code guidance

Read and follow [`AGENTS.md`](AGENTS.md). It is the canonical guidance for every
coding agent working in this repository.

`profile-and-optimize` is the Claude Code adapter for the broader
`gpu-perf-tune` project. Keep Claude-specific packaging and hook instructions
in that adapter. Keep shared project policy in `AGENTS.md`.

The shared value bar is that every promoted result must be
adversarially-confirmed to add value.

## Health Stack

- typecheck: make typecheck
- lint: make lint
- test: make pytest
- shell: make lint-shell
