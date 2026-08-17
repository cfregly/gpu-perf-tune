# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-08-16

### Added
- Added repository-wide Ruff, Pyright, and ShellCheck gates to `make all`,
  pre-commit, and CI.
- Added an upgrade guide, a Claude Code MCP health check, and a Codex MCP
  registration check.

### Security
- Kept Artificial Analysis API keys out of subprocess arguments, command
  receipts, standard output, standard error, CLI JSON, and MCP responses.
- Rejected custom AIPerf commands that embed `--api-key` before they can run or
  create evidence artifacts.

### Changed
- Required an explicit `--client` in the direct MCP installer and client
  configuration helper. The installer no longer writes Cursor configuration
  when the caller omits a client.
- Updated the pinned Artificial Analysis dependency closure and held the MCP
  runtime below version 2 pending a deliberate compatibility migration.

### Fixed
- Corrected production type contracts across the server, hooks, skills, and
  repository scripts.
- Fixed a variable-shadowing bug that could corrupt K-suffixed benchmark cell
  identifiers when variant metadata was present.
- Documented the search-tool fallback when `rg` is not installed.

## [0.3.1] - 2026-08-16

### Security
- Restricted the standalone TTFO probe to loopback HTTP targets, an exact API
  path, explicit unprivileged ports, and requests that do not follow redirects.
- Hardened operator-controlled path handling found by the first CodeQL scan,
  including derived S3 object keys and campaign lookup slugs.
- Replaced secret-shaped installer test fixtures with inert markers.

### Changed
- Added focused tests for validated probe URLs, safe derived paths, and explicit
  local CLI file inputs.

## [0.3.0] - 2026-08-16

### Added
- First-class onboarding for Claude Code and Codex, including Codex Agent
  Skills installation. Other stdio MCP client helpers remain best-effort.
- A canonical `AGENTS.md`, official Agent Skills validation, installer tests,
  full Markdown link checks, and a Python 3.11 through 3.14 CI matrix.
- Private security reporting guidance, CODEOWNERS, Dependabot configuration,
  issue intake controls, and third-party notices.
- An idempotent release workflow for annotated tags and GitHub Releases.

### Changed
- Positioned `gpu-perf-tune` as the project, with the Claude Code plugin as one
  adapter for the `profile-and-optimize` skills package.
- Replaced agent-specific runtime discovery markers with package files.
- Pinned GitHub Actions, the kernel profiling image, and the Agent Skills
  reference validator to immutable revisions.
- Made the contributor checks independent of the Claude Code CLI.
- Removed private AI tuning defaults. Matrix, proposal, report, validation, and
  experiment creation commands now require an explicit `--space`. MLPerf rules
  validation now requires an explicit `--rules` file.
- Rejected acknowledgement flags embedded in raw MCP arguments. Job submission
  now requires the structured `i_understand_this_submits_jobs=true` field, and
  external publication requires
  `i_understand_this_publishes_externally=true`.

### Fixed
- Made MCP client configuration updates atomic, idempotent, and private during
  dry runs.
- Corrected stale repository URLs, schema identifiers, tool documentation, and
  contributor commands.
- Restored CI by using the current workload handoff flag and removing an
  unavailable Git dependency.

## [0.2.0] - 2026-08-03

### Added
- `inference-performance-hints`, a read-only estimate-then-measure triage skill
  adapted from Jeff Dean and Sanjay Ghemawat's official Performance Hints.
- An MCP-readable performance-hints resource and a synthetic 70B decode estimate.

### Changed
- Benchmark, tuning, model-optimization, kernel-profile, methodology, and agent
  guidance now use a bounded cost ledger before broad sweeps or captures.

## [0.1.0] - 2026-06-13

### Added
- 31 profile-and-optimize skills with a bundled MCP server, shipped as a Claude
  Code plugin.
- Measurement-rigor methodology and the skill, tool, and version count lints.
- `scripts/check_docs.py` doc-correctness gate and a CI workflow.
