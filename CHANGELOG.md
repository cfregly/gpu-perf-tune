# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
