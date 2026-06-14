# Runbooks

Searchable operator runbooks indexed by the `search_runbooks` MCP tool.
The internal launch runbooks were not carried over. The runbooks here are
generic SOPs that reference only tools and docs shipped in this repository.

## Index

| Runbook | What it covers |
|---|---|
| [profile-a-regression.md](profile-a-regression.md) | Detect a regression (`perf_baseline_diff`) -> hygiene-gated nsys capture (`scripts/nsys-validate-capture.sh`) -> `profile_profile_diff` -> per-kernel attribution (`import_nsys` / `import_ncu` / `dcgm_correlate`) -> classify via the `docs/METHODOLOGY.md` kernel rubric -> fix -> re-baseline |

Add new SOPs as standalone `.md` files in this directory and list them in the
table above (e.g. host-overhead checklists, per-stream MFU triage).
