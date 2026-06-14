# Public API

Status: Active
Audience: engineers and MCP/tool authors importing repo helpers.

The public Python API is intentionally small and CLI-free. Import the
stable shared primitives from the `tools.shared` modules listed below.
Operator commands remain under [`tools/`](/plugins/profile-and-optimize/server/tools).

Do not import other `tools.pipeline...` modules directly from external
code. Their layout and signatures are internal and may change without
notice.

## Stable Exports

| Symbol | Source of truth | Purpose |
| --- | --- | --- |
| `BENCHMARK_COLUMNS` | `tools.shared.validation_schema` | Supported summary columns for artifact validation. |
| `REQUIRED_SUMMARY_FIELDS` | `tools.shared.validation_schema` | Required summary fields for result rows. |
| `FINAL_LOG_PPL_TARGETS` | `tools.shared.mlperf_targets` | Per-benchmark final log perplexity thresholds. |
| `QUALITY_TARGETS` | `tools.shared.mlperf_targets` | Non-log-ppl quality targets. |
| `parse_mllog_file` | `tools.shared.mllog_parser` | MLLOG parser shared by validators and tests. |

Everything else under `tools/` should be treated as internal unless it is
promoted into this table in the same change that documents the contract.
