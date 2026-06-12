Status: Active
Audience: contributors writing or reviewing artifacts under experiments/artifacts.

# Evidence Shape

Every durable run, gate, or investigation artifact should be easy to audit
without replaying the chat that produced it.

## Required Shape

- Put durable evidence under `experiments/artifacts/<family>/<run-id>/`.
- Include `SOURCE.md` with where the evidence came from, how it was produced,
  and which command or tool created it.
- Include `summary.md` for the human-readable result.
- Preserve raw payloads next to normalized outputs when evidence comes from
  metrics, logs, ClickHouse, Prometheus, Grafana, Slurm, or MCP tools.
- Record enough IDs to reconnect the evidence: benchmark, run ID, Slurm job ID,
  node names, cohort path, time window, datasource, and query text where
  applicable.
- Keep the `experiments/artifacts/` index short and hand-maintained. It is a
  reviewer map, not a generated inventory.

## Active Surfaces

- `tools/shared/audit/audit_artifact_paths.py`
- `tools/shared/audit/audit_evidence_bundle.py`
- the `evidence` CLI (`python -m evidence`)

