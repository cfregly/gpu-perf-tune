# findings.yaml schema

The `findings_cli.py` MCP verb family (`findings_record`, `findings_render`,
`findings_diff`) accepts a single `findings.yaml` file per evidence bundle.

## Schema (one-of-list)

```yaml
findings:
  - id: <slug>                    # required; unique within the bundle
    severity: critical | high | medium | low | informational    # required
    source_skill: <skill-name>    # required; matches a SKILL.md `name:`
    source_query: <label>         # required; the query label from the source
                                  # skill's output, or a free-form descriptor
    evidence_path: <relative path to raw response JSON; e.g. raw/W1-ib-bw-top-20.json>
    headline: <one-line summary>  # required; what an operator sees first
    recommended_action: <one-line action>   # required; what to do next
    status: open | in_progress | resolved   # required; default `open`
    affected_entities:            # optional; list of node/cluster/zone names
      - kind: <node | dpu | cluster | zone | leafgroup | switch | pod>
        value: <name>
    references:                   # optional; relative paths or URLs
      - <relative-path or url>
    notes: <free-form>            # optional
    detected_at_utc: <RFC3339>    # auto-filled by `findings_record` if absent
```

## Example

```yaml
findings:
  - id: dcgm-thermal-throttle-node12
    severity: critical
    source_skill: inference-dcgm-correlate
    source_query: dcgm-thermal-violations
    evidence_path: raw/W8-dcgm-thermal.json
    headline: "GPU thermal throttling on node12 during the load sweep"
    recommended_action: "Inspect node12 cooling before re-running the bench"
    status: open
    affected_entities:
      - kind: node
        value: node12
    references:
      - docs/METHODOLOGY.md
    detected_at_utc: "2026-05-21T15:32:33Z"
  - id: nccl-busbw-regression-pool-a
    severity: high
    source_skill: inference-perf-bench
    source_query: all-reduce-busbw
    evidence_path: raw/W1-nccl-all-reduce.json
    headline: "all_reduce busBW 18% below baseline on 4 of 8 nodes"
    recommended_action: "Fabric cable / optic / NIC inspection on the 4 affected nodes"
    status: open
    affected_entities:
      - kind: node
        value: node03
      - kind: node
        value: node05
    detected_at_utc: "2026-05-21T15:32:36Z"
```

## Diff semantics

`findings_diff <bundle-a> <bundle-b>` emits markdown like:

```markdown
## Findings diff: <bundle-a> -> <bundle-b>

### New findings (in B, not in A)

- dcgm-thermal-throttle-node12 (critical) -- GPU thermal throttling on node12 during the load sweep

### Resolved findings (in A as open, not in B)

- nccl-busbw-regression-pool-a (high) -- was 18% below baseline, now absent

### Status changes

- (none)
```

Two findings match by `id`. New ones are added. Ones in A but absent (or status=resolved) in B are flagged as resolved.

## Render semantics

`findings_render` emits a markdown roll-up grouped by severity:

```markdown
## Critical (drop everything)

| # | Finding | Source | Action |
| --- | --- | --- | --- |
| C1 | <headline> | <source_skill> / <source_query> | <recommended_action> |

## High (significant ops impact)

...
```

Numbering (`C1`, `H1`, `M1`, `L1`) is auto-assigned in input order.

## When to use which verb

- **`findings_record`** - append a finding to an evidence bundle's
  `findings.yaml` (created if missing). Operator-callable.
- **`findings_render`** - convert `findings.yaml` to a presentable
  `findings.md` for sharing or attaching to a ticket. Read-only.
- **`findings_diff`** - compare two `findings.yaml` files (e.g.,
  yesterday's bundle vs today's) and report drift. Read-only.

## Contact

Open an issue using the [`question.md`](/.github/ISSUE_TEMPLATE/question.md) template.
