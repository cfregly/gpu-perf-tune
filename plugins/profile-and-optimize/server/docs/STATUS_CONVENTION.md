# Doc Status Convention

Status: Active
Audience: contributors adding or rewriting docs.

Per the review conventions in [`REVIEWERS.md`](../../../../REVIEWERS.md), every
document under [`docs/`](.) carries a `Status:` header so
search and review surfaces can distinguish authoritative material from
historical evidence at a glance.

## Allowed values

| Status | Meaning | Example |
| --- | --- | --- |
| `Active` | Authoritative, current. Edit deliberately and keep links honest. | [`start-here.md`](start-here.md), [`first-hour.md`](first-hour.md). |
| `Reference` | Authoritative but slow-changing background context that reviewers consult on demand. | [`mcp-tool-io-contract.md`](mcp-tool-io-contract.md). |
| `Append-only ADR log` | Architecture Decision Records. Superseded entries link forward, never get rewritten. | An `architecture-decisions.md` log. |
| `Captured evidence` | Snapshot of an external thread, audit, or post-mortem. Read for audit, do not edit retroactively. | A captured incident post-mortem. |
| `Historical` | Superseded narrative kept for reviewer context. Pointers in the file route the reader to the current source of truth. | A dated operator handoff. |
| `Generated` | Recomputed from a source-of-truth manifest. Do not hand-edit. | A generated index under [`../experiments/artifacts/`](../experiments/artifacts). |

## Where the header goes

The header is the first non-blank lines of the file, *before* the H1
title. Use the form:

```markdown
Status: Historical
Audience: reviewers auditing the gate execution captured on the dated handoff.
Forward-looking handoffs live in ../README.md...

# Doc title
```

`Audience` and forward-routing prose are required for `Historical` and
`Captured evidence` so reviewers landing on a stale document are
immediately routed to the current authoritative one.

## Drift gate

The header is enforced in review. Run `make check-doc-links` from the repository root to verify that every documentation link resolves.

New docs MUST land with a `Status:` header regardless.
