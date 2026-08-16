# Bundled MCP server guidance

This directory is the source of truth for the `profile_and_optimize` MCP
server. Shared project policy lives in the repository root `AGENTS.md`.
This file adds rules for server tools, experiments, and performance evidence.

The runtime finds this tree from `PROFILE_AND_OPTIMIZE_REPO_ROOT`, or by
walking upward until it finds `pyproject.toml`, `mcp_surface.py`, and `tools/`.
Agent instruction filenames are not runtime markers.

## Safety and acknowledgements

Treat a tool as read-only unless its contract declares a mutating safety class.
Local artifact writers use explicit output paths and do not require a separate
acknowledgement. A call whose contract declares an `i_understand_this_*` field
must receive it from the operator in the current turn. Never infer or cache
that acknowledgement. Current ack-gated tools submit jobs, change Slurm node
state, or publish data to external storage.

Fail closed when a required path, command, credential, workload identity, or
cluster target is missing. A dry run must not install packages, create a virtual
environment, call a client CLI, write configuration, or reveal existing private
configuration.

## Reproducibility-grade evidence

A result needs a durable evidence bundle. Record the source revision, operator,
environment, exact commands, standard output, standard error, exit codes, raw
measurements, and summary. Preserve raw payloads before aggregation.

A future reviewer must be able to identify what ran and decide whether the
claim follows from the captured data. Empty, truncated, or implausible captures
are capture defects until a validation step proves otherwise.

## Benchmark methodology hygiene

Every number carries its full context:

- model and exact weights revision
- GPU model, count, topology, and interconnect
- weight, activation, and KV cache precision
- tensor, pipeline, expert, and data parallelism
- serving engine, image, source revision, and material launch flags
- input length, output length, concurrency, dataset, and cache regime
- trial count, aggregation method, baseline, and changed variable

State the denominator for a percentage. State both absolute values for a
speedup. Distinguish a warm best case from an SLA-qualified operating point.

## Verdict rigor: DRAFT vs VERDICT

Label a single observation, extrapolation, unverified capture, or predicted
improvement as DRAFT. Name the missing rerun or control that would promote it.

Use VERDICT only after production-shaped, variance-controlled measurement
against a representative baseline. A kernel cause also needs validated profile
evidence. Never promote a prediction because it sounds plausible.

## Validate the matrix, never generalize from one cell

Quantization, backend, and scheduling winners can change with concurrency and
workload shape. A universal claim requires the relevant backend by concurrency
matrix on matched hardware and workload conditions. Report the winner for each
measured regime when the matrix does not have one winner.

## Experiment isolation and traceability

Use the evidence run ID as the experiment ID and join key. Every disposable
cluster object uses an experiment-specific name and the matching
`experiment=<id-slug>` label. Do not reuse standing, platform, migration, or
shared cache names.

Tear down experiment resources by label and verify standing resources remain
ready. Record created object names and published campaign paths in the evidence
bundle.

## Experiment delivery ladder

Use the smallest delivery method that proves a source change:

1. Use a file-level `subPath` ConfigMap overlay for a small, explicit patch.
2. Use an init container patch set when several files must change on a pullable
   base image.
3. Build a pinned image when compiled code, dependencies, or startup behavior
   changes.
4. Use an infrastructure patch only when the experiment needs platform-owned
   deployment changes.

Record the delivery method, source commit, patch, and image digest. A result
from one delivery tier cannot prove a different tier without a matched rerun.

## All attribution claims must match collected profile data

Every where or why claim cites the exact profile artifact. Every recommendation
cites its evidence paths and evidence rigor. A predicted change is DRAFT until a
controlled comparison proves it.

Do not smooth over conflicting tools. Record the conflict and resolve capture
quality, time windows, workload identity, and metric meaning first.

## Kernel rubric

Classify a kernel before optimizing it:

- `K`, known-good against its relevant roofline
- `R`, reducible through algorithm or fusion work
- `H`, hidden behind launch, synchronization, or host overhead
- `P`, parallelism-starved through occupancy or load imbalance
- `A`, attribution error caused by the capture or analysis

Record the candidate and baseline classifications. A win against a weaker or
misclassified baseline remains DRAFT.

## Speed-of-light framing

Compare measured work with the named hardware ceiling in
`configs/sol-ceilings.yaml`. Cite the exact key and record the evidence rigor.
Do not copy peak values into a skill or report. A ceiling is an upper bound, not
an expected result.

## Validate every generated asset

Open and inspect rendered documents. Parse emitted JSON. Round-trip generated
configuration through its consumer. Compare output row counts with the source.
Check plot labels and axes against raw values. Exit code zero is not enough.

## The Grind Mandate

End each measured result with the next candidate lever, its expected direction
and rough magnitude, its cost, and the gate that will prove or refute it. Bank
lower-ranked candidates with evidence. An empty frontier needs a documented
speed-of-light wall.

## Server checks

Run these from the repository root:

```bash
make pytest
make smoke-mcp-runtime
make lint-tool-counts
make lint-skill-mcp-args
```

`mcp_surface.py` owns the public library and tool counts. Update its constants,
tests, and count-bearing docs together when the surface changes.
