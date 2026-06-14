---
name: mirage-graph-coverage
last_validated: 2026-06-05
description: >-
  Read-only coverage auditor for the mirage / MPK persistent-megakernel's
  GENERATED task graph: cross-references DECLARED tensors (all_tensors[...] in
  kernel_N.cu) against tensors CONSUMED by tasks (inputs/outputs base_ptr in
  task_graph_N.json) and flags any tensor declared but never consumed as an
  input - the "feature defined in code but never wired into the task graph" bug
  class (e.g. RoPE cos/sin tables allocated but consumed by 0 tasks =>
  position-blind attention => incoherent / degenerate-repetition output). A
  numerics / relL2 harness silently PASSES on this class, so run THIS first when
  a megakernel compiles + exits 0 but emits incoherent output. Distinct from
  inference-graph-diff (vLLM FX / Inductor graphs). Triggers on "graph
  coverage", "task graph audit", "megakernel wiring", "declared but unconsumed
  tensor", "rope not wired", "missing task", or "megakernel / mirage / MPK /
  task-graph" with "coverage / wiring / unconsumed / unwired / audit".
---

# mirage megakernel graph-coverage auditor

Catches the **missing-wiring** bug class in a mirage/MPK generated megakernel:
a tensor (or whole feature) is defined and allocated but never wired into a
task, so the kernel runs without it. The canonical case is GLM-5.1 on GB300
where the RoPE `cos`/`sin` tables were attached but consumed by **0 tasks**, so
attention ran position-blind and the model emitted fluent-but-incoherent,
repetition-collapsing tokens - while every per-kernel numerics test passed.

## Quick start

```bash
python3 assets/mk-graph-coverage-audit.py <gendir>
# <gendir> holds the generated kernel_<rank>.cu + task_graph_<rank>.json
#   (e.g. mirage demo/deepseek_v3/). Defaults to /work/mirage-perop2/demo/deepseek_v3
# options: --critical rope_cos,rope_sin,<scale-tensor>   --ranks 0,1,2,3
```
Exit 0 = PASS (every critical tensor consumed as input), 1 = FAIL (a critical
tensor is declared but never consumed = missing wiring), 2 = no artifacts.

## What it does

Per TP rank, it parses two generated artifacts and diffs them:
- **declared** = `all_tensors["NAME"] = NAME;` lines in `kernel_<rank>.cu`
- **consumed** = `base_ptr` of every `all_tasks[].inputs[]` / `.outputs[]` in
  `task_graph_<rank>.json`, mapped to the set of consuming `task_type`s + counts

It then classifies each declared tensor and FAILs iff a *critical* tensor
(`rope_cos`,`rope_sin`,`cos_pos_embed`,`sin_pos_embed` by default) has **0 input
consumers**.

## Reading the output
- `[OK] <t>: input_refs=N consumer_task_types=[...]` - wired (consumed by a task).
- `[MISSING-WIRING] <t>: input_refs=0` - declared but never read by any task => the bug.
- `INFO write-only` - produced by a task but never read (often a final output like
  `output_token`. Suspicious only for an intermediate/feature tensor).
- `INFO dead` - declared, 0 refs at all (allocated scratch / codegen leftover. The
  `nullptr` sentinel shows here as a benign false positive).

## Worked example (GLM-5.1 on GB300)
Before the wiring fix the auditor FAILs: `rope_cos` `input_refs=0` (rope absent from the
graph). After the fix it PASSes: `rope_cos`/`rope_sin` `input_refs=156`,
`consumer_task_types=[278, 296]` (296 = q-rope-apply, 278 = k_pe / KV-cache
gather). Same artifacts a relL2 numerics harness would have called clean in both
states - graph-coverage is what distinguishes them.

## Triage rule
For a megakernel coherence bug (compiles, exits 0, incoherent output), run the
graph-coverage audit **before** building a per-kernel numerics harness. A
missing/mis-wired task makes a numerics harness pass while the model stays wrong,
the audit names the unwired tensor directly. Only once coverage is clean does a
relL2 harness make sense.

## Caveat
Verifies **structural wiring** (is the tensor consumed by a task), not runtime
numerical coherence. A PASS removes "feature absent" as the suspect. It does not
by itself prove the output is correct.
