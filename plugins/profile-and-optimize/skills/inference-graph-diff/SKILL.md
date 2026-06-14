---
name: inference-graph-diff
last_validated: 2026-05-26
description: >-
  Diff the compiled FX / Inductor graphs across two vLLM versions or two
  helm configs to see exactly which fused kernels / passes / partitions
  changed. Uses `torch._dynamo.explain` + `torch.compile`'s graph-dump
  hooks. Useful when you anticipate a model-graph-level
  change (e.g. swapping cudagraph_mode, enabling fuse_allreduce_rms,
  toggling pass_config.fuse_attn_quant). Triggers on "graph diff",
  "dynamo explain", "fx graph", "compile graph", "torch.compile diff",
  "compilation pass change", or any combination of "graph / fx /
  inductor / dynamo" with "diff / compare / explain / dump".
allowed-tools:
  - Bash(kubectl:exec,*)
  - Bash(kubectl:cp,*)
  - Bash(jq:*)
  - Bash(diff:*)
  - Bash(python3:*)
  - Read
  - Write
---

# inference-graph-diff

## Purpose

Diff the **compiled torch FX / Inductor graphs** between two vLLM
configs to surface exactly which compilation choices changed. The most
common need: "I added `pass_config.fuse_allreduce_rms=true`,
which graph nodes were affected?" - torch.compile decisions are not
visible in the vllm startup logs alone.

Three diff layers cover the stack:
- `inference-kernel-profile` (nsys `.nsys-rep` timelines - runtime)
- `analyze-zymtrace-workload` (per-category kernel time)
- This skill (the COMPILATION graph - build-time)

## When to use

- Confirming a `compilation_config` change actually took effect (the
  vllm logs print the requested config, not the post-fusion graph)
- Investigating an unexpected perf delta after a 1-line config tweak
  ("did the fused op activate?")
- Pre-deploy gate when migrating between vllm versions (catch silent
  inductor pass behavior changes)
- Documenting WHY a variant landed for the eventual upstream PR or
  release notes

Do **not** use this skill for:

- Runtime timeline analysis - use nsys or zymtrace instead
- Per-request latency breakdown - those are different layers
- **SGLang arms** - this skill is **vLLM-specific** (it diffs vLLM's
  `torch.compile` FX / Inductor graph dumps via `TORCH_LOGS=+dynamo,+inductor`).
  SGLang has a different compilation/fusion model (no vLLM `compilation_config` /
  `cudagraph_mode`), so this skill fails closed on an SGLang arm. To compare two
  SGLang configs' execution structure, capture a per-kernel **nsys timeline** for
  each (launch-wrap `sglang.launch_server` under nsys)
  and diff the `cuda_gpu_kern_sum` kernel sets instead.

## Recipe

### 1. Dump the graph from each side of the diff

For each helm config A and B:

```bash
NS=inference
TARGET_POD=$(kubectl -n $NS get pods -l app=basic-inference -o jsonpath='{.items[0].metadata.name}')

# Need vllm to log the compile graph. Set the env var BEFORE engine init:
kubectl -n $NS set env deployment/basic-inference \
  TORCH_LOGS="+dynamo,+inductor,+graph_breaks" \
  TORCHDYNAMO_VERBOSE=1 \
  PT_LOGGING_LEVEL=DEBUG

# Wait for rolling restart...
# Then grab the logs which contain the FX dump:
kubectl -n $NS logs $TARGET_POD --tail=10000 > side-A.log
```

Repeat with the alternate config to get `side-B.log`.

### 2. Extract the FX graphs

```python
import re, pathlib
def extract_fx_graphs(log_path):
    text = pathlib.Path(log_path).read_text(errors="replace")
    graphs = re.findall(r"=== FX GRAPH ===\n(.*?)\n=== END FX GRAPH ===", text, re.DOTALL)
    return graphs

a_graphs = extract_fx_graphs("side-A.log")
b_graphs = extract_fx_graphs("side-B.log")
```

(The exact log marker strings depend on vLLM version + torch version,
the recipe above is the v0.21.x default. For other versions, dump via
`TORCH_COMPILE_DEBUG=1` and parse the `compile_debug_*.txt` files
torch.compile writes under `~/.cache/torch/`.)

### 3. Diff the graphs

```bash
diff -u side-A-graph0.fx side-B-graph0.fx > graph0.diff
# For the inductor lowered graph:
diff -u side-A-inductor.py side-B-inductor.py > inductor.diff
```

### 4. Optional: render summary with `torch._dynamo.explain`

```bash
kubectl -n $NS exec $TARGET_POD -- python3 -c "
import torch._dynamo as dynamo
# torch._dynamo.explain takes a callable + sample inputs and prints
# the compile decisions (recompiles, graph breaks, ops list)
# For vllm the callable is the model.forward; sample inputs are mocked.
# See the dynamo.explain doc for the full surface.
"
```

### 5. Emit a structured `graph_diff.json` into the bundle

```json
{
  "schema": "inference_graph_diff_v1",
  "side_a": {"helm_rev": 28, "config_summary": "...", "compile_passes": [...]},
  "side_b": {"helm_rev": 29, "config_summary": "...", "compile_passes": [...]},
  "added_passes": ["fuse_allreduce_rms"],
  "removed_passes": [],
  "graph_size_delta": {"nodes_before": 1234, "nodes_after": 1187, "delta_pct": -3.8},
  "notes": "1-line helm bump landed; -47 nodes from the allreduce-rms fusion."
}
```

## Output

Bundle artifact layout:

```
experiments/artifacts/inference-perf-bench/<bundle>/
  graph-diff/
    side-A.log              # raw torch.compile log from config A
    side-B.log              # raw torch.compile log from config B
    side-A-graph0.fx        # extracted FX graph
    side-B-graph0.fx
    graph0.diff             # unified diff
    inductor.diff           # diff of the lowered inductor code
    graph_diff.json         # structured summary
```

## Skill maturity

This skill is **research-grade** (v0.1 - same status as the
nsys-sidecar approach). The recipe is documented but not yet wrapped in
a dedicated MCP verb because the dump-format depends on torch version.
A future v1.x of profile_and_optimize could add a `perf_tune_report_graph_diff` verb
that automates the recipe end-to-end.

## Kernel rubric (K/R/H/P/A)

A compilation-graph change is often a **representation (R)** or **hardware
specialization (H)** move in the kernel rubric (`docs/METHODOLOGY.md`
"Kernel-work classification"). A fusion pass that replaces a generic Triton
op with a library/`sm100f` tensor-core kernel raises both R (R2→R1) and H (toward H4),
`fuse_allreduce_rms` / `fuse_attn_quant` shift which hardware path the graph dispatches.
When a graph diff backs a custom-kernel comparison, **note the R/H delta in
`graph_diff.json` `notes`** and carry the candidate + baseline `(K,R,H,P,A)` coordinates
into the bundle's `SOURCE.md`/`summary.md`. The graph diff shows R/H *changed*. It does
NOT prove the new path engages tensor cores or hits its roofline - defer that H + P
proof to [`inference-kernel-ncu-profile`](/plugins/profile-and-optimize/skills/inference-kernel-ncu-profile/SKILL.md), the
gate's enforcement point. A win over a strictly-lower-H/R baseline stays a **DRAFT, never
a VERDICT**. If the comparison campaign reaches L4 (an ncu roofline renders), the
candidate + baseline `(K,R,H,P,A)` must also be emitted as a structured `krhpa:` block
in `config.yaml` (see [`inference-kernel-ncu-profile`](/plugins/profile-and-optimize/skills/inference-kernel-ncu-profile/SKILL.md)
for the YAML) - `publish_to_lake` fails closed without it.

## Cross-references

- `perf-baseline-diff` for run-over-run baseline diffs
- `analyze-zymtrace-workload` for per-category zymtrace diffs
- `inference-kernel-profile` for sidecar nsys captures

## Full-context reporting (no bare numbers)

Per the canon "Every performance number carries its full context (no bare numbers)"
(`docs/METHODOLOGY.md` "Full-context reporting"): every number this
skill emits MUST carry its full measurement-context descriptor, and every comparison MUST be
matched on it. A bare `tok/s` / TPOT / BW / %SoL / speedup is a defect - it cannot set a
default, ship a config, or appear in a report.
- **Identity:** model (+HF path), hardware (exact ceiling token `GB300`/`B200`), quant, kv-cache dtype.
- **Parallelism:** TP, DP (replicas), PP, EP, parallel_strategy.
- **Serving cfg:** max-num-seqs, max-num-batched-tokens, gpu-memory-utilization, max-model-len, cudagraph_mode/enforce_eager, async_scheduling, prefix-caching.
- **Workload:** dataset, ISL/OSL (or mean in/out tokens), concurrency, num-prompts.
- **Regime:** warm vs cold. Latency vs throughput tier.
- **Stack:** image/vllm commit, bench backend, serving engine.
- **Grounding:** `%SoL` (+ ceiling key from `configs/sol-ceilings.yaml` - never inline a peak), sol_rigor (L1-L4), trials n (mean±std), same-node, baseline named.
- **Per-number exact shape (no smoothing):** when reporting more than one number, keep EACH with its own exact shape (ISL/OSL, concurrency, dataset, regime) - never normalize a set to one uniform descriptor that hides per-point variation (e.g. `c=1 @ ISL1024/OSL256` + `c=64 @ ISL4096/OSL512`, NOT one shared "random").

A graph diff is structural, not a measurement, so it is exempt from the %SoL
requirement. If you pair it with a perf measurement (tok/s, latency, speedup),
that measurement follows the rigor discipline: capture L1+L3 (L4 where
feasible) Speed-of-Light + publish `--strict`. Canonical map:
`docs/METHODOLOGY.md`.

## Next lever / BREAKTHROUGH (Grind Mandate)

If this skill emits a measured result, its output MUST end by naming the **next perf lever**,
its **expected unlock** (direction + rough magnitude), and the **gate** that proves/refutes it,
per `docs/METHODOLOGY.md` "Always be grinding (next-lever framing)". A
measured win is the new floor, not the finish -- so **do everything we can to find the next
BREAKTHROUGH**: the highest-EV unlock toward Speed-of-Light (a new champion / kernel / router /
quant / parallelism / spec-decode win, or an unblocked stack), not just the next micro-lever.
Rank the candidate breakthrough levers by value x cost (the GRIND FRONTIER, `perftunereport
value_view`), pursue the top, bank the rest with evidence. Record WHY a refuted lever loses,
update the standing frontier in the active bundle's `HANDOFF.md`. Never conclude
"exhausted/optimal/done" without an explicit next-lever frontier (an empty frontier AND a
documented SoL wall only). Delete this section ONLY if the skill produces no measurements.
