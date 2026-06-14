---
name: inference-capacity-sizing
last_validated: 2026-06-09
description: >-
  SLA-first GPU capacity sizing for a serving deployment: given a tokens-per-minute
  (TPM) target AND the interactivity SLA (output tokens/s/user), compute the pods and
  GPUs needed from a model's measured tok/s/user-vs-concurrency curve. Sizing MUST start
  from the tok/s/user SLA, not the TPM number alone: the SLA picks the concurrency, which
  sets per-pod throughput, which sets the GPU count. Also translates validated
  optimizations into delta-GPUs at the SLA-pinned concurrency. Backed by the standalone
  capacity_sizing.py tool. Use to answer "how many GPUs for N TPM", "size this for the
  customer", "pods for the SLA", or to sanity-check a sizing. Triggers on "capacity
  sizing", "how many GPUs", "how many pods", "tokens per minute sizing", "TPM target",
  "size for the customer", "interactivity SLA", "tokens per second per user", "GPU count
  for throughput", or any combination of "size / capacity / how-many" with "GPU / pod /
  TPM / SLA / tok-per-user / concurrency".
allowed-tools:
  - mcp__profile_and_optimize__search_runbooks
  - mcp__profile_and_optimize__search_evidence
  - Bash(python3:*)
  - Read
  - Write
---

# inference-capacity-sizing

## Purpose

Answer "how many GPUs to meet a TPM target" the correct way: start from the customer's
interactivity SLA (output tokens/s/user), not the TPM number alone. A TPM target by itself
is underdetermined. 3M TPM is 50,000 output tok/s, but that can be a few users each getting
many tokens/s or many users each getting few. The tokens/s/user SLA is what picks the
concurrency, the concurrency sets the per-pod throughput, and that sets the pod and GPU
count. Sizing from a fixed concurrency and back-deriving users is the backwards direction.

## When to use

- A customer gives a throughput requirement (TPM, RPS, or total tok/s) and you must size the
  GPU count.
- You need to sanity-check a sizing recommendation ("N pods for M TPM").
- You want to show a perf optimization as a customer-facing win (delta-GPUs at the SLA).

Do not use this to MEASURE a model (that is `inference-perf-bench` / `inference-aa-workload`)
or to rank models (that is `inference-fleet-leaderboard`). This skill consumes an
already-measured tok/s/user-vs-concurrency curve.

## The method (SLA-first)

```
SLA S (tok/s/user)  -> c* = max concurrency per replica where tok/s/user(c*) >= S
                    -> per-replica throughput at c*  (= c* * tok/s/user(c*))
                    -> replicas = ceil( (TPM/60) / (per-replica * util) )
                    -> GPUs = replicas * gpus_per_pod
```

util is the headroom for the TTFT SLA + failover (70 percent is the usual planning number).
The pod definition is the GPUs per replica (a GB300 node = 4 GPUs = TP4). Report GPUs as the
invariant. A "2x TP4 pod" is 8 GPUs.

## Workflow

1. Get the measured curve. Pull the model's tok/s/user anchors from the fleet leaderboards
   ([`inference-fleet-leaderboard`](/plugins/profile-and-optimize/skills/inference-fleet-leaderboard/SKILL.md)
   gives c=1 and c=10. The throughput leaderboard gives the knee), or, for a VERDICT, the
   full roofline sweep (c=8..128).
2. Run the tool:

```bash
python3 ${PROFILE_AND_OPTIMIZE_REPO_ROOT}/tools/capacity_sizing.py \
  --tpm 3000000 --sla "20,50,100,150,215" \
  --anchors "1:215.5,10:122.2,256:41.8" \
  --gpus-per-pod 4 --util 0.70 --model "MiniMax-M2-NVFP4 (GB300 TP4)" \
  --emit <bundle-dir>
```

   It emits the SLA -> c* -> per-pod tok/s -> pods -> GPUs -> $/month table
   (CAPACITY-SIZING.md + capacity_sizing.json). It fails loud on missing/degenerate inputs
   (no anchors, non-monotonic curve, tpm<=0).
3. Read the table SLA-first. The pod count is flat for any SLA at or below the throughput
   knee, then rises steeply as the SLA tightens. Lead with "the GPU count is the SLA you
   commit to".
4. Match the workload. The curve MUST be measured at the customer's workload shape
   (ISL/OSL, multi-turn). An AA-shape curve is not valid for a long-context routing-stress
   ask. Using the wrong curve is the most common sizing error.

## Optimization leverage (delta-GPUs)

Translate each validated win into its effect on the GPU count at the SLA-pinned concurrency,
each at its own matched baseline + tier (never multiplied together): tuned high-concurrency
config (an untuned serve needs several times the GPUs), async scheduling + cudagraph (in the
tuned curve, do not disable), load- and KV-cache-aware routing (multi-turn cache-hit, the
routing-stress regime), NVFP4-KV capacity (more streams or longer context per pod, MLA
models), TP right-size (small-active MoE at TP1, fewer GPUs), MTP (raises tok/s/user so a
tight SLA is met at a higher concurrency). Compute the delta-GPUs at the SLA-pinned c from
the matched delta for the target workload. Cite the campaign + tier for each.

## Full-context reporting (no bare numbers)

Per `docs/METHODOLOGY.md` "Full-context reporting" (no bare numbers): every number
this skill emits MUST carry its full measurement-context descriptor, and every comparison
MUST be matched on it. A bare tok/s / TPOT / tok/s/user / GPU-count is a defect.
- Identity: model (+HF path), hardware (exact ceiling token GB300/B200), quant, kv-cache dtype.
- Parallelism: TP, DP (replicas), PP, EP.
- Serving cfg: max-num-seqs, max-num-batched-tokens, gpu-memory-utilization, max-model-len, cudagraph_mode, async_scheduling, prefix-caching.
- Workload: dataset, ISL/OSL (or mean in/out tokens), concurrency, num-prompts.
- Regime: warm vs cold. Latency vs throughput tier.
- Stack: image/vllm commit, serving engine (vllm/sglang/dynamo).
- Grounding: %SoL + ceiling key, sol_rigor (L1-L4), trials n, same-node, baseline named.
- The sizing inherits the curve's rigor: an anchored + interpolated sizing is a DRAFT. A
  full-roofline-sweep sizing is a VERDICT. Always state which, and the curve's workload.

## Asset validation (review + FAIL LOUD)

Every asset this skill emits (the sizing table / json) is held to `docs/METHODOLOGY.md`
"Asset validation": the tool FAILS LOUDLY
on missing/bad/degenerate inputs (no anchors, non-monotonic tok/s/user curve, tpm<=0,
util out of range) and never writes a silent placeholder, and the agent REVIEWS the rendered
table for human-sense + accuracy (the pod count rises monotonically as the SLA tightens, the
knee plateau is present, each number traces to the curve + the formula) and rebuilds it if
wrong, never shipping a wrong/confusing sizing with a caveat.

## Next lever / BREAKTHROUGH (Grind Mandate)

If this skill emits a measured-curve-backed sizing, its output MUST end by naming the next
lever, its expected unlock, and the gate that proves/refutes it, per `docs/METHODOLOGY.md`
"Always be grinding (next-lever framing)". For sizing the standing next lever is:
upgrade the DRAFT (anchored + interpolated) to a VERDICT by feeding the full roofline sweep
at the customer's workload. Then rank the delta-GPU optimization levers by value x cost and
pursue the top one (fewer GPUs at the SLA is the breakthrough). Delete this section ONLY if
the skill produces no measurements.

## Provenance

Backed by the standalone `server/tools/capacity_sizing.py` (sibling of
`tp_rightsize_advisor.py`. Pure-Python, fail-loud, unit-tested in `test_capacity_sizing.py`).
Companions:
[`inference-fleet-leaderboard`](/plugins/profile-and-optimize/skills/inference-fleet-leaderboard/SKILL.md)
(ranks models. Supplies the tok/s/user anchors),
[`inference-perf-bench`](/plugins/profile-and-optimize/skills/inference-perf-bench/SKILL.md)
(measures the curve).
