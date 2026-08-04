# Dean-Ghemawat performance hints for GPU inference

Status: Active

Jeff Dean and Sanjay Ghemawat's official
[Performance Hints](https://abseil.io/fast/hints.html) guide describes how they
estimate, measure, and improve hot code. The guide says its concrete examples
focus on single-binary C++ rather than distributed systems or ML hardware. This
document adapts the reusable principles to GPU inference without pretending the
original CPU examples are GPU measurements.

## The operating loop

Use one loop for every performance investigation:

1. Identify whether the code runs at setup, once per request, once per token, or
   inside a library used by many workloads.
2. Write a rough cost ledger before spending cluster time.
3. Measure the production-shaped baseline and profile the dominant path.
4. Change one mechanism, then rerun the same measurement.
5. Keep the change only when the end-to-end metric and the profile agree.

Choosing a faster implementation early is useful when it does not add meaningful
complexity. Profile evidence is still required before a disruptive or obscure
optimization.

## Back-of-the-envelope ledger

An estimate is a ranking tool, not a verdict. List the expensive operations,
multiply work by an approximate unit cost, and account for which costs serialize
versus overlap. Round to one significant digit when the inputs are uncertain.

For GPU inference, estimate these terms when they apply:

| Term | Rough calculation | Preferred input |
| --- | --- | --- |
| Decode weight traffic | active weight bytes per step / aggregate effective HBM bytes/s | measured active parameters, actual quant format, DCGM or ncu bandwidth |
| KV-cache traffic | KV bytes read and written per step / effective HBM bytes/s | actual layers, heads, sequence length, cache dtype |
| Prefill compute | FLOPs / effective tensor-core FLOP/s | model shape plus measured ncu or roofline efficiency |
| Collectives | topology-adjusted bytes / measured fabric bytes/s + collective latency | same-message-size `nccl-tests` or profile data |
| Launch and host gaps | launches per step x measured launch cost, plus observed host gaps | nsys timeline or decode-step budget |
| Model loading | checkpoint bytes / measured end-to-end loader bytes/s | loader logs, including staging and deserialization |
| User latency | queueing + prefill + generated tokens x decode-step time | the production-shaped concurrency and ISL/OSL distribution |

If terms overlap, their lower bound is usually the maximum of the terms. If they
serialize, add them. State the assumption. Batching means replica throughput is
not simply the reciprocal of one user's latency.

Use [`configs/sol-ceilings.yaml`](/configs/sol-ceilings.yaml) for published GPU
ceilings and local evidence for achieved rates. Do not copy a hardware peak into
a skill or estimate. Name the ceiling key and the efficiency assumption.

The official guide's updated operation table is still useful for scale intuition:
cache, branch, lock, and DRAM operations are roughly nanoseconds. Small compression,
SSD reads, datacenter round trips, and 1 MB memory or fast-network transfers range
from microseconds to about a millisecond. Disk and intercontinental operations are
milliseconds to hundreds of milliseconds. Those values are rough CPU, storage,
and network reference points. Measure the current host, GPU, fabric, and software
stack before using any value in a decision.

## Translate the hints to inference

| Dean-Ghemawat principle | GPU-inference application | Evidence that closes the loop |
| --- | --- | --- |
| Prefer an algorithmic improvement | Reduce active work, padding, or sequential decode steps before tuning a local instruction | End-to-end A/B plus workload and quality gates |
| Add bulk APIs | Batch requests, tokenize in batches, fuse operator chains, and amortize control-plane crossings | Request-level latency and throughput at matched shapes |
| Use compact representations and better layout | Quantize weights or KV cache, keep tensors contiguous, separate hot metadata, reduce indirection | Bytes moved, cache dtype, ncu sectors and bandwidth, accuracy gate |
| Reduce allocations and copies | Reuse KV blocks and staging buffers, avoid host-device copies, keep temporary tensors out of per-token paths | Allocation profile, memcpy timeline, peak memory, TPOT |
| Avoid repeated work | Reuse compiled graphs, cache stable prefixes, precompute masks and descriptors, move invariants outside decode loops | Cache-hit and graph-replay counters plus cold and warm A/Bs |
| Add a common-case fast path | Capture frequent shapes, specialize a proven hot path, and keep fallback behavior explicit | Shape coverage plus profile share and correctness tests |
| Help the compiler only after profiling | Inspect generated CUDA/SASS for the measured hot kernel, then adjust fusion, vectorization, or specialization | ncu instruction, occupancy, stall, and roofline data |
| Reduce stats and logging costs | Move verbose logs and expensive metric aggregation off per-token paths | CPU profile and same-telemetry-level A/B |
| Exploit parallelism in batches | Parallelize at a useful grain and amortize dispatch rather than launching work per item | Scaling curve, launch count, utilization, and queueing |
| Shorten synchronization and pipeline | Overlap host work, HBM traffic, and collectives, and remove unnecessary device or stream synchronization | CPU and GPU timelines with the same filter |
| Treat a flat profile as a system signal | Accumulate several measured small wins or move higher in the call graph to change the loop or representation | Stable microbenchmarks plus an end-to-end confirmation |

Do not apply a hint mechanically. Caching can create misleading warm wins, extra
parallelism can increase queueing or contention, and compact formats can fail
quality gates. The workload decides.

## Rank experiments by possible impact

Use the current profile to cap each idea's potential. If a contributor occupies
8% of end-to-end time, deleting it entirely cannot produce more than an 8%
latency reduction under the same workload. For each candidate, record:

- the measured contributor and its share,
- the proposed mechanism,
- the maximum plausible end-to-end improvement,
- confidence in the diagnosis,
- experiment cost and rollback path,
- the metric, artifact, and stop condition that prove or refute it.

Start with the highest expected-value experiment whose baseline is trustworthy.
A broad sweep is not a substitute for a hypothesis. A within-noise result refutes
the lever for that workload and should not be promoted as a default.

## Measurement rules

- Build and run the production optimization mode with useful symbols.
- Match model, precision, topology, engine, workload shape, cache regime, and
  concurrency between arms.
- Use a microbenchmark for turnaround, then confirm at system level. A microbench
  can omit queueing, cache, allocation, synchronization, and compiler interactions.
- Profile CPU and GPU sides. Low GPU utilization can be a host, launch, copy, or
  synchronization problem.
- When a profile is flat, inspect loops high in the call graph, allocation and
  hardware-counter profiles, overly general paths, and the cumulative value of
  multiple small changes.
- Label every estimate and single observation DRAFT. Promotion to VERDICT follows
  [`docs/METHODOLOGY.md`](/docs/METHODOLOGY.md).

## Output contract

Every hint pass returns:

1. an estimate ledger with source, unit, and confidence for every input,
2. the measured baseline or the explicit missing-measurement gate,
3. the top three candidate mechanisms ranked by bounded impact and experiment cost,
4. one next experiment with a matched control and stop condition,
5. the sibling skill that should execute it.

See the synthetic
[`70B decode estimate`](/examples/performance-hints/README.md) for the expected
shape. Search this document through the bundled MCP with `search_runbooks` using
queries such as `back-of-the-envelope`, `flat profile`, `bulk APIs`, or
`reduce allocations`.

## Sources

- Jeff Dean and Sanjay Ghemawat,
  [Performance Hints](https://abseil.io/fast/hints.html), original 2023-07-27,
  updated 2025-12-16.
- Jeff Dean and Luiz Andre Barroso,
  [The Tail at Scale](https://research.google/pubs/the-tail-at-scale/), for why
  distributed serving must measure the latency distribution rather than averages
  alone.
