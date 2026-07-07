# Workload Proof Packet

This is the buyer-facing packet shape for neocloud workload proof. It turns a
GPU claim into one inspectable artifact: workload, cloud target, software stack,
run command, measurements, baseline, raw evidence, gates, and verdict.

Use it when the audience is asking:

- Did this run on the target neocloud, region, GPU SKU, and topology?
- Was the workload shape named, including model, dataset, token lengths,
  concurrency, and request count?
- Can I rerun or inspect the exact command, environment, output, logs, and
  profiler trace?
- Is the baseline comparable?
- What is proven, what is not proven, and what should happen next?

## Contract

The schema is [`schemas/workload-proof-packet-v1.json`](../schemas/workload-proof-packet-v1.json).
The completeness gate is [`scripts/check_workload_proof_packets.py`](../scripts/check_workload_proof_packets.py).
The checked-in example is [`examples/workload-proof-packet/workload-proof-packet.json`](../examples/workload-proof-packet/workload-proof-packet.json).

A packet must track these dimensions:

- Workload: name, model source, dataset, input and output tokens, concurrency,
  request count, success criteria.
- Target: neocloud, region, zone, GPU SKU, GPU count, topology, interconnect,
  availability source.
- Stack: container image, serving engine, engine version, CUDA version, driver,
  NCCL, launch flags.
- Run: exact command tuple, working directory, stdout, stderr, exit file, exit
  code, sanitized environment, secrets policy.
- Measurements: latency, throughput, tokens per GPU, utilization, power, cost,
  reliability.
- Baseline: name, kind, source, comparability flag, comparable measurements.
- Evidence: raw outputs, logs, profiler traces, normalized summary, source repo,
  source commit, dirty flag.
- Gates: named checks with pass status, required-for-verdict status, and evidence
  path.
- Verdict: claim, proof scope, not-proven list, caveats, next lever.

## ProofPlane handoff

GPU and inference pilots can attach this packet to a ProofPlane proof pack as
`workload_level_evidence`. The optional top-level `proofplane_handoff` object
names the ProofPlane proof-pack or pilot id, the workflow, access-stage movement,
what the workload packet proves, what ProofPlane proves, and what remains outside
the claim.

Keep the layers separate. This packet proves workload target, stack, command,
measurements, baseline, profiler evidence, and verdict scope. ProofPlane proves
workflow authority, replay, gates, hosted evidence, and promotion. A packet can
pass `--require-proofplane-handoff` while still being `status: "draft"`.

Use `--require-verdict` before external sharing. It is the separate gate that
requires verdict status, passing verdict gates, a comparable baseline, and a
clean source tree.

## Commands

Validate every checked-in packet:

```bash
make workload-proof-check
```

Validate a specific packet as ProofPlane-attachable workload evidence:

```bash
python3 scripts/check_workload_proof_packets.py path/to/workload-proof-packet.json --require-proofplane-handoff
```

Validate a specific packet as verdict-ready:

```bash
python3 scripts/check_workload_proof_packets.py path/to/workload-proof-packet.json --require-verdict
```

Use `--require-verdict` before sharing a packet externally. It fails if a
required verdict gate has not passed, if the baseline is not comparable, or if
the source tree was dirty during capture.

## Status

The checked-in example is a synthetic draft fixture. It demonstrates the packet
and handoff contract, not real B200, CoreWeave, Verda, or customer-workflow
evidence.

Use `status: "draft"` for single-run or incomplete evidence. A draft packet can
still be valuable, but it must name what is not proven.

Use `status: "verdict"` only when the required gates pass, the baseline is
comparable, and the source tree is clean. Verdict means the packet is ready for
skeptical review inside the proof scope it states.
