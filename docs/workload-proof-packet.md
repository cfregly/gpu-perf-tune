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

## Workflow handoff

GPU and inference pilots can attach this packet to a broader workflow record as
`workload_level_evidence`. The optional top-level `workflow_handoff` object
names the integration id, workflow, access-stage movement, what the workload
packet proves, what the consuming workflow system proves, and what remains
outside the claim.

Keep the layers separate. This packet proves workload target, stack, command,
measurements, baseline, profiler evidence, and verdict scope. The consuming
workflow system proves workflow authority, replay, hosted evidence, promotion,
or other workflow-level facts. A packet can pass `--require-workflow-handoff`
while still being `status: "draft"`.

Use `--require-verdict` before external sharing. It is the separate gate that
requires verdict status, passing verdict gates, a comparable baseline, and a
clean source tree.

## Local workload proof contract

The workload proof packet is the local contract. It is complete without another
repository: schema, validator, checked fixture, and Makefile gate all live here.
The `workflow_handoff` object is only the bridge into another workflow system. It
names the integration id, access-stage movement, workload facts, workflow-system
facts, and non-claims.

Backend compiler or profiler artifacts may be referenced as evidence only when
the packet also names the workload command, run evidence, comparable baseline,
gates, and verdict scope. A compile or profile artifact does not become a
buyer-facing workload claim until this packet accepts it.

Runtime or harness outputs can support a workflow lane, but they do not prove GPU
performance, backend correctness, or workload cost by themselves.

## Optional integrations

ProofPlane is one possible consumer of `workflow_handoff` metadata. In that
case, ProofPlane should consume the packet as workload-level evidence and keep
its workflow-level claims separate from this repo's GPU workload claims. This
repo does not require ProofPlane to validate, install, or run.

## Commands

Validate every checked-in packet:

```bash
make workload-proof-check
```

Validate a specific packet as workflow-attachable workload evidence:

```bash
python3 scripts/check_workload_proof_packets.py path/to/workload-proof-packet.json --require-workflow-handoff
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
