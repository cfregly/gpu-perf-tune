# Runbook: profile a performance regression

A generic SOP for taking a suspected serving/training performance regression
from "the number got worse" to a classified root cause, a fix, and a new
recorded baseline. Every step names the repo tool that performs it. Nothing
here assumes a specific cluster, model, or vendor account.

Methodology canon: [`docs/METHODOLOGY.md`](../../../../docs/METHODOLOGY.md)
(verdict rigor, full-context reporting, kernel-work classification, capture
hygiene). Hardware ceilings: [`configs/sol-ceilings.yaml`](../../../../configs/sol-ceilings.yaml).

All module commands below run from `plugins/profile-and-optimize/server/`,
each is also exposed as an MCP tool of the same verb name (see
[`docs/mcp-composition.md`](../docs/mcp-composition.md)).

---

## 0. Preconditions

- A **recorded baseline** for the metric that regressed. If none exists, stop
  and record one first (`perf_baseline_record`, skill
  [`perf-baseline-record`](../../skills/perf-baseline-record/SKILL.md)) - a
  regression without a baseline is an anecdote.
- The serving config under test passes its known-good check, so a missing
  required flag is ruled out before any profiling:

  ```bash
  python3 -m tools.known_good_config.known_good_config_cli check \
    --model <model-id> --registry <path-to-known-good-configs.yaml> \
    --serve-args "<the live serve argv>" --json
  ```

  (Example registry shape: [`configs/known-good-configs.example.yaml`](../../../../configs/known-good-configs.example.yaml).)

## 1. Detect and confirm the regression (`perf_baseline_diff`)

Diff the current measurement against the registered baseline. The verb emits
per-dimension deltas and a GREEN / YELLOW / RED verdict:

```bash
python3 -m tools.perf_baseline.perf_baseline_cli diff \
  --baseline <family>/<measurement>/<slug> \
  --current ./current-measurement.json \
  --tolerance-percent 5 --json
```

- **GREEN** - no regression. Stop here (noise or expectation error).
- **YELLOW / RED** - reproduce once before profiling: rerun the *same* bench
  with the *same* workload shape (ISL/OSL, concurrency, dataset). A
  single-observation regression is a DRAFT, not a finding
  (METHODOLOGY.md "Verdict rigor").

Record the confirming measurement so the investigation has a pinned "bad"
artifact: `python3 -m tools.perf_baseline.perf_baseline_cli record ...`
(skill: [`perf-baseline-diff`](../../skills/perf-baseline-diff/SKILL.md)).

## 2. Capture an nsys profile - with hygiene gates

Capture one profile of the **regressed** build and ensure one exists (or is
captured) for the **baseline** build, under driven load matching the regressing
workload.

Sidecar capture into a serving pod (ack-gated. Submits a real profiling job):

```bash
python3 -m tools.perf_tune_report.perf_tune_report_cli kernel_profile \
  --namespace <ns> --pod <pod> --target-container <container> \
  --output-dir ./profiles/<tag> --duration-seconds 60 \
  --i-understand-this-submits-jobs --json
```

(Skill: [`inference-kernel-profile`](../../skills/inference-kernel-profile/SKILL.md).)

**Gate every capture before analyzing it** - an empty or implausible profile
is a capture bug until proven otherwise (METHODOLOGY.md "Capture hygiene"):

```bash
REP=./profiles/<tag>/capture.nsys-rep LOCAL=1 \
  DEPLOY_ARGS="<the nsys argv used>" \
  bash scripts/nsys-validate-capture.sh
```

The script enforces the 4-point gate: (1) `--cuda-graph-trace=node` was in the
nsys argv (graph-resident kernels are opaque otherwise), (2) the rep is not
suspiciously small (idle/untrafficked window), (3) the rep exists and
finalized, (4) the exported sqlite KERNEL table has nonzero rows. Exit 0 =
safe to analyze. Exit 1 = RETRY with the printed reason. Do **not** run stats
on a rep that fails the gate.

If you are sampling with a zymtrace-style continuous profiler instead, gate
queries on ingest lag first: `bash scripts/zymtrace-ingest-wait.sh` (skill:
[`zymtrace-anchored-query`](../../skills/zymtrace-anchored-query/SKILL.md)).

## 3. Diff the two profiles (`profile_profile_diff`)

Rank what actually changed between baseline and regressed reps:

```bash
python3 -m tools.pipeline.submission.profile.profile_cli profile-diff \
  --baseline ./profiles/baseline/capture.nsys-rep \
  --candidate ./profiles/regressed/capture.nsys-rep \
  --baseline-label good --candidate-label bad \
  --out ./profiles/profile-diff.md --json
```

Output: top NVTX-range deltas, top kernel-mix deltas (by total device time and
call count), and NCCL collective deltas. On hosts without `nsys` on PATH, pass
pre-extracted CSV dirs via `--baseline-csv-dir` / `--candidate-csv-dir`.

If the kernel mix is *unchanged* but wall time grew, suspect host-side
overhead and run `profile_host_overhead` (verb `host-overhead` in the same
CLI. Subverbs `top` / `record` / `dump`) before chasing kernels.

## 4. Per-kernel attribution

Fold the per-kernel evidence into a perf-report campaign cell so the renderer
and the lake see the same attribution (campaign scaffolding:
`perf_tune_report_campaign_init`, example config
[`configs/campaigns/example-campaign.yaml`](../../../../configs/campaigns/example-campaign.yaml)).

1. **nsys kernel summary -> kernels.json** (time-share attribution):

   ```bash
   python3 -m tools.perf_tune_report.perf_tune_report_cli import_nsys \
     --campaign <slug> --cell-id <cell> --bundle ./profiles/regressed --json
   ```

   The bundle needs `capture_sources.json` declaring `nsys` plus
   `nsys/cuda_gpu_kern_sum.txt` (from `nsys stats --report cuda_gpu_kern_sum`).

2. **ncu per-kernel Speed-of-Light** (the highest-rigor per-kernel layer),
   when an ncu bundle exists (skill:
   [`inference-kernel-ncu-profile`](../../skills/inference-kernel-ncu-profile/SKILL.md)):

   ```bash
   python3 -m tools.perf_tune_report.perf_tune_report_cli import_ncu \
     --campaign <slug> --cell-id <cell> --bundle <ncu-bundle-dir> --json
   ```

3. **Workload-level byte/FLOP grounding** against the published ceilings in
   `configs/sol-ceilings.yaml`, from a frozen DCGM snapshot of the bench
   window (template: [`configs/dcgm-frozen/example.yaml`](../../../../configs/dcgm-frozen/example.yaml)):

   ```bash
   python3 -m tools.perf_tune_report.perf_tune_report_cli dcgm_correlate \
     --campaign <slug> --cell-id <cell> \
     --frozen-yaml configs/dcgm-frozen/<name>.yaml --json
   ```

Every "%SoL" you quote must name the ceiling it is a percentage of
(METHODOLOGY.md "Speed-of-light framing". Roofline math:
[`tools/perf_tune_report/ROOFLINE-METHODOLOGY.md`](../tools/perf_tune_report/ROOFLINE-METHODOLOGY.md)).

## 5. Classify the offending kernel(s)

Apply the METHODOLOGY.md "Kernel-work classification" rubric to each kernel
the diff ranked. Climbing the wrong category wastes the engagement:

| Class | Signature in the data | Action |
|---|---|---|
| **K**nown-good | matches roofline expectation for its ceiling | move on - the regression is elsewhere |
| **R**educible | algorithmic/fusion headroom vs roofline | optimize the kernel or its launch shape |
| **H**idden | wall-time grew, device time did not. Gaps between kernels | chase launch/sync/host overhead (`profile_host_overhead`) |
| **P**arallelism-starved | low occupancy / load imbalance at fixed work | fix grid sizing, batching, or balance |
| **A**ttribution-error | implausible totals, empty windows, profile disagrees with DCGM | fix capture hygiene FIRST (back to step 2) |

## 6. Fix, then re-measure

Apply the smallest plausible fix (flag, fusion, backend pin, batching change).
Then repeat steps 2-3 on the fixed build: re-capture (hygiene-gated),
`profile-diff` fixed-vs-regressed, and rerun the original bench at the
original workload shape. A fix that cannot be seen in both the profile diff
*and* the end-to-end metric is not yet a fix.

## 7. Re-baseline and close the loop

1. **Record the new baseline** so the next regression diffs against the fixed
   build: `python3 -m tools.perf_baseline.perf_baseline_cli record ...`.
2. **If the fix is a required serve flag**, register it so it is never
   re-discovered the hard way:

   ```bash
   python3 -m tools.known_good_config.known_good_config_cli record \
     --model <model-id> --registry <registry.yaml> \
     --required-flag '<flag>|<match-regex>|<severity>|<why>' --json
   ```

3. **Report with full context** (METHODOLOGY.md "Full-context reporting"):
   hardware/topology, precision, engine + flags, workload shape, both absolute
   values, and the capture artifacts. Label the claim DRAFT or VERDICT.
4. **Name the next lever** (METHODOLOGY.md "Always be grinding") - e.g. add
   the finding and its `next_lever` to your value-findings registry (example
   shape: [`configs/value-findings.example.yaml`](../../../../configs/value-findings.example.yaml)).

---

## Quick reference

| Step | Tool / file |
|---|---|
| Detect / confirm | `perf_baseline_diff` (`tools/perf_baseline/perf_baseline_cli.py`) |
| Capture | `perf_tune_report_kernel_profile`. Gate with `scripts/nsys-validate-capture.sh` |
| Profile diff | `profile_profile_diff` (`tools/pipeline/submission/profile/profile_cli.py`) |
| Host overhead | `profile_host_overhead` (same CLI, `host-overhead` verb) |
| Per-kernel attribution | `perf_tune_report_import_nsys` / `perf_tune_report_import_ncu` |
| Byte/FLOP grounding | `perf_tune_report_dcgm_correlate` + `configs/sol-ceilings.yaml` |
| Classification rubric | `docs/METHODOLOGY.md` "Kernel-work classification" |
| Re-baseline | `perf_baseline_record`, `known_good_config_record` |
