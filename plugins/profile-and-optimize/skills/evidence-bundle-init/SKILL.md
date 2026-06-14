---
name: evidence-bundle-init
last_validated: 2026-05-21
description: >-
  Scaffold a new evidence bundle directory ready for reproducibility-grade
  evidence capture: SOURCE.md (operator + cluster + git
  SHA + UTC timestamp), summary.md (verdict skeleton), commands/ (for the
  four-file .cmd/.stdout/.stderr/.exit tuple capture per shell command).
  Workload-agnostic. Works for any experiment family. Operator names family
  + run-id (or accepts default). Triggers on "new evidence bundle", "init
  evidence", "scaffold bundle", "start a new artifact bundle", "new run-id",
  "evidence-bundle-init", "set up a bundle", or any combination of "new /
  init / scaffold / create / start" with "evidence / bundle / artifact /
  run-id / experiment".
allowed-tools:
  - mcp__profile_and_optimize__evidence_init
  - Bash(mkdir:*)
  - Bash(date:*)
  - Bash(git:*)
  - Bash(hostname:*)
  - Bash(whoami:*)
  - Read
  - Write
---

# evidence-bundle-init

## Purpose

Set up a new evidence bundle directory under `experiments/artifacts/<family>/<run-id>/` with the skeleton reproducibility-grade evidence requires:

- `SOURCE.md` - operator identity, cluster, git SHA, UTC timestamp, the original prompt / intent.
- `summary.md` - verdict skeleton operator fills in as the experiment progresses.
- `commands/` - directory for the four-file `<NN>-<step>.{cmd,stdout,stderr,exit}` tuples one per shell command run during the experiment.
- `.gitkeep` markers so the directory layout survives an empty commit.

This skill is the operator-facing convenience over `mkdir + cat > SOURCE.md`. Half a minute of friction is enough that people skip the discipline. This skill makes it 5 seconds.

## Experiment isolation & traceability (required for any cluster-touching experiment)

The bundle's run-id IS the **experiment-id** - the single join key across the evidence bundle, the cluster objects, and the perf-lake. When the experiment creates cluster resources:

- Every Deployment/Pod/PVC/PV/Secret/ConfigMap/Service MUST use an experiment-unique name derived from the id (e.g. `glm51-expt-deepep-ll`, PV `glm51-deepep-expt-pv`) and carry the label `experiment=<id-slug>`.
- NEVER reuse a standing/platform/migration name (e.g. a shared `*-inference*` deployment, a standing `*-cache*` PV, or anything labeled `migration=*`). Cluster-scoped PV names are global. A collision silently breaks another owner's PVC.
- Tear down by label: `kubectl delete deploy,pod,pvc,secret -l experiment=<id-slug>`. For `Retain` PVs pre-clear the attacher finalizer before delete.
- Record the created object names + the perf-lake `campaign=<id>` in `SOURCE.md` (template below has the block).

## Observations vs mechanisms + roofline companions (measurement bundles)

A measurement bundle that produces a roofline / SoL analysis MUST separate **observations** (what
the instruments reported) from **mechanisms** (the causal "why"), and MUST carry a source-code
provenance block. This is the discipline a perf reviewer asks for: "separate the mechanisms from
the observations".

Scaffold these in any measurement bundle:

- `findings/01-observations.md` - **measured tables ONLY** (DCGM SM/tensor/DRAM %, tok/s, TPOT, AA
  numbers). No interpretation, no "because". Each number is reproducible from `commands/`.
- `findings/02-mechanisms.md` - one item per claim, formatted `OBSERVATION -> MECHANISM (causal) ->
  CONFIDENCE (+ what would raise it)`. A mechanism claim ("decode plateaus at 41% HBM because the
  sparse-MoE+MLA kernel mix has low DRAM efficiency") needs a profile (DCGM/zymtrace/nsys/ncu) -
  the rooflines are the observation. The mechanism is the separately-evidenced interpretation.
- `findings/00-ANSWERS-*.md` (optional) - the live-sync handout that answers the reviewer's questions
  directly, each pointing at 01/02.
- A ` ```provenance ` block (`experiment_provenance_v1`) in `SOURCE.md` pinning the exact
  vLLM/SGLang commit + delivery + patch, so the rendered roofline carries a source link (see
  [`server/tools/perf_tune_report/ROOFLINE-METHODOLOGY.md`](/plugins/profile-and-optimize/server/tools/perf_tune_report/ROOFLINE-METHODOLOGY.md)).
  Record the REAL `delivery` (`image|overlay|patchedVllm|infr-patch`) the bundle ran -- it is the
  code-under-test identity: a number from this bundle may be cited only as
  evidence for THAT delivery, never cross-tier (an `overlay`/offline-prepped run is not evidence for
  an `infr-patch`, even if the kernels match).

The prefill/decode roofline itself (page 7) is captured + always-published via the
`inference-perf-tune-report` / `inference-perf-bench` pipeline. This bundle just holds the obs/mechanisms
narrative + provenance that the report links to.

> This skill is backed by a native MCP verb: `mcp__profile_and_optimize__evidence_init`. The verb does the entire scaffold atomically (mkdir + SOURCE.md + summary.md + commands/README.md + .gitkeep) and returns the bundle path. The Bash-tool path documented below remains supported as a fallback.

## Why a bundle, not a flat directory

This repo's reproducibility-grade-evidence convention requires that significant experiments produce a bundle, not loose files:

- The bundle's path is the durable handle that future skills (`search_evidence`, `perf-baseline-record`, etc.) reach for.
- `SOURCE.md` is the audit trail that says "this evidence was captured by X on cluster Y on date Z from prompt W".
- `summary.md` is the human-readable verdict at the bottom of the funnel.
- `commands/` is the four-file tuple capture that makes every shell action replayable.

## When to use

- Starting any experiment that will produce >1 artifact file.
- Reviewer asked "where's the evidence for that claim?" and the answer is a bundle.
- Periodic capture (weekly perf-of-record snapshot, monthly drain audit, etc.).
- Pairs with every skill that writes artifacts ([`prometheus-anchored-query`](/plugins/profile-and-optimize/skills/prometheus-anchored-query/SKILL.md), [`perf-baseline-record`](/plugins/profile-and-optimize/skills/perf-baseline-record/SKILL.md)).

Do **not** use this skill for:

- One-off shell commands whose output you'll throw away - no bundle needed.
- Adding to an existing bundle - just `cd` into the bundle and add files.
- Benchmark families whose runbooks define their own bundle layout and naming conventions - follow those. This skill is the generic scaffolder.

## Example prompts

- "Init a new evidence bundle for the nccl sweep I'm about to run."
- "Scaffold a bundle under cluster-health family."
- "New evidence bundle for the gpu-burn soak."
- "Set up a bundle for the b200 8b regression investigation."
- `/evidence-bundle-init --family cluster-health --run-id rack-a-validation`
- `/evidence-bundle-init --family perf-baselines --measurement nccl_busbw`

## Prerequisites

1. **`PROFILE_AND_OPTIMIZE_REPO_ROOT`** for the bundle path. The skill writes to `${PROFILE_AND_OPTIMIZE_REPO_ROOT}/experiments/artifacts/<family>/<run-id>/`.
2. **Family** - `--family <name>` (e.g. `cluster-health`, `nccl-tests`, `gpu-burn`, `campaign/llama31_8b`).
3. **Run-id** - `--run-id <slug>` (default: `<UTC-timestamp>` if not supplied).
4. **Operator intent** - `--intent "<one-line description>"` (gets written into `SOURCE.md`).

## Interaction style

Fast and autonomous (3-5 seconds). Single optional pause: confirm the family + run-id + intent before write.

## Workflow

### Phase 0: resolve bundle path

```text
bundle = ${PROFILE_AND_OPTIMIZE_REPO_ROOT}/experiments/artifacts/<family>/<run-id>/
```

If the bundle already exists, **stop**. Bundles are immutable. New captures use a new run-id.

### Phase 1: gather provenance

In parallel:

- `Bash(date -u +%Y-%m-%dT%H:%M:%SZ)` - UTC timestamp.
- `Bash(hostname)` - workstation hostname.
- `Bash(whoami)` - operator user.
- `Bash(git -C ${PROFILE_AND_OPTIMIZE_REPO_ROOT} rev-parse HEAD)` - current SHA of the bundled server tree.
- `Bash(git -C ${PROFILE_AND_OPTIMIZE_REPO_ROOT} remote get-url origin)` - repo remote URL for the Provenance section.

### Phase 2: write the skeleton

**Preferred (MCP verb):**

```text
mcp__profile_and_optimize__evidence_init with:
  args: ["--family", "<family>",
         "--intent", "<operator one-line intent>",
         "--run-id", "<slug>",
         "--json"]
```

The verb does everything atomically and returns the bundle directory path. Skip to Phase 3.

**Fallback (Bash-tool):**

```text
mkdir -p ${bundle}/commands

Write ${bundle}/SOURCE.md with:
  # SOURCE

  **Family:** `<family>`
  **Run-id:** `<run-id>`
  **Created at (UTC):** `<ts>`
  **Created by:** `<USER>` on `<hostname>`
  **PROFILE_AND_OPTIMIZE SHA (bundled server):** `<git-sha>`

  ## Intent

  <operator's --intent text>

  ## Provenance

  - Workstation kernel: `<uname -a>`
  - Repo: `<git -C ${PROFILE_AND_OPTIMIZE_REPO_ROOT} remote get-url origin>` (this plugin marketplace).
  - Bundle path: `experiments/artifacts/<family>/<run-id>/`

  ## Experiment isolation & traceability

  The run-id IS the experiment-id: the single join key across this bundle, the
  cluster objects, and the perf-lake. (Matches the `mcp__profile_and_optimize__evidence_init`
  scaffold. Keep these as structured `- key: value` lines so `publish_to_lake` /
  `experiments_index` can read them.)

  - experiment_id: <run-id>
  - family: <e.g. nvfp4-kv | warp-decode | deepep | (blank)>
  - object label (EVERY cluster object, on metadata AND pod template): `experiment=<run-id>`
  - cluster resources created (fill in as you create them. Every
    Deployment/Pod/Job/PVC/PV/Secret/ConfigMap/Service, experiment-unique-named,
    NEVER a standing/migration name):
    -
  - perf-lake campaign: `campaign=<run-id>` (run `perftunereport campaign_init
    --experiment-id <run-id> --family <family> --evidence-bundle <this-bundle>` so
    campaign_id == experiment_id. The `s3://perf-lake/...` atlas_v1 + campaign_v1
    paths are auto-appended here by `publish_to_lake`).
  - pre-apply label gate: verify every manifest carries `experiment=<run-id>`
    before `kubectl apply`.

  ## Cross-references

  - `docs/METHODOLOGY.md` - the measurement-methodology canon.

Write ${bundle}/summary.md with:
  # Summary

  **Status:** in-progress

  ## Verdict

  <to-be-filled-in by operator at end of experiment>

  ## Findings

  -

  ## Recommendations

  -

  ## Open questions

  -

Write ${bundle}/commands/README.md with:
  # commands/

  Every shell command run during this experiment is captured as a four-file
  tuple:

      00-<step-slug>.cmd       # the exact command
      00-<step-slug>.stdout    # captured stdout
      00-<step-slug>.stderr    # captured stderr
      00-<step-slug>.exit      # exit code

  Filenames are zero-padded sequential (00, 01, 02, ...) so the chronological order
  is preserved in `ls`. Use a helper to capture all four atomically, e.g.:

      run() {
        local n="$1". Shift
        local slug="$1". Shift
        local prefix="$(printf '%02d-%s' "${n}" "${slug}")"
        printf '%s ' "$@" > "${prefix}.cmd". Echo >> "${prefix}.cmd"
        "$@" > "${prefix}.stdout" 2> "${prefix}.stderr"
        echo $? > "${prefix}.exit"
      }
      run 0 ls-image ls /mnt/data/images/

Touch ${bundle}/commands/.gitkeep so the empty directory survives a git add.
```

### Phase 3: report

Print the bundle path and the next-step pointer:

- "Add captures: `cd <bundle>` then run your commands with the `run` helper above."
- "Finalize: edit `summary.md` with verdict + findings before sharing."
- "Register a baseline if applicable: `perf-baseline-record --source <bundle>`."

## Output bundle layout

```
${PROFILE_AND_OPTIMIZE_REPO_ROOT}/experiments/artifacts/<family>/<run-id>/
  SOURCE.md
  summary.md
  commands/
    README.md
    .gitkeep
```

## Safety

- **Never overwrite an existing bundle.** Bundles are immutable. The skill refuses to init over a populated directory.
- **Audit trail.** `SOURCE.md` records the operator's `${USER}` + hostname so future readers know who captured the evidence, where, and from what prompt.
- **No automatic commit.** The skill creates the directory + scaffold files but does NOT run `git add` / `git commit`. The operator does that after the experiment is captured.

## Source-of-truth references

- [`docs/METHODOLOGY.md`](/docs/METHODOLOGY.md) - the measurement-methodology canon every bundle feeds.
- [`server/AGENTS.md`](/plugins/profile-and-optimize/server/AGENTS.md) - bundled-server discovery contract.
- All sibling skills that write artifacts - they all assume the bundle this skill creates.
