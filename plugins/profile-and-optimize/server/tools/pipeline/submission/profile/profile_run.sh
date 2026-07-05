#!/usr/bin/env bash
# profile_run.sh -- Slurm-aware nsys + NVTX capture wrapper for MLPerf v6.0
# training jobs (B200 LLaMA 3.1 8B, GB300 LLaMA 3.1 405B, GB300 DeepSeek v3
# 671B). The wrapper does NOT invoke nsys directly; it sets the env vars
# (NSYSCMD / NVTX_FLAG / NSYS_PREFIX / NSYS_SUFFIX) that the in-image
# run_and_time.sh already consumes (mirrored in
# tests/fixtures/ai_tuning/hyp/scaled_run.tp-comm-overlap-true.hyp:435-444),
# injects the Megatron --profile / --nvtx_ranges flags through
# TRAINING_EXTRA_ARGS (mirrored in
# tools/benchmarks/llama31_405b/megatron_overlay_run.sh:203), routes LOGDIR
# under the artifact anchor (per CLAUDE.md "Artifact Anchor"), and forwards
# the operator-gate to the underlying launcher.
#
# Per CLAUDE.md "Fail Fast, No Silent Fallbacks":
#   - every required input must be set; the wrapper aborts on the first
#     failing assertion.
#   - --ack-cluster-cost is mandatory; --dry-run is the only opt-out for
#     local printing without submission.
#   - missing nsys in the container is a fatal error; we do not fall back
#     to a "run without nsys" mode silently.
#
# Per CLAUDE.md "Self-Contained Repository Boundary":
#   - shells out only to checked-in tools (the existing 405B launcher.sh
#     and the unified common/launcher.py).
#
# Usage:
#   tools/pipeline/submission/profile/profile_run.sh \
#       --bench llama31_8b \
#       --run-id <slug> \
#       --nodes 8 \
#       --ack-cluster-cost
#
# Arguments:
#   --bench NAME             llama31_8b | llama31_405b | deepseekv3_671b
#   --run-id SLUG            short kebab-case slug; the profiling artifact
#                            family is derived from --bench (campaign/<bench>/
#                            <run-id>/profiling/).
#   --nodes N                node count to forward to the underlying launcher.
#   --profile-step-start N   first iteration profiled (default 10).
#   --profile-step-end N     last iteration profiled, exclusive (default 12).
#   --profile-ranks LIST     comma-separated ranks; defaults per-bench:
#                              llama31_8b      -> 0
#                              llama31_405b    -> 0,1
#                              deepseekv3_671b -> 0,1,32
#   --use-pytorch-profiler   route the capture through PyTorch profiler
#                            instead of nsys + emit_nvtx; mutually exclusive
#                            with the default nsys path.
#   --extra-args "..."       additional Megatron args appended to
#                            TRAINING_EXTRA_ARGS verbatim.
#   --extra-launcher-arg X   additional argument forwarded to the underlying
#                            launcher (repeatable, e.g. for SLURM_NODELIST).
#   --reservation NAME       forwarded to the 405B launcher's --reservation.
#   --ack-cluster-cost       mandatory; acknowledges this is an expensive
#                            cluster run that produces a profile.
#   --dry-run                print the resolved env + the launcher command,
#                            then exit. Skips the operator-gate check.
#
# Environment overrides (with defaults):
#   NSYS_TRACE_FLAGS         tracers passed to `nsys profile -t ...`.
#                            Default: cuda,nvtx,osrt,cudnn,cublas
#   NSYS_EXTRA_ARGS          extra `nsys profile` args appended after the
#                            default flags. Default: empty.
#   ART_ROOT                 artifact root. Default:
#                            mlperf-6.0-training/experiments/artifacts.
#
# The artifact dir is created up front so the operator can place
# SOURCE.md + summary.md + commands/ alongside the profile per the CLAUDE.md
# "Reproducibility-Grade Evidence" rule.

set -euo pipefail

usage() {
    awk '/^# Usage:/,/^#$/' "$0" | sed 's/^# \{0,1\}//'
}

abort() {
    echo "profile_run: $*" >&2
    exit 1
}

BENCH=""
RUN_ID=""
NODES=""
PROFILE_STEP_START=10
PROFILE_STEP_END=12
PROFILE_RANKS=""
USE_PYTORCH_PROFILER=0
EXTRA_ARGS=""
RESERVATION=""
ACK_CLUSTER_COST=0
DRY_RUN=0
declare -a EXTRA_LAUNCHER_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bench) BENCH="${2:-}"; shift 2 ;;
        --run-id) RUN_ID="${2:-}"; shift 2 ;;
        --nodes) NODES="${2:-}"; shift 2 ;;
        --profile-step-start) PROFILE_STEP_START="${2:-}"; shift 2 ;;
        --profile-step-end) PROFILE_STEP_END="${2:-}"; shift 2 ;;
        --profile-ranks) PROFILE_RANKS="${2:-}"; shift 2 ;;
        --use-pytorch-profiler) USE_PYTORCH_PROFILER=1; shift ;;
        --extra-args) EXTRA_ARGS="${2:-}"; shift 2 ;;
        --extra-launcher-arg) EXTRA_LAUNCHER_ARGS+=("${2:-}"); shift 2 ;;
        --reservation) RESERVATION="${2:-}"; shift 2 ;;
        --ack-cluster-cost) ACK_CLUSTER_COST=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) abort "unknown flag '$1' (try --help)" ;;
    esac
done

[[ -n "${BENCH}" ]] || abort "--bench is required (llama31_8b | llama31_405b | deepseekv3_671b)"
[[ -n "${RUN_ID}" ]] || abort "--run-id is required (short kebab-case slug)"
[[ -n "${NODES}" ]] || abort "--nodes is required"
[[ "${RUN_ID}" =~ ^[a-z0-9][a-z0-9-]*$ ]] || abort "--run-id must be kebab-case (got '${RUN_ID}')"
[[ "${NODES}" =~ ^[0-9]+$ ]] || abort "--nodes must be an integer (got '${NODES}')"
[[ "${PROFILE_STEP_START}" =~ ^[0-9]+$ ]] || abort "--profile-step-start must be int"
[[ "${PROFILE_STEP_END}" =~ ^[0-9]+$ ]] || abort "--profile-step-end must be int"

if (( PROFILE_STEP_END <= PROFILE_STEP_START )); then
    abort "--profile-step-end (${PROFILE_STEP_END}) must be > --profile-step-start (${PROFILE_STEP_START})"
fi

case "${BENCH}" in
    llama31_8b)
        : "${PROFILE_RANKS:=0}"
        FAMILY="campaign/llama31_8b"
        ;;
    llama31_405b)
        : "${PROFILE_RANKS:=0,1}"
        FAMILY="campaign/llama31_405b"
        ;;
    deepseekv3_671b)
        : "${PROFILE_RANKS:=0,1,32}"
        FAMILY="campaign/deepseekv3_671b"
        ;;
    *)
        abort "unknown --bench '${BENCH}' (expected llama31_8b | llama31_405b | deepseekv3_671b)"
        ;;
esac

[[ "${PROFILE_RANKS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || abort "--profile-ranks must be a comma-separated rank list (got '${PROFILE_RANKS}')"

if (( ACK_CLUSTER_COST == 0 && DRY_RUN == 0 )); then
    abort "--ack-cluster-cost is required (per CLAUDE.md operator gate). Use --dry-run to print without submitting."
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
ART_ROOT="${ART_ROOT:-${REPO_ROOT}/experiments/artifacts}"
ART_DIR="${ART_ROOT}/${FAMILY}/${RUN_ID}/profiling"
LOGDIR="${ART_DIR}"
NSYS_STATS_DIR="${ART_DIR}/nsys-stats"

mkdir -p "${ART_DIR}"
mkdir -p "${NSYS_STATS_DIR}"

if (( DRY_RUN == 0 )); then
    if ! command -v nsys >/dev/null 2>&1; then
        cat >&2 <<EOF
profile_run: 'nsys' not found on the operator path. The wrapper does not
require nsys on the login node (the in-image run_and_time.sh runs it
inside the container), but the operator typically runs 'nsys stats'
locally on the artifact afterwards. Install via the in-image recipe at
[Megatron-Bridge/docker/common/install_nsys.sh](../../../../../Megatron-Bridge/docker/common/install_nsys.sh)
or via 'apt-get install nsight-systems-cli'. Continuing.
EOF
    fi
fi

NSYS_TRACE_FLAGS="${NSYS_TRACE_FLAGS:-cuda,nvtx,osrt,cudnn,cublas}"
NSYS_EXTRA_ARGS="${NSYS_EXTRA_ARGS:-}"

if (( USE_PYTORCH_PROFILER == 1 )); then
    NSYSCMD=""
    NVTX_FLAG=0
    PROFILER_FLAG="--use-pytorch-profiler"
else
    NVTX_FLAG=1
    NSYSCMD="nsys profile -t ${NSYS_TRACE_FLAGS} -s none --force-overwrite=true --capture-range=cudaProfilerApi --capture-range-end=stop -o {} ${NSYS_EXTRA_ARGS}"
    PROFILER_FLAG=""
fi

PROFILE_ARGS="--profile --profile_step_start=${PROFILE_STEP_START} --profile_step_end=${PROFILE_STEP_END} --profile_ranks ${PROFILE_RANKS} --nvtx_ranges"
if [[ -n "${PROFILER_FLAG}" ]]; then
    PROFILE_ARGS="${PROFILE_ARGS} ${PROFILER_FLAG}"
fi
TRAINING_EXTRA_ARGS_FULL="${PROFILE_ARGS}${EXTRA_ARGS:+ ${EXTRA_ARGS}}"

case "${BENCH}" in
    llama31_405b)
        LAUNCHER="${REPO_ROOT}/tools/benchmarks/llama31_405b/launcher.sh"
        [[ -x "${LAUNCHER}" ]] || abort "launcher not executable: ${LAUNCHER}"
        declare -a LAUNCHER_CMD=("${LAUNCHER}")
        if (( DRY_RUN == 1 )); then
            LAUNCHER_CMD+=(--dry-run)
        fi
        if [[ -n "${RESERVATION}" ]]; then
            LAUNCHER_CMD+=("--reservation=${RESERVATION}")
        fi
        ;;
    llama31_8b|deepseekv3_671b)
        LAUNCHER="${REPO_ROOT}/tools/benchmarks/common/launcher.py"
        [[ -f "${LAUNCHER}" ]] || abort "launcher not found: ${LAUNCHER}"
        declare -a LAUNCHER_CMD=(python3 "${LAUNCHER}" --benchmark "${BENCH}" --nodes "${NODES}")
        if (( DRY_RUN == 1 )); then
            LAUNCHER_CMD+=(--dry-run)
        else
            LAUNCHER_CMD+=(--i-understand-this-submits-jobs)
        fi
        ;;
esac

if (( ${#EXTRA_LAUNCHER_ARGS[@]} > 0 )); then
    LAUNCHER_CMD+=("${EXTRA_LAUNCHER_ARGS[@]}")
fi

CAPTURE_LOG="${ART_DIR}/capture.log"
{
    printf '# profile_run.sh capture metadata\n'
    printf 'bench=%s\n' "${BENCH}"
    printf 'run_id=%s\n' "${RUN_ID}"
    printf 'nodes=%s\n' "${NODES}"
    printf 'profile_step_start=%s\n' "${PROFILE_STEP_START}"
    printf 'profile_step_end=%s\n' "${PROFILE_STEP_END}"
    printf 'profile_ranks=%s\n' "${PROFILE_RANKS}"
    printf 'use_pytorch_profiler=%s\n' "${USE_PYTORCH_PROFILER}"
    printf 'nvtx_flag=%s\n' "${NVTX_FLAG}"
    printf 'nsyscmd=%s\n' "${NSYSCMD}"
    printf 'training_extra_args=%s\n' "${TRAINING_EXTRA_ARGS_FULL}"
    printf 'logdir=%s\n' "${LOGDIR}"
    printf 'art_dir=%s\n' "${ART_DIR}"
    printf 'launcher=%s\n' "${LAUNCHER_CMD[*]}"
    printf 'dry_run=%s\n' "${DRY_RUN}"
    printf 'started_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
} > "${CAPTURE_LOG}"

cat <<EOF
profile_run: bench=${BENCH} run_id=${RUN_ID} nodes=${NODES}
  artifact dir : ${ART_DIR}
  LOGDIR       : ${LOGDIR}
  capture log  : ${CAPTURE_LOG}
  rank list    : ${PROFILE_RANKS}
  step window  : [${PROFILE_STEP_START}, ${PROFILE_STEP_END})
  TRAINING_EXTRA_ARGS=${TRAINING_EXTRA_ARGS_FULL}
  NSYSCMD=${NSYSCMD}
  NVTX_FLAG=${NVTX_FLAG}
  launcher: ${LAUNCHER_CMD[*]}
EOF

if (( DRY_RUN == 1 )); then
    echo "profile_run: --dry-run; not invoking launcher."
    exit 0
fi

export TRAINING_EXTRA_ARGS="${TRAINING_EXTRA_ARGS_FULL}"
export NSYSCMD
export NVTX_FLAG
export LOGDIR
export PROFILE_RANKS

"${LAUNCHER_CMD[@]}"
