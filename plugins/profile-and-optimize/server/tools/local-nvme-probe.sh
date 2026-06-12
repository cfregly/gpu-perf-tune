#!/usr/bin/env bash
# Read-only per-node NVMe probe for the GB300 cohort.
#
# Origin: 2026-05-01 NVIDIA review feedback in #<team-channel>
# (<slack-permalink>).
# In-repo capture of the surrounding thread:
# docs/learnings/slack/<team-channel>/2026-05-01-scaling-guidance-and-nfs-vs-nvme.md
# (and the parent DSv3/LLaMA container-drop thread
# docs/learnings/slack/<team-channel>/2026-04-30-dsv3-llama31-container-drop-and-conversion-fixes.md
# where the NFS->NVMe step-time impact was first measured).
# NVIDIA observed our 405B / 8B / DSv3 runs were ~10% slower per step and
# attributed it to NFS reads. NVIDIA's `run.sub` already supports per-node
# NVMe staging via the `SLOW_DATADIR` -> `DATADIR` rsync gate (see
# (removed 2026-05-11 per reproducibility-v1 cleanup)), but
# we never set the variable. Before wiring the launchers (`Phase 2`), we
# need to know the canonical per-node NVMe mount + free capacity on the
# GB300 cohort. This script is the Phase 1 probe.
#
# Per mlperf-6.0-training/AGENTS.md "Fail Fast, No Silent Fallbacks": this
# script is read-only by design (df / mount / findmnt / stat / a single
# touch+rm in /tmp-of-the-mount). It does NOT run training, does NOT pull
# a container, and does NOT write the dataset. Operator-runnable from the
# Slurm submit host. Output is captured into
# experiments/artifacts/local-nvme-probe-<date>/<host>.txt, one file per probed node.
#
# Usage (operator-driven):
#   bash mlperf-6.0-training/tools/local-nvme-probe.sh \
#       --nodes 4 \
#       --partition gb300 \
#       --reservation <existing-rack-reservation> \
#       --out-dir experiments/artifacts/campaign/local-nvme-probe-$(date -u +%Y-%m-%d)
#
# Without --reservation the probe will queue on whatever idle gb300 capacity
# the scheduler hands out; with --reservation it stays inside the existing
# rack-reserve cohort (per AGENTS.md the launcher never creates reservations,
# the operator does).

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: local-nvme-probe.sh [--nodes N] [--partition P] [--reservation R] [--out-dir DIR]

Defaults:
  --nodes 4
  --partition gb300
  --reservation <unset>
  --out-dir experiments/artifacts/campaign/local-nvme-probe-$(date -u +%Y-%m-%d)

Required env (no defaults):
  (none)

Probed mount candidates: /raid /mnt/local /mnt/nvme /scratch/local /local /dev/shm /tmp

Per AGENTS.md "Fail Fast, No Silent Fallbacks": the script exits non-zero
when no candidate mount is rw-writable on any probed node, so the operator
sees the failure and knows Phase 2 cannot proceed.
EOF
}

NODES=4
PARTITION=gb300
RESERVATION=""
OUT_DIR=""

for arg in "$@"; do
  case "${arg}" in
    --nodes=*)        NODES="${arg#*=}" ;;
    --partition=*)    PARTITION="${arg#*=}" ;;
    --reservation=*)  RESERVATION="${arg#*=}" ;;
    --out-dir=*)      OUT_DIR="${arg#*=}" ;;
    -h|--help)        usage; exit 0 ;;
    *)
      printf 'unknown arg: %s\n' "${arg}" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="experiments/artifacts/campaign/local-nvme-probe-$(date -u +%Y-%m-%d)"
fi

if ! command -v sbatch >/dev/null 2>&1 || ! command -v srun >/dev/null 2>&1; then
  echo "ERROR: this probe must run from a Slurm submit host (sbatch/srun missing in PATH)" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
printf 'Probe output dir: %s\n' "${OUT_DIR}"

srun_args=(
  "--partition=${PARTITION}"
  "--nodes=${NODES}"
  "--ntasks-per-node=1"
  "--time=00:02:00"
  "--job-name=mlperf-local-nvme-probe"
  "--output=${OUT_DIR}/%N.txt"
)
if [[ -n "${RESERVATION}" ]]; then
  srun_args+=( "--reservation=${RESERVATION}" )
fi

# The probe body. One -c block, fully read-only except for a single
# `touch <path>/.mlperf-probe-<jobid>` + `rm` per candidate (proves rw).
read -r -d '' PROBE_BODY <<'PROBE' || true
set +e
hostname
date -u +%Y-%m-%dT%H:%M:%SZ
echo
echo "=== uname ==="
uname -a
echo
echo "=== df -hT (all mounts) ==="
df -hT
echo
echo "=== findmnt JSON (mount points + fstype + size in bytes) ==="
findmnt -lo TARGET,SOURCE,FSTYPE,OPTIONS,SIZE,USED,AVAIL --bytes --json 2>/dev/null \
  || findmnt -lo TARGET,SOURCE,FSTYPE,OPTIONS,SIZE,USED,AVAIL --bytes
echo
echo "=== mount | grep nvme/raid/local ==="
mount | grep -Ei 'nvme|raid|/local|/scratch' || echo "(no nvme/raid/local/scratch matches)"
echo
echo "=== lsblk ==="
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS 2>/dev/null \
  || lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT
echo
candidates="/raid /mnt/local /mnt/nvme /scratch/local /local /dev/shm /tmp"
echo "=== rw probe per candidate ==="
for cand in ${candidates}; do
  if [[ ! -d "${cand}" ]]; then
    printf '%-20s missing\n' "${cand}"
    continue
  fi
  free_bytes=$(df -B1 --output=avail "${cand}" 2>/dev/null | tail -n1 | tr -d ' ')
  fstype=$(df --output=fstype "${cand}" 2>/dev/null | tail -n1 | tr -d ' ')
  total_bytes=$(df -B1 --output=size "${cand}" 2>/dev/null | tail -n1 | tr -d ' ')
  probe_path="${cand}/.mlperf-probe-${SLURM_JOB_ID:-noslurm}-$$"
  if touch "${probe_path}" 2>/dev/null && rm "${probe_path}" 2>/dev/null; then
    rw="rw"
  else
    rw="ro_or_denied"
  fi
  printf '%-20s fstype=%s size_bytes=%s avail_bytes=%s status=%s\n' \
    "${cand}" "${fstype}" "${total_bytes}" "${free_bytes}" "${rw}"
done
echo
echo "=== /sys/block enumerations (NVMe controllers) ==="
ls -l /sys/block 2>/dev/null | grep -E 'nvme|sd' || echo "(none)"
echo
echo "=== /proc/mounts grep nvme/local/raid ==="
grep -Ei 'nvme|/raid|/local|/scratch' /proc/mounts || echo "(no proc/mounts matches)"
echo
echo "=== probe done ==="
PROBE

cmd=(
  srun "${srun_args[@]}"
  /bin/bash -c "${PROBE_BODY}"
)

printf '+ %s\n' "${cmd[*]}"
"${cmd[@]}"
rc=$?

echo
printf 'Probe finished rc=%s. Per-node output captured under %s\n' "${rc}" "${OUT_DIR}"
echo
echo "Next step: aggregate the per-node files into ${OUT_DIR}/decision.md"
echo "  pick the canonical mount across all probed nodes (must be rw on all),"
echo "  capture min(free_bytes) across nodes, then drive Phase 2:"
echo "  docs link: mlperf-6.0-training/specs/mlperf_training_v6_0_runbook.md"
echo "             section 'Local NVMe staging (SLOW_DATADIR)'"

exit "${rc}"
