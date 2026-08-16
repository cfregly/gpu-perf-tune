"""Catalog of common Slurm-job failure signatures.

Used by ``slurm_cli.triage`` to classify a failed job by grep-matching the
``slurm-<jobid>.out`` file. Each pattern has:

- ``klass`` -- a short string identifier (consumed by skills).
- ``regex`` -- a compiled regex (case-insensitive, multiline).
- ``description`` -- a human-readable one-liner.
- ``next_probe`` -- what the triage skill should recommend next.

Patterns are checked in priority order; the first to match wins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from re import Pattern


@dataclass(frozen=True)
class Signature:
    klass: str
    regex: Pattern[str]
    description: str
    next_probe: str


def _re(pattern: str) -> Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE)


def _re_dotall(pattern: str) -> Pattern[str]:
    """Compile a multi-line cross-newline pattern (used by shm_bloat_oom).

    The default ``_re`` keeps ``.`` line-local so multi-line slurm-out
    sections don't accidentally pull unrelated context into a match. The
    ``shm_bloat_oom`` signature explicitly needs DOTALL because the OOM
    line and the ``/dev/shm`` evidence are usually on adjacent lines, not
    the same line.
    """
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL)


SIGNATURES: tuple[Signature, ...] = (
    # NOTE: shm_bloat_oom MUST come before the generic `oom` so the more
    # specific pattern wins on priority order. Origin: the 2026-05-15
    # DSv3 671B 11-job NODE_FAIL incident root-caused to leftover /dev/shm
    # filling RAM and OOM-killing the dataloader worker (NVIDIA's worker
    # consumes ~50 GiB). The generic `oom` next_probe ("bump --mem") is
    # wrong for this root cause. The fix is to inspect and clear stale
    # shared-memory files under the operator's recovery policy.
    Signature(
        klass="shm_bloat_oom",
        regex=_re_dotall(
            # The OOM line and the /dev/shm evidence are usually on
            # adjacent lines. ``.{0,4000}?`` bounds the cross-line window
            # so the regex can't lazily match across hundreds of KB.
            r"(?:(?:out of memory.*?killed.*?signal\s*9|oomkilled|cgroup.*?oom-kill)"
            r".{0,4000}?(?:/dev/shm|Shmem:|nccl-[A-Za-z0-9]{6}|tmpfs.{0,40}?(?:100|9[5-9])\s*%))"
            r"|(?:(?:/dev/shm|Shmem:|nccl-[A-Za-z0-9]{6}|tmpfs.{0,40}?(?:100|9[5-9])\s*%)"
            r".{0,4000}?(?:out of memory.*?killed.*?signal\s*9|oomkilled|cgroup.*?oom-kill))"
            r"|(?:no space left on device.{0,200}?/dev/shm)"
            r"|(?:/dev/shm.{0,200}?no space left on device)"
        ),
        description=(
            "OOM kill correlated with /dev/shm bloat (leftover NCCL / python-mp / torch-shm / "
            "kernel-cache files filling tmpfs and pushing RAM out)"
        ),
        next_probe=(
            "Inspect /dev/shm occupancy and file ownership on the suspect nodes. "
            "Use k8s-troubleshooting for cluster evidence and slurm_drain before any "
            "operator-approved cleanup. Do not increase --mem until stale shared memory is ruled out."
        ),
    ),
    Signature(
        klass="oom",
        regex=_re(r"out of memory.*killed.*signal\s*9|oomkilled|cgroup.*oom-kill"),
        description="OOM kill detected (kernel OOM-killer signal 9 or cgroup OOM event)",
        next_probe="increase --mem in the next sbatch by ~25%, or reduce per-task batch / micro-batch",
    ),
    Signature(
        klass="nccl_hang",
        regex=_re(r"NCCL\s+WARN.*timeout|all[-_]?reduce.*timed out|NCCL INFO.*hang"),
        description="NCCL collective timeout / hang",
        next_probe="Use k8s-troubleshooting to inspect fabric evidence. Drain an unhealthy node with slurm_drain before retrying",
    ),
    Signature(
        klass="nccl_setup",
        regex=_re(r"ECONNREFUSED|connection refused.*NCCL|NCCL.*setup.*failed|failed to setup.*rdma"),
        description="NCCL setup failure (refused connection / RDMA QP setup error)",
        next_probe="Use k8s-troubleshooting to verify network reachability between the suspect node and its peers",
    ),
    Signature(
        klass="dataset_missing",
        regex=_re(r"no such file or directory.*dataset|cannot stat.*data|file not found.*\.bin"),
        description="Dataset path missing or unreadable",
        next_probe="Fix the dataset path. This is an operator-side issue, not a cluster issue",
    ),
    Signature(
        klass="image_missing",
        regex=_re(r"not found.*sqsh|cannot open.*image|pyxis.*image|enroot.*not found"),
        description="Pyxis / Enroot image missing or unreadable",
        next_probe="Fix the --container-image path and verify the image is readable from every target node",
    ),
    Signature(
        klass="walltime",
        regex=_re(r"DUE TO TIME LIMIT|SLURM job timeout|cancelled.*time limit"),
        description="Walltime exceeded",
        next_probe="either the workload is slower than expected (profile it) or the walltime guess was too tight",
    ),
    Signature(
        klass="node_failure",
        regex=_re(r"NODE[_ ]FAIL|node failure detected|srun.*epilog.*failed|slurmctld.*node.*down"),
        description="Slurm reported a node failure mid-job",
        next_probe="Use k8s-troubleshooting to inspect node evidence. Drain the node with slurm_drain if it remains unhealthy",
    ),
    Signature(
        klass="gpu_xid",
        regex=_re(r"NVRM.*Xid|Xid\s+\d+|dcgm.*xid.*error"),
        description="GPU XID error fired during the run",
        next_probe="Use k8s-troubleshooting to capture GPU evidence. Drain the failing node with slurm_drain and follow the operator hardware process",
    ),
    Signature(
        klass="fabric",
        regex=_re(r"couldn't find any working interface for libibverbs|IB.*HCA.*error|infiniband.*link.*down"),
        description="IB / libibverbs interface error",
        next_probe="Use k8s-troubleshooting to inspect the suspect node and peers. Drain the node with slurm_drain if the link is down",
    ),
)


def match_signatures(text: str) -> list[tuple[Signature, int]]:
    """Return a list of (signature, hit_count) for every signature that matched.

    Sorted by hit_count descending then by SIGNATURES order. Caller can use
    confidence-from-hit-count heuristics (>=3 hits => HIGH, 1-2 => MEDIUM, etc.).
    """
    hits: list[tuple[Signature, int]] = []
    for sig in SIGNATURES:
        matches = sig.regex.findall(text)
        if matches:
            hits.append((sig, len(matches)))
    hits.sort(key=lambda pair: (-pair[1], SIGNATURES.index(pair[0])))
    return hits
