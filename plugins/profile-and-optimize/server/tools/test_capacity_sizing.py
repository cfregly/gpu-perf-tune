"""Unit tests for the SLA-first capacity sizer (single source of truth for TPM -> pods/GPUs).

Cases are anchored on the GB300 MiniMax-M2.7-NVFP4 leaderboard cells and the 3M-TPM thread's
own arithmetic (18 pods x 2916 tok/s/pod ~= 50k tok/s = 3M TPM)."""
import math

import pytest

from capacity_sizing import Anchor, parse_anchors, render_md, resolve

# GB300 MiniMax-M2.7-NVFP4 TP4 measured tok/s/user anchors (AA aa-10k, L3): c=1, c=10, knee c=256.
MINIMAX = [Anchor(1, 215.5), Anchor(10, 122.2), Anchor(256, 41.8)]
TPM_3M = 3_000_000  # = 50,000 output tok/s


def test_thread_18_pods_crosscheck():
    # The thread: per-pod 2916 tok/s at c=128 (tok/s/user 22.78), 8-GPU pod, 100% util -> 18 pods.
    r = resolve(tpm=TPM_3M, sla_list=[2916 / 128], anchors=[Anchor(128, 2916 / 128)],
                gpus_per_pod=8, util=1.0, model="thread-crosscheck")
    row = r.rows[0]
    assert row.replicas == 18
    assert row.gpus == 144
    assert abs(row.per_replica_tps - 2916.0) < 1.0


def test_minimax_knee_sla():
    # At/below the knee SLA (~42 tok/s/user) the per-pod throughput plateaus -> 7 pods @70%, 28 GPUs.
    r = resolve(tpm=TPM_3M, sla_list=[41.8], anchors=MINIMAX, gpus_per_pod=4, util=0.70, model="MiniMax-M2.7")
    row = r.rows[0]
    assert row.concurrency == 256
    assert row.replicas == 7
    assert row.gpus == 28
    assert row.regime == "below-knee-plateau"


def test_minimax_c10_measured_anchor():
    # SLA == the c=10 measured anchor (122.2): per-pod 1222 -> 59 pods @70%, 236 GPUs.
    r = resolve(tpm=TPM_3M, sla_list=[122.2], anchors=MINIMAX, gpus_per_pod=4, util=0.70, model="MiniMax-M2.7")
    row = r.rows[0]
    assert row.concurrency == 10
    assert row.regime == "measured-anchor"
    assert row.replicas == 59
    assert row.gpus == 236


def test_minimax_interp_row():
    # SLA=100 is interpolated between c=10 and c=256: c* ~24.5 -> 30 pods @70%, 120 GPUs.
    r = resolve(tpm=TPM_3M, sla_list=[100.0], anchors=MINIMAX, gpus_per_pod=4, util=0.70, model="MiniMax-M2.7")
    row = r.rows[0]
    assert row.regime == "interp"
    assert 24.0 <= row.concurrency <= 25.0
    assert row.replicas == 30
    assert row.gpus == 120


def test_minimax_below_knee_plateau_is_flat():
    # SLA below the knee (20) shares the knee per-pod throughput -> same pod count as the knee.
    r = resolve(tpm=TPM_3M, sla_list=[20.0], anchors=MINIMAX, gpus_per_pod=4, util=0.70, model="MiniMax-M2.7")
    row = r.rows[0]
    assert row.regime == "below-knee-plateau"
    assert row.replicas == 7


def test_sla_above_c1_is_latency_tier_infeasible():
    # An SLA tighter than the c=1 anchor (300 > 215.5): latency tier, clamps to c=1, huge pod count.
    r = resolve(tpm=TPM_3M, sla_list=[300.0], anchors=MINIMAX, gpus_per_pod=4, util=0.70, model="MiniMax-M2.7")
    row = r.rows[0]
    assert row.regime == "latency-tier-infeasible"
    assert row.concurrency == 1
    assert row.replicas == 332


def test_gpu_count_swings_with_sla():
    # Same TPM, same model: GPU count is SLA-driven (relaxed << tight).
    r = resolve(tpm=TPM_3M, sla_list=[42, 122.2], anchors=MINIMAX, gpus_per_pod=4, util=0.70)
    relaxed = next(x for x in r.rows if x.sla_toks_per_user == 42)
    tight = next(x for x in r.rows if x.sla_toks_per_user == 122.2)
    assert tight.gpus > 5 * relaxed.gpus


def test_render_md_has_table():
    r = resolve(tpm=TPM_3M, sla_list=[42, 100], anchors=MINIMAX, model="MiniMax-M2.7")
    md = render_md(r)
    assert "Capacity sizing (SLA-first): MiniMax-M2.7" in md
    assert "| SLA tok/s/user |" in md


def test_parse_anchors_roundtrip():
    a = parse_anchors("1:215.5, 10:122.2, 256:41.8")
    assert [x.concurrency for x in a] == [1, 10, 256]
    assert abs(a[0].per_replica_tps - 215.5) < 1e-6


def test_fail_loud_non_monotonic_anchors():
    # tok/s/user must be non-increasing in concurrency
    with pytest.raises(ValueError):
        parse_anchors("10:100,256:200")


def test_fail_loud_empty_anchors():
    with pytest.raises(ValueError):
        parse_anchors("")


def test_fail_loud_bad_tpm():
    with pytest.raises(ValueError):
        resolve(tpm=0, sla_list=[50], anchors=MINIMAX)


def test_fail_loud_bad_sla():
    with pytest.raises(ValueError):
        resolve(tpm=TPM_3M, sla_list=[-5], anchors=MINIMAX)


def test_fail_loud_bad_util():
    with pytest.raises(ValueError):
        resolve(tpm=TPM_3M, sla_list=[50], anchors=MINIMAX, util=1.5)
