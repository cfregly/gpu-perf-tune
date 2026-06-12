"""Unit tests for the TP right-size resolver (single source of truth for the energy/OPEX right-size).

Cases are anchored on the FLEET-ENERGY-AUDIT measured fleet (GB300, NVFP4, 2026-06-09)."""
from tp_rightsize_advisor import render_md, resolve


def test_nemotron_30b_a3b_tp4_is_over_provisioned():
    # measured: 3.6B active over TP4 = 0.9B/GPU, 2.73x worse tok/s/GPU than Qwen3-30B TP1
    r = resolve(total_params_b=30, active_params_b=3.6, dtype="nvfp4", current_tp=4)
    assert r.over_provisioned is True
    assert r.recommended_tp == 1
    assert r.gpu_reduction_factor == 4.0
    assert r.confidence == "measured"
    assert r.monthly_usd_saved_per_replica and r.monthly_usd_saved_per_replica > 0


def test_qwen3_30b_a3b_tp1_is_right_sized():
    # the measured right-sized anchor: 3B active / 1 GPU = loaded, 11,968 tok/s/GPU
    r = resolve(total_params_b=30, active_params_b=3.0, dtype="nvfp4", current_tp=1)
    assert r.over_provisioned is False
    assert r.recommended_tp == 1
    assert r.gpu_reduction_factor == 1.0


def test_qwen3_next_80b_a3b_tp4_over_but_extrapolated():
    # 3B active / TP4 = 0.75B/GPU underfilled, but total 80B > 40B -> verify w/ a TP A/B
    r = resolve(total_params_b=80, active_params_b=3.0, dtype="nvfp4", current_tp=4)
    assert r.over_provisioned is True
    assert r.recommended_tp == 1
    assert r.confidence == "extrapolated"
    assert any("extrapolated" in x for x in r.reasons)


def test_v4_flash_tp4_not_flagged_active_dense():
    # 13B active / TP4 = 3.25B/GPU >= 3B target -> NOT over-provisioned (large-active MoE)
    r = resolve(total_params_b=119, active_params_b=13.0, dtype="nvfp4", current_tp=4)
    assert r.over_provisioned is False
    assert r.recommended_tp == 4


def test_exaone_236b_a23b_tp4_not_flagged_even_though_fits_tp2():
    # active-density gate protects it: 23B / TP4 = 5.75B/GPU loaded, even though memory fits TP2
    r = resolve(total_params_b=236, active_params_b=23.0, dtype="fp8", current_tp=4)
    assert r.over_provisioned is False
    assert r.recommended_tp == 4


def test_dsv32_671b_a37b_memory_driven_tp4():
    # memory forces min TP4 (335GB NVFP4); active 37B/4 = 9.25B loaded -> not over, recommend stays 4
    r = resolve(total_params_b=671, active_params_b=37.0, dtype="nvfp4", current_tp=4)
    assert r.min_tp_memory == 4
    assert r.over_provisioned is False
    assert r.recommended_tp == 4


def test_no_current_tp_recommends_loaded_tp():
    r = resolve(total_params_b=30, active_params_b=3.0, dtype="nvfp4", current_tp=None)
    assert r.recommended_tp == 1
    assert r.over_provisioned is False
    assert r.gpu_reduction_factor is None


def test_render_md_flags_over_provisioning():
    r = resolve(total_params_b=30, active_params_b=3.6, dtype="nvfp4", current_tp=4)
    md = render_md(r)
    assert "RECOMMENDED TP: 1" in md
    assert "OVER-PROVISIONED" in md
    assert "[warn] **active_density**" in md
    assert "## Rationale" in md
