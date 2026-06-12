"""Unit tests for the TPM-across-hardware rollup (tpm_summary).

Coverage:

- peak operating-point selection (max output_tps_per_gpu)
- SLA operating-point selection + threshold filtering (TTFT and TPOT/ITL)
- the three capacity bases (per_gpu / per_replica / per_node)
- output-only vs total TPM (total null when total_tps_per_gpu absent)
- SLA thresholds unset -> sla=None, sla_computed=False
- grouping collapses cells with the same identity
- markdown / csv / json serializers
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.perf_tune_report.schema import AtlasCell
from tools.perf_tune_report.tpm_summary import (
    DEFAULT_GPUS_PER_NODE,
    DEFAULT_TPOT_SLA_MS,
    DEFAULT_TTFT_SLA_MS,
    DEFAULT_USD_PER_GPU_HOUR,
    SECONDS_PER_MINUTE,
    compute_tpm_summary,
    discover_tpm_config,
)


def _row(**overrides: Any) -> AtlasCell:
    base = dict(
        cell_id="c",
        model="GLM-5.1-NVFP4",
        hardware="B200",
        quant="NVFP4",
        tensor_parallel=8,
        parallel_strategy="TP",
        mtp=False,
        max_num_batched_tokens=4096,
        concurrency=1,
        status="full",
        ttft_avg_ms=100.0,
        request_throughput_avg=1.0,
        output_tps_per_user=50.0,
        output_tps_per_gpu=10.0,
        total_tps_per_gpu=40.0,
        tpot_median_ms=20.0,
        backend="vllm-sweep",
    )
    base.update(overrides)
    return AtlasCell(**base)


def test_peak_picks_max_output_tps_per_gpu():
    rows = [
        _row(concurrency=1, output_tps_per_gpu=10.0),
        _row(concurrency=8, output_tps_per_gpu=30.0),
        _row(concurrency=32, output_tps_per_gpu=25.0),
    ]
    s = compute_tpm_summary(rows)
    assert len(s.groups) == 1
    peak = s.groups[0].peak
    assert peak is not None
    assert peak.concurrency == 8
    assert peak.output_tps_per_gpu == 30.0


def test_three_bases_scale_correctly():
    rows = [_row(output_tps_per_gpu=10.0, tensor_parallel=8, total_tps_per_gpu=40.0)]
    s = compute_tpm_summary(rows, gpus_per_node=8)
    peak = s.groups[0].peak
    # output: 10 tok/s/GPU * 60 = 600 TPM/GPU
    assert peak.output_tpm_per_gpu == 10.0 * SECONDS_PER_MINUTE
    assert peak.output_tpm_per_replica == 10.0 * 8 * SECONDS_PER_MINUTE
    assert peak.output_tpm_per_node == 10.0 * 8 * SECONDS_PER_MINUTE
    # total: 40 tok/s/GPU * 60
    assert peak.total_tpm_per_gpu == 40.0 * SECONDS_PER_MINUTE
    assert peak.total_tpm_per_replica == 40.0 * 8 * SECONDS_PER_MINUTE


def test_per_node_basis_independent_of_tp():
    rows = [_row(output_tps_per_gpu=10.0, tensor_parallel=16)]
    s = compute_tpm_summary(rows, gpus_per_node=8)
    peak = s.groups[0].peak
    assert peak.output_tpm_per_replica == 10.0 * 16 * SECONDS_PER_MINUTE
    assert peak.output_tpm_per_node == 10.0 * 8 * SECONDS_PER_MINUTE


def test_total_tpm_null_when_total_absent():
    rows = [_row(total_tps_per_gpu=None)]
    s = compute_tpm_summary(rows)
    peak = s.groups[0].peak
    assert peak.total_tpm_per_gpu is None
    assert peak.total_tpm_per_replica is None
    assert peak.total_tpm_per_node is None
    # markdown renders the missing total as n/a, not a crash.
    md = s.to_markdown()
    assert "n/a" in md


def test_sla_not_computed_when_no_threshold():
    rows = [_row(output_tps_per_gpu=10.0), _row(concurrency=8, output_tps_per_gpu=30.0)]
    s = compute_tpm_summary(rows)
    assert s.sla_computed is False
    assert s.groups[0].sla is None
    assert "not computed" in s.to_markdown() or "not set" in s.to_markdown()


def test_sla_picks_highest_throughput_under_thresholds():
    rows = [
        _row(concurrency=1, output_tps_per_gpu=10.0, ttft_avg_ms=100.0, tpot_median_ms=20.0),
        _row(concurrency=8, output_tps_per_gpu=30.0, ttft_avg_ms=400.0, tpot_median_ms=45.0),
        _row(concurrency=32, output_tps_per_gpu=50.0, ttft_avg_ms=900.0, tpot_median_ms=80.0),
    ]
    # SLA: TTFT<=500, TPOT<=50 -> c=32 fails (900/80), c=8 passes (400/45),
    # c=1 passes. Highest-throughput passing point is c=8.
    s = compute_tpm_summary(rows, ttft_sla_ms=500.0, tpot_sla_ms=50.0)
    assert s.sla_computed is True
    sla = s.groups[0].sla
    assert sla is not None
    assert sla.concurrency == 8
    assert sla.output_tps_per_gpu == 30.0
    # peak is still the unconstrained max (c=32).
    assert s.groups[0].peak.concurrency == 32


def test_sla_none_when_no_point_meets():
    rows = [_row(concurrency=1, output_tps_per_gpu=10.0, ttft_avg_ms=5000.0, tpot_median_ms=200.0)]
    s = compute_tpm_summary(rows, ttft_sla_ms=500.0, tpot_sla_ms=50.0)
    assert s.sla_computed is True
    assert s.groups[0].sla is None
    # markdown shows an explicit "no point met SLA" row, not silent omission.
    assert "no point met SLA" in s.to_markdown()


def test_sla_uses_itl_when_tpot_absent():
    rows = [_row(output_tps_per_gpu=10.0, ttft_avg_ms=100.0, tpot_median_ms=None, itl_avg_ms=30.0)]
    # TPOT absent but ITL=30 <= 40 -> meets SLA.
    s = compute_tpm_summary(rows, tpot_sla_ms=40.0)
    assert s.groups[0].sla is not None
    # ITL=30 > 20 -> fails.
    s2 = compute_tpm_summary(rows, tpot_sla_ms=20.0)
    assert s2.groups[0].sla is None


def test_grouping_collapses_same_identity():
    rows = [
        _row(cell_id="a", concurrency=1, output_tps_per_gpu=10.0),
        _row(cell_id="b", concurrency=8, output_tps_per_gpu=30.0),
    ]
    s = compute_tpm_summary(rows)
    assert len(s.groups) == 1  # same model/hw/quant/TP/strategy/mtp


def test_distinct_hardware_make_distinct_groups():
    rows = [
        _row(hardware="B200", output_tps_per_gpu=30.0),
        _row(hardware="H100", tensor_parallel=16, output_tps_per_gpu=20.0),
        _row(hardware="GB300", tensor_parallel=4, output_tps_per_gpu=40.0),
    ]
    s = compute_tpm_summary(rows)
    assert {g.hardware for g in s.groups} == {"B200", "H100", "GB300"}
    # Display order: H100, B200, GB300.
    assert [g.hardware for g in s.groups] == ["H100", "B200", "GB300"]


def test_rows_without_output_tps_are_skipped():
    rows = [_row(output_tps_per_gpu=None, status="failed")]
    s = compute_tpm_summary(rows)
    assert s.groups == []


def test_discover_tpm_config_reads_block(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "name: c\ntpm:\n  ttft_sla_ms: 500\n  tpot_sla_ms: 50\n  gpus_per_node: 4\n"
    )
    cfg = discover_tpm_config(tmp_path)
    assert cfg.ttft_sla_ms == 500.0
    assert cfg.tpot_sla_ms == 50.0
    assert cfg.gpus_per_node == 4


def test_discover_tpm_config_defaults_when_absent(tmp_path):
    # No config.yaml at all -> code-defaults (v1.49.0): default SLA + default
    # cost table so SLA-TPM + $/1M populate for every campaign.
    cfg = discover_tpm_config(tmp_path)
    assert cfg.ttft_sla_ms == DEFAULT_TTFT_SLA_MS
    assert cfg.tpot_sla_ms == DEFAULT_TPOT_SLA_MS
    assert cfg.gpus_per_node == DEFAULT_GPUS_PER_NODE
    assert cfg.usd_per_gpu_hour == DEFAULT_USD_PER_GPU_HOUR
    assert cfg.usd_per_gpu_hour["B200"] == 8.60
    # GB300 = assumed estimate (no public list rate).
    assert cfg.usd_per_gpu_hour["GB300"] == 12.00


def test_discover_tpm_config_defaults_when_no_block(tmp_path):
    (tmp_path / "config.yaml").write_text("name: c\ncells: []\n")
    cfg = discover_tpm_config(tmp_path)
    assert cfg.ttft_sla_ms == DEFAULT_TTFT_SLA_MS
    assert cfg.gpus_per_node == DEFAULT_GPUS_PER_NODE
    assert cfg.usd_per_gpu_hour == DEFAULT_USD_PER_GPU_HOUR


def test_discover_tpm_config_partial_block_fills_defaults(tmp_path):
    # Only ttft set; tpot + gpus_per_node absent -> per-field default fallback.
    (tmp_path / "config.yaml").write_text("tpm:\n  ttft_sla_ms: 750\n")
    cfg = discover_tpm_config(tmp_path)
    assert cfg.ttft_sla_ms == 750.0
    assert cfg.tpot_sla_ms == DEFAULT_TPOT_SLA_MS
    assert cfg.gpus_per_node == DEFAULT_GPUS_PER_NODE


def test_discover_tpm_config_cost_block_overlays_defaults(tmp_path):
    # A cost: block overrides one default rate AND adds a brand-new hardware
    # key; the other default rates remain.
    (tmp_path / "config.yaml").write_text(
        "cost:\n  usd_per_gpu_hour:\n    B200: 5.0\n    MI325X: 3.0\n"
    )
    cfg = discover_tpm_config(tmp_path)
    assert cfg.usd_per_gpu_hour["B200"] == 5.0          # override of default
    assert cfg.usd_per_gpu_hour["MI325X"] == 3.0        # brand-new key added
    assert cfg.usd_per_gpu_hour["H100"] == 6.16         # default retained
    assert cfg.usd_per_gpu_hour["GB300"] == 12.00       # default estimate retained
    assert "campaign config cost: block" in cfg.cost_rate_source


def test_discover_tpm_config_malformed_yaml(tmp_path):
    (tmp_path / "config.yaml").write_text("tpm:\n  ttft_sla_ms: [unterminated\n")
    cfg = discover_tpm_config(tmp_path)  # must not raise
    assert cfg.gpus_per_node == DEFAULT_GPUS_PER_NODE
    assert cfg.usd_per_gpu_hour == DEFAULT_USD_PER_GPU_HOUR


def _campaign_under_cost_yaml(tmp_path, cost_yaml_text, config_text=""):
    """Nest a campaign dir under perf-tune-report/configs/cost.yaml so the walk-up finds it.
    A published campaign always has a config.yaml (just no cost: block) -- write a minimal
    one so discover_tpm_config does not early-return before the overlay."""
    configs = tmp_path / "perf-tune-report" / "configs"
    configs.mkdir(parents=True)
    (configs / "cost.yaml").write_text(cost_yaml_text)
    camp = tmp_path / "perf-tune-report" / "campaigns" / "mycamp"
    camp.mkdir(parents=True)
    (camp / "config.yaml").write_text(config_text or "tpm:\n  ttft_sla_ms: 2000\n")
    return camp


def test_discover_tpm_config_overlays_fleet_cost_yaml(tmp_path):
    # No per-campaign cost: block -> the fleet perf-tune-report/configs/cost.yaml is overlaid
    # on the defaults (the A2 outcome: published cost_v1 uses fleet rates with no block).
    camp = _campaign_under_cost_yaml(
        tmp_path,
        "usd_per_gpu_hour:\n  GB300: 8.60\n  B200: 6.50\n  default: 8.60\n",
    )
    cfg = discover_tpm_config(camp)
    assert cfg.usd_per_gpu_hour["GB300"] == 8.60   # cost.yaml overrides the 12.00 default
    assert cfg.usd_per_gpu_hour["B200"] == 6.50    # cost.yaml overrides the 8.60 default
    assert cfg.usd_per_gpu_hour["H200"] == 6.31    # default retained (absent from cost.yaml)
    assert "default" not in cfg.usd_per_gpu_hour   # the fallback key is not a hardware
    assert "cost.yaml" in cfg.cost_rate_source


def test_discover_tpm_config_campaign_block_wins_over_cost_yaml(tmp_path):
    # Precedence: campaign cost: block > cost.yaml > defaults.
    camp = _campaign_under_cost_yaml(
        tmp_path,
        "usd_per_gpu_hour:\n  GB300: 8.60\n",
        config_text="cost:\n  usd_per_gpu_hour:\n    GB300: 4.00\n",
    )
    cfg = discover_tpm_config(camp)
    assert cfg.usd_per_gpu_hour["GB300"] == 4.00   # campaign block wins
    assert "campaign config cost: block" in cfg.cost_rate_source


def test_group_carries_mean_isl_osl_and_cache_mode():
    rows = [
        _row(concurrency=1, mean_input_tokens=3200.0, mean_output_tokens=512.0, cache_mode="warm"),
        _row(concurrency=8, output_tps_per_gpu=30.0, mean_input_tokens=3200.0,
             mean_output_tokens=512.0, cache_mode="warm"),
    ]
    s = compute_tpm_summary(rows)
    g = s.groups[0]
    assert g.mean_isl == 3200.0
    assert g.mean_osl == 512.0
    assert g.cache_mode == "warm"


def test_cost_per_1m_tokens_math():
    # 10 tok/s/GPU output, 40 total; $4.50/GPU-hr.
    rows = [_row(output_tps_per_gpu=10.0, total_tps_per_gpu=40.0)]
    s = compute_tpm_summary(rows, usd_per_gpu_hour={"B200": 4.50})
    peak = s.groups[0].peak
    # $/1M out = 4.5*1e6/(10*3600) = 125.0 ; $/1M total = 4.5*1e6/(40*3600) = 31.25
    assert abs(peak.usd_per_1m_output_tokens - 125.0) < 1e-6
    assert abs(peak.usd_per_1m_total_tokens - 31.25) < 1e-6
    assert peak.usd_per_gpu_hour == 4.50


def test_cost_null_when_hardware_not_in_map():
    rows = [_row(hardware="H100", output_tps_per_gpu=10.0)]
    s = compute_tpm_summary(rows, usd_per_gpu_hour={"B200": 4.50})  # no H100 entry
    assert s.groups[0].peak.usd_per_1m_output_tokens is None


def test_cost_key_mismatch_warns(capsys):
    rows = [_row(hardware="B200", output_tps_per_gpu=10.0)]
    # lowercase 'b200' typo matches no atlas hardware ('B200') -> warning.
    compute_tpm_summary(rows, usd_per_gpu_hour={"b200": 4.50})
    err = capsys.readouterr().err
    assert "match no atlas hardware" in err and "b200" in err


def test_cost_key_match_no_warning(capsys):
    rows = [_row(hardware="B200", output_tps_per_gpu=10.0)]
    compute_tpm_summary(rows, usd_per_gpu_hour={"B200": 4.50})
    assert "match no atlas hardware" not in capsys.readouterr().err


def test_point_carries_cell_id():
    rows = [_row(cell_id="cellZ", output_tps_per_gpu=10.0)]
    s = compute_tpm_summary(rows)
    assert s.groups[0].peak.cell_id == "cellZ"


def test_csv_and_json_roundtrip():
    rows = [_row(output_tps_per_gpu=10.0)]
    s = compute_tpm_summary(rows, ttft_sla_ms=500.0, tpot_sla_ms=50.0)
    # json
    d = json.loads(s.to_json())
    assert d["schema_version"] == "tpm_summary_v1"
    assert d["sla_computed"] is True
    assert len(d["groups"]) == 1
    # csv: header + peak(3 bases) + sla(3 bases) = 7 lines (+ trailing newline)
    csv_lines = [ln for ln in s.to_csv().splitlines() if ln.strip()]
    assert csv_lines[0].startswith("model,hardware,quant")
    assert len(csv_lines) == 1 + 3 + 3
