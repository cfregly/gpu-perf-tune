"""Unit tests for the perf_tune_report value_view verb (leadership value ledger)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.perf_tune_report.perf_tune_report_cli import main
from tools.perf_tune_report.value_view import (
    build_value_view,
    default_registry_path,
    frontier_rows,
    render_frontier,
    render_markdown,
    render_report,
)


def _mk_campaign(campaigns: Path, cid: str, *, sol_rigor: str = "L3",
                 dcgm: bool = True, tier: str = "verdict",
                 baseline_named: bool = True) -> Path:
    d = campaigns / cid
    d.mkdir(parents=True)
    (d / "report_status.json").write_text(json.dumps({
        "sol_rigor": sol_rigor, "sol_complete": True,
        "dcgm_grounded": dcgm, "focus": "mixed",
    }))
    (d / "verdict.json").write_text(json.dumps({
        "tier": tier, "trials": 3, "same_node": True,
        "baseline_named": baseline_named,
    }))
    return d


def _registry() -> dict:
    return {
        "baseline_context": "vLLM + FlashInfer-TRTLLM",
        "findings": [
            {"id": "win1", "title": "Win One", "lifecycle": "done", "baseline": "fp8",
             "win": "1.3x", "hardware": "GB300", "deploy_readiness": "deployed",
             "campaign_ids": ["camp-good"], "next_lever": "extend to fleet", "next_value": "high",
             "source_refs": [{"repo": "example/vllm", "branch": "feature/x",
                              "commit": "abc1234", "delivery": "overlay"}]},
            {"id": "wip1", "title": "WIP One", "lifecycle": "in_progress", "baseline": "ctrl",
             "win": "tbd", "hardware": "GB300", "deploy_readiness": "fork-local",
             "campaign_ids": ["camp-missing"], "next_lever": "finish revalidation", "next_value": "med",
             "source_refs": [{"repo": "vllm-project/vllm", "delivery": "image",
                              "image": "infr/vllm:v2.12.3"}]},
            {"id": "neg1", "title": "Neg One", "lifecycle": "closed_negative", "baseline": "FI",
             "win": "slower", "hardware": "GB300", "deploy_readiness": "n/a",
             "campaign_ids": [], "next_lever": "frontier-exhausted: H1 cannot beat H4", "next_value": "low"},
        ],
    }


def test_build_value_view_joins_live_status(tmp_path):
    campaigns = tmp_path / "campaigns"
    campaigns.mkdir()
    _mk_campaign(campaigns, "camp-good", sol_rigor="L3", dcgm=True)
    view = build_value_view(_registry(), campaigns)
    assert len(view["findings"]) == 3
    by_id = {f["id"]: f for f in view["findings"]}
    # found + grounded + baseline_named -> no flags
    assert by_id["win1"]["live"]["best_sol_rigor"] == "L3"
    assert by_id["win1"]["live"]["flags"] == []
    # missing campaign -> flagged not-found
    assert any("not found locally" in fl for fl in by_id["wip1"]["live"]["flags"])
    # closed negative with no campaigns -> not flagged (by design)
    assert by_id["neg1"]["live"]["flags"] == []


def test_ungrounded_and_unnamed_flags(tmp_path):
    campaigns = tmp_path / "campaigns"
    campaigns.mkdir()
    _mk_campaign(campaigns, "camp-weak", sol_rigor="none", dcgm=False, baseline_named=False)
    reg = {"baseline_context": "x", "findings": [
        {"id": "w", "title": "W", "lifecycle": "done", "baseline": "b", "win": "1x",
         "hardware": "GB300", "deploy_readiness": "deployed", "campaign_ids": ["camp-weak"],
         "next_lever": "next", "next_value": "med", "source_refs": [{"repo": "x/vllm", "delivery": "overlay"}]},
    ]}
    view = build_value_view(reg, campaigns)
    flags = view["findings"][0]["live"]["flags"]
    assert any("ungrounded" in f for f in flags)
    assert any("baseline_named=false" in f for f in flags)


def test_render_markdown_and_report_group(tmp_path):
    campaigns = tmp_path / "campaigns"
    campaigns.mkdir()
    _mk_campaign(campaigns, "camp-good")
    view = build_value_view(_registry(), campaigns)
    md = render_markdown(view)
    assert "A. DONE" in md and "Win One" in md
    # source-code attribution column is rendered (durable-lineage)
    assert "Source" in md and "feature/x@abc1234" in md
    rep = render_report(view)
    assert "Summary:" in rep and "**Win One**" in rep
    assert "DONE" in rep and "CLOSED" in rep


def test_missing_source_refs_flagged(tmp_path):
    """A done/in_progress finding without source_refs is flagged (link wins to code)."""
    campaigns = tmp_path / "campaigns"
    campaigns.mkdir()
    _mk_campaign(campaigns, "camp-good")
    reg = {"baseline_context": "x", "findings": [
        {"id": "w", "title": "W", "lifecycle": "done", "baseline": "b", "win": "1x",
         "hardware": "GB300", "deploy_readiness": "deployed", "campaign_ids": ["camp-good"]},
    ]}
    view = build_value_view(reg, campaigns)
    assert any("no source_refs" in f for f in view["findings"][0]["live"]["flags"])


def test_default_registry_path(tmp_path):
    campaigns = tmp_path / "perf-tune-report" / "campaigns"
    campaigns.mkdir(parents=True)
    p = default_registry_path(campaigns)
    assert p.name == "value-findings.yaml"
    assert p.parent.name == "configs"


def test_missing_next_lever_flagged(tmp_path):
    """Performance ratchet: a finding with no next_lever is flagged."""
    campaigns = tmp_path / "campaigns"
    campaigns.mkdir()
    _mk_campaign(campaigns, "camp-good")
    reg = {"baseline_context": "x", "findings": [
        {"id": "w", "title": "W", "lifecycle": "closed_negative", "baseline": "b", "win": "1x",
         "hardware": "GB300", "deploy_readiness": "n/a", "campaign_ids": []},
    ]}
    view = build_value_view(reg, campaigns)
    assert any("no next_lever" in f for f in view["findings"][0]["live"]["flags"])


def test_frontier_ranks_and_banks(tmp_path):
    """GRIND FRONTIER: active levers ranked high>med, frontier-exhausted banked + excluded."""
    campaigns = tmp_path / "campaigns"
    campaigns.mkdir()
    _mk_campaign(campaigns, "camp-good")
    view = build_value_view(_registry(), campaigns)
    rows = frontier_rows(view)
    active = [r for r in rows if not r["exhausted"]]
    # win1 (high) ranks before wip1 (med); neg1 is frontier-exhausted -> banked, not active.
    assert [r["id"] for r in active] == ["win1", "wip1"]
    assert any(r["exhausted"] and r["id"] == "neg1" for r in rows)
    fr = render_frontier(view)
    assert "GRIND FRONTIER" in fr
    assert fr.index("extend to fleet") < fr.index("finish revalidation")  # high before med
    assert "frontier-exhausted" in fr.lower() or "neg1" in fr
    # the frontier rides along in the full markdown render
    assert "GRIND FRONTIER" in render_markdown(view)


def test_cli_value_view_report(tmp_path, capsys):
    campaigns = tmp_path / "campaigns"
    campaigns.mkdir()
    _mk_campaign(campaigns, "camp-good")
    reg = tmp_path / "value-findings.yaml"
    reg.write_text(yaml.safe_dump(_registry()))
    rc = main(["value_view", "--registry", str(reg),
               "--campaigns-dir", str(campaigns), "--format", "report"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "value prop" in out and "**Win One**" in out


def test_cli_value_view_resolves_gpu_hr_from_cost_yaml(tmp_path, capsys):
    # configs/cost.yaml sits next to campaigns/ (campaigns_root.parent / "configs"); the
    # GRIND FRONTIER economics resolve the GB300 rate from it (override absent).
    campaigns = tmp_path / "campaigns"
    campaigns.mkdir()
    _mk_campaign(campaigns, "camp-good")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "cost.yaml").write_text(
        "usd_per_gpu_hour:\n  GB300: 7.77\n  default: 7.77\n"
    )
    reg = tmp_path / "value-findings.yaml"
    reg.write_text(yaml.safe_dump(_registry()))
    rc = main(["value_view", "--registry", str(reg),
               "--campaigns-dir", str(campaigns), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["gpu_hr"] == 7.77  # from cost.yaml, not the 8.60 GB300 default


def test_cli_value_view_gpu_hr_override_wins_over_cost_yaml(tmp_path, capsys):
    campaigns = tmp_path / "campaigns"
    campaigns.mkdir()
    _mk_campaign(campaigns, "camp-good")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "cost.yaml").write_text("usd_per_gpu_hour:\n  GB300: 7.77\n")
    reg = tmp_path / "value-findings.yaml"
    reg.write_text(yaml.safe_dump(_registry()))
    rc = main(["value_view", "--registry", str(reg), "--campaigns-dir", str(campaigns),
               "--gpu-hr", "12.34", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["gpu_hr"] == 12.34  # explicit override wins


def _gain_registry() -> dict:
    """Registry exercising the blended gain ranking: a throughput speedup with tok/s/GPU
    (-> $ columns), a latency speedup without tok/s/GPU, and a high-next_value lever w/ no gain."""
    src = [{"repo": "x/vllm", "delivery": "overlay"}]
    return {"baseline_context": "x", "findings": [
        {"id": "thr_big", "title": "Thr Big", "lifecycle": "done", "baseline": "b", "win": "2.3x",
         "hardware": "GB300", "deploy_readiness": "deployed", "campaign_ids": [],
         "next_lever": "roll to fleet", "next_value": "med", "source_refs": src,
         "gain": {"speedup": 2.3, "tier": "throughput", "tps_gpu_peak": 481, "baseline_tps_gpu": 369}},
        {"id": "lat_mid", "title": "Lat Mid", "lifecycle": "done", "baseline": "b", "win": "2.1x",
         "hardware": "GB300", "deploy_readiness": "deployed", "campaign_ids": [],
         "next_lever": "port to fleet", "next_value": "low", "source_refs": src,
         "gain": {"speedup": 2.1, "tier": "latency"}},
        {"id": "no_gain_high", "title": "No Gain High", "lifecycle": "in_progress", "baseline": "b",
         "win": "tbd", "hardware": "GB300", "deploy_readiness": "fork-local", "campaign_ids": [],
         "next_lever": "infra fix", "next_value": "high", "source_refs": src},
    ]}


def test_frontier_ranks_by_gain_then_next_value(tmp_path):
    campaigns = tmp_path / "campaigns"
    campaigns.mkdir()
    view = build_value_view(_gain_registry(), campaigns)
    rows = frontier_rows(view)
    # speedup-ranked levers first (2.3 thr > 2.1 lat), THEN the no-gain high-next_value lever
    assert [r["id"] for r in rows] == ["thr_big", "lat_mid", "no_gain_high"]
    thr = rows[0]
    assert thr["dollars_per_1m"] is not None
    assert thr["dollars_saved_per_1m"] is not None
    assert thr["gpu_hours_saved_per_1m"] is not None
    # latency lever has no tok/s/GPU -> no dollar columns, but ranked on its speedup label
    assert rows[1]["dollars_per_1m"] is None
    assert "2.1x lat" in rows[1]["gain_label"]
    # no-gain lever falls back to the next_value bucket label
    assert rows[2]["gain_label"] == "high"


def test_frontier_dollar_columns_render(tmp_path):
    campaigns = tmp_path / "campaigns"
    campaigns.mkdir()
    view = build_value_view(_gain_registry(), campaigns)
    fr = render_frontier(view, gpu_hr=8.60)
    assert "$/1M out (peak)" in fr and "GPU-hrs saved/1M" in fr
    assert "2.3x thr" in fr
    # $/1M at 481 tok/s/GPU @ $8.60/GPU-hr ~= $4.97
    assert "$4.9" in fr


def test_evidence_ids_render_supporting_count_no_flags(tmp_path):
    """evidence_ids render as a '+N supporting' note and add NO flags (only campaign_ids flag)."""
    campaigns = tmp_path / "campaigns"
    campaigns.mkdir()
    _mk_campaign(campaigns, "camp-good")
    reg = {"baseline_context": "x", "findings": [
        {"id": "w", "title": "W", "lifecycle": "done", "baseline": "b", "win": "1x",
         "hardware": "GB300", "deploy_readiness": "deployed", "campaign_ids": ["camp-good"],
         "next_lever": "next", "next_value": "med",
         "source_refs": [{"repo": "x/vllm", "delivery": "overlay"}],
         "evidence_ids": ["e1-20260101T000000Z", "e2-20260101T000001Z"]},
    ]}
    view = build_value_view(reg, campaigns)
    # evidence_ids add NO flags (the clean-ledger design goal: only campaign_ids carry flags)
    assert view["findings"][0]["live"]["flags"] == []
    assert "+2 supporting" in render_markdown(view)
    assert "2 supporting runs" in render_report(view)
