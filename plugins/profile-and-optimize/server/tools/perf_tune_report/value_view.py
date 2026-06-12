"""value_view: render the leadership value-prop ledger.

Joins the curated ``value-findings.yaml`` registry (the human-verified wins +
baselines + lifecycle) with the LIVE perf-lake campaigns -- each campaign's
``report_status.json`` (``sol_rigor`` / ``dcgm_grounded`` / ``focus``) and
``verdict.json`` (``tier`` / ``baseline_named``) -- and renders a grouped
DONE / IN-PROGRESS / NOT-DONE / CLOSED-NEGATIVE markdown table.

The registry win numbers are NOT recomputed here (they are human-verified +
traced to the campaign / audit-table); value_view validates that each backing
campaign EXISTS locally and reports its live rigor, flagging any finding whose
campaign is missing or ungrounded so the ledger never silently drifts from the lake.

It also renders the **GRIND FRONTIER** (the performance ratchet, AGENTS "Always be
grinding"): every finding's ``next_lever`` ranked by ``next_value``, and a flag for any
finding missing one -- so "what do we grind next?" is answered from the ledger, and a
champion is never a finish line (it is the next baseline).

Dependency-light by design: reads only small JSON files (no pyarrow / matplotlib),
so it runs in a minimal install. Added in profile-and-optimize (perf_tune_report value_view verb).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Reuse the fleet leaderboard's $/1M-output-token cost model (single source of truth for
# the $/GPU-hour -> $/1M-tok formula). fleet_leaderboard is dependency-light (schema only),
# so value_view stays minimal-install-safe.
from tools.perf_tune_report import fleet_leaderboard as _fleet
from tools.perf_tune_report import provenance as _prov

# Lifecycle groups, in leadership reading order.
LIFECYCLE_GROUPS: list[tuple[str, str]] = [
    ("done", "A. DONE -- validated, deployable wins"),
    ("in_progress", "B. IN-PROGRESS -- revalidating"),
    ("not_done", "C. NOT-DONE -- research, de-risked"),
    ("closed_negative", "D. CLOSED NEGATIVES -- prevented wasted spend"),
]

_RIGOR_ORDER = {"none": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4}


def _max_rigor(a: str, b: str) -> str:
    return a if _RIGOR_ORDER.get(a, 0) >= _RIGOR_ORDER.get(b, 0) else b


def default_registry_path(campaigns_root: Path) -> Path:
    """The registry lives at ``perf-tune-report/configs/value-findings.yaml`` -- a
    sibling of the ``campaigns/`` dir."""
    return campaigns_root.parent / "configs" / "value-findings.yaml"


def _gentle_resolve(campaign_id: str, campaigns_root: Path) -> Path | None:
    """Resolve a campaign dir WITHOUT raising (unlike helpers.resolve_campaign_dir):
    return None when not found so a missing campaign is flagged, not fatal."""
    direct = campaigns_root / campaign_id
    if direct.is_dir():
        return direct
    for pat in (f"*-{campaign_id}", f"{campaign_id}*", f"*{campaign_id}*"):
        for entry in sorted(campaigns_root.glob(pat)):
            if entry.is_dir():
                return entry
    return None


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _read_campaign_provenance(campaign_dir: Path) -> dict:
    """The cited campaign's code-under-test provenance, dep-light: prefer the flattened
    ``provenance.json`` (JSON, no yaml needed in the minimal install); {} when absent."""
    pj = _read_json(campaign_dir / "provenance.json")
    return pj if isinstance(pj, dict) else {}


def _campaign_status(campaign_dir: Path) -> dict[str, Any]:
    """Read live rigor + verdict directly from the campaign's small JSON files
    (no lake_writer import -> no heavy deps)."""
    rs = _read_json(campaign_dir / "report_status.json")
    vj = _read_json(campaign_dir / "verdict.json")
    return {
        "sol_rigor": rs.get("sol_rigor", "none"),
        "sol_complete": bool(rs.get("sol_complete", False)),
        "dcgm_grounded": bool(rs.get("dcgm_grounded", False)),
        "focus": rs.get("focus"),
        "verdict_tier": str(vj.get("tier", "draft")),
        "baseline_named": bool(vj.get("baseline_named", False)),
        "finding_tag": vj.get("finding"),
    }


def build_value_view(registry: dict, campaigns_root: Path) -> dict[str, Any]:
    """Join the curated registry with live campaign status."""
    out_findings: list[dict[str, Any]] = []
    for f in registry.get("findings", []):
        camps: list[dict[str, Any]] = []
        flags: list[str] = []
        best_rigor = "none"
        any_found = False
        cids = f.get("campaign_ids") or []
        for cid in cids:
            cdir = _gentle_resolve(cid, campaigns_root)
            if cdir is None:
                camps.append({"campaign": cid, "found": False})
                flags.append(f"{cid}: not found locally (S3-only or unpublished)")
                continue
            any_found = True
            st = _campaign_status(cdir)
            st["campaign"] = cid
            st["found"] = True
            camps.append(st)
            best_rigor = _max_rigor(best_rigor, st["sol_rigor"])
            if not st["dcgm_grounded"] and st["sol_rigor"] in ("none", "L1"):
                flags.append(f"{cid}: ungrounded (sol_rigor={st['sol_rigor']})")
            if f.get("finding_baseline_check", True) and not st["baseline_named"] \
                    and f.get("lifecycle") in ("done", "in_progress"):
                flags.append(f"{cid}: baseline_named=false (tag verdict.json)")
            # Code-under-test provenance match (rigor principle p): the finding's
            # source_refs delivery+commit MUST match the cited campaign's provenance --
            # an overlay/offline-prepped campaign cited as an infr-patch benefit is a
            # cross-tier DRAFT defect (even when the kernels match).
            flags.extend(_prov.provenance_match_problems(
                f.get("source_refs"), _read_campaign_provenance(cdir), label=cid,
            ))
        if cids and not any_found:
            flags.append("no backing campaign found locally")
        if f.get("lifecycle") in ("done", "in_progress") and not cids:
            flags.append("no campaign_ids declared")
        # Source-code attribution (durable-lineage): a deployable / in-flight win
        # MUST name the source it ran (branch/commit/delivery or image), so the
        # ledger links each win to deployable code, not just a campaign id.
        if f.get("lifecycle") in ("done", "in_progress") and not f.get("source_refs"):
            flags.append("no source_refs declared (add the vLLM branch/commit/delivery or image)")
        # Performance ratchet (AGENTS "Always be grinding"): every finding MUST name a
        # next_lever so the GRIND FRONTIER is never empty. A champion is the next baseline;
        # a closed negative points at the next lever. Flag a missing one.
        if not str(f.get("next_lever", "") or "").strip():
            flags.append("no next_lever (performance ratchet: name the next lever or 'frontier-exhausted: <evidence>')")
        out_findings.append({
            **f,
            "live": {"best_sol_rigor": best_rigor, "campaigns": camps, "flags": flags},
        })
    return {
        "baseline_context": registry.get("baseline_context", ""),
        "findings": out_findings,
    }


def _source_summary(f: dict) -> str:
    """Compact source-code attribution string for a finding (durable-lineage):
    branch@commit (delivery) for a patch, or image <tag> for a stock-image run."""
    refs = f.get("source_refs") or []
    parts: list[str] = []
    for r in refs[:2]:
        if not isinstance(r, dict):
            continue
        repo = str(r.get("repo", "") or "").split("/")[-1]
        commit = str(r.get("commit", "") or "")[:9]
        if r.get("branch"):
            seg = f"{r['branch']}@{commit}".rstrip("@")
        elif r.get("image"):
            seg = f"image {str(r['image']).split(':')[-1]}"
        elif commit:
            seg = f"{repo}@{commit}"
        else:
            seg = repo or "?"
        if r.get("delivery"):
            seg += f" ({r['delivery']})"
        parts.append(seg)
    return "; ".join(parts) or "(none)"


def _count_by_lifecycle(view: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for f in view["findings"]:
        counts[f.get("lifecycle", "?")] = counts.get(f.get("lifecycle", "?"), 0) + 1
    return counts


def render_markdown(view: dict, *, title: str = "Value ledger",
                    gpu_hr: float = _fleet.GPU_HR_DEFAULT) -> str:
    counts = _count_by_lifecycle(view)
    lines: list[str] = [
        f"# {title} (generated by `perftunereport value_view`)",
        "",
        f"**Baseline:** {view['baseline_context']}",
        "",
        "Summary: " + " | ".join(
            f"{lc}={counts.get(lc, 0)}" for lc, _ in LIFECYCLE_GROUPS
        ),
        "",
        "> Win numbers are human-verified in the registry; the `live sol_rigor / verdict`"
        " + `Flags` columns are read from the perf-lake campaigns at render time.",
        "",
    ]
    by_lc: dict[str, list] = {}
    for f in view["findings"]:
        by_lc.setdefault(f.get("lifecycle", "?"), []).append(f)
    for lc, heading in LIFECYCLE_GROUPS:
        fs = by_lc.get(lc, [])
        if not fs:
            continue
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(
            "| Finding | Win vs baseline | Baseline | HW | live sol_rigor / verdict | Deploy | Source | Backing campaign(s) | Flags |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for f in fs:
            live = f["live"]
            camps = ", ".join(c["campaign"] for c in live["campaigns"]) or "(none)"
            n_ev = len(f.get("evidence_ids") or [])
            if n_ev:
                camps += f" (+{n_ev} supporting)"
            tiers = ", ".join(sorted({
                c.get("verdict_tier", "?") for c in live["campaigns"] if c.get("found")
            })) or "-"
            flags = "; ".join(live["flags"]) or "ok"
            lines.append(
                f"| {f.get('title','')} | {f.get('win','')} | {f.get('baseline','')} | "
                f"{f.get('hardware','')} | {live['best_sol_rigor']} / {tiers} | "
                f"{f.get('deploy_readiness','')} | {_source_summary(f)} | {camps} | {flags} |"
            )
        lines.append("")
    lines.append(render_frontier(view, gpu_hr=gpu_hr))
    return "\n".join(lines)


_NEXT_VALUE_ORDER = {"high": 0, "med": 1, "low": 2}
_LIFECYCLE_RANK = {lc: i for i, (lc, _) in enumerate(LIFECYCLE_GROUPS)}
# Gain tiers whose `speedup` is a real tok/s-or-TPOT multiplier we can rank on (the blended,
# tok/s-led ranking). capacity / enablement / none fall back to the next_value bucket.
_GAIN_RANK_TIERS = {"throughput", "latency"}
_TIER_TAG = {"throughput": "thr", "latency": "lat", "capacity": "cap", "enablement": "enb"}


def _finding_economics(f: dict, gpu_hr: float) -> dict[str, Any]:
    """Quantified gain + dollar economics for a finding's optional ``gain`` block.

    Reuses ``fleet_leaderboard._cost`` ($/1M output tok at $/GPU-hour). All dollar/GPU-hour
    figures are PER 1M OUTPUT TOKENS (volume-independent -- no fleet-size assumption baked
    in). ``$ saved`` + ``GPU-hours saved`` require BOTH a peak and a matched-concurrency
    baseline tok/s/GPU; otherwise only ``$/1M (peak)`` (or nothing) is reported."""
    g = f.get("gain")
    if not isinstance(g, dict):  # a malformed str/list gain must degrade, not crash the ledger
        g = {}
    speedup = g.get("speedup")
    tier = str(g.get("tier", "") or "").lower()
    ranked = isinstance(speedup, (int, float)) and tier in _GAIN_RANK_TIERS
    tps_peak = g.get("tps_gpu_peak")
    base_tps = g.get("baseline_tps_gpu")
    dollars_peak = _fleet._cost(tps_peak, gpu_hr) if isinstance(tps_peak, (int, float)) else None
    dollars_base = _fleet._cost(base_tps, gpu_hr) if isinstance(base_tps, (int, float)) else None
    dollars_saved = (
        dollars_base - dollars_peak
        if (dollars_peak is not None and dollars_base is not None) else None
    )
    gpu_hours_saved = (
        (1.0 / 3600.0) * 1e6 * (1.0 / base_tps - 1.0 / tps_peak)
        if isinstance(tps_peak, (int, float)) and isinstance(base_tps, (int, float))
        and tps_peak and base_tps else None
    )
    if isinstance(speedup, (int, float)):
        gain_label = f"{speedup:g}x {_TIER_TAG.get(tier, tier or 'x')}".strip()
    else:
        gain_label = str(f.get("next_value", "med") or "med").lower()
    return {
        "speedup": speedup if isinstance(speedup, (int, float)) else None,
        "tier": tier or None,
        "ranked_by_speedup": ranked,
        "gain_label": gain_label,
        "dollars_per_1m": dollars_peak,
        "dollars_per_1m_baseline": dollars_base,
        "dollars_saved_per_1m": dollars_saved,
        "gpu_hours_saved_per_1m": gpu_hours_saved,
    }


def frontier_rows(view: dict, *, gpu_hr: float = _fleet.GPU_HR_DEFAULT) -> list[dict[str, Any]]:
    """The GRIND FRONTIER (performance ratchet): every finding's ``next_lever``, ranked by
    blended gain magnitude (throughput tok/s/GPU speedup + latency TPOT speedup) first, then
    by ``next_value`` (high>med>low) then lifecycle for levers without a numeric speedup.
    ``frontier-exhausted: ...`` entries are banked (sorted last), not active levers."""
    rows: list[dict[str, Any]] = []
    for f in view["findings"]:
        nl = str(f.get("next_lever", "") or "").strip()
        if not nl:
            continue
        econ = _finding_economics(f, gpu_hr)
        rows.append({
            "id": f.get("id", ""),
            "title": f.get("title", ""),
            "lifecycle": f.get("lifecycle", "?"),
            "next_lever": nl,
            "next_value": str(f.get("next_value", "med") or "med").lower(),
            "exhausted": nl.lower().startswith("frontier-exhausted"),
            **econ,
        })
    rows.sort(key=lambda r: (
        r["exhausted"],
        0 if r["ranked_by_speedup"] else 1,
        -(r["speedup"] or 0.0) if r["ranked_by_speedup"] else 0.0,
        _NEXT_VALUE_ORDER.get(r["next_value"], 1),
        _LIFECYCLE_RANK.get(r["lifecycle"], 9),
    ))
    return rows


def render_frontier(view: dict, *, gpu_hr: float = _fleet.GPU_HR_DEFAULT) -> str:
    """Render the ranked GRIND FRONTIER section (the always-be-grinding view: what to
    grind next, highest-gain first, with the $/1M-output-token economics). Banked
    frontier-exhausted findings are listed separately so the active queue is never confused
    with dead-ends."""
    rows = frontier_rows(view, gpu_hr=gpu_hr)
    active = [r for r in rows if not r["exhausted"]]
    exhausted = [r for r in rows if r["exhausted"]]
    missing = [f.get("id", "?") for f in view["findings"]
               if not str(f.get("next_lever", "") or "").strip()]
    lines: list[str] = [
        "## GRIND FRONTIER -- ranked next levers (performance ratchet)",
        "",
        "_A champion is the next baseline; every finding names its next lever. Ranked by "
        "blended gain magnitude (throughput tok/s/GPU speedup + latency TPOT speedup), with "
        f"$/1M-output-tok economics at ${gpu_hr:g}/GPU-hour (per 1M tokens, volume-independent). "
        "AGENTS 'Always be grinding'._",
        "",
    ]
    if not active:
        lines.append("> WARNING: no ACTIVE next levers queued -- the frontier looks exhausted. "
                     "Verify every finding's `next_lever` (the ratchet forbids a stall).")
        lines.append("")
    else:
        lines.append("| # | gain | $/1M out (peak) | $ saved/1M | GPU-hrs saved/1M | Next lever | From finding | Lifecycle |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(active, 1):
            dpm = f"${r['dollars_per_1m']:.2f}" if r["dollars_per_1m"] is not None else "-"
            dsv = f"${r['dollars_saved_per_1m']:.2f}" if r["dollars_saved_per_1m"] is not None else "-"
            ghs = f"{r['gpu_hours_saved_per_1m']:.4f}" if r["gpu_hours_saved_per_1m"] is not None else "-"
            lines.append(
                f"| {i} | {r['gain_label']} | {dpm} | {dsv} | {ghs} | "
                f"{r['next_lever']} | {r['title']} | {r['lifecycle']} |"
            )
        lines.append("")
    if missing:
        lines.append(f"> RATCHET FLAG: {len(missing)} finding(s) missing `next_lever`: "
                     + ", ".join(missing) + " -- name the next lever.")
        lines.append("")
    if exhausted:
        lines.append(f"_Banked / frontier-exhausted ({len(exhausted)}): "
                     + ", ".join(r["id"] for r in exhausted) + "._")
        lines.append("")
    return "\n".join(lines)


_GROUP_LABEL = {
    "done": "DONE -- deploy now",
    "in_progress": "IN-PROGRESS -- revalidating",
    "not_done": "NOT-DONE -- research / pursue",
    "closed_negative": "CLOSED -- banked negatives (prevented wasted spend)",
}


def render_report(view: dict, *, title: str = "Inference perf wins -- value prop",
                  gpu_hr: float = _fleet.GPU_HR_DEFAULT) -> str:
    """Compact, copy-paste-ready markdown for a report or Slack: a one-line summary
    + grouped bullets (one per finding), instead of the wide audit table."""
    counts = _count_by_lifecycle(view)
    summary = ", ".join(
        f"{_GROUP_LABEL[lc].split(' -- ')[0]} {counts.get(lc, 0)}"
        for lc, _ in LIFECYCLE_GROUPS
    )
    lines: list[str] = [
        f"# {title}",
        f"_Baseline: {view['baseline_context']}_",
        "",
        f"**Summary:** {summary} (total {len(view['findings'])}).",
        "",
    ]
    by_lc: dict[str, list] = {}
    for f in view["findings"]:
        by_lc.setdefault(f.get("lifecycle", "?"), []).append(f)
    for lc, _ in LIFECYCLE_GROUPS:
        fs = by_lc.get(lc, [])
        if not fs:
            continue
        lines.append(f"**{_GROUP_LABEL[lc]} ({len(fs)})**")
        for f in fs:
            live = f["live"]
            meta = [b for b in (
                f.get("baseline"),
                f.get("hardware"),
                f"sol_rigor {live['best_sol_rigor']}",
                f.get("deploy_readiness"),
                (f"src {_source_summary(f)}" if f.get("source_refs") else None),
                (f"+{len(f['evidence_ids'])} supporting runs" if f.get("evidence_ids") else None),
            ) if b]
            lines.append(
                f"- **{f.get('title', '')}** -- {f.get('win', '')}  _({'; '.join(meta)})_"
            )
        lines.append("")
    lines.append(render_frontier(view, gpu_hr=gpu_hr))
    flags = sorted({fl for f in view["findings"] for fl in f["live"]["flags"]})
    if flags:
        shown = "; ".join(flags[:6]) + ("; ..." if len(flags) > 6 else "")
        lines.append(f"_Data caveats ({len(flags)}): {shown}_")
    lines.append(
        "_Generated by `perftunereport value_view --format report`; every win traces to a named "
        "perf-lake campaign (see value-findings.yaml)._"
    )
    return "\n".join(lines)
