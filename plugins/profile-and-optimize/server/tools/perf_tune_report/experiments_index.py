"""Cross-experiment index: enumerate all perf-report campaigns into one table.

This is the analyst entry point the workspace lacked. Each campaign PDF is
standalone; this verb walks the campaigns dir and emits ONE row per experiment
keyed by the experiment-id join key (AGENTS.md "Experiment Isolation &
Traceability"), so a human can answer "what experiments exist and how did they
do" -- e.g. filter ``family == "nvfp4-kv"`` and read peak tok/s/GPU vs the fp8
baseline -- without grepping prose across 65+ cluster-probes.

Reads only local campaign artifacts (SOURCE.md join keys + report_status.json
focus/sol_rigor + verdict.json tier + atlas.jsonl headline metrics). No pyarrow,
no network. Optional best-effort S3 enumeration marks which campaigns reached the
lake. Outputs a tracked ``experiments-index.jsonl`` + ``EXPERIMENTS-INDEX.md``
in the perf-report bundle (data, not Python -- per perf-tune-report/AGENTS.md).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from tools.perf_tune_report.lake_writer import (
    parse_campaign_utc,
    parse_source_md,
    read_render_status,
    read_verdict,
)
from tools.perf_tune_report.schema import read_jsonl


def enumerate_published_campaign_ids(
    cfg: Any,
    *,
    bucket: str,
    s3_client_factory: Callable[[Any], Any] | None = None,
) -> set[str]:
    """List campaign ids already published to the lake's ``campaign_v1`` prefix.

    Paginates ``list_objects_v2`` over ``S3_PREFIX/campaign_v1/`` and parses the
    Hive ``campaign=<id>`` token from each key. Isolated + factory-injectable so
    tests stub the S3 client (mirrors ``upload_to_s3``)."""
    from tools.perf_tune_report.lake_writer import (
        S3_PREFIX,
        CAMPAIGN_TABLE_NAME,
        _make_s3_client,
    )

    factory = s3_client_factory or _make_s3_client
    client = factory(cfg)
    prefix = f"{S3_PREFIX}/{CAMPAIGN_TABLE_NAME}/"
    ids: set[str] = set()
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            m = re.search(r"campaign=([^/]+)/", obj.get("Key", ""))
            if m:
                ids.add(m.group(1))
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return ids


def _safe_min(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return min(vals) if vals else None


def _safe_max(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def _is_campaign_dir(d: Path) -> bool:
    if not d.is_dir():
        return False
    return any((d / f).exists() for f in ("atlas.jsonl", "config.yaml", "SOURCE.md"))


def build_index_row(campaign_dir: Path, published: set[str] | None = None) -> dict[str, Any]:
    """One index row for a single campaign directory.

    ``published`` is the set of campaign ids confirmed in the lake (from
    ``enumerate_published_campaign_ids``); when None (``--include-s3`` off or
    creds missing) ``published_to_lake`` defaults to False = "not confirmed"."""
    campaign_id = campaign_dir.name
    meta = parse_source_md(campaign_dir / "SOURCE.md")
    status = read_render_status(campaign_dir)
    verdict = read_verdict(campaign_dir)

    atlas_path = campaign_dir / "atlas.jsonl"
    rows = read_jsonl(atlas_path) if atlas_path.is_file() else []
    models = sorted({r.model for r in rows if r.model})
    hardware = sorted({r.hardware for r in rows if r.hardware})
    quant = sorted({r.quant for r in rows if r.quant})

    try:
        captured_at = parse_campaign_utc(campaign_id).isoformat()
    except ValueError:
        captured_at = ""

    return {
        "campaign_id": campaign_id,
        # Default experiment_id to campaign_id so the join key is never empty.
        "experiment_id": meta.get("experiment_id") or campaign_id,
        "family": meta.get("family", ""),
        "captured_at_utc": captured_at,
        "focus": status.focus,
        "sol_rigor": status.sol_rigor,
        "sol_complete": status.sol_complete,
        "dcgm_grounded": status.dcgm_grounded,
        "rendered": status.rendered,
        "verdict_tier": verdict.tier,
        "models": ",".join(models),
        "hardware": ",".join(hardware),
        "quant": ",".join(quant),
        "peak_output_tps_per_gpu": _safe_max([r.output_tps_per_gpu for r in rows]),
        "min_ttft_ms": _safe_min([r.ttft_avg_ms for r in rows]),
        "min_tpot_median_ms": _safe_min([r.tpot_median_ms for r in rows]),
        "cell_count": len({r.cell_id for r in rows}),
        "atlas_rows": len(rows),
        "evidence_bundle_path": meta.get("evidence_bundle_path", ""),
        "published_to_lake": (campaign_id in published) if published is not None else False,
    }


def build_index(campaigns_root: Path, published: set[str] | None = None) -> list[dict[str, Any]]:
    """Enumerate every local campaign into index rows, sorted by captured_at."""
    rows = [
        build_index_row(d, published)
        for d in sorted(campaigns_root.iterdir())
        if _is_campaign_dir(d)
    ]
    rows.sort(key=lambda r: (r["captured_at_utc"], r["campaign_id"]))
    return rows


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.1f}"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


def render_index_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Experiments index",
        "",
        f"- Campaigns: **{len(rows)}**",
        "- One row per experiment (join key = `experiment_id`). Generated by "
        "`perftunereport experiments_index`; do not hand-edit.",
        "",
        "| experiment_id | family | focus | sol_rigor | verdict | lake | model | hw | "
        "peak tok/s/GPU | min TTFT ms | min TPOT ms | bundle |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        bundle = r["evidence_bundle_path"]
        bundle_disp = "`" + bundle.split("/the external workspace")[-1] + "`" if bundle else "-"
        lines.append(
            f"| `{r['experiment_id']}` | {r['family'] or '-'} | {r['focus']} "
            f"| {r['sol_rigor']} | {r['verdict_tier']} | {_fmt(r.get('published_to_lake', False))} "
            f"| {r['models'] or '-'} "
            f"| {r['hardware'] or '-'} | {_fmt(r['peak_output_tps_per_gpu'])} "
            f"| {_fmt(r['min_ttft_ms'])} | {_fmt(r['min_tpot_median_ms'])} | {bundle_disp} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_index(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "experiments-index.jsonl"
    md_path = out_dir / "EXPERIMENTS-INDEX.md"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    md_path.write_text(render_index_md(rows), encoding="utf-8")
    return {"jsonl": str(jsonl_path), "md": str(md_path)}


# --------------------------------------------------------------------------- #
# experiment_inventory: the canonical "how many experiments" count.
#
# ``experiments_index`` enumerates only the LOCAL campaigns dir. The full
# experiment universe also lives in the run-id-stamped evidence bundles under
# each deploy bundle's ``experiments/artifacts/**`` + ``cluster-probes/*`` (the
# 185/338 the coverage audit walks). This verb UNIFIES the two into ONE headline
# count keyed by the run-id join key (AGENTS.md "Experiment Isolation &
# Traceability": campaign_id == bundle run-id), ending the campaigns-vs-bundles
# ambiguity. profile_and_optimize stays workspace-agnostic: bundle roots are passed in via
# ``--bundle-root`` (repeatable); with none given the count is campaigns-only.
# --------------------------------------------------------------------------- #

#: A directory is an "experiment" iff its name carries a UTC run-id stamp. Same
#: pattern + bundle-discovery logic as ``perf-lake-coverage-audit.py`` so the
#: inventory count matches the audit universe.
RUNID_STAMP = re.compile(r"\d{8}T\d{6}Z")


def find_bundle_dirs(bundle_roots: list[Path]) -> list[Path]:
    """Run-id-stamped evidence-bundle dirs (carrying SOURCE.md or summary.md) under the
    given roots, deduped by resolved path. Mirrors the coverage audit's discovery."""
    seen: set[Path] = set()
    out: list[Path] = []
    for root in bundle_roots:
        if not root.is_dir():
            continue
        for marker in ("SOURCE.md", "summary.md"):
            for f in root.rglob(marker):
                d = f.parent.resolve()
                if RUNID_STAMP.search(d.name) and d not in seen:
                    seen.add(d)
                    out.append(d)
    return out


def build_inventory(
    campaigns_root: Path,
    bundle_roots: list[Path] | None = None,
    published: set[str] | None = None,
) -> dict[str, Any]:
    """Unify local campaigns + run-id-stamped evidence bundles into one headline count.

    The run-id (campaign dir name == bundle dir name) is the join key, so a campaign and
    its source bundle dedupe to ONE experiment; ``total_experiments`` is the union."""
    campaigns = build_index(campaigns_root, published)
    campaign_ids = {r["campaign_id"] for r in campaigns}
    bundles = find_bundle_dirs(bundle_roots or [])
    bundle_ids = {b.name for b in bundles}
    all_ids = campaign_ids | bundle_ids
    bundle_only = sorted(bundle_ids - campaign_ids)

    by_family: dict[str, int] = {}
    by_model: dict[str, int] = {}
    for r in campaigns:
        fam = r.get("family") or "(unfamilied)"
        by_family[fam] = by_family.get(fam, 0) + 1
        for m in (r.get("models") or "").split(","):
            m = m.strip()
            if m:
                by_model[m] = by_model.get(m, 0) + 1

    return {
        "total_experiments": len(all_ids),
        "campaign_count": len(campaigns),
        "bundle_count": len(bundles),
        "bundle_only_count": len(bundle_only),
        "published_in_lake": (len(published) if published is not None else None),
        "published_local_campaigns": (
            sum(1 for r in campaigns if r.get("published_to_lake"))
            if published is not None else None
        ),
        "by_family": dict(sorted(by_family.items(), key=lambda kv: (-kv[1], kv[0]))),
        "by_model": dict(sorted(by_model.items(), key=lambda kv: (-kv[1], kv[0]))),
        "bundle_only": bundle_only,
        "campaigns": campaigns,
    }


def render_inventory_md(inv: dict[str, Any]) -> str:
    lines = [
        "# Experiment inventory",
        "",
        f"- **Total distinct experiments: {inv['total_experiments']}** "
        "(union of local perf-report campaigns + run-id-stamped evidence bundles, "
        "deduped by run-id).",
        f"- Local perf-report campaigns: **{inv['campaign_count']}**",
        f"- Evidence bundles (run-id-stamped, SOURCE.md/summary.md): **{inv['bundle_count']}**",
        f"- Bundles with no local campaign (bundle-only): **{inv['bundle_only_count']}**",
    ]
    if inv["published_in_lake"] is not None:
        lines.append(f"- Published in the lake (campaign_v1): **{inv['published_in_lake']}**")
    lines += [
        "",
        "Generated by `perftunereport experiment_inventory`; do not hand-edit. The run-id is the "
        "join key across evidence bundle, cluster `experiment=<id>` label, and perf-lake "
        "campaign (AGENTS.md \"Experiment Isolation & Traceability\"). Pass `--bundle-root` "
        "(repeatable) to include the deploy-bundle trees; with none given the count is "
        "campaigns-only.",
        "",
        "## By family (local campaigns)",
        "",
        "| family | campaigns |",
        "| --- | ---: |",
    ]
    for fam, n in inv["by_family"].items():
        lines.append(f"| {fam} | {n} |")
    lines += [
        "",
        "## By model (local campaigns with atlas)",
        "",
        "| model | campaigns |",
        "| --- | ---: |",
    ]
    for m, n in inv["by_model"].items():
        lines.append(f"| {m} | {n} |")
    lines.append("")
    return "\n".join(lines)


def write_inventory(inv: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "EXPERIMENT-INVENTORY.md"
    json_path = out_dir / "experiment-inventory.json"
    md_path.write_text(render_inventory_md(inv), encoding="utf-8")
    # Keep the JSON headline light: drop the full per-campaign list (that is
    # experiments-index.jsonl's job).
    summary = {k: v for k, v in inv.items() if k != "campaigns"}
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"md": str(md_path), "json": str(json_path)}
