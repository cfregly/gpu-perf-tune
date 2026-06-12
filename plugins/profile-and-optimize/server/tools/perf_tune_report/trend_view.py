"""trend_view: longitudinal (model, variant) perf/quality trend across campaigns.

The fleet/value/champion views are SNAPSHOTS (latest-wins). This verb answers the
"did this regress / improve over time + across engine versions" question: it groups
atlas rows by ``(model, variant_key, concurrency)`` -- the stable serving-variant hash from
``capture_signature`` (the same key the lake's ``atlas_v1.variant_key`` carries) -- orders
each group by ``captured_at``, and reports the first->last delta + a regression flag, with the
serving ``image`` per point (the engine-version axis).

Reads LOCAL campaigns by default (dependency-light, like ``fleet_leaderboard``); the identical
grouping applies to a published-lake parquet pull (lake rows carry ``variant_key`` +
``captured_at`` + ``image`` directly), so this is the local-first longitudinal view.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tools.perf_tune_report.capture_signature import variant_key_for
from tools.perf_tune_report.fleet_leaderboard import canon_model
from tools.perf_tune_report.schema import AtlasCell

# Metrics where LOWER is better (a regression is an INCREASE). Everything else (tok/s) is
# higher-better (a regression is a DECREASE).
_LOWER_BETTER_HINTS = ("tpot", "ttft", "itl", "ms")


def _is_lower_better(metric: str) -> bool:
    m = metric.lower()
    return any(h in m for h in _LOWER_BETTER_HINTS)


def _variant_key(row: AtlasCell) -> str:
    # Prefer the row's lake variant_key (image-independent); else compute it. Both EXCLUDE
    # image so the same config trends across engine-image versions (the image is the axis).
    vk = getattr(row, "variant_key", "") or ""
    if vk:
        return vk
    try:
        return variant_key_for(row)
    except Exception:  # noqa: BLE001 - a row we can't key is skipped from trends
        return ""


def read_lake_rows(lake_dir: Path, *, hardware_filter: str | None = None) -> list[Any]:
    """Read published ``atlas_v1`` parquet from a pulled lake snapshot, joining
    ``campaign_v1.vllm_commit`` as the engine-version axis (so the trend shows the engine
    COMMIT, not just the image tag). pyarrow is imported lazily -- only this path needs it.

    Returns AtlasCell-lite rows (``SimpleNamespace`` carrying every parquet column) that
    duck-type into ``build_trends`` (it only reads model / variant_key / concurrency /
    captured_at / image / the metric). Lake rows always carry ``variant_key`` + ``captured_at``,
    so no signature recompute is needed. ``lake_dir`` may be the snapshot root or any ancestor
    of the ``atlas_v1/`` + ``campaign_v1/`` partition dirs (matched by path substring)."""
    import pyarrow.parquet as pq  # type: ignore[import-not-found]  # lazy: only the lake path

    atlas_files: list[Path] = []
    campaign_files: list[Path] = []
    for p in sorted(lake_dir.rglob("*.parquet")):
        s = str(p)
        if "atlas_v1" in s:
            atlas_files.append(p)
        elif "campaign_v1" in s:
            campaign_files.append(p)

    # campaign_id -> vllm_commit (the engine-version axis joined onto each atlas row).
    commit_by_campaign: dict[str, str] = {}
    for cf in campaign_files:
        for rec in pq.read_table(cf).to_pylist():
            cid, commit = rec.get("campaign_id"), (rec.get("vllm_commit") or "")
            if cid and commit:
                commit_by_campaign[cid] = commit

    rows: list[Any] = []
    for af in atlas_files:
        for rec in pq.read_table(af).to_pylist():
            if hardware_filter and rec.get("hardware") != hardware_filter:
                continue
            rec = dict(rec)
            commit = commit_by_campaign.get(rec.get("campaign_id"), "")
            if commit:  # prefer the joined commit for the engine-version axis
                rec["image"] = commit
            rows.append(SimpleNamespace(**rec))
    return rows


def build_trends(
    rows: list[AtlasCell],
    *,
    metric: str = "output_tps_per_gpu",
    concurrency: int | None = None,
    regression_pct: float = 10.0,
) -> dict[str, Any]:
    """Group rows into per-(model, variant, concurrency) time-series and flag regressions.

    A trend needs >= 2 distinct captured_at points to compute a delta; single-point variants
    are reported with their one point (delta None). ``concurrency`` filters to one c (else every
    c is its own trend line). ``regression_pct`` is the |delta| that flags a regression in the
    wrong direction for the metric."""
    lower_better = _is_lower_better(metric)
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for r in rows:
        val = getattr(r, metric, None)
        if not isinstance(val, (int, float)):
            continue
        if concurrency is not None and r.concurrency != concurrency:
            continue
        vk = _variant_key(r)
        if not vk:
            continue
        key = (canon_model(r.model), vk, int(r.concurrency))
        groups.setdefault(key, []).append({
            "captured_at": r.captured_at or "",
            "value": float(val),
            "image": getattr(r, "image", "") or "",
        })

    trends: list[dict[str, Any]] = []
    for (model, vk, conc), pts in groups.items():
        pts.sort(key=lambda p: p["captured_at"])
        first, last = pts[0], pts[-1]
        delta_pct: float | None = None
        regression = False
        if len(pts) >= 2 and first["value"]:
            delta_pct = round((last["value"] / first["value"] - 1) * 100, 1)
            regression = (delta_pct >= regression_pct) if lower_better else (delta_pct <= -regression_pct)
        trends.append({
            "model": model,
            "variant_key": vk[:12],
            "concurrency": conc,
            "n_points": len(pts),
            "first_value": first["value"],
            "last_value": last["value"],
            "delta_pct": delta_pct,
            "regression": regression,
            "first_captured_at": first["captured_at"],
            "last_captured_at": last["captured_at"],
            "images": sorted({p["image"] for p in pts if p["image"]}),
            "points": pts,
        })
    # Regressions first, then most-tracked (most points) first, then model.
    trends.sort(key=lambda t: (not t["regression"], -t["n_points"], t["model"]))
    return {
        "metric": metric,
        "lower_better": lower_better,
        "concurrency": concurrency,
        "regression_pct": regression_pct,
        "n_trends": len(trends),
        "n_regressions": sum(1 for t in trends if t["regression"]),
        "trends": trends,
    }


def render_markdown(view: dict[str, Any], *, title: str = "Perf/quality trend over time") -> str:
    metric = view["metric"]
    direction = "lower is better" if view["lower_better"] else "higher is better"
    lines = [
        f"# {title} (generated by `perftunereport trend_view`)",
        "",
        f"Metric: **{metric}** ({direction}). Grouped by `(model, variant_key, concurrency)` -- the "
        "stable `capture_signature` serving-variant hash -- ordered by `captured_at`. A regression "
        f"is a >= {view['regression_pct']:g}% move in the wrong direction across the tracked window. "
        "`images` is the engine-version axis. Reads local campaigns (same shape as the lake's "
        "`atlas_v1.variant_key` + `captured_at` for a published-lake pull).",
        "",
        f"Trends: **{view['n_trends']}** | regressions: **{view['n_regressions']}**.",
        "",
        "| | model | variant | c | n | first | last | delta% | window | images |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for t in view["trends"]:
        flag = "REGRESSION" if t["regression"] else ""
        dpct = f"{t['delta_pct']:+.1f}%" if t["delta_pct"] is not None else "-"
        window = (
            f"{t['first_captured_at'][:10]}..{t['last_captured_at'][:10]}"
            if t["n_points"] >= 2 else (t["last_captured_at"][:10] or "-")
        )
        imgs = ", ".join(i.split(":")[-1] for i in t["images"]) or "-"
        lines.append(
            f"| {flag} | {t['model']} | `{t['variant_key']}` | {t['concurrency']} | {t['n_points']} "
            f"| {t['first_value']:.1f} | {t['last_value']:.1f} | {dpct} | {window} | {imgs} |"
        )
    lines.append("")
    return "\n".join(lines)
