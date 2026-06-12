"""Cross-model fleet leaderboards from the local campaigns dir.

The fleet-wide analyst view the workspace lacked: each campaign PDF is one model,
but "which model do I pick" is a *cross-campaign* question. This module walks the
campaigns dir (like ``experiments_index``) and emits three complementary
leaderboards keyed off the AA latency cells + the roofline/throughput cells in
``atlas.jsonl``:

- ``AA-FLEET-LEADERBOARD-<hw>.md``        latency tier (aa-1k/10k/100k, c=1 + c=10)
- ``THROUGHPUT-FLEET-LEADERBOARD-<hw>.md`` throughput tier (peak tok/s/GPU, TP=4)
- ``FLEET-MODEL-SELECTION-<hw>.md``        decision capstone (perf Pareto frontier)

Provenance: this productizes the four hand-written generators
(``perf-tune-report/gen-aa-leaderboard.py`` etc.) into one re-runnable verb. Reads
only local ``atlas.jsonl`` (no pyarrow, no network). The per-tier *why* (DCGM L3
+ zymtrace L1: the fleet is host/KV-bound, not memory-bound) is captured in
``perf-tune-report/FLEET-THROUGHPUT-ATTRIBUTION-*.md``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from tools.perf_tune_report.schema import AtlasCell, read_jsonl

GPU_HR_DEFAULT = 8.60  # GB300 $/gpu-hour
SHAPES = ("aa-1k", "aa-10k", "aa-100k")


# --------------------------------------------------------------------------- #
# model-name normalization (atlas `model` strings drift across campaigns)
# --------------------------------------------------------------------------- #
def canon_model(raw: str | None) -> str:
    """Collapse drifted atlas model strings to one display name (throughput tier)."""
    if not raw:
        return "unknown"
    m = raw.split("/")[-1]
    if "GGUF" in m or "gguf" in m or "Q2_K" in m:
        return "GLM-5.1 (GGUF Q2_K_XL)"
    m = re.sub(r"-?(NVFP4|MXFP4|FP8|BF16)$", "", m, flags=re.IGNORECASE)
    m = re.sub(r"-Thinking(-\d+)?$", "", m, flags=re.IGNORECASE)
    m = re.sub(r"-(it|Instruct)$", "", m, flags=re.IGNORECASE)
    m = re.sub(r"-2507$", "", m)
    table = {
        "gemma-4-26B-A4B": "Gemma-4-26B-A4B", "Gemma-4-26B-A4B": "Gemma-4-26B-A4B",
        "Llama-3.1-8B": "Llama-3.1-8B", "Llama-3.3-70B": "Llama-3.3-70B",
    }
    return table.get(m, m)


def aa_display(raw: str | None) -> str:
    """AA-tier display name (adds the GLM-5.1 sparse annotation)."""
    base = canon_model(raw)
    return "GLM-5.1 (DSA sparse)" if base == "GLM-5.1" else base


def read_all_rows(campaigns_root: Path, hardware_filter: str = "GB300") -> list[AtlasCell]:
    """Concatenate atlas.jsonl rows across every campaign dir, filtered to hardware."""
    rows: list[AtlasCell] = []
    for d in sorted(campaigns_root.iterdir()):
        atlas = d / "atlas.jsonl"
        if atlas.is_file():
            try:
                for r in read_jsonl(atlas):
                    if hardware_filter in (r.hardware or ""):
                        rows.append(r)
            except Exception:  # noqa: BLE001 - skip a malformed campaign, never fail the fleet view
                continue
    return rows


def _cost(tps: float | None, gpu_hr: float) -> float | None:
    return (gpu_hr / 3600.0) / tps * 1e6 if isinstance(tps, (int, float)) and tps else None


def resolve_gpu_hr(hardware: str, configs_dir: Path | None = None,
                   override: float | None = None) -> float:
    """Resolve the $/GPU-hour rate (the adjustable cost knob). An explicit --gpu-hr override
    wins; else read ``perf-tune-report/configs/cost.yaml`` ``usd_per_gpu_hour[hardware]`` (or
    ``[default]``); else ``GPU_HR_DEFAULT``. Cost is a simple lookup, not a model."""
    if override is not None:
        return float(override)
    if configs_dir is not None:
        cost_yaml = Path(configs_dir) / "cost.yaml"
        if cost_yaml.is_file():
            try:
                import yaml  # type: ignore
                rates = (yaml.safe_load(cost_yaml.read_text()) or {}).get("usd_per_gpu_hour") or {}
                if hardware in rates:
                    return float(rates[hardware])
                if "default" in rates:
                    return float(rates["default"])
            except Exception:  # noqa: BLE001 - cost is best-effort; fall back to the default
                pass
    return GPU_HR_DEFAULT


# --------------------------------------------------------------------------- #
# AA latency tier
# --------------------------------------------------------------------------- #
def build_aa(rows: list[AtlasCell]) -> dict[str, dict[tuple[str, int], dict[str, Any]]]:
    """model -> {(shape, conc): {ttft, opu, opg}} (latest captured wins)."""
    data: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    seen_at: dict[tuple[str, tuple[str, int]], str] = {}
    for r in rows:
        shape = r.cell_id if r.cell_id in SHAPES else (getattr(r, "extra", None) or {}).get("aa_shape")
        if shape not in SHAPES:
            continue
        model = aa_display(r.model)
        key = (shape, r.concurrency)
        cap = r.captured_at or ""
        # Reasoning-aware preference: a row carrying think-token data (parser-ON
        # measurement) outranks a later parser-OFF/plain row for the TTFO fields;
        # ttft/opu/opg still follow latest-captured.
        prev = data.get(model, {}).get(key)
        if seen_at.get((model, key), "") <= cap:
            seen_at[(model, key)] = cap
            entry = {
                "ttft": r.ttft_avg_ms, "opu": r.output_tps_per_user, "opg": r.output_tps_per_gpu,
                "ttfo": getattr(r, "ttfo_avg_ms", None),
                "cov": getattr(r, "ttfo_coverage", None),
                "rtoks": getattr(r, "reasoning_token_count", None),
            }
            if prev is not None and prev.get("rtoks") is not None and entry["rtoks"] is None:
                for k in ("ttfo", "cov", "rtoks"):
                    entry[k] = prev[k]
            data.setdefault(model, {})[key] = entry
        elif prev is not None and prev.get("rtoks") is None and getattr(r, "reasoning_token_count", None) is not None:
            prev.update({"ttfo": getattr(r, "ttfo_avg_ms", None),
                         "cov": getattr(r, "ttfo_coverage", None),
                         "rtoks": getattr(r, "reasoning_token_count", None)})
    return data


# --------------------------------------------------------------------------- #
# throughput tier
# --------------------------------------------------------------------------- #
def build_throughput(rows: list[AtlasCell]) -> list[dict[str, Any]]:
    """One dict per (model, quant, TP) config with its peak-throughput point."""
    best: dict[tuple[str, str, int], dict[str, Any]] = {}
    for r in rows:
        if r.status not in ("full", "ok", "None"):
            continue
        if (r.cell_id or "").startswith("aa-"):
            continue
        tps = r.output_tps_per_gpu
        if not isinstance(tps, (int, float)):
            continue
        tp = r.tensor_parallel or 4
        key = (canon_model(r.model), str(r.quant or "").upper(), tp)
        if key not in best or tps > best[key]["tps_gpu"]:
            best[key] = {
                "model": key[0], "quant": key[1], "tp": tp, "tps_gpu": tps,
                "conc": r.concurrency, "tps_user": r.output_tps_per_user,
                "ttft": r.ttft_avg_ms, "mnbt": r.max_num_batched_tokens,
            }
    return list(best.values())


# --------------------------------------------------------------------------- #
# Pareto decision tier
# --------------------------------------------------------------------------- #
def build_pareto(rows: list[AtlasCell]) -> dict[str, dict[str, float]]:
    """model -> {lat_user, lat_ttft (aa-1k c=1), tps_gpu (peak TP=4 non-AA)}.

    Only models with all three axes measured (a fair Pareto needs all axes)."""
    M: dict[str, dict[str, Any]] = {}
    for r in rows:
        m = canon_model(r.model)
        e = M.setdefault(m, {"lat_user": None, "lat_ttft": None, "tps_gpu": None})
        cid = r.cell_id or ""
        if cid == "aa-1k" and r.concurrency == 1:
            u = r.output_tps_per_user
            if isinstance(u, (int, float)) and (e["lat_user"] is None or u > e["lat_user"]):
                e["lat_user"] = u
                e["lat_ttft"] = r.ttft_avg_ms
        elif not cid.startswith("aa-") and (r.tensor_parallel or 4) == 4:
            g = r.output_tps_per_gpu
            if isinstance(g, (int, float)) and (e["tps_gpu"] is None or g > e["tps_gpu"]):
                e["tps_gpu"] = g
    return {m: e for m, e in M.items() if all(e[k] is not None for k in ("lat_user", "lat_ttft", "tps_gpu"))}


def build_quality(rows: list[AtlasCell]) -> dict[str, dict[str, float]]:
    """model -> {metric_name: value} from import_model_eval ``eval_acc`` cells (latest wins).

    Serving quality (GPQA/MMLU-Pro) is the REAL model-selection driver; this surfaces it next
    to the perf Pareto, kept SEPARATE per the PERF != quality caveat. Empty until evals are
    imported via ``perftunereport import_model_eval``."""
    data: dict[str, dict[str, float]] = {}
    seen: dict[tuple[str, str], str] = {}
    for r in rows:
        extra = getattr(r, "extra", None) or {}
        if extra.get("metric_kind") != "eval_acc":
            continue
        qm = extra.get("quality_metrics")
        if not isinstance(qm, dict):
            continue
        m = canon_model(r.model)
        cap = r.captured_at or ""
        for name, val in qm.items():
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                continue
            key = (m, str(name))
            if seen.get(key, "") > cap:
                continue
            seen[key] = cap
            data.setdefault(m, {})[str(name)] = float(val)
    return data


def dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    """a perf-dominates b: >= on all 3 (ttft inverted) and strictly > on >=1."""
    ge = a["lat_user"] >= b["lat_user"] and a["lat_ttft"] <= b["lat_ttft"] and a["tps_gpu"] >= b["tps_gpu"]
    gt = a["lat_user"] > b["lat_user"] or a["lat_ttft"] < b["lat_ttft"] or a["tps_gpu"] > b["tps_gpu"]
    return ge and gt


# --------------------------------------------------------------------------- #
# renderers
# --------------------------------------------------------------------------- #
def _f(x: Any, n: int = 1) -> str:
    return f"%.{n}f" % x if isinstance(x, (int, float)) else "NA"


def render_aa_md(data: dict, hw: str, gpu_hr: float) -> str:
    L = [f"# Artificial Analysis (AA) fleet leaderboard - {hw} TP4 NVFP4", "",
         f"AUTO-GENERATED by `perftunereport fleet_leaderboard` ({len(data)} models). Re-run after any new AA "
         "campaign publishes. Same methodology (AIPerf synthetic, temp 0 / top_p 1 / min_tokens+ignore_eos, "
         f"o200k_base), same HW ({hw}, TP=4, NVFP4). Read-only synthesis.", "",
         "## Latency tier (c=1, single prompt) - sorted by output tok/s/user"]
    for s in SHAPES:
        ranked = sorted([(m, data[m][(s, 1)]) for m in data if (s, 1) in data[m]],
                        key=lambda r: -(r[1].get("opu") or 0))
        if not ranked:
            continue
        L += ["", f"### {s} (c=1)", "", "| rank | model | TTFT (ms) | output tok/s/user | tok/s/GPU |",
              "| ---: | --- | ---: | ---: | ---: |"]
        for i, (m, v) in enumerate(ranked, 1):
            L.append(f"| {i} | {m} | {_f(v['ttft'])} | {_f(v['opu'])} | {_f(v['opg'])} |")
    L += ["", "## Answer latency (TTFO, reasoning-aware; c=1) - TTFO is an ANSWERED-SUBSET stat, always read with its coverage"]
    for s in SHAPES:
        ranked = sorted([(m, data[m][(s, 1)]) for m in data
                         if (s, 1) in data[m] and data[m][(s, 1)].get("cov") is not None],
                        key=lambda r: (r[1].get("ttfo") is None, r[1].get("ttfo") or 0))
        if not ranked:
            continue
        L += ["", f"### {s} (c=1)", "",
              "| rank | model | TTFO (ms) | coverage | think toks (all reqs) | TTFT (ms) |",
              "| ---: | --- | ---: | ---: | ---: | ---: |"]
        for i, (m, v) in enumerate(ranked, 1):
            cov = f"{v['cov']:.0%}" if v.get("cov") is not None else "NA"
            ttfo = _f(v["ttfo"]) if v.get("ttfo") is not None else "never answers"
            rt = _f(v["rtoks"]) if v.get("rtoks") is not None else "-"
            L.append(f"| {i} | {m} | {ttfo} | {cov} | {rt} | {_f(v['ttft'])} |")
    L += ["", "## Parallel tier (c=10) - sorted by tok/s/GPU"]
    for s in ("aa-1k", "aa-10k"):
        ranked = sorted([(m, data[m][(s, 10)]) for m in data if (s, 10) in data[m]],
                        key=lambda r: -(r[1].get("opg") or 0))
        if not ranked:
            continue
        L += ["", f"### {s} (c=10)", "", "| rank | model | TTFT (ms) | tok/s/user | tok/s/GPU |",
              "| ---: | --- | ---: | ---: | ---: |"]
        for i, (m, v) in enumerate(ranked, 1):
            L.append(f"| {i} | {m} | {_f(v['ttft'])} | {_f(v['opu'])} | {_f(v['opg'])} |")
    L += ["", "## Read",
          "- Active-param count drives decode speed (per-token HBM traffic), but absolute GPU utilization is "
          "LOW (HBM <15%, L3-grounded) -> host/KV-scheduling-bound, NOT memory-bound. Grounded cross-model "
          "attribution: `FLEET-THROUGHPUT-ATTRIBUTION-GB300.md`.",
          "- MiniMax-M2.7 typically has the lowest TTFT; GLM-5.1 (DSA sparse) is the long-context outlier.", "",
          f"## Cost ($/1M output tok @ ${gpu_hr}/GPU-hour) - cheapest first", "",
          "| model | aa-1k c=1 | aa-1k c=10 | aa-10k c=1 |", "| --- | ---: | ---: | ---: |"]
    crows = []
    for m in data:
        c1 = _cost(data[m].get(("aa-1k", 1), {}).get("opg"), gpu_hr)
        c10 = _cost(data[m].get(("aa-1k", 10), {}).get("opg"), gpu_hr)
        c10k = _cost(data[m].get(("aa-10k", 1), {}).get("opg"), gpu_hr)
        crows.append((m, c1, c10, c10k))
    crows.sort(key=lambda r: (r[1] if r[1] is not None else 9e9))
    dol = lambda x: ("$%.2f" % x) if isinstance(x, (int, float)) else "(DNF)"
    for m, c1, c10, c10k in crows:
        L.append(f"| {m} | {dol(c1)} | {dol(c10)} | {dol(c10k)} |")
    L += ["", "Batching (c=10) is ~7x cheaper than single-prompt (c=1). Companion: "
          "`THROUGHPUT-FLEET-LEADERBOARD-GB300.md`, `FLEET-MODEL-SELECTION-GB300.md`.", ""]
    return "\n".join(L)


def render_throughput_md(allrows: list[dict], hw: str, gpu_hr: float) -> str:
    def table(rs: list[dict], header: str) -> list[str]:
        rs = sorted(rs, key=lambda r: -r["tps_gpu"])
        out = [header, "", "| rank | model | quant | TP | peak tok/s/GPU | @conc | tok/s/user@peak | "
               "TTFT@peak (ms) | $/1M out |", "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for i, r in enumerate(rs, 1):
            c = _cost(r["tps_gpu"], gpu_hr)
            out.append(f"| {i} | {r['model']} | {r['quant']} | {r['tp']} | {_f(r['tps_gpu'])} | {r['conc']} | "
                       f"{_f(r['tps_user'])} | {_f(r['ttft'])} | {('$%.2f' % c) if c else 'NA'} |")
        return out
    tp4 = [r for r in allrows if r["tp"] == 4]
    alt = [r for r in allrows if r["tp"] in (2, 8)]
    nmodels = len({r["model"] for r in tp4})
    L = [f"# Throughput fleet leaderboard (peak tok/s/GPU) - {hw} NVFP4", "",
         "> **PROVISIONAL** when generated mid-sweep (the roofline sweep is an active workstream; numbers are "
         "best-so-far). Re-run `perftunereport fleet_leaderboard` to refresh.", "",
         f"AUTO-GENERATED ({nmodels} models, TP=4). Peak = max sustained output tok/s/GPU across each config's "
         f"concurrency sweep, with the latency AT that peak. {hw} / status=full only; AA cells excluded "
         "(latency leaderboard). TP=1 microbenches excluded; TP=2/8 = sharding study.", ""]
    L += table(tp4, "## Production tier (TP=4) - sorted by peak tok/s/GPU")
    if alt:
        L += ["", *table(alt, "## Alternate-sharding study (TP=2 / TP=8)")]
    L += ["", "## Read",
          "- Peak throughput is a batch-decode number (high concurrency) - the inverse-latency end of the "
          "Pareto front. DCGM-grounded: even at peak the small-active MoEs sit at HBM <8% / tensor <3% "
          "(GPU-starved, host/KV-bound) - see `FLEET-THROUGHPUT-ATTRIBUTION-GB300.md`.",
          "- $/1M here is the throughput-optimal cost (~7x cheaper than the AA c=1 latency-optimal cost).", ""]
    return "\n".join(L)


def render_pareto_md(M: dict, hw: str, gpu_hr: float, quality: dict | None = None) -> str:
    names = list(M)
    frontier = [n for n in names if not any(dominates(M[o], M[n]) for o in names if o != n)]
    dominated = [n for n in names if n not in frontier]
    best_decode = max(names, key=lambda n: M[n]["lat_user"]) if names else None
    best_ttft = min(names, key=lambda n: M[n]["lat_ttft"]) if names else None
    best_tput = max(names, key=lambda n: M[n]["tps_gpu"]) if names else None
    L = [f"# Fleet model-selection guide (perf Pareto frontier) - {hw} TP4 NVFP4", "",
         "> **PROVISIONAL** where the throughput axis is still filling. Capstone over the AA latency + "
         "throughput leaderboards. Re-run `perftunereport fleet_leaderboard`.", "",
         f"AUTO-GENERATED ({len(names)} models with all 3 axes). Axes: decode latency (aa-1k c=1 tok/s/user, "
         "higher better), first-token (aa-1k c=1 TTFT, lower better), batch throughput (peak tok/s/GPU TP=4, "
         "higher better).", ""]
    if names:
        L += ["## Pick by priority", "", "| optimize for... | pick | number |", "| --- | --- | --- |",
              f"| interactive decode | **{best_decode}** | {_f(M[best_decode]['lat_user'])} tok/s/user |",
              f"| lowest first-token | **{best_ttft}** | {_f(M[best_ttft]['lat_ttft'])} ms TTFT |",
              f"| batch throughput / $ | **{best_tput}** | {_f(M[best_tput]['tps_gpu'])} tok/s/GPU "
              f"(${_cost(M[best_tput]['tps_gpu'], gpu_hr):.2f}/1M) |", ""]
    L += ["## Perf-efficient frontier (non-dominated - choose from these)", "",
          "| model | decode tok/s/user | TTFT (ms) | peak tok/s/GPU | $/1M out |",
          "| --- | ---: | ---: | ---: | ---: |"]
    for n in sorted(frontier, key=lambda x: -M[x]["tps_gpu"]):
        e = M[n]
        L.append(f"| {n} | {_f(e['lat_user'])} | {_f(e['lat_ttft'])} | {_f(e['tps_gpu'])} | "
                 f"${_cost(e['tps_gpu'], gpu_hr):.2f} |")
    L += ["", "## Perf-dominated (beaten on ALL three axes)", ""]
    if dominated:
        L += ["| model | decode tok/s/user | TTFT (ms) | peak tok/s/GPU | dominated by |",
              "| --- | ---: | ---: | ---: | --- |"]
        for n in sorted(dominated, key=lambda x: -M[x]["tps_gpu"]):
            e = M[n]
            dom = [o for o in frontier if dominates(M[o], e)]
            L.append(f"| {n} | {_f(e['lat_user'])} | {_f(e['lat_ttft'])} | {_f(e['tps_gpu'])} | "
                     f"{', '.join(dom) or '-'} |")
    else:
        L.append("_None - every measured model is on the perf frontier._")
    L += ["", "## Model quality (lm-eval serving evals, where measured)",
          "Quality is the REAL model-selection driver, shown SEPARATELY from the perf Pareto "
          "(a perf-dominated model may be the right pick on quality). Captured via "
          "`perftunereport import_model_eval` -> quality_v1; empty until serving evals are imported.",
          ""]
    measured = {m: q for m, q in (quality or {}).items() if q}
    if measured:
        metric_names = sorted({n for q in measured.values() for n in q})
        L += ["| model | " + " | ".join(metric_names) + " |",
              "| --- | " + " | ".join(["---:"] * len(metric_names)) + " |"]
        for m in sorted(measured):
            cells = " | ".join(
                (_f(measured[m].get(n), 3) if measured[m].get(n) is not None else "-")
                for n in metric_names
            )
            L.append(f"| {m} | {cells} |")
    else:
        L.append("_None measured yet -- run `inference-model-eval` + `perftunereport import_model_eval`._")
    L += ["", "## Critical caveat: PERF-only Pareto, NOT model quality",
          "- The Pareto frontier above ranks PERF only; \"dominated\" = perf-dominated on the AA "
          "short-context shape, NOT \"never use\". Cross-reference the quality table above (reasoning, "
          "long-context, agentic quality is orthogonal and is usually the real selection driver). "
          "GLM-5.1's value is DSA-sparse long-context, not the AA 1k-shape decode this frontier measures.", ""]
    return "\n".join(L)


def write_leaderboards(rows: list[AtlasCell], out_dir: Path, hw: str = "GB300",
                       gpu_hr: float = GPU_HR_DEFAULT) -> dict[str, str]:
    """Render + write all three leaderboards; return {name: path}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    aa = build_aa(rows)
    tput = build_throughput(rows)
    pareto = build_pareto(rows)
    quality = build_quality(rows)
    files = {
        "aa": out_dir / f"AA-FLEET-LEADERBOARD-{hw}.md",
        "throughput": out_dir / f"THROUGHPUT-FLEET-LEADERBOARD-{hw}.md",
        "pareto": out_dir / f"FLEET-MODEL-SELECTION-{hw}.md",
    }
    files["aa"].write_text(render_aa_md(aa, hw, gpu_hr) + "\n", encoding="utf-8")
    files["throughput"].write_text(render_throughput_md(tput, hw, gpu_hr) + "\n", encoding="utf-8")
    files["pareto"].write_text(render_pareto_md(pareto, hw, gpu_hr, quality) + "\n", encoding="utf-8")
    return {
        "aa_models": len(aa),
        "throughput_configs": len(tput),
        "pareto_models": len(pareto),
        "quality_models": len(quality),
        **{k: str(v) for k, v in files.items()},
    }
