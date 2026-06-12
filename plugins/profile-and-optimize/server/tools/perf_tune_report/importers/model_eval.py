"""Import an lm-eval-harness ``results.json`` into a perf-report quality cell.

`inference-model-eval` (GPQA / MMLU-Pro via lm-eval-harness) writes a standalone
``results_<ts>.json`` -- it was never captured into a campaign or the perf-lake, so
serving quality was invisible to the model-selection view. This importer parses that
file into ONE accuracy cell (``cells/<cell-id>/normalized.json``) with
``extra["metric_kind"] = "eval_acc"`` + ``extra["quality_metrics"] = {task_metric: value}``,
so ``atlas_aggregate`` -> ``publish_to_lake`` lands it in the existing ``quality_v1`` table
(``build_quality_table`` already reads ``metric_kind`` + ``quality_metrics``). The cell carries
NO throughput/latency metrics, so it is exempt from the warm/cold + shape methodology gate
(``_row_is_measured`` is False), and the campaign should use ``focus: accuracy``.

Dependency-light (stdlib + schema only).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.perf_tune_report.schema import STATUS_FULL, AtlasCell

# lm-eval metric keys we treat as quality scores; ``*_stderr`` and hyperparameters are dropped.
# lm-eval emits keys like ``"acc,none"`` / ``"acc_norm,none"`` / ``"exact_match,none"``.
_EVAL_METRIC_PREFIXES = ("acc", "acc_norm", "exact_match", "f1", "pass@1", "score", "em")


def parse_eval_metrics(results: dict[str, Any]) -> dict[str, float]:
    """Flatten lm-eval ``results`` into ``{task_metric: value}`` quality scores.

    ``results`` is the lm-eval ``results.json`` top-level dict (``{"results": {task: {...}}}``)
    or the inner ``results`` mapping directly. Keys like ``"acc,none"`` are normalized to
    ``"<task>_acc"``; ``*_stderr`` and non-accuracy keys are dropped."""
    inner = results.get("results") if isinstance(results.get("results"), dict) else results
    out: dict[str, float] = {}
    for task, md in (inner or {}).items():
        if not isinstance(md, dict):
            continue
        for k, v in md.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            name = str(k).split(",")[0]  # "acc,none" -> "acc"
            if "stderr" in name.lower():
                continue
            if not any(name == p or name.startswith(p) for p in _EVAL_METRIC_PREFIXES):
                continue
            out[f"{task}_{name}"] = float(v)
    return out


def import_model_eval(
    results_path: Path,
    campaign_dir: Path,
    *,
    model: str,
    hardware: str,
    quant: str,
    tensor_parallel: int = 1,
    cell_id: str = "model-eval",
    parallel_strategy: str = "TP",
    kv_cache_dtype: str = "unknown",
    image: str = "unknown",
) -> dict[str, Any]:
    """Parse ``results_path`` (lm-eval results.json) -> a quality cell under ``campaign_dir``.

    Writes ``cells/<cell_id>/normalized.json`` (a JSON list with one AtlasCell) + ``status.txt``
    so ``atlas_aggregate`` includes it. Returns a summary dict."""
    results = json.loads(Path(results_path).read_text())
    metrics = parse_eval_metrics(results)
    if not metrics:
        raise ValueError(
            f"no eval metrics parsed from {results_path} (expected lm-eval 'results' with "
            "acc/acc_norm/exact_match keys)"
        )
    tasks = sorted({name.rsplit("_", 1)[0] for name in metrics})
    row = AtlasCell(
        cell_id=cell_id,
        model=model,
        hardware=hardware,
        quant=quant,
        tensor_parallel=int(tensor_parallel),
        parallel_strategy=parallel_strategy,
        mtp=False,
        max_num_batched_tokens=0,  # accuracy run: no serving shape (exempt from the methodology gate)
        concurrency=1,
        status=STATUS_FULL,
        dataset="eval",
        kv_cache_dtype=kv_cache_dtype,
        image=image,
        notes="lm-eval-harness serving quality",
        extra={
            "metric_kind": "eval_acc",
            "quality_metrics": metrics,
            "runner": "lm-eval-harness",
            "eval_tasks": tasks,
        },
    )
    cell_dir = Path(campaign_dir) / "cells" / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    normalized = cell_dir / "normalized.json"
    normalized.write_text(json.dumps([row.to_dict()], indent=2, sort_keys=True) + "\n")
    (cell_dir / "status.txt").write_text(STATUS_FULL + "\n")
    return {
        "cell_id": cell_id,
        "n_metrics": len(metrics),
        "tasks": tasks,
        "metrics": metrics,
        "normalized": str(normalized),
    }
