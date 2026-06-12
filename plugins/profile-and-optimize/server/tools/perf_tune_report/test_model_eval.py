"""Tests for the lm-eval-harness -> quality cell importer (import_model_eval)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.perf_tune_report.importers.model_eval import import_model_eval, parse_eval_metrics
from tools.perf_tune_report.perf_tune_report_cli import main

_RESULTS = {
    "results": {
        "gpqa_main_zeroshot": {"alias": "gpqa", "acc,none": 0.52, "acc_stderr,none": 0.07},
        "mmlu_pro": {"acc,none": 0.903, "acc_norm,none": 0.91},
    },
    "config": {"model_args": "model=GLM-5.1,base_url=http://localhost:8000/v1/completions"},
}


def test_parse_eval_metrics_keeps_acc_drops_stderr():
    m = parse_eval_metrics(_RESULTS)
    assert m["gpqa_main_zeroshot_acc"] == 0.52
    assert m["mmlu_pro_acc"] == 0.903
    assert m["mmlu_pro_acc_norm"] == 0.91
    assert all("stderr" not in k for k in m)  # *_stderr dropped


def test_import_model_eval_writes_quality_cell(tmp_path: Path):
    res = tmp_path / "results.json"
    res.write_text(json.dumps(_RESULTS))
    camp = tmp_path / "campaigns" / "glm51-eval-20260607T000000Z"
    camp.mkdir(parents=True)
    out = import_model_eval(res, camp, model="GLM-5.1-NVFP4", hardware="GB300",
                            quant="NVFP4", tensor_parallel=4)
    assert out["n_metrics"] == 3
    norm = json.loads((camp / "cells" / "model-eval" / "normalized.json").read_text())
    assert isinstance(norm, list) and len(norm) == 1
    cell = norm[0]
    # metric_kind=eval_acc + quality_metrics -> lands in quality_v1 on publish.
    assert cell["extra"]["metric_kind"] == "eval_acc"
    assert cell["extra"]["quality_metrics"]["mmlu_pro_acc"] == 0.903
    assert cell["dataset"] == "eval"
    assert cell["max_num_batched_tokens"] == 0  # accuracy run; exempt from methodology gate
    assert cell["status"] == "full"


def test_import_model_eval_no_metrics_raises(tmp_path: Path):
    res = tmp_path / "r.json"
    res.write_text(json.dumps({"results": {"x": {"perplexity,none": 5.0}}}))
    camp = tmp_path / "c-20260607T000000Z"
    camp.mkdir()
    with pytest.raises(ValueError):
        import_model_eval(res, camp, model="m", hardware="GB300", quant="NVFP4")


def test_cli_import_model_eval(tmp_path: Path, capsys):
    res = tmp_path / "results.json"
    res.write_text(json.dumps(_RESULTS))
    camps = tmp_path / "campaigns"
    camp = camps / "glm51-eval-20260607T000000Z"
    camp.mkdir(parents=True)
    rc = main(["import_model_eval", "--results", str(res), "--campaign", str(camp),
               "--model", "GLM-5.1-NVFP4", "--hardware", "GB300", "--quant", "NVFP4",
               "--tensor-parallel", "4", "--campaigns-dir", str(camps), "--json"])
    assert rc == 0
    env = json.loads(capsys.readouterr().out)
    assert env["n_metrics"] == 3 and env["cell_id"] == "model-eval"
    assert "gpqa_main_zeroshot" in env["tasks"]
