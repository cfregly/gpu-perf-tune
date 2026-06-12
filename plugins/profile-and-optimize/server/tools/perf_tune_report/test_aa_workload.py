"""Unit tests for the Artificial Analysis (AA) workload backend.

Covers the shared command builder + dataset generator + normalizer
(``aa_workload``), the thin runner (``aa_bench``), the ``cell_run --backend
aa --dry-run`` CLI path, and a drift-guard that the bundled standalone
script's ``AA_SHAPES`` stay in sync with the package module.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from tools.perf_tune_report.runners.aa_bench import run_cell as run_cell_aa
from tools.perf_tune_report.runners.aa_workload import (
    AA_SHAPES,
    MODE_DATASET_REPLAY,
    MODE_SYNTHETIC,
    build_aiperf_command,
    generate_aa_dataset,
    normalize_outputs,
)
from tools.perf_tune_report.runners.common import CellConfig
from tools.perf_tune_report.schema import BACKEND_AIPERF, STATUS_FAILED, STATUS_FULL


# --- shapes ---------------------------------------------------------------


def test_aa_shapes_match_methodology():
    assert AA_SHAPES["aa-1k"].input_tokens == 1000
    assert AA_SHAPES["aa-1k"].output_tokens == 1000
    assert AA_SHAPES["aa-10k"].input_tokens == 10000
    assert AA_SHAPES["aa-10k"].output_tokens == 1500
    assert AA_SHAPES["aa-100k"].input_tokens == 100000
    assert AA_SHAPES["aa-100k"].output_tokens == 2000


def test_standalone_script_shapes_drift_guard():
    """The bundled self-contained script must declare the same shapes as the
    package module (the script intentionally has no package import)."""
    script = (
        Path(__file__).resolve().parents[3]
        / "skills"
        / "inference-aa-workload"
        / "assets"
        / "repro_artificialanalysis.py"
    )
    spec = importlib.util.spec_from_file_location("aa_repro_script", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pkg = {k: (v.input_tokens, v.output_tokens) for k, v in AA_SHAPES.items()}
    assert mod.AA_SHAPES == pkg


# --- command builder ------------------------------------------------------


def test_build_command_synthetic():
    cmd = build_aiperf_command(
        AA_SHAPES["aa-10k"],
        aiperf_cmd=["aiperf"],
        model="m",
        url="http://x:8000",
        output_artifact_dir="/tmp/out",
        concurrency=4,
        request_count=20,
        api_key="secret",
        tokenizer="m",
        mode=MODE_SYNTHETIC,
    )
    assert cmd[:2] == ["aiperf", "profile"]
    assert "--synthetic-input-tokens-mean" in cmd
    assert cmd[cmd.index("--synthetic-input-tokens-mean") + 1] == "10000"
    assert cmd[cmd.index("--output-tokens-mean") + 1] == "1500"
    assert "--extra-inputs" in cmd and "temperature:0" in cmd and "top_p:1" in cmd
    assert "min_tokens:1500" in cmd and "ignore_eos:true" in cmd
    assert cmd[cmd.index("--api-key") + 1] == "secret"
    assert cmd[cmd.index("--concurrency") + 1] == "4"
    assert "--input-file" not in cmd


def test_build_command_dataset_replay():
    cmd = build_aiperf_command(
        AA_SHAPES["aa-1k"],
        aiperf_cmd=["aiperf"],
        model="m",
        url="http://x:8000",
        output_artifact_dir="/tmp/out",
        mode=MODE_DATASET_REPLAY,
        input_file="/data/aa-1k.jsonl",
    )
    assert cmd[cmd.index("--input-file") + 1] == "/data/aa-1k.jsonl"
    assert cmd[cmd.index("--custom-dataset-type") + 1] == "mooncake_trace"
    assert "--synthetic-input-tokens-mean" not in cmd


def test_build_command_no_extra_output_controls():
    cmd = build_aiperf_command(
        AA_SHAPES["aa-1k"],
        aiperf_cmd=["aiperf"],
        model="m",
        url="http://x:8000",
        output_artifact_dir="/tmp/out",
        extra_output_controls=False,
    )
    assert "min_tokens:1000" not in cmd
    assert "ignore_eos:true" not in cmd
    # temperature/top_p are still applied regardless.
    assert "temperature:0" in cmd


def test_build_command_replay_requires_input_file():
    import pytest

    with pytest.raises(ValueError):
        build_aiperf_command(
            AA_SHAPES["aa-1k"],
            aiperf_cmd=["aiperf"],
            model="m",
            url="http://x:8000",
            output_artifact_dir="/tmp/out",
            mode=MODE_DATASET_REPLAY,
        )


# --- dataset generator ----------------------------------------------------


def test_generate_aa_dataset(tmp_path: Path):
    out = tmp_path / "aa-1k.jsonl"
    info = generate_aa_dataset(AA_SHAPES["aa-1k"], 5, out, use_tiktoken=False)
    assert info["rows"] == 5
    assert info["used_tiktoken"] is False
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 5
    row = json.loads(lines[0])
    assert "text_input" in row
    assert row["output_length"] == 1000
    # Heuristic should land in a sane ballpark for ~1000 tokens.
    assert len(row["text_input"].split()) > 500


# --- normalizer -----------------------------------------------------------


def _cell() -> CellConfig:
    return CellConfig(
        cell_id="aa-10k",
        model="m",
        hardware="B200",
        quant="NVFP4",
        tensor_parallel=8,
        parallel_strategy="TP",
        mtp=False,
        max_num_batched_tokens=4096,
        concurrencies=(1,),
    )


def test_normalize_outputs_happy_path(tmp_path: Path):
    cell = _cell()
    cell_dir = tmp_path / "cells" / cell.cell_id
    raw = cell_dir / "raw" / "c1"
    raw.mkdir(parents=True)
    (raw / "profile_export_aiperf.json").write_text(
        json.dumps(
            {
                "median_ttft_ms": 123.4,
                "request_throughput": 2.5,
                "output_throughput": 800.0,
            }
        )
    )
    rows, status = normalize_outputs(
        cell, cell_dir / "raw", cell_dir, shape=AA_SHAPES["aa-10k"], mode=MODE_SYNTHETIC
    )
    assert status == STATUS_FULL
    assert len(rows) == 1
    r = rows[0]
    assert r.concurrency == 1
    assert r.ttft_avg_ms == 123.4
    assert r.backend == BACKEND_AIPERF
    assert r.output_tps_per_gpu == 800.0 / 8
    assert r.extra["aa_shape"] == "aa-10k"
    assert r.extra["aa_mode"] == MODE_SYNTHETIC
    # AA shape ISL/OSL promoted to typed fields + dataset tagged (2026-06-07): the
    # leaderboard/lake ground the ranking at the real AA shape, not dataset=unknown.
    assert r.mean_input_tokens == 10000.0
    assert r.mean_output_tokens == 1500.0
    assert r.dataset == "aa"
    assert r.bench_backend == "aiperf"


def test_normalize_outputs_aiperf_0_9_nested_schema(tmp_path: Path):
    """aiperf>=0.9 nests metrics as {"unit","avg"} and renames the throughput /
    ttft keys; the normalizer must parse it (regression: the flat-key extractor
    silently produced 0 rows -> STATUS_FAILED on a successful c=1 GLM-5.1 run)."""
    cell = _cell()
    cell_dir = tmp_path / "cells" / cell.cell_id
    raw = cell_dir / "raw" / "c1"
    raw.mkdir(parents=True)
    (raw / "profile_export_aiperf.json").write_text(
        json.dumps(
            {
                "time_to_first_token": {"unit": "ms", "avg": 230.5},
                "request_throughput": {"unit": "requests/sec", "avg": 0.188},
                "output_token_throughput": {"unit": "tokens/sec", "avg": 188.4},
                "output_sequence_length": {"unit": "tokens", "avg": 1000.0},
            }
        )
    )
    rows, status = normalize_outputs(
        cell, cell_dir / "raw", cell_dir, shape=AA_SHAPES["aa-1k"], mode=MODE_SYNTHETIC
    )
    assert status == STATUS_FULL
    assert len(rows) == 1
    r = rows[0]
    assert r.ttft_avg_ms == 230.5
    assert r.request_throughput_avg == 0.188
    assert r.output_tps_per_gpu == 188.4 / 8
    assert r.output_tps_per_user == 188.4 / 1


def test_normalize_outputs_no_reports_is_failed(tmp_path: Path):
    cell = _cell()
    cell_dir = tmp_path / "cells" / cell.cell_id
    (cell_dir / "raw").mkdir(parents=True)
    rows, status = normalize_outputs(
        cell, cell_dir / "raw", cell_dir, shape=AA_SHAPES["aa-10k"], mode=MODE_SYNTHETIC
    )
    assert rows == []
    assert status == STATUS_FAILED


# --- runner dry-run -------------------------------------------------------


def test_run_cell_aa_dry_run(tmp_path: Path):
    cell = CellConfig(
        cell_id="aa-1k",
        model="m",
        hardware="B200",
        quant="NVFP4",
        tensor_parallel=8,
        parallel_strategy="TP",
        mtp=False,
        max_num_batched_tokens=4096,
        concurrencies=(1, 10),
    )
    result = run_cell_aa(
        cell,
        tmp_path,
        shape_name="aa-1k",
        model="m",
        url="http://x:8000",
        mode=MODE_SYNTHETIC,
        dry_run=True,
    )
    assert result.dry_run is True
    assert result.shape == "aa-1k"
    assert len(result.commands) == 2  # one per concurrency
    # The .cmd capture file is written even on dry-run.
    assert (result.cell_dir / "commands" / "aa-sweep.cmd").is_file()


def test_run_cell_aa_unknown_shape(tmp_path: Path):
    import pytest

    cell = _cell()
    with pytest.raises(ValueError):
        run_cell_aa(
            cell,
            tmp_path,
            shape_name="aa-nope",
            model="m",
            url="http://x:8000",
            dry_run=True,
        )


# --- CLI dispatch ---------------------------------------------------------


def test_cli_cell_run_aa_dry_run(tmp_path: Path, capsys):
    from tools.perf_tune_report.perf_tune_report_cli import build_parser

    campaign = tmp_path / "20260529T000000Z-aa"
    (campaign / "cells").mkdir(parents=True)
    config = {
        "name": "aa",
        "cells": [
            {
                "cell_id": "aa-10k",
                "model": "m",
                "hardware": "B200",
                "quant": "NVFP4",
                "tensor_parallel": 8,
                "parallel_strategy": "TP",
                "mtp": False,
                "max_num_batched_tokens": 4096,
                "concurrencies": [1, 10],
                "aa": {
                    "model": "m",
                    "url": "http://x:8000",
                    "shape": "aa-10k",
                    "mode": "synthetic",
                },
            }
        ],
    }
    import yaml

    (campaign / "config.yaml").write_text(yaml.safe_dump(config))

    parser = build_parser()
    args = parser.parse_args(
        [
            "cell_run",
            "--campaign",
            str(campaign),
            "--cell",
            "aa-10k",
            "--backend",
            "aa",
            "--dry-run",
            "--json",
        ]
    )
    rc = args.func(args)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["backend"] == "aa"
    assert out["aa_shape"] == "aa-10k"
    assert out["dry_run"] is True
    assert len(out["commands"]) == 2
