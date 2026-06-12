"""Unit tests for the bench-all-workloads importer (import_workloads verb)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.perf_tune_report.importers.workloads import import_workloads
from tools.perf_tune_report.perf_tune_report_cli import main

# A representative vllm-bench-serve output block (no token totals -> mean ISL/OSL falls
# back to the nominal shape from bench-workloads.json).
_BENCH = """============ Serving Benchmark Result ============
Successful requests:                     64
Benchmark duration (s):                  63.34
Request throughput (req/s):              1.01
Output token throughput (tok/s):         517.31
Total token throughput (tok/s):          4659.51
Median TTFT (ms):                        285.48
Median TPOT (ms):                        14.41
"""


def _bench_dir(tmp_path: Path, files: dict[str, str], workloads: list[dict]) -> Path:
    bench = tmp_path / "bench-out"
    bench.mkdir()
    for name, text in files.items():
        (bench / name).write_text(text)
    (bench / "bench-workloads.json").write_text(
        json.dumps({"schema": "bench_workloads_v1", "workloads": workloads})
    )
    return bench


def _campaign(tmp_path: Path, name: str = "camp") -> Path:
    camp = tmp_path / name
    (camp / "cells").mkdir(parents=True)
    return camp


def test_import_workloads_tags_dataset_and_isl_osl(tmp_path):
    bench = _bench_dir(
        tmp_path,
        {"sonnet-c1.txt": _BENCH, "sonnet-c64.txt": _BENCH, "random-c1.txt": _BENCH},
        [
            {"tag": "sonnet", "dataset": "sonnet", "isl": 512, "osl": 256},
            {"tag": "random", "dataset": "random", "isl": 1024, "osl": 512},
        ],
    )
    camp = _campaign(tmp_path)
    res = import_workloads(
        bench, camp, model="GLM-5.1", hardware="GB300", tensor_parallel=4,
        kv_cache_dtype="fp8_e4m3", image="infr/vllm:v2.12.3", max_num_batched_tokens=2048,
    )
    assert res.n_cells == 2
    assert set(res.tags) == {"sonnet", "random"}
    assert res.n_rows == 3
    sonnet = json.loads((camp / "cells" / "sonnet" / "normalized.json").read_text())
    assert {r["concurrency"] for r in sonnet} == {1, 64}
    r = sonnet[0]
    assert r["dataset"] == "sonnet"
    # No token totals in the bench output -> nominal ISL/OSL from bench-workloads.json.
    assert r["mean_input_tokens"] == 512.0
    assert r["mean_output_tokens"] == 256.0
    assert r["output_tps_per_gpu"] == pytest.approx(517.31 / 4)
    assert r["total_tps_per_gpu"] == pytest.approx(4659.51 / 4)
    assert r["output_tps_per_user"] == pytest.approx(1000.0 / 14.41)
    assert r["bench_backend"] == "openai"
    assert r["kv_cache_dtype"] == "fp8_e4m3"
    assert r["image"] == "infr/vllm:v2.12.3"
    assert r["backend"] == "vllm-sweep"


def test_import_workloads_measured_mean_overrides_nominal(tmp_path):
    # When the bench output carries token totals, the MEASURED mean wins over the nominal.
    bench_text = _BENCH + "Total input tokens:                      6400\n" \
                          "Total generated tokens:                  3200\n"
    bench = _bench_dir(
        tmp_path, {"random-c1.txt": bench_text},
        [{"tag": "random", "dataset": "random", "isl": 1024, "osl": 512}],
    )
    camp = _campaign(tmp_path)
    import_workloads(bench, camp, model="M", hardware="GB300", tensor_parallel=4)
    r = json.loads((camp / "cells" / "random" / "normalized.json").read_text())[0]
    assert r["mean_input_tokens"] == pytest.approx(6400 / 64)   # measured, not 1024 nominal
    assert r["mean_output_tokens"] == pytest.approx(3200 / 64)


def test_import_workloads_multihyphen_tag(tmp_path):
    # aa-1k tag (multi-hyphen) must resolve to tag=aa-1k, c=1 (greedy anchor on last -c<d>).
    bench = _bench_dir(
        tmp_path, {"aa-1k-c1.txt": _BENCH},
        [{"tag": "aa-1k", "dataset": "aa", "isl": 1000, "osl": 1000}],
    )
    camp = _campaign(tmp_path)
    res = import_workloads(bench, camp, model="M", hardware="GB300", tensor_parallel=4)
    assert res.tags == ["aa-1k"]
    rows = json.loads((camp / "cells" / "aa-1k" / "normalized.json").read_text())
    assert rows[0]["dataset"] == "aa" and rows[0]["concurrency"] == 1


def test_import_workloads_dry_run_writes_nothing(tmp_path):
    bench = _bench_dir(
        tmp_path, {"sonnet-c1.txt": _BENCH},
        [{"tag": "sonnet", "dataset": "sonnet", "isl": 512, "osl": 256}],
    )
    camp = _campaign(tmp_path)
    res = import_workloads(bench, camp, model="M", hardware="GB300", tensor_parallel=4,
                           dry_run=True)
    assert res.n_cells == 1 and res.n_rows == 1
    assert not (camp / "cells" / "sonnet").exists()


def test_import_workloads_missing_workloads_json_defaults_dataset_to_tag(tmp_path):
    # No bench-workloads.json -> dataset defaults to the tag, ISL/OSL None (still imports).
    bench = tmp_path / "bench-out"
    bench.mkdir()
    (bench / "sonnet-c1.txt").write_text(_BENCH)
    camp = _campaign(tmp_path)
    import_workloads(bench, camp, model="M", hardware="GB300", tensor_parallel=4)
    r = json.loads((camp / "cells" / "sonnet" / "normalized.json").read_text())[0]
    assert r["dataset"] == "sonnet"          # tag is the fallback dataset
    assert r["mean_input_tokens"] is None    # no nominal, no totals


def test_cli_import_workloads(tmp_path, capsys):
    bench = _bench_dir(
        tmp_path, {"sonnet-c1.txt": _BENCH},
        [{"tag": "sonnet", "dataset": "sonnet", "isl": 512, "osl": 256}],
    )
    camp = _campaign(tmp_path, "campaigns/mycamp")
    rc = main([
        "import_workloads", "--bench-dir", str(bench), "--campaign", str(camp),
        "--model", "GLM-5.1", "--hardware", "GB300", "--tensor-parallel", "4",
        "--kv-cache-dtype", "fp8_e4m3", "--image", "infr/vllm:v2.12.3", "--json",
    ])
    assert rc == 0
    env = json.loads(capsys.readouterr().out)
    assert env["verb"] == "import_workloads"
    assert env["n_cells"] == 1 and env["tags"] == ["sonnet"]
