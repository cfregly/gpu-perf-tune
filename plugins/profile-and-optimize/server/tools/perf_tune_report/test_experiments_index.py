"""Unit tests for the perf_tune_report experiments_index verb (cross-experiment index)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.perf_tune_report.experiments_index import (
    build_index,
    build_index_row,
    build_inventory,
    enumerate_published_campaign_ids,
    find_bundle_dirs,
    render_index_md,
    render_inventory_md,
    write_index,
    write_inventory,
)
from tools.perf_tune_report.perf_tune_report_cli import main
from tools.perf_tune_report.schema import AtlasCell, write_jsonl


class _FakeS3List:
    """Stub S3 client returning a fixed set of campaign_v1 keys (paginated)."""

    def __init__(self, campaign_ids, page_size=2):
        self._keys = [
            f"perflake/perf-report/campaign_v1/dt=2026-05-31/campaign={c}/part-0.parquet"
            for c in campaign_ids
        ]
        self._page = page_size

    def list_objects_v2(self, **kw):
        token = int(kw.get("ContinuationToken", 0))
        chunk = self._keys[token:token + self._page]
        nxt = token + self._page
        more = nxt < len(self._keys)
        return {
            "Contents": [{"Key": k} for k in chunk],
            "IsTruncated": more,
            "NextContinuationToken": str(nxt) if more else None,
        }


def test_enumerate_published_campaign_ids_paginates():
    ids = enumerate_published_campaign_ids(
        cfg=None, bucket="perf-lake",
        s3_client_factory=lambda _cfg: _FakeS3List(["a-20260531T010000Z", "b-20260531T020000Z",
                                                    "c-20260531T030000Z"]),
    )
    assert ids == {"a-20260531T010000Z", "b-20260531T020000Z", "c-20260531T030000Z"}


def test_build_index_marks_published_to_lake(tmp_path: Path):
    _stage(tmp_path, "pub-20260531T010000Z", rows=[_row()])
    _stage(tmp_path, "unpub-20260531T020000Z", rows=[_row()])
    rows = build_index(tmp_path, published={"pub-20260531T010000Z"})
    by_id = {r["campaign_id"]: r for r in rows}
    assert by_id["pub-20260531T010000Z"]["published_to_lake"] is True
    assert by_id["unpub-20260531T020000Z"]["published_to_lake"] is False


def test_build_index_published_defaults_false_when_not_checked(tmp_path: Path):
    _stage(tmp_path, "x-20260531T010000Z", rows=[_row()])
    rows = build_index(tmp_path)  # published=None -> not checked
    assert rows[0]["published_to_lake"] is False


def _stage(campaigns_root: Path, name: str, *, family: str = "", experiment_id: str | None = None,
           rows: list[AtlasCell] | None = None, rendered: bool = True) -> Path:
    d = campaigns_root / name
    d.mkdir(parents=True)
    src = ["# Campaign", "", "- captured_at: x", "- config: /tmp/x.yaml"]
    if experiment_id:
        src.append(f"- experiment_id: {experiment_id}")
    if family:
        src.append(f"- family: {family}")
    (d / "SOURCE.md").write_text("\n".join(src) + "\n")
    if rows is not None:
        write_jsonl(rows, d / "atlas.jsonl")
    if rendered:
        (d / "report_status.json").write_text(json.dumps(
            {"sol_complete": True, "plot_ready_points": 1, "omitted_pages": [],
             "dcgm_grounded": True, "focus": "throughput", "sol_rigor": "L3"}))
    return d


def _row(**kw) -> AtlasCell:
    base = dict(cell_id="c1", model="GLM-5.1-NVFP4", hardware="B200", quant="NVFP4",
                tensor_parallel=8, parallel_strategy="TP", mtp=False,
                max_num_batched_tokens=8192, concurrency=1, status="full",
                ttft_avg_ms=120.0, request_throughput_avg=0.2, output_tps_per_user=70.0,
                output_tps_per_gpu=400.0, tpot_median_ms=13.0, backend="vllm-sweep")
    base.update(kw)
    return AtlasCell(**base)


def test_build_index_row_reads_join_keys_and_headline(tmp_path: Path):
    d = _stage(tmp_path, "glm51-nvfp4kv-ab-20260531T120000Z", family="nvfp4-kv",
               experiment_id="glm51-nvfp4kv-ab-20260531T120000Z",
               rows=[_row(output_tps_per_gpu=400.0, ttft_avg_ms=120.0),
                     _row(concurrency=8, output_tps_per_gpu=900.0, ttft_avg_ms=200.0)])
    r = build_index_row(d)
    assert r["experiment_id"] == "glm51-nvfp4kv-ab-20260531T120000Z"
    assert r["family"] == "nvfp4-kv"
    assert r["focus"] == "throughput"
    assert r["sol_rigor"] == "L3"
    assert r["peak_output_tps_per_gpu"] == 900.0
    assert r["min_ttft_ms"] == 120.0
    assert r["models"] == "GLM-5.1-NVFP4"


def test_build_index_defaults_experiment_id_to_campaign_id(tmp_path: Path):
    d = _stage(tmp_path, "legacy-20260525T010101Z", rows=[_row()])
    r = build_index_row(d)
    assert r["experiment_id"] == "legacy-20260525T010101Z"
    assert r["family"] == ""


def test_build_index_enumerates_and_sorts(tmp_path: Path):
    _stage(tmp_path, "b-20260531T020000Z", rows=[_row()])
    _stage(tmp_path, "a-20260531T010000Z", rows=[_row()])
    (tmp_path / "not-a-campaign").mkdir()  # ignored (no atlas/config/SOURCE)
    rows = build_index(tmp_path)
    assert [r["campaign_id"] for r in rows] == [
        "a-20260531T010000Z", "b-20260531T020000Z"]


def test_write_index_emits_jsonl_and_md(tmp_path: Path):
    _stage(tmp_path, "x-20260531T030000Z", family="deepep", rows=[_row()])
    rows = build_index(tmp_path)
    out = write_index(rows, tmp_path / "out")
    assert Path(out["jsonl"]).is_file()
    assert Path(out["md"]).is_file()
    lines = Path(out["jsonl"]).read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["family"] == "deepep"
    assert "Experiments index" in Path(out["md"]).read_text()


def test_cli_experiments_index_family_filter(tmp_path: Path, capsys):
    _stage(tmp_path, "kv-20260531T040000Z", family="nvfp4-kv", rows=[_row()])
    _stage(tmp_path, "ep-20260531T050000Z", family="deepep", rows=[_row()])
    rc = main(["experiments_index", "--campaigns-dir", str(tmp_path),
               "--family", "nvfp4-kv", "--out", str(tmp_path / "o"), "--json"])
    assert rc == 0
    env = json.loads(capsys.readouterr().out)
    assert env["campaign_count"] == 1
    rows = [json.loads(l) for l in
            (tmp_path / "o" / "experiments-index.jsonl").read_text().splitlines()]
    assert rows[0]["family"] == "nvfp4-kv"


# --- experiment_inventory (canonical count: campaigns + bundles) ------------------------

def _stage_bundle(root: Path, name: str, *, marker: str = "SOURCE.md") -> Path:
    """A run-id-stamped evidence bundle under a deploy-bundle-style tree."""
    d = root / name
    d.mkdir(parents=True)
    (d / marker).write_text("# bundle\n")
    return d


def test_find_bundle_dirs_runid_only(tmp_path: Path):
    arts = tmp_path / "experiments" / "artifacts" / "fam"
    probes = tmp_path / "cluster-probes"
    _stage_bundle(arts, "glm51-ab-20260531T010000Z")
    _stage_bundle(probes, "glm51-probe-20260531T020000Z", marker="summary.md")
    _stage_bundle(probes, "no-runid-here")  # excluded: no UTC stamp
    found = {d.name for d in find_bundle_dirs([tmp_path])}
    assert found == {"glm51-ab-20260531T010000Z", "glm51-probe-20260531T020000Z"}


def test_build_inventory_unions_and_dedupes_by_runid(tmp_path: Path):
    campaigns = tmp_path / "campaigns"
    campaigns.mkdir()
    # one campaign that ALSO exists as a bundle (same run-id -> one experiment)
    _stage(campaigns, "shared-20260531T010000Z", family="nvfp4-kv", rows=[_row()])
    # one campaign with no bundle
    _stage(campaigns, "camp-only-20260531T020000Z", family="deepep", rows=[_row()])
    deploy = tmp_path / "x-deploy"
    _stage_bundle(deploy / "cluster-probes", "shared-20260531T010000Z")  # dedupes w/ campaign
    _stage_bundle(deploy / "cluster-probes", "bundle-only-20260531T030000Z")  # bundle-only

    inv = build_inventory(campaigns, [deploy])
    assert inv["campaign_count"] == 2
    assert inv["bundle_count"] == 2
    # union of {2 campaign ids} + {2 bundle ids, 1 shared} = 3 distinct experiments
    assert inv["total_experiments"] == 3
    assert inv["bundle_only_count"] == 1
    assert inv["bundle_only"] == ["bundle-only-20260531T030000Z"]
    assert inv["by_family"] == {"deepep": 1, "nvfp4-kv": 1}
    assert inv["by_model"].get("GLM-5.1-NVFP4") == 2
    # md headline renders the unified count
    md = render_inventory_md(inv)
    assert "Total distinct experiments: 3" in md


def test_build_inventory_campaigns_only_when_no_bundle_root(tmp_path: Path):
    campaigns = tmp_path / "campaigns"
    campaigns.mkdir()
    _stage(campaigns, "a-20260531T010000Z", rows=[_row()])
    inv = build_inventory(campaigns, None)
    assert inv["total_experiments"] == inv["campaign_count"] == 1
    assert inv["bundle_count"] == 0


def test_write_inventory_emits_md_and_json(tmp_path: Path):
    campaigns = tmp_path / "campaigns"
    campaigns.mkdir()
    _stage(campaigns, "a-20260531T010000Z", family="quant-ab", rows=[_row()])
    inv = build_inventory(campaigns, None)
    out = write_inventory(inv, tmp_path / "out")
    assert Path(out["md"]).is_file() and Path(out["json"]).is_file()
    summary = json.loads(Path(out["json"]).read_text())
    assert summary["total_experiments"] == 1
    assert "campaigns" not in summary  # heavy per-campaign list excluded from the json headline


def test_cli_experiment_inventory(tmp_path: Path, capsys):
    campaigns = tmp_path / "campaigns"
    campaigns.mkdir()
    _stage(campaigns, "shared-20260531T010000Z", rows=[_row()])
    deploy = tmp_path / "x-deploy"
    _stage_bundle(deploy / "cluster-probes", "shared-20260531T010000Z")
    _stage_bundle(deploy / "experiments" / "artifacts", "extra-20260531T040000Z")
    rc = main(["experiment_inventory", "--campaigns-dir", str(campaigns),
               "--bundle-root", str(deploy), "--out", str(tmp_path / "o"), "--json"])
    assert rc == 0
    env = json.loads(capsys.readouterr().out)
    assert env["total_experiments"] == 2  # shared (deduped) + extra-bundle
    assert env["campaign_count"] == 1
    assert env["bundle_count"] == 2
    assert env["bundle_only_count"] == 1
    assert "Total distinct experiments: 2" in (tmp_path / "o" / "EXPERIMENT-INVENTORY.md").read_text()
