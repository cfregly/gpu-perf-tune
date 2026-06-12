"""Tests for the zymtrace_kernels declared-coverage contract.

Mirrors the 8 Phase D validation scenarios from the plan, but as proper
pytest cases so the contract is enforced in CI not just /tmp scripts:

1. positive_path      -- declared + 5 valid TSVs -> kernels.json emitted
2. declared_empty     -- declared + 0-byte TSV -> ZymtraceTSVMissing
3. declared_malformed -- declared + bad TSV -> ZymtraceTSVMalformed
4. no_manifest_skip   -- no manifest -> silent skip, no kernels.json
5. legacy_bundle_skip -- 0-byte TSVs but NO manifest -> silent skip
6. renderer_3page     -- kernels.json present -> 3-page PDF
7. renderer_2page     -- no kernels.json -> 2-page PDF (no error)
8. renderer_malformed -- kernels.json malformed -> KernelsJsonMalformed
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.perf_tune_report.importers.zymtrace_kernels import (
    ZymtraceTSVMalformed,
    ZymtraceTSVMissing,
    import_zymtrace_kernels,
)
from tools.perf_tune_report.renderer.render_report import (
    KernelsJsonMalformed,
    discover_kernels_payloads,
    render_report,
)
from tools.perf_tune_report.schema import (
    BACKEND_VLLM_SWEEP,
    STATUS_FULL,
    AtlasCell,
)


# ---------------------------------------------------------------------------
# Synthetic-bundle helpers
# ---------------------------------------------------------------------------

_VALID_TSVS = {
    "kernel-class.tsv": (
        "event_kind\tkind\tsamples\n"
        "cuda\tnative\t1000\n"
        "cuda\tcuda\t500\n"
    ),
    "top-gpu-frames.tsv": (
        "kernel\tsamples\n"
        "multimem_all_reduce_kernel<bfloat16>\t558\n"
        "bmm_E2m1E2m1_Fp32_sm100f\t328\n"
        "triton_poi_fused_mul_silu_slice_0\t442\n"
        "cublasLt_splitKreduce_kernel\t944\n"
        "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512\t180\n"
    ),
    "per-gpu.tsv": (
        "gpu_name\tgpu_uuid\tsamples\n"
        "NVIDIA B200\t507bd556-0c4d-0f4e-1b63-6b0fa6698c41\t35961\n"
        "NVIDIA B200\t92f929f2-1fea-72bb-7d96-834eb0f4e9c8\t35917\n"
    ),
    "per-category.tsv": (
        "category\tsamples\n"
        "cuBLAS\t119199\n"
        "Triton-fused\t61950\n"
        "NCCL\t16510\n"
        "BMM-NVFP4\t19212\n"
        "FMHA\t9226\n"
    ),
    "top-python-during-cuda.tsv": (
        "python_frame\tsamples\n"
        "vllm.engine.async_llm_engine.AsyncLLMEngine._run_engine\t12345\n"
        "torch._dynamo.eval_frame._fn\t6789\n"
    ),
}

_VALID_MANIFEST = {
    "schema_version": 1,
    "captured_sources": ["zymtrace"],
    "captured_at": "2026-05-26T05:50:00Z",
    "captured_by": "test_zymtrace_kernels@pytest",
    "cluster": "synthetic",
    "pod_name": "test-pod-1",
}


def _make_bundle(root: Path, *, manifest: dict | None, tsvs: dict[str, str | None]) -> Path:
    """Build a bundle directory with optional manifest + arbitrary TSV contents.

    ``tsvs`` value of None means do NOT create that file at all (vs ""
    which creates a 0-byte file).
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "raw").mkdir(exist_ok=True)
    (root / "raw" / "sweep-c1.txt").write_text("Serving Benchmark Result\n")
    if manifest is not None:
        (root / "capture_sources.json").write_text(json.dumps(manifest, indent=2))
    if tsvs:
        (root / "zymtrace").mkdir(exist_ok=True)
        for name, content in tsvs.items():
            if content is None:
                continue
            (root / "zymtrace" / name).write_text(content)
    return root


def _mk_atlas(campaign_dir: Path, cell_id: str) -> Path:
    campaign_dir.mkdir(parents=True, exist_ok=True)
    cell = AtlasCell(
        cell_id=cell_id,
        model="kimi-k2.6",
        hardware="B200",
        quant="NVFP4",
        tensor_parallel=8,
        parallel_strategy="TP",
        mtp=False,
        max_num_batched_tokens=4096,
        concurrency=1,
        status=STATUS_FULL,
        ttft_avg_ms=100.0,
        request_throughput_avg=0.5,
        output_tps_per_user=20.0,
        output_tps_per_gpu=12.5,
        backend=BACKEND_VLLM_SWEEP,
        raw_path="/dev/null",
        captured_at="2026-05-26T05:50:00Z",
        notes="test",
        extra={},
    )
    atlas_path = campaign_dir / "atlas.jsonl"
    atlas_path.write_text(json.dumps(dataclasses.asdict(cell)) + "\n")
    return atlas_path


# ---------------------------------------------------------------------------
# Importer-side tests (scenarios 1-5)
# ---------------------------------------------------------------------------


def test_zymtrace_positive_path(tmp_path):
    """S1: manifest declares zymtrace + all 5 TSVs valid -> kernels.json emitted."""
    bundle = _make_bundle(tmp_path / "bundle", manifest=_VALID_MANIFEST, tsvs=_VALID_TSVS)
    cell_dir = tmp_path / "campaign" / "cells" / "cell-1"

    result = import_zymtrace_kernels(bundle, cell_dir)

    assert result.kernels_json_path is not None
    assert result.kernels_json_path.is_file()
    assert result.skipped_reason is None
    assert result.top_kernel_count == 5
    assert result.category_count == 5
    assert result.gpu_count == 2

    payload = json.loads(result.kernels_json_path.read_text())
    assert payload["schema_version"] == 1
    assert "zymtrace" in payload["captured_sources"]
    assert payload["per_category"]["cuBLAS"] == 119199
    # category derivation: multimem_all_reduce_kernel -> NCCL
    assert any(
        k["name"].startswith("multimem_all_reduce") and k["category"] == "NCCL"
        for k in payload["top_kernels"]
    )
    # category derivation: bmm_E2m1E2m1 -> BMM-NVFP4
    assert any(
        k["name"].startswith("bmm_E2m1") and k["category"] == "BMM-NVFP4"
        for k in payload["top_kernels"]
    )


def test_zymtrace_declared_but_empty_tsv_raises(tmp_path):
    """S2: declared + per-gpu.tsv is 0 bytes -> ZymtraceTSVMissing(reason=empty)."""
    tsvs = dict(_VALID_TSVS)
    tsvs["per-gpu.tsv"] = ""  # truncate to 0 bytes
    bundle = _make_bundle(tmp_path / "bundle", manifest=_VALID_MANIFEST, tsvs=tsvs)
    cell_dir = tmp_path / "campaign" / "cells" / "cell-2"

    with pytest.raises(ZymtraceTSVMissing) as ei:
        import_zymtrace_kernels(bundle, cell_dir)
    assert ei.value.reason == "empty"
    assert "per-gpu.tsv" in str(ei.value.path)


def test_zymtrace_declared_but_absent_tsv_raises(tmp_path):
    """S2 variant: declared + per-category.tsv missing entirely -> absent."""
    tsvs = dict(_VALID_TSVS)
    tsvs["per-category.tsv"] = None  # don't write the file
    bundle = _make_bundle(tmp_path / "bundle", manifest=_VALID_MANIFEST, tsvs=tsvs)
    cell_dir = tmp_path / "campaign" / "cells" / "cell-2b"

    with pytest.raises(ZymtraceTSVMissing) as ei:
        import_zymtrace_kernels(bundle, cell_dir)
    assert ei.value.reason == "absent"
    assert "per-category.tsv" in str(ei.value.path)


def test_zymtrace_declared_but_malformed_raises(tmp_path):
    """S3: declared + top-gpu-frames.tsv has wrong column count -> ZymtraceTSVMalformed."""
    tsvs = dict(_VALID_TSVS)
    tsvs["top-gpu-frames.tsv"] = "kernel\tsamples\nthis_is_one_cell_only\n"
    bundle = _make_bundle(tmp_path / "bundle", manifest=_VALID_MANIFEST, tsvs=tsvs)
    cell_dir = tmp_path / "campaign" / "cells" / "cell-3"

    with pytest.raises(ZymtraceTSVMalformed):
        import_zymtrace_kernels(bundle, cell_dir)


def test_zymtrace_empty_tsv_message_names_ingest_lag(tmp_path):
    """An empty (0-byte) TSV stays a loud ZymtraceTSVMissing, but the message now
    names ClickHouse INGEST LAG + the recapture path so an empty-now is not
    mistaken for a permanent gap. reason stays 'empty' (fail-fast unchanged)."""
    tsvs = dict(_VALID_TSVS)
    tsvs["per-gpu.tsv"] = ""  # 0 bytes -> ingest-lag shape
    bundle = _make_bundle(tmp_path / "bundle", manifest=_VALID_MANIFEST, tsvs=tsvs)
    cell_dir = tmp_path / "campaign" / "cells" / "cell-lag1"

    with pytest.raises(ZymtraceTSVMissing) as ei:
        import_zymtrace_kernels(bundle, cell_dir)
    assert ei.value.reason == "empty"  # fail-fast classification unchanged
    msg = str(ei.value)
    assert "INGEST LAG" in msg
    assert "zymtrace-query-hygiene.md" in msg
    assert "RE-CAPTURE" in msg


def test_zymtrace_header_only_tsv_names_ingest_lag(tmp_path):
    """A header-only TSV (the 'query returned headers but no rows' shape ingest lag
    produces) stays a loud ZymtraceTSVMalformed, with the ingest-lag hint."""
    tsvs = dict(_VALID_TSVS)
    tsvs["kernel-class.tsv"] = "event_kind\tkind\tsamples\n"  # header only, no rows
    bundle = _make_bundle(tmp_path / "bundle", manifest=_VALID_MANIFEST, tsvs=tsvs)
    cell_dir = tmp_path / "campaign" / "cells" / "cell-lag2"

    with pytest.raises(ZymtraceTSVMalformed) as ei:
        import_zymtrace_kernels(bundle, cell_dir)
    assert ei.value.reason.startswith("header-only")
    assert "INGEST LAG" in str(ei.value)


def test_zymtrace_absent_tsv_message_no_ingest_lag_hint(tmp_path):
    """An ABSENT file (capture never wrote it) is NOT ingest lag -> no lag hint,
    so the two failure shapes stay distinguishable in the message."""
    tsvs = dict(_VALID_TSVS)
    tsvs["per-category.tsv"] = None  # file absent
    bundle = _make_bundle(tmp_path / "bundle", manifest=_VALID_MANIFEST, tsvs=tsvs)
    cell_dir = tmp_path / "campaign" / "cells" / "cell-lag3"

    with pytest.raises(ZymtraceTSVMissing) as ei:
        import_zymtrace_kernels(bundle, cell_dir)
    assert ei.value.reason == "absent"
    assert "INGEST LAG" not in str(ei.value)


def test_zymtrace_no_manifest_silent_skip(tmp_path):
    """S4: no manifest at all -> silent skip, no kernels.json emitted."""
    bundle = _make_bundle(tmp_path / "bundle", manifest=None, tsvs={})
    cell_dir = tmp_path / "campaign" / "cells" / "cell-4"

    result = import_zymtrace_kernels(bundle, cell_dir)

    assert result.kernels_json_path is None
    assert result.skipped_reason is not None
    assert "absent" in result.skipped_reason or "does not declare" in result.skipped_reason


def test_zymtrace_legacy_bundle_skip(tmp_path):
    """S5: legacy bundle (0-byte TSVs + NO manifest) -> silent skip.

    This is the case for every glm51-* historical bundle: their zymtrace/
    subdir has 0-byte TSVs from the broken-curl-port-9123 capture era but
    they never wrote a manifest. The importer must NOT retroactively brick
    them; their capture history is already documented in Phase 5 SOURCE.md.
    """
    bundle = _make_bundle(
        tmp_path / "bundle",
        manifest=None,
        tsvs={name: "" for name in _VALID_TSVS},  # all 0 bytes
    )
    cell_dir = tmp_path / "campaign" / "cells" / "cell-5"

    result = import_zymtrace_kernels(bundle, cell_dir)

    assert result.kernels_json_path is None
    assert "absent" in result.skipped_reason or "does not declare" in result.skipped_reason


def test_zymtrace_manifest_present_but_zymtrace_not_in_sources(tmp_path):
    """Manifest declares OTHER sources (e.g. only nsys) -> skip silently.

    Forward-compat check: when nsys capture lands as a sibling source, a
    bundle that declares ["nsys"] but not ["zymtrace"] should not be
    required to ship zymtrace TSVs.
    """
    manifest_nsys_only = dict(_VALID_MANIFEST)
    manifest_nsys_only["captured_sources"] = ["nsys"]
    bundle = _make_bundle(tmp_path / "bundle", manifest=manifest_nsys_only, tsvs={})
    cell_dir = tmp_path / "campaign" / "cells" / "cell-6"

    result = import_zymtrace_kernels(bundle, cell_dir)
    assert result.kernels_json_path is None


def test_zymtrace_dry_run_does_not_write(tmp_path):
    """dry_run=True parses + validates but emits no file."""
    bundle = _make_bundle(tmp_path / "bundle", manifest=_VALID_MANIFEST, tsvs=_VALID_TSVS)
    cell_dir = tmp_path / "campaign" / "cells" / "cell-7"

    result = import_zymtrace_kernels(bundle, cell_dir, dry_run=True)
    assert result.kernels_json_path is not None  # returns the would-be path
    assert not result.kernels_json_path.exists()  # but didn't write
    assert result.top_kernel_count == 5


# ---------------------------------------------------------------------------
# Renderer-side tests (scenarios 6-8)
# ---------------------------------------------------------------------------


def test_renderer_three_page_when_kernels_json_present(tmp_path):
    """S6: campaign with kernels.json -> 3-page PDF."""
    bundle = _make_bundle(tmp_path / "bundle", manifest=_VALID_MANIFEST, tsvs=_VALID_TSVS)
    campaign = tmp_path / "campaign"
    cell_dir = campaign / "cells" / "cell-1"
    import_zymtrace_kernels(bundle, cell_dir)

    atlas = _mk_atlas(campaign, "cell-1")
    out_pdf = tmp_path / "out.pdf"
    render_report(atlas, out_pdf, title="3-page test")

    assert out_pdf.is_file()
    assert out_pdf.stat().st_size > 5000
    pdf_bytes = out_pdf.read_bytes()
    # PdfPages writes "/Type /Page" (with trailing space or newline) per page.
    page_count = pdf_bytes.count(b"/Type /Page\n") + pdf_bytes.count(b"/Type /Page ")
    assert page_count >= 3


def test_renderer_two_page_when_no_kernels_json(tmp_path):
    """S7: campaign without kernels.json -> pages 1+2 + a completeness page."""
    campaign = tmp_path / "campaign"
    (campaign / "cells" / "cell-1").mkdir(parents=True)
    atlas = _mk_atlas(campaign, "cell-1")
    out_pdf = tmp_path / "out.pdf"
    render_report(atlas, out_pdf, title="2-page test")

    assert out_pdf.is_file()
    pdf_bytes = out_pdf.read_bytes()
    page_count = pdf_bytes.count(b"/Type /Page\n") + pdf_bytes.count(b"/Type /Page ")
    # The conditional pages are no longer silently dropped: a loud
    # completeness page is appended, so 2 data pages + 2b (TPM) + 1
    # completeness = 4.
    assert page_count == 4


def test_renderer_malformed_kernels_json_raises(tmp_path):
    """S8: kernels.json present but missing required fields -> KernelsJsonMalformed."""
    campaign = tmp_path / "campaign"
    cell_dir = campaign / "cells" / "cell-1"
    cell_dir.mkdir(parents=True)
    (cell_dir / "kernels.json").write_text(
        '{"schema_version": 1, "captured_sources": ["zymtrace"]}'  # missing fields
    )
    atlas = _mk_atlas(campaign, "cell-1")
    out_pdf = tmp_path / "out.pdf"

    with pytest.raises(KernelsJsonMalformed) as ei:
        render_report(atlas, out_pdf, title="malformed test")
    assert "missing required fields" in ei.value.reason


def test_renderer_invalid_json_kernels_raises(tmp_path):
    """kernels.json is not valid JSON -> KernelsJsonMalformed (parse error)."""
    campaign = tmp_path / "campaign"
    cell_dir = campaign / "cells" / "cell-1"
    cell_dir.mkdir(parents=True)
    (cell_dir / "kernels.json").write_text("{this is not json")
    atlas = _mk_atlas(campaign, "cell-1")
    out_pdf = tmp_path / "out.pdf"

    with pytest.raises(KernelsJsonMalformed):
        render_report(atlas, out_pdf, title="invalid-json test")


def test_discover_kernels_payloads_empty_when_no_cells_dir(tmp_path):
    """discover_kernels_payloads returns empty OrderedDict when no cells/ tree."""
    result = discover_kernels_payloads(tmp_path)
    assert len(result) == 0
