"""Perf-report CLI: 17 verbs that build a campaign atlas + render the PDF + publish to the data lake.

Verbs (mirrored as MCP tools ``perf_tune_report_<verb>`` via
[`mcp_surface.py`](../../mcp_surface.py)):

- ``campaign_init``    -- scaffold ``campaigns/<UTC>-<slug>/`` from a config
- ``cell_run``         -- run one cell via ``vllm-sweep`` / ``aiperf`` / ``aa`` backend (ack-gated)
- ``atlas_aggregate``  -- union per-cell normalized.json -> ``atlas.jsonl``
- ``report_render``    -- render the multi-page PDF via PdfPages
- ``report_smoke``     -- render PDF from the bundled synthetic fixture (no cluster)
- ``publish_to_lake``  -- write atlas + campaign + SoL + TPM as Parquet to S3 (the perf lake BYOB lane)
- ``tpm_summary``      -- per-hardware tokens-per-minute capacity rollup for pricing (v1.35.0)
- ``import_perf_bench``-- import an inference-perf-bench bundle into a campaign cell
- ``campaign_run``     -- matrix orchestrator (drain -> helm -> bench -> aggregate -> render) (ack-gated)
- ``graph_diff``       -- diff two torch.compile dynamo/inductor log dumps
- ``kernel_profile``   -- capture an nsys per-kernel profile from a live vLLM pod (ack-gated)
- ``raw_bench_compare``-- multi-bundle vllm-bench-serve linear-comparison PDF
- ``import_nsys``      -- import an nsys cuda_gpu_kern_sum into a cell's kernels.json
- ``import_ncu``       -- import an ncu per-kernel bundle into a cell's ncu_kernels.json
- ``dcgm_correlate``   -- fold a frozen DCGM YAML into a cell's dcgm_correlation.json
- ``experiments_index``-- build a cross-experiment index over all local campaigns
- ``value_view``       -- leadership value-prop ledger (curated registry x live campaigns)

See the skill at
[`skills/inference-perf-tune-report/SKILL.md`](../../skills/inference-perf-tune-report/SKILL.md)
for the operator workflow.

Default campaigns dir: ``./campaigns/`` (operator-
relocatable via the ``PERFREPORT_CAMPAIGNS_DIR`` env var).

Added in profile-and-optimize v1.10.0. ``publish_to_lake`` verb added in v1.16.0.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from tools.perf_tune_report.aggregator import aggregate
from tools.perf_tune_report.coverage import summarize
from tools.perf_tune_report.helpers import (
    emit,
    load_yaml,
    resolve_campaign_dir,
    resolve_campaigns_dir,
    slugify,
    synthetic_fixture_path,
    utc_timestamp_slug,
)
from tools.perf_tune_report.importers import import_bundle_auto, import_perf_bench_bundle
from tools.perf_tune_report.importers import import_roofline_sweep_bundle
from tools.perf_tune_report.importers import detect_variant_ab, import_variant_ab_bundle
from tools.perf_tune_report.importers.ncu_kernels import import_ncu_kernels
from tools.perf_tune_report.importers.nsys_kernels import import_nsys_kernels
from tools.perf_tune_report.kernel_profile import (
    KernelProfileResult,
    capture_kernel_profile,
)
from tools.perf_tune_report.graph_diff import (
    GraphDiffResult,
    diff_graph_logs,
)
from tools.perf_tune_report.kernel_reproducer import scaffold_reproducer
from tools.perf_tune_report.orchestrator import CellPlan, run_campaign
from tools.perf_tune_report.runners.aa_bench import run_cell as run_cell_aa
from tools.perf_tune_report.runners.aiperf_bench import run_cell as run_cell_aiperf
from tools.perf_tune_report.runners.common import cell_config_from_dict
from tools.perf_tune_report.runners.vllm_sweep import run_cell as run_cell_vllm_sweep
from tools.perf_tune_report.schema import read_jsonl
from tools.perf_tune_report import champion_select as champion_select_lib
from tools.perf_tune_report.capture_signature import (
    build_plan as build_capture_plan,
    materialize_reuse,
)
from tools.perf_tune_report.value_view import (
    build_value_view,
    default_registry_path,
    render_markdown,
    render_report,
)


CONTRACT: dict[str, dict[str, Any]] = {
    "value_view": {
        "safety": "writes_artifacts",
        "required": (),
        "optional": ("--registry", "--out", "--format", "--title", "--gpu-hr",
                     "--campaigns-dir", "--json"),
        "json": True,
        "ack": None,
        "description": "Render the leadership value-prop ledger: join the curated "
        "value-findings.yaml registry with live perf-lake campaigns (sol_rigor + "
        "verdict tier) into a grouped DONE / IN-PROGRESS / NOT-DONE / CLOSED-NEGATIVE "
        "table. Read-only on the lake; flags any finding whose backing campaign is "
        "missing locally or ungrounded. Writes markdown to --out (or stdout).",
    },
    "portability_view": {
        "safety": "writes_artifacts",
        "required": (),
        "optional": ("--registry", "--out", "--title", "--campaigns-dir", "--json"),
        "json": True,
        "ack": None,
        "description": "Render the lever-by-model portability matrix from value-findings.yaml: "
        "rows = perf levers, columns = fleet models, cells = validated / candidate / refuted / "
        "untested, plus a per-model 'try-next' candidate list. Answers 'which of our proven "
        "levers should I try on model X' in one lookup. Writes markdown to --out (or stdout).",
    },
    "campaign_init": {
        "safety": "writes_artifacts",
        "required": ("--config",),
        "optional": ("--slug", "--experiment-id", "--family", "--evidence-bundle",
                     "--campaigns-dir", "--json"),
        "json": True,
        "ack": None,
        "description": "Scaffold a new campaign directory from a YAML config. Pass "
        "--experiment-id to make campaign_id == the evidence-bundle run-id (the "
        "single join key across bundle / cluster label / perf-lake).",
    },
    "experiments_index": {
        "safety": "writes_artifacts",
        "required": (),
        "optional": ("--family", "--out", "--include-s3", "--s3-endpoint",
                     "--s3-bucket", "--s3-access-key-file", "--s3-secret-key-file",
                     "--campaigns-dir", "--json"),
        "json": True,
        "ack": None,
        "description": "Enumerate all local campaigns into a cross-experiment index "
        "(experiments-index.jsonl + EXPERIMENTS-INDEX.md), keyed by experiment_id, so "
        "an analyst can compare effectiveness across experiments (filter by --family).",
    },
    "experiment_inventory": {
        "safety": "writes_artifacts",
        "required": (),
        "optional": ("--bundle-root", "--out", "--include-s3", "--s3-endpoint",
                     "--s3-bucket", "--s3-access-key-file", "--s3-secret-key-file",
                     "--campaigns-dir", "--json"),
        "json": True,
        "ack": None,
        "description": "Canonical experiment count: unify local perf-report campaigns with "
        "run-id-stamped evidence bundles (--bundle-root, repeatable) into ONE headline count "
        "+ per-family/model breakdown (EXPERIMENT-INVENTORY.md + experiment-inventory.json), "
        "deduped by the run-id join key. Answers 'how many experiments have we run' without "
        "the campaigns-vs-bundles ambiguity.",
    },
    "import_model_eval": {
        "safety": "writes_artifacts",
        "required": ("--results", "--campaign", "--model", "--hardware", "--quant"),
        "optional": ("--tensor-parallel", "--cell-id", "--parallel-strategy",
                     "--kv-cache-dtype", "--image", "--campaigns-dir", "--json"),
        "json": True,
        "ack": None,
        "description": "Import an lm-eval-harness results.json into a perf-report quality cell "
        "(extra.metric_kind=eval_acc + quality_metrics) so GPQA/MMLU-Pro serving quality lands "
        "in quality_v1 on publish. The cell carries no throughput (use focus: accuracy).",
    },
    "import_workloads": {
        "safety": "writes_artifacts",
        "required": ("--bench-dir", "--campaign", "--model", "--hardware", "--tensor-parallel"),
        "optional": ("--quant", "--parallel-strategy", "--max-num-batched-tokens",
                     "--kv-cache-dtype", "--image", "--cudagraph-mode",
                     "--gpu-memory-utilization", "--bench-backend", "--dry-run",
                     "--campaigns-dir", "--json"),
        "json": True,
        "ack": None,
        "description": "Import a bench-all-workloads.sh output dir (one <tag>-c<c>.txt per "
        "workload x concurrency + bench-workloads.json) into per-workload campaign cells, each "
        "row tagged with its dataset + typed ISL/OSL -- closing dataset=unknown at the source so "
        "atlas_aggregate -> publish lands the full multi-workload suite (the one-call companion "
        "to bench-all-workloads.sh --import-campaign).",
    },
    "trend_view": {
        "safety": "writes_artifacts",
        "required": (),
        "optional": ("--metric", "--concurrency", "--regression-pct", "--hardware",
                     "--out", "--campaigns-dir", "--lake-dir", "--json"),
        "json": True,
        "ack": None,
        "description": "Longitudinal (model, variant_key) perf/quality trend across campaigns: "
        "group atlas rows by the stable capture_signature variant key + concurrency, order by "
        "captured_at, flag regressions, and show the serving image (engine-version axis). "
        "Local-first; same row shape as the lake's atlas_v1.variant_key for a published-lake pull.",
    },
    "fleet_leaderboard": {
        "safety": "writes_artifacts",
        "required": (),
        "optional": ("--hardware", "--gpu-hr", "--out", "--campaigns-dir", "--json"),
        "json": True,
        "ack": None,
        "description": "Render the cross-model fleet leaderboards from local campaigns: "
        "AA-FLEET-LEADERBOARD (latency tier), THROUGHPUT-FLEET-LEADERBOARD (peak tok/s/GPU), "
        "and FLEET-MODEL-SELECTION (perf Pareto frontier: which model to pick). Auto-discovers "
        "every model's AA + roofline cells; re-run after new campaigns publish.",
    },
    "cell_run": {
        "safety": "submits_jobs",
        "required": ("--campaign", "--cell", "--backend"),
        "optional": (
            "--serve-cmd",
            "--bench-cmd",
            "--namespace",
            "--bench-pod",
            "--kube-context",
            "--endpoint-url",
            "--served-model",
            "--dataset-split",
            "--conversation-count",
            "--aa-shape",
            "--aa-mode",
            "--request-count",
            "--dry-run",
            "--campaigns-dir",
            "--json",
        ),
        "json": True,
        "ack": "--i-understand-this-submits-jobs",
        "description": "Run one cell via vllm-sweep, aiperf, or aa backend (ack-gated).",
    },
    "atlas_aggregate": {
        "safety": "writes_artifacts",
        "required": ("--campaign",),
        "optional": ("--campaigns-dir", "--json"),
        "json": True,
        "ack": None,
        "description": "Aggregate per-cell normalized.json into atlas.jsonl + coverage summary.",
    },
    "report_render": {
        "safety": "writes_artifacts",
        "required": ("--campaign",),
        "optional": (
            "--out",
            "--title",
            "--variants-line",
            "--data-source-line",
            "--strict",
            "--allow-ungrounded",
            "--campaigns-dir",
            "--json",
        ),
        "json": True,
        "ack": None,
        "description": (
            "Render the perf-report PDF from a campaign's atlas.jsonl. Omitted "
            "SoL/kernel/DCGM pages and empty charts are surfaced loudly (why + "
            "how-to-fix) on a completeness page + report_status.json; --strict "
            "exits non-zero when SoL is incomplete or 0 plot-ready points."
        ),
    },
    "report_smoke": {
        "safety": "read_only",
        "required": (),
        "optional": ("--out", "--title", "--json"),
        "json": True,
        "ack": None,
        "description": "Render the PDF from the bundled synthetic fixture (no cluster needed).",
    },
    "publish_to_lake": {
        "safety": "writes_artifacts",
        "required": ("--campaign",),
        "optional": (
            "--s3-endpoint",
            "--s3-bucket",
            "--s3-access-key-file",
            "--s3-secret-key-file",
            "--if-exists",
            "--strict",
            "--no-strict",
            "--allow-incomplete",
            "--allow-ungrounded",
            "--dry-run",
            "--campaigns-dir",
            "--json",
        ),
        "json": True,
        "ack": None,
        "description": (
            "Publish a campaign's atlas + provenance as Parquet to S3 "
            "(s3://perf-lake/perflake/perf-report/). Ready for downstream "
            "Iceberg registration into warehouse intake tables. Under --strict the "
            "methodology gate enforces each measured row's full descriptor + its OWN "
            "ISL/OSL shape (per-number exact shape, no smoothing -- "
            "docs/METHODOLOGY.md)."
        ),
    },
    "campaign_run": {
        "safety": "submits_jobs",
        "required": ("--config", "--campaign"),
        "optional": (
            "--continue-on-red",
            "--dry-run",
            "--campaigns-dir",
            "--json",
        ),
        "json": True,
        "ack": "--i-understand-this-submits-jobs",
        "description": (
            "Campaign-level orchestrator (v1.20.0): loops over a matrix YAML "
            "and for each cell runs the full 10-step pipeline (drain -> helm "
            "upgrade -> warmup -> bench -> zymtrace -> import -> aggregate -> "
            "render -> baseline-record -> baseline-diff). Always-resume on "
            "Ctrl-C / exception via try/finally. Fail-fast on RED verdict "
            "unless --continue-on-red is passed."
        ),
    },
    "graph_diff": {
        "safety": "writes_artifacts",
        "required": ("--side-a-log", "--side-b-log", "--output-dir"),
        "optional": (
            "--side-a-label",
            "--side-b-label",
            "--notes",
            "--dry-run",
            "--json",
        ),
        "json": True,
        "ack": None,
        "description": (
            "Diff two torch.compile dynamo+inductor log dumps and emit a "
            "structured graph_diff.json + per-graph unified diffs (v1.21.0). "
            "The operator pre-collects each side's log via "
            "TORCH_LOGS=+dynamo,+inductor,+graph_breaks; this verb is "
            "read-only on the cluster (parses local log files only). Output: "
            "side-<label>-graph<n>.fx + graph<n>.diff + graph_diff.json per "
            "the inference_graph_diff_v1 schema."
        ),
    },
    "kernel_reproducer_scaffold": {
        "safety": "writes_artifacts",
        "required": ("--kernel-name", "--header", "--output-dir"),
        "optional": (
            "--mma-m",
            "--mma-n",
            "--batch",
            "--out-dim",
            "--k",
            "--mirage-tree",
            "--arch",
            "--dry-run",
            "--json",
        ),
        "json": True,
        "ack": None,
        "description": (
            "Scaffold a standalone CUDA/CUTLASS kernel reproducer (.cu + build "
            "script) for white-box kernel debugging (v1.69.0) -- Track B of the "
            "inference-kernel-whitebox-debug skill. Emits a self-contained harness "
            "modeled on the proven GLM-5.1 linear_sm100_mpk reproducer, "
            "parameterized by the GEMM dims + mirage tree + GPU arch: it "
            "instantiates the kernel template, feeds CONTROLLED inputs (all-ones "
            "then optional real dump) and diffs vs a host GEMM. The operator "
            "transcribes the EXACT template params + tma_2d descriptor types from "
            "the codegen site (task_register.cc) into the marked block. Read-only "
            "on the cluster (writes local artifacts only)."
        ),
    },
    "kernel_profile": {
        "safety": "submits_jobs",
        "required": ("--namespace", "--pod", "--target-container", "--output-dir"),
        "optional": (
            "--sidecar-image",
            "--duration-seconds",
            "--sample",
            "--trace",
            "--sampling-frequency",
            "--vllm-pid-pattern",
            "--bundle",
            "--dry-run",
            "--json",
        ),
        "json": True,
        "ack": "--i-understand-this-submits-jobs",
        "description": (
            "Capture per-kernel CUDA profile from a live vLLM inference pod "
            "via the nsys-sidecar (v1.21.0). Uses ``kubectl debug "
            "--share-processes`` to attach an ephemeral container that runs "
            "nsys profile against the engine PID, then extracts .nsys-rep + "
            "summary CSVs into --output-dir. Optionally patches a bundle's "
            "inference_perfbench_v1.json so the renderer picks up the "
            "per-kernel breakdown. ALWAYS ack-gated: this mutates the cluster "
            "(adds an ephemeral container). Use --dry-run to print the step "
            "commands without executing."
        ),
    },
    "raw_bench_compare": {
        "safety": "writes_artifacts",
        "required": ("--manifest", "--out"),
        "optional": ("--json",),
        "json": True,
        "ack": None,
        "description": (
            "Render a multi-bundle vllm-bench-serve linear-comparison PDF "
            "from a raw_bench_compare_v1 YAML manifest (v1.24.0). Sibling "
            "to report_render: where report_render produces a faceted "
            "multi-page PDF from atlas.jsonl, raw_bench_compare overlays "
            "N bundles' per-concurrency curves onto a single chart per "
            "metric (throughput / TTFT / TPOT) + a peak-bars chart with "
            "%gain-vs-baseline + a summary table. Targeted at the "
            "'6-variant champion comparison' use case where faceting hides "
            "the linear story. Promotes the pre-v1.24.0 workshop renderers "
            "from ./campaigns workspacescripts/."
        ),
    },
    "import_perf_bench": {
        "safety": "writes_artifacts",
        "required": ("--campaign", "--bundle"),
        "optional": (
            "--cell-id",
            "--model",
            "--hardware",
            "--quant",
            "--tensor-parallel",
            "--parallel-strategy",
            "--mtp",
            "--max-num-batched-tokens",
            "--max-num-seqs",
            "--patched-vllm-enabled",
            "--notes",
            "--cache-mode",
            "--dataset",
            "--cudagraph-mode",
            "--enforce-eager",
            "--gpu-memory-utilization",
            "--kv-cache-dtype",
            "--image",
            "--delivery",
            "--overlay-mode",
            "--patch-files",
            "--data-parallel",
            "--pipeline-parallel",
            "--concurrency",
            "--expected-reqs",
            "--dry-run",
            "--campaigns-dir",
            "--json",
        ),
        "json": True,
        "ack": None,
        "description": (
            "Import an existing inference-perf-bench bundle "
            "(*-deploy/experiments/artifacts/inference-perf-bench/<bundle>/) "
            "into a perf-report campaign as cells/<cell-id>/normalized.json. "
            "Auto-detects bundle pattern: parses raw/sweep-c*.txt + "
            "raw/sweep-K*-c*.txt (vLLM bench-serve text format, GLM/DSv4 "
            "layout) OR bench-c<NNN>/raw/load.jsonl + raw/load.jsonl "
            "(drive_load.py JSONL format, Kimi layout). v1.21.0 adds the "
            "drive_load auto-dispatch. Metadata is sourced from the bundle's "
            "inference_perfbench_v1.json if present; any missing required field "
            "MUST be passed via --model / --hardware / ... overrides. "
            "--concurrency is required only for single-c drive_load bundles "
            "that have raw/load.jsonl without bench-c<NNN>/ subdirs. Records each "
            "cell's OWN ISL/OSL shape (per-number exact shape, no smoothing -- "
            "docs/METHODOLOGY.md); the publish/render --strict gate enforces it "
            "per-row (methodology_problems), and the renderer labels per-cell via "
            "shape_label_problems() rather than smoothing heterogeneous cells to one caption."
        ),
    },
    "import_roofline_sweep": {
        "safety": "writes_artifacts",
        "required": ("--campaign", "--bundle"),
        "optional": (
            "--cell-id",
            "--model",
            "--hardware",
            "--quant",
            "--kv-dtype",
            "--model-config",
            "--tensor-parallel",
            "--parallel-strategy",
            "--mtp",
            "--max-num-batched-tokens",
            "--cache-mode",
            "--dataset",
            "--cudagraph-mode",
            "--enforce-eager",
            "--gpu-memory-utilization",
            "--kv-cache-dtype",
            "--image",
            "--delivery",
            "--overlay-mode",
            "--patch-files",
            "--data-parallel",
            "--pipeline-parallel",
            "--dry-run",
            "--campaigns-dir",
            "--json",
        ),
        "json": True,
        "ack": None,
        "description": (
            "Import an always-on prefill+decode roofline sweep bundle "
            "(*-deploy/profiling/roofline-sweep.sh output: decode_sweep.jsonl + "
            "prefill_sweep.jsonl) into a campaign. Emits cells/<id>-decode + "
            "<id>-prefill normalized.json (AtlasCell rows carrying per-(phase, "
            "concurrency/ISL) DCGM PROF utilization -- SM/tensor/DRAM active -- plus "
            "the analytical roofline coords -- in extra, flowing to atlas_v1.extra_json + "
            "roofline_v1 in the lake) plus a cells/<id>-decode/roofline_sweep.json the "
            "prefill/decode roofline renderer page consumes (embeds the analytical model "
            "shape from --model-config / a captured model_config.json / the registry, so "
            "render + lake are self-contained). Metadata defaults from "
            "roofline_sweep_manifest.json; override hardware/TP/quant/kv-dtype via flags."
        ),
    },
    "import_variant_ab": {
        "safety": "writes_artifacts",
        "required": ("--campaign", "--bundle", "--model"),
        "optional": (
            "--hardware",
            "--quant",
            "--tensor-parallel",
            "--parallel-strategy",
            "--mtp",
            "--max-num-batched-tokens",
            "--cache-mode",
            "--notes",
            "--dataset",
            "--cudagraph-mode",
            "--enforce-eager",
            "--gpu-memory-utilization",
            "--kv-cache-dtype",
            "--image",
            "--delivery",
            "--overlay-mode",
            "--patch-files",
            "--data-parallel",
            "--pipeline-parallel",
            "--require-plot-ready",
            "--dry-run",
            "--campaigns-dir",
            "--json",
        ),
        "json": True,
        "ack": None,
        "description": (
            "Import a cross-engine variant A/B bundle (run-variant-ab.sh output: "
            "<bundle>/<arm>/c<C>-t<T>.txt per arm) into a campaign as one "
            "cells/<arm>/normalized.json per arm, trial-averaged, engine-tagged "
            "(vllm-sweep | sglang-sweep from each arm's result.json/name) so vLLM "
            "and SGLang arms are first-class + cross-engine-comparable. Per-arm "
            "zymtrace L1 SoL is auto-ingested when run-variant-ab.sh wrote "
            "<arm>/capture_sources.json (declared-coverage: a broken TSV aborts "
            "the import). This is the first-class, discoverable form of the "
            "variant-A/B path that import_perf_bench auto-dispatches; use it when "
            "you know the bundle is a cross-engine A/B (it feeds champion_select). "
            "--require-plot-ready hard-fails if any arm lacks the throughput-scatter "
            "fields a strict publish needs."
        ),
    },
    "capture_plan": {
        "safety": "writes_artifacts",
        "required": ("--campaign",),
        "optional": ("--source-campaign", "--out", "--campaigns-dir", "--json"),
        "json": True,
        "ack": None,
        "description": (
            "Build an exact-variant capture reuse plan for one target campaign. "
            "Scans local atlas rows + cells/* capture artifacts, computes a "
            "conservative serving-variant signature, groups missing captures by "
            "signature, and lists exact-match reuse candidates from source campaigns. "
            "Writes the plan JSON when --out is supplied."
        ),
    },
    "materialize_capture_reuse": {
        "safety": "writes_artifacts",
        "required": ("--plan",),
        "optional": ("--dry-run", "--json"),
        "json": True,
        "ack": None,
        "description": (
            "Copy exact-match capture artifacts from a capture_plan JSON into target "
            "cells and write capture_reuse.json provenance. Copies only candidates "
            "whose source artifact exists and whose target artifact is still absent."
        ),
    },
    "dcgm_correlate": {
        "safety": "writes_artifacts",
        "required": ("--campaign", "--cell-id", "--frozen-yaml"),
        "optional": (
            "--kernels-json",
            "--ceilings",
            "--dry-run",
            "--campaigns-dir",
            "--json",
        ),
        "json": True,
        "ack": None,
        "description": (
            "Fold a frozen DCGM measurement YAML (dcgm_frozen_v1, e.g. "
            "perf-tune-report/configs/dcgm-frozen/<name>.yaml) into a campaign "
            "cell's cells/<cell-id>/dcgm_correlation.json -- the L2/L3 "
            "byte/FLOP workload-level Speed-of-Light grounding the renderer's "
            "page 6 consumes. When the cell has a kernels.json (or --kernels-json "
            "is passed), the zymtrace x DCGM per-category cross-attribution "
            "(page 6b) is also populated. This is the byte-grounding step that "
            "flips a campaign from sol_complete-only to dcgm_grounded=true. The "
            "frozen YAML is the offline-reproducible path; the live Prometheus "
            "correlate() path is a library API (tools.perf_tune_report.dcgm_correlate."
            "correlate) for callers that can construct a PrometheusClient (a "
            "standalone CLI cannot reach the Prometheus MCP-mcp)."
        ),
    },
    "import_nsys": {
        "safety": "writes_artifacts",
        "required": ("--campaign", "--cell-id", "--bundle"),
        "optional": (
            "--kern-sum-name",
            "--dry-run",
            "--campaigns-dir",
            "--json",
        ),
        "json": True,
        "ack": None,
        "description": (
            "Import an nsys per-kernel bundle (with capture_sources.json declaring "
            "'nsys' + nsys/cuda_gpu_kern_sum.txt from `nsys stats --report "
            "cuda_gpu_kern_sum`) into a campaign cell as cells/<cell-id>/kernels.json "
            "(zymtrace-compatible schema; Total GPU time(ns) as the per-kernel weight), "
            "which the renderer's page-3 kernel breakdown + page-6b cross-attribution "
            "consume. Use when a zymtrace GPU flamegraph is unavailable (e.g. the "
            "per-process implant did not intercept)."
        ),
    },
    "import_ncu": {
        "safety": "writes_artifacts",
        "required": ("--campaign", "--cell-id", "--bundle"),
        "optional": (
            "--hw-key",
            "--dry-run",
            "--campaigns-dir",
            "--json",
        ),
        "json": True,
        "ack": None,
        "description": (
            "Import an ncu per-kernel bundle "
            "(*-deploy/experiments/artifacts/ncu-perkernel/<bundle>/, with "
            "capture_sources.json declaring 'ncu' + ncu-profiles/*-sol.csv "
            "+ *-raw.csv pairs) into a perf-report campaign as "
            "cells/<cell-id>/ncu_kernels.json, which the renderer's page-5 "
            "Speed-of-Light roofline scatter consumes. Handles both ncu wide "
            "(--page raw) and ncu-2026 long/melted (--page details) CSV "
            "shapes. Kernels captured with --set=basic (no FLOPS/DRAM-byte "
            "counters) import with measured %SoL but null arithmetic "
            "intensity; page 5 plots those as %SoL-only points. --hw-key "
            "selects the sol-ceilings.yaml hardware row (default b200_sm100)."
        ),
    },
    "tpm_summary": {
        "safety": "writes_artifacts",
        "required": ("--campaign",),
        "optional": (
            "--ttft-sla-ms",
            "--tpot-sla-ms",
            "--gpus-per-node",
            "--context",
            "--out-dir",
            "--campaigns-dir",
            "--json",
        ),
        "json": True,
        "ack": None,
        "description": (
            "Roll a campaign's atlas.jsonl into a per-hardware tokens-per-minute "
            "(TPM) capacity summary for pricing / capacity discussions (v1.35.0). "
            "For each (model, hardware, quant, TP, strategy, MTP) group it reports "
            "a peak-capacity point and -- when --ttft-sla-ms / --tpot-sla-ms are "
            "supplied -- a latency-SLA-bounded point, each at per-GPU, per-replica "
            "(=TP GPUs), and per-node (--gpus-per-node, default 8) bases, for both "
            "output-only and total (input+output) TPM. Pure post-processing of "
            "already-measured atlas data (no cluster runs). Writes "
            "tpm_summary.{json,csv,md} to the campaign dir (or --out-dir). "
            "Total-TPM is n/a for backends that emit no total-token line."
        ),
    },
    "champion_select": {
        "safety": "writes_artifacts",
        "required": ("--campaign",),
        "optional": (
            "--focus",
            "--focus-c",
            "--top",
            "--baseline",
            "--metric",
            "--slo-rel",
            "--slo-abs-ms",
            "--trials",
            "--same-node",
            "--require-workloads",
            "--workloads-present",
            "--accuracy-gate",
            "--accuracy-floor",
            "--out",
            "--title",
            "--campaigns-dir",
            "--json",
        ),
        "json": True,
        "ack": None,
        "description": (
            "Select the production champion: from a campaign's atlas + per-cell "
            "SoL artifacts, rank the cross-engine (vLLM + SGLang) variant arms "
            "under the focus metric (tok/s/GPU or median TPOT) + a TPOT SLO, pick "
            "the baseline + top-X (default 3), summarize each across the 4-layer "
            "SoL ladder (L1 zymtrace / L2 / L3 DCGM / L4 ncu) + gather the "
            "roofline operating points, and emit a tiered (DRAFT/VERDICT) "
            "production recommendation. A VERDICT requires the variance "
            "(--same-node + --trials>=3), multi-workload (--workloads-present "
            "covers the canonical suite), and accuracy (--accuracy-gate pass) "
            "gates AND L3 byte-grounding of the champion; anything short is a "
            "DRAFT. Writes CHAMPION.md + champion_select.json (the artifact the "
            "renderer champion page + the perf-lake champion rows consume). Pure "
            "post-processing -- no cluster runs."
        ),
    },
}


# ----------------------------------------------------------------------------
# Verb implementations
# ----------------------------------------------------------------------------

def cmd_campaign_init(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        print(f"FATAL: --config does not exist: {config_path}", file=sys.stderr)
        return 2
    config = load_yaml(config_path)

    slug = args.slug or slugify(config.get("name") or config_path.stem)
    campaigns_root = resolve_campaigns_dir(args.campaigns_dir)
    campaigns_root.mkdir(parents=True, exist_ok=True)

    # Experiment-id is the single join key (AGENTS.md "Experiment Isolation &
    # Traceability"): it is the evidence-bundle run-id AND the cluster label
    # value AND -- when supplied -- the campaign_id (this dir's basename). When
    # an --experiment-id (or config experiment_id:) is given, the campaign dir is
    # named EXACTLY that, so the published campaign_id joins back to the cluster
    # objects (experiment=<id-slug>) and the evidence bundle. Otherwise we fall
    # back to the historical <UTC>-<slug> name and the campaign_id becomes the
    # experiment_id (still self-consistent, just not pre-linked to a bundle).
    experiment_id = args.experiment_id or config.get("experiment_id")
    family = args.family or config.get("family", "") or ""
    evidence_bundle = args.evidence_bundle or config.get("evidence_bundle_path", "") or ""

    if experiment_id:
        # Lazy import: lake_writer's module top is light (pyarrow is imported
        # inside its functions), so this does not pull pyarrow into campaign_init.
        from tools.perf_tune_report.lake_writer import _CAMPAIGN_UTC_RE
        if not _CAMPAIGN_UTC_RE.search(experiment_id):
            print(
                f"FATAL: --experiment-id {experiment_id!r} must contain a "
                f"YYYYMMDDTHHMMSSZ stamp (it is the evidence-bundle run-id); "
                f"publish derives the dt= partition from it.",
                file=sys.stderr,
            )
            return 2
        campaign_dir = campaigns_root / experiment_id
    else:
        campaign_dir = campaigns_root / f"{utc_timestamp_slug()}-{slug}"

    if campaign_dir.exists():
        print(f"FATAL: campaign already exists: {campaign_dir}", file=sys.stderr)
        return 2
    campaign_dir.mkdir(parents=True)
    (campaign_dir / "cells").mkdir()
    (campaign_dir / "commands").mkdir()

    # campaign_id == dir basename throughout lake_writer; default experiment_id
    # to it so the campaign_v1 row always carries a non-empty join key.
    if not experiment_id:
        experiment_id = campaign_dir.name

    # Freeze the config alongside the campaign for reproducibility.
    frozen_config = campaign_dir / "config.yaml"
    shutil.copy2(config_path, frozen_config)

    # Source-code attribution (durable-lineage workstream): lift the bundle's
    # ```provenance``` block (the machine-readable link to the vLLM commit/branch
    # + image + delivery under test) into the campaign so it flows to the lake.
    # Precedence: the evidence bundle's SOURCE.md block, then a config
    # `provenance:` mapping. We persist it three ways so every consumer finds it:
    #   (1) provenance.json sidecar (lineage_view / audits),
    #   (2) appended `provenance:` mapping in the frozen config.yaml (so it is
    #       covered by config_yaml_sha256), and
    #   (3) flat `- <key>: <value>` bullets in SOURCE.md, which parse_source_md
    #       lifts into the campaign_v1 source columns.
    from tools.perf_tune_report import provenance as provenance_mod

    prov: dict[str, Any] | None = None
    if evidence_bundle:
        prov = provenance_mod.parse_file(Path(evidence_bundle).expanduser() / "SOURCE.md")
    if prov is None and isinstance(config.get("provenance"), dict):
        prov = config["provenance"]
    prov_bullets = ""
    if prov:
        import json as _json

        (campaign_dir / "provenance.json").write_text(
            _json.dumps(prov, indent=2, sort_keys=True)
        )
        if "provenance" not in config:
            # Append (not re-dump) so the original config bytes are preserved.
            import yaml as _yaml

            dumped = _yaml.safe_dump({"provenance": prov}, sort_keys=False)
            with frozen_config.open("a", encoding="utf-8") as fh:
                fh.write("\n# --- source-code attribution (campaign_init) ---\n")
                fh.write(dumped)
        prov_bullets = provenance_mod.flat_bullets(prov)

    cells = config.get("cells", [])
    cell_ids = [c["cell_id"] for c in cells]
    source_md = (
        f"# Perf-report campaign: {slug}\n\n"
        f"- captured_at: {utc_timestamp_slug()}\n"
        f"- config: {config_path}\n"
        f"- cells: {len(cells)}\n"
        f"- backend(s): {sorted({c.get('backend', 'unspecified') for c in cells})}\n"
        f"- experiment_id: {experiment_id}\n"
        f"- family: {family}\n"
        f"- evidence_bundle_path: {evidence_bundle}\n"
        + prov_bullets
    )
    (campaign_dir / "SOURCE.md").write_text(source_md)
    (campaign_dir / "summary.md").write_text(
        "# Verdict\n\n_TBD: fill in after the run is complete._\n"
    )

    payload = {
        "tool": "perf_tune_report_campaign_init",
        "library": "perf_tune_report",
        "verb": "campaign_init",
        "safety": CONTRACT["campaign_init"]["safety"],
        "campaign_dir": str(campaign_dir),
        "slug": slug,
        "experiment_id": experiment_id,
        "family": family,
        "cells": cell_ids,
        "cell_count": len(cells),
    }
    emit(payload, as_json=args.json)
    return 0


def cmd_cell_run(args: argparse.Namespace) -> int:
    if not args.dry_run and not getattr(args, "i_understand_this_submits_jobs", False):
        print(
            "FATAL: cell_run is ack-gated (safety=submits_jobs). "
            "Pass --i-understand-this-submits-jobs to actually run, "
            "or --dry-run to print the command without executing.",
            file=sys.stderr,
        )
        return 2

    campaigns_root = resolve_campaigns_dir(args.campaigns_dir)
    campaign_dir = resolve_campaign_dir(args.campaign, campaigns_root)
    config_path = campaign_dir / "config.yaml"
    if not config_path.is_file():
        print(f"FATAL: campaign config not found: {config_path}", file=sys.stderr)
        return 2
    config = load_yaml(config_path)
    cells_by_id = {c["cell_id"]: c for c in config.get("cells", [])}
    if args.cell not in cells_by_id:
        print(
            f"FATAL: cell {args.cell!r} not in campaign config; available: "
            f"{sorted(cells_by_id)}",
            file=sys.stderr,
        )
        return 2

    cell_dict = cells_by_id[args.cell]
    cell_cfg = cell_config_from_dict(cell_dict)

    if args.backend == "vllm-sweep":
        serve_cmd = args.serve_cmd or cell_dict.get("vllm_sweep", {}).get("serve_cmd")
        bench_cmd = args.bench_cmd or cell_dict.get("vllm_sweep", {}).get("bench_cmd")
        if not serve_cmd or not bench_cmd:
            print(
                "FATAL: vllm-sweep backend requires --serve-cmd and --bench-cmd "
                "(or vllm_sweep.serve_cmd / vllm_sweep.bench_cmd in the cell config).",
                file=sys.stderr,
            )
            return 2
        result = run_cell_vllm_sweep(
            cell_cfg,
            campaign_dir,
            serve_cmd=serve_cmd,
            bench_cmd=bench_cmd,
            dry_run=args.dry_run,
        )
        payload = {
            "tool": "perf_tune_report_cell_run",
            "library": "perf_tune_report",
            "verb": "cell_run",
            "safety": CONTRACT["cell_run"]["safety"],
            "ack_required": True,
            "ack_field": "i_understand_this_submits_jobs",
            "campaign_dir": str(campaign_dir),
            "cell_id": cell_cfg.cell_id,
            "backend": "vllm-sweep",
            "status": result.status,
            "row_count": result.row_count,
            "dry_run": result.dry_run,
            "command": result.command,
        }
    elif args.backend == "aiperf":
        aiperf_cfg = cell_dict.get("aiperf", {})
        namespace = args.namespace or aiperf_cfg.get("namespace")
        bench_pod = args.bench_pod or aiperf_cfg.get("bench_pod")
        kube_context = args.kube_context or aiperf_cfg.get("kube_context")
        endpoint_url = args.endpoint_url or aiperf_cfg.get("endpoint_url")
        served_model = args.served_model or aiperf_cfg.get("served_model")
        dataset_split = args.dataset_split or aiperf_cfg.get("dataset_split", "2025_07")
        conv_count = args.conversation_count or aiperf_cfg.get("conversation_count")
        for name, value in (
            ("--namespace", namespace),
            ("--bench-pod", bench_pod),
            ("--kube-context", kube_context),
            ("--endpoint-url", endpoint_url),
            ("--served-model", served_model),
        ):
            if not value:
                print(
                    f"FATAL: aiperf backend requires {name} (CLI or cell.aiperf.* field).",
                    file=sys.stderr,
                )
                return 2
        result = run_cell_aiperf(
            cell_cfg,
            campaign_dir,
            namespace=namespace,
            bench_pod=bench_pod,
            kube_context=kube_context,
            endpoint_url=endpoint_url,
            served_model=served_model,
            dataset_split=dataset_split,
            conversation_count=conv_count,
            dry_run=args.dry_run,
        )
        payload = {
            "tool": "perf_tune_report_cell_run",
            "library": "perf_tune_report",
            "verb": "cell_run",
            "safety": CONTRACT["cell_run"]["safety"],
            "ack_required": True,
            "ack_field": "i_understand_this_submits_jobs",
            "campaign_dir": str(campaign_dir),
            "cell_id": cell_cfg.cell_id,
            "backend": "aiperf",
            "status": result.status,
            "row_count": result.row_count,
            "dry_run": result.dry_run,
            "commands": result.commands,
        }
    elif args.backend == "aa":
        aa_cfg = cell_dict.get("aa", {})
        model = args.served_model or aa_cfg.get("model")
        url = args.endpoint_url or aa_cfg.get("url")
        shape_name = getattr(args, "aa_shape", None) or aa_cfg.get("shape")
        mode = getattr(args, "aa_mode", None) or aa_cfg.get("mode", "synthetic")
        request_count = getattr(args, "request_count", None) or aa_cfg.get("request_count", 10)
        for name, value in (
            ("--served-model (or cell.aa.model)", model),
            ("--endpoint-url (or cell.aa.url)", url),
            ("--aa-shape (or cell.aa.shape)", shape_name),
        ):
            if not value:
                print(
                    f"FATAL: aa backend requires {name}.",
                    file=sys.stderr,
                )
                return 2
        # API key is read from the named env var (never the YAML / CLI) so it
        # is not written into the campaign config or evidence bundle.
        api_key_env = aa_cfg.get("api_key_env", "WANDB_INFERENCE_API_KEY")
        api_key = os.environ.get(api_key_env) if api_key_env else None
        kube = None
        namespace = args.namespace or aa_cfg.get("namespace")
        if namespace:
            kube = {
                "namespace": namespace,
                "bench_pod": args.bench_pod or aa_cfg.get("bench_pod"),
                "kube_context": args.kube_context or aa_cfg.get("kube_context"),
            }
        try:
            result = run_cell_aa(
                cell_cfg,
                campaign_dir,
                shape_name=shape_name,
                model=model,
                url=url,
                endpoint=aa_cfg.get("endpoint", "/v1/chat/completions"),
                endpoint_type=aa_cfg.get("endpoint_type", "chat"),
                api_key=api_key,
                tokenizer=aa_cfg.get("tokenizer"),
                tokenizer_trust_remote_code=bool(aa_cfg.get("tokenizer_trust_remote_code", True)),
                mode=mode,
                request_count=int(request_count),
                custom_dataset_type=aa_cfg.get("custom_dataset_type", "mooncake_trace"),
                extra_output_controls=bool(aa_cfg.get("extra_output_controls", True)),
                input_file=aa_cfg.get("input_file"),
                dataset_count=aa_cfg.get("dataset_count"),
                aiperf_cmd=aa_cfg.get("aiperf_cmd"),
                kube=kube,
                # Spec-decode AL capture is on by default whenever a bench pod
                # is set (kube mode); cell.aa.spec_scrape: false opts out.
                spec_scrape=bool(aa_cfg.get("spec_scrape", True)),
                # Settle discipline (opt-in, GB300 settle audit):
                # cell.aa.prewarm_shapes + cell.aa.burn_in + cell.aa.settle_s.
                prewarm_shapes=aa_cfg.get("prewarm_shapes"),
                burn_in=bool(aa_cfg.get("burn_in", False)),
                settle_s=int(aa_cfg.get("settle_s", 30)),
                dry_run=args.dry_run,
            )
        except ValueError as exc:
            print(f"FATAL: {exc}", file=sys.stderr)
            return 2
        payload = {
            "tool": "perf_tune_report_cell_run",
            "library": "perf_tune_report",
            "verb": "cell_run",
            "safety": CONTRACT["cell_run"]["safety"],
            "ack_required": True,
            "ack_field": "i_understand_this_submits_jobs",
            "campaign_dir": str(campaign_dir),
            "cell_id": cell_cfg.cell_id,
            "backend": "aa",
            "aa_shape": result.shape,
            "aa_mode": result.mode,
            "status": result.status,
            "row_count": result.row_count,
            "dry_run": result.dry_run,
            "commands": result.commands,
        }
        if result.dataset_info:
            payload["dataset_info"] = result.dataset_info
        if result.spec_windows:
            payload["spec_decode"] = {
                str(c): {"al": w["al"], "accept_rate": w["accept_rate"]}
                for c, w in sorted(result.spec_windows.items())
            }
    else:
        print(
            f"FATAL: --backend must be one of: vllm-sweep, aiperf, aa (got {args.backend!r})",
            file=sys.stderr,
        )
        return 2

    emit(payload, as_json=args.json)
    return 0


def cmd_value_view(args: argparse.Namespace) -> int:
    campaigns_root = resolve_campaigns_dir(args.campaigns_dir)
    registry_path = (
        Path(args.registry).expanduser()
        if args.registry
        else default_registry_path(campaigns_root)
    )
    if not registry_path.is_file():
        print(
            f"FATAL: value-findings registry not found at {registry_path}; pass "
            "--registry or create perf-tune-report/configs/value-findings.yaml.",
            file=sys.stderr,
        )
        return 2
    registry = load_yaml(registry_path)
    view = build_value_view(registry, campaigns_root)
    # Resolve the $/GPU-hour: --gpu-hr override wins, else perf-tune-report/configs/cost.yaml
    # (the fleet is GB300), else GPU_HR_DEFAULT -- shared with fleet_leaderboard's knob.
    from tools.perf_tune_report.fleet_leaderboard import resolve_gpu_hr
    gpu_hr = resolve_gpu_hr("GB300", campaigns_root.parent / "configs",
                            getattr(args, "gpu_hr", None))
    if getattr(args, "format", "table") == "report":
        md = render_report(view, title=args.title or "Inference perf wins -- value prop",
                           gpu_hr=gpu_hr)
    else:
        md = render_markdown(view, title=args.title or "Value ledger", gpu_hr=gpu_hr)
    out_path = Path(args.out).expanduser() if args.out else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md)
    elif not args.json:
        print(md)
    all_flags = [fl for f in view["findings"] for fl in f["live"]["flags"]]
    payload = {
        "tool": "perf_tune_report_value_view",
        "library": "perf_tune_report",
        "verb": "value_view",
        "safety": CONTRACT["value_view"]["safety"],
        "registry": str(registry_path),
        "campaigns_dir": str(campaigns_root),
        "finding_count": len(view["findings"]),
        "out_path": str(out_path) if out_path else None,
        "flag_count": len(all_flags),
        "flags": all_flags,
        "gpu_hr": gpu_hr,
    }
    if out_path is not None or args.json:
        emit(payload, as_json=args.json)
    return 0


def cmd_portability_view(args: argparse.Namespace) -> int:
    """Render the lever-by-model portability matrix from value-findings.yaml."""
    from tools.perf_tune_report.portability_view import build_portability, render_markdown

    campaigns_root = resolve_campaigns_dir(args.campaigns_dir)
    registry_path = (
        Path(args.registry).expanduser()
        if args.registry
        else default_registry_path(campaigns_root)
    )
    if not registry_path.is_file():
        print(
            f"FATAL: value-findings registry not found at {registry_path}; pass "
            "--registry or create perf-tune-report/configs/value-findings.yaml.",
            file=sys.stderr,
        )
        return 2
    registry = load_yaml(registry_path)
    view = build_portability(registry)
    md = render_markdown(view, title=args.title or "Lever x model portability matrix")
    out_path = Path(args.out).expanduser() if args.out else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md)
    elif not args.json:
        print(md)
    payload = {
        "tool": "perf_tune_report_portability_view",
        "library": "perf_tune_report",
        "verb": "portability_view",
        "safety": CONTRACT["portability_view"]["safety"],
        "registry": str(registry_path),
        "model_count": len(view["models"]),
        "lever_count": len(view["rows"]),
        "models": view["models"],
        "out_path": str(out_path) if out_path else None,
    }
    if out_path is not None or args.json:
        emit(payload, as_json=args.json)
    return 0


def cmd_champion_select(args: argparse.Namespace) -> int:
    campaigns_root = resolve_campaigns_dir(args.campaigns_dir)
    campaign_dir = resolve_campaign_dir(args.campaign, campaigns_root)
    workloads_present = (
        tuple(w.strip() for w in args.workloads_present.split(",") if w.strip())
        if args.workloads_present
        else None
    )
    require_workloads = tuple(
        w.strip() for w in (args.require_workloads or "").split(",") if w.strip()
    ) or champion_select_lib.CANONICAL_WORKLOADS
    try:
        result = champion_select_lib.select(
            campaign_dir,
            focus=args.focus,
            focus_c=args.focus_c,
            top=args.top,
            baseline=args.baseline,
            metric=args.metric,
            slo_rel=args.slo_rel,
            slo_abs_ms=args.slo_abs_ms,
            trials=args.trials,
            same_node=args.same_node,
            require_workloads=require_workloads,
            workloads_present=workloads_present,
            accuracy_gate=args.accuracy_gate,
            accuracy_floor=args.accuracy_floor,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    out_md = Path(args.out).expanduser() if args.out else None
    if args.dry_run:
        json_path = campaign_dir / "champion_select.json"
        md_path = out_md or (campaign_dir / "CHAMPION.md")
    else:
        json_path, md_path = champion_select_lib.write_outputs(
            result, campaign_dir, out_md=out_md, title=args.title
        )
    if not args.json:
        print(champion_select_lib.render_markdown(result, title=args.title))
    payload = {
        "tool": "perf_tune_report_champion_select",
        "library": "perf_tune_report",
        "verb": "champion_select",
        "safety": CONTRACT["champion_select"]["safety"],
        "campaign_dir": str(campaign_dir),
        "champion_json": str(json_path),
        "champion_md": str(md_path),
        "focus": result.focus,
        "focus_c": result.focus_c,
        "metric": result.metric,
        "baseline_cell": result.baseline_cell,
        "recommended_cell": result.recommended_cell,
        "recommended_engine": result.recommended_engine,
        "tier": result.tier,
        "variant_count": len(result.variants),
        "gates": [g.to_dict() for g in result.gates],
        "dry_run": args.dry_run,
    }
    if args.json:
        emit(payload, as_json=True)
    return 0


def cmd_capture_plan(args: argparse.Namespace) -> int:
    campaigns_root = resolve_campaigns_dir(args.campaigns_dir)
    target_campaign = resolve_campaign_dir(args.campaign, campaigns_root)
    source_campaigns = [
        resolve_campaign_dir(c, campaigns_root)
        for c in (args.source_campaign or [])
    ]
    if not source_campaigns:
        source_campaigns = [target_campaign]
    try:
        plan = build_capture_plan([target_campaign], source_campaigns)
    except ValueError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    payload = plan.to_dict()
    payload.update({
        "tool": "perf_tune_report_capture_plan",
        "library": "perf_tune_report",
        "verb": "capture_plan",
        "safety": CONTRACT["capture_plan"]["safety"],
        "campaign_dir": str(target_campaign),
        "cell_count": len(plan.cells),
        "reuse_candidate_count": len(plan.reuse_candidates),
        "missing_group_count": len(plan.missing_groups),
    })
    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n")
        payload["out"] = str(out)
    emit(payload, as_json=args.json)
    return 0


def cmd_materialize_capture_reuse(args: argparse.Namespace) -> int:
    plan_path = Path(args.plan).expanduser()
    try:
        result = materialize_reuse(plan_path, dry_run=args.dry_run)
    except ValueError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    payload = result.to_dict()
    payload.update({
        "tool": "perf_tune_report_materialize_capture_reuse",
        "library": "perf_tune_report",
        "verb": "materialize_capture_reuse",
        "safety": CONTRACT["materialize_capture_reuse"]["safety"],
        "plan": str(plan_path),
        "copied_count": len(result.copied),
        "skipped_count": len(result.skipped),
        "dry_run": args.dry_run,
    })
    emit(payload, as_json=args.json)
    return 0


def cmd_atlas_aggregate(args: argparse.Namespace) -> int:
    campaigns_root = resolve_campaigns_dir(args.campaigns_dir)
    campaign_dir = resolve_campaign_dir(args.campaign, campaigns_root)
    try:
        result = aggregate(campaign_dir)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    payload = {
        "tool": "perf_tune_report_atlas_aggregate",
        "library": "perf_tune_report",
        "verb": "atlas_aggregate",
        "safety": CONTRACT["atlas_aggregate"]["safety"],
        "campaign_dir": str(campaign_dir),
        "atlas_path": str(result.atlas_path),
        "row_count": result.row_count,
        "cell_count": result.cell_count,
        "coverage": {
            "atlas_cells": result.coverage.atlas_cells,
            "full_sweeps": result.coverage.full_sweeps,
            "partial_sweeps": result.coverage.partial_sweeps,
            "failed_cells": result.coverage.failed_cells,
            "plot_ready_points": result.coverage.plot_ready_points,
            "evicted_cells": result.coverage.evicted_cells,
            "non_plot_ready_full_cells": result.coverage.non_plot_ready_full_cells,
            "header_line": result.coverage.header_line(),
        },
    }
    # Loud warning: a 0-plot-ready or full-but-unplottable atlas yields a
    # blank scatter downstream. Surface it here instead of silently passing
    # an empty atlas to report_render.
    if result.coverage.plot_ready_points == 0:
        print(
            "WARNING: 0 plot-ready concurrency points in this atlas -- the "
            "report scatter will be empty. How to fix: ensure each cell's bench "
            "output includes 'Median TTFT (ms)' and 'Request throughput (req/s)', "
            "then re-import + re-aggregate.",
            file=sys.stderr,
        )
    elif result.coverage.non_plot_ready_full_cells:
        print(
            f"WARNING: {result.coverage.non_plot_ready_full_cells} STATUS_FULL "
            "cell(s) are not plot-ready (missing ttft_avg_ms and/or "
            "request_throughput_avg) and will not produce scatter points. "
            "How to fix: re-import those cells with complete bench output.",
            file=sys.stderr,
        )
    emit(payload, as_json=args.json)
    return 0


def cmd_report_render(args: argparse.Namespace) -> int:
    from tools.perf_tune_report.renderer.render_report import render_report

    campaigns_root = resolve_campaigns_dir(args.campaigns_dir)
    campaign_dir = resolve_campaign_dir(args.campaign, campaigns_root)
    atlas_path = campaign_dir / "atlas.jsonl"
    if not atlas_path.is_file():
        print(
            f"FATAL: atlas.jsonl not found at {atlas_path}; run `perftunereport atlas aggregate` first.",
            file=sys.stderr,
        )
        return 2
    out_pdf = Path(args.out).expanduser() if args.out else campaign_dir / "report.pdf"
    status = render_report(
        atlas_path,
        out_pdf,
        title=args.title or "perf-report",
        variants_line=args.variants_line,
        data_source_line=args.data_source_line,
    )
    rows = read_jsonl(atlas_path)
    coverage = summarize(rows)
    payload = {
        "tool": "perf_tune_report_report_render",
        "library": "perf_tune_report",
        "verb": "report_render",
        "safety": CONTRACT["report_render"]["safety"],
        "campaign_dir": str(campaign_dir),
        "out_pdf": str(out_pdf),
        "size_bytes": out_pdf.stat().st_size,
        "coverage": {
            "atlas_cells": coverage.atlas_cells,
            "full_sweeps": coverage.full_sweeps,
            "partial_sweeps": coverage.partial_sweeps,
            "failed_cells": coverage.failed_cells,
            "plot_ready_points": coverage.plot_ready_points,
            "evicted_cells": coverage.evicted_cells,
            "non_plot_ready_full_cells": coverage.non_plot_ready_full_cells,
            "header_line": coverage.header_line(),
        },
        "render_status": {
            "sol_complete": status.sol_complete,
            "focus": status.focus,
            "sol_rigor": status.sol_rigor,
            "dcgm_grounded": status.dcgm_grounded,
            "plot_ready_points": status.plot_ready_points,
            "non_plot_ready_full_cells": status.non_plot_ready_full_cells,
            "rendered_pages": status.rendered_pages,
            "omitted_pages": status.omitted_pages,
            "report_status_json": str(campaign_dir / "report_status.json"),
        },
    }

    # Never let an omission pass silently: list each omitted page (why +
    # how-to-fix) on stderr. The same detail is on the PDF completeness page
    # + report_status.json.
    for omission in status.omitted_pages:
        print(
            f"WARNING: omitted {omission['page']} -- {omission['why']} "
            f"How to fix: {omission['how_to_fix']}",
            file=sys.stderr,
        )

    # DCGM byte-grounding is the L2/L3 analog of the L1 SoL roofline. Surface
    # its absence loudly even when sol_complete=True, so a zymtrace-only
    # campaign is never mistaken for a fully byte-grounded one.
    if status.sol_complete and not status.dcgm_grounded:
        print(
            "WARNING: sol_complete=True but dcgm_grounded=False -- this campaign "
            "has the L1 zymtrace SoL roofline (page 4) but NO DCGM workload-level "
            "byte/FLOP grounding (pages 6/6b). Run dcgm_correlate per cell "
            "(inference-dcgm-correlate skill) then re-render + re-publish for a "
            "fully byte-grounded campaign.",
            file=sys.stderr,
        )

    if args.strict and (not status.sol_complete or status.plot_ready_points == 0):
        print(
            "FATAL: --strict and the report is incomplete "
            f"(sol_complete={status.sol_complete}, "
            f"plot_ready_points={status.plot_ready_points}). "
            "Capture the missing SoL/roofline inputs (see the warnings above and "
            "the report completeness page), then re-render. The PDF was still "
            f"written to {out_pdf} for inspection.",
            file=sys.stderr,
        )
        emit(payload, as_json=args.json)
        return 2

    # --strict also enforces DCGM byte-grounding: a campaign with the L1 SoL
    # roofline but no DCGM workload-level grounding (pages 6/6b) fails strict
    # unless --allow-ungrounded. Mirrors the publish_to_lake fail-closed gate.
    if (
        args.strict
        and status.sol_complete
        and not status.dcgm_grounded
        and not getattr(args, "allow_ungrounded", False)
    ):
        print(
            "FATAL: --strict and dcgm_grounded=False -- the report has the L1 "
            "zymtrace SoL roofline (page 4) but NO DCGM workload-level byte/FLOP "
            "grounding (pages 6/6b). Byte-grounding is mandatory: run `perftunereport "
            "dcgm_correlate` per cell (capture DCGM over the bench window + a "
            "dcgm_frozen_v1 YAML; see the inference-dcgm-correlate skill), then "
            f"re-render. The PDF was still written to {out_pdf} for inspection. "
            "Pass --allow-ungrounded to accept a deliberately zymtrace-only L1 report.",
            file=sys.stderr,
        )
        emit(payload, as_json=args.json)
        return 2

    emit(payload, as_json=args.json)
    return 0


def cmd_tpm_summary(args: argparse.Namespace) -> int:
    """Roll a campaign's atlas into a per-hardware TPM capacity summary."""
    from tools.perf_tune_report.tpm_summary import compute_tpm_summary

    campaigns_root = resolve_campaigns_dir(args.campaigns_dir)
    campaign_dir = resolve_campaign_dir(args.campaign, campaigns_root)
    atlas_path = campaign_dir / "atlas.jsonl"
    if not atlas_path.is_file():
        print(
            f"FATAL: atlas.jsonl not found at {atlas_path}; run "
            "`perftunereport atlas_aggregate --campaign <slug>` first.",
            file=sys.stderr,
        )
        return 2

    # SLA thresholds + node size: the campaign config.yaml `tpm:` block is the
    # shared source (so the verb, the PDF page, and the lake all agree). CLI
    # flags override the config when explicitly passed.
    from tools.perf_tune_report.tpm_summary import discover_tpm_config

    tpm_cfg = discover_tpm_config(campaign_dir)
    ttft_sla_ms = args.ttft_sla_ms if args.ttft_sla_ms is not None else tpm_cfg.ttft_sla_ms
    tpot_sla_ms = args.tpot_sla_ms if args.tpot_sla_ms is not None else tpm_cfg.tpot_sla_ms
    gpus_per_node = args.gpus_per_node if args.gpus_per_node is not None else tpm_cfg.gpus_per_node

    # Best-effort ISL/OSL context: the campaign config's top-level
    # description usually carries the shape (e.g. "ISL=5K/OSL=1K"). --context
    # overrides it. Neither is required.
    context_line = args.context
    if context_line is None:
        config_path = campaign_dir / "config.yaml"
        if config_path.is_file():
            try:
                cfg = load_yaml(config_path)
                desc = cfg.get("description")
                if isinstance(desc, str) and desc.strip():
                    context_line = " ".join(desc.split())
            except Exception:  # noqa: BLE001 - context is best-effort
                context_line = None

    rows = read_jsonl(atlas_path)
    summary = compute_tpm_summary(
        rows,
        ttft_sla_ms=ttft_sla_ms,
        tpot_sla_ms=tpot_sla_ms,
        gpus_per_node=gpus_per_node,
        context_line=context_line,
        usd_per_gpu_hour=tpm_cfg.usd_per_gpu_hour,
        cost_rate_source=tpm_cfg.cost_rate_source,
    )

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else campaign_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "tpm_summary.json"
    csv_path = out_dir / "tpm_summary.csv"
    md_path = out_dir / "tpm_summary.md"
    json_path.write_text(summary.to_json() + "\n", encoding="utf-8")
    csv_path.write_text(summary.to_csv(), encoding="utf-8")
    md_path.write_text(summary.to_markdown() + "\n", encoding="utf-8")

    if not summary.groups:
        print(
            "WARNING: 0 throughput-bearing atlas rows -- the TPM summary is "
            "empty. How to fix: ensure each cell's bench output includes "
            "'Output token throughput (tok/s)', then re-import + re-aggregate.",
            file=sys.stderr,
        )

    payload = {
        "tool": "perf_tune_report_tpm_summary",
        "library": "perf_tune_report",
        "verb": "tpm_summary",
        "safety": CONTRACT["tpm_summary"]["safety"],
        "campaign_dir": str(campaign_dir),
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "md_path": str(md_path),
        "group_count": len(summary.groups),
        "sla_computed": summary.sla_computed,
        "ttft_sla_ms": summary.ttft_sla_ms,
        "tpot_sla_ms": summary.tpot_sla_ms,
        "gpus_per_node": summary.gpus_per_node,
        "hardwares": _ordered_unique_hw(summary),
    }
    emit(payload, as_json=args.json)
    return 0


def _ordered_unique_hw(summary: Any) -> list[str]:
    seen: dict[str, None] = {}
    for g in summary.groups:
        seen.setdefault(g.hardware, None)
    return list(seen.keys())


def cmd_publish_to_lake(args: argparse.Namespace) -> int:
    """Publish a campaign's atlas + provenance as Parquet to S3."""
    from tools.perf_tune_report.lake_writer import (
        IF_EXISTS_CHOICES,
        IF_EXISTS_FAIL,
        CampaignIncompleteError,
        parse_source_md,
        publish,
        resolve_s3_config,
    )

    campaigns_root = resolve_campaigns_dir(args.campaigns_dir)
    campaign_dir = resolve_campaign_dir(args.campaign, campaigns_root)

    if_exists = args.if_exists or IF_EXISTS_FAIL
    if if_exists not in IF_EXISTS_CHOICES:
        print(
            f"FATAL: --if-exists must be one of {IF_EXISTS_CHOICES}; got {if_exists!r}",
            file=sys.stderr,
        )
        return 2

    try:
        cfg = resolve_s3_config(
            endpoint=args.s3_endpoint,
            bucket=args.s3_bucket,
            access_key_file=args.s3_access_key_file,
            secret_key_file=args.s3_secret_key_file,
        )
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2

    # Strict by default (workspace rigor policy). --no-strict OR the deprecated
    # --allow-incomplete alias opts out for a first-class intentional-gap publish.
    strict = getattr(args, "strict", True) and not args.allow_incomplete
    try:
        result = publish(
            campaign_dir,
            cfg=cfg,
            dry_run=args.dry_run,
            if_exists=if_exists,
            allow_incomplete=args.allow_incomplete,
            allow_ungrounded=getattr(args, "allow_ungrounded", False),
            strict=strict,
        )
    except CampaignIncompleteError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2

    payload = {
        "tool": "perf_tune_report_publish_to_lake",
        "library": "perf_tune_report",
        "verb": "publish_to_lake",
        "safety": CONTRACT["publish_to_lake"]["safety"],
        "campaign_dir": str(result.campaign_dir),
        "campaign_id": result.campaign_id,
        "captured_at_utc": result.captured_at_utc.isoformat(),
        "published_at_utc": result.published_at_utc.isoformat(),
        "dry_run": result.dry_run,
        "bucket": result.bucket,
        "endpoint": result.endpoint,
        "if_exists": if_exists,
        "tables": {
            "atlas_v1": {
                "local_path": str(result.atlas.local_path),
                "s3_key": result.atlas.s3_key,
                "size_bytes": result.atlas.size_bytes,
                "sha256": result.atlas.sha256,
                "row_count": result.atlas.row_count,
                "skipped": result.atlas.skipped,
            },
            "campaign_v1": {
                "local_path": str(result.campaign.local_path),
                "s3_key": result.campaign.s3_key,
                "size_bytes": result.campaign.size_bytes,
                "sha256": result.campaign.sha256,
                "row_count": result.campaign.row_count,
                "skipped": result.campaign.skipped,
            },
            "sol_v1": {
                "local_path": str(result.sol.local_path),
                "s3_key": result.sol.s3_key,
                "size_bytes": result.sol.size_bytes,
                "sha256": result.sol.sha256,
                "row_count": result.sol.row_count,
                "skipped": result.sol.skipped,
            },
            "tpm_v1": {
                "local_path": str(result.tpm.local_path),
                "s3_key": result.tpm.s3_key,
                "size_bytes": result.tpm.size_bytes,
                "sha256": result.tpm.sha256,
                "row_count": result.tpm.row_count,
                "skipped": result.tpm.skipped,
            },
            "cost_v1": {
                "local_path": str(result.cost.local_path),
                "s3_key": result.cost.s3_key,
                "size_bytes": result.cost.size_bytes,
                "sha256": result.cost.sha256,
                "row_count": result.cost.row_count,
                "skipped": result.cost.skipped,
            },
            "quality_v1": {
                "local_path": str(result.quality.local_path),
                "s3_key": result.quality.s3_key,
                "size_bytes": result.quality.size_bytes,
                "sha256": result.quality.sha256,
                "row_count": result.quality.row_count,
                "skipped": result.quality.skipped,
            },
            **({"champion_v1": {
                "local_path": str(result.champion.local_path),
                "s3_key": result.champion.s3_key,
                "size_bytes": result.champion.size_bytes,
                "sha256": result.champion.sha256,
                "row_count": result.champion.row_count,
                "skipped": result.champion.skipped,
            }} if result.champion is not None else {}),
            **({"roofline_v1": {
                "local_path": str(result.roofline.local_path),
                "s3_key": result.roofline.s3_key,
                "size_bytes": result.roofline.size_bytes,
                "sha256": result.roofline.sha256,
                "row_count": result.roofline.row_count,
                "skipped": result.roofline.skipped,
            }} if result.roofline is not None else {}),
        },
    }

    # Close the traceability loop: write the campaign + s3 paths back into the
    # EVIDENCE BUNDLE's SOURCE.md so the bundle records where its results landed
    # (this was operator-dependent and frequently missing). Best-effort + only on
    # a real publish; the bundle path comes from the campaign SOURCE.md's
    # evidence_bundle_path (written by campaign_init --evidence-bundle).
    if not result.dry_run:
        bundle_note = _append_lake_provenance_to_bundle(
            campaign_dir, parse_source_md, result
        )
        if bundle_note:
            payload["evidence_bundle_updated"] = bundle_note

    emit(payload, as_json=args.json)
    return 0


def _append_lake_provenance_to_bundle(campaign_dir, parse_source_md, result) -> str | None:
    """Append the campaign + s3 paths to the evidence bundle's SOURCE.md.

    Idempotent (skips if the campaign id is already recorded). Returns the bundle
    SOURCE.md path on write, else None. Never raises -- traceability is a
    best-effort convenience, not a publish gate.
    """
    try:
        meta = parse_source_md(campaign_dir / "SOURCE.md")
        bundle = meta.get("evidence_bundle_path", "")
        if not bundle:
            return None
        bundle_source = Path(bundle).expanduser() / "SOURCE.md"
        if not bundle_source.is_file():
            return None
        existing = bundle_source.read_text(encoding="utf-8")
        if f"campaign={result.campaign_id}" in existing:
            return None  # already recorded
        block = (
            "\n## Perf-lake publish (auto-appended by perftunereport publish_to_lake)\n\n"
            f"- campaign={result.campaign_id}\n"
            f"- atlas_v1: s3://{result.bucket}/{result.atlas.s3_key}\n"
            f"- campaign_v1: s3://{result.bucket}/{result.campaign.s3_key}\n"
            f"- published_at_utc: {result.published_at_utc.isoformat()}\n"
        )
        bundle_source.write_text(existing + block, encoding="utf-8")
        return str(bundle_source)
    except OSError:
        return None


def cmd_experiments_index(args: argparse.Namespace) -> int:
    """Enumerate all campaigns into a cross-experiment index (jsonl + md)."""
    from tools.perf_tune_report.experiments_index import build_index, write_index

    campaigns_root = resolve_campaigns_dir(args.campaigns_dir)
    if not campaigns_root.is_dir():
        print(f"FATAL: campaigns dir does not exist: {campaigns_root}", file=sys.stderr)
        return 2

    # Optional best-effort lake enumeration: mark which local campaigns are
    # actually published (campaign_v1 prefix). Warn-and-continue (never fail) when
    # creds are missing, so the local index always renders.
    published = None
    if getattr(args, "include_s3", False):
        from tools.perf_tune_report.experiments_index import enumerate_published_campaign_ids
        from tools.perf_tune_report.lake_writer import resolve_s3_config
        try:
            cfg = resolve_s3_config(
                endpoint=args.s3_endpoint,
                bucket=args.s3_bucket,
                access_key_file=args.s3_access_key_file,
                secret_key_file=args.s3_secret_key_file,
            )
            published = enumerate_published_campaign_ids(cfg, bucket=cfg.bucket)
            print(f"== --include-s3: {len(published)} published campaign(s) in the lake ==",
                  file=sys.stderr)
        except SystemExit as exc:
            print(f"WARNING: --include-s3 skipped (S3 creds unavailable): {exc}",
                  file=sys.stderr)
            published = None
        except Exception as exc:  # noqa: BLE001 - best-effort; never fail the index
            print(f"WARNING: --include-s3 lake enumeration failed: {exc}", file=sys.stderr)
            published = None

    rows = build_index(campaigns_root, published)
    if args.family:
        rows = [r for r in rows if r["family"] == args.family]
    # Default output: the perf-report bundle (campaigns_root's parent), so the
    # index is tracked alongside configs (data, not Python) per perf-tune-report/AGENTS.md.
    out_dir = Path(args.out).expanduser().resolve() if args.out else campaigns_root.parent
    paths = write_index(rows, out_dir)

    payload = {
        "tool": "perf_tune_report_experiments_index",
        "library": "perf_tune_report",
        "verb": "experiments_index",
        "safety": CONTRACT["experiments_index"]["safety"],
        "campaigns_dir": str(campaigns_root),
        "campaign_count": len(rows),
        "published_in_lake": (len(published) if published is not None else None),
        "jsonl": paths["jsonl"],
        "md": paths["md"],
    }
    emit(payload, as_json=args.json)
    return 0


def cmd_experiment_inventory(args: argparse.Namespace) -> int:
    """Canonical experiment count: unify local campaigns + run-id-stamped evidence bundles."""
    from tools.perf_tune_report.experiments_index import build_inventory, write_inventory

    campaigns_root = resolve_campaigns_dir(args.campaigns_dir)
    if not campaigns_root.is_dir():
        print(f"FATAL: campaigns dir does not exist: {campaigns_root}", file=sys.stderr)
        return 2

    bundle_roots = [Path(p).expanduser().resolve() for p in (args.bundle_root or [])]

    # Optional best-effort lake enumeration for the published count (same warn-and-continue
    # contract as experiments_index: never fail the inventory when creds are missing).
    published = None
    if getattr(args, "include_s3", False):
        from tools.perf_tune_report.experiments_index import enumerate_published_campaign_ids
        from tools.perf_tune_report.lake_writer import resolve_s3_config
        try:
            cfg = resolve_s3_config(
                endpoint=args.s3_endpoint,
                bucket=args.s3_bucket,
                access_key_file=args.s3_access_key_file,
                secret_key_file=args.s3_secret_key_file,
            )
            published = enumerate_published_campaign_ids(cfg, bucket=cfg.bucket)
            print(f"== --include-s3: {len(published)} published campaign(s) in the lake ==",
                  file=sys.stderr)
        except SystemExit as exc:
            print(f"WARNING: --include-s3 skipped (S3 creds unavailable): {exc}",
                  file=sys.stderr)
            published = None
        except Exception as exc:  # noqa: BLE001 - best-effort; never fail the inventory
            print(f"WARNING: --include-s3 lake enumeration failed: {exc}", file=sys.stderr)
            published = None

    inv = build_inventory(campaigns_root, bundle_roots, published)
    out_dir = Path(args.out).expanduser().resolve() if args.out else campaigns_root.parent
    paths = write_inventory(inv, out_dir)

    payload = {
        "tool": "perf_tune_report_experiment_inventory",
        "library": "perf_tune_report",
        "verb": "experiment_inventory",
        "safety": CONTRACT["experiment_inventory"]["safety"],
        "campaigns_dir": str(campaigns_root),
        "bundle_roots": [str(p) for p in bundle_roots],
        "total_experiments": inv["total_experiments"],
        "campaign_count": inv["campaign_count"],
        "bundle_count": inv["bundle_count"],
        "bundle_only_count": inv["bundle_only_count"],
        "published_in_lake": inv["published_in_lake"],
        "md": paths["md"],
        "json": paths["json"],
    }
    emit(payload, as_json=args.json)
    return 0


def cmd_trend_view(args: argparse.Namespace) -> int:
    """Longitudinal (model, variant_key) perf/quality trend across campaigns."""
    from tools.perf_tune_report.fleet_leaderboard import read_all_rows
    from tools.perf_tune_report.trend_view import build_trends, read_lake_rows, render_markdown

    source = "local-campaigns"
    if getattr(args, "lake_dir", None):
        # Published-lake mode: read atlas_v1 parquet from a pulled snapshot + join the
        # campaign_v1 vllm_commit. Local campaigns stay the default.
        lake_dir = Path(args.lake_dir).expanduser()
        if not lake_dir.is_dir():
            print(f"FATAL: --lake-dir does not exist: {lake_dir}", file=sys.stderr)
            return 2
        rows = read_lake_rows(lake_dir, hardware_filter=args.hardware)
        campaigns_root = lake_dir
        source = "published-lake"
    else:
        campaigns_root = resolve_campaigns_dir(args.campaigns_dir)
        if not campaigns_root.is_dir():
            print(f"FATAL: campaigns dir does not exist: {campaigns_root}", file=sys.stderr)
            return 2
        rows = read_all_rows(campaigns_root, hardware_filter=args.hardware)
    view = build_trends(rows, metric=args.metric, concurrency=args.concurrency,
                        regression_pct=args.regression_pct)
    md = render_markdown(view)
    out_path = Path(args.out).expanduser() if args.out else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md)
    elif not args.json:
        print(md)
    payload = {
        "tool": "perf_tune_report_trend_view",
        "library": "perf_tune_report",
        "verb": "trend_view",
        "safety": CONTRACT["trend_view"]["safety"],
        "campaigns_dir": str(campaigns_root),
        "source": source,
        "hardware": args.hardware,
        "metric": args.metric,
        "n_trends": view["n_trends"],
        "n_regressions": view["n_regressions"],
        "out_path": str(out_path) if out_path else None,
    }
    emit(payload, as_json=args.json)
    return 0


def cmd_import_model_eval(args: argparse.Namespace) -> int:
    """Import an lm-eval-harness results.json into a campaign quality cell (eval_acc)."""
    from tools.perf_tune_report.importers.model_eval import import_model_eval

    campaigns_root = resolve_campaigns_dir(args.campaigns_dir)
    campaign_dir = resolve_campaign_dir(args.campaign, campaigns_root)
    results = Path(args.results).expanduser()
    if not results.is_file():
        print(f"FATAL: --results not found: {results}", file=sys.stderr)
        return 2
    try:
        result = import_model_eval(
            results, campaign_dir,
            model=args.model, hardware=args.hardware, quant=args.quant,
            tensor_parallel=args.tensor_parallel, cell_id=args.cell_id,
            parallel_strategy=args.parallel_strategy,
            kv_cache_dtype=args.kv_cache_dtype, image=args.image,
        )
    except ValueError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    payload = {
        "tool": "perf_tune_report_import_model_eval",
        "library": "perf_tune_report",
        "verb": "import_model_eval",
        "safety": CONTRACT["import_model_eval"]["safety"],
        "campaign_dir": str(campaign_dir),
        **result,
    }
    emit(payload, as_json=args.json)
    return 0


def cmd_import_workloads(args: argparse.Namespace) -> int:
    """Import a bench-all-workloads.sh output dir into dataset-tagged campaign cells."""
    from tools.perf_tune_report.importers.workloads import import_workloads

    campaigns_root = resolve_campaigns_dir(args.campaigns_dir)
    campaign_dir = resolve_campaign_dir(args.campaign, campaigns_root)
    bench_dir = Path(args.bench_dir).expanduser()
    if not bench_dir.is_dir():
        print(f"FATAL: --bench-dir not found: {bench_dir}", file=sys.stderr)
        return 2
    # F1 fix-forward: default full-context descriptors from the bench bundle's captured
    # run-manifest.json when the operator left the placeholder defaults ("unknown"/None,
    # or a 'full' cudagraph default that an eager run-manifest should correct). Operator
    # values otherwise win.
    _dd = _run_manifest_descriptor_defaults(bench_dir)
    if _dd:
        if args.kv_cache_dtype in (None, "unknown") and _dd.get("kv_cache_dtype"):
            args.kv_cache_dtype = _dd["kv_cache_dtype"]
        if (not args.image or args.image == "unknown") and _dd.get("image"):
            args.image = _dd["image"]
        if args.gpu_memory_utilization is None and _dd.get("gpu_memory_utilization") is not None:
            args.gpu_memory_utilization = _dd["gpu_memory_utilization"]
        if _dd.get("cudagraph_mode") == "eager":
            args.cudagraph_mode = "eager"
    try:
        result = import_workloads(
            bench_dir, campaign_dir,
            model=args.model, hardware=args.hardware,
            tensor_parallel=args.tensor_parallel, quant=args.quant,
            parallel_strategy=args.parallel_strategy,
            max_num_batched_tokens=args.max_num_batched_tokens,
            kv_cache_dtype=args.kv_cache_dtype, image=args.image,
            cudagraph_mode=args.cudagraph_mode,
            gpu_memory_utilization=args.gpu_memory_utilization,
            bench_backend=args.bench_backend, dry_run=args.dry_run,
        ).to_dict()
    except (FileNotFoundError, ValueError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    payload = {
        "tool": "perf_tune_report_import_workloads",
        "library": "perf_tune_report",
        "verb": "import_workloads",
        "safety": CONTRACT["import_workloads"]["safety"],
        **result,
    }
    emit(payload, as_json=args.json)
    return 0


def cmd_fleet_leaderboard(args: argparse.Namespace) -> int:
    """Render the cross-model fleet leaderboards (AA latency + throughput + Pareto)."""
    from tools.perf_tune_report.fleet_leaderboard import read_all_rows, write_leaderboards

    campaigns_root = resolve_campaigns_dir(args.campaigns_dir)
    if not campaigns_root.is_dir():
        print(f"FATAL: campaigns dir does not exist: {campaigns_root}", file=sys.stderr)
        return 2

    hw = args.hardware
    rows = read_all_rows(campaigns_root, hardware_filter=hw)
    if not rows:
        print(f"FATAL: no {hw} atlas rows found under {campaigns_root}", file=sys.stderr)
        return 2

    # Resolve the $/GPU-hour: --gpu-hr override wins, else perf-tune-report/configs/cost.yaml.
    from tools.perf_tune_report.fleet_leaderboard import resolve_gpu_hr
    gpu_hr = resolve_gpu_hr(hw, campaigns_root.parent / "configs", args.gpu_hr)

    # Default output: the perf-report bundle (campaigns_root's parent), alongside configs.
    out_dir = Path(args.out).expanduser().resolve() if args.out else campaigns_root.parent
    result = write_leaderboards(rows, out_dir, hw=hw, gpu_hr=gpu_hr)

    payload = {
        "tool": "perf_tune_report_fleet_leaderboard",
        "library": "perf_tune_report",
        "verb": "fleet_leaderboard",
        "safety": CONTRACT["fleet_leaderboard"]["safety"],
        "campaigns_dir": str(campaigns_root),
        "hardware": hw,
        "atlas_rows": len(rows),
        **result,
    }
    emit(payload, as_json=args.json)
    return 0


def cmd_campaign_run(args: argparse.Namespace) -> int:
    """Run the full campaign orchestrator (Phase 2b)."""
    if not args.dry_run and not getattr(args, "i_understand_this_submits_jobs", False):
        print(
            "FATAL: campaign_run is ack-gated (safety=submits_jobs). "
            "Pass --i-understand-this-submits-jobs to actually run, "
            "or --dry-run to print the plan without submitting jobs.",
            file=sys.stderr,
        )
        return 2

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        print(f"FATAL: --config does not exist: {config_path}", file=sys.stderr)
        return 2

    config = load_yaml(config_path)
    campaign_cfg = config.get("campaign", {})
    cells_cfg = config.get("cells", [])

    if not cells_cfg:
        print("FATAL: config has no 'cells' list", file=sys.stderr)
        return 2

    # Build CellPlan objects
    cells: list[CellPlan] = []
    for c in cells_cfg:
        try:
            cells.append(CellPlan(
                id=c["id"],
                backend=c.get("backend", "vllm-sweep"),
                concurrencies=tuple(c.get("concurrencies", [])),
                helm_overrides=c.get("helm_overrides", {}) or {},
                profile=c.get("profile", {}) or {},
                notes=c.get("notes", ""),
                backend_config=c.get("aa") or c.get("aiperf") or {},
            ))
        except (KeyError, ValueError) as exc:
            print(f"FATAL: cell config invalid: {exc}", file=sys.stderr)
            return 2

    campaigns_root = resolve_campaigns_dir(args.campaigns_dir)
    campaign_dir = resolve_campaign_dir(args.campaign, campaigns_root)

    if args.dry_run:
        # In dry-run mode we DON'T construct the StepFns (which would require
        # cluster credentials) — we just emit the 10-step plan in JSON.
        plan: list[dict[str, Any]] = []
        endpoint_only_backends = ("aiperf", "aa")
        for cell in cells:
            endpoint_only = cell.backend in endpoint_only_backends
            plan.append({
                "cell_id": cell.id,
                "backend": cell.backend,
                "concurrencies": list(cell.concurrencies),
                "steps": [
                    "1. drain Slurm-on-K8s co-tenants" if campaign_cfg.get("drain_nodes") else "1. (no Slurm-on-K8s nodes; drain skipped)",
                    f"2. (backend={cell.backend} targets existing endpoint; helm_upgrade skipped)" if endpoint_only else "2. helm upgrade with cell helm_overrides",
                    f"3. (backend={cell.backend} targets existing endpoint; warmup skipped)" if endpoint_only else "3. warmup probe (5-prompt c=4)",
                    f"4. cell_run backend={cell.backend}",
                    "5. zymtrace anchored query" if cell.profile.get("zymtrace", "on") == "on" else "5. (zymtrace disabled)",
                    "6. import_perf_bench (raw -> normalized.json)",
                    "7. atlas_aggregate (campaign rollup)",
                    "7b. dcgm_correlate (byte-grounding -> dcgm_correlation.json, pages 6/6b) if cells/<id>/dcgm-frozen.yaml present",
                    "8. report_render (after-each-cell PDF refresh)",
                    "9. baseline_record",
                    "10. baseline_diff (returns verdict)",
                    "FINALLY: Slurm-on-K8s resume (always, even on Ctrl-C)",
                ],
                "notes": cell.notes,
            })
        payload = {
            "tool": "perf_tune_report_campaign_run",
            "library": "perf_tune_report",
            "verb": "campaign_run",
            "safety": CONTRACT["campaign_run"]["safety"],
            "ack_required": True,
            "ack_field": "i_understand_this_submits_jobs",
            "campaign_dir": str(campaign_dir),
            "cells_count": len(cells),
            "dry_run": True,
            "fail_fast_on_red": not args.continue_on_red,
            "always_resume_on_exception": True,
            "plan": plan,
        }
        emit(payload, as_json=args.json)
        return 0

    # NON-DRY-RUN path: wire the production step functions (v1.21.0).
    # The step functions use real subprocess (kubectl/helm) + library calls
    # (import_bundle_auto / aggregate / render_report / perf_baseline). The
    # cluster mutation calls are timeout-bounded; the orchestrator's
    # try/finally always-resume contract still holds even if any step fails.
    from tools.perf_tune_report.orchestrator import production_step_fns

    step_fns = production_step_fns()

    target_namespace = campaign_cfg.get("target_namespace", "inference")
    target_release = campaign_cfg.get("target_release", "")
    if not target_release:
        print(
            "FATAL: campaign.target_release is required for non-dry-run "
            "(must name the helm release to upgrade per cell).",
            file=sys.stderr,
        )
        return 2

    chart_dir_str = campaign_cfg.get("chart_dir", "")
    if not chart_dir_str:
        print(
            "FATAL: campaign.chart_dir is required for non-dry-run "
            "(must point at the helm chart directory).",
            file=sys.stderr,
        )
        return 2
    chart_dir = Path(chart_dir_str).expanduser().resolve()

    base_values_str = campaign_cfg.get("base_values", "")
    base_values = Path(base_values_str).expanduser().resolve() if base_values_str else Path()

    drain_nodes = tuple(campaign_cfg.get("drain_nodes", []) or ())
    comparator = campaign_cfg.get("comparator_baseline", "") or ""

    from tools.perf_tune_report.orchestrator import run_campaign

    try:
        result = run_campaign(
            cells,
            campaign_dir=campaign_dir,
            target_namespace=target_namespace,
            target_release=target_release,
            chart_dir=chart_dir,
            base_values=base_values,
            drain_nodes=drain_nodes,
            comparator_baseline=comparator,
            step_fns=step_fns,
            continue_on_red=args.continue_on_red,
            dry_run=False,
        )
    except ValueError as exc:
        print(f"FATAL: campaign_run failed: {exc}", file=sys.stderr)
        return 2

    payload: dict[str, Any] = {
        "tool": "perf_tune_report_campaign_run",
        "library": "perf_tune_report",
        "verb": "campaign_run",
        "safety": CONTRACT["campaign_run"]["safety"],
        "ack_required": True,
        "ack_field": "i_understand_this_submits_jobs",
        "dry_run": False,
    }
    payload.update(result.to_dict())
    emit(payload, as_json=args.json)
    # Return non-zero only when the campaign was unable to complete any cell.
    return 0 if result.cells_completed > 0 else 1


def cmd_graph_diff(args: argparse.Namespace) -> int:
    """Diff two torch.compile dynamo+inductor log dumps (v1.21.0)."""
    side_a = Path(args.side_a_log).expanduser().resolve()
    side_b = Path(args.side_b_log).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not side_a.is_file():
        print(f"FATAL: --side-a-log does not exist: {side_a}", file=sys.stderr)
        return 2
    if not side_b.is_file():
        print(f"FATAL: --side-b-log does not exist: {side_b}", file=sys.stderr)
        return 2

    try:
        result = diff_graph_logs(
            side_a_log=side_a,
            side_b_log=side_b,
            output_dir=output_dir,
            side_a_label=args.side_a_label or "side-A",
            side_b_label=args.side_b_label or "side-B",
            notes=args.notes or "",
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    payload: dict[str, Any] = {
        "tool": "perf_tune_report_graph_diff",
        "library": "perf_tune_report",
        "verb": "graph_diff",
        "safety": CONTRACT["graph_diff"]["safety"],
    }
    payload.update(result.to_dict())
    payload["dry_run"] = args.dry_run
    emit(payload, as_json=args.json)
    return 0


def cmd_kernel_reproducer_scaffold(args: argparse.Namespace) -> int:
    """Scaffold a standalone CUDA/CUTLASS kernel reproducer .cu + build script (v1.69.0)."""
    output_dir = Path(args.output_dir).expanduser().resolve()
    try:
        result = scaffold_reproducer(
            kernel_name=args.kernel_name,
            header=args.header,
            out_dir=output_dir,
            mma_m=args.mma_m,
            mma_n=args.mma_n,
            batch=args.batch,
            out=args.out_dim,
            k=args.k,
            mirage_tree=args.mirage_tree,
            arch=args.arch,
            dry_run=args.dry_run,
        )
    except (ValueError, OSError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    payload: dict[str, Any] = {
        "tool": "perf_tune_report_kernel_reproducer_scaffold",
        "library": "perf_tune_report",
        "verb": "kernel_reproducer_scaffold",
        "safety": CONTRACT["kernel_reproducer_scaffold"]["safety"],
    }
    payload.update(result.to_dict())
    payload["dry_run"] = args.dry_run
    emit(payload, as_json=args.json)
    return 0


def cmd_kernel_profile(args: argparse.Namespace) -> int:
    """Capture per-kernel CUDA profile from a live vLLM pod (v1.21.0)."""
    if not args.dry_run and not getattr(args, "i_understand_this_submits_jobs", False):
        print(
            "FATAL: kernel_profile is ack-gated (safety=submits_jobs). "
            "Pass --i-understand-this-submits-jobs to actually attach the "
            "sidecar, or --dry-run to print the commands without executing.",
            file=sys.stderr,
        )
        return 2

    bundle_path: Path | None = None
    if args.bundle:
        bundle_path = Path(args.bundle).expanduser().resolve()
        if not bundle_path.is_dir():
            print(f"FATAL: --bundle does not exist: {bundle_path}", file=sys.stderr)
            return 2

    output_dir = Path(args.output_dir).expanduser().resolve()

    try:
        result = capture_kernel_profile(
            namespace=args.namespace,
            pod=args.pod,
            target_container=args.target_container,
            output_dir=output_dir,
            sidecar_image=args.sidecar_image,
            duration_s=args.duration_seconds,
            sample=args.sample,
            trace=args.trace,
            sampling_frequency=args.sampling_frequency,
            vllm_pid_pattern=args.vllm_pid_pattern,
            bundle=bundle_path,
            dry_run=args.dry_run,
        )
    except RuntimeError as exc:
        print(f"FATAL: kernel_profile step failed: {exc}", file=sys.stderr)
        return 2

    payload: dict[str, Any] = {
        "tool": "perf_tune_report_kernel_profile",
        "library": "perf_tune_report",
        "verb": "kernel_profile",
        "safety": CONTRACT["kernel_profile"]["safety"],
        "ack_required": True,
        "ack_field": "i_understand_this_submits_jobs",
    }
    payload.update(result.to_dict())
    emit(payload, as_json=args.json)
    return 0


def _run_manifest_descriptor_defaults(bundle: Path) -> dict[str, Any]:
    """F1 fix-forward: best-effort full-context descriptors from a capture-run-env
    ``run-manifest.json`` found in the bench bundle (or up to 3 parents). Lets an
    import default ``image``/``kv_cache_dtype``/``gpu_memory_utilization``/
    ``cudagraph_mode`` from the ACTUAL captured deploy env when the operator omits the
    CLI flags -- real captured values, never guesses. ``enforce_eager`` maps to
    ``cudagraph_mode`` (eager | the vLLM-default 'full'). Absent/unreadable -> {}."""
    base = bundle if bundle.is_dir() else bundle.parent
    seen: list[Path] = []
    for d in (base, *list(base.parents)[:3]):
        seen += [d / "run-manifest.json", d / "run-env.json"]
    for p in seen:
        if not p.is_file():
            continue
        try:
            m = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(m, dict):
            continue
        out: dict[str, Any] = {}
        if m.get("image"):
            out["image"] = m["image"]
        kv = m.get("kv_cache_dtype")
        if isinstance(kv, str) and kv and kv != "unknown":
            out["kv_cache_dtype"] = kv
        gmu = m.get("gpu_memory_utilization")
        if isinstance(gmu, (int, float)) and not isinstance(gmu, bool):
            out["gpu_memory_utilization"] = float(gmu)
        ee = m.get("enforce_eager")
        if isinstance(ee, bool):
            out["cudagraph_mode"] = "eager" if ee else "full"
        if out:
            return out
    return {}


def cmd_import_perf_bench(args: argparse.Namespace) -> int:
    """Import an existing inference-perf-bench bundle into a campaign."""
    campaigns_root = resolve_campaigns_dir(args.campaigns_dir)
    campaign_dir = resolve_campaign_dir(args.campaign, campaigns_root)
    bundle_path = Path(args.bundle).expanduser().resolve()

    # Build override dict from CLI flags. Only include fields the operator
    # explicitly passed; let the importer fall back to bundle metadata for
    # everything else.
    overrides: dict[str, Any] = {}
    if args.cell_id:
        overrides["cell_id"] = args.cell_id
    if args.model:
        overrides["model"] = args.model
    if args.hardware:
        overrides["hardware"] = args.hardware
    if args.quant:
        overrides["quant"] = args.quant
    if args.tensor_parallel is not None:
        overrides["tensor_parallel"] = args.tensor_parallel
    if args.parallel_strategy:
        overrides["parallel_strategy"] = args.parallel_strategy
    if args.mtp is not None:
        overrides["mtp"] = args.mtp
    if args.max_num_batched_tokens is not None:
        overrides["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.max_num_seqs is not None:
        overrides["max_num_seqs"] = args.max_num_seqs
    if args.patched_vllm_enabled is not None:
        overrides["patched_vllm_enabled"] = args.patched_vllm_enabled
    if args.notes:
        overrides["notes"] = args.notes
    if getattr(args, "cache_mode", None):
        overrides["cache_mode"] = args.cache_mode
    # Full-context descriptor overrides (2026-06-07).
    for _fld in (
        "dataset",
        "cudagraph_mode",
        "enforce_eager",
        "gpu_memory_utilization",
        "kv_cache_dtype",
        "image",
        "delivery",
        "overlay_mode",
        "patch_files",
        "data_parallel",
        "pipeline_parallel",
    ):
        _val = getattr(args, _fld, None)
        if _val is not None:
            overrides[_fld] = _val
    if getattr(args, "expected_reqs", None):
        overrides["expected_reqs"] = args.expected_reqs

    # F1 fix-forward: fill any full-context descriptor the operator did NOT pass from
    # the bundle's captured run-manifest.json (capture-run-env.sh). Operator CLI flags
    # always win (setdefault only fills gaps); real captured values, never guesses.
    for _k, _v in _run_manifest_descriptor_defaults(bundle_path).items():
        overrides.setdefault(_k, _v)

    try:
        result = import_bundle_auto(
            bundle=bundle_path,
            campaign_dir=campaign_dir,
            overrides=overrides,
            dry_run=args.dry_run,
            concurrency_override=getattr(args, "concurrency", None),
            require_plot_ready=getattr(args, "require_plot_ready", False),
        )
    except ValueError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    # Determine which importer was dispatched (so the payload tells the
    # operator which code path actually ran). The drive_load result has an
    # importer="inference_drive_load" attribute; the bench-serve one does
    # not (it predates v1.21.0).
    importer_name = getattr(result, "importer", "inference_perf_bench")

    payload = {
        "tool": "perf_tune_report_import_perf_bench",
        "library": "perf_tune_report",
        "verb": "import_perf_bench",
        "safety": CONTRACT["import_perf_bench"]["safety"],
        "importer": importer_name,
        "campaign_dir": str(result.campaign_dir),
        # lws_summary emits multiple cells -> LwsSummaryImportResult has no single
        # cell_id/cell_dir/normalized_path; guard so the multi-cell path doesn't crash.
        "cell_id": getattr(result, "cell_id", None),
        "cell_dir": (str(result.cell_dir) if getattr(result, "cell_dir", None) else None),
        "normalized_path": (
            str(result.normalized_path)
            if getattr(result, "normalized_path", None)
            else None
        ),
        "bundle_path": (str(result.bundle_path) if getattr(result, "bundle_path", None) else None),
        "row_count": result.row_count,
        "concurrencies": getattr(result, "concurrencies", None),
        "k_values": getattr(result, "k_values", None),
        "status": result.status,
        "dry_run": args.dry_run,
    }
    emit(payload, as_json=args.json)
    return 0


def cmd_import_variant_ab(args: argparse.Namespace) -> int:
    """Import a cross-engine variant A/B bundle (one engine-tagged cell per arm)."""
    campaigns_root = resolve_campaigns_dir(args.campaigns_dir)
    campaign_dir = resolve_campaign_dir(args.campaign, campaigns_root)
    bundle_path = Path(args.bundle).expanduser().resolve()

    if not detect_variant_ab(bundle_path):
        print(
            f"FATAL: {bundle_path} is not a variant-A/B bundle (no <arm>/c<C>-t<T>.txt "
            "subdirs). Use import_perf_bench for sweep/drive_load/aiperf layouts.",
            file=sys.stderr,
        )
        return 2

    overrides: dict[str, Any] = {}
    for attr in (
        "model", "hardware", "quant", "tensor_parallel", "parallel_strategy",
        "mtp", "max_num_batched_tokens", "cache_mode", "notes",
        "dataset", "cudagraph_mode", "enforce_eager", "gpu_memory_utilization",
        "kv_cache_dtype", "image", "delivery", "overlay_mode", "patch_files",
        "data_parallel", "pipeline_parallel",
    ):
        val = getattr(args, attr, None)
        if val is not None:
            overrides[attr] = val

    try:
        result = import_variant_ab_bundle(
            bundle=bundle_path,
            campaign_dir=campaign_dir,
            overrides=overrides,
            dry_run=args.dry_run,
            require_plot_ready=getattr(args, "require_plot_ready", False),
        )
    except ValueError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    payload = {
        "tool": "perf_tune_report_import_variant_ab",
        "library": "perf_tune_report",
        "verb": "import_variant_ab",
        "safety": CONTRACT["import_variant_ab"]["safety"],
        "importer": "variant_ab",
        "campaign_dir": str(result.campaign_dir),
        "bundle_path": str(result.bundle_path),
        "cells": result.cells,
        "cell_count": len(result.cells),
        "row_count": result.row_count,
        "concurrencies": result.concurrencies,
        "status": result.status,
        "dry_run": args.dry_run,
    }
    emit(payload, as_json=args.json)
    return 0


def cmd_import_roofline_sweep(args: argparse.Namespace) -> int:
    """Import a prefill+decode roofline sweep bundle into a campaign cell."""
    campaigns_root = resolve_campaigns_dir(args.campaigns_dir)
    campaign_dir = resolve_campaign_dir(args.campaign, campaigns_root)
    bundle_path = Path(args.bundle).expanduser().resolve()

    overrides: dict[str, Any] = {}
    for attr in (
        "cell_id", "model", "hardware", "quant", "kv_dtype", "tensor_parallel",
        "parallel_strategy", "mtp", "max_num_batched_tokens", "cache_mode",
        "model_config_path", "kv_cache_dtype",
        # full-context descriptors (so roofline cells can pass publish --strict)
        "dataset", "cudagraph_mode", "enforce_eager", "gpu_memory_utilization",
        "image", "delivery", "overlay_mode", "patch_files",
        "data_parallel", "pipeline_parallel",
    ):
        val = getattr(args, attr, None)
        if val is not None:
            overrides[attr] = val

    # F1 fix-forward: fill any full-context descriptor the operator did not pass from
    # the sweep bundle's captured run-manifest.json (capture-run-env.sh). Operator flags win.
    for _k, _v in _run_manifest_descriptor_defaults(bundle_path).items():
        overrides.setdefault(_k, _v)

    try:
        result = import_roofline_sweep_bundle(
            bundle=bundle_path,
            campaign_dir=campaign_dir,
            overrides=overrides,
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    payload = {
        "tool": "perf_tune_report_import_roofline_sweep",
        "library": "perf_tune_report",
        "verb": "import_roofline_sweep",
        "safety": CONTRACT["import_roofline_sweep"]["safety"],
        "campaign_dir": str(result.campaign_dir),
        "cell_id": result.cell_id,
        "cell_dirs": [str(x) for x in result.cell_dirs],
        "decode_points": result.decode_points,
        "prefill_points": result.prefill_points,
        "status": result.status,
        "dry_run": args.dry_run,
    }
    emit(payload, as_json=args.json)
    return 0


def _resolve_ceilings_yaml(explicit: str | None, campaign_dir: Path) -> Path | None:
    """Locate sol-ceilings.yaml: --ceilings, then SOL_CEILINGS_YAML env, then
    walk up from the campaign dir for configs/sol-ceilings.yaml."""
    if explicit:
        p = Path(explicit).expanduser().resolve()
        return p if p.is_file() else None
    env_override = os.environ.get("SOL_CEILINGS_YAML", "").strip()
    if env_override and env_override != "disable":
        p = Path(env_override).expanduser().resolve()
        return p if p.is_file() else None
    # Canonical bundle name first, then a name-agnostic bundle-root fallback
    # (<bundle>/configs/sol-ceilings.yaml) so a future submodule rename needs no
    # code edit; the campaign's own bundle root is reached before higher ancestors.
    relcands = (Path("perf-tune-report") / "configs" / "sol-ceilings.yaml",
                Path("configs") / "sol-ceilings.yaml")
    cur = campaign_dir.resolve()
    for parent in [cur, *cur.parents]:
        for relpath in relcands:
            candidate = parent / relpath
            if candidate.is_file():
                return candidate
    return None


def cmd_dcgm_correlate(args: argparse.Namespace) -> int:
    """Fold a frozen DCGM YAML into a cell's dcgm_correlation.json (page 6/6b)."""
    from tools.perf_tune_report.dcgm_correlate import (
        FrozenYamlMalformed,
        correlate_from_frozen,
        write_correlation,
    )
    from tools.perf_tune_report.renderer.sol_roofline import load_ceilings

    campaigns_root = resolve_campaigns_dir(args.campaigns_dir)
    campaign_dir = resolve_campaign_dir(args.campaign, campaigns_root)
    cell_dir = campaign_dir / "cells" / args.cell_id
    if not cell_dir.is_dir():
        print(f"FATAL: cell dir not found: {cell_dir}", file=sys.stderr)
        return 2

    frozen = Path(args.frozen_yaml).expanduser().resolve()
    if not frozen.is_file():
        print(f"FATAL: --frozen-yaml does not exist: {frozen}", file=sys.stderr)
        return 2

    ceilings_path = _resolve_ceilings_yaml(args.ceilings, campaign_dir)
    if ceilings_path is None:
        print(
            "FATAL: could not locate sol-ceilings.yaml. Pass --ceilings, set "
            "SOL_CEILINGS_YAML, or run from a tree containing "
            "configs/sol-ceilings.yaml.",
            file=sys.stderr,
        )
        return 2
    ceilings = load_ceilings(ceilings_path)

    # kernels.json: explicit override, else the cell's own (for page 6b).
    kernels_json_path: Path | None = None
    if args.kernels_json:
        kernels_json_path = Path(args.kernels_json).expanduser().resolve()
        if not kernels_json_path.is_file():
            print(f"FATAL: --kernels-json does not exist: {kernels_json_path}", file=sys.stderr)
            return 2
    elif (cell_dir / "kernels.json").is_file():
        kernels_json_path = cell_dir / "kernels.json"

    if args.dry_run:
        payload = {
            "tool": "perf_tune_report_dcgm_correlate",
            "library": "perf_tune_report",
            "verb": "dcgm_correlate",
            "safety": CONTRACT["dcgm_correlate"]["safety"],
            "campaign_dir": str(campaign_dir),
            "cell_id": args.cell_id,
            "frozen_yaml": str(frozen),
            "ceilings": str(ceilings_path),
            "kernels_json": str(kernels_json_path) if kernels_json_path else None,
            "dry_run": True,
            "would_write": str(cell_dir / "dcgm_correlation.json"),
        }
        emit(payload, as_json=args.json)
        return 0

    try:
        result = correlate_from_frozen(
            frozen, ceilings, cell_dir=cell_dir, kernels_json_path=kernels_json_path
        )
    except FrozenYamlMalformed as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    out_path = write_correlation(result, cell_dir)

    d = result.to_dict()
    payload = {
        "tool": "perf_tune_report_dcgm_correlate",
        "library": "perf_tune_report",
        "verb": "dcgm_correlate",
        "safety": CONTRACT["dcgm_correlate"]["safety"],
        "campaign_dir": str(campaign_dir),
        "cell_id": args.cell_id,
        "frozen_yaml": str(frozen),
        "ceilings": str(ceilings_path),
        "kernels_json": str(kernels_json_path) if kernels_json_path else None,
        "dcgm_correlation_json": str(out_path),
        "captured_sources": d.get("captured_sources"),
        "n_resources": len(d.get("resources", [])),
        "per_category_attribution_rows": len(d.get("per_category_attribution", [])),
        "dry_run": False,
    }
    emit(payload, as_json=args.json)
    return 0


def cmd_import_nsys(args: argparse.Namespace) -> int:
    """Import an nsys cuda_gpu_kern_sum into a campaign cell (page-3/6b input)."""
    from tools.perf_tune_report.importers.nsys_kernels import (
        NsysKernSumMalformed,
        NsysKernSumMissing,
    )

    campaigns_root = resolve_campaigns_dir(args.campaigns_dir)
    campaign_dir = resolve_campaign_dir(args.campaign, campaigns_root)
    bundle_path = Path(args.bundle).expanduser().resolve()
    cell_dir = campaign_dir / "cells" / args.cell_id

    try:
        result = import_nsys_kernels(
            bundle_path,
            cell_dir,
            kern_sum_name=args.kern_sum_name,
            dry_run=args.dry_run,
        )
    except (NsysKernSumMissing, NsysKernSumMalformed) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    payload = {
        "tool": "perf_tune_report_import_nsys",
        "library": "perf_tune_report",
        "verb": "import_nsys",
        "safety": CONTRACT["import_nsys"]["safety"],
        "campaign_dir": str(campaign_dir),
        "cell_id": args.cell_id,
        "cell_dir": str(cell_dir),
        "bundle_path": str(bundle_path),
        "kernels_json_path": (
            str(result.kernels_json_path) if result.kernels_json_path else None
        ),
        "top_kernel_count": result.top_kernel_count,
        "category_count": result.category_count,
        "skipped_reason": result.skipped_reason,
        "dry_run": args.dry_run,
    }
    emit(payload, as_json=args.json)
    return 0


def cmd_import_ncu(args: argparse.Namespace) -> int:
    """Import an ncu per-kernel bundle into a campaign cell (page-5 input)."""
    from tools.perf_tune_report.importers.ncu_kernels import NcuCsvMalformed, NcuCsvMissing

    campaigns_root = resolve_campaigns_dir(args.campaigns_dir)
    campaign_dir = resolve_campaign_dir(args.campaign, campaigns_root)
    bundle_path = Path(args.bundle).expanduser().resolve()
    cell_dir = campaign_dir / "cells" / args.cell_id

    try:
        result = import_ncu_kernels(
            bundle_path,
            cell_dir,
            hw_key=args.hw_key,
            dry_run=args.dry_run,
        )
    except (NcuCsvMissing, NcuCsvMalformed) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    payload = {
        "tool": "perf_tune_report_import_ncu",
        "library": "perf_tune_report",
        "verb": "import_ncu",
        "safety": CONTRACT["import_ncu"]["safety"],
        "campaign_dir": str(campaign_dir),
        "cell_id": args.cell_id,
        "cell_dir": str(cell_dir),
        "bundle_path": str(bundle_path),
        "hw_key": args.hw_key,
        "ncu_kernels_json_path": (
            str(result.ncu_kernels_json_path) if result.ncu_kernels_json_path else None
        ),
        "kernel_count": result.kernel_count,
        "skipped_reason": result.skipped_reason,
        "dry_run": args.dry_run,
    }
    emit(payload, as_json=args.json)
    return 0


def cmd_raw_bench_compare(args: argparse.Namespace) -> int:
    """v1.24.0: render a multi-bundle vllm-bench-serve comparison PDF."""
    from tools.perf_tune_report.raw_bench_compare import (
        RawBenchCompareManifestMalformed,
        render_comparison,
    )

    manifest = Path(args.manifest).expanduser().resolve()
    out_pdf = Path(args.out).expanduser().resolve()
    try:
        result = render_comparison(manifest, out_pdf)
    except FileNotFoundError as e:
        print(f"FATAL: manifest not found: {e}", file=sys.stderr)
        return 2
    except RawBenchCompareManifestMalformed as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 1

    payload = {
        "tool": "perf_tune_report_raw_bench_compare",
        "library": "perf_tune_report",
        "verb": "raw_bench_compare",
        "safety": CONTRACT["raw_bench_compare"]["safety"],
        **result.to_dict(),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for key, val in payload.items():
            if key == "peaks":
                continue
            print(f"{key}: {val}")
        if result.peaks:
            print("peaks:")
            for p in result.peaks:
                pct = p.get("pct_vs_baseline")
                pct_str = f" {pct:+.1f}%" if pct is not None else ""
                print(f"  {p['short']:24s} {p['peak_output_tps']:>8.0f} tok/s  @ c={p['peak_c']}{pct_str}")
    return 0


def cmd_report_smoke(args: argparse.Namespace) -> int:
    import shutil
    import tempfile

    from tools.perf_tune_report.renderer.render_report import render_report

    fixture = synthetic_fixture_path()
    if not fixture.is_file():
        print(f"FATAL: bundled synthetic fixture missing: {fixture}", file=sys.stderr)
        return 2
    out_pdf = Path(args.out).expanduser() if args.out else Path("/tmp/perftunereport-smoke.pdf")
    # Render from a temp copy so the read-only smoke never writes
    # report_status.json into the bundled fixtures dir.
    with tempfile.TemporaryDirectory(prefix="perftunereport-smoke-") as tmp:
        tmp_atlas = Path(tmp) / "atlas.jsonl"
        shutil.copy2(fixture, tmp_atlas)
        render_report(
            tmp_atlas,
            out_pdf,
            title=args.title or "glm5p1 benchmark report",
            variants_line=(
                "Variants: GLM-5.1-FP8 (FP8, H100) | "
                "GLM-5.1-NVFP4 (NVFP4, B200) | "
                "GLM-5.1-NVFP4 (NVFP4, GB300)"
            ),
            data_source_line=(
                "Data source: synthetic chat | each concurrency point targeted 600s "
                "steady-state | prompt 28.8k shared + 3.2k unique = 32k input | "
                "OSL 4k | cache target 90% | tokenizer zai-org/GLM-5.1"
            ),
        )
    rows = read_jsonl(fixture)
    coverage = summarize(rows)
    payload = {
        "tool": "perf_tune_report_report_smoke",
        "library": "perf_tune_report",
        "verb": "report_smoke",
        "safety": CONTRACT["report_smoke"]["safety"],
        "fixture": str(fixture),
        "out_pdf": str(out_pdf),
        "size_bytes": out_pdf.stat().st_size,
        "coverage": {
            "atlas_cells": coverage.atlas_cells,
            "full_sweeps": coverage.full_sweeps,
            "partial_sweeps": coverage.partial_sweeps,
            "failed_cells": coverage.failed_cells,
            "plot_ready_points": coverage.plot_ready_points,
            "evicted_cells": coverage.evicted_cells,
            "header_line": coverage.header_line(),
        },
    }
    emit(payload, as_json=args.json)
    return 0


# ----------------------------------------------------------------------------
# Parser plumbing
# ----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="perftunereport",
        description=(
            "Build multi-page benchmark report PDFs from vllm-sweep / AIPerf "
            "output. Backs the inference-perf-tune-report skill + 5 MCP tools."
        ),
    )
    sub = parser.add_subparsers(dest="verb", required=True)

    # campaign_init
    ci = sub.add_parser("campaign_init", description=CONTRACT["campaign_init"]["description"])
    ci.add_argument("--config", required=True, help="Path to a campaign YAML config")
    ci.add_argument("--slug", default=None, help="Override the campaign slug (default: config name)")
    ci.add_argument(
        "--experiment-id",
        default=None,
        help="Experiment-id = evidence-bundle run-id (the single join key). When "
        "set, the campaign dir is named EXACTLY this so campaign_id == experiment-id "
        "joins back to the cluster objects (experiment=<id-slug>) + the bundle. Must "
        "contain a YYYYMMDDTHHMMSSZ stamp. Falls back to config experiment_id:.",
    )
    ci.add_argument(
        "--family",
        default=None,
        help="Experiment family for cross-experiment grouping (e.g. nvfp4-kv, "
        "warp-decode, deepep). Falls back to config family:.",
    )
    ci.add_argument(
        "--evidence-bundle",
        default=None,
        help="Path to the evidence bundle this campaign is derived from. Falls back "
        "to config evidence_bundle_path:.",
    )
    ci.add_argument("--campaigns-dir", default=None, help="Override the campaigns root")
    ci.add_argument("--json", action="store_true", help="Emit JSON envelope")
    ci.set_defaults(func=cmd_campaign_init)

    # experiments_index
    ei = sub.add_parser("experiments_index",
                        description=CONTRACT["experiments_index"]["description"])
    ei.add_argument("--family", default=None,
                    help="Only include experiments in this family (e.g. nvfp4-kv)")
    ei.add_argument("--out", default=None,
                    help="Output dir (default: the perf-report bundle, campaigns' parent)")
    ei.add_argument("--include-s3", action="store_true",
                    help="Also enumerate the lake's campaign_v1 prefix to mark each row's "
                    "published_to_lake (best-effort; warns + continues if S3 creds absent)")
    ei.add_argument("--s3-endpoint", default=None, help="S3 endpoint (with --include-s3)")
    ei.add_argument("--s3-bucket", default=None, help="S3 bucket (with --include-s3)")
    ei.add_argument("--s3-access-key-file", default=None,
                    help="File with S3 access key (with --include-s3)")
    ei.add_argument("--s3-secret-key-file", default=None,
                    help="File with S3 secret key (with --include-s3)")
    ei.add_argument("--campaigns-dir", default=None, help="Override the campaigns root")
    ei.add_argument("--json", action="store_true", help="Emit JSON envelope")
    ei.set_defaults(func=cmd_experiments_index)

    # experiment_inventory
    einv = sub.add_parser("experiment_inventory",
                          description=CONTRACT["experiment_inventory"]["description"])
    einv.add_argument("--bundle-root", action="append", default=None,
                      help="Evidence-bundle tree root to also walk for run-id-stamped bundles "
                      "(repeatable; e.g. a *-deploy dir). Campaigns are always counted.")
    einv.add_argument("--out", default=None,
                      help="Output dir (default: the perf-report bundle, campaigns' parent)")
    einv.add_argument("--include-s3", action="store_true",
                      help="Also enumerate the lake's campaign_v1 prefix for the published count "
                      "(best-effort; warns + continues if S3 creds absent)")
    einv.add_argument("--s3-endpoint", default=None, help="S3 endpoint (with --include-s3)")
    einv.add_argument("--s3-bucket", default=None, help="S3 bucket (with --include-s3)")
    einv.add_argument("--s3-access-key-file", default=None,
                      help="File with S3 access key (with --include-s3)")
    einv.add_argument("--s3-secret-key-file", default=None,
                      help="File with S3 secret key (with --include-s3)")
    einv.add_argument("--campaigns-dir", default=None, help="Override the campaigns root")
    einv.add_argument("--json", action="store_true", help="Emit JSON envelope")
    einv.set_defaults(func=cmd_experiment_inventory)

    # import_model_eval
    ime = sub.add_parser("import_model_eval",
                         description=CONTRACT["import_model_eval"]["description"])
    ime.add_argument("--results", required=True, help="lm-eval-harness results.json path")
    ime.add_argument("--campaign", required=True, help="Campaign dir path OR slug")
    ime.add_argument("--model", required=True, help="Model (matches the serving campaign's model)")
    ime.add_argument("--hardware", required=True, help="Hardware token (GB300 / B200)")
    ime.add_argument("--quant", required=True, help="Quant (NVFP4 / FP8 / ...)")
    ime.add_argument("--tensor-parallel", type=int, default=1, dest="tensor_parallel")
    ime.add_argument("--cell-id", default="model-eval", dest="cell_id")
    ime.add_argument("--parallel-strategy", default="TP", dest="parallel_strategy",
                     choices=("TP", "EP"))
    ime.add_argument("--kv-cache-dtype", default="unknown", dest="kv_cache_dtype")
    ime.add_argument("--image", default="unknown", help="Serving image tag the eval ran against")
    ime.add_argument("--campaigns-dir", default=None, help="Override the campaigns root")
    ime.add_argument("--json", action="store_true", help="Emit JSON envelope")
    ime.set_defaults(func=cmd_import_model_eval)

    # import_workloads
    iw = sub.add_parser("import_workloads",
                        description=CONTRACT["import_workloads"]["description"])
    iw.add_argument("--bench-dir", required=True, dest="bench_dir",
                    help="bench-all-workloads.sh output dir (<tag>-c<c>.txt + bench-workloads.json)")
    iw.add_argument("--campaign", required=True, help="Campaign dir path OR slug")
    iw.add_argument("--model", required=True, help="Served model (matches the campaign's model)")
    iw.add_argument("--hardware", required=True, help="Hardware token (GB300 / B200)")
    iw.add_argument("--tensor-parallel", required=True, type=int, dest="tensor_parallel")
    iw.add_argument("--quant", default="NVFP4", help="Quant (NVFP4 / FP8 / ...)")
    iw.add_argument("--parallel-strategy", default="TP", dest="parallel_strategy",
                    choices=("TP", "EP"))
    iw.add_argument("--max-num-batched-tokens", type=int, default=0,
                    dest="max_num_batched_tokens")
    iw.add_argument("--kv-cache-dtype", default="unknown", dest="kv_cache_dtype",
                    help="Serve kv-cache dtype (required-context: set for a publish --strict run)")
    iw.add_argument("--image", default="unknown",
                    help="Serving image tag (required-context: set for a publish --strict run)")
    iw.add_argument("--cudagraph-mode", default="full", dest="cudagraph_mode")
    iw.add_argument("--gpu-memory-utilization", type=float, default=None,
                    dest="gpu_memory_utilization")
    iw.add_argument("--bench-backend", default="openai", dest="bench_backend",
                    help="bench CLIENT backend (bench-all-workloads.sh uses --backend openai)")
    iw.add_argument("--dry-run", action="store_true", help="Parse + report, write nothing")
    iw.add_argument("--campaigns-dir", default=None, help="Override the campaigns root")
    iw.add_argument("--json", action="store_true", help="Emit JSON envelope")
    iw.set_defaults(func=cmd_import_workloads)

    # trend_view
    tv = sub.add_parser("trend_view", description=CONTRACT["trend_view"]["description"])
    tv.add_argument("--metric", default="output_tps_per_gpu",
                    help="atlas metric to trend (output_tps_per_gpu | tpot_median_ms | ttft_avg_ms | ...)")
    tv.add_argument("--concurrency", type=int, default=None,
                    help="filter to one concurrency (else each c is its own trend line)")
    tv.add_argument("--regression-pct", type=float, default=10.0, dest="regression_pct",
                    help="abs %% move in the wrong direction that flags a regression (default 10)")
    tv.add_argument("--hardware", default="GB300", help="hardware token filter (default GB300)")
    tv.add_argument("--out", default=None, help="write the trend markdown to this path")
    tv.add_argument("--campaigns-dir", default=None, help="Override the campaigns root")
    tv.add_argument("--lake-dir", default=None,
                    help="Read published atlas_v1 parquet from a pulled lake snapshot "
                         "(joins campaign_v1 vllm_commit as the engine-version axis); "
                         "local campaigns are the default")
    tv.add_argument("--json", action="store_true", help="Emit JSON envelope")
    tv.set_defaults(func=cmd_trend_view)

    # fleet_leaderboard
    fl = sub.add_parser("fleet_leaderboard",
                        description=CONTRACT["fleet_leaderboard"]["description"])
    fl.add_argument("--hardware", default="GB300",
                    help="Hardware token to filter atlas rows + name outputs (default: GB300)")
    fl.add_argument("--gpu-hr", type=float, default=None,
                    help="$/GPU-hour for the cost columns (overrides perf-tune-report/configs/cost.yaml; "
                    "default: resolve from cost.yaml by hardware, else 8.60)")
    fl.add_argument("--out", default=None,
                    help="Output dir (default: the perf-report bundle, campaigns' parent)")
    fl.add_argument("--campaigns-dir", default=None, help="Override the campaigns root")
    fl.add_argument("--json", action="store_true", help="Emit JSON envelope")
    fl.set_defaults(func=cmd_fleet_leaderboard)

    # cell_run
    cr = sub.add_parser("cell_run", description=CONTRACT["cell_run"]["description"])
    cr.add_argument("--campaign", required=True, help="Campaign dir path OR slug")
    cr.add_argument("--cell", required=True, help="cell_id to run")
    cr.add_argument(
        "--backend",
        required=True,
        choices=("vllm-sweep", "aiperf", "aa"),
        help="Which backend driver to use for this cell",
    )
    # vllm-sweep knobs
    cr.add_argument("--serve-cmd", default=None, help="vllm-sweep --serve-cmd")
    cr.add_argument("--bench-cmd", default=None, help="vllm-sweep --bench-cmd")
    # aiperf knobs
    cr.add_argument("--namespace", default=None)
    cr.add_argument("--bench-pod", default=None)
    cr.add_argument("--kube-context", default=None)
    cr.add_argument("--endpoint-url", default=None)
    cr.add_argument("--served-model", default=None)
    cr.add_argument("--dataset-split", default=None)
    cr.add_argument("--conversation-count", type=int, default=None)
    # aa (Artificial Analysis) knobs
    cr.add_argument(
        "--aa-shape",
        dest="aa_shape",
        default=None,
        choices=("aa-1k", "aa-10k", "aa-100k"),
        help="aa backend: which AA workload shape this cell runs (or cell.aa.shape)",
    )
    cr.add_argument(
        "--aa-mode",
        dest="aa_mode",
        default=None,
        choices=("synthetic", "dataset-replay"),
        help="aa backend: synthetic prompt generation vs replay of a generated JSONL",
    )
    cr.add_argument(
        "--request-count",
        dest="request_count",
        type=int,
        default=None,
        help="aa backend: requests per concurrency point (or cell.aa.request_count)",
    )
    cr.add_argument("--dry-run", action="store_true", help="Print the command without executing")
    cr.add_argument(
        "--i-understand-this-submits-jobs",
        dest="i_understand_this_submits_jobs",
        action="store_true",
        help="Ack flag (safety_class=submits_jobs)",
    )
    cr.add_argument("--campaigns-dir", default=None, help="Override the campaigns root")
    cr.add_argument("--json", action="store_true", help="Emit JSON envelope")
    cr.set_defaults(func=cmd_cell_run)

    # atlas_aggregate
    aa = sub.add_parser("atlas_aggregate", description=CONTRACT["atlas_aggregate"]["description"])
    aa.add_argument("--campaign", required=True, help="Campaign dir path OR slug")
    aa.add_argument("--campaigns-dir", default=None, help="Override the campaigns root")
    aa.add_argument("--json", action="store_true", help="Emit JSON envelope")
    aa.set_defaults(func=cmd_atlas_aggregate)

    # report_render
    rr = sub.add_parser("report_render", description=CONTRACT["report_render"]["description"])
    rr.add_argument("--campaign", required=True, help="Campaign dir path OR slug")
    rr.add_argument("--out", default=None, help="Output PDF path (default: <campaign>/report.pdf)")
    rr.add_argument("--title", default=None, help="Title rendered on page 1 header")
    rr.add_argument("--variants-line", default=None, help="Variants line for the header")
    rr.add_argument("--data-source-line", default=None, help="Data-source line for the header")
    rr.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if SoL rooflines are incomplete, there are 0 plot-ready points, or dcgm_grounded=False",
    )
    rr.add_argument(
        "--allow-ungrounded",
        dest="allow_ungrounded",
        action="store_true",
        help="Under --strict, accept dcgm_grounded=False (deliberately zymtrace-only L1 report) instead of failing",
    )
    rr.add_argument("--campaigns-dir", default=None, help="Override the campaigns root")
    rr.add_argument("--json", action="store_true", help="Emit JSON envelope")
    rr.set_defaults(func=cmd_report_render)

    # tpm_summary (v1.35.0): per-hardware TPM capacity rollup for pricing
    ts = sub.add_parser("tpm_summary", description=CONTRACT["tpm_summary"]["description"])
    ts.add_argument("--campaign", required=True, help="Campaign dir path OR slug")
    ts.add_argument(
        "--ttft-sla-ms",
        dest="ttft_sla_ms",
        type=float,
        default=None,
        help="TTFT SLA threshold in ms; SLA point = highest tok/s/GPU under this (and --tpot-sla-ms). Default: config.yaml tpm.ttft_sla_ms",
    )
    ts.add_argument(
        "--tpot-sla-ms",
        dest="tpot_sla_ms",
        type=float,
        default=None,
        help="TPOT/ITL SLA threshold in ms; SLA point = highest tok/s/GPU under this (and --ttft-sla-ms). Default: config.yaml tpm.tpot_sla_ms",
    )
    ts.add_argument(
        "--gpus-per-node",
        dest="gpus_per_node",
        type=int,
        default=None,
        help="GPUs per node for the per-node TPM basis (default: config.yaml tpm.gpus_per_node, else 8)",
    )
    ts.add_argument(
        "--context",
        default=None,
        help="ISL/OSL or data-source context line for the summary header (default: campaign config description)",
    )
    ts.add_argument(
        "--out-dir",
        dest="out_dir",
        default=None,
        help="Output dir for tpm_summary.{json,csv,md} (default: the campaign dir)",
    )
    ts.add_argument("--campaigns-dir", default=None, help="Override the campaigns root")
    ts.add_argument("--json", action="store_true", help="Emit JSON envelope")
    ts.set_defaults(func=cmd_tpm_summary)

    # campaign_run (v1.20.0; Phase 2b)
    cp = sub.add_parser("campaign_run", description=CONTRACT["campaign_run"]["description"])
    cp.add_argument("--config", required=True, help="Path to a campaign matrix YAML")
    cp.add_argument("--campaign", required=True, help="Campaign dir path OR slug")
    cp.add_argument("--continue-on-red", action="store_true", dest="continue_on_red", help="Continue past RED cell verdicts instead of fail-fast")
    cp.add_argument("--dry-run", action="store_true", help="Print the 10-step plan JSON without submitting jobs")
    cp.add_argument(
        "--i-understand-this-submits-jobs",
        dest="i_understand_this_submits_jobs",
        action="store_true",
        help="Ack flag (safety_class=submits_jobs)",
    )
    cp.add_argument("--campaigns-dir", default=None, help="Override the campaigns root")
    cp.add_argument("--json", action="store_true", help="Emit JSON envelope")
    cp.set_defaults(func=cmd_campaign_run)

    # import_perf_bench (v1.18.0)
    ip = sub.add_parser("import_perf_bench", description=CONTRACT["import_perf_bench"]["description"])
    ip.add_argument("--campaign", required=True, help="Campaign dir path OR slug")
    ip.add_argument(
        "--bundle",
        required=True,
        help=(
            "Path to a *-deploy/experiments/artifacts/inference-perf-bench/<bundle>/ dir. "
            "Must contain raw/sweep-c*.txt or raw/sweep-K*-c*.txt files."
        ),
    )
    ip.add_argument("--cell-id", default=None, help="Override the cell_id (default: bundle dirname)")
    ip.add_argument("--model", default=None, help="Model name (default: from bundle's inference_perfbench_v1.json)")
    ip.add_argument("--hardware", default=None, help="e.g. B200, GB300, H100 (default: B200)")
    ip.add_argument("--quant", default=None, help="NVFP4 | FP8 | BF16 | ... (default: inferred from model name)")
    ip.add_argument("--tensor-parallel", type=int, default=None, dest="tensor_parallel", help="TP size (default: 8)")
    ip.add_argument(
        "--parallel-strategy",
        default=None,
        dest="parallel_strategy",
        choices=("TP", "EP"),
        help="TP or EP (default: TP)",
    )
    ip.add_argument(
        "--mtp",
        type=lambda s: s.lower() in ("true", "1", "yes"),
        default=None,
        help="true|false — was MTP enabled? (default: inferred from speculative_decoding metadata)",
    )
    ip.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        dest="max_num_batched_tokens",
        help="max-num-batched-tokens (default: from bundle metadata; falls back to 4096)",
    )
    ip.add_argument("--max-num-seqs", type=int, default=None, dest="max_num_seqs", help="max-num-seqs (extra)")
    ip.add_argument(
        "--patched-vllm-enabled",
        type=lambda s: s.lower() in ("true", "1", "yes"),
        default=None,
        dest="patched_vllm_enabled",
        help="true|false — was the AG patch overlay enabled? (extra)",
    )
    ip.add_argument("--notes", default=None, help="Free-form notes string appended to each row")
    ip.add_argument(
        "--cache-mode",
        dest="cache_mode",
        default=None,
        choices=("warm", "cold", "unknown"),
        help="Warm-vs-cold methodology label for these rows (default: bundle meta or 'unknown')",
    )
    # Full-context descriptor overrides (2026-06-07; AGENTS.md "Every performance number
    # carries its full context"). Required (via flag or bundle meta) for a measured campaign
    # to pass publish/render --strict.
    ip.add_argument("--dataset", dest="dataset", default=None, help="Workload dataset (random|sharegpt|sonnet|aa|code|...) -- full-context descriptor")
    ip.add_argument("--cudagraph-mode", dest="cudagraph_mode", default=None, help="full|piecewise|none|eager -- full-context descriptor (the eager/cudagraph trap)")
    ip.add_argument("--enforce-eager", dest="enforce_eager", type=lambda s: s.lower() in ("true", "1", "yes"), default=None, help="true|false -- sets cudagraph_mode=eager when no explicit --cudagraph-mode")
    ip.add_argument("--gpu-memory-utilization", dest="gpu_memory_utilization", type=float, default=None, help="gmu the number was measured at -- full-context descriptor")
    ip.add_argument("--kv-cache-dtype", dest="kv_cache_dtype", default=None, help="fp8_e4m3|bf16|nvfp4|... -- full-context descriptor")
    ip.add_argument("--image", dest="image", default=None, help="serving image tag / vllm commit -- full-context descriptor")
    ip.add_argument("--delivery", dest="delivery", default=None, choices=("image", "overlay", "patchedVllm", "infr-patch"), help="how code reached the cluster (delivery ladder) -- full-context descriptor")
    ip.add_argument("--overlay-mode", dest="overlay_mode", default=None, choices=("subpath", "patchset-initcontainer", "pythonpath-sitecustomize"), help="overlay sub-tier when --delivery=overlay")
    ip.add_argument("--patch-files", dest="patch_files", default=None, help="comma-separated patch files applied (e.g. infr 0006..0026) -- full-context descriptor")
    ip.add_argument("--data-parallel", dest="data_parallel", type=int, default=None, help="DP replica count (default 1)")
    ip.add_argument("--pipeline-parallel", dest="pipeline_parallel", type=int, default=None, help="PP size (default 1)")
    ip.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help=(
            "Concurrency override (v1.21.0). Required only for single-c "
            "drive_load bundles that have raw/load.jsonl without "
            "bench-c<NNN>/ subdirs. Ignored for sweep-c*.txt bundles."
        ),
    )
    ip.add_argument(
        "--expected-reqs",
        dest="expected_reqs",
        type=lambda s: {int(k): int(v) for k, v in __import__("json").loads(s).items()},
        default=None,
        help=(
            "JSON dict {concurrency: expected_request_count} overriding the AIPerf "
            "full-vs-partial turn-count expectation (default Replay 2025_07: c=1->70, "
            "c=8->284, c=16->559, c=32->1135). Use for AA synthetic-shape runs that "
            "intentionally send a fixed request-count, e.g. '{\"1\": 10}' so a "
            "10-request aa-1k/10k/100k cell imports as full, not partial."
        ),
    )
    ip.add_argument(
        "--require-plot-ready",
        action="store_true",
        dest="require_plot_ready",
        help=(
            "Hard-fail at import if any STATUS_FULL cell lacks 'Median TTFT (ms)' / "
            "'Request throughput (req/s)' (the strict throughput-scatter fields). Set "
            "for any throughput-focus campaign so an incomplete (grep-dropped) capture "
            "is caught at import, not later at the render/publish --strict gate."
        ),
    )
    ip.add_argument("--dry-run", action="store_true", help="Parse + validate; do NOT write files")
    ip.add_argument("--campaigns-dir", default=None, help="Override the campaigns root")
    ip.add_argument("--json", action="store_true", help="Emit JSON envelope")
    ip.set_defaults(func=cmd_import_perf_bench)

    # import_roofline_sweep (v1.61.0): prefill+decode roofline sweep + per-(c,ISL) DCGM
    irs = sub.add_parser("import_roofline_sweep", description=CONTRACT["import_roofline_sweep"]["description"])
    irs.add_argument("--campaign", required=True, help="Campaign dir path OR slug")
    irs.add_argument("--bundle", required=True, help="roofline-sweep.sh output dir (decode_sweep.jsonl + prefill_sweep.jsonl)")
    irs.add_argument("--cell-id", default=None, dest="cell_id", help="Override cell_id base (default: bundle dirname)")
    irs.add_argument("--model", default=None, help="Model name (default: from manifest / zai-org/GLM-5.1)")
    irs.add_argument("--hardware", default=None, help="e.g. B200, GB300, H100 (default: GB300)")
    irs.add_argument("--quant", default=None, help="NVFP4 | FP8 | BF16 (default: NVFP4)")
    irs.add_argument("--kv-dtype", default=None, dest="kv_dtype", help="KV cache dtype for byte-grounded HBM utilization (default: fp8)")
    irs.add_argument("--kv-cache-dtype", default=None, dest="kv_cache_dtype", help="Alias for --kv-dtype; matches atlas full-context naming")
    irs.add_argument("--model-config", default=None, dest="model_config_path", help="Path to the model config.json (for analytical roofline math when the family is not in the registry)")
    irs.add_argument("--tensor-parallel", type=int, default=None, dest="tensor_parallel", help="TP size (default: 4)")
    irs.add_argument("--parallel-strategy", default=None, dest="parallel_strategy", choices=("TP", "EP"), help="TP or EP (default: TP)")
    irs.add_argument("--mtp", type=lambda s: s.lower() in ("true", "1", "yes"), default=None, help="true|false (default: false)")
    irs.add_argument("--max-num-batched-tokens", type=int, default=None, dest="max_num_batched_tokens", help="default 12288")
    irs.add_argument("--cache-mode", dest="cache_mode", default=None, choices=("warm", "cold", "unknown"), help="methodology label")
    # full-context descriptors (2026-06-07) so roofline cells pass publish_to_lake --strict
    irs.add_argument("--dataset", default=None, help="full-context: workload dataset (roofline-sweep.sh drives random) -- default random")
    irs.add_argument("--cudagraph-mode", dest="cudagraph_mode", default=None, help="full-context: full|piecewise|none|eager (the eager/cudagraph trap)")
    irs.add_argument("--enforce-eager", dest="enforce_eager", type=lambda s: s.lower() in ("true", "1", "yes"), default=None, help="full-context: sets cudagraph_mode=eager when no explicit --cudagraph-mode")
    irs.add_argument("--gpu-memory-utilization", dest="gpu_memory_utilization", type=float, default=None, help="full-context: gmu the sweep ran at")
    irs.add_argument("--image", default=None, help="full-context: serving image tag / vllm commit the sweep ran on")
    irs.add_argument("--delivery", dest="delivery", default=None, choices=("image", "overlay", "patchedVllm", "infr-patch"), help="how code reached the cluster (delivery ladder) -- full-context descriptor")
    irs.add_argument("--overlay-mode", dest="overlay_mode", default=None, choices=("subpath", "patchset-initcontainer", "pythonpath-sitecustomize"), help="overlay sub-tier when --delivery=overlay")
    irs.add_argument("--patch-files", dest="patch_files", default=None, help="comma-separated patch files applied -- full-context descriptor")
    irs.add_argument("--data-parallel", dest="data_parallel", type=int, default=None, help="full-context: DP replica count")
    irs.add_argument("--pipeline-parallel", dest="pipeline_parallel", type=int, default=None, help="full-context: PP size")
    irs.add_argument("--dry-run", action="store_true", help="Parse + validate; do NOT write files")
    irs.add_argument("--campaigns-dir", default=None, help="Override the campaigns root")
    irs.add_argument("--json", action="store_true", help="Emit JSON envelope")
    irs.set_defaults(func=cmd_import_roofline_sweep)

    # import_variant_ab (v1.66.0): first-class cross-engine variant A/B import
    iva = sub.add_parser("import_variant_ab", description=CONTRACT["import_variant_ab"]["description"])
    iva.add_argument("--campaign", required=True, help="Campaign dir path OR slug")
    iva.add_argument("--bundle", required=True, help="run-variant-ab.sh output dir (<arm>/c<C>-t<T>.txt per arm)")
    iva.add_argument("--model", required=True, help="Served model id (arm result.json carries no model)")
    iva.add_argument("--hardware", default=None, help="e.g. GB300, B200 (default: B200)")
    iva.add_argument("--quant", default=None, help="NVFP4 | FP8 | BF16 (default: NVFP4)")
    iva.add_argument("--tensor-parallel", type=int, default=None, dest="tensor_parallel", help="TP size (default: arm result.json tp, else 8)")
    iva.add_argument("--parallel-strategy", default=None, dest="parallel_strategy", choices=("TP", "EP"), help="TP or EP (default: TP)")
    iva.add_argument("--mtp", type=lambda s: s.lower() in ("true", "1", "yes"), default=None, help="true|false (default: inferred from arm name)")
    iva.add_argument("--max-num-batched-tokens", type=int, default=None, dest="max_num_batched_tokens", help="default 4096")
    iva.add_argument("--cache-mode", dest="cache_mode", default=None, choices=("warm", "cold", "unknown"), help="methodology label (default: arm result.json warm flag)")
    iva.add_argument("--notes", default=None, help="Free-form notes recorded on each cell")
    iva.add_argument("--dataset", dest="dataset", default=None, help="Workload dataset (random|sharegpt|sonnet|aa|code|...) -- full-context descriptor")
    iva.add_argument("--cudagraph-mode", dest="cudagraph_mode", default=None, help="full|piecewise|none|eager -- full-context descriptor")
    iva.add_argument("--enforce-eager", dest="enforce_eager", type=lambda s: s.lower() in ("true", "1", "yes"), default=None, help="true|false -- sets cudagraph_mode=eager when no explicit --cudagraph-mode")
    iva.add_argument("--gpu-memory-utilization", dest="gpu_memory_utilization", type=float, default=None, help="gmu the number was measured at -- full-context descriptor")
    iva.add_argument("--kv-cache-dtype", dest="kv_cache_dtype", default=None, help="fp8_e4m3|bf16|nvfp4|... -- full-context descriptor")
    iva.add_argument("--image", dest="image", default=None, help="serving image tag / vllm commit -- full-context descriptor")
    iva.add_argument("--delivery", dest="delivery", default=None, choices=("image", "overlay", "patchedVllm", "infr-patch"), help="how code reached the cluster (delivery ladder) -- full-context descriptor")
    iva.add_argument("--overlay-mode", dest="overlay_mode", default=None, choices=("subpath", "patchset-initcontainer", "pythonpath-sitecustomize"), help="overlay sub-tier when --delivery=overlay")
    iva.add_argument("--patch-files", dest="patch_files", default=None, help="comma-separated patch files applied -- full-context descriptor")
    iva.add_argument("--data-parallel", dest="data_parallel", type=int, default=None, help="DP replica count (default 1)")
    iva.add_argument("--pipeline-parallel", dest="pipeline_parallel", type=int, default=None, help="PP size (default 1)")
    iva.add_argument("--require-plot-ready", action="store_true", dest="require_plot_ready",
                     help="Hard-fail at import if any arm lacks Median TTFT / Request throughput (strict-publish fields)")
    iva.add_argument("--dry-run", action="store_true", help="Parse + validate; do NOT write files")
    iva.add_argument("--campaigns-dir", default=None, help="Override the campaigns root")
    iva.add_argument("--json", action="store_true", help="Emit JSON envelope")
    iva.set_defaults(func=cmd_import_variant_ab)

    # dcgm_correlate: frozen DCGM YAML -> cells/<id>/dcgm_correlation.json (page 6/6b)
    dc = sub.add_parser("dcgm_correlate", description=CONTRACT["dcgm_correlate"]["description"])
    dc.add_argument("--campaign", required=True, help="Campaign dir path OR slug")
    dc.add_argument("--cell-id", required=True, dest="cell_id", help="Target cell id under cells/")
    dc.add_argument(
        "--frozen-yaml",
        required=True,
        dest="frozen_yaml",
        help="Path to a dcgm_frozen_v1 YAML (e.g. perf-tune-report/configs/dcgm-frozen/<name>.yaml)",
    )
    dc.add_argument(
        "--kernels-json",
        default=None,
        dest="kernels_json",
        help="Override path to the cell's zymtrace kernels.json (default: cells/<id>/kernels.json if present) -> enables page-6b per-category cross-attribution",
    )
    dc.add_argument(
        "--ceilings",
        default=None,
        help="Override sol-ceilings.yaml path (default: SOL_CEILINGS_YAML env or configs/sol-ceilings.yaml discovered up-tree)",
    )
    dc.add_argument("--dry-run", action="store_true", help="Resolve inputs + print plan; write nothing")
    dc.add_argument("--campaigns-dir", default=None, help="Override the campaigns root")
    dc.add_argument("--json", action="store_true", help="Emit JSON envelope")
    dc.set_defaults(func=cmd_dcgm_correlate)

    # import_nsys: nsys cuda_gpu_kern_sum -> cells/<id>/kernels.json (page-3/6b input)
    ins = sub.add_parser("import_nsys", description=CONTRACT["import_nsys"]["description"])
    ins.add_argument("--campaign", required=True, help="Campaign dir path OR slug")
    ins.add_argument("--cell-id", required=True, help="Target cell id under cells/")
    ins.add_argument(
        "--bundle",
        required=True,
        help="nsys bundle dir (capture_sources.json declaring 'nsys' + nsys/cuda_gpu_kern_sum.txt)",
    )
    ins.add_argument(
        "--kern-sum-name",
        default="cuda_gpu_kern_sum.txt",
        help="filename of the cuda_gpu_kern_sum report under <bundle>/nsys/ (default cuda_gpu_kern_sum.txt)",
    )
    ins.add_argument("--dry-run", action="store_true", help="Parse + validate; write nothing")
    ins.add_argument("--campaigns-dir", default=None, help="Override the campaigns root")
    ins.add_argument("--json", action="store_true", help="Emit JSON envelope")
    ins.set_defaults(func=cmd_import_nsys)

    # import_ncu (v1.31.0): ncu per-kernel bundle -> cells/<id>/ncu_kernels.json
    inc = sub.add_parser("import_ncu", description=CONTRACT["import_ncu"]["description"])
    inc.add_argument("--campaign", required=True, help="Campaign dir path OR slug")
    inc.add_argument("--cell-id", required=True, help="Target cell id under cells/")
    inc.add_argument(
        "--bundle",
        required=True,
        help="ncu-perkernel bundle dir (capture_sources.json + ncu-profiles/)",
    )
    inc.add_argument(
        "--hw-key",
        default="b200_sm100",
        help="sol-ceilings.yaml hardware key (default b200_sm100)",
    )
    inc.add_argument("--dry-run", action="store_true", help="Parse + validate; write nothing")
    inc.add_argument("--campaigns-dir", default=None, help="Override the campaigns root")
    inc.add_argument("--json", action="store_true", help="Emit JSON envelope")
    inc.set_defaults(func=cmd_import_ncu)

    # graph_diff (v1.21.0)
    gd = sub.add_parser(
        "graph_diff",
        description=CONTRACT["graph_diff"]["description"],
    )
    gd.add_argument(
        "--side-a-log",
        required=True,
        dest="side_a_log",
        help="Path to side-A torch.compile log (TORCH_LOGS=+dynamo,+inductor)",
    )
    gd.add_argument(
        "--side-b-log",
        required=True,
        dest="side_b_log",
        help="Path to side-B torch.compile log",
    )
    gd.add_argument(
        "--output-dir",
        required=True,
        dest="output_dir",
        help="Where to write graph_diff.json + per-graph .fx + .diff artifacts",
    )
    gd.add_argument(
        "--side-a-label",
        default=None,
        dest="side_a_label",
        help="Filename prefix for side-A artifacts (default: 'side-A')",
    )
    gd.add_argument(
        "--side-b-label",
        default=None,
        dest="side_b_label",
        help="Filename prefix for side-B artifacts (default: 'side-B')",
    )
    gd.add_argument(
        "--notes",
        default=None,
        help="Free-form notes appended to graph_diff.json",
    )
    gd.add_argument("--dry-run", action="store_true", help="Parse + validate; do NOT write files")
    gd.add_argument("--json", action="store_true", help="Emit JSON envelope")
    gd.set_defaults(func=cmd_graph_diff)

    # kernel_reproducer_scaffold (v1.69.0)
    krs = sub.add_parser(
        "kernel_reproducer_scaffold",
        description=CONTRACT["kernel_reproducer_scaffold"]["description"],
    )
    krs.add_argument("--kernel-name", required=True, dest="kernel_name",
                     help="Kernel task-impl template name, e.g. linear_sm100_mpk_task_impl")
    krs.add_argument("--header", required=True,
                     help="Kernel header to #include, e.g. tasks/blackwell/linear_sm100_mpk.cuh")
    krs.add_argument("--output-dir", required=True, dest="output_dir",
                     help="Directory to write the .cu + build script")
    krs.add_argument("--mma-m", type=int, default=128, dest="mma_m")
    krs.add_argument("--mma-n", type=int, default=16, dest="mma_n")
    krs.add_argument("--batch", type=int, default=8, help="BATCH_SIZE")
    krs.add_argument("--out-dim", type=int, default=1024, dest="out_dim", help="OUTPUT_SIZE")
    krs.add_argument("--k", type=int, default=6144, help="REDUCTION_SIZE")
    krs.add_argument("--mirage-tree", default="/work/mirage-perop2", dest="mirage_tree")
    krs.add_argument("--arch", default="compute_103a,code=sm_103a", help="nvcc -gencode arch")
    krs.add_argument("--dry-run", action="store_true", help="Render but do NOT write files")
    krs.add_argument("--json", action="store_true", help="Emit JSON envelope")
    krs.set_defaults(func=cmd_kernel_reproducer_scaffold)

    # kernel_profile (v1.21.0)
    kp = sub.add_parser(
        "kernel_profile",
        description=CONTRACT["kernel_profile"]["description"],
    )
    kp.add_argument("--namespace", required=True, help="Kubernetes namespace of the target pod")
    kp.add_argument("--pod", required=True, help="Target pod name")
    kp.add_argument(
        "--target-container",
        required=True,
        dest="target_container",
        help="Name of the container whose PID namespace we attach to (e.g. 'basic-inference')",
    )
    kp.add_argument(
        "--output-dir",
        required=True,
        dest="output_dir",
        help="Local directory to write .nsys-rep + summary CSV + kernel_profile.json",
    )
    kp.add_argument(
        "--sidecar-image",
        dest="sidecar_image",
        default="ghcr.io/cfregly/nsys-sidecar:0.1.0",
        help="nsys sidecar image (public canonical image; override with your own mirror)",
    )
    kp.add_argument(
        "--duration-seconds",
        type=int,
        dest="duration_seconds",
        default=120,
        help="nsys profile duration (default 120s)",
    )
    kp.add_argument("--sample", default="cpu", help="nsys --sample value (default: cpu)")
    kp.add_argument(
        "--trace",
        default="cuda,nvtx,osrt",
        help="nsys --trace value (default: cuda,nvtx,osrt)",
    )
    kp.add_argument(
        "--sampling-frequency",
        type=int,
        dest="sampling_frequency",
        default=1000,
        help="nsys --sampling-frequency value in Hz (default 1000)",
    )
    kp.add_argument(
        "--vllm-pid-pattern",
        dest="vllm_pid_pattern",
        default="vllm serve",
        help="pgrep pattern for the engine process (default: 'vllm serve')",
    )
    kp.add_argument(
        "--bundle",
        default=None,
        help=(
            "Optional inference-perf-bench bundle dir to patch with the new "
            "kernel_profile metadata. Updates the bundle's "
            "inference_perfbench_v1.json in place."
        ),
    )
    kp.add_argument(
        "--i-understand-this-submits-jobs",
        dest="i_understand_this_submits_jobs",
        action="store_true",
        help="Ack flag (safety_class=submits_jobs)",
    )
    kp.add_argument("--dry-run", action="store_true", help="Print step commands; do NOT execute kubectl")
    kp.add_argument("--json", action="store_true", help="Emit JSON envelope")
    kp.set_defaults(func=cmd_kernel_profile)

    # raw_bench_compare (v1.24.0)
    rbc = sub.add_parser(
        "raw_bench_compare",
        description=CONTRACT["raw_bench_compare"]["description"],
    )
    rbc.add_argument("--manifest", required=True, help="raw_bench_compare_v1 YAML manifest")
    rbc.add_argument("--out", required=True, help="Output PDF path")
    rbc.add_argument("--json", action="store_true", help="Emit JSON envelope")
    rbc.set_defaults(func=cmd_raw_bench_compare)

    # report_smoke
    rs = sub.add_parser("report_smoke", description=CONTRACT["report_smoke"]["description"])
    rs.add_argument("--out", default=None, help="Output PDF path (default: /tmp/perftunereport-smoke.pdf)")
    rs.add_argument("--title", default=None, help="Title rendered on page 1 header")
    rs.add_argument("--json", action="store_true", help="Emit JSON envelope")
    rs.set_defaults(func=cmd_report_smoke)

    # publish_to_lake
    pl = sub.add_parser("publish_to_lake", description=CONTRACT["publish_to_lake"]["description"])
    pl.add_argument("--campaign", required=True, help="Campaign dir path OR slug")
    pl.add_argument(
        "--s3-endpoint",
        dest="s3_endpoint",
        default=None,
        help="S3 endpoint URL (default: env PERFLAKE_LAKE_S3_ENDPOINT or https://object-store.example.com)",
    )
    pl.add_argument(
        "--s3-bucket",
        dest="s3_bucket",
        default=None,
        help="S3 bucket name (default: env PERFLAKE_LAKE_S3_BUCKET or perf-lake)",
    )
    pl.add_argument(
        "--s3-access-key-file",
        dest="s3_access_key_file",
        default=None,
        help="File holding the S3 access key (default: env PERFLAKE_LAKE_S3_ACCESS_KEY)",
    )
    pl.add_argument(
        "--s3-secret-key-file",
        dest="s3_secret_key_file",
        default=None,
        help="File holding the S3 secret key (default: env PERFLAKE_LAKE_S3_SECRET_KEY)",
    )
    pl.add_argument(
        "--if-exists",
        dest="if_exists",
        choices=("fail", "skip", "overwrite"),
        default=None,
        help="Behavior when an S3 object already exists (default: fail).",
    )
    pl.add_argument(
        "--allow-incomplete",
        dest="allow_incomplete",
        action="store_true",
        help=(
            "DEPRECATED alias for --no-strict (publish an incomplete-SoL campaign). "
            "Kept for back-compat; prefer --no-strict for a first-class "
            "intentional-gap publish (latency-bound / ncu-only / proxy)."
        ),
    )
    # STRICT BY DEFAULT (workspace rigor policy, docs/METHODOLOGY.md):
    # publish REFUSES a campaign with no rendered SoL page / 0 throughput-scatter
    # points / unsupported verdict, so a no-SoL campaign cannot land silently.
    # --strict is kept as an explicit no-op for back-compat; --no-strict is the
    # first-class opt-out for a deliberate intentional-gap publish.
    pl.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        help="No-op (strict is the default since the strict-by-default flip). Kept for back-compat.",
    )
    pl.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help=(
            "Opt out of the strict gate for a FIRST-CLASS intentional-gap publish "
            "(latency-bound / proxy / dcgm_grounded=false). The gap is recorded on "
            "the lake row (sol_complete/focus/sol_rigor); never use silently to hide "
            "a missing capture."
        ),
    )
    pl.set_defaults(strict=True)
    pl.add_argument(
        "--allow-ungrounded",
        dest="allow_ungrounded",
        action="store_true",
        help=(
            "Escape the MANDATORY DCGM byte-grounding gate (v1.33.0): by default "
            "publish FAILS-CLOSED when a campaign has the L1 SoL roofline but "
            "dcgm_grounded=False (no DCGM workload-level byte/FLOP grounding, "
            "pages 6/6b). Pass this only for a deliberately zymtrace-only L1 "
            "campaign where DCGM is unavailable; the lake row records "
            "dcgm_grounded=false."
        ),
    )
    pl.add_argument("--dry-run", dest="dry_run", action="store_true", help="Build + write parquet locally; skip S3 upload.")
    pl.add_argument("--campaigns-dir", default=None, help="Override the campaigns root")
    pl.add_argument("--json", action="store_true", help="Emit JSON envelope")
    pl.set_defaults(func=cmd_publish_to_lake)

    vv = sub.add_parser("value_view", description=CONTRACT["value_view"]["description"])
    vv.add_argument("--registry", default=None,
                    help="Path to value-findings.yaml (default: <campaigns>/../configs/value-findings.yaml)")
    vv.add_argument("--out", default=None, help="Write the rendered markdown ledger to this path")
    vv.add_argument("--format", choices=["table", "report"], default="table",
                    help="table = wide audit table (default); report = compact copy-paste summary for a report/Slack")
    vv.add_argument("--title", default=None, help="Ledger title")
    vv.add_argument("--gpu-hr", type=float, default=None,
                    help="$/GPU-hour for the GRIND FRONTIER $/1M-token economics "
                         "(override; else perf-tune-report/configs/cost.yaml, else 8.60 GB300)")
    vv.add_argument("--campaigns-dir", default=None, help="Override the campaigns root")
    vv.add_argument("--json", action="store_true", help="Emit JSON envelope")
    vv.set_defaults(func=cmd_value_view)

    pv = sub.add_parser("portability_view",
                        description=CONTRACT["portability_view"]["description"])
    pv.add_argument("--registry", default=None,
                    help="Path to value-findings.yaml (default: <campaigns>/../configs/value-findings.yaml)")
    pv.add_argument("--out", default=None, help="Write the rendered matrix markdown to this path")
    pv.add_argument("--title", default=None, help="Matrix title")
    pv.add_argument("--campaigns-dir", default=None, help="Override the campaigns root")
    pv.add_argument("--json", action="store_true", help="Emit JSON envelope")
    pv.set_defaults(func=cmd_portability_view)

    # champion_select (v1.66.0): baseline vs top-X cross-engine champion selection
    cs = sub.add_parser("champion_select", description=CONTRACT["champion_select"]["description"])
    cs.add_argument("--campaign", required=True, help="Campaign dir path OR slug")
    cs.add_argument("--focus", default=None, choices=("throughput", "latency", "mixed"),
                    help="Champion metric focus (default: campaign config.focus, else throughput)")
    cs.add_argument("--focus-c", type=int, default=None, dest="focus_c",
                    help="Concurrency at which the champion is selected (default: 1 latency / 32 throughput)")
    cs.add_argument("--top", type=int, default=3, help="Number of top variants to keep (default 3)")
    cs.add_argument("--baseline", default=None, help="Baseline cell_id (default: a *-base/*-baseline arm, vLLM-preferred)")
    cs.add_argument("--metric", default=None, choices=("tok_s_gpu", "tpot"),
                    help="Override the ranking metric (default: derived from --focus)")
    cs.add_argument("--slo-rel", type=float, default=1.10, dest="slo_rel",
                    help="TPOT SLO = baseline TPOT x this (throughput focus; default 1.10)")
    cs.add_argument("--slo-abs-ms", type=float, default=None, dest="slo_abs_ms",
                    help="Absolute TPOT SLO ceiling in ms (overrides --slo-rel)")
    cs.add_argument("--trials", type=int, default=None,
                    help="Trials/arm of the A/B (VERDICT variance gate needs >=3 + --same-node)")
    cs.add_argument("--same-node", action="store_true", dest="same_node",
                    help="Assert the A/B arms ran same-node (VERDICT variance gate)")
    cs.add_argument("--require-workloads", default=None, dest="require_workloads",
                    help="Comma list the VERDICT multi-workload gate requires (default: aa,sonnet,sharegpt,random,code)")
    cs.add_argument("--workloads-present", default=None, dest="workloads_present",
                    help="Comma list of workloads actually benched (multi-workload gate; omit => unknown)")
    cs.add_argument("--accuracy-gate", default="unknown", dest="accuracy_gate",
                    choices=("pass", "fail", "unknown"), help="Accuracy-eval gate result (VERDICT gate)")
    cs.add_argument("--accuracy-floor", type=float, default=None, dest="accuracy_floor",
                    help="Derive the accuracy gate from the campaign's local eval_acc cells "
                    "(import_model_eval): pass iff the worst measured metric >= this floor")
    cs.add_argument("--out", default=None, help="CHAMPION.md output path (default: <campaign>/CHAMPION.md)")
    cs.add_argument("--title", default=None, help="Report title")
    cs.add_argument("--dry-run", action="store_true", help="Compute + print; do NOT write files")
    cs.add_argument("--campaigns-dir", default=None, help="Override the campaigns root")
    cs.add_argument("--json", action="store_true", help="Emit JSON envelope")
    cs.set_defaults(func=cmd_champion_select)

    cp = sub.add_parser("capture_plan", description=CONTRACT["capture_plan"]["description"])
    cp.add_argument("--campaign", required=True, help="Target campaign dir path OR slug")
    cp.add_argument(
        "--source-campaign",
        action="append",
        default=[],
        dest="source_campaign",
        help=(
            "Source campaign dir path OR slug to search for exact-match capture artifacts. "
            "Repeatable; defaults to the target campaign."
        ),
    )
    cp.add_argument("--out", default=None, help="Write the plan JSON to this path")
    cp.add_argument("--campaigns-dir", default=None, help="Override the campaigns root")
    cp.add_argument("--json", action="store_true", help="Emit JSON envelope")
    cp.set_defaults(func=cmd_capture_plan)

    mcr = sub.add_parser(
        "materialize_capture_reuse",
        description=CONTRACT["materialize_capture_reuse"]["description"],
    )
    mcr.add_argument("--plan", required=True, help="capture_plan JSON path")
    mcr.add_argument("--dry-run", action="store_true", help="Validate/calculate copies without writing")
    mcr.add_argument("--json", action="store_true", help="Emit JSON envelope")
    mcr.set_defaults(func=cmd_materialize_capture_reuse)

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


# Optional deps that only land with `install.sh --full` (the perf_tune_report +
# leaderboard extras). A bare `claude plugin install` install does NOT pull
# these, so report_render / report_smoke / publish_to_lake fail without them.
_RENDERER_OPTIONAL_DEPS = frozenset(
    {"matplotlib", "pandas", "numpy", "pyarrow", "boto3", "tiktoken", "openpyxl"}
)


def _renderer_dep_hint(missing: str) -> str:
    server_root = Path(__file__).resolve().parents[2]
    return (
        f"\n[perftunereport] Optional dependency '{missing}' is not installed.\n"
        f"The perf_tune_report renderer / publish-to-lake verbs need the perf_tune_report "
        f"extras, which a minimal install skips. Install them with EITHER:\n"
        f"    bash {server_root}/install.sh --full\n"
        f'    pip install -e "{server_root}[perf_tune_report,leaderboard]" '
        f"-c {server_root}/constraints-aa.txt\n"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except ModuleNotFoundError as exc:
        missing = (exc.name or "").split(".")[0]
        if missing in _RENDERER_OPTIONAL_DEPS:
            print(_renderer_dep_hint(missing), file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
