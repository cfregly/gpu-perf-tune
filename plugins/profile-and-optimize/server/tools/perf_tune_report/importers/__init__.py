"""Importers: bridge from existing perf-bench evidence bundles to perf-report
``cells/<id>/normalized.json`` files.

The canonical end-to-end flow expected by the perf-report library is:

    campaign_init -> cell_run -> atlas_aggregate -> report_render -> publish_to_lake

But operators frequently have **existing** ``*-deploy/experiments/artifacts/
inference-perf-bench/<bundle>/`` directories from ad-hoc / non-MCP runs (e.g.
the GLM-5.1 Phase 1-14 campaigns + the V1/V2/V3 AG-patch verification bundles
captured before ``perf_tune_report_campaign_run`` was implemented). The importers
in this package adapt those bundles into the same ``cells/<id>/normalized.json``
shape the aggregator expects, so the rest of the pipeline (aggregate -> render
-> publish) works without any per-campaign bash glue.

Importer surface
----------------

- ``import_perf_bench_bundle`` (v1.18.0): vLLM ``bench serve`` text output
  (one ``raw/sweep-c<N>.txt`` per concurrency, optionally K-suffixed).
  GLM-5.1 + DSv4 bundle layout.
- ``import_drive_load_bundle`` (v1.21.0): ``drive_load.py`` JSONL load-driver
  output (``bench-c<NNN>/raw/load.jsonl`` or ``raw/load.jsonl``). Kimi K2.6
  bundle layout.
- ``import_lws_summary_bundle`` (v1.23.1): pre-aggregated multi-variant
  ``summary.json`` (GLM-5.1 LWS-baseline-vs-champions layout). Emits
  AtlasCell rows directly to ``<campaign_dir>/atlas.jsonl`` (bypasses
  the cells/<id>/normalized.json -> atlas_aggregate path because the
  upstream variant-runner already aggregated).
- ``import_bundle_auto`` (v1.21.0; v1.23.1: + lws_summary arm): inspects
  the bundle and dispatches to the correct importer. The CLI
  ``perf_tune_report_import_perf_bench`` verb calls this so a single command
  handles all three layouts without the operator having to remember
  which is which.

Both back the ``mcp__profile_and_optimize__perf_tune_report_import_perf_bench`` verb.
"""

from pathlib import Path
from typing import Any

from tools.perf_tune_report.importers.inference_perf_bench import (
    ImportResult,
    import_perf_bench_bundle,
)
from tools.perf_tune_report.importers.inference_drive_load import (
    DriveLoadImportResult,
    detect_bundle_pattern,
    import_drive_load_bundle,
)
from tools.perf_tune_report.importers.lws_summary import (
    LwsSummaryImportResult,
    detect_lws_summary,
    import_lws_summary_bundle,
)
from tools.perf_tune_report.importers.aiperf_export import (
    AiperfImportResult,
    detect_aiperf_bundle,
    import_aiperf_bundle,
)
from tools.perf_tune_report.importers.variant_ab import (
    VariantAbImportResult,
    detect_variant_ab,
    import_variant_ab_bundle,
)
from tools.perf_tune_report.importers.roofline_sweep import (
    RooflineSweepImportResult,
    import_roofline_sweep_bundle,
)


def import_bundle_auto(
    bundle: Path,
    campaign_dir: Path,
    *,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
    captured_at: str | None = None,
    concurrency_override: int | None = None,
    require_plot_ready: bool = False,
) -> ImportResult | DriveLoadImportResult | LwsSummaryImportResult:
    """Detect the bundle pattern and dispatch to the right importer.

    Detection priority: lws_summary -> inference_perf_bench -> inference_drive_load.
    The lws_summary arm is checked first because it requires a
    ``summary.json`` at the bundle root -- a unique signature that the
    other two layouts never produce.

    Raises:
        ValueError: bundle does not exist, or no pattern matches.
    """
    bundle = bundle.expanduser().resolve()
    if not bundle.is_dir():
        raise ValueError(f"import_bundle_auto: bundle does not exist: {bundle}")

    if detect_lws_summary(bundle):
        return import_lws_summary_bundle(
            bundle=bundle,
            campaign_dir=campaign_dir,
            overrides=overrides,
            dry_run=dry_run,
            captured_at=captured_at,
        )

    # variant-A/B layout: <bundle>/<arm>/c<C>-t<T>.txt (run-variant-ab.sh). Unique
    # subdir/c<C>-t<T>.txt signature; emits one cell per arm (trial-averaged).
    if detect_variant_ab(bundle):
        return import_variant_ab_bundle(
            bundle=bundle,
            campaign_dir=campaign_dir,
            overrides=overrides,
            dry_run=dry_run,
            captured_at=captured_at,
            require_plot_ready=require_plot_ready,
        )

    # AIPerf per-variant layout: <bundle>/c<N>/profile_export_aiperf.csv. Checked
    # before the sweep/drive_load arms because its c<N>/ subdir signature is
    # unique. Requires overrides['model'] (no model inference for AIPerf).
    if detect_aiperf_bundle(bundle):
        return import_aiperf_bundle(
            bundle=bundle,
            campaign_dir=campaign_dir,
            overrides=overrides,
            dry_run=dry_run,
            captured_at=captured_at,
        )

    pattern = detect_bundle_pattern(bundle)
    if pattern == "inference_perf_bench":
        return import_perf_bench_bundle(
            bundle=bundle,
            campaign_dir=campaign_dir,
            overrides=overrides,
            dry_run=dry_run,
            captured_at=captured_at,
            require_plot_ready=require_plot_ready,
        )
    if pattern == "inference_drive_load":
        return import_drive_load_bundle(
            bundle=bundle,
            campaign_dir=campaign_dir,
            overrides=overrides,
            dry_run=dry_run,
            captured_at=captured_at,
            concurrency_override=concurrency_override,
        )
    raise ValueError(
        f"import_bundle_auto: no recognized importer pattern in {bundle} "
        f"(looked for summary.json, raw/sweep-c*.txt, raw/sweep-K*-c*.txt, "
        f"bench-c<NNN>/raw/load.jsonl, and raw/load.jsonl)"
    )


__all__ = [
    "ImportResult",
    "DriveLoadImportResult",
    "LwsSummaryImportResult",
    "AiperfImportResult",
    "VariantAbImportResult",
    "import_perf_bench_bundle",
    "import_drive_load_bundle",
    "import_lws_summary_bundle",
    "import_aiperf_bundle",
    "import_variant_ab_bundle",
    "import_roofline_sweep_bundle",
    "RooflineSweepImportResult",
    "import_bundle_auto",
    "detect_bundle_pattern",
    "detect_lws_summary",
    "detect_aiperf_bundle",
    "detect_variant_ab",
]
