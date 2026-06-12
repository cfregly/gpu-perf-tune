"""Publish a perf-report campaign as Parquet to S3 (the perf lake BYOB lane).

This is the data-lake side of the inference-perf-tune-report skill. Where the
existing 5 verbs (``campaign_init``/``cell_run``/``atlas_aggregate``/
``report_render``/``report_smoke``) own the local-evidence pipeline, the
``publish_to_lake`` verb (added in profile-and-optimize v1.16.0) takes a green
campaign and writes two parquet files to the perf lake's S3 BYOB bucket
``s3://perf-lake/perflake/perf-report/``. A downstream (operator-side) Spark
job reads the raw parquet and registers it in the warehouse -- e.g. Iceberg
intake tables promoted via dbt staging/model layers and served to consumers
through an external catalog (Trino / StarRocks / Spark SQL) -- the standard
raw -> staging -> model medallion pattern. This module owns only the raw
parquet contract; the warehouse side is whatever pipeline your data platform
already runs.

Authentication: S3 endpoint / bucket / access-key / secret-key are
resolved in priority order

1. CLI flags ``--s3-endpoint``, ``--s3-bucket``,
   ``--s3-access-key-file``, ``--s3-secret-key-file``.
2. Env vars ``PERFLAKE_LAKE_S3_ENDPOINT`` (default
   ``https://object-store.example.com``), ``PERFLAKE_LAKE_S3_BUCKET`` (default
   ``perf-lake``), ``PERFLAKE_LAKE_S3_ACCESS_KEY``,
   ``PERFLAKE_LAKE_S3_SECRET_KEY``. Matches the existing perflake
   ``.env`` convention so one env file works for both tools.

S3 layout (Hive-style so a downstream Spark job can register it later):

- ``s3://<bucket>/perflake/perf-report/atlas_v1/dt=<YYYY-MM-DD>/campaign=<campaign_id>/part-0.parquet``
- ``s3://<bucket>/perflake/perf-report/campaign_v1/dt=<YYYY-MM-DD>/campaign=<campaign_id>/part-0.parquet``
- ``s3://<bucket>/perflake/perf-report/sol_v1/dt=<YYYY-MM-DD>/campaign=<campaign_id>/part-0.parquet``
- ``s3://<bucket>/perflake/perf-report/tpm_v1/dt=<YYYY-MM-DD>/campaign=<campaign_id>/part-0.parquet``
- ``s3://<bucket>/perflake/perf-report/cost_v1/dt=<YYYY-MM-DD>/campaign=<campaign_id>/part-0.parquet``

Idempotency: writes are keyed by ``campaign_id`` so re-runs overwrite the
same object. Default ``--if-exists=fail`` to avoid silent clobber;
``--if-exists=overwrite`` for explicit re-publish; ``--if-exists=skip``
for backfill loops.

Completeness gate (v1.26.0): ``publish`` refuses to land a campaign that
was never rendered, has no Speed-of-Light rooflines, or has 0 plot-ready
points, unless ``--allow-incomplete`` is passed. The ``campaign_v1`` table
gained three columns -- ``sol_complete`` (bool), ``plot_ready_points``
(int64), ``omitted_pages`` (string, comma-joined) -- sourced from the
renderer's ``report_status.json``, plus ``dcgm_grounded`` (bool, v1.29.0)
recording whether DCGM workload-level byte/FLOP grounding (page 6) is
present, plus ``partial_pages`` (string, comma-joined, v1.30.0) recording
pages that rendered but carry less than a full measurement (e.g. a page-5
%SoL-only scatter with arithmetic intensity unmeasured) so consumers can
filter partial roofline data. Downstream intake tables should append these
columns (all NOT NULL; older rows predate them).
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import re
import socket
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from tools.perf_tune_report.capture_signature import variant_key_for
from tools.perf_tune_report.schema import (
    REQUIRED_CONTEXT_STR_FIELDS,
    AtlasCell,
    engine_for_backend,
    read_jsonl,
)


S3_ENV_ENDPOINT = "PERFLAKE_LAKE_S3_ENDPOINT"
S3_ENV_BUCKET = "PERFLAKE_LAKE_S3_BUCKET"
S3_ENV_ACCESS_KEY = "PERFLAKE_LAKE_S3_ACCESS_KEY"
S3_ENV_SECRET_KEY = "PERFLAKE_LAKE_S3_SECRET_KEY"
PUBLISHER_ENV = "PERFREPORT_PUBLISHER"

S3_DEFAULT_ENDPOINT = "https://object-store.example.com"
S3_DEFAULT_BUCKET = "perf-lake"
S3_PREFIX = "perflake/perf-report"

# atlas_v1: one row per benchmark OPERATING POINT. Grain (natural key /
# uniqueness): (campaign_id, cell_id, concurrency, mean_input_tokens,
# mean_output_tokens, cache_mode, num_speculative_tokens). A single cell_id is
# swept over BOTH concurrency (decode sweep) AND prompt shape (prefill ISL
# sweep -> same cell_id + c=1, different mean_input_tokens), and the same point
# may be re-measured warm vs cold (cache_mode) and at different MTP/EAGLE K
# (num_speculative_tokens). So (campaign_id, cell_id, concurrency) ALONE is NOT
# unique -- it collapses prefill-ISL points, warm/cold pairs, and MTP-K arms.
# This mirrors roofline_v1's (campaign_id, cell_id, phase, concurrency, isl)
# grain; atlas has no `phase` column, so cell_id (the -prefill/-decode suffix) +
# ISL/OSL encode it. `variant_key` (capture_signature hash, set at publish) is
# the belt-and-suspenders cross-campaign variant guard.
ATLAS_TABLE_NAME = "atlas_v1"
CAMPAIGN_TABLE_NAME = "campaign_v1"
# sol_v1 (added v1.36.0): per-category L1-L4 Speed-of-Light rows, one row per
# (campaign_id, cell_id, category, sol_level[, kernel_name]). L1 = zymtrace
# sample-share + ceiling (no %SoL), L2 = zymtrace x DCGM cross-attribution,
# L3 = DCGM workload resources, L4 = ncu per-kernel. Empty when a campaign has
# no cells/*/ SoL artifacts (still published, 0 rows).
SOL_TABLE_NAME = "sol_v1"
# tpm_v1 (added v1.35.0): per-hardware tokens-per-minute capacity rows for
# pricing / capacity discussions, one row per (campaign_id, model, hardware,
# quant, tensor_parallel, parallel_strategy, mtp, operating_point[peak|sla],
# basis[per_gpu|per_replica|per_node]). Empty when a campaign has no
# throughput-bearing atlas rows (still published, 0 rows).
TPM_TABLE_NAME = "tpm_v1"
# cost_v1 (added v1.42.0): per-(group, operating_point) economics/TCO rows --
# $/1M tokens (when a cost: config block supplies $/GPU-hour) + tokens-per-watt
# (when DCGM power was captured). One row per (campaign_id, model, hardware,
# quant, tensor_parallel, parallel_strategy, mtp, operating_point[peak|sla]).
# Empty when no throughput-bearing atlas rows; cost / energy columns null when
# the respective inputs are absent (recorded, not blocking).
COST_TABLE_NAME = "cost_v1"
# quality_v1 (added v1.63.0): training-accuracy / draft-acceptance rows in LONG
# format -- one row per (campaign_id, cell_id, metric_kind, metric_name) for every
# cell that declares extra["metric_kind"] (e.g. train_accuracy_proxy, acceptance).
# metric_name/value come from the canonical extra["quality_metrics"]={name: value}
# sub-dict, falling back to flat accuracy/loss/delta keys for campaigns predating
# the convention. Empty (0 rows) for pure serving campaigns (no metric_kind).
QUALITY_TABLE_NAME = "quality_v1"
# champion_v1 (added v1.66.0): the production-choice synthesis -- one row per
# SELECTED variant (baseline + top-X) from a campaign's champion_select.json,
# carrying the focus metric, %win vs baseline, SLO verdict, the 4-layer SoL
# summary (sol_rigor + HBM/tensor/SM %), and is_recommended / champion_tier so
# "which X variants did we pick + which one ships + its proof" is one query.
# Empty (0 rows) when a campaign has no champion_select.json (still published).
CHAMPION_TABLE_NAME = "champion_v1"
# roofline_v1 (added v1.67.0): per-(c, ISL) prefill/decode roofline operating
# points -- one row per cells/*/roofline_sweep.json point. Carries the analytical
# arithmetic intensity + achieved-compute/GPU + delivered-HBM-BW (the roofline
# x/y), the DCGM SM/tensor/DRAM active fractions (the utilization-vs-concurrency
# curves), and the per-GPU ceilings, so Superset can render the prefill/decode
# roofline scatter + the HBM%/tensor%/SM%-vs-C lines from the lake directly
# (page 7 was previously renderer-local + atlas extra_json only). Empty (0 rows)
# when a campaign has no roofline sweep (still published).
ROOFLINE_TABLE_NAME = "roofline_v1"

IF_EXISTS_FAIL = "fail"
IF_EXISTS_SKIP = "skip"
IF_EXISTS_OVERWRITE = "overwrite"
IF_EXISTS_CHOICES = (IF_EXISTS_FAIL, IF_EXISTS_SKIP, IF_EXISTS_OVERWRITE)


class CampaignIncompleteError(RuntimeError):
    """Raised by ``publish`` when a campaign is missing SoL rooflines or has
    0 plot-ready points and ``allow_incomplete`` was not set.

    The perf-lake is the traceability source of truth; landing an
    incomplete campaign silently is the failure this gate prevents. The
    message names what is missing + how to populate it, mirroring the
    renderer's report_status.json.
    """

# Matches the campaign-dir UTC timestamp token anywhere in the name (no
# anchor): handles both the canonical campaign_init layout
# "<UTC>-<slug>" (e.g. "20260525T081650Z-glm51-phase6", timestamp prefix)
# and the legacy "<slug>-<UTC>" layout (timestamp suffix). Both capture
# "2026-05-25T08:16:50Z". `.search()` returns the first match.
_CAMPAIGN_UTC_RE = re.compile(
    r"(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})T"
    r"(?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})Z"
)


@dataclass(frozen=True)
class S3Config:
    """Resolved S3 connection params.

    ``access_key`` and ``secret_key`` are read from files (matches the
    existing ``--s3-access-key-file`` / ``--s3-secret-key-file`` pattern
    used by ``perflake/incremental-export-to-lake``). Strings here, never
    paths, so the actual file content never lands in tool argv or logs.
    """

    endpoint: str
    bucket: str
    access_key: str
    secret_key: str


@dataclass(frozen=True)
class ObjectWriteResult:
    """Per-object write outcome returned by ``publish``."""

    table: str
    local_path: Path
    s3_key: str
    size_bytes: int
    sha256: str
    row_count: int
    skipped: bool = False


@dataclass(frozen=True)
class PublishResult:
    """Outcome of one ``publish_to_lake`` invocation."""

    campaign_dir: Path
    campaign_id: str
    captured_at_utc: datetime
    atlas: ObjectWriteResult
    campaign: ObjectWriteResult
    sol: ObjectWriteResult
    tpm: ObjectWriteResult
    cost: ObjectWriteResult
    quality: ObjectWriteResult
    dry_run: bool
    bucket: str
    endpoint: str
    published_at_utc: datetime
    # champion_v1 (added v1.66.0): the production-choice synthesis table. None on
    # older callers that predate the field; publish() always populates it.
    champion: ObjectWriteResult | None = None
    # roofline_v1 (added v1.67.0): per-(c, ISL) prefill/decode roofline points.
    # None on older callers that predate the field; publish() always populates it.
    roofline: ObjectWriteResult | None = None


# ---------------------------------------------------------------------------
# Provenance + schema construction
# ---------------------------------------------------------------------------


def parse_campaign_utc(campaign_id: str) -> datetime:
    """Extract the UTC timestamp baked into a campaign dir name.

    Returns the parsed timestamp on success, or raises ``ValueError`` if
    the dir name does not end with the canonical ``YYYYMMDDTHHMMSSZ``
    suffix the workspace's evidence-bundle-init convention enforces.
    """
    match = _CAMPAIGN_UTC_RE.search(campaign_id)
    if not match:
        raise ValueError(
            f"FATAL: campaign_id {campaign_id!r} does not end with "
            f"YYYYMMDDTHHMMSSZ; cannot derive captured_at_utc."
        )
    return datetime(
        int(match["y"]),
        int(match["m"]),
        int(match["d"]),
        int(match["H"]),
        int(match["M"]),
        int(match["S"]),
        tzinfo=timezone.utc,
    )


def parse_source_md(source_md_path: Path) -> dict[str, str]:
    """Parse SOURCE.md for ``operator:`` and ``cluster:`` style metadata.

    The campaign_init verb writes SOURCE.md with ``- captured_at: ...``
    and ``- config: ...`` lines, but operators sometimes hand-edit it to
    add ``- operator:`` or ``- cluster_context:``. We honor any of those
    if present and silently fill empty strings otherwise.
    """
    metadata: dict[str, str] = {}
    if not source_md_path.is_file():
        return metadata
    for line in source_md_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("-"):
            continue
        body = stripped.lstrip("-").strip()
        if ":" not in body:
            continue
        key, _, value = body.partition(":")
        metadata[key.strip().lower()] = value.strip()
    return metadata


@dataclass(frozen=True)
class RenderStatusSummary:
    """The completeness signal read from ``report_status.json``.

    ``sol_complete`` / ``plot_ready_points`` / ``omitted_pages`` mirror the
    renderer's RenderStatus. ``rendered`` is False when no report_status.json
    exists yet (render never ran) -- treated as incomplete so the publish
    gate fails loudly rather than landing an unrendered campaign.
    """

    rendered: bool
    sol_complete: bool
    plot_ready_points: int
    omitted_pages: str
    # L2/L3 DCGM byte-grounding present (page 6 rendered). Defaults False
    # when the field is absent (older report_status.json) so a campaign is
    # never silently recorded as DCGM-grounded without proof.
    dcgm_grounded: bool = False
    # Comma-joined names of pages that rendered but carry less than a full
    # measurement (e.g. page-5 %SoL-only / AI-unmeasured). Empty string when
    # none. Recorded (not gated) so the lake row flags the limitation.
    partial_pages: str = ""
    # focus (added v1.33.0): "latency" | "throughput" | "mixed". Recorded so
    # latency-focused runs are first-class published results.
    focus: str = "mixed"
    # sol_rigor (added v1.33.0): highest SoL evidence level present --
    # "L4" (ncu) | "L3" (DCGM) | "L1" (zymtrace proxy) | "none". The
    # proxy-vs-tight distinction is a recorded field, not a publish blocker.
    sol_rigor: str = "none"
    # PER-ARM SoL coverage (added v1.68.0): mirrors the renderer's RenderStatus
    # per-arm fields. ``sol_complete`` above is CAMPAIGN-level ("any SoL page
    # rendered"); these say whether baseline AND every variant carries a
    # roofline. The defaults make a pre-field report_status.json (rendered by an
    # older renderer) trivially complete, so the per-arm gate bites only once a
    # campaign is re-rendered with the per-arm-aware renderer -- the
    # sol-coverage audit --fail backstop catches already-rendered partials.
    arms_total: int = 0
    arms_uncovered: tuple[str, ...] = ()
    sol_per_arm_complete: bool = True


def read_render_status(campaign_dir: Path) -> RenderStatusSummary:
    """Read ``<campaign_dir>/report_status.json`` (written by render_report).

    Returns a ``RenderStatusSummary`` with ``rendered=False`` and
    ``sol_complete=False`` when the file is absent, so publish cannot land a
    campaign that was never rendered.
    """
    status_path = campaign_dir / "report_status.json"
    if not status_path.is_file():
        return RenderStatusSummary(
            rendered=False, sol_complete=False, plot_ready_points=0, omitted_pages="",
            dcgm_grounded=False,
        )
    data = json.loads(status_path.read_text(encoding="utf-8"))
    omitted = ",".join(o.get("page", "") for o in data.get("omitted_pages", []))
    partial = ",".join(p.get("page", "") for p in data.get("partial_pages", []))
    return RenderStatusSummary(
        rendered=True,
        sol_complete=bool(data.get("sol_complete", False)),
        plot_ready_points=int(data.get("plot_ready_points", 0)),
        omitted_pages=omitted,
        dcgm_grounded=bool(data.get("dcgm_grounded", False)),
        partial_pages=partial,
        focus=str(data.get("focus", "mixed")),
        sol_rigor=str(data.get("sol_rigor", "none")),
        arms_total=int(data.get("arms_total", 0) or 0),
        arms_uncovered=tuple(data.get("arms_uncovered", []) or []),
        # Back-compat: absent -> True (older report_status.json predates the
        # per-arm fields, so it is not retroactively refused; re-render arms it).
        sol_per_arm_complete=bool(data.get("sol_per_arm_complete", True)),
    )


@dataclass
class VerdictSummary:
    """Author-declared verdict tier + provenance, read from ``<campaign>/verdict.json``.

    Defaults to a **DRAFT** (ungated) when verdict.json is absent. A campaign that
    declares ``tier == "verdict"`` is held to the controlled+metric+baseline
    provenance by the publish gate (AGENTS.md "Verdict rigor: DRAFT vs VERDICT").
    """

    tier: str = "draft"
    trials: int = 0
    same_node: bool = False
    decode_metric: str = ""          # tpot | itl | throughput | ""
    baseline_named: bool = False
    latency_claim: bool = False      # campaign makes a decode-latency claim
    per_kernel_ref: bool = False
    which_kernel_claim: bool = False  # campaign makes a which-kernel attribution


def read_verdict(campaign_dir: Path) -> VerdictSummary:
    """Read ``<campaign_dir>/verdict.json`` (author/skill-declared). Absent -> DRAFT."""
    path = campaign_dir / "verdict.json"
    if not path.is_file():
        return VerdictSummary()
    d = json.loads(path.read_text(encoding="utf-8"))
    return VerdictSummary(
        tier=str(d.get("tier", "draft")),
        trials=int(d.get("trials", 0)),
        same_node=bool(d.get("same_node", False)),
        decode_metric=str(d.get("decode_metric", "")),
        baseline_named=bool(d.get("baseline_named", False)),
        latency_claim=bool(d.get("latency_claim", False)),
        per_kernel_ref=bool(d.get("per_kernel_ref", False)),
        which_kernel_claim=bool(d.get("which_kernel_claim", False)),
    )


def verdict_problems(v: VerdictSummary) -> list[str]:
    """Unmet VERDICT-tier provenance requirements (empty list when draft or OK)."""
    if v.tier != "verdict":
        return []
    p: list[str] = []
    if v.trials < 3:
        p.append(
            f"verdict_tier=verdict needs >=3 trials (got {v.trials}) -- run a "
            "repeated-trial controlled A/B"
        )
    if not v.same_node:
        p.append(
            "verdict_tier=verdict needs same_node=true (pin both arms to one node) "
            "-- a cross-node delta is a DRAFT"
        )
    if v.latency_claim and v.decode_metric not in ("tpot", "itl"):
        p.append(
            f"decode-latency verdict needs decode_metric in tpot|itl (got "
            f"{v.decode_metric!r}) -- output tok/s at small num_prompts is TTFT-noisy"
        )
    if not v.baseline_named:
        p.append(
            "verdict_tier=verdict needs baseline_named=true (name the "
            "production-representative baseline)"
        )
    if v.which_kernel_claim and not v.per_kernel_ref:
        p.append(
            "which-kernel verdict needs per_kernel_ref=true (nsys/ncu per-kernel "
            "data) -- a DCGM regime % is necessary but not sufficient"
        )
    return p


def _effective_verdict_tier(campaign_dir: Path) -> str:
    """Verdict tier to record in the lake. An author-declared
    ``verdict_tier=verdict`` that fails the rigor checks is downgraded to
    ``draft`` (so the campaign still publishes, with an honest tier) rather
    than blocking publish (always-publish policy v1.33.0)."""
    v = read_verdict(campaign_dir)
    if v.tier == "verdict" and verdict_problems(v):
        return "draft"
    return v.tier


def read_next_lever(campaign_dir: Path) -> str:
    """Author-declared ``next_lever`` for the campaign -- the "Always ship an
    actionable path-forward" / performance-ratchet mandate (AGENTS.md). The
    specific next change expected to move a metric further, OR an explicit
    ``frontier-exhausted: <evidence>`` when a dimension is grounded at its
    Speed-of-Light ceiling. Read from the campaign ``config.yaml`` ``next_lever:``
    field, falling back to a ``SOURCE.md`` ``next_lever`` bullet. Empty when neither
    is set. (Campaign-level analog of the per-finding ``next_lever`` the
    ``value_view`` GRIND FRONTIER already enforces.)"""
    config_path = campaign_dir / "config.yaml"
    if config_path.is_file():
        try:
            import yaml  # lazy: keep the publish path import-light

            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            nl = str(cfg.get("next_lever", "") or "").strip()
            if nl:
                return nl
        except Exception:  # noqa: BLE001 - a malformed config must never crash publish
            pass
    return str(parse_source_md(campaign_dir / "SOURCE.md").get("next_lever", "") or "").strip()


def next_lever_problems(campaign_dir: Path) -> list[str]:
    """Performance-ratchet publish gate (AGENTS.md "Always ship an actionable
    path-forward" / "Always be grinding"). A campaign MUST declare a ``next_lever``
    so the lake row answers "what is the next lever for X" and the grind never
    silently stalls. The allowed escape for a genuinely-maxed dimension is
    ``frontier-exhausted: <evidence>`` (still a non-empty value). Empty/absent ->
    a problem: REFUSED under ``--strict`` (the default), recorded+warned otherwise --
    the same machinery as the SoL / verdict / krhpa gates."""
    if read_next_lever(campaign_dir):
        return []
    return [
        "no next_lever / path-forward declared (performance ratchet: set `next_lever:` in "
        "the campaign config.yaml -- the specific next change to move a metric further, or "
        "'frontier-exhausted: <evidence>' when a dimension is at its Speed-of-Light ceiling)"
    ]


VALID_CLOSE_REASONS = ("beat-target", "measured-plateau", "infra-wall")


def read_close_reason(campaign_dir: Path) -> str:
    """Author-declared ``close_reason`` for the campaign (AGENTS.md "Always be grinding",
    principle i). A perf investigation may CLOSE only on a MEASURED outcome: ``beat-target``,
    ``measured-plateau`` (variance-controlled), or ``infra-wall`` (documented blocker). Empty =
    still open (no close claimed). Read from config.yaml ``close_reason:`` then SOURCE.md."""
    config_path = campaign_dir / "config.yaml"
    if config_path.is_file():
        try:
            import yaml  # lazy: keep the publish path import-light

            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            cr = str(cfg.get("close_reason", "") or "").strip()
            if cr:
                return cr
        except Exception:  # noqa: BLE001 - a malformed config must never crash publish
            pass
    return str(parse_source_md(campaign_dir / "SOURCE.md").get("close_reason", "") or "").strip()


def close_reason_problems(campaign_dir: Path) -> list[str]:
    """Grind-discipline publish gate (AGENTS.md "Always be grinding", principle i). A campaign that
    CLOSES an investigation must record a MEASURED close_reason in {beat-target, measured-plateau,
    infra-wall}; closing on a first-principles / cost / temporary-plateau argument is forbidden. An
    empty close_reason = open campaign (fine -- next_lever is still required by next_lever_problems). A
    non-empty but invalid value -> a problem: REFUSED under --strict, recorded+warned otherwise."""
    cr = read_close_reason(campaign_dir)
    if not cr or cr in VALID_CLOSE_REASONS:
        return []
    return [
        f"close_reason='{cr}' is not a measured close-reason (grind discipline: must be one of "
        f"{', '.join(VALID_CLOSE_REASONS)} -- never close on a first-principles / cost / "
        "temporary-plateau argument; set `close_reason:` only when CLOSING on measured proof)"
    ]


def _row_is_measured(row: AtlasCell) -> bool:
    """True iff the atlas row carries a throughput/latency number, so the
    warm/cold + shape methodology rule applies. A failed/evicted cell with
    all-None metrics is not a measurement and is exempt."""
    return any(
        getattr(row, name, None) is not None
        for name in (
            "ttft_avg_ms",
            "request_throughput_avg",
            "output_tps_per_user",
            "output_tps_per_gpu",
            "tpot_median_ms",
            "itl_avg_ms",
        )
    )


def methodology_problems(rows: list[AtlasCell]) -> list[str]:
    """Benchmark-methodology hygiene gate (AGENTS.md "Benchmark methodology
    hygiene"). Every MEASURED atlas row must carry a warm-vs-cold label
    (``cache_mode`` in {warm, cold}) and shape provenance
    (``max_num_batched_tokens`` > 0). A throughput/latency number left at
    ``cache_mode=unknown`` is the warm-vs-cold comparability trap (a warm
    sweep-tail point silently compared against a cold single-shot). Mirrors the
    verdict gate: under ``--strict`` these RAISE; otherwise they are recorded
    (the warm/cold gap is already visible on ``atlas_v1.cache_mode``) + warned.
    Returns the problem strings (empty when no measured rows, or all labelled)."""
    measured = [r for r in rows if _row_is_measured(r)]
    if not measured:
        return []
    p: list[str] = []
    unlabeled = sorted(
        {
            r.cell_id
            for r in measured
            if (getattr(r, "cache_mode", "unknown") or "unknown") == "unknown"
        }
    )
    if unlabeled:
        p.append(
            "warm/cold label missing (cache_mode=unknown) on measured cell(s): "
            + ", ".join(unlabeled)
            + " -- label every throughput/latency number warm (cache-primed / "
            "sweep-tail) or cold (fresh / single-shot) via the importer "
            "--cache-mode (AGENTS.md 'Benchmark methodology hygiene')"
        )
    no_shape = sorted(
        {
            r.cell_id
            for r in measured
            if int(getattr(r, "max_num_batched_tokens", 0) or 0) <= 0
        }
    )
    if no_shape:
        p.append(
            "shape provenance missing (max_num_batched_tokens<=0) on measured "
            "cell(s): " + ", ".join(no_shape)
            + " -- the bench shape must be recorded, not inferred from a label"
        )
    # ISL/OSL shape precision -- the PER-ROW half of "per-number exact shape (no smoothing)"
    # (added 2026-06-08; docs/METHODOLOGY.md "no bare numbers" / AGENTS.md "Per-number exact
    # shape"). Each measured cell carries its OWN ISL/OSL; a number is NEVER smoothed to one
    # campaign-level shape. (The CROSS-CELL half -- one shared shape label rendered over
    # heterogeneous-shape cells -- is handled at RENDER time by the render-layer
    # renderer/tpm_table.py shape_label_problems() detector, which makes _shape_caption
    # label per-cell; it is not a publish --strict gate, since heterogeneous shapes across
    # cells are allowed -- only smoothing them to one caption is the defect.)
    # For vllm-bench datasets (random / sonnet / sharegpt / replay / code) the per-request
    # ISL/OSL IS the shape and must be recorded -- max_num_batched_tokens alone does not
    # pin the workload. aiperf / drive_load / aa-* workloads legitimately leave ISL/OSL
    # None (the shape is defined by the dataset name / token-mean -- see the
    # AtlasCell.mean_input_tokens docstring), so they are EXEMPT (no false positive).
    def _isl_osl_exempt(r: AtlasCell) -> bool:
        be = str(getattr(r, "bench_backend", "") or "").lower()
        ds = str(getattr(r, "dataset", "") or "").lower()
        return be in ("aiperf", "drive_load") or ds.startswith("aa")

    no_isl_osl = sorted(
        {
            r.cell_id
            for r in measured
            if not _isl_osl_exempt(r)
            and (
                float(getattr(r, "mean_input_tokens", 0) or 0) <= 0
                or float(getattr(r, "mean_output_tokens", 0) or 0) <= 0
            )
        }
    )
    if no_isl_osl:
        p.append(
            "shape provenance missing (mean_input_tokens/mean_output_tokens<=0 on a "
            "vllm-bench dataset) on measured cell(s): " + ", ".join(no_isl_osl)
            + " -- record the workload ISL/OSL the number was measured at (AGENTS.md "
            "'Every performance number carries its full context'); aiperf/drive_load/aa-* "
            "are exempt (shape defined by the dataset name)"
        )
    # Full-context descriptor gate (AGENTS.md "Every performance number carries its full
    # context (no bare numbers)" / rule docs/METHODOLOGY.md): a measured number is a
    # defect without its descriptor. Flag any measured row whose str descriptor field is
    # still "unknown", or whose gpu_memory_utilization is unset.
    for fld in REQUIRED_CONTEXT_STR_FIELDS:
        missing = sorted(
            {
                r.cell_id
                for r in measured
                if (str(getattr(r, fld, "unknown") or "unknown")) == "unknown"
            }
        )
        if missing:
            p.append(
                f"full-context descriptor missing ({fld}=unknown) on measured cell(s): "
                + ", ".join(missing)
                + f" -- set {fld} via the importer/runner override or bundle metadata "
                "(AGENTS.md 'Every performance number carries its full context')"
            )
    no_gmu = sorted(
        {
            r.cell_id
            for r in measured
            if getattr(r, "gpu_memory_utilization", None) is None
        }
    )
    if no_gmu:
        p.append(
            "full-context descriptor missing (gpu_memory_utilization unset) on measured "
            "cell(s): " + ", ".join(no_gmu)
            + " -- record the gpu-memory-utilization the number was measured at"
        )
    return p


def _read_krhpa(campaign_dir: Path) -> dict[str, Any] | None:
    """Read the ``krhpa:`` block from the frozen campaign ``config.yaml`` (None
    when absent / unreadable). Mirrors ``discover_tpm_config``'s
    read-config-at-publish pattern (lazy yaml import keeps the publish path
    light)."""
    cfg_path = campaign_dir / "config.yaml"
    if not cfg_path.is_file():
        return None
    try:
        import yaml  # lazy: keep the module import light for the publish path

        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    blk = data.get("krhpa")
    return blk if isinstance(blk, dict) else None


def _read_krhpa_exempt_reason(campaign_dir: Path) -> str:
    """Read the explicit L4 non-custom-kernel exemption reason from config.yaml."""
    cfg_path = campaign_dir / "config.yaml"
    if not cfg_path.is_file():
        return ""
    try:
        import yaml  # lazy: keep the module import light for the publish path

        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return ""
    reason = data.get("krhpa_exempt_reason")
    return str(reason).strip() if reason else ""


def krhpa_problems(campaign_dir: Path, render_status: RenderStatusSummary) -> list[str]:
    """Kernel-rubric gate (AGENTS.md "Custom-kernel work: classify before you
    climb"). A kernel-comparison campaign -- identified by an L4 ncu roofline
    having rendered (``sol_rigor == "L4"``, i.e. page 5 from a
    ``cells/*/ncu_kernels.json``) -- MUST carry a ``krhpa:`` block in
    ``config.yaml`` classifying BOTH the candidate AND the named baseline on
    all five axes (K, R, H, P, A in 1..4 == L1..L4), so a win over a
    strictly-lower-H/R baseline cannot be silently published as a VERDICT (you
    cannot beat an H4 tensor-core baseline with an H1 kernel). Non-L4 campaigns
    are exempt (return [])."""
    if render_status.sol_rigor != "L4":
        return []
    if _read_krhpa_exempt_reason(campaign_dir):
        return []
    blk = _read_krhpa(campaign_dir)
    if blk is None:
        return [
            "L4 kernel-comparison campaign (sol_rigor=L4) is missing a krhpa: "
            "block in config.yaml -- classify the candidate AND the named "
            "baseline on (K,R,H,P,A) per AGENTS.md 'Custom-kernel work: classify "
            "before you climb' (a win over a lower-H/R baseline is a DRAFT, not a "
            "VERDICT)"
        ]
    p: list[str] = []
    for role in ("candidate", "baseline"):
        sub = blk.get(role)
        if not isinstance(sub, dict):
            p.append(f"krhpa.{role} missing or not a mapping (need K,R,H,P,A + name)")
            continue
        for axis in ("K", "R", "H", "P", "A"):
            v = sub.get(axis)
            if not (isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 4):
                p.append(f"krhpa.{role}.{axis} must be an int in 1..4 (L1-L4), got {v!r}")
        name = sub.get("name")
        if not (isinstance(name, str) and name.strip()):
            p.append(f"krhpa.{role}.name must be a non-empty string")
    return p


def _read_provenance(campaign_dir: Path) -> dict[str, Any] | None:
    """Read the ``provenance:`` mapping from the frozen campaign ``config.yaml``
    (campaign_init copies the bundle's ```provenance``` block there), falling
    back to a ``provenance.json`` sidecar. None when absent. Mirrors
    ``_read_krhpa``'s lazy-yaml read-at-publish pattern."""
    cfg_path = campaign_dir / "config.yaml"
    if cfg_path.is_file():
        try:
            import yaml  # lazy: keep the publish path import-light

            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            blk = data.get("provenance")
            if isinstance(blk, dict):
                return blk
        except Exception:
            pass
    js = campaign_dir / "provenance.json"
    if js.is_file():
        try:
            import json

            blk = json.loads(js.read_text(encoding="utf-8"))
            if isinstance(blk, dict):
                return blk
        except Exception:
            pass
    return None


def source_problems(campaign_dir: Path) -> list[str]:
    """Source-code attribution gate (durable-lineage workstream). A VERDICT-tier
    campaign must be attributable to a clean, pinned source commit. Reads the
    AUTHOR-DECLARED verdict tier (not the auto-downgraded one) so a campaign that
    is otherwise rigorous but lacks pinned source is caught. Empty list for
    drafts / OK. Mirrors ``verdict_problems`` / ``krhpa_problems``: under
    ``--strict`` these RAISE, otherwise they are warned + recorded."""
    from tools.perf_tune_report import provenance as _prov

    declared_tier = read_verdict(campaign_dir).tier
    return _prov.source_provenance_problems(_read_provenance(campaign_dir), declared_tier)


def roofline_problems(render_status: RenderStatusSummary) -> list[str]:
    """Page-7 (prefill/decode roofline) gate (added v1.67.0). The phase-separated
    roofline is MANDATORY for a serving THROUGHPUT/MIXED campaign (one with
    plot-ready throughput points) -- the "what C maxes the TFLOPs / is decode
    >=75% HBM / which sharding degree" questions are always wanted. Latency-only,
    kernel-probe, and accuracy runs are EXEMPT (focus=latency/accuracy, or no
    throughput points). Under ``--strict`` this RAISES; otherwise it is recorded
    on the campaign_v1 row + warned (a first-class intentional-gap publish)."""
    if render_status.focus not in ("throughput", "mixed"):
        return []
    if render_status.plot_ready_points == 0:
        return []
    if "page 7" in (render_status.omitted_pages or "").lower():
        return [
            "no prefill/decode roofline (page 7) on a throughput/mixed serving "
            "campaign -- run *-deploy/profiling/roofline-sweep.sh + "
            "perftunereport import_roofline_sweep, then re-render (ROOFLINE-METHODOLOGY.md)"
        ]
    return []


def read_roofline_gap_arms(campaign_dir: Path) -> dict[str, str]:
    """Author-declared per-arm roofline opt-out from config.yaml
    ``roofline_gap_arms:`` -- the per-variant analog of ``--ack-telemetry-gap``.
    An arm that is genuinely un-capturable (B200 cluster evacuated, MTP engine
    instability, deleted overlay ConfigMap) is listed here so it publishes but
    is RECORDED, never silently "complete". Accepts a mapping
    ``{<arm>: <reason>}`` or a bare list ``[<arm>, ...]`` (reason defaults to
    ""). Empty dict when absent / unreadable (mirrors ``_read_krhpa``'s lazy
    read-at-publish pattern; a malformed config must never crash publish)."""
    cfg_path = campaign_dir / "config.yaml"
    if not cfg_path.is_file():
        return {}
    try:
        import yaml  # lazy: keep the publish path import-light

        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - a malformed config must never crash publish
        return {}
    blk = data.get("roofline_gap_arms")
    if isinstance(blk, dict):
        return {str(k): str(v) for k, v in blk.items()}
    if isinstance(blk, (list, tuple)):
        return {str(a): "" for a in blk}
    return {}


def per_arm_roofline_problems(
    render_status: RenderStatusSummary, campaign_dir: Path
) -> list[str]:
    """Per-arm roofline gate (added v1.68.0). ``roofline_problems`` above refuses
    only when page 7 is ENTIRELY absent (campaign-level "no arm has a roofline").
    This refuses when SOME arm lacks a roofline -- the "baseline + EACH variant
    must carry a roofline" rule that the campaign-level ``sol_complete`` /
    page-7-omitted checks silently miss (a 1-of-N-covered campaign reads
    complete).

    Exempt unless the campaign is a MULTI-ARM serving throughput/mixed run
    (``focus in {throughput, mixed}``, plot-ready points, ``arms_total > 1``):
    latency / accuracy / kernel-probe / single-arm runs are out of scope, and a
    pre-field report_status.json (``sol_per_arm_complete`` defaulting True) is a
    no-op until re-rendered. The author opt-out ``roofline_gap_arms:``
    (config.yaml) drops genuinely un-capturable arms so they publish + are
    recorded. Under ``--strict`` a remaining gap RAISES; otherwise it is warned +
    recorded (the same machinery as the SoL / verdict / roofline gates)."""
    if render_status.focus not in ("throughput", "mixed"):
        return []
    if render_status.plot_ready_points == 0:
        return []
    if render_status.arms_total <= 1:
        return []
    if render_status.sol_per_arm_complete:
        return []
    gap = read_roofline_gap_arms(campaign_dir)
    missing = sorted(a for a in render_status.arms_uncovered if a not in gap)
    if not missing:
        return []
    return [
        f"per-arm roofline gap: {len(missing)} of {render_status.arms_total} arm(s) "
        "carry no roofline (baseline + EACH variant must): "
        + ", ".join(missing)
        + " -- run *-deploy/profiling/roofline-sweep.sh + perftunereport "
        "import_roofline_sweep for each arm, then re-render; or declare a "
        "genuinely un-capturable arm in config.yaml roofline_gap_arms: "
        "{<arm>: <reason>} (B200 / MTP-engine-blocked / overlay-gone)"
    ]


def _sha256_of(path: Path) -> str:
    if not path.is_file():
        return ""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# Matches a spec-decode K marker in a note / extra text: "mtp-K3", "mtp-num_spec2",
# "num_speculative_tokens=2", "num_speculative_tokens: 2". `.search` takes the first.
_MTP_K_NOTE_RE = re.compile(
    r"(?:mtp[-_]?(?:k|num[-_]?spec(?:ulative[-_]?tokens)?)|num_speculative_tokens)"
    r"\s*[=:\-]?\s*(\d+)",
    re.IGNORECASE,
)


def _effective_num_speculative_tokens(row: AtlasCell) -> int | None:
    """Return the row's spec-decode K, lifting it from ``extra``/``notes`` when
    the typed ``num_speculative_tokens`` field is unset.

    The typed field is the source of truth; this only fills it for rows (mostly
    historical / model-optimize A/B) that recorded the K in ``extra``
    (``num_speculative_tokens`` / ``spec_decode_k``) or in a ``mtp-K<n>`` /
    ``num_speculative_tokens=<n>`` note rather than the typed field. Without the
    lift, two MTP arms (K=2 vs K=3) at the same operating point collapse on the
    atlas grain (they differ only by K). Returns None when there is no K signal.
    This is the single publish-time chokepoint that normalizes K across ALL
    atlas-producing paths (import_perf_bench / variant_ab / drive_load /
    vllm_sweep / hand-built normalized.json) and on re-publish of historical
    campaigns.
    """
    typed = getattr(row, "num_speculative_tokens", None)
    if typed is not None and not isinstance(typed, bool):
        return int(typed)
    extra = getattr(row, "extra", None) or {}
    for key in ("num_speculative_tokens", "spec_decode_k"):
        val = extra.get(key)
        if isinstance(val, bool):
            continue
        if isinstance(val, int):
            return val
        if isinstance(val, str) and val.strip().isdigit():
            return int(val.strip())
    for text in (extra.get("note"), getattr(row, "notes", None)):
        if isinstance(text, str):
            match = _MTP_K_NOTE_RE.search(text)
            if match:
                return int(match.group(1))
    return None


def build_atlas_table(rows: list[AtlasCell], campaign_id: str) -> Any:
    """Build a ``pyarrow.Table`` for the perf_tune_report_atlas_v1 schema.

    Grain (natural key / uniqueness): one row per ``(campaign_id, cell_id,
    concurrency, mean_input_tokens, mean_output_tokens, cache_mode,
    num_speculative_tokens)``. A cell_id is swept over concurrency (decode) AND
    prompt shape (prefill ISL), and re-measured warm/cold and at different
    MTP-K, so ``(campaign_id, cell_id, concurrency)`` alone is NOT unique.
    ``variant_key`` is the publish-time cross-campaign guard.

    Schema (stable across campaigns; ``extra_json`` is the JSON-serialized
    free-form ``extra`` dict so the Iceberg schema does not drift between
    campaigns that use different ``extra`` keys):

    - campaign_id, cell_id, model, hardware, quant (string)
    - tensor_parallel, max_num_batched_tokens, concurrency (int64)
    - parallel_strategy (string), mtp (bool), status (string)
    - ttft_avg_ms, request_throughput_avg, output_tps_per_user,
      output_tps_per_gpu (float64, nullable)
    - backend, raw_path, captured_at, notes (string)
    - extra_json (string)
    """
    import pyarrow as pa  # type: ignore[import-not-found]

    schema = pa.schema(
        [
            pa.field("campaign_id", pa.string(), nullable=False),
            pa.field("cell_id", pa.string(), nullable=False),
            pa.field("model", pa.string(), nullable=False),
            pa.field("hardware", pa.string(), nullable=False),
            pa.field("quant", pa.string(), nullable=False),
            pa.field("tensor_parallel", pa.int64(), nullable=False),
            pa.field("parallel_strategy", pa.string(), nullable=False),
            pa.field("mtp", pa.bool_(), nullable=False),
            pa.field("max_num_batched_tokens", pa.int64(), nullable=False),
            pa.field("concurrency", pa.int64(), nullable=False),
            pa.field("status", pa.string(), nullable=False),
            pa.field("ttft_avg_ms", pa.float64(), nullable=True),
            pa.field("request_throughput_avg", pa.float64(), nullable=True),
            pa.field("output_tps_per_user", pa.float64(), nullable=True),
            pa.field("output_tps_per_gpu", pa.float64(), nullable=True),
            # total_tps_per_gpu (added v1.35.0): total (input+output) token
            # throughput per GPU -- the input to total-TPM in tpm_v1. Null for
            # backends that emit no total-token line. Downstream: append (default
            # null older partitions).
            pa.field("total_tps_per_gpu", pa.float64(), nullable=True),
            # tpot_median_ms + itl_avg_ms (added v1.33.0): decode-latency
            # metrics so the atlas carries a consistent schema every run
            # regardless of focus. Downstream: append (default null older partitions).
            pa.field("tpot_median_ms", pa.float64(), nullable=True),
            pa.field("itl_avg_ms", pa.float64(), nullable=True),
            # Analysis-carry-through (added v1.42.0). Downstream: append these
            # columns (defaults null / 'unknown' for older partitions).
            pa.field("mean_input_tokens", pa.float64(), nullable=True),
            pa.field("mean_output_tokens", pa.float64(), nullable=True),
            pa.field("prefix_cache_hit_rate", pa.float64(), nullable=True),
            pa.field("cache_mode", pa.string(), nullable=False),
            # Full-context descriptor (added 2026-06-07; AGENTS.md "Every performance
            # number carries its full context"). Downstream: append these columns (defaults
            # 'unknown' / null / 1 for older partitions).
            pa.field("dataset", pa.string(), nullable=False),
            pa.field("cudagraph_mode", pa.string(), nullable=False),
            pa.field("gpu_memory_utilization", pa.float64(), nullable=True),
            pa.field("kv_cache_dtype", pa.string(), nullable=False),
            pa.field("image", pa.string(), nullable=False),
            pa.field("data_parallel", pa.int64(), nullable=False),
            pa.field("pipeline_parallel", pa.int64(), nullable=False),
            # Serving-variant descriptor (added 2026-06-07): the knobs needed to answer
            # "which variant + why" + a stable cross-campaign variant_key (capture_signature
            # hash). Downstream: append (defaults null/'' for older partitions).
            # num_speculative_tokens = MTP/EAGLE K; bench_backend = bench CLIENT (vllm/openai).
            pa.field("num_speculative_tokens", pa.int64(), nullable=True),
            pa.field("async_scheduling", pa.bool_(), nullable=True),
            pa.field("max_num_seqs", pa.int64(), nullable=True),
            pa.field("enable_prefix_caching", pa.bool_(), nullable=True),
            pa.field("bench_backend", pa.string(), nullable=False),
            pa.field("variant_key", pa.string(), nullable=False),
            pa.field("backend", pa.string(), nullable=False),
            # Serving engine normalized from backend (vllm/sglang/trtllm; "" for
            # aiperf). Downstream: append (default '' for older partitions).
            pa.field("serving_engine", pa.string(), nullable=False),
            # Ledger-to-atlas data-capture gaps (added 2026-06-07; see perf-report
            # UPSTREAM-REQUEST-atlas-ledger-datacapture-gaps.md). Downstream: append these
            # columns (defaults '' / null for older partitions).
            pa.field("router_policy", pa.string(), nullable=False),
            pa.field("prefix_reuse", pa.float64(), nullable=True),
            pa.field("per_replica_cache_hit", pa.float64(), nullable=True),
            pa.field("acceptance_length", pa.float64(), nullable=True),
            # Spec-decode accept rate = accepted/draft_tokens over the same
            # window as acceptance_length (added 2026-06-10, first-class AL
            # capture). Downstream: append (default null for older partitions).
            pa.field("spec_accept_rate", pa.float64(), nullable=True),
            pa.field("kv_cache_tokens", pa.int64(), nullable=True),
            pa.field("ep_mode", pa.string(), nullable=False),
            pa.field("dcgm_sm_active", pa.float64(), nullable=True),
            pa.field("dcgm_dram_active", pa.float64(), nullable=True),
            pa.field("dcgm_tensor_active", pa.float64(), nullable=True),
            pa.field("raw_path", pa.string(), nullable=False),
            pa.field("captured_at", pa.string(), nullable=False),
            pa.field("notes", pa.string(), nullable=False),
            pa.field("extra_json", pa.string(), nullable=False),
        ]
    )

    columns: dict[str, list[Any]] = {field.name: [] for field in schema}
    for row in rows:
        columns["campaign_id"].append(campaign_id)
        columns["cell_id"].append(row.cell_id)
        columns["model"].append(row.model)
        columns["hardware"].append(row.hardware)
        columns["quant"].append(row.quant)
        columns["tensor_parallel"].append(int(row.tensor_parallel))
        columns["parallel_strategy"].append(row.parallel_strategy)
        columns["mtp"].append(bool(row.mtp))
        columns["max_num_batched_tokens"].append(
            int(row.max_num_batched_tokens) if row.max_num_batched_tokens is not None else 0
        )
        columns["concurrency"].append(int(row.concurrency))
        columns["status"].append(row.status)
        columns["ttft_avg_ms"].append(row.ttft_avg_ms)
        columns["request_throughput_avg"].append(row.request_throughput_avg)
        columns["output_tps_per_user"].append(row.output_tps_per_user)
        columns["output_tps_per_gpu"].append(row.output_tps_per_gpu)
        columns["total_tps_per_gpu"].append(getattr(row, "total_tps_per_gpu", None))
        columns["tpot_median_ms"].append(getattr(row, "tpot_median_ms", None))
        columns["itl_avg_ms"].append(getattr(row, "itl_avg_ms", None))
        columns["mean_input_tokens"].append(getattr(row, "mean_input_tokens", None))
        columns["mean_output_tokens"].append(getattr(row, "mean_output_tokens", None))
        columns["prefix_cache_hit_rate"].append(getattr(row, "prefix_cache_hit_rate", None))
        columns["cache_mode"].append(getattr(row, "cache_mode", "unknown") or "unknown")
        columns["dataset"].append(getattr(row, "dataset", "unknown") or "unknown")
        columns["cudagraph_mode"].append(getattr(row, "cudagraph_mode", "unknown") or "unknown")
        columns["gpu_memory_utilization"].append(getattr(row, "gpu_memory_utilization", None))
        columns["kv_cache_dtype"].append(getattr(row, "kv_cache_dtype", "unknown") or "unknown")
        columns["image"].append(getattr(row, "image", "unknown") or "unknown")
        columns["data_parallel"].append(int(getattr(row, "data_parallel", 1) or 1))
        columns["pipeline_parallel"].append(int(getattr(row, "pipeline_parallel", 1) or 1))
        _nst = _effective_num_speculative_tokens(row)
        columns["num_speculative_tokens"].append(int(_nst) if _nst is not None else None)
        columns["async_scheduling"].append(getattr(row, "async_scheduling", None))
        _mns = getattr(row, "max_num_seqs", None)
        columns["max_num_seqs"].append(int(_mns) if _mns is not None else None)
        columns["enable_prefix_caching"].append(getattr(row, "enable_prefix_caching", None))
        columns["bench_backend"].append(getattr(row, "bench_backend", "") or "")
        # Stable cross-campaign variant key (capture_signature hash) so a (model, variant)
        # time-series is joinable in the lake. Compute when the row did not carry one;
        # never fail the publish if the signature can't be built (e.g. accuracy rows).
        _vk = getattr(row, "variant_key", "") or ""
        if not _vk:
            try:
                _vk = variant_key_for(row)  # image-INDEPENDENT: stable across engine versions for the time-series
            except Exception:  # noqa: BLE001 - variant_key is best-effort, never blocks publish
                _vk = ""
        columns["variant_key"].append(_vk)
        columns["backend"].append(row.backend or "")
        columns["serving_engine"].append(
            getattr(row, "serving_engine", "") or engine_for_backend(row.backend or "")
        )
        # Ledger-to-atlas data-capture gaps (added 2026-06-07).
        columns["router_policy"].append(getattr(row, "router_policy", "") or "")
        columns["prefix_reuse"].append(getattr(row, "prefix_reuse", None))
        columns["per_replica_cache_hit"].append(getattr(row, "per_replica_cache_hit", None))
        columns["acceptance_length"].append(getattr(row, "acceptance_length", None))
        columns["spec_accept_rate"].append(getattr(row, "spec_accept_rate", None))
        columns["kv_cache_tokens"].append(
            int(row.kv_cache_tokens) if getattr(row, "kv_cache_tokens", None) is not None else None
        )
        columns["ep_mode"].append(getattr(row, "ep_mode", "") or "")
        columns["dcgm_sm_active"].append(getattr(row, "dcgm_sm_active", None))
        columns["dcgm_dram_active"].append(getattr(row, "dcgm_dram_active", None))
        columns["dcgm_tensor_active"].append(getattr(row, "dcgm_tensor_active", None))
        columns["raw_path"].append(row.raw_path or "")
        columns["captured_at"].append(row.captured_at or "")
        columns["notes"].append(row.notes or "")
        columns["extra_json"].append(json.dumps(row.extra or {}, sort_keys=True))

    return pa.table(columns, schema=schema)


def _read_champion_select(campaign_dir: Path) -> dict[str, Any] | None:
    """Load ``<campaign_dir>/champion_select.json`` (champion_select verb output)
    or None. Lenient: a malformed file is treated as absent (the lake row records
    the empty pick rather than failing the publish)."""
    path = campaign_dir / "champion_select.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def build_campaign_row(
    campaign_dir: Path,
    rows: list[AtlasCell],
    *,
    published_at_utc: datetime | None = None,
    publisher_operator: str | None = None,
    publisher_host: str | None = None,
) -> Any:
    """Build a single-row ``pyarrow.Table`` for perf_tune_report_campaign_v1.

    Provenance fields:

    - ``campaign_id``: campaign dir basename
    - ``slug``: same as campaign_id (kept as a separate field for join
      ergonomics in dbt downstream)
    - ``captured_at_utc``: parsed from the campaign dir's
      ``YYYYMMDDTHHMMSSZ`` suffix
    - ``operator``, ``cluster_context``: parsed from SOURCE.md if
      present, else empty strings
    - ``config_yaml_sha256``, ``source_md_sha256``: file digests for
      audit/reproducibility
    - ``cell_count``, ``backends``, ``atlas_row_count``: derived from
      the loaded rows
    - ``published_at_utc``, ``publisher_operator``, ``publisher_host``:
      who/when ran the publish step
    """
    import pyarrow as pa  # type: ignore[import-not-found]

    campaign_id = campaign_dir.name
    captured_at = parse_campaign_utc(campaign_id)
    source_md_path = campaign_dir / "SOURCE.md"
    config_yaml_path = campaign_dir / "config.yaml"
    metadata = parse_source_md(source_md_path)

    backends = sorted({row.backend for row in rows if row.backend})
    cell_count = len({row.cell_id for row in rows})
    render_status = read_render_status(campaign_dir)

    schema = pa.schema(
        [
            pa.field("campaign_id", pa.string(), nullable=False),
            pa.field("slug", pa.string(), nullable=False),
            pa.field(
                "captured_at_utc",
                pa.timestamp("us", tz="UTC"),
                nullable=False,
            ),
            pa.field("operator", pa.string(), nullable=False),
            pa.field("cluster_context", pa.string(), nullable=False),
            pa.field("config_yaml_sha256", pa.string(), nullable=False),
            pa.field("source_md_sha256", pa.string(), nullable=False),
            pa.field("cell_count", pa.int64(), nullable=False),
            pa.field("backends", pa.string(), nullable=False),
            pa.field("atlas_row_count", pa.int64(), nullable=False),
            # Completeness provenance (added v1.26.0). Downstream Iceberg
            # registration: append these three columns to the campaign_v1
            # intake table. ``sol_complete``
            # is the consumer-facing flag for "this campaign carries
            # Speed-of-Light rooflines"; ``omitted_pages`` is a comma-joined
            # list of report pages that were omitted (with reasons recorded
            # in the campaign's report_status.json).
            pa.field("sol_complete", pa.bool_(), nullable=False),
            pa.field("plot_ready_points", pa.int64(), nullable=False),
            pa.field("omitted_pages", pa.string(), nullable=False),
            # dcgm_grounded (added v1.29.0): True when the campaign carries
            # DCGM workload-level byte/FLOP grounding (page 6). The L2/L3
            # analog of sol_complete (L1 roofline) -- a campaign can be
            # sol_complete=True off zymtrace alone yet dcgm_grounded=False.
            # Downstream: append this column (default False for older partitions).
            pa.field("dcgm_grounded", pa.bool_(), nullable=False),
            # sol_per_arm_complete (added v1.71.0): True when baseline AND every
            # variant arm carries a roofline (PER-ARM coverage), vs the
            # CAMPAIGN-level sol_complete above ("any SoL page rendered"). Lets
            # lake consumers find multi-arm campaigns with an uncovered variant
            # (the gap the per-arm publish gate closes). Downstream: append this
            # column (default True for older partitions).
            pa.field("sol_per_arm_complete", pa.bool_(), nullable=False),
            # partial_pages (added v1.30.0): comma-joined names of pages that
            # rendered but carry less than a full measurement (e.g. page-5
            # %SoL-only / arithmetic-intensity-unmeasured). Empty string when
            # none. Recorded, not gated -- lets lake consumers filter partial
            # roofline data. Downstream: append this column (default '' for older
            # partitions).
            pa.field("partial_pages", pa.string(), nullable=False),
            # verdict_tier (added v1.32.0): "draft" (default) or "verdict". A
            # campaign published as "verdict" passed the verdict-rigor gate
            # (controlled + metric-isolated + fair-baseline provenance); "draft"
            # is exploratory/provisional. Downstream: append this column (default
            # 'draft' for older partitions).
            pa.field("verdict_tier", pa.string(), nullable=False),
            # focus + sol_rigor (added v1.33.0): EVERY measurement run now
            # publishes (no latency-bound/proxy refusal). `focus` is the run's
            # intent ("latency"|"throughput"|"mixed"); `sol_rigor` is the highest
            # SoL evidence level present ("L4" ncu | "L3" DCGM | "L1" zymtrace
            # proxy | "none"). Consumers filter side-by-side comparisons on
            # focus and judge SoL tightness on sol_rigor (not on whether the row
            # exists). Downstream: append these two columns (defaults 'mixed' /
            # 'none' for older partitions).
            pa.field("focus", pa.string(), nullable=False),
            pa.field("sol_rigor", pa.string(), nullable=False),
            pa.field(
                "published_at_utc",
                pa.timestamp("us", tz="UTC"),
                nullable=False,
            ),
            pa.field("publisher_operator", pa.string(), nullable=False),
            pa.field("publisher_host", pa.string(), nullable=False),
            # experiment_id / experiment_family / evidence_bundle_path (added
            # v1.34.0): the cross-experiment join + grouping keys. ``experiment_id``
            # is the evidence-bundle run-id = cluster ``experiment=<id-slug>`` label
            # value; when campaign_init was run with --experiment-id it equals
            # ``campaign_id`` (so a single key joins lake row <-> cluster objects
            # <-> evidence bundle), otherwise it defaults to ``campaign_id``.
            # ``experiment_family`` (e.g. "nvfp4-kv", "warp-decode", "deepep") makes
            # "all <family> vs the fp8 baseline" a first-class GROUP BY instead of a
            # grep over notes. ``evidence_bundle_path`` is the on-disk bundle dir.
            # Downstream: append these three columns to
            # the campaign_v1 intake table (defaults: experiment_id
            # = campaign_id, '' for the other two on older partitions).
            pa.field("experiment_id", pa.string(), nullable=False),
            pa.field("experiment_family", pa.string(), nullable=False),
            pa.field("evidence_bundle_path", pa.string(), nullable=False),
            # Source-code attribution (added for the durable-lineage workstream):
            # the experiment's link to the ACTUAL code under test, lifted from the
            # bundle's ```provenance``` block (campaign_init copies it into the
            # campaign config.yaml + flat SOURCE.md bullets; see provenance.py).
            # This closes the gap where the vLLM commit/branch lived only in
            # SOURCE.md prose and never in the lake -- now
            # "run_id -> source commit/branch -> verdict -> metrics" is one query.
            # ``experiment_status`` is the lifecycle (active|verified|refuted|
            # incomplete|superseded), distinct from atlas_v1.status (the per-cell
            # full|partial|failed|evicted), so negative + partial results are
            # first-class + queryable. Downstream: append these columns to
            # the campaign_v1 intake table (default '' / 'active').
            pa.field("title", pa.string(), nullable=False),
            pa.field("experiment_status", pa.string(), nullable=False),
            pa.field("tags", pa.string(), nullable=False),
            pa.field("hypothesis", pa.string(), nullable=False),
            pa.field("vllm_repo", pa.string(), nullable=False),
            pa.field("vllm_branch", pa.string(), nullable=False),
            pa.field("vllm_commit", pa.string(), nullable=False),
            pa.field("vllm_image", pa.string(), nullable=False),
            # image_digest = the IMMUTABLE sha256 content pin (the tag is mutable), the
            # image-side analog of vllm_commit. '' for older partitions. Downstream: append.
            pa.field("vllm_image_digest", pa.string(), nullable=False),
            pa.field("vllm_pip_version", pa.string(), nullable=False),
            pa.field("delivery", pa.string(), nullable=False),
            # overlay_mode = the delivery-ladder sub-tier when delivery=overlay
            # (subpath|patchset-initcontainer|pythonpath-sitecustomize). '' otherwise. Downstream: append.
            pa.field("overlay_mode", pa.string(), nullable=False),
            pa.field("code_repo", pa.string(), nullable=False),
            pa.field("code_sha", pa.string(), nullable=False),
            # Champion synthesis (added v1.66.0): the production pick from the
            # campaign's champion_select.json. ``champion_tier`` is the headline
            # DRAFT/VERDICT decision; ``recommended_cell`` / ``recommended_engine``
            # name the variant to ship; ``champion_baseline_cell`` is the
            # reference it beat. All '' when the campaign carries no
            # champion_select.json. Downstream: append these four columns (default
            # '' for older partitions). The per-variant detail is champion_v1.
            pa.field("recommended_cell", pa.string(), nullable=False),
            pa.field("recommended_engine", pa.string(), nullable=False),
            pa.field("champion_tier", pa.string(), nullable=False),
            pa.field("champion_baseline_cell", pa.string(), nullable=False),
            # next_lever (added: AGENTS.md "Always ship an actionable path-forward" /
            # performance-ratchet mandate). The campaign-level path-forward: the next
            # change to move a metric further, or "frontier-exhausted: <evidence>".
            # Empty is REFUSED under publish_to_lake --strict (next_lever_problems), so
            # the lake itself answers "what is the next lever for X". Downstream: append
            # this column (default '' for older partitions).
            pa.field("next_lever", pa.string(), nullable=False),
            # close_reason (added: AGENTS.md "Always be grinding" / grind discipline, principle i).
            # When a campaign CLOSES an investigation it must record a MEASURED reason -- one of
            # {beat-target, measured-plateau, infra-wall}; '' = still open (no close claimed). An
            # invalid/absent-but-claimed-close reason is REFUSED under --strict, recorded+warned
            # otherwise. Downstream: append this column (default '' for older partitions).
            pa.field("close_reason", pa.string(), nullable=False),
        ]
    )
    champion = _read_champion_select(campaign_dir) or {}
    published_at = published_at_utc or datetime.now(timezone.utc)
    publisher_op = publisher_operator or os.environ.get(PUBLISHER_ENV) or getpass.getuser()
    publisher_h = publisher_host or socket.gethostname()

    columns: dict[str, list[Any]] = {
        "campaign_id": [campaign_id],
        "slug": [campaign_id],
        "captured_at_utc": [captured_at],
        "operator": [metadata.get("operator", "")],
        "cluster_context": [metadata.get("cluster_context", metadata.get("cluster", ""))],
        "config_yaml_sha256": [_sha256_of(config_yaml_path)],
        "source_md_sha256": [_sha256_of(source_md_path)],
        "cell_count": [cell_count],
        "backends": [",".join(backends)],
        "atlas_row_count": [len(rows)],
        "sol_complete": [render_status.sol_complete],
        "plot_ready_points": [render_status.plot_ready_points],
        "omitted_pages": [render_status.omitted_pages],
        "dcgm_grounded": [render_status.dcgm_grounded],
        "sol_per_arm_complete": [render_status.sol_per_arm_complete],
        # Auto-downgrade an unsupported verdict claim to "draft" instead of
        # refusing to publish (always-publish policy v1.33.0): a verdict_tier=
        # verdict without the controlled/metric/baseline provenance lands as
        # draft (visible in the lake) rather than blocking the run.
        "verdict_tier": [_effective_verdict_tier(campaign_dir)],
        "partial_pages": [render_status.partial_pages],
        "focus": [render_status.focus],
        "sol_rigor": [render_status.sol_rigor],
        "published_at_utc": [published_at],
        "publisher_operator": [publisher_op],
        "publisher_host": [publisher_h],
        # Join + grouping keys; experiment_id defaults to campaign_id so the row
        # always carries a non-empty join key even for pre-v1.34.0 campaigns.
        "experiment_id": [metadata.get("experiment_id") or campaign_id],
        "experiment_family": [metadata.get("family", "")],
        "evidence_bundle_path": [metadata.get("evidence_bundle_path", "")],
        # Source-code attribution columns. campaign_init wrote these as flat
        # `- <key>: <value>` bullets into the campaign SOURCE.md (from the
        # bundle's ```provenance``` block), so parse_source_md already lifted
        # them into `metadata`. Default '' / 'active' when a bundle predates the
        # provenance convention (still publishes; the gap is visible in the lake).
        "title": [metadata.get("title", "")],
        "experiment_status": [metadata.get("experiment_status", "active")],
        "tags": [metadata.get("tags", "")],
        "hypothesis": [metadata.get("hypothesis", "")],
        "vllm_repo": [metadata.get("vllm_repo", "")],
        "vllm_branch": [metadata.get("vllm_branch", "")],
        "vllm_commit": [metadata.get("vllm_commit", "")],
        "vllm_image": [metadata.get("vllm_image", "")],
        "vllm_image_digest": [metadata.get("vllm_image_digest", "")],
        "vllm_pip_version": [metadata.get("vllm_pip_version", "")],
        "delivery": [metadata.get("delivery", "")],
        "overlay_mode": [metadata.get("overlay_mode", "")],
        "code_repo": [metadata.get("code_repo", "")],
        "code_sha": [metadata.get("code_sha", "")],
        "recommended_cell": [champion.get("recommended_cell") or ""],
        "recommended_engine": [champion.get("recommended_engine") or ""],
        "champion_tier": [champion.get("tier") or ""],
        "champion_baseline_cell": [champion.get("baseline_cell") or ""],
        "next_lever": [read_next_lever(campaign_dir)],
        "close_reason": [read_close_reason(campaign_dir)],
    }
    return pa.table(columns, schema=schema)


def _walk_cell_artifacts(campaign_dir: Path, filename: str) -> "list[tuple[str, Path, dict[str, Any]]]":
    """Walk ``<campaign_dir>/cells/*/<filename>`` and load each JSON.

    Returns a list of ``(cell_id, path, payload)`` ordered by cell dir name.
    Light-weight (no matplotlib / renderer import): the publish path must not
    drag in the rendering deps. Malformed JSON is skipped with a stderr warning
    rather than aborting publish (the renderer already gates malformed payloads
    at render time, which must run before publish).
    """
    out: list[tuple[str, Path, dict[str, Any]]] = []
    cells_dir = campaign_dir / "cells"
    if not cells_dir.is_dir():
        return out
    for cell_dir in sorted(p for p in cells_dir.iterdir() if p.is_dir()):
        path = cell_dir / filename
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"WARNING: skipping malformed {path} for sol_v1", file=sys.stderr)
            continue
        out.append((cell_dir.name, path, payload))
    return out


def _resolve_sol_ceilings(campaign_dir: Path, rows: list[AtlasCell]) -> "tuple[dict[str, Any] | None, str | None, str]":
    """Find + load sol-ceilings.yaml and pick the hardware key.

    Mirrors ``render_report.discover_sol_inputs`` but without importing the
    (matplotlib-pulling) renderer package. Returns
    ``(ceilings_dict_or_None, hw_key_or_None, yaml_sha256)``. The sha is ''
    when no YAML was found.
    """
    from tools.perf_tune_report.renderer import sol_roofline  # local: stdlib-only module

    env_override = os.environ.get("SOL_CEILINGS_YAML", "").strip()
    if env_override == "disable":
        return None, None, ""
    yaml_path: Path | None = None
    if env_override:
        cand = Path(env_override).expanduser().resolve()
        yaml_path = cand if cand.is_file() else None
    else:
        cur = campaign_dir.resolve()
        # Canonical bundle name first, then a name-agnostic bundle-root fallback
        # (<bundle>/configs/sol-ceilings.yaml) so a future submodule rename needs
        # no code edit; the campaign's own bundle root is reached first.
        relcands = (Path("perf-tune-report") / "configs" / "sol-ceilings.yaml",
                    Path("configs") / "sol-ceilings.yaml")
        for parent in [cur, *cur.parents]:
            for rel in relcands:
                cand = parent / rel
                if cand.is_file():
                    yaml_path = cand
                    break
            if yaml_path is not None:
                break
    if yaml_path is None:
        return None, None, ""
    yaml_sha = _sha256_of(yaml_path)
    try:
        ceilings = sol_roofline.load_ceilings(yaml_path)
    except Exception:  # noqa: BLE001 - malformed YAML: degrade to no-ceiling L1 rows, still publish
        return None, None, yaml_sha
    hw_key = sol_roofline.hardware_key_for_atlas(rows)
    if hw_key is None or hw_key not in ceilings:
        return ceilings, None, yaml_sha
    return ceilings, hw_key, yaml_sha


def build_sol_table(
    campaign_dir: Path,
    campaign_id: str,
    rows: list[AtlasCell],
    *,
    captured_at_utc: datetime,
    published_at_utc: datetime,
    focus: str,
    sol_rigor: str,
) -> Any:
    """Build a ``pyarrow.Table`` for the perf_tune_report_sol_v1 schema.

    One row per ``(cell_id, category, sol_level)`` (L4 adds ``kernel_name``):

    - **L1** (``cells/*/kernels.json`` + sol-ceilings.yaml): zymtrace per-category
      sample-share + the natural ceiling each category is bound by. ``pct_sol`` is
      NULL at L1 -- the artifact carries time-share, not byte/FLOP attribution.
    - **L2** (``cells/*/dcgm_correlation.json`` ``per_category_attribution``):
      zymtrace x DCGM cross-attribution -- ``pct_sol`` = ``sol_pct_bw`` (bandwidth
      categories) or ``sol_pct_compute`` (compute categories).
    - **L3** (``dcgm_correlation.json`` ``resources``): DCGM workload-level
      byte/FLOP grounding; ``category`` carries the resource ``peak_key``.
    - **L4** (``cells/*/ncu_kernels.json`` ``kernels``): ncu per-kernel
      arithmetic intensity + achieved %peak.

    Empty (0 rows) when the campaign has no ``cells/*/`` SoL artifacts -- the
    table is still written + uploaded so the lake records "this campaign carried
    no per-category SoL evidence" explicitly.
    """
    import pyarrow as pa  # type: ignore[import-not-found]

    schema = pa.schema(
        [
            pa.field("campaign_id", pa.string(), nullable=False),
            pa.field("cell_id", pa.string(), nullable=False),
            pa.field("category", pa.string(), nullable=False),
            pa.field("sol_level", pa.string(), nullable=False),
            pa.field("gpu_time_share_pct", pa.float64(), nullable=True),
            pa.field("pct_sol", pa.float64(), nullable=True),
            pa.field("bound", pa.string(), nullable=True),
            pa.field("ceiling_key", pa.string(), nullable=True),
            pa.field("ceiling_value", pa.float64(), nullable=True),
            pa.field("ceiling_units", pa.string(), nullable=True),
            pa.field("measured_value", pa.float64(), nullable=True),
            pa.field("measured_units", pa.string(), nullable=True),
            pa.field("attributed_bytes_total", pa.float64(), nullable=True),
            pa.field("attributed_flops_total", pa.float64(), nullable=True),
            pa.field("arithmetic_intensity_flops_per_byte", pa.float64(), nullable=True),
            pa.field("kernel_name", pa.string(), nullable=True),
            pa.field("hw_key", pa.string(), nullable=True),
            pa.field("source_artifact", pa.string(), nullable=False),
            pa.field("source_artifact_sha256", pa.string(), nullable=False),
            pa.field("sol_ceilings_yaml_sha256", pa.string(), nullable=False),
            pa.field("captured_at_utc", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("published_at_utc", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("focus", pa.string(), nullable=False),
            pa.field("sol_rigor", pa.string(), nullable=False),
        ]
    )
    columns: dict[str, list[Any]] = {field.name: [] for field in schema}

    def emit(**kw: Any) -> None:
        for f in schema:
            if f.name in ("campaign_id", "captured_at_utc", "published_at_utc", "focus", "sol_rigor"):
                continue
            columns[f.name].append(kw.get(f.name))
        columns["campaign_id"].append(campaign_id)
        columns["captured_at_utc"].append(captured_at_utc)
        columns["published_at_utc"].append(published_at_utc)
        columns["focus"].append(focus)
        columns["sol_rigor"].append(sol_rigor)

    ceilings, hw_key, yaml_sha = _resolve_sol_ceilings(campaign_dir, rows)

    # L1 -- zymtrace per-category sample-share + ceiling
    from tools.perf_tune_report.renderer import sol_roofline  # stdlib-only
    for cell_id, path, payload in _walk_cell_artifacts(campaign_dir, "kernels.json"):
        sha = _sha256_of(path)
        if ceilings is not None and hw_key is not None:
            hw_data = ceilings.get(hw_key, {})
            cat_map = ceilings.get("category_ceiling_map", {})
            for r in sol_roofline.compute_category_sol(payload, hw_data, cat_map):
                emit(
                    cell_id=cell_id, category=r["category"], sol_level="L1",
                    gpu_time_share_pct=r["time_share_pct"], pct_sol=None,
                    bound=r["bound"], ceiling_key=r["ceiling_metric"],
                    ceiling_value=r["ceiling_value"], ceiling_units=r["ceiling_units"],
                    measured_value=None, measured_units=None,
                    attributed_bytes_total=None, attributed_flops_total=None,
                    arithmetic_intensity_flops_per_byte=None, kernel_name=None,
                    hw_key=hw_key, source_artifact="kernels.json",
                    source_artifact_sha256=sha, sol_ceilings_yaml_sha256=yaml_sha,
                )
        else:
            # No ceilings resolvable: still record per-category time-share.
            per_cat = payload.get("per_category", {}) or {}
            total = sum(int(v) for v in per_cat.values()) or 1
            for cat, samples in per_cat.items():
                if int(samples) == 0:
                    continue
                emit(
                    cell_id=cell_id, category=cat, sol_level="L1",
                    gpu_time_share_pct=100.0 * int(samples) / total, pct_sol=None,
                    bound=None, ceiling_key=None, ceiling_value=None, ceiling_units=None,
                    measured_value=None, measured_units=None,
                    attributed_bytes_total=None, attributed_flops_total=None,
                    arithmetic_intensity_flops_per_byte=None, kernel_name=None,
                    hw_key=hw_key, source_artifact="kernels.json",
                    source_artifact_sha256=sha, sol_ceilings_yaml_sha256=yaml_sha,
                )

    # L2 + L3 -- DCGM cross-attribution + workload resources
    for cell_id, path, payload in _walk_cell_artifacts(campaign_dir, "dcgm_correlation.json"):
        sha = _sha256_of(path)
        cell_hw = payload.get("hw_key")
        for c in payload.get("per_category_attribution", []) or []:
            bound = c.get("bound")
            if bound == "compute":
                pct, meas, munits = c.get("sol_pct_compute"), c.get("effective_tflops_during_category_window"), "TFLOPS"
            elif bound == "bandwidth":
                pct, meas, munits = c.get("sol_pct_bw"), c.get("effective_bw_during_category_window"), "bytes/s"
            else:
                pct = c.get("sol_pct_bw") if c.get("sol_pct_bw") is not None else c.get("sol_pct_compute")
                meas = munits = None
            emit(
                cell_id=cell_id, category=c.get("category"), sol_level="L2",
                gpu_time_share_pct=c.get("time_share_pct"), pct_sol=pct,
                bound=bound, ceiling_key=c.get("ceiling_metric"),
                ceiling_value=None, ceiling_units=None,
                measured_value=meas, measured_units=munits,
                attributed_bytes_total=c.get("attributed_bytes_total"),
                attributed_flops_total=c.get("attributed_flops_total"),
                arithmetic_intensity_flops_per_byte=None, kernel_name=None,
                hw_key=cell_hw, source_artifact="dcgm_correlation.json",
                source_artifact_sha256=sha, sol_ceilings_yaml_sha256=yaml_sha,
            )
        for res in payload.get("resources", []) or []:
            bytes_ps, tflops = res.get("measured_bytes_per_s"), res.get("measured_tflops_avg")
            if bytes_ps is not None:
                meas, munits = bytes_ps, "bytes/s"
            elif tflops is not None:
                meas, munits = tflops, "TFLOPS"
            else:
                meas, munits = None, None
            emit(
                cell_id=cell_id, category=res.get("peak_key"), sol_level="L3",
                gpu_time_share_pct=None, pct_sol=res.get("sol_pct"),
                bound=None, ceiling_key=res.get("peak_key"),
                ceiling_value=res.get("peak_per_gpu"), ceiling_units=res.get("peak_per_gpu_units"),
                measured_value=meas, measured_units=munits,
                attributed_bytes_total=res.get("measured_bytes_total"),
                attributed_flops_total=None,
                arithmetic_intensity_flops_per_byte=None, kernel_name=None,
                hw_key=cell_hw, source_artifact="dcgm_correlation.json",
                source_artifact_sha256=sha, sol_ceilings_yaml_sha256=yaml_sha,
            )

    # L4 -- ncu per-kernel arithmetic intensity
    for cell_id, path, payload in _walk_cell_artifacts(campaign_dir, "ncu_kernels.json"):
        sha = _sha256_of(path)
        cell_hw = payload.get("hw_key")
        for k in payload.get("kernels", []) or []:
            pct = k.get("achieved_dram_pct_peak")
            if pct is None:
                pct = k.get("achieved_sm_pct_peak")
            emit(
                cell_id=cell_id, category=k.get("category") or "Other", sol_level="L4",
                gpu_time_share_pct=None, pct_sol=pct, bound=None,
                ceiling_key=None, ceiling_value=None, ceiling_units=None,
                measured_value=k.get("achieved_tflops"), measured_units="TFLOPS",
                attributed_bytes_total=k.get("dram_bytes_total"),
                attributed_flops_total=k.get("sm_flops_total"),
                arithmetic_intensity_flops_per_byte=k.get("arithmetic_intensity_flops_per_byte"),
                kernel_name=k.get("name"),
                hw_key=cell_hw, source_artifact="ncu_kernels.json",
                source_artifact_sha256=sha, sol_ceilings_yaml_sha256=yaml_sha,
            )

    return pa.table(columns, schema=schema)


def build_tpm_table(
    rows: list[AtlasCell],
    campaign_id: str,
    *,
    captured_at_utc: datetime,
    published_at_utc: datetime,
    gpus_per_node: int = 8,
    ttft_sla_ms: float | None = None,
    tpot_sla_ms: float | None = None,
) -> Any:
    """Build a ``pyarrow.Table`` for the perf_tune_report_tpm_v1 schema.

    One row per ``(model, hardware, quant, tensor_parallel, parallel_strategy,
    mtp, operating_point, basis)``. ``operating_point`` is ``peak`` always, plus
    ``sla`` when the campaign ``config.yaml`` ``tpm:`` block declared SLA
    thresholds (``publish`` reads them via ``discover_tpm_config`` and passes
    ``ttft_sla_ms`` / ``tpot_sla_ms`` here). ``basis`` is one of ``per_gpu`` /
    ``per_replica`` / ``per_node``. ``total_tpm`` is null for backends that emit
    no total-token line.

    Empty (0 rows) when the campaign has no throughput-bearing atlas rows -- the
    table is still written + uploaded so the lake records the absence explicitly.
    """
    import pyarrow as pa  # type: ignore[import-not-found]

    from tools.perf_tune_report.tpm_summary import BASES, compute_tpm_summary

    schema = pa.schema(
        [
            pa.field("campaign_id", pa.string(), nullable=False),
            pa.field("model", pa.string(), nullable=False),
            pa.field("hardware", pa.string(), nullable=False),
            pa.field("quant", pa.string(), nullable=False),
            pa.field("tensor_parallel", pa.int64(), nullable=False),
            pa.field("parallel_strategy", pa.string(), nullable=False),
            pa.field("mtp", pa.bool_(), nullable=False),
            pa.field("operating_point", pa.string(), nullable=False),
            pa.field("basis", pa.string(), nullable=False),
            pa.field("gpus_per_node", pa.int64(), nullable=False),
            pa.field("concurrency", pa.int64(), nullable=False),
            pa.field("output_tpm", pa.float64(), nullable=True),
            pa.field("total_tpm", pa.float64(), nullable=True),
            pa.field("ttft_avg_ms", pa.float64(), nullable=True),
            pa.field("tpot_median_ms", pa.float64(), nullable=True),
            pa.field("itl_avg_ms", pa.float64(), nullable=True),
            # Shape + warm/cold carry-through (added v1.42.0) so pricing can
            # filter capacity by ISL/OSL + cache mode.
            pa.field("mean_isl", pa.float64(), nullable=True),
            pa.field("mean_osl", pa.float64(), nullable=True),
            pa.field("cache_mode", pa.string(), nullable=False),
            pa.field("captured_at_utc", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("published_at_utc", pa.timestamp("us", tz="UTC"), nullable=False),
        ]
    )
    columns: dict[str, list[Any]] = {field.name: [] for field in schema}

    summary = compute_tpm_summary(
        rows,
        ttft_sla_ms=ttft_sla_ms,
        tpot_sla_ms=tpot_sla_ms,
        gpus_per_node=gpus_per_node,
    )
    for g in summary.groups:
        for point in (g.peak, g.sla):
            if point is None:
                continue
            for basis in BASES:
                columns["campaign_id"].append(campaign_id)
                columns["model"].append(g.model)
                columns["hardware"].append(g.hardware)
                columns["quant"].append(g.quant)
                columns["tensor_parallel"].append(int(g.tensor_parallel))
                columns["parallel_strategy"].append(g.parallel_strategy)
                columns["mtp"].append(bool(g.mtp))
                columns["operating_point"].append(point.operating_point)
                columns["basis"].append(basis)
                columns["gpus_per_node"].append(int(gpus_per_node))
                columns["concurrency"].append(int(point.concurrency))
                columns["output_tpm"].append(getattr(point, f"output_tpm_{basis}"))
                columns["total_tpm"].append(getattr(point, f"total_tpm_{basis}"))
                columns["ttft_avg_ms"].append(point.ttft_avg_ms)
                columns["tpot_median_ms"].append(point.tpot_median_ms)
                columns["itl_avg_ms"].append(point.itl_avg_ms)
                columns["mean_isl"].append(g.mean_isl)
                columns["mean_osl"].append(g.mean_osl)
                columns["cache_mode"].append(g.cache_mode or "unknown")
                columns["captured_at_utc"].append(captured_at_utc)
                columns["published_at_utc"].append(published_at_utc)

    return pa.table(columns, schema=schema)


def _cell_power_watts_per_gpu(campaign_dir: Path, cell_id: str) -> float | None:
    """Read mean per-GPU power (watts) from a cell's dcgm_correlation.json.

    Returns None when the file/field is absent (no DCGM power captured) so
    tokens-per-watt degrades to null rather than blocking publish."""
    if not cell_id:
        return None
    path = campaign_dir / "cells" / cell_id / "dcgm_correlation.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    val = payload.get("power_watts_per_gpu")
    return float(val) if isinstance(val, (int, float)) and val > 0 else None


def build_cost_table(
    campaign_dir: Path,
    rows: list[AtlasCell],
    campaign_id: str,
    *,
    captured_at_utc: datetime,
    published_at_utc: datetime,
    gpus_per_node: int = 8,
    ttft_sla_ms: float | None = None,
    tpot_sla_ms: float | None = None,
    usd_per_gpu_hour: dict[str, float] | None = None,
) -> Any:
    """Build a ``pyarrow.Table`` for the perf_tune_report_cost_v1 economics/TCO schema.

    One row per ``(model, hardware, quant, tensor_parallel, parallel_strategy,
    mtp, operating_point)``. Cost columns (``usd_per_1m_*``, ``usd_per_gpu_hour``)
    are null unless the campaign ``cost:`` block supplied the hardware's
    $/GPU-hour; energy columns (``tokens_per_watt``, ``power_watts_per_gpu``)
    are null unless the point's cell carries DCGM power in dcgm_correlation.json.
    Both gaps are RECORDED (not blocking) per the always-publish policy.
    """
    import pyarrow as pa  # type: ignore[import-not-found]

    from tools.perf_tune_report.tpm_summary import compute_tpm_summary

    schema = pa.schema(
        [
            pa.field("campaign_id", pa.string(), nullable=False),
            pa.field("model", pa.string(), nullable=False),
            pa.field("hardware", pa.string(), nullable=False),
            pa.field("quant", pa.string(), nullable=False),
            pa.field("tensor_parallel", pa.int64(), nullable=False),
            pa.field("parallel_strategy", pa.string(), nullable=False),
            pa.field("mtp", pa.bool_(), nullable=False),
            pa.field("operating_point", pa.string(), nullable=False),
            pa.field("concurrency", pa.int64(), nullable=False),
            pa.field("usd_per_gpu_hour", pa.float64(), nullable=True),
            pa.field("usd_per_1m_output_tokens", pa.float64(), nullable=True),
            pa.field("usd_per_1m_total_tokens", pa.float64(), nullable=True),
            pa.field("power_watts_per_gpu", pa.float64(), nullable=True),
            pa.field("tokens_per_watt", pa.float64(), nullable=True),
            pa.field("captured_at_utc", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("published_at_utc", pa.timestamp("us", tz="UTC"), nullable=False),
        ]
    )
    columns: dict[str, list[Any]] = {field.name: [] for field in schema}

    summary = compute_tpm_summary(
        rows,
        ttft_sla_ms=ttft_sla_ms,
        tpot_sla_ms=tpot_sla_ms,
        gpus_per_node=gpus_per_node,
        usd_per_gpu_hour=usd_per_gpu_hour,
    )
    for g in summary.groups:
        for point in (g.peak, g.sla):
            if point is None:
                continue
            power = _cell_power_watts_per_gpu(campaign_dir, point.cell_id)
            tokens_per_watt = (
                point.output_tps_per_gpu / power
                if power is not None and power > 0
                else None
            )
            columns["campaign_id"].append(campaign_id)
            columns["model"].append(g.model)
            columns["hardware"].append(g.hardware)
            columns["quant"].append(g.quant)
            columns["tensor_parallel"].append(int(g.tensor_parallel))
            columns["parallel_strategy"].append(g.parallel_strategy)
            columns["mtp"].append(bool(g.mtp))
            columns["operating_point"].append(point.operating_point)
            columns["concurrency"].append(int(point.concurrency))
            columns["usd_per_gpu_hour"].append(point.usd_per_gpu_hour)
            columns["usd_per_1m_output_tokens"].append(point.usd_per_1m_output_tokens)
            columns["usd_per_1m_total_tokens"].append(point.usd_per_1m_total_tokens)
            columns["power_watts_per_gpu"].append(power)
            columns["tokens_per_watt"].append(tokens_per_watt)
            columns["captured_at_utc"].append(captured_at_utc)
            columns["published_at_utc"].append(published_at_utc)

    return pa.table(columns, schema=schema)


_QUALITY_KEY_HINTS = ("acc", "accept", "eval", "loss", "delta")


def _quality_metrics(extra: dict[str, Any]) -> dict[str, float]:
    """Extract numeric quality metrics from a cell's ``extra`` dict.

    Canonical (profile-and-optimize v1.63.0+): a ``quality_metrics: {name: value}`` sub-dict
    that accuracy/acceptance campaigns emit (distinct from hyperparameters). For
    campaigns predating that convention, fall back to flat numeric keys whose name
    looks like an accuracy / acceptance / eval / loss / delta metric (which
    excludes hyperparameters like batch_size, global_batch, learning_rate).
    """
    canonical = extra.get("quality_metrics")
    if isinstance(canonical, dict):
        return {
            str(k): float(v)
            for k, v in canonical.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
    out: dict[str, float] = {}
    for k, v in extra.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        if any(hint in str(k).lower() for hint in _QUALITY_KEY_HINTS):
            out[str(k)] = float(v)
    return out


def build_quality_table(
    rows: list[AtlasCell],
    campaign_id: str,
    *,
    captured_at_utc: datetime,
    published_at_utc: datetime,
) -> Any:
    """Build a ``pyarrow.Table`` for the perf_tune_report_quality_v1 schema.

    LONG format: one row per ``(campaign_id, cell_id, metric_kind, metric_name)``
    for every cell that declares ``extra["metric_kind"]`` (a training-accuracy /
    draft-acceptance run). Serving cells (no ``metric_kind``) contribute no rows,
    so a pure-serving campaign yields an empty (0-row) table -- still published.
    Heterogeneous per-campaign accuracy keys are normalized to (metric_name,
    metric_value) pairs via ``_quality_metrics`` so dashboards filter on
    metric_name instead of extracting JSON out of ``atlas_v1.extra_json``.
    """
    import pyarrow as pa  # type: ignore[import-not-found]

    schema = pa.schema(
        [
            pa.field("campaign_id", pa.string(), nullable=False),
            pa.field("cell_id", pa.string(), nullable=False),
            pa.field("model", pa.string(), nullable=False),
            pa.field("hardware", pa.string(), nullable=False),
            pa.field("quant", pa.string(), nullable=False),
            pa.field("tensor_parallel", pa.int64(), nullable=False),
            pa.field("parallel_strategy", pa.string(), nullable=False),
            pa.field("mtp", pa.bool_(), nullable=False),
            pa.field("metric_kind", pa.string(), nullable=False),
            pa.field("metric_name", pa.string(), nullable=False),
            pa.field("metric_value", pa.float64(), nullable=True),
            pa.field("captured_at_utc", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("published_at_utc", pa.timestamp("us", tz="UTC"), nullable=False),
        ]
    )
    columns: dict[str, list[Any]] = {field.name: [] for field in schema}

    for row in rows:
        extra = row.extra or {}
        metric_kind = extra.get("metric_kind")
        if not metric_kind:
            continue
        for metric_name, metric_value in sorted(_quality_metrics(extra).items()):
            columns["campaign_id"].append(campaign_id)
            columns["cell_id"].append(row.cell_id)
            columns["model"].append(row.model)
            columns["hardware"].append(row.hardware)
            columns["quant"].append(row.quant)
            columns["tensor_parallel"].append(int(row.tensor_parallel))
            columns["parallel_strategy"].append(row.parallel_strategy)
            columns["mtp"].append(bool(row.mtp))
            columns["metric_kind"].append(str(metric_kind))
            columns["metric_name"].append(metric_name)
            columns["metric_value"].append(metric_value)
            columns["captured_at_utc"].append(captured_at_utc)
            columns["published_at_utc"].append(published_at_utc)

    return pa.table(columns, schema=schema)


def build_champion_table(
    campaign_dir: Path,
    campaign_id: str,
    *,
    captured_at_utc: datetime,
    published_at_utc: datetime,
) -> Any:
    """Build a ``pyarrow.Table`` for the perf_tune_report_champion_v1 schema.

    One row per SELECTED variant (baseline + top-X) from the campaign's
    ``champion_select.json``. Carries the focus metric, %win vs baseline, SLO
    verdict, the 4-layer SoL summary, and is_recommended / champion_tier, so the
    "which X variants did we pick, which one ships, and what is its proof"
    question is one query against the lake. Empty (0 rows) when the campaign has
    no champion_select.json (a campaign that never ran champion_select).
    """
    import pyarrow as pa  # type: ignore[import-not-found]

    schema = pa.schema(
        [
            pa.field("campaign_id", pa.string(), nullable=False),
            pa.field("cell_id", pa.string(), nullable=False),
            pa.field("engine", pa.string(), nullable=False),
            pa.field("is_baseline", pa.bool_(), nullable=False),
            pa.field("is_recommended", pa.bool_(), nullable=False),
            pa.field("champion_tier", pa.string(), nullable=False),
            pa.field("focus", pa.string(), nullable=False),
            pa.field("focus_c", pa.int64(), nullable=False),
            pa.field("metric", pa.string(), nullable=False),
            pa.field("focus_metric", pa.float64(), nullable=True),
            pa.field("output_tps_per_gpu", pa.float64(), nullable=True),
            pa.field("tpot_median_ms", pa.float64(), nullable=True),
            pa.field("pct_win_vs_baseline", pa.float64(), nullable=True),
            pa.field("slo_verdict", pa.string(), nullable=False),
            pa.field("sol_rigor", pa.string(), nullable=False),
            pa.field("hbm_pct_sol", pa.float64(), nullable=True),
            pa.field("tensor_pct_sol", pa.float64(), nullable=True),
            pa.field("sm_active_pct", pa.float64(), nullable=True),
            pa.field("l1_present", pa.bool_(), nullable=False),
            pa.field("l2_present", pa.bool_(), nullable=False),
            pa.field("l3_present", pa.bool_(), nullable=False),
            pa.field("l4_present", pa.bool_(), nullable=False),
            pa.field("captured_at_utc", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("published_at_utc", pa.timestamp("us", tz="UTC"), nullable=False),
        ]
    )
    columns: dict[str, list[Any]] = {field.name: [] for field in schema}
    champion = _read_champion_select(campaign_dir)
    if champion:
        rec = champion.get("recommended_cell")
        tier = champion.get("tier") or ""
        focus = champion.get("focus") or ""
        focus_c = int(champion.get("focus_c") or 0)
        metric = champion.get("metric") or ""
        for v in champion.get("variants", []):
            sol = v.get("sol") or {}
            columns["campaign_id"].append(campaign_id)
            columns["cell_id"].append(v.get("cell_id", ""))
            columns["engine"].append(v.get("engine", ""))
            columns["is_baseline"].append(bool(v.get("is_baseline")))
            columns["is_recommended"].append(v.get("cell_id") == rec)
            columns["champion_tier"].append(tier)
            columns["focus"].append(focus)
            columns["focus_c"].append(focus_c)
            columns["metric"].append(metric)
            columns["focus_metric"].append(v.get("focus_metric"))
            columns["output_tps_per_gpu"].append(v.get("output_tps_per_gpu"))
            columns["tpot_median_ms"].append(v.get("tpot_median_ms"))
            columns["pct_win_vs_baseline"].append(v.get("pct_win_vs_baseline"))
            columns["slo_verdict"].append(v.get("slo_verdict", ""))
            columns["sol_rigor"].append(sol.get("sol_rigor", "none"))
            columns["hbm_pct_sol"].append(sol.get("hbm_pct_sol"))
            columns["tensor_pct_sol"].append(sol.get("tensor_pct_sol"))
            columns["sm_active_pct"].append(sol.get("sm_active_pct"))
            columns["l1_present"].append(bool(sol.get("l1_present")))
            columns["l2_present"].append(bool(sol.get("l2_present")))
            columns["l3_present"].append(bool(sol.get("l3_present")))
            columns["l4_present"].append(bool(sol.get("l4_present")))
            columns["captured_at_utc"].append(captured_at_utc)
            columns["published_at_utc"].append(published_at_utc)
    return pa.table(columns, schema=schema)


_HW_TOKEN_TO_KEY = {"B200": "b200_sm100", "GB300": "gb300_nvl72", "H100": "h100_sxm"}


def _peak_for(ceilings: dict[str, Any] | None, hw_token: str, quant: str) -> "tuple[float | None, float | None]":
    """(compute_peak_pflops_per_gpu, hbm_peak_tbps_per_gpu) from sol-ceilings,
    keyed by the roofline payload's bare hardware token + quant. (None, None)
    when ceilings are unavailable -> the %-of-peak columns are left null."""
    if not ceilings:
        return None, None
    key = _HW_TOKEN_TO_KEY.get((hw_token or "").split(" ")[0])
    hw = ceilings.get(key) if key else None
    if not hw:
        return None, None
    q = (quant or "").upper()
    ck = "nvfp4_dense_pflops"
    if q == "FP8":
        ck = "fp8_dense_pflops"
    elif q in ("BF16", "FP16"):
        ck = "bf16_dense_pflops"
    comp = (hw.get(ck) or {}).get("value")
    hbm = (hw.get("hbm3e_tbps") or hw.get("hbm3_tbps") or {}).get("value")
    return (float(comp) if comp else None), (float(hbm) if hbm else None)


def build_roofline_table(
    campaign_dir: Path,
    campaign_id: str,
    rows: list[AtlasCell],
    *,
    captured_at_utc: datetime,
    published_at_utc: datetime,
) -> Any:
    """Build a ``pyarrow.Table`` for the perf_tune_report_roofline_v1 schema.

    One row per prefill/decode roofline operating point across all
    ``cells/*/roofline_sweep.json`` artifacts. Carries the analytical roofline
    coordinates (arithmetic intensity, achieved compute/GPU, delivered HBM
    bytes/s/GPU) + the measured DCGM active fractions + the per-GPU ceilings, so
    a Superset scatter (x=AI, y=achieved TFLOP/s, line=ceiling) and the
    HBM%/tensor%/SM%-vs-concurrency curves render straight from the lake. Empty
    (0 rows) when the campaign has no roofline sweep.
    """
    import pyarrow as pa  # type: ignore[import-not-found]

    schema = pa.schema(
        [
            pa.field("campaign_id", pa.string(), nullable=False),
            pa.field("cell_id", pa.string(), nullable=False),
            pa.field("model", pa.string(), nullable=False),
            pa.field("hardware", pa.string(), nullable=False),
            pa.field("quant", pa.string(), nullable=False),
            pa.field("kv_dtype", pa.string(), nullable=False),
            pa.field("tensor_parallel", pa.int64(), nullable=False),
            pa.field("phase", pa.string(), nullable=False),          # decode | prefill
            pa.field("concurrency", pa.int64(), nullable=True),       # decode C (null for prefill)
            pa.field("isl", pa.int64(), nullable=True),
            pa.field("osl", pa.int64(), nullable=True),
            pa.field("median_tpot_ms", pa.float64(), nullable=True),
            pa.field("median_ttft_ms", pa.float64(), nullable=True),
            pa.field("output_throughput", pa.float64(), nullable=True),
            # analytical roofline coordinates (x, y, delivered BW)
            pa.field("arithmetic_intensity", pa.float64(), nullable=True),
            pa.field("achieved_tflops_per_gpu", pa.float64(), nullable=True),
            pa.field("hbm_delivered_Bps_per_gpu", pa.float64(), nullable=True),
            pa.field("hbm_delivered_pct", pa.float64(), nullable=True),   # delivered / peak * 100
            # measured DCGM active fractions (the utilization-vs-C curves)
            pa.field("dram_active_pct", pa.float64(), nullable=True),     # DRAM_ACTIVE * 100 (proxy)
            pa.field("tensor_active_pct", pa.float64(), nullable=True),
            pa.field("sm_active_pct", pa.float64(), nullable=True),
            # per-GPU ceilings + ridge (so the roofline line is queryable)
            pa.field("compute_peak_pflops_per_gpu", pa.float64(), nullable=True),
            pa.field("hbm_peak_tbps_per_gpu", pa.float64(), nullable=True),
            pa.field("ridge_ai", pa.float64(), nullable=True),
            pa.field("captured_at_utc", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("published_at_utc", pa.timestamp("us", tz="UTC"), nullable=False),
        ]
    )
    from tools.perf_tune_report import roofline_math  # pure module (no matplotlib)

    def _shape_for(payload: dict[str, Any]) -> "roofline_math.ModelShape | None":
        emb = payload.get("analytical_shape")
        if isinstance(emb, dict) and emb.get("hidden_size"):
            try:
                return roofline_math.shape_from_dict(emb)
            except Exception:  # noqa: BLE001
                pass
        return roofline_math.shape_for_model(payload.get("model", ""))

    def _coords(pt: dict[str, Any], phase: str, shape, quant: str, tp: int, kvd: str):
        """(arithmetic_intensity, achieved_tflops_per_gpu, hbm_delivered_Bps_per_gpu),
        preferring the importer-embedded per-point fields, else recomputing from the
        model shape (so re-rendered pre-embed campaigns get correct lake rows too)."""
        ai = pt.get("arithmetic_intensity")
        ach = pt.get("achieved_tflops_per_gpu")
        deliv = pt.get("hbm_delivered_Bps_per_gpu")
        if ai is not None or shape is None:
            return ai, ach, deliv
        if phase == "decode":
            c, rate = pt.get("c") or 1, pt.get("output_throughput")
            if not rate:
                return None, None, None
            ctx = int((pt.get("isl") or 256) + (pt.get("osl") or 512) // 2)
            ai = shape.decode_arithmetic_intensity(c, ctx, quant, kvd)
            union = (min(shape.n_routed_experts, shape.n_experts_per_tok * c)
                     if shape.is_moe else 0)
            bpt = shape.active_weight_bytes(union, quant) / c + shape.kv_bytes_per_token(ctx, kvd)
        else:
            isl, inp, dur = pt.get("isl"), pt.get("total_input_tokens"), pt.get("duration")
            rate = (inp / dur) if (inp and dur) else None
            if not rate or not isl:
                return None, None, None
            ai = shape.prefill_arithmetic_intensity(isl, quant)
            experts = shape.n_routed_experts if shape.is_moe else 0
            bpt = shape.active_weight_bytes(experts, quant) / max(int(isl), 1)
        return ai, shape.flop_per_token * rate / tp / 1e12, bpt * rate / tp

    columns: dict[str, list[Any]] = {field.name: [] for field in schema}
    ceilings, _hw_key, _sha = _resolve_sol_ceilings(campaign_dir, rows)
    for cell_id, _path, payload in _walk_cell_artifacts(campaign_dir, "roofline_sweep.json"):
        if payload.get("schema") != "roofline_sweep_points_v1":
            continue
        model = str(payload.get("model", ""))
        hw = str(payload.get("hardware", ""))
        quant = str(payload.get("quant", "NVFP4"))
        kvd = str(payload.get("kv_dtype", "fp8"))
        tp = int(payload.get("tensor_parallel") or 1)
        comp_pf, hbm_tb = _peak_for(ceilings, hw, quant)
        ridge = ((comp_pf * 1e15) / (hbm_tb * 1e12)) if (comp_pf and hbm_tb) else None
        shape = _shape_for(payload)
        for phase in ("decode", "prefill"):
            for pt in payload.get(phase, []) or []:
                ai_v, ach_v, delivered = _coords(pt, phase, shape, quant, tp, kvd)
                hbm_pct = (delivered / (hbm_tb * 1e12) * 100) if (delivered and hbm_tb) else None
                dram = pt.get("dram_active")
                ten = pt.get("tensor_active")
                sm = pt.get("sm_active")
                columns["campaign_id"].append(campaign_id)
                columns["cell_id"].append(cell_id)
                columns["model"].append(model)
                columns["hardware"].append(hw)
                columns["quant"].append(quant)
                columns["kv_dtype"].append(kvd)
                columns["tensor_parallel"].append(tp)
                columns["phase"].append(phase)
                columns["concurrency"].append(int(pt["c"]) if pt.get("c") is not None else None)
                columns["isl"].append(int(pt["isl"]) if pt.get("isl") is not None else None)
                columns["osl"].append(int(pt["osl"]) if pt.get("osl") is not None else None)
                columns["median_tpot_ms"].append(pt.get("median_tpot_ms"))
                columns["median_ttft_ms"].append(pt.get("median_ttft_ms"))
                columns["output_throughput"].append(pt.get("output_throughput"))
                columns["arithmetic_intensity"].append(ai_v)
                columns["achieved_tflops_per_gpu"].append(ach_v)
                columns["hbm_delivered_Bps_per_gpu"].append(delivered)
                columns["hbm_delivered_pct"].append(hbm_pct)
                columns["dram_active_pct"].append(dram * 100 if dram is not None else None)
                columns["tensor_active_pct"].append(ten * 100 if ten is not None else None)
                columns["sm_active_pct"].append(sm * 100 if sm is not None else None)
                columns["compute_peak_pflops_per_gpu"].append(comp_pf)
                columns["hbm_peak_tbps_per_gpu"].append(hbm_tb)
                columns["ridge_ai"].append(ridge)
                columns["captured_at_utc"].append(captured_at_utc)
                columns["published_at_utc"].append(published_at_utc)
    return pa.table(columns, schema=schema)


# ---------------------------------------------------------------------------
# Local + S3 writes
# ---------------------------------------------------------------------------


def write_local_parquet(table: Any, out_path: Path) -> Path:
    """Write a pyarrow Table to a local parquet file. Returns the path."""
    import pyarrow.parquet as pq  # type: ignore[import-not-found]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd")
    return out_path


def s3_key_for(table_name: str, campaign_id: str, captured_at: datetime) -> str:
    """Return the canonical S3 key for one (table, campaign) pair.

    Layout: ``perflake/perf-report/<table>/dt=<YYYY-MM-DD>/campaign=<id>/part-0.parquet``.
    The ``dt=`` partition uses the captured-at UTC date (not the publish
    date) so re-publishing the same campaign always lands at the same
    key. The downstream Spark MERGE INTO can then key on campaign_id.
    """
    dt = captured_at.strftime("%Y-%m-%d")
    return (
        f"{S3_PREFIX}/{table_name}/"
        f"dt={dt}/campaign={campaign_id}/part-0.parquet"
    )


def _make_s3_client(cfg: S3Config) -> Any:
    """Return a boto3 S3 client wired to S3.

    Uses virtual-host-style addressing (``<bucket>.<endpoint-host>``) because
    S3 rejects path-style requests with
    ``PathStyleRequestNotAllowed``. Matches the existing perflake conf
    convention ``FMA_S3_HOST_BUCKET=%(bucket)s.<endpoint-host>`` from the
    perflake ``.env``.

    Isolated as a one-line wrapper so tests can monkeypatch
    ``_make_s3_client`` to return a stub without pulling in moto/etc.
    """
    import boto3  # type: ignore[import-not-found]
    from botocore.config import Config  # type: ignore[import-not-found]

    return boto3.client(
        "s3",
        endpoint_url=cfg.endpoint,
        aws_access_key_id=cfg.access_key,
        aws_secret_access_key=cfg.secret_key,
        config=Config(s3={"addressing_style": "virtual"}),
    )


def upload_to_s3(
    local_path: Path,
    *,
    bucket: str,
    key: str,
    cfg: S3Config,
    if_exists: str,
    s3_client_factory: Callable[[S3Config], Any] | None = None,
) -> dict[str, Any]:
    """Upload one parquet file to S3.

    Returns a dict ``{key, size_bytes, skipped}``. Respects
    ``if_exists`` semantics (fail / skip / overwrite). Idempotency is
    keyed on the S3 object key (not ETag), matching the downstream loader's
    expectation that each campaign re-publish overwrites in place.
    """
    factory = s3_client_factory or _make_s3_client
    client = factory(cfg)

    exists = False
    try:
        client.head_object(Bucket=bucket, Key=key)
        exists = True
    except Exception:  # noqa: BLE001 - boto3 raises ClientError; we want any "not found" to fall through
        exists = False

    if exists:
        if if_exists == IF_EXISTS_FAIL:
            raise FileExistsError(
                f"s3://{bucket}/{key} already exists and --if-exists={IF_EXISTS_FAIL}; "
                f"pass --if-exists={IF_EXISTS_OVERWRITE} to clobber or "
                f"--if-exists={IF_EXISTS_SKIP} to no-op."
            )
        if if_exists == IF_EXISTS_SKIP:
            return {"key": key, "size_bytes": local_path.stat().st_size, "skipped": True}

    with local_path.open("rb") as body:
        client.put_object(Bucket=bucket, Key=key, Body=body)
    return {"key": key, "size_bytes": local_path.stat().st_size, "skipped": False}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def resolve_s3_config(
    *,
    endpoint: str | None,
    bucket: str | None,
    access_key_file: str | None,
    secret_key_file: str | None,
) -> S3Config:
    """Resolve the four S3 fields with CLI > env > default precedence.

    Access key + secret key MUST come from a file (CLI flag) or env var;
    we never accept the secret as a CLI arg directly to keep it out of
    ``argv`` / process lists / shell history.
    """
    resolved_endpoint = endpoint or os.environ.get(S3_ENV_ENDPOINT, S3_DEFAULT_ENDPOINT)
    resolved_bucket = bucket or os.environ.get(S3_ENV_BUCKET, S3_DEFAULT_BUCKET)
    if access_key_file:
        ak = Path(access_key_file).expanduser().read_text().strip()
    else:
        ak = os.environ.get(S3_ENV_ACCESS_KEY, "").strip()
    if secret_key_file:
        sk = Path(secret_key_file).expanduser().read_text().strip()
    else:
        sk = os.environ.get(S3_ENV_SECRET_KEY, "").strip()
    if not ak or not sk:
        raise SystemExit(
            "FATAL: S3 access/secret key missing. Set "
            f"{S3_ENV_ACCESS_KEY} + {S3_ENV_SECRET_KEY} in the "
            "environment (e.g. via `set -a; source "
            "./.env; set +a`), or pass "
            "--s3-access-key-file + --s3-secret-key-file."
        )
    return S3Config(
        endpoint=resolved_endpoint,
        bucket=resolved_bucket,
        access_key=ak,
        secret_key=sk,
    )


def publish(
    campaign_dir: Path,
    *,
    cfg: S3Config,
    dry_run: bool = False,
    if_exists: str = IF_EXISTS_FAIL,
    allow_incomplete: bool = False,
    allow_ungrounded: bool = False,
    strict: bool = False,
    publisher_operator: str | None = None,
    publisher_host: str | None = None,
    published_at_utc: datetime | None = None,
    s3_client_factory: Callable[[S3Config], Any] | None = None,
) -> PublishResult:
    """Read ``<campaign>/atlas.jsonl``, build atlas + campaign tables,
    write them locally to ``<campaign>/lake-cache/``, and (unless
    ``dry_run``) upload to S3.

    Publish policy (v1.33.0): EVERY measurement run publishes -- a
    latency-bound / ncu-only / zymtrace-proxy campaign is a first-class result,
    not a refusal. Incompleteness (no SoL evidence, 0 throughput-scatter points,
    an unsupported verdict claim) is recorded on the lake row (``sol_complete``,
    ``focus``, ``sol_rigor``, ``plot_ready_points``, ``omitted_pages``,
    ``verdict_tier``) + warned loudly, never hidden. The ONLY hard requirement
    is that ``report_render`` ran first (a campaign with no report_status.json
    cannot be published -- run render first). An unsupported ``verdict_tier=
    verdict`` claim is downgraded to ``draft`` (it still publishes).

    ``strict=True`` restores the old fail-loud behavior (raise
    ``CampaignIncompleteError`` on any completeness/verdict problem) for callers
    that want a gate. ``allow_incomplete`` is retained for back-compat (no-op
    now that landing is the default).

    Returns a ``PublishResult`` with per-table outcomes.
    """
    if if_exists not in IF_EXISTS_CHOICES:
        raise ValueError(
            f"if_exists must be one of {IF_EXISTS_CHOICES}; got {if_exists!r}"
        )

    campaign_dir = campaign_dir.resolve()
    if not campaign_dir.is_dir():
        raise SystemExit(f"FATAL: campaign dir does not exist: {campaign_dir}")
    campaign_id = campaign_dir.name
    captured_at = parse_campaign_utc(campaign_id)

    atlas_path = campaign_dir / "atlas.jsonl"
    if not atlas_path.is_file():
        raise SystemExit(
            f"FATAL: atlas.jsonl not found at {atlas_path}; run "
            "`perftunereport atlas_aggregate --campaign <slug>` first."
        )
    rows = read_jsonl(atlas_path)

    # Completeness signal. The ONLY hard requirement at the LIBRARY level is that
    # report_render ran (no report_status.json -> nothing to publish). This
    # function still defaults `strict=False` for programmatic back-compat, so
    # callers that want the permissive always-publish behavior get it. NOTE: the
    # `perftunereport publish_to_lake` CLI now passes `strict=True` BY DEFAULT (workspace
    # rigor policy, docs/METHODOLOGY.md) -- an operator must pass
    # `--no-strict` (or the deprecated `--allow-incomplete`) for a first-class
    # intentional-gap publish (latency-bound / ncu-only / proxy). Under strict the
    # problems below RAISE; otherwise they are RECORDED on the lake row + warned.
    render_status = read_render_status(campaign_dir)
    if not render_status.rendered:
        raise CampaignIncompleteError(
            f"cannot publish campaign {campaign_id!r}: no report_status.json "
            "(report_render was never run) -- run `perftunereport report_render "
            "--campaign <slug>` first."
        )
    problems: list[str] = []
    if not render_status.sol_complete:
        problems.append(
            "no Speed-of-Light evidence (sol_complete=false, sol_rigor=none; "
            f"omitted: {render_status.omitted_pages or 'n/a'}) -- capture "
            "zymtrace/ncu/DCGM SoL inputs and re-render for a tighter roofline"
        )
    if render_status.plot_ready_points == 0 and render_status.focus not in ("latency", "accuracy"):
        problems.append(
            "0 throughput-scatter points -- ensure bench output has 'Median "
            "TTFT (ms)' + 'Request throughput (req/s)' (or set focus=latency for "
            "a decode-latency/kernel run, or focus=accuracy for a training-accuracy/"
            "acceptance run), then re-import + re-render"
        )
    problems.extend(verdict_problems(read_verdict(campaign_dir)))
    problems.extend(methodology_problems(rows))
    problems.extend(krhpa_problems(campaign_dir, render_status))
    problems.extend(source_problems(campaign_dir))
    problems.extend(next_lever_problems(campaign_dir))
    problems.extend(close_reason_problems(campaign_dir))
    problems.extend(roofline_problems(render_status))
    problems.extend(per_arm_roofline_problems(render_status, campaign_dir))
    if problems:
        if strict:
            raise CampaignIncompleteError(
                f"--strict: refusing to publish incomplete campaign "
                f"{campaign_id!r}: " + "; ".join(problems)
            )
        print(
            f"WARNING: publishing campaign {campaign_id!r} with recorded gaps "
            f"(focus={render_status.focus}, sol_rigor={render_status.sol_rigor}): "
            + "; ".join(problems)
            + ". These are RECORDED on the campaign_v1 row (not hidden); pass "
            "--strict to refuse instead.",
            file=sys.stderr,
        )

    # DCGM byte-grounding gate (v1.33.0): byte-grounding is MANDATORY by
    # default for any campaign that HAS Speed-of-Light evidence. A campaign
    # that is sol_complete=True (L1 zymtrace roofline) but dcgm_grounded=False
    # (no DCGM workload-level byte/FLOP grounding, pages 6/6b) is FAIL-CLOSED:
    # publish refuses unless the operator explicitly passes --allow-ungrounded
    # (e.g. for a deliberately zymtrace-only L1 campaign where DCGM is
    # unavailable). This makes "every campaign is byte-grounded before publish"
    # the enforced default rather than an easily-skipped manual step. (A
    # sol_complete=False campaign has no SoL evidence at all, so DCGM grounding
    # is moot -- that case is governed by the completeness policy above.)
    # DCGM byte-grounding is RECORDED (dcgm_grounded column), not a refusal
    # (always-publish policy v1.33.0): a zymtrace-only or ncu-only campaign is a
    # first-class published result; its lack of DCGM byte-grounding is visible in
    # the lake row + warned, never hidden. `strict=True` (or the legacy
    # `allow_ungrounded=False` under strict) restores the fail-closed gate.
    if render_status.rendered and render_status.sol_complete and not render_status.dcgm_grounded:
        if strict and not allow_ungrounded:
            raise CampaignIncompleteError(
                f"--strict: refusing to publish campaign {campaign_id!r} with "
                "dcgm_grounded=FALSE (no DCGM workload-level byte/FLOP grounding, "
                "pages 6/6b). Run `perftunereport dcgm_correlate` per cell + re-render, "
                "or pass --allow-ungrounded."
            )
        print(
            f"WARNING: publishing campaign {campaign_id!r} with dcgm_grounded=FALSE "
            "-- it has SoL evidence (sol_rigor="
            f"{render_status.sol_rigor}) but NO DCGM workload-level byte/FLOP grounding "
            "(pages 6/6b). The lake row records dcgm_grounded=false. Run dcgm_correlate "
            "for a fully byte-grounded campaign.",
            file=sys.stderr,
        )

    published_at = published_at_utc or datetime.now(timezone.utc)
    atlas_table = build_atlas_table(rows, campaign_id)
    campaign_table = build_campaign_row(
        campaign_dir,
        rows,
        published_at_utc=published_at,
        publisher_operator=publisher_operator,
        publisher_host=publisher_host,
    )
    sol_table = build_sol_table(
        campaign_dir,
        campaign_id,
        rows,
        captured_at_utc=captured_at,
        published_at_utc=published_at,
        focus=render_status.focus,
        sol_rigor=render_status.sol_rigor,
    )
    # TPM SLA thresholds + node size come from the campaign config.yaml `tpm:`
    # block, so the lake's tpm_v1 carries the same peak AND sla points the PDF
    # page shows (no block -> peak-only at the default node size).
    from tools.perf_tune_report.tpm_summary import discover_tpm_config

    tpm_cfg = discover_tpm_config(campaign_dir)
    tpm_table = build_tpm_table(
        rows,
        campaign_id,
        captured_at_utc=captured_at,
        published_at_utc=published_at,
        gpus_per_node=tpm_cfg.gpus_per_node,
        ttft_sla_ms=tpm_cfg.ttft_sla_ms,
        tpot_sla_ms=tpm_cfg.tpot_sla_ms,
    )
    # cost_v1 (economics/TCO): $/1M tokens (from the cost: block) + tokens-per-watt
    # (from per-cell DCGM power). Cost/energy columns null when the inputs are absent.
    cost_table = build_cost_table(
        campaign_dir,
        rows,
        campaign_id,
        captured_at_utc=captured_at,
        published_at_utc=published_at,
        gpus_per_node=tpm_cfg.gpus_per_node,
        ttft_sla_ms=tpm_cfg.ttft_sla_ms,
        tpot_sla_ms=tpm_cfg.tpot_sla_ms,
        usd_per_gpu_hour=tpm_cfg.usd_per_gpu_hour,
    )
    # quality_v1 (training-accuracy / draft-acceptance, long-format). Empty for
    # pure serving campaigns (no cell declares extra["metric_kind"]).
    quality_table = build_quality_table(
        rows,
        campaign_id,
        captured_at_utc=captured_at,
        published_at_utc=published_at,
    )
    # champion_v1 (production-choice synthesis): one row per selected variant from
    # champion_select.json. Empty for campaigns that never ran champion_select.
    champion_table = build_champion_table(
        campaign_dir,
        campaign_id,
        captured_at_utc=captured_at,
        published_at_utc=published_at,
    )
    # roofline_v1 (per-(c, ISL) prefill/decode roofline points). Empty for
    # campaigns with no roofline sweep.
    roofline_table = build_roofline_table(
        campaign_dir,
        campaign_id,
        rows,
        captured_at_utc=captured_at,
        published_at_utc=published_at,
    )

    cache_dir = campaign_dir / "lake-cache"
    atlas_local = cache_dir / "atlas_v1.parquet"
    campaign_local = cache_dir / "campaign_v1.parquet"
    sol_local = cache_dir / "sol_v1.parquet"
    tpm_local = cache_dir / "tpm_v1.parquet"
    cost_local = cache_dir / "cost_v1.parquet"
    quality_local = cache_dir / "quality_v1.parquet"
    champion_local = cache_dir / "champion_v1.parquet"
    roofline_local = cache_dir / "roofline_v1.parquet"
    write_local_parquet(atlas_table, atlas_local)
    write_local_parquet(campaign_table, campaign_local)
    write_local_parquet(sol_table, sol_local)
    write_local_parquet(tpm_table, tpm_local)
    write_local_parquet(cost_table, cost_local)
    write_local_parquet(quality_table, quality_local)
    write_local_parquet(champion_table, champion_local)
    write_local_parquet(roofline_table, roofline_local)

    atlas_key = s3_key_for(ATLAS_TABLE_NAME, campaign_id, captured_at)
    campaign_key = s3_key_for(CAMPAIGN_TABLE_NAME, campaign_id, captured_at)
    sol_key = s3_key_for(SOL_TABLE_NAME, campaign_id, captured_at)
    tpm_key = s3_key_for(TPM_TABLE_NAME, campaign_id, captured_at)
    cost_key = s3_key_for(COST_TABLE_NAME, campaign_id, captured_at)
    quality_key = s3_key_for(QUALITY_TABLE_NAME, campaign_id, captured_at)
    champion_key = s3_key_for(CHAMPION_TABLE_NAME, campaign_id, captured_at)
    roofline_key = s3_key_for(ROOFLINE_TABLE_NAME, campaign_id, captured_at)

    if dry_run:
        atlas_outcome = {"key": atlas_key, "size_bytes": atlas_local.stat().st_size, "skipped": True}
        campaign_outcome = {
            "key": campaign_key,
            "size_bytes": campaign_local.stat().st_size,
            "skipped": True,
        }
        sol_outcome = {"key": sol_key, "size_bytes": sol_local.stat().st_size, "skipped": True}
        tpm_outcome = {"key": tpm_key, "size_bytes": tpm_local.stat().st_size, "skipped": True}
        cost_outcome = {"key": cost_key, "size_bytes": cost_local.stat().st_size, "skipped": True}
        quality_outcome = {"key": quality_key, "size_bytes": quality_local.stat().st_size, "skipped": True}
        champion_outcome = {"key": champion_key, "size_bytes": champion_local.stat().st_size, "skipped": True}
        roofline_outcome = {"key": roofline_key, "size_bytes": roofline_local.stat().st_size, "skipped": True}
    else:
        atlas_outcome = upload_to_s3(
            atlas_local,
            bucket=cfg.bucket,
            key=atlas_key,
            cfg=cfg,
            if_exists=if_exists,
            s3_client_factory=s3_client_factory,
        )
        campaign_outcome = upload_to_s3(
            campaign_local,
            bucket=cfg.bucket,
            key=campaign_key,
            cfg=cfg,
            if_exists=if_exists,
            s3_client_factory=s3_client_factory,
        )
        sol_outcome = upload_to_s3(
            sol_local,
            bucket=cfg.bucket,
            key=sol_key,
            cfg=cfg,
            if_exists=if_exists,
            s3_client_factory=s3_client_factory,
        )
        tpm_outcome = upload_to_s3(
            tpm_local,
            bucket=cfg.bucket,
            key=tpm_key,
            cfg=cfg,
            if_exists=if_exists,
            s3_client_factory=s3_client_factory,
        )
        cost_outcome = upload_to_s3(
            cost_local,
            bucket=cfg.bucket,
            key=cost_key,
            cfg=cfg,
            if_exists=if_exists,
            s3_client_factory=s3_client_factory,
        )
        quality_outcome = upload_to_s3(
            quality_local,
            bucket=cfg.bucket,
            key=quality_key,
            cfg=cfg,
            if_exists=if_exists,
            s3_client_factory=s3_client_factory,
        )
        champion_outcome = upload_to_s3(
            champion_local,
            bucket=cfg.bucket,
            key=champion_key,
            cfg=cfg,
            if_exists=if_exists,
            s3_client_factory=s3_client_factory,
        )
        roofline_outcome = upload_to_s3(
            roofline_local,
            bucket=cfg.bucket,
            key=roofline_key,
            cfg=cfg,
            if_exists=if_exists,
            s3_client_factory=s3_client_factory,
        )

    return PublishResult(
        campaign_dir=campaign_dir,
        campaign_id=campaign_id,
        captured_at_utc=captured_at,
        atlas=ObjectWriteResult(
            table=ATLAS_TABLE_NAME,
            local_path=atlas_local,
            s3_key=atlas_outcome["key"],
            size_bytes=atlas_outcome["size_bytes"],
            sha256=_sha256_of(atlas_local),
            row_count=len(rows),
            skipped=bool(atlas_outcome.get("skipped", False)) and not dry_run,
        ),
        campaign=ObjectWriteResult(
            table=CAMPAIGN_TABLE_NAME,
            local_path=campaign_local,
            s3_key=campaign_outcome["key"],
            size_bytes=campaign_outcome["size_bytes"],
            sha256=_sha256_of(campaign_local),
            row_count=1,
            skipped=bool(campaign_outcome.get("skipped", False)) and not dry_run,
        ),
        sol=ObjectWriteResult(
            table=SOL_TABLE_NAME,
            local_path=sol_local,
            s3_key=sol_outcome["key"],
            size_bytes=sol_outcome["size_bytes"],
            sha256=_sha256_of(sol_local),
            row_count=sol_table.num_rows,
            skipped=bool(sol_outcome.get("skipped", False)) and not dry_run,
        ),
        tpm=ObjectWriteResult(
            table=TPM_TABLE_NAME,
            local_path=tpm_local,
            s3_key=tpm_outcome["key"],
            size_bytes=tpm_outcome["size_bytes"],
            sha256=_sha256_of(tpm_local),
            row_count=tpm_table.num_rows,
            skipped=bool(tpm_outcome.get("skipped", False)) and not dry_run,
        ),
        cost=ObjectWriteResult(
            table=COST_TABLE_NAME,
            local_path=cost_local,
            s3_key=cost_outcome["key"],
            size_bytes=cost_outcome["size_bytes"],
            sha256=_sha256_of(cost_local),
            row_count=cost_table.num_rows,
            skipped=bool(cost_outcome.get("skipped", False)) and not dry_run,
        ),
        quality=ObjectWriteResult(
            table=QUALITY_TABLE_NAME,
            local_path=quality_local,
            s3_key=quality_outcome["key"],
            size_bytes=quality_outcome["size_bytes"],
            sha256=_sha256_of(quality_local),
            row_count=quality_table.num_rows,
            skipped=bool(quality_outcome.get("skipped", False)) and not dry_run,
        ),
        champion=ObjectWriteResult(
            table=CHAMPION_TABLE_NAME,
            local_path=champion_local,
            s3_key=champion_outcome["key"],
            size_bytes=champion_outcome["size_bytes"],
            sha256=_sha256_of(champion_local),
            row_count=champion_table.num_rows,
            skipped=bool(champion_outcome.get("skipped", False)) and not dry_run,
        ),
        roofline=ObjectWriteResult(
            table=ROOFLINE_TABLE_NAME,
            local_path=roofline_local,
            s3_key=roofline_outcome["key"],
            size_bytes=roofline_outcome["size_bytes"],
            sha256=_sha256_of(roofline_local),
            row_count=roofline_table.num_rows,
            skipped=bool(roofline_outcome.get("skipped", False)) and not dry_run,
        ),
        dry_run=dry_run,
        bucket=cfg.bucket,
        endpoint=cfg.endpoint,
        published_at_utc=published_at,
    )
