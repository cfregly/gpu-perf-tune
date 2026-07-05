"""Compose page 1 (scatter grid), page 2 (heatmap tables), optional
page 3 (GPU kernel breakdown), and optional page 4 (Speed-of-Light
roofline) into one PDF.

Uses matplotlib's PdfPages backend, mirroring the file the source GLM-5.1
PDF carries in its Creator metadata (``Matplotlib v3.10.9, pdf backend``).

Page 3 is conditional on the campaign's cells/ tree containing at least
one valid ``kernels.json`` (emitted by
``tools.perf_tune_report.importers.zymtrace_kernels`` when the bundle's
``capture_sources.json`` declares zymtrace). The decision logic:

- No ``kernels.json`` files anywhere under ``<campaign>/cells/*/``: the
  page is NOT silently dropped. The omission is recorded in the returned
  ``RenderStatus``, listed on the loud "Report completeness" page (with a
  why + how-to-fix), and written to ``report_status.json``.
- Any ``kernels.json`` file exists but cannot be parsed as JSON or is
  missing required fields: raise ``KernelsJsonMalformed`` and abort the
  whole render. The silent-degradation pattern Phase 5 SOURCE.md fell
  into is not allowed back in.
- At least one valid ``kernels.json`` is found: render page 3 using all
  of them (one per variant in cell-id order).

Page 4 (Speed-of-Light roofline) is conditional on page 3 being drawn AND
on a ``sol-ceilings.yaml`` being findable. ``discover_sol_inputs`` returns
the loaded ceilings dict + the hardware key for the campaign's atlas, or
``None`` when absent -- in which case the omission is recorded loudly (see
above) and ``RenderStatus.sol_complete`` is set ``False``. Malformed YAML
raises ``SoLCeilingsMalformed`` (mirrors the ``KernelsJsonMalformed``
no-silent-degradation rule).
"""

from __future__ import annotations

import json
import os
import textwrap
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from tools.perf_tune_report.coverage import summarize
from tools.perf_tune_report.renderer import (
    champion_select,
    dcgm_category_attribution,
    dcgm_sol,
    heatmap_tables,
    kernel_breakdown,
    prefill_decode_roofline,
    scatter_grid,
    sol_roofline,
    sol_roofline_scatter,
    tpm_table,
)
from tools.perf_tune_report.renderer.render_status import OMISSION_REASONS, OmissionReason, RenderStatus
from tools.perf_tune_report.schema import AtlasCell, read_jsonl


_NCU_REQUIRED_FIELDS = (
    "schema_version",
    "captured_sources",
    "hw_key",
    "kernels",
)

_DCGM_REQUIRED_FIELDS = (
    "schema_version",
    "captured_sources",
    "hw_key",
    "resources",
    "queries",
)


class NcuKernelsJsonMalformed(Exception):
    """Raised when a ``ncu_kernels.json`` file is present but unparseable.

    Mirrors ``KernelsJsonMalformed``: fires at render time when the
    renderer is asked to draw page 5 against bad payload data.
    """

    def __init__(self, path: Path, reason: str):
        super().__init__(f"ncu_kernels.json malformed: {path} ({reason})")
        self.path = path
        self.reason = reason


class DcgmCorrelationJsonMalformed(Exception):
    """Raised when a ``dcgm_correlation.json`` file is present but unparseable."""

    def __init__(self, path: Path, reason: str):
        super().__init__(f"dcgm_correlation.json malformed: {path} ({reason})")
        self.path = path
        self.reason = reason


def discover_ncu_payloads(
    campaign_dir: Path,
) -> "OrderedDict[str, dict[str, Any]]":
    """Walk ``<campaign_dir>/cells/*/ncu_kernels.json``, parse + validate.

    Same pattern as ``discover_kernels_payloads`` for the zymtrace
    kernels.json. Empty dict means no campaign cell has ncu coverage;
    the renderer skips page 5 silently.
    """
    out: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    cells_dir = campaign_dir / "cells"
    if not cells_dir.is_dir():
        return out
    for cell_dir in sorted(p for p in cells_dir.iterdir() if p.is_dir()):
        path = cell_dir / "ncu_kernels.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise NcuKernelsJsonMalformed(path, f"not valid JSON: {e}") from e
        missing = [f for f in _NCU_REQUIRED_FIELDS if f not in payload]
        if missing:
            raise NcuKernelsJsonMalformed(
                path, f"missing required fields: {missing}"
            )
        out[cell_dir.name] = payload
    return out


def discover_dcgm_payloads(
    campaign_dir: Path,
) -> "OrderedDict[str, dict[str, Any]]":
    """Walk ``<campaign_dir>/cells/*/dcgm_correlation.json``, parse + validate.

    Empty dict means no campaign cell has DCGM coverage; the renderer
    skips page 6 silently.
    """
    out: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    cells_dir = campaign_dir / "cells"
    if not cells_dir.is_dir():
        return out
    for cell_dir in sorted(p for p in cells_dir.iterdir() if p.is_dir()):
        path = cell_dir / "dcgm_correlation.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise DcgmCorrelationJsonMalformed(path, f"not valid JSON: {e}") from e
        missing = [f for f in _DCGM_REQUIRED_FIELDS if f not in payload]
        if missing:
            raise DcgmCorrelationJsonMalformed(
                path, f"missing required fields: {missing}"
            )
        out[cell_dir.name] = payload
    return out


def discover_roofline_payloads(
    campaign_dir: Path,
) -> "OrderedDict[str, dict[str, Any]]":
    """Walk ``<campaign_dir>/cells/*/roofline_sweep.json`` (written by the
    roofline_sweep importer). Empty dict -> the prefill/decode roofline page
    is skipped silently. Malformed JSON raises (no silent degradation)."""
    out: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    cells_dir = campaign_dir / "cells"
    if not cells_dir.is_dir():
        return out
    for cell_dir in sorted(p for p in cells_dir.iterdir() if p.is_dir()):
        path = cell_dir / "roofline_sweep.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise prefill_decode_roofline.RooflineSweepJsonMalformed(
                cell_dir.name, f"not valid JSON: {e}"
            ) from e
        out[cell_dir.name] = payload
    return out


def discover_champion_payload(campaign_dir: Path) -> dict[str, Any] | None:
    """Load ``<campaign_dir>/champion_select.json`` (written by the
    ``champion_select`` verb). None -> the champion-selection page is omitted
    loudly. Malformed JSON raises (no silent degradation)."""
    path = campaign_dir / "champion_select.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise champion_select.ChampionSelectJsonMalformed(f"{path}: not valid JSON: {e}") from e


_KERNELS_REQUIRED_FIELDS = (
    "schema_version",
    "captured_sources",
    "top_kernels",
    "per_gpu",
    "per_category",
    "top_python_during_cuda",
)


class KernelsJsonMalformed(Exception):
    """Raised when a ``kernels.json`` file is present but unparseable or
    missing one of the required schema fields.

    Distinct from ``ZymtraceTSVMalformed`` (importer-side) -- this fires
    at render time when the renderer is asked to draw page 3 against bad
    payload data. Either way, the silent-skip path is the wrong answer.
    """

    def __init__(self, path: Path, reason: str):
        super().__init__(f"kernels.json malformed: {path} ({reason})")
        self.path = path
        self.reason = reason


def discover_kernels_payloads(
    campaign_dir: Path,
) -> "OrderedDict[str, dict[str, Any]]":
    """Walk ``<campaign_dir>/cells/*/kernels.json``, parse + validate each.

    Returns an ``OrderedDict`` keyed by cell_id (= cell directory name),
    ordered by cell_id. Empty dict means no campaign cell has zymtrace
    coverage; the renderer skips page 3 silently.

    Raises ``KernelsJsonMalformed`` if any ``kernels.json`` file exists
    but is unparseable or missing required fields.
    """
    out: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    cells_dir = campaign_dir / "cells"
    if not cells_dir.is_dir():
        return out
    for cell_dir in sorted(p for p in cells_dir.iterdir() if p.is_dir()):
        kernels_path = cell_dir / "kernels.json"
        if not kernels_path.is_file():
            continue
        try:
            payload = json.loads(kernels_path.read_text())
        except json.JSONDecodeError as e:
            raise KernelsJsonMalformed(kernels_path, f"not valid JSON: {e}") from e
        missing = [f for f in _KERNELS_REQUIRED_FIELDS if f not in payload]
        if missing:
            raise KernelsJsonMalformed(
                kernels_path, f"missing required fields: {missing}"
            )
        out[cell_dir.name] = payload
    return out


def _is_roofline_shard(name: str) -> bool:
    """A roofline SHARD cell (``<arm>-decode`` / ``<arm>-prefill``, written by
    ``import_roofline_sweep``) is not a standalone arm -- it collapses to its
    base arm for per-arm coverage."""
    return name.endswith("-decode") or name.endswith("-prefill")


def compute_per_arm_coverage(
    campaign_dir: Path,
    rows: "Sequence[AtlasCell]",
    cell_kernels: "dict[str, Any]",
    cell_dcgm: "dict[str, Any]",
    cell_roofline: "dict[str, Any]",
    *,
    page4: bool,
    page6: bool,
    page7: bool,
) -> "tuple[int, int, list[str]]":
    """Per-arm Speed-of-Light coverage rollup -- the source of truth the
    publish gate, teardown hook, and sol-coverage audit all key on.

    ``sol_complete`` is CAMPAIGN-level ("any SoL page rendered"), so a multi-arm
    campaign whose baseline carries a roofline but whose variants do NOT reads
    complete. This rollup makes the per-variant gap explicit.

    An ``arm`` == an atlas cell (the distinct serving config under comparison);
    its ``-decode`` / ``-prefill`` roofline shards collapse to the base arm. An
    arm is COVERED when it carries a SoL input whose page actually rendered:
    page-4 (``kernels.json`` / L1 zymtrace), page-6 (``dcgm_correlation.json`` /
    L3 DCGM), or page-7 (``roofline_sweep.json`` on the base or a shard / L3
    prefill-decode roofline). Returns ``(arms_total, arms_with_roofline,
    sorted(arms_uncovered))``.

    The definition is kept identical to ``perf-tune-glm51/sol-coverage-audit.py``
    so the audit can defer to these renderer-recorded fields without drifting.
    """
    cells_dir = campaign_dir / "cells"
    cell_dirs = (
        {p.name for p in cells_dir.iterdir() if p.is_dir()}
        if cells_dir.is_dir()
        else set()
    )
    atlas_arm_dirs: dict[str, str] = {}
    for r in rows:
        cell_id = getattr(r, "cell_id", None)
        if not cell_id or _is_roofline_shard(cell_id):
            continue
        arm = (getattr(r, "extra", {}) or {}).get("arm")
        if isinstance(arm, str) and arm and (
            arm in cell_dirs or f"{arm}-decode" in cell_dirs or f"{arm}-prefill" in cell_dirs
        ):
            atlas_arm_dirs.setdefault(cell_id, arm)
        else:
            atlas_arm_dirs.setdefault(cell_id, cell_id)
    shard_bases = {n.rsplit("-", 1)[0] for n in cell_dirs if _is_roofline_shard(n)}
    if atlas_arm_dirs:
        arms = sorted(atlas_arm_dirs)
    else:
        arms = sorted({a for a in cell_dirs if not _is_roofline_shard(a)} | shard_bases)
    covered_n = 0
    uncovered: list[str] = []
    for arm in arms:
        artifact_arm = atlas_arm_dirs.get(arm, arm)
        has_roofline = (
            artifact_arm in cell_roofline
            or f"{artifact_arm}-decode" in cell_roofline
            or f"{artifact_arm}-prefill" in cell_roofline
        )
        is_covered = (
            (artifact_arm in cell_kernels and page4)
            or (artifact_arm in cell_dcgm and page6)
            or (has_roofline and page7)
        )
        if is_covered:
            covered_n += 1
        else:
            uncovered.append(arm)
    return len(arms), covered_n, uncovered


# Default search path for the workspace-shared ceilings YAML. Resolved
# relative to the campaign's ``atlas.jsonl`` parent by walking up the
# directory tree looking for ``configs/sol-ceilings.yaml``.
# Operators can override via the ``SOL_CEILINGS_YAML`` env var.
_SOL_CEILINGS_RELPATH = Path("perf-tune-report") / "configs" / "sol-ceilings.yaml"
_SOURCE_REGISTRY_RELPATH = Path("perf-tune-report") / "configs" / "source-registry.yaml"


def discover_provenance(campaign_dir: Path) -> "dict[str, Any] | None":
    """Read the campaign's ``provenance.json`` (the experiment_provenance_v1
    block copied in by ``campaign_init``). None when absent (older campaigns)."""
    p = campaign_dir / "provenance.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def discover_source_registry(campaign_dir: Path) -> "dict[str, Any] | None":
    """Walk up for ``perf-tune-report/configs/source-registry.yaml`` (branch ->
    purpose). None when not found (the source page then omits purposes)."""
    cur = campaign_dir.resolve()
    for parent in [cur, *cur.parents]:
        cand = parent / _SOURCE_REGISTRY_RELPATH
        if cand.is_file():
            try:
                import yaml
                return yaml.safe_load(cand.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                return None
    return None


def discover_sol_inputs(
    campaign_dir: Path,
    rows: Sequence[AtlasCell],
    fallback_hw_key: str | None = None,
) -> tuple[dict[str, Any], str] | None:
    """Locate + load ``sol-ceilings.yaml`` and pick the hardware key.

    Returns ``(ceilings_dict, hardware_key)`` when both are available, or
    ``None`` to skip page 4 silently. Returns ``None`` when:

    - the campaign's atlas carries no rows with a recognised ``hardware``
      string (we don't know which ceiling column to use) AND no usable
      ``fallback_hw_key`` was supplied, or
    - no ``sol-ceilings.yaml`` is findable on the search path, or
    - the env var ``SOL_CEILINGS_YAML=disable`` is set (escape hatch).

    ``fallback_hw_key`` lets a caller supply a hardware key from another
    source (e.g. the ``ncu_kernels.json`` payload's own ``hw_key``) for the
    case where the bench atlas is empty -- a pure-ncu roofline campaign with
    no serve sweep. It is used ONLY when the atlas-derived key is absent /
    unknown, so existing bench-driven callers (no fallback) are unchanged.

    Raises ``sol_roofline.SoLCeilingsMalformed`` when a YAML is found but
    is unparseable / missing required keys (no silent degradation).
    """
    env_override = os.environ.get("SOL_CEILINGS_YAML", "").strip()
    if env_override == "disable":
        return None

    if env_override:
        yaml_path = Path(env_override).expanduser().resolve()
        if not yaml_path.is_file():
            return None
    else:
        yaml_path = None
        cur = campaign_dir.resolve()
        # Try the canonical bundle name first, then a name-agnostic
        # ``<bundle>/configs/sol-ceilings.yaml`` fallback so a future submodule
        # rename (perf-report -> perf-tune-report -> ...) needs no code edit.
        # The campaign's own bundle root (campaigns/'s parent) is reached before
        # higher ancestors, so the fallback matches the bundle's own configs/.
        _relcands = (_SOL_CEILINGS_RELPATH, Path("configs") / "sol-ceilings.yaml")
        for parent in [cur, *cur.parents]:
            for _rel in _relcands:
                candidate = parent / _rel
                if candidate.is_file():
                    yaml_path = candidate
                    break
            if yaml_path is not None:
                break
        if yaml_path is None:
            return None

    ceilings = sol_roofline.load_ceilings(yaml_path)
    hw_key = sol_roofline.hardware_key_for_atlas(rows)
    # Fall back to a caller-supplied key (e.g. the ncu payload's hw_key) when
    # the atlas can't supply one -- the pure-ncu, zero-bench-row case. (`None
    # in ceilings` is safely False, so bench-driven callers are unaffected.)
    if (hw_key is None or hw_key not in ceilings) and fallback_hw_key in ceilings:
        hw_key = fallback_hw_key
    if hw_key is None or hw_key not in ceilings:
        return None
    return ceilings, hw_key


def _render_empty_banner(fig: Any, reason: OmissionReason) -> None:
    """Overlay a loud why+how banner on a page whose chart has no data.

    Used when page 1's scatter has 0 plot-ready points: the empty grid
    still draws, but this banner makes the absence impossible to miss and
    tells the reviewer exactly how to populate it.
    """
    why = "\n".join(textwrap.wrap(reason.why, width=92))
    how = "\n".join(textwrap.wrap("How to fix: " + reason.how_to_fix, width=92))
    fig.text(
        0.5,
        0.5,
        f"NO PLOT-READY DATA\n\n{why}\n\n{how}",
        ha="center",
        va="center",
        fontsize=11,
        color="#b30000",
        wrap=True,
        bbox={"boxstyle": "round", "facecolor": "#fff3f3", "edgecolor": "#b30000", "linewidth": 1.5},
    )


def _render_completeness_page(fig: Any, status: RenderStatus) -> None:
    """Draw the loud "Report completeness" page enumerating every omission
    AND every partial (rendered-but-limited) page.

    Rendered whenever any conditional page was omitted OR rendered partial, so
    a reviewer never reads a short or limited report as complete. Each entry
    lists the page, why, and the exact next step to populate / complete it.
    """
    fig.clf()
    lines: list[str] = []
    header = "REPORT COMPLETENESS -- omitted + partial pages & how to fix"
    sol_state = "COMPLETE" if status.sol_complete else "INCOMPLETE"
    lines.append(
        f"Speed-of-Light rooflines: {sol_state}    |    "
        f"plot-ready points: {status.plot_ready_points}    |    "
        f"full-but-unplottable cells: {status.non_plot_ready_full_cells}"
    )
    lines.append("")
    # Per-arm roofline coverage (v1.68.0): a multi-arm campaign is not complete
    # until baseline AND every variant carries a roofline. Surface any gap loud.
    if status.arms_total:
        per_arm_state = "COMPLETE" if status.sol_per_arm_complete else "INCOMPLETE"
        lines.append(
            f"Per-arm rooflines: {per_arm_state}    |    "
            f"arms with a roofline: {status.arms_with_roofline}/{status.arms_total}"
        )
        if status.arms_uncovered:
            lines.append("  ARMS MISSING A ROOFLINE (baseline + each variant must carry one):")
            for wrapped in textwrap.wrap("    " + ", ".join(status.arms_uncovered), width=100):
                lines.append(wrapped)
            lines.append(
                "  How to fix: run *-deploy/profiling/roofline-sweep.sh + perftunereport "
                "import_roofline_sweep for each arm, then re-render -- or declare each "
                "genuinely un-capturable arm in config.yaml roofline_gap_arms: "
                "{<arm>: <reason>} (B200 / MTP-engine-blocked / overlay-gone)."
            )
        lines.append("")
    for omission in status.omitted_pages:
        lines.append(f"OMITTED: {omission['page']}")
        for wrapped in textwrap.wrap("  Why: " + omission["why"], width=100):
            lines.append(wrapped)
        for wrapped in textwrap.wrap("  How to fix: " + omission["how_to_fix"], width=100):
            lines.append(wrapped)
        lines.append("")

    if status.partial_pages:
        lines.append("PARTIAL (rendered but limited -- do NOT read as a full measurement):")
        lines.append("")
        for partial in status.partial_pages:
            lines.append(f"PARTIAL: {partial['page']}")
            for wrapped in textwrap.wrap("  Why: " + partial["why"], width=100):
                lines.append(wrapped)
            for wrapped in textwrap.wrap("  How to fix: " + partial["how_to_fix"], width=100):
                lines.append(wrapped)
            lines.append("")

    fig.text(0.5, 0.95, header, ha="center", va="top", fontsize=13, color="#b30000", weight="bold")
    fig.text(
        0.04,
        0.88,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        color="#333333",
    )


def _render_source_page(fig: Any, links: "list[dict[str, str]]", ident: dict[str, Any]) -> None:
    """Draw the "Source under test" page: the EXACT code each roofline ran
    (vLLM/SGLang repo + branch + commit + delivery + infr patch), with a GitHub
    URL per entry. A roofline with no traceable source is not reproducible; this
    page makes the link first-class in the PDF (from the provenance block +
    source-registry.yaml)."""
    fig.clf()
    lines: list[str] = []
    title = ident.get("title") or ""
    if title:
        for wrapped in textwrap.wrap(f"experiment: {title}", width=100):
            lines.append(wrapped)
        lines.append("")
    for link in links:
        deliv = link.get("delivery") or "image"
        lines.append(f"{link['repo'] or '(repo unset)'}   [delivery: {deliv}]")
        if link.get("branch") or link.get("commit"):
            lines.append(f"  branch: {link.get('branch') or '(default)'}    "
                         f"commit: {link.get('commit') or '(unpinned)'}")
        if link.get("url"):
            lines.append(f"  {link['url']}")
        if link.get("purpose"):
            for wrapped in textwrap.wrap("  purpose: " + link["purpose"], width=100):
                lines.append(wrapped)
        if link.get("image"):
            lines.append(f"  image: {link['image']}")
        if link.get("patch"):
            for wrapped in textwrap.wrap("  patch: " + link["patch"], width=100):
                lines.append(wrapped)
        lines.append("")
    if not links:
        lines.append("(no experiment_provenance_v1 block -- run capture-provenance.sh on the bundle)")
    fig.text(0.5, 0.95, "SOURCE UNDER TEST -- exact code each roofline ran",
             ha="center", va="top", fontsize=13, color="#1a5276", weight="bold")
    fig.text(0.04, 0.88, "\n".join(lines), ha="left", va="top", fontsize=9,
             family="monospace", color="#333333")


def build_pdf_provenance(
    campaign_id: str,
    rows: Sequence[AtlasCell],
    rendered_at: datetime,
) -> dict[str, Any]:
    """Build UTC provenance for the rendered PDF: infodict + a page footer.

    The OS file mtime of the written PDF is local wall-clock time and is
    NOT authoritative -- e.g. at ~22:41 local (UTC-7) it reads a full
    calendar day behind the ``YYYYMMDDTHHMMSSZ`` campaign run-id. This
    embeds the canonical UTC instant (``rendered_at``) + the run-id + the
    bench-capture window directly into the artifact (PDF ``CreationDate``
    metadata and a page-1 footer) so consumers never have to trust the
    local mtime. All emitted timestamps are UTC.
    """
    rendered_utc = rendered_at.astimezone(timezone.utc)
    rendered_iso = rendered_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    captured = sorted({r.captured_at for r in rows if r.captured_at})
    if not captured:
        bench_window = "unknown"
    elif len(captured) == 1:
        bench_window = captured[0]
    else:
        bench_window = f"{captured[0]} .. {captured[-1]}"
    footer = (
        f"campaign={campaign_id}  |  rendered={rendered_iso}  |  "
        f"bench-window={bench_window}  |  "
        "all times UTC (file mtime is local, not authoritative)"
    )
    infodict = {
        "Subject": f"perf-report campaign {campaign_id}",
        "Keywords": (
            f"campaign={campaign_id} rendered_utc={rendered_iso} "
            f"bench_window={bench_window}"
        ),
        "CreationDate": rendered_utc,
        "ModDate": rendered_utc,
    }
    return {
        "infodict": infodict,
        "footer": footer,
        "rendered_iso": rendered_iso,
        "bench_window": bench_window,
    }


def _discover_focus(campaign_dir: Path) -> str:
    """Read the campaign's intent focus ("latency" | "throughput" | "mixed")
    from ``<campaign_dir>/config.yaml`` (``focus:`` key). Defaults "mixed".

    Line-scan parse (no yaml dependency) so this works for any config shape.
    The focus is RECORDED on every published campaign so latency-focused runs
    (c=1 decode, kernel probes) are first-class results, not "drafts".
    """
    valid = {"latency", "throughput", "mixed"}
    cfg = campaign_dir / "config.yaml"
    if cfg.is_file():
        for raw in cfg.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("focus:"):
                val = line.split(":", 1)[1].strip().strip("\"'").lower()
                if val in valid:
                    return val
    return "mixed"


def render_report(
    atlas_jsonl_path: Path,
    out_pdf: Path,
    *,
    title: str = "glm5p1 benchmark report",
    variants_line: str | None = None,
    data_source_line: str | None = None,
    page_size: tuple[float, float] = (scatter_grid.PAGE_WIDTH_IN, scatter_grid.PAGE_HEIGHT_IN),
) -> RenderStatus:
    """Render the PDF from an ``atlas.jsonl`` file; return a ``RenderStatus``.

    Pages 1 + 2 always render. Pages 3-6b (GPU kernel breakdown, SoL
    roofline, ncu scatter, DCGM SoL, DCGM cross-attribution) are conditional
    on their input data being present under ``<campaign_dir>/cells/*/``.

    No conditional page is ever silently dropped: when its input is missing
    the omission is recorded in the returned ``RenderStatus``, surfaced on a
    loud "Report completeness" page (why + how-to-fix per omission), and
    written to ``<campaign_dir>/report_status.json``. A page-1 scatter with
    0 plot-ready points gets a loud why+how banner. Malformed input data
    still raises (``KernelsJsonMalformed`` etc.) -- the no-silent-degradation
    rule applies to both absence and corruption.
    """
    rows: Sequence[AtlasCell] = read_jsonl(atlas_jsonl_path)
    coverage = summarize(rows)
    status = RenderStatus(
        plot_ready_points=coverage.plot_ready_points,
        non_plot_ready_full_cells=coverage.non_plot_ready_full_cells,
    )

    # Discover + validate kernel payloads BEFORE drawing anything, so a
    # malformed kernels.json fails the whole render at the start (the
    # opposite of a silent-skip degradation).
    campaign_dir = atlas_jsonl_path.parent
    cell_kernels = discover_kernels_payloads(campaign_dir)
    cell_ncu = discover_ncu_payloads(campaign_dir)
    cell_dcgm = discover_dcgm_payloads(campaign_dir)
    cell_roofline = discover_roofline_payloads(campaign_dir)
    champion_payload = discover_champion_payload(campaign_dir)

    # Run intent focus (recorded on every published campaign).
    status.focus = _discover_focus(campaign_dir)
    # Track which Speed-of-Light evidence LEVELS render (L1 zymtrace page4 /
    # L4 ncu page5 / L3 DCGM page6). sol_complete is now "any SoL evidence
    # rendered" (not just page4), and sol_rigor records the highest level so
    # the proxy-vs-tight distinction is a field, not a publish blocker.
    sol_l1 = sol_l3 = sol_l4 = False

    # UTC provenance stamp (embedded in PDF metadata + a page-1 footer) so the
    # artifact self-documents its canonical UTC instant + run-id, and nobody
    # has to read the local-wall-clock OS file mtime.
    provenance = build_pdf_provenance(campaign_dir.name, rows, datetime.now(timezone.utc))

    # Discover SoL inputs only when page 3 is in scope (page 4 layers on
    # top of page 3's zymtrace per_category data).
    sol_inputs = discover_sol_inputs(campaign_dir, rows) if cell_kernels else None
    # Page 5 wants SoL ceilings too (for the roofline lines), but is
    # independent of page 3's zymtrace data; it can render even when
    # only ncu_kernels.json (no kernels.json) is present. For a pure-ncu
    # campaign the bench atlas is empty (0 rows -> no atlas-derived hw key),
    # so fall back to the ncu payload's own hw_key to resolve the ceilings.
    ncu_fallback_hw = next(iter(cell_ncu.values())).get("hw_key") if cell_ncu else None
    ncu_sol_inputs = (
        discover_sol_inputs(campaign_dir, rows, fallback_hw_key=ncu_fallback_hw)
        if (cell_ncu and sol_inputs is None)
        else sol_inputs
    )

    # Lazy matplotlib import so callers that only need the schema don't pay
    # the ~150ms import cost.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if out_pdf.is_dir():
        raise ValueError(f"--out must be a file path, not a directory: {out_pdf}")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_pdf) as pdf:
        # Embed UTC provenance in the PDF metadata. CreationDate is set to a
        # tz-aware UTC datetime (not matplotlib's default local-clock value),
        # so the document's own date is unambiguous regardless of the writer's
        # timezone or the file's local mtime.
        info = pdf.infodict()
        info["Title"] = title
        info["Subject"] = provenance["infodict"]["Subject"]
        info["Keywords"] = provenance["infodict"]["Keywords"]
        info["CreationDate"] = provenance["infodict"]["CreationDate"]
        info["ModDate"] = provenance["infodict"]["ModDate"]

        # Page 1: scatter grid. A 0-plot-ready atlas still draws the grid,
        # but gets a loud why+how banner so the empty chart is never silent.
        fig1 = plt.figure(figsize=page_size)
        scatter_grid.render_page(
            fig1,
            rows,
            coverage,
            title=title,
            variants_line=variants_line,
            data_source_line=data_source_line,
        )
        if coverage.plot_ready_points == 0:
            _render_empty_banner(fig1, OMISSION_REASONS["scatter_empty"])
            status.omit("scatter_empty")
        else:
            status.rendered_pages.append("scatter (page 1)")
        # UTC provenance footer in the clear bottom margin (scatter grid
        # subplots start at bottom=0.05).
        fig1.text(
            0.5,
            0.015,
            provenance["footer"],
            ha="center",
            va="bottom",
            fontsize=6,
            color="#888888",
        )
        pdf.savefig(fig1)
        plt.close(fig1)

        # Page 2: heatmap tables.
        fig2 = plt.figure(figsize=page_size)
        heatmap_tables.render_page(fig2, rows)
        status.rendered_pages.append("heatmap (page 2)")
        pdf.savefig(fig2)
        plt.close(fig2)

        # Page 2b (conditional): TPM-supported-across-hardware capacity table
        # for pricing discussions. SLA thresholds + gpus_per_node come from the
        # campaign config.yaml `tpm:` block (discover_tpm_config), so the PDF
        # shows the same peak AND sla points the lake (tpm_v1) carries; no block
        # -> peak-only at the default node size. Drawn when >=1 group carries
        # output_tps_per_gpu; else omitted loudly.
        from tools.perf_tune_report.tpm_summary import (
            compute_tpm_summary,
            discover_tpm_config,
        )

        tpm_cfg = discover_tpm_config(campaign_dir)
        tpm_summary = compute_tpm_summary(
            rows,
            ttft_sla_ms=tpm_cfg.ttft_sla_ms,
            tpot_sla_ms=tpm_cfg.tpot_sla_ms,
            gpus_per_node=tpm_cfg.gpus_per_node,
            context_line=data_source_line,
            usd_per_gpu_hour=tpm_cfg.usd_per_gpu_hour,
        )
        # Per-cell mean GPU power (from dcgm_correlation.json) -> the TPM page
        # shows tokens-per-watt at the peak point when power was captured.
        power_by_cell: dict[str, float] = {}
        for cid, payload in cell_dcgm.items():
            w = payload.get("power_watts_per_gpu")
            if isinstance(w, (int, float)) and w > 0:
                power_by_cell[cid] = float(w)
        fig_tpm = plt.figure(figsize=page_size)
        if tpm_table.render_page(fig_tpm, rows, summary=tpm_summary, power_by_cell=power_by_cell):
            status.rendered_pages.append("TPM across hardware (page 2b)")
            pdf.savefig(fig_tpm)
        else:
            status.omit("tpm_table")
        plt.close(fig_tpm)

        # Page 3 (conditional): GPU kernel breakdown.
        if cell_kernels:
            fig3 = plt.figure(figsize=page_size)
            kernel_breakdown.render_page(fig3, cell_kernels)
            status.rendered_pages.append("kernel breakdown (page 3)")
            pdf.savefig(fig3)
            plt.close(fig3)
        else:
            status.omit("kernel_breakdown")

        # Page 4 (conditional on page 3 + sol-ceilings.yaml): L1 zymtrace SoL
        # roofline. sol_complete is computed AFTER all SoL pages (page4/5/6) as
        # "any SoL evidence rendered" -- so an ncu-only or DCGM-only campaign is
        # still sol_complete (with sol_rigor recording the level).
        if cell_kernels and sol_inputs is not None:
            ceilings, hw_key = sol_inputs
            fig4 = plt.figure(figsize=page_size)
            sol_roofline.render_page(fig4, cell_kernels, rows, ceilings, hw_key)
            status.rendered_pages.append("SoL roofline (page 4)")
            sol_l1 = True
            pdf.savefig(fig4)
            plt.close(fig4)
        else:
            status.omit("sol_roofline")

        # Page 5 (conditional on ncu_kernels.json + sol-ceilings.yaml):
        # byte-grounded per-kernel SoL roofline scatter.
        if cell_ncu and ncu_sol_inputs is not None:
            ceilings, hw_key = ncu_sol_inputs
            # ncu_kernels.json carries its own hw_key; prefer it over the
            # atlas-row-derived one when they disagree (e.g. a B200
            # capture re-rendered in a multi-hw campaign).
            first_payload_hw = next(iter(cell_ncu.values())).get("hw_key")
            if first_payload_hw and first_payload_hw in ceilings:
                hw_key = first_payload_hw
            fig5 = plt.figure(figsize=page_size)
            # v1.23.2: pass cell_dcgm as fallback so page 5 plots
            # category-level points from DCGM per_category_attribution
            # when ncu kernels have all-null AI/tflops (e.g. --set=basic
            # capture).
            page5 = sol_roofline_scatter.render_page(
                fig5, cell_ncu, ceilings, hw_key, cell_dcgm=cell_dcgm
            )
            status.rendered_pages.append("ncu SoL scatter (page 5)")
            sol_l4 = True
            pdf.savefig(fig5)
            plt.close(fig5)
            # The page rendered, but if it carries only %SoL-only markers (AI
            # unmeasured) or no points at all, record it as PARTIAL so the
            # report is never read as a full measurement.
            if page5 is not None and page5.partial and page5.reason:
                status.mark_partial(page5.reason)
        else:
            status.omit("ncu_scatter")

        # Page 6 (conditional on dcgm_correlation.json): DCGM workload-level
        # byte-grounded SoL. Independent of page 4/5: this page draws
        # whenever a dcgm_correlation.json is present, even on bundles
        # that never ran ncu or zymtrace.
        if cell_dcgm:
            fig6 = plt.figure(figsize=page_size)
            dcgm_sol.render_page(fig6, cell_dcgm)
            status.rendered_pages.append("DCGM SoL (page 6)")
            sol_l3 = True
            pdf.savefig(fig6)
            plt.close(fig6)
        else:
            status.omit("dcgm_sol")
            # No dcgm_correlation.json -> no L2/L3 byte-grounding. Flag it
            # loudly (mirrors sol_complete) so a zymtrace-only campaign is
            # never published as if it were DCGM-grounded.
            status.dcgm_grounded = False

        # Page 6b (conditional on dcgm_correlation.json carrying a
        # non-empty per_category_attribution block): Level-2 zymtrace x
        # DCGM cross-attribution. Independent of page 5 (ncu); enabled
        # when correlate() was invoked with kernels_json_path=.
        if cell_dcgm and next(iter(cell_dcgm.values())).get("per_category_attribution"):
            fig6b = plt.figure(figsize=page_size)
            dcgm_category_attribution.render_page(fig6b, cell_dcgm)
            status.rendered_pages.append("zymtrace x DCGM cross-attribution (page 6b)")
            pdf.savefig(fig6b)
            plt.close(fig6b)
        else:
            status.omit("dcgm_xattr")

        # Page 7 (conditional on cells/*/roofline_sweep.json): the always-on
        # prefill/decode roofline + per-(c,ISL) DCGM utilization. L3
        # (DCGM-grounded), so it sets sol_l3 + dcgm_grounded when it draws.
        roofline_inputs = discover_sol_inputs(campaign_dir, rows)
        if roofline_inputs is None and cell_roofline:
            hwtok = next(iter(cell_roofline.values())).get("hardware")
            fb = {"B200": "b200_sm100", "GB300": "gb300_nvl72", "H100": "h100_sxm"}.get(hwtok)
            roofline_inputs = discover_sol_inputs(campaign_dir, rows, fallback_hw_key=fb)
        if cell_roofline and roofline_inputs is not None:
            r_ceilings, r_hw = roofline_inputs
            fig7 = plt.figure(figsize=page_size)
            prefill_decode_roofline.render_page(fig7, cell_roofline, r_ceilings, r_hw)
            status.rendered_pages.append("prefill/decode roofline (page 7)")
            sol_l3 = True
            status.dcgm_grounded = True
            pdf.savefig(fig7)
            plt.close(fig7)
        else:
            status.omit("prefill_decode_roofline")

        # Page 8 (conditional on champion_select.json): the production-choice
        # synthesis -- baseline vs top-X (cross-engine) + the overlaid roofline +
        # the DRAFT/VERDICT recommendation. Reuses the page-7 roofline ceilings
        # (falling back to the champion payload's hardware) for the overlay.
        if champion_payload is not None:
            champ_ceilings = champ_hw = None
            if roofline_inputs is not None:
                champ_ceilings, champ_hw = roofline_inputs
            else:
                hwtok = champion_payload.get("hardware")
                fb = {"B200": "b200_sm100", "GB300": "gb300_nvl72", "H100": "h100_sxm"}.get(hwtok)
                champ_inputs = discover_sol_inputs(campaign_dir, rows, fallback_hw_key=fb)
                if champ_inputs is not None:
                    champ_ceilings, champ_hw = champ_inputs
            fig8 = plt.figure(figsize=page_size)
            champion_select.render_page(fig8, champion_payload, champ_ceilings, champ_hw)
            status.rendered_pages.append("champion selection (page 8)")
            pdf.savefig(fig8)
            plt.close(fig8)
        else:
            status.omit("champion_select")

        # "Source under test" page (always-on when a provenance block exists):
        # the exact vLLM/SGLang commit + delivery + infr patch each roofline ran,
        # with a GitHub URL per entry. Reproducibility is not optional.
        prov_block = discover_provenance(campaign_dir)
        if prov_block:
            from tools.perf_tune_report import provenance as _prov_mod
            links = _prov_mod.source_links(prov_block, discover_source_registry(campaign_dir))
            fig_src = plt.figure(figsize=page_size)
            _render_source_page(fig_src, links, prov_block.get("identity", {}) or {})
            pdf.savefig(fig_src)
            plt.close(fig_src)
            status.rendered_pages.append("source under test")

        # Per-arm SoL coverage (v1.68.0): record which arms (baseline + each
        # variant) carry a RENDERED roofline page, computed here -- after every
        # page has been appended to rendered_pages -- so the page-4/6/7 flags
        # are accurate. The publish gate / teardown hook / audit key on these so
        # "baseline + EACH variant has a roofline" is enforced, not just the
        # campaign-level "some arm does" (sol_complete, set below).
        _rp = status.rendered_pages
        a_total, a_roofline, a_uncovered = compute_per_arm_coverage(
            campaign_dir,
            rows,
            cell_kernels,
            cell_dcgm,
            cell_roofline,
            page4=any("(page 4)" in s for s in _rp),
            page6=any("(page 6)" in s for s in _rp),
            page7=any("(page 7)" in s for s in _rp),
        )
        status.arms_total = a_total
        status.arms_with_roofline = a_roofline
        status.arms_uncovered = a_uncovered
        status.sol_per_arm_complete = not a_uncovered

        # Loud "Report completeness" page: rendered whenever anything was
        # omitted OR rendered partial OR an arm lacks a roofline, so a short,
        # limited, or per-arm-incomplete report is never a mystery and never
        # reads as complete.
        if status.omitted_pages or status.partial_pages or not status.sol_per_arm_complete:
            fig_done = plt.figure(figsize=page_size)
            _render_completeness_page(fig_done, status)
            pdf.savefig(fig_done)
            plt.close(fig_done)

    # Degenerate-asset guard (asset-validation rule m / CLAUDE.md "Validate every generated
    # asset"): a near-empty PDF means the render silently failed -- FAIL LOUD rather than
    # leaving a broken report on disk for publish_to_lake to land.
    _pdf_bytes = os.path.getsize(out_pdf) if os.path.exists(out_pdf) else 0
    if _pdf_bytes < 10000:
        raise RuntimeError(
            f"degenerate report: {out_pdf} is {_pdf_bytes} bytes -- the render produced no real "
            "content. Fix the campaign data and re-render; do not publish a broken report "
            "(CLAUDE.md 'Validate every generated asset')."
        )

    # sol_complete = ANY Speed-of-Light evidence rendered (L1 zymtrace page4 OR
    # L4 ncu page5 OR L3 DCGM page6). sol_rigor records the highest level so the
    # proxy-vs-tight distinction is a recorded field, never a publish blocker --
    # a latency-bound/ncu-only/proxy run is a first-class published result.
    status.sol_complete = sol_l1 or sol_l3 or sol_l4
    status.sol_rigor = (
        "L4" if sol_l4 else "L3" if sol_l3 else "L1" if sol_l1 else "none"
    )

    # Machine-readable sidecar consumed by publish_to_lake + the CLI.
    status_path = campaign_dir / "report_status.json"
    status_path.write_text(json.dumps(status.to_dict(), indent=2) + "\n", encoding="utf-8")

    return status
