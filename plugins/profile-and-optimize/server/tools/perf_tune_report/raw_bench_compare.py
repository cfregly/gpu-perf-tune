"""Multi-bundle vllm-bench-serve comparison report (v1.24.0).

Promotes the GLM-LWS workshop renderers
(``./campaigns workspacescripts/render_lws_baseline_report.py``
+ ``render_phase_a_report.py``) into a reusable profile-and-optimize helper.

Use case
--------

The existing ``perftunereport report_render`` produces a faceted multi-page
PDF from an ``atlas.jsonl`` (5x2 scatter facet by
``max_num_batched_tokens`` + 3x2 per-concurrency heatmap tables). That
is the rich-dataset analysis view.

``raw_bench_compare`` is the complementary *linear comparison* view:
operator supplies a YAML manifest of N bundles + per-bundle display
metadata (label, short, color, marker, knob, is_baseline); the verb
overlays each bundle's per-concurrency curve onto a single chart per
metric (throughput / TTFT / TPOT) + a peak-bars chart with
percent-gain-vs-baseline + a summary table. Targeted at the
"6-variant champion comparison" use case where faceting hides the
linear story.

Bypass-the-pipeline by design: doesn't write to ``cells/<id>/``,
doesn't call ``atlas_aggregate``. The operator just runs this once
per comparison and gets a PDF directly.

YAML manifest schema (``raw_bench_compare_v1``)
-----------------------------------------------

.. code-block:: yaml

    schema_version: 1
    campaign_name: "GLM-5.1 LWS baseline vs champions"
    hardware: "B200"
    model: "GLM-5.1"
    notes: "Single 8x B200 g1fb686 sandbox; TP=8, NVLink-only."
    bundles_root: "<external-workspace>"
    bundles:
      - glob: "glm51-LWS-baseline-*"
        label: "LWS-baseline (mns=40, mbt=32768, kv=fp8)"
        short: "LWS-baseline"
        knob: "mns=40, mbt=32768, kv=fp8"
        color: "#444444"
        marker: "s"
        is_baseline: true
      - glob: "glm51-E4-mns128-*"
        label: "E4 (mns=128) ULTIMATE-CHAMPION"
        short: "E4-mns128"
        knob: "mns 96 -> 128 + kv -> fp8_e4m3"
        color: "#9467bd"
        marker: "D"

When ``bundles_root`` is absent or a relative path, each bundle's
``glob`` is resolved relative to the manifest's parent directory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_SWEEP_C_RE = re.compile(r"^sweep-c(\d+)\.txt$")


class RawBenchCompareManifestMalformed(Exception):
    """Raised when a manifest YAML is present but unusable."""

    def __init__(self, path: Path, reason: str):
        super().__init__(f"raw_bench_compare manifest malformed: {path} ({reason})")
        self.path = path
        self.reason = reason


@dataclass
class BundleSpec:
    """One entry from the manifest, resolved + parsed."""

    glob: str
    label: str
    short: str
    knob: str = ""
    color: str = "#1f77b4"
    marker: str = "o"
    is_baseline: bool = False
    # Resolved at load time:
    bundle_path: Path | None = None
    rows: list[dict[str, float | int | str]] = field(default_factory=list)


@dataclass
class RawBenchCompareResult:
    """Returned by ``render_comparison()`` for the JSON envelope."""

    manifest_path: Path
    out_pdf: Path
    campaign_name: str
    n_bundles: int
    n_bundles_with_data: int
    n_rows_total: int
    baseline_short: str | None
    baseline_peak_tps: float | None
    peaks: list[dict[str, Any]]  # one entry per bundle with data

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "out_pdf": str(self.out_pdf),
            "campaign_name": self.campaign_name,
            "n_bundles": self.n_bundles,
            "n_bundles_with_data": self.n_bundles_with_data,
            "n_rows_total": self.n_rows_total,
            "baseline_short": self.baseline_short,
            "baseline_peak_tps": self.baseline_peak_tps,
            "peaks": self.peaks,
        }


def _parse_sweep_file(
    path: Path, *, concurrency: int | None = None
) -> dict[str, float | int | str] | None:
    """Parse one vllm-bench-serve output file's metrics.

    Reuses the regex set from ``importers.inference_perf_bench._REGEX`` so the parser
    stays in sync. The concurrency is taken from the ``sweep-c<N>.txt`` filename by
    default; pass ``concurrency`` explicitly to parse a differently-named file (e.g.
    ``import_workloads`` parses ``<tag>-c<c>.txt`` and supplies the ``c`` itself).
    Returns ``None`` if the filename does not match and no concurrency was supplied,
    or if the run did not complete.
    """
    from tools.perf_tune_report.importers.inference_perf_bench import _REGEX

    if concurrency is None:
        m = _SWEEP_C_RE.match(path.name)
        if not m:
            return None
        concurrency = int(m.group(1))
    text = path.read_text(errors="replace")
    out: dict[str, float | int | str] = {"path": str(path), "c": int(concurrency)}
    # Aliases the workshop scripts used (ttft_median_ms etc.) -> existing keys.
    for key, rx in _REGEX.items():
        match = rx.search(text)
        if match:
            val = match.group(1)
            out[key] = int(val) if key == "n_reqs" else float(val)
    # Surface the workshop-friendly aliases too.
    if "ttft_med_ms" in out:
        out["ttft_median_ms"] = out["ttft_med_ms"]
    if "tpot_med_ms" in out:
        out["tpot_median_ms"] = out["tpot_med_ms"]
    # Require minimal completeness.
    if "output_tps" not in out and "req_per_s" not in out:
        return None
    return out


def _resolve_bundle(spec: BundleSpec, bundles_root: Path) -> Path | None:
    """Pick the most recent bundle matching the spec's glob."""
    matches = sorted(bundles_root.glob(spec.glob))
    if not matches:
        return None
    return matches[-1]


def _load_bundle(bundle_path: Path) -> list[dict[str, float | int | str]]:
    """Load + parse all ``raw/sweep-c<N>.txt`` files in one bundle."""
    raw = bundle_path / "raw"
    if not raw.is_dir():
        return []
    rows: list[dict[str, float | int | str]] = []
    for fp in sorted(raw.glob("sweep-c*.txt"), key=lambda p: int(p.stem.split("c")[1])):
        row = _parse_sweep_file(fp)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda r: r["c"])  # type: ignore[arg-type]
    return rows


def load_manifest(manifest_path: Path) -> tuple[dict[str, Any], list[BundleSpec]]:
    """Load + validate a ``raw_bench_compare_v1`` YAML manifest.

    Returns ``(meta_dict, bundle_specs)``. Each ``BundleSpec`` is
    populated with ``bundle_path`` resolved (via ``bundles_root`` walk)
    and ``rows`` parsed from disk.

    Raises ``RawBenchCompareManifestMalformed`` if the manifest is
    missing required keys.
    """
    import yaml as _yaml

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    try:
        data = _yaml.safe_load(manifest_path.read_text())
    except _yaml.YAMLError as e:
        raise RawBenchCompareManifestMalformed(
            manifest_path, reason=f"YAML parse error: {e}"
        ) from e
    if not isinstance(data, dict):
        raise RawBenchCompareManifestMalformed(
            manifest_path, reason="top-level not a mapping"
        )
    bundles_in = data.get("bundles")
    if not isinstance(bundles_in, list) or not bundles_in:
        raise RawBenchCompareManifestMalformed(
            manifest_path, reason="bundles[] missing or empty"
        )

    bundles_root_raw = data.get("bundles_root")
    if bundles_root_raw:
        bundles_root = Path(bundles_root_raw).expanduser()
        if not bundles_root.is_absolute():
            bundles_root = (manifest_path.parent / bundles_root).resolve()
    else:
        bundles_root = manifest_path.parent.resolve()

    specs: list[BundleSpec] = []
    for entry in bundles_in:
        if not isinstance(entry, dict):
            raise RawBenchCompareManifestMalformed(
                manifest_path, reason=f"bundle entry not a mapping: {entry!r}"
            )
        for required in ("glob", "label", "short"):
            if required not in entry:
                raise RawBenchCompareManifestMalformed(
                    manifest_path,
                    reason=f"bundle entry missing '{required}': {entry!r}",
                )
        spec = BundleSpec(
            glob=entry["glob"],
            label=entry["label"],
            short=entry["short"],
            knob=entry.get("knob", ""),
            color=entry.get("color", "#1f77b4"),
            marker=entry.get("marker", "o"),
            is_baseline=bool(entry.get("is_baseline", False)),
        )
        resolved = _resolve_bundle(spec, bundles_root)
        if resolved is not None:
            spec.bundle_path = resolved
            spec.rows = _load_bundle(resolved)
        specs.append(spec)

    return data, specs


def render_comparison(
    manifest_path: Path,
    out_pdf: Path,
) -> RawBenchCompareResult:
    """Render the multi-bundle comparison PDF.

    Args:
        manifest_path: ``raw_bench_compare_v1`` YAML.
        out_pdf: destination PDF path.

    Returns:
        ``RawBenchCompareResult`` summarising bundle resolution + peaks.

    Raises:
        FileNotFoundError: manifest path absent.
        RawBenchCompareManifestMalformed: manifest YAML unusable.
        ValueError: no bundles produced any parseable rows (the PDF
            would be empty; refuse to render).
    """
    meta, specs = load_manifest(manifest_path)

    n_with_data = sum(1 for s in specs if s.rows)
    if n_with_data == 0:
        raise ValueError(
            f"render_comparison: no bundles in {manifest_path} yielded "
            f"parseable rows; PDF would be empty"
        )

    # Lazy matplotlib import.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    campaign_name = meta.get("campaign_name", "raw bench comparison")
    hardware = meta.get("hardware", "")
    model = meta.get("model", "")
    notes = meta.get("notes", "")

    baseline = next((s for s in specs if s.is_baseline and s.rows), None)
    baseline_peak_tps = (
        max((row["output_tps"] for row in baseline.rows if "output_tps" in row), default=None)
        if baseline
        else None
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    peaks_summary: list[dict[str, Any]] = []

    with PdfPages(out_pdf) as pdf:
        # Page 1: cover
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.85, campaign_name, ha="center", fontsize=18, fontweight="bold")
        sub = f"{model} on {hardware}" if (model or hardware) else "raw vllm-bench-serve comparison"
        ax.text(0.5, 0.78, sub, ha="center", fontsize=11)
        ax.text(0.5, 0.73, f"manifest: {manifest_path.name}", ha="center", fontsize=8, color="#666")
        ax.text(
            0.5, 0.65,
            f"{n_with_data} of {len(specs)} bundles produced parseable data",
            ha="center", fontsize=10,
        )
        if notes:
            ax.text(
                0.5, 0.55, notes, ha="center", va="top", fontsize=9, color="#444",
                wrap=True,
            )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: throughput vs concurrency
        _render_line_chart(
            pdf, specs,
            metric_key="output_tps",
            title=f"Output throughput vs concurrency ({campaign_name})",
            ylabel="Output tokens / sec",
            logx=True,
        )
        # Page 3: TTFT vs concurrency
        _render_line_chart(
            pdf, specs,
            metric_key="ttft_med_ms",
            title="Median TTFT vs concurrency (lower = better)",
            ylabel="Median TTFT (ms)",
            logx=True,
            logy=True,
        )
        # Page 4: TPOT vs concurrency
        _render_line_chart(
            pdf, specs,
            metric_key="tpot_med_ms",
            title="Median TPOT vs concurrency (lower = better)",
            ylabel="Median TPOT (ms / token)",
            logx=False,
        )
        # Page 5: peak bars + %gain vs baseline
        peaks_summary = _render_peak_bars(pdf, specs, baseline_peak_tps)
        # Page 6: summary table
        _render_summary_table(pdf, specs, baseline_peak_tps)

    return RawBenchCompareResult(
        manifest_path=manifest_path,
        out_pdf=out_pdf,
        campaign_name=campaign_name,
        n_bundles=len(specs),
        n_bundles_with_data=n_with_data,
        n_rows_total=sum(len(s.rows) for s in specs),
        baseline_short=baseline.short if baseline else None,
        baseline_peak_tps=baseline_peak_tps,
        peaks=peaks_summary,
    )


def _render_line_chart(pdf, specs, *, metric_key, title, ylabel, logx=False, logy=False):
    """One line chart, all bundles overlaid."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 8))
    plotted = 0
    for spec in specs:
        if not spec.rows:
            continue
        xs = [r["c"] for r in spec.rows if metric_key in r]
        ys = [r[metric_key] for r in spec.rows if metric_key in r]
        if not xs:
            continue
        lw = 2.5 if spec.is_baseline else 1.8
        ls = "--" if spec.is_baseline else "-"
        ax.plot(
            xs, ys, color=spec.color, marker=spec.marker, label=spec.label,
            linewidth=lw, linestyle=ls, markersize=8,
        )
        plotted += 1
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Concurrency" + (" (log2)" if logx else ""))
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linestyle="--")
    if plotted > 0:
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
    else:
        ax.text(
            0.5, 0.5,
            f"No bundles carry '{metric_key}' rows",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="#aa3333",
        )
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _render_peak_bars(pdf, specs, baseline_peak_tps: float | None) -> list[dict[str, Any]]:
    """Bar chart of peak output_tps per bundle + %gain-vs-baseline annotation."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 7))
    labels: list[str] = []
    peaks: list[float] = []
    peak_cs: list[int] = []
    colors: list[str] = []
    peaks_summary: list[dict[str, Any]] = []
    for spec in specs:
        if not spec.rows:
            continue
        ranked = [r for r in spec.rows if "output_tps" in r]
        if not ranked:
            continue
        best = max(ranked, key=lambda r: r["output_tps"])
        labels.append(spec.short)
        peaks.append(float(best["output_tps"]))
        peak_cs.append(int(best["c"]))
        colors.append(spec.color)
        peaks_summary.append(
            {
                "short": spec.short,
                "peak_output_tps": float(best["output_tps"]),
                "peak_c": int(best["c"]),
                "pct_vs_baseline": (
                    ((best["output_tps"] / baseline_peak_tps) - 1.0) * 100.0
                    if baseline_peak_tps
                    else None
                ),
            }
        )
    if not peaks:
        ax.text(
            0.5, 0.5,
            "No bundle has output_tps rows; peak-bars chart skipped.",
            transform=ax.transAxes, ha="center", va="center", color="#aa3333",
        )
    else:
        bars = ax.bar(labels, peaks, color=colors, edgecolor="black", linewidth=0.8)
        for bar, peak, pc in zip(bars, peaks, peak_cs):
            pct = ((peak / baseline_peak_tps) - 1.0) * 100.0 if baseline_peak_tps else 0.0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{peak:.0f}\n@ c={pc}" + (f"\n{pct:+.0f}%" if baseline_peak_tps else ""),
                ha="center", va="bottom", fontsize=9,
            )
        ax.set_ylim(0, max(peaks) * 1.18)
    ax.set_title(
        "Peak output throughput per variant"
        + (" (% gain vs baseline)" if baseline_peak_tps else ""),
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("Peak output tokens / sec")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    return peaks_summary


def _render_summary_table(pdf, specs, baseline_peak_tps: float | None) -> None:
    """Per-variant tabular summary."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.axis("off")
    headers = ["Variant", "Knob change", "Peak tok/s", "@ c (peak)", "% vs baseline"]
    rows_table: list[list[str]] = []
    for spec in specs:
        if not spec.rows:
            rows_table.append([spec.short, spec.knob, "n/a", "n/a", "n/a"])
            continue
        ranked = [r for r in spec.rows if "output_tps" in r]
        if not ranked:
            rows_table.append([spec.short, spec.knob, "n/a", "n/a", "n/a"])
            continue
        best = max(ranked, key=lambda r: r["output_tps"])
        pct = (
            f"{((best['output_tps'] / baseline_peak_tps) - 1.0) * 100.0:+.1f}%"
            if baseline_peak_tps
            else "n/a"
        )
        rows_table.append(
            [
                spec.short,
                spec.knob or "",
                f"{best['output_tps']:.0f}",
                str(best["c"]),
                pct,
            ]
        )
    table = ax.table(cellText=rows_table, colLabels=headers, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)
    ax.set_title(
        "Summary table (per-variant peak + % gain vs baseline)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
