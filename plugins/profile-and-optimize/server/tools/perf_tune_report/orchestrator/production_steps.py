"""Production ``StepFns`` for ``campaign_run``.

Wires the 10-step pipeline to real library + subprocess calls. Each step
is a small wrapper that adapts the orchestrator's typed callable shape to
the underlying production function.

Tests should NOT use this module directly — they should construct a
``StepFns`` with stub callables (see ``test_campaign_run.py``). This
module exists so the CLI entry point can pass through to the real
implementations when ``--i-understand-this-submits-jobs`` is set.

Step inventory
--------------

============================  ========================================================
Step                          Implementation
============================  ========================================================
drain                         ``kubectl cordon`` (k8s) — for Slurm-on-K8s partitions, use
                              ``scontrol update`` instead (operator-supplied list)
helm_upgrade                  ``helm upgrade --install -f base_values -f overlay``
warmup                        ``kubectl exec curl /v1/models`` retry up to 60s
bench                         dispatch on backend: vllm_sweep / aiperf / aa / trtllm
zymtrace                      no-op placeholder (zymtrace anchored query is fetched
                              by the operator out-of-band; bundle re-imports it)
import_bundle                 ``import_bundle_auto`` (handles both Kimi + GLM patterns)
aggregate                     ``aggregator.aggregate``
render                        ``renderer.render_report.render_report``
baseline_record               ``perf_baseline_cli.cmd_record`` via argparse.Namespace
baseline_diff                 ``perf_baseline_cli.cmd_diff``  via argparse.Namespace
resume                        ``kubectl uncordon`` paired with drain
============================  ========================================================

All cluster mutations honor a 60s timeout on each ``kubectl`` / ``helm``
call to prevent hangs from blocking always-resume.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import math
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from tools.perf_tune_report.helpers import resolve_operator_path
from tools.perf_tune_report.importers.zymtrace_kernels import (
    ZymtraceTSVMalformed,
    ZymtraceTSVMissing,
)
from tools.perf_tune_report.orchestrator.campaign_run import CellPlan, StepFns

_CMD_TIMEOUT_S = 120
_WARMUP_TIMEOUT_S = 60
_WARMUP_POLL_INTERVAL_S = 5
_SERVER_ROOT = Path(__file__).resolve().parents[3]
logger = logging.getLogger(__name__)

_BASELINE_METRICS: dict[str, tuple[str, str, str]] = {
    "output_tps_per_gpu": ("higher-is-better", "max", "tokens/s/GPU"),
    "request_throughput_avg": ("higher-is-better", "max", "requests/s"),
    "tpot_median_ms": ("lower-is-better", "min", "ms"),
    "itl_avg_ms": ("lower-is-better", "min", "ms"),
    "ttft_avg_ms": ("lower-is-better", "min", "ms"),
}


# ---------------------------------------------------------------------------
# Cluster mutation steps (subprocess)
# ---------------------------------------------------------------------------


def _run_cmd(cmd: list[str], *, timeout: int = _CMD_TIMEOUT_S) -> bool:
    """Run a command, returning True on exit 0, False on any failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except FileNotFoundError:
        return False


def step_drain(nodes: Sequence[str], cell_id: str) -> bool:
    """``kubectl cordon`` each node before running on it.

    Slurm-on-K8s partition drains require ``scontrol update partition=<x>
    state=DRAIN`` which is a separate code path the operator wires when
    the cluster uses Slurm. This step only handles the k8s-side cordon.
    """
    ok = True
    for node in nodes:
        if not _run_cmd(["kubectl", "cordon", node]):
            ok = False
    return ok


def step_resume(nodes: Sequence[str], cell_id: str) -> bool:
    """Pair to ``step_drain``: ``kubectl uncordon`` each node."""
    ok = True
    for node in nodes:
        if not _run_cmd(["kubectl", "uncordon", node]):
            ok = False
    return ok


def step_helm_upgrade(
    namespace: str,
    release: str,
    chart_dir: Path,
    overrides: dict[str, Any],
) -> bool:
    """``helm upgrade --install`` with the cell's overrides.

    Overrides can be (a) a path to an overlay yaml under
    ``overrides["path"]``, or (b) inline ``--set`` key=value pairs under
    ``overrides["set"]``.
    """
    cmd = [
        "helm",
        "upgrade",
        "--install",
        release,
        str(chart_dir),
        "--namespace",
        namespace,
    ]
    if "path" in overrides:
        cmd.extend(["-f", str(overrides["path"])])
    for k, v in overrides.get("set", {}).items():
        cmd.extend(["--set", f"{k}={v}"])
    return _run_cmd(cmd, timeout=300)


def step_warmup(namespace: str, release: str, cell_id: str) -> bool:
    """Poll the ``/v1/models`` endpoint via ``kubectl exec`` until 200 or timeout."""
    deadline = time.monotonic() + _WARMUP_TIMEOUT_S
    while time.monotonic() < deadline:
        cmd = [
            "kubectl",
            "-n",
            namespace,
            "exec",
            f"deploy/{release}",
            "--",
            "curl",
            "-sf",
            "http://localhost:8000/v1/models",
        ]
        if _run_cmd(cmd, timeout=10):
            return True
        time.sleep(_WARMUP_POLL_INTERVAL_S)
    return False


# ---------------------------------------------------------------------------
# Library-level steps (no subprocess; direct Python imports)
# ---------------------------------------------------------------------------


def _cell_config_from_plan(cell: CellPlan):
    """Build a runner ``CellConfig`` from a ``CellPlan``.

    Canonical identity axes come from the cell's top-level runner config.
    Legacy matrices that stored those fields in ``helm_overrides`` continue to
    work, with the same defaults used before canonical matrix normalization.
    """
    from tools.perf_tune_report.runners.common import CellConfig

    def value(name: str, default: Any) -> Any:
        return cell.runner_config.get(name, cell.helm_overrides.get(name, default))

    return CellConfig(
        cell_id=cell.id,
        model=str(value("model", "unknown")),
        hardware=str(value("hardware", "B200")),
        quant=str(value("quant", "NVFP4")),
        tensor_parallel=int(value("tensor_parallel", 8)),
        parallel_strategy=str(value("parallel_strategy", "TP")),
        mtp=bool(value("mtp", False)),
        max_num_batched_tokens=int(value("max_num_batched_tokens", 4096)),
        concurrencies=tuple(cell.concurrencies),
        extras=dict(value("extras", {})),
    )


def step_bench(campaign_dir: Path, cell: CellPlan) -> bool:
    """Dispatch to the right runner based on ``cell.backend``.

    Returns True if the runner produced a cell row, False otherwise.
    Runner errors (e.g. TRT-LLM stub raising NotImplementedError) are
    caught + reported to the operator via the per-cell receipt.
    """
    try:
        if cell.backend == "vllm-sweep":
            from tools.perf_tune_report.runners.vllm_sweep import run_cell as run_vllm

            cfg = _cell_config_from_plan(cell)
            serve_cmd = cell.backend_config.get("serve_cmd", cell.helm_overrides.get("serve_cmd", ""))
            bench_cmd = cell.backend_config.get("bench_cmd", cell.helm_overrides.get("bench_cmd", ""))
            if not serve_cmd or not bench_cmd:
                return False
            res = run_vllm(
                cfg,
                campaign_dir,
                serve_cmd=serve_cmd,
                bench_cmd=bench_cmd,
                dry_run=False,
            )
            return res.row_count > 0
        if cell.backend == "aiperf":
            # Replay mooncake-trace replay against an existing endpoint URL.
            # Config comes from the cell's ``aiperf:`` block (cell.backend_config).
            from tools.perf_tune_report.runners.aiperf_bench import run_cell as run_aiperf

            cfgd = cell.backend_config
            required = (
                "namespace",
                "bench_pod",
                "kube_context",
                "endpoint_url",
                "served_model",
            )
            if any(not cfgd.get(k) for k in required):
                return False
            res = run_aiperf(
                _cell_config_from_plan(cell),
                campaign_dir,
                namespace=cfgd["namespace"],
                bench_pod=cfgd["bench_pod"],
                kube_context=cfgd["kube_context"],
                endpoint_url=cfgd["endpoint_url"],
                served_model=cfgd["served_model"],
                dataset_split=cfgd.get("dataset_split", "2025_07"),
                conversation_count=cfgd.get("conversation_count"),
                dry_run=False,
            )
            return res.row_count > 0
        if cell.backend == "aa":
            # Artificial-Analysis synthetic fixed-shape (or dataset-replay) via
            # AIPerf against an endpoint URL. Config from the ``aa:`` block.
            import os

            from tools.perf_tune_report.runners.aa_bench import run_cell as run_aa

            cfgd = cell.backend_config
            model = cfgd.get("model")
            url = cfgd.get("url")
            shape_name = cfgd.get("shape")
            if not model or not url or not shape_name:
                return False
            api_key_env = cfgd.get("api_key_env", "WANDB_INFERENCE_API_KEY")
            api_key = os.environ.get(api_key_env) if api_key_env else None
            kube = None
            namespace = cfgd.get("namespace")
            if namespace:
                kube = {
                    "namespace": namespace,
                    "bench_pod": cfgd.get("bench_pod"),
                    "kube_context": cfgd.get("kube_context"),
                }
            res = run_aa(
                _cell_config_from_plan(cell),
                campaign_dir,
                shape_name=shape_name,
                model=model,
                url=url,
                endpoint=cfgd.get("endpoint", "/v1/chat/completions"),
                endpoint_type=cfgd.get("endpoint_type", "chat"),
                api_key=api_key,
                tokenizer=cfgd.get("tokenizer"),
                tokenizer_trust_remote_code=bool(cfgd.get("tokenizer_trust_remote_code", True)),
                mode=cfgd.get("mode", "synthetic"),
                request_count=int(cfgd.get("request_count", 10)),
                custom_dataset_type=cfgd.get("custom_dataset_type", "mooncake_trace"),
                extra_output_controls=bool(cfgd.get("extra_output_controls", True)),
                input_file=cfgd.get("input_file"),
                dataset_count=cfgd.get("dataset_count"),
                aiperf_cmd=cfgd.get("aiperf_cmd"),
                kube=kube,
                dry_run=False,
            )
            return res.row_count > 0
        if cell.backend == "trtllm":
            return False
    except (
        ImportError,
        KeyError,
        NotImplementedError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        TypeError,
        ValueError,
    ) as exc:
        logger.debug("Benchmark step failed for cell %s: %s", cell.id, exc)
        return False
    return False


def step_zymtrace(campaign_dir: Path, cell_id: str) -> bool:
    """Best-effort zymtrace coverage check.

    The full anchored-query workflow lives in the
    ``prometheus-anchored-query`` skill and is fetched out-of-band by the
    operator. This step just verifies that a zymtrace coverage manifest
    exists in the cell's directory (i.e. the operator pre-staged it
    via the skill). Missing -> returns False but does NOT abort the cell.
    """
    cell_dir = campaign_dir / "cells" / cell_id
    return (cell_dir / "capture_sources.json").is_file()


def step_import(cell_dir: Path, campaign_dir: Path, cell_id: str) -> bool:
    """Wrap ``import_bundle_auto`` so it adheres to the StepFns signature."""
    try:
        from tools.perf_tune_report.importers import import_bundle_auto

        # The cell_dir IS the bundle dir under the new pipeline (where the
        # bench output lands). For backward compat, accept either layout:
        # cell_dir/raw or cell_dir/bench-c<NNN>.
        import_bundle_auto(
            bundle=cell_dir,
            campaign_dir=campaign_dir,
            overrides={"cell_id": cell_id},
            dry_run=False,
        )
        return True
    except (ImportError, OSError, TypeError, ValueError, ZymtraceTSVMalformed, ZymtraceTSVMissing) as exc:
        logger.debug("Import step failed for cell %s: %s", cell_id, exc)
        return False


def step_aggregate(campaign_dir: Path) -> bool:
    try:
        from tools.perf_tune_report.aggregator import aggregate

        aggregate(campaign_dir)
        return True
    except (ImportError, OSError, TypeError, ValueError) as exc:
        logger.debug("Aggregate step failed for %s: %s", campaign_dir, exc)
        return False


def _discover_ceilings_yaml(campaign_dir: Path) -> Path | None:
    """Locate sol-ceilings.yaml: SOL_CEILINGS_YAML env, else walk up from the
    campaign dir for configs/sol-ceilings.yaml."""
    import os

    env_override = os.environ.get("SOL_CEILINGS_YAML", "").strip()
    if env_override and env_override != "disable":
        p = resolve_operator_path(env_override, label="SOL_CEILINGS_YAML")
        return p if p.is_file() else None
    relpath = Path("perf-tune-report") / "configs" / "sol-ceilings.yaml"
    cur = campaign_dir.resolve()
    for parent in [cur, *cur.parents]:
        candidate = parent / relpath
        if candidate.is_file():
            return candidate
    return None


def step_dcgm_correlate(campaign_dir: Path, cell_id: str) -> bool:
    """Fold a per-cell frozen DCGM YAML into cells/<id>/dcgm_correlation.json.

    Byte-grounding step (L2/L3): produces the renderer's page 6 (workload
    byte/FLOP SoL) + page 6b (zymtrace x DCGM cross-attribution) input, which
    flips a campaign from sol_complete-only to dcgm_grounded=true.

    Convention: the operator/skill drops a ``dcgm-frozen.yaml`` (dcgm_frozen_v1)
    into the cell dir before this step (the live Prometheus correlate() path
    is a library API the orchestrator subprocess cannot reach). Missing frozen
    YAML -> returns False (loud-skip via the caller's note), does NOT abort.
    """
    try:
        from tools.perf_tune_report.dcgm_correlate import (
            FrozenYamlMalformed,
            correlate_from_frozen,
            write_correlation,
        )
    except ImportError as exc:
        logger.debug("DCGM correlation imports failed for cell %s: %s", cell_id, exc)
        return False

    try:
        cell_dir = campaign_dir / "cells" / cell_id
        frozen = cell_dir / "dcgm-frozen.yaml"
        if not frozen.is_file():
            return False  # no DCGM input for this cell; caller logs a loud skip
        ceilings_path = _discover_ceilings_yaml(campaign_dir)
        if ceilings_path is None:
            return False
        ceilings = yaml.safe_load(ceilings_path.read_text())
        kernels = cell_dir / "kernels.json"
        result = correlate_from_frozen(
            frozen,
            ceilings,
            cell_dir=cell_dir,
            kernels_json_path=kernels if kernels.is_file() else None,
        )
        write_correlation(result, cell_dir)
        return True
    except (
        FrozenYamlMalformed,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as exc:
        logger.debug("DCGM correlation step failed for cell %s: %s", cell_id, exc)
        return False


def step_render(campaign_dir: Path, out_pdf: Path) -> bool:
    try:
        from tools.perf_tune_report.renderer.champion_select import (
            ChampionSelectJsonMalformed,
        )
        from tools.perf_tune_report.renderer.prefill_decode_roofline import (
            RooflineSweepJsonMalformed,
        )
        from tools.perf_tune_report.renderer.render_report import (
            DcgmCorrelationJsonMalformed,
            KernelsJsonMalformed,
            NcuKernelsJsonMalformed,
            render_report,
        )
        from tools.perf_tune_report.renderer.sol_roofline import SoLCeilingsMalformed
    except ImportError as exc:
        logger.debug("Render imports failed for %s: %s", campaign_dir, exc)
        return False

    try:
        atlas = campaign_dir / "atlas.jsonl"
        if not atlas.is_file():
            return False
        render_report(atlas, out_pdf, title=f"campaign {campaign_dir.name}")
        return True
    except (
        ChampionSelectJsonMalformed,
        DcgmCorrelationJsonMalformed,
        KernelsJsonMalformed,
        NcuKernelsJsonMalformed,
        OSError,
        RooflineSweepJsonMalformed,
        RuntimeError,
        SoLCeilingsMalformed,
        TypeError,
        ValueError,
    ) as exc:
        logger.debug("Render step failed for %s: %s", campaign_dir, exc)
        return False


def _baseline_repo_root() -> Path:
    """Resolve the artifact root without depending on the MCP client's cwd."""
    import os

    override = os.environ.get("PROFILE_AND_OPTIMIZE_REPO_ROOT")
    return resolve_operator_path(override, label="PROFILE_AND_OPTIMIZE_REPO_ROOT") if override else _SERVER_ROOT


def _cell_headline(campaign_dir: Path, cell_id: str) -> dict[str, Any] | None:
    """Materialize one comparable scalar headline for a campaign cell."""
    config_path = campaign_dir / "config.yaml"
    atlas_path = campaign_dir / "atlas.jsonl"
    if not config_path.is_file() or not atlas_path.is_file():
        return None
    config = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(config, dict):
        return None
    campaign_config = config.get("campaign")
    if not isinstance(campaign_config, dict):
        campaign_config = {}
    focus = campaign_config.get("focus", config.get("focus"))
    metric = campaign_config.get("baseline_metric", config.get("baseline_metric"))
    if metric is None:
        if focus == "throughput":
            metric = "output_tps_per_gpu"
        elif focus == "latency":
            metric = "tpot_median_ms"
        else:
            return None
    if metric not in _BASELINE_METRICS:
        return None
    direction, reduction, unit = _BASELINE_METRICS[metric]
    requested_concurrency = campaign_config.get("baseline_concurrency", config.get("baseline_concurrency"))

    candidates: list[tuple[float, int | None]] = []
    try:
        for line in atlas_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("cell_id") != cell_id:
                continue
            concurrency = row.get("concurrency")
            if requested_concurrency is not None and concurrency != requested_concurrency:
                continue
            value = row.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                continue
            candidates.append((numeric, concurrency if isinstance(concurrency, int) else None))
    except (OSError, json.JSONDecodeError):
        return None
    if not candidates:
        return None

    selected = max(candidates) if reduction == "max" else min(candidates)
    value, concurrency = selected
    measurement = metric if concurrency is None else f"{metric}-c{concurrency}"
    cell_dir = campaign_dir / "cells" / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    source = cell_dir / "baseline-headline.json"
    payload = {
        "schema": "campaign_headline_v1",
        "cell_id": cell_id,
        "metric": metric,
        "concurrency": concurrency,
        "direction": direction,
        "unit": unit,
        "value": value,
    }
    source.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {
        **payload,
        "measurement": measurement,
        "source": source,
    }


def step_baseline_record(campaign_dir: Path, cell_id: str) -> bool:
    """Register one directional scalar headline for the campaign cell."""
    try:
        from tools.perf_baseline.perf_baseline_cli import cmd_record

        headline = _cell_headline(campaign_dir, cell_id)
        if headline is None:
            return False
        ns = argparse.Namespace(
            family="inference",
            measurement=headline["measurement"],
            source=str(headline["source"]),
            value=headline["value"],
            unit=headline["unit"],
            direction=headline["direction"],
            schema=None,
            notes=f"auto-recorded by campaign_run, cell_id={cell_id}",
            repo_root=str(_baseline_repo_root()),
            json=True,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            return cmd_record(ns) == 0
    except SystemExit:
        return False
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError, yaml.YAMLError) as exc:
        logger.debug("Baseline record step failed for cell %s: %s", cell_id, exc)
        return False


def step_baseline_diff(campaign_dir: Path, cell_id: str, comparator: str) -> str:
    """Diff the cell's atlas against ``comparator``; return verdict.

    Returns ``GREEN`` / ``YELLOW`` / ``RED`` / ``NA``. If no comparator is
    set or the comparator file is missing, returns ``NA``.
    """
    if not comparator:
        return "NA"
    comparator_path = Path(comparator).expanduser()
    if not comparator_path.is_dir():
        return "RED"
    try:
        from tools.perf_baseline.perf_baseline_cli import cmd_diff

        headline = _cell_headline(campaign_dir, cell_id)
        if headline is None:
            return "RED"
        baseline_payload = json.loads((comparator_path / "baseline.json").read_text())
        if (
            baseline_payload.get("measurement") != headline["measurement"]
            or baseline_payload.get("direction", "two-sided") != headline["direction"]
        ):
            return "RED"
        ns = argparse.Namespace(
            baseline=str(comparator_path),
            current=str(headline["source"]),
            tolerance_percent=2.0,
            tolerance_absolute=None,
            repo_root=str(_baseline_repo_root()),
            json=True,
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rc = cmd_diff(ns)
        if rc != 0:
            return "RED"
        payload = json.loads(output.getvalue())
        verdict = payload.get("verdict")
        return verdict if verdict in {"GREEN", "YELLOW", "RED"} else "RED"
    except SystemExit:
        return "RED"
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError, yaml.YAMLError) as exc:
        logger.debug("Baseline diff step failed for cell %s: %s", cell_id, exc)
        return "RED"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def production_step_fns() -> StepFns:
    """Build the production ``StepFns`` bundle (real subprocess + libs)."""
    return StepFns(
        drain=step_drain,
        helm_upgrade=step_helm_upgrade,
        warmup=step_warmup,
        bench=step_bench,
        zymtrace=step_zymtrace,
        import_bundle=step_import,
        aggregate=step_aggregate,
        render=step_render,
        baseline_record=step_baseline_record,
        baseline_diff=step_baseline_diff,
        resume=step_resume,
        dcgm_correlate=step_dcgm_correlate,
    )
