"""Artificial Analysis (AA) workload shapes: shared command builder + dataset generator.

Reproduces the Artificial Analysis language-model performance benchmarking
workload shapes documented at
https://artificialanalysis.ai/methodology/performance-benchmarking

AA tests three text shapes (plus an optional vision shape) with
``temperature 0`` / ``top_p 1``, tokens counted as ``o200k_base`` (tiktoken),
and an "at least N answer tokens" guarantee enforced via vLLM-style
``min_tokens`` + ``ignore_eos`` request fields. AA generates a fresh prompt
per run to fill a target token budget and replays the *same* prompt across
every endpoint it covers.

This module is the single source of truth shared by two callers:

- the standalone ``repro_artificialanalysis.py`` script bundled with the
  ``inference-aa-workload`` skill (placement A), and
- the ``aa_bench`` perf-report runner (placement B) that flows AA cells
  through ``atlas_aggregate -> report_render -> publish_to_lake``.

Two run modes are supported:

- ``synthetic``      -- AIPerf synthesizes a prompt at the target token mean
  (``--synthetic-input-tokens-mean``). Faithful to the original bash script.
- ``dataset-replay`` -- a generated JSONL of real-text prompts is replayed
  identically (``--input-file`` + ``--custom-dataset-type mooncake_trace``
  with the ``text_input`` field), giving better cross-endpoint comparability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Reuse the AIPerf JSON parsing the aiperf runner already proved out, so the
# two runners stay in lock-step on AIPerf's version-varying report layout.
from tools.perf_tune_report.runners.aiperf_bench import _extract_metrics, _parse_aiperf_json
from tools.perf_tune_report.runners.common import CellConfig, utc_now_iso
from tools.perf_tune_report.schema import (
    BACKEND_AIPERF,
    STATUS_FAILED,
    STATUS_FULL,
    STATUS_PARTIAL,
    AtlasCell,
)

# AA API parameters for non-reasoning models (methodology "API Parameters").
AA_TEMPERATURE = 0
AA_TOP_P = 1

# AA counts every token as an OpenAI GPT-4 ``o200k_base`` tiktoken token so the
# same text is the same token count across model tokenizers.
O200K_ENCODING = "o200k_base"

# AIPerf custom-dataset-type for exact-text replay. The upstream
# inference-tools perf-bench runner uses the hyphenated ``mooncake-trace``;
# AIPerf accepts the underscore form in its docs. We default to the underscore
# form here (the AA reproducer is a fresh path) but keep it overridable.
DEFAULT_CUSTOM_DATASET_TYPE = "mooncake_trace"

# Run modes.
MODE_SYNTHETIC = "synthetic"
MODE_DATASET_REPLAY = "dataset-replay"
MODES = (MODE_SYNTHETIC, MODE_DATASET_REPLAY)


@dataclass(frozen=True)
class AAShape:
    """One AA workload shape: a target input-token budget + min answer tokens."""

    name: str
    input_tokens: int
    output_tokens: int


# Canonical AA shapes (single source of truth). The standalone script and the
# test drift-guard both assert against this table. Numbers per the AA
# methodology "Workload Types" table:
#   1k   -> ~1,000 input  / at least 1,000 answer tokens
#   10k  -> ~10,000 input / at least 1,500 answer tokens (AA site default)
#   100k -> ~100,000 input / at least 2,000 answer tokens
AA_SHAPES: dict[str, AAShape] = {
    "aa-1k": AAShape("aa-1k", 1000, 1000),
    "aa-10k": AAShape("aa-10k", 10000, 1500),
    "aa-100k": AAShape("aa-100k", 100000, 2000),
}

# Order the bash script runs them in (and the script's default selection).
AA_DEFAULT_SHAPE_ORDER = ("aa-1k", "aa-10k", "aa-100k")


class TokenizerUnavailable(RuntimeError):
    """Raised when neither tiktoken nor a fallback heuristic was requested."""


def count_tokens(text: str, encoding_name: str = O200K_ENCODING) -> int:
    """Count tokens with tiktoken's ``o200k_base`` encoding.

    Raises ``TokenizerUnavailable`` if tiktoken is not installed so callers
    can decide whether to fall back to a heuristic (and warn loudly) rather
    than silently producing a wrong token count.
    """
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - exercised via fallback path
        raise TokenizerUnavailable(
            "tiktoken is not installed; cannot count o200k_base tokens"
        ) from exc
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


# A small deterministic word pool. Repeated/truncated to hit a token budget.
# Real-ish prose so the prompt exercises a realistic tokenizer path rather
# than a single repeated token (which speculative decoding would game).
_FILLER_WORDS = (
    "the quick brown fox jumps over the lazy dog while a curious cat watches "
    "from the windowsill as morning light spills across the wooden floor and "
    "somewhere distant a kettle begins to whistle softly in the quiet kitchen "
    "where yesterday's newspaper still lies folded beside an empty coffee cup "
).split()


def _approx_words_for_tokens(target_tokens: int) -> int:
    # o200k_base averages ~0.75 tokens per English word; invert to get words.
    return max(1, round(target_tokens / 0.75))


def generate_prompt_text(
    target_tokens: int,
    *,
    encoding_name: str = O200K_ENCODING,
    use_tiktoken: bool = True,
) -> tuple[str, int, bool]:
    """Build deterministic filler text that tokenizes to ~``target_tokens``.

    Returns ``(text, measured_or_estimated_tokens, used_tiktoken)``. When
    tiktoken is available the text is grown/trimmed until the exact token
    count is hit; otherwise a ~0.75 tokens/word heuristic is used and the
    third tuple element is ``False`` so the caller can warn.
    """
    can_count = use_tiktoken
    if can_count:
        try:
            count_tokens("warmup", encoding_name)
        except TokenizerUnavailable:
            can_count = False

    if not can_count:
        n_words = _approx_words_for_tokens(target_tokens)
        words = [_FILLER_WORDS[i % len(_FILLER_WORDS)] for i in range(n_words)]
        text = " ".join(words)
        return text, target_tokens, False

    # tiktoken available: grow to >= target, then trim by word to land on target.
    words: list[str] = []
    i = 0
    # Grow in chunks to avoid re-encoding on every single word for 100k shapes.
    while count_tokens(" ".join(words), encoding_name) < target_tokens:
        chunk = [_FILLER_WORDS[(i + j) % len(_FILLER_WORDS)] for j in range(64)]
        words.extend(chunk)
        i += 64
    # Trim word-by-word down to the largest prefix that is <= target_tokens,
    # then add one word back so we land at >= target (AA is "approximately").
    while words and count_tokens(" ".join(words), encoding_name) > target_tokens:
        words.pop()
    text = " ".join(words)
    return text, count_tokens(text, encoding_name), True


def generate_aa_dataset(
    shape: AAShape,
    count: int,
    out_path: Path,
    *,
    encoding_name: str = O200K_ENCODING,
    use_tiktoken: bool = True,
) -> dict:
    """Write a ``count``-row replay JSONL for ``shape`` to ``out_path``.

    Each row is an AIPerf ``mooncake_trace`` text-input record:
    ``{"text_input": "<prose>", "output_length": <shape.output_tokens>}``.
    AIPerf sends each ``text_input`` exactly as-is, in file order, so the
    same file replays an identical request sequence across every endpoint.

    Returns a small provenance dict (rows written, token accounting, whether
    the o200k_base tokenizer was actually used).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text, measured, used_tiktoken = generate_prompt_text(
        shape.input_tokens, encoding_name=encoding_name, use_tiktoken=use_tiktoken
    )
    rows = [
        {"text_input": text, "output_length": shape.output_tokens}
        for _ in range(count)
    ]
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")
    return {
        "shape": shape.name,
        "rows": count,
        "target_input_tokens": shape.input_tokens,
        "measured_input_tokens": measured,
        "output_length": shape.output_tokens,
        "tokenizer": encoding_name if used_tiktoken else "heuristic-0.75-tok-per-word",
        "used_tiktoken": used_tiktoken,
        "path": str(out_path),
    }


def build_aiperf_command(
    shape: AAShape,
    *,
    aiperf_cmd: list[str],
    model: str,
    url: str,
    output_artifact_dir: str,
    endpoint: str = "/v1/chat/completions",
    endpoint_type: str = "chat",
    concurrency: int = 1,
    request_count: int = 10,
    api_key: str | None = None,
    tokenizer: str | None = None,
    tokenizer_trust_remote_code: bool = True,
    mode: str = MODE_SYNTHETIC,
    input_file: str | None = None,
    custom_dataset_type: str = DEFAULT_CUSTOM_DATASET_TYPE,
    extra_output_controls: bool = True,
) -> list[str]:
    """Build one ``aiperf profile`` argv for one AA shape at one concurrency.

    Mirrors the original ``repro_artificialanalysis.sh`` flag set. In
    ``synthetic`` mode AIPerf generates the prompt at the token mean; in
    ``dataset-replay`` mode it replays ``input_file`` exactly. The
    ``temperature:0`` / ``top_p:1`` and (optional) ``min_tokens`` /
    ``ignore_eos`` extra-inputs are applied in both modes.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == MODE_DATASET_REPLAY and not input_file:
        raise ValueError("dataset-replay mode requires input_file")

    cmd = list(aiperf_cmd) + [
        "profile",
        "--model",
        model,
        "--endpoint-type",
        endpoint_type,
        "--endpoint",
        endpoint,
        "--streaming",
        "--url",
        url,
    ]
    if api_key:
        cmd += ["--api-key", api_key]
    if tokenizer:
        cmd += ["--tokenizer", tokenizer]
        if tokenizer_trust_remote_code:
            cmd += ["--tokenizer-trust-remote-code"]

    if mode == MODE_SYNTHETIC:
        cmd += [
            "--synthetic-input-tokens-mean",
            str(shape.input_tokens),
            "--synthetic-input-tokens-stddev",
            "0",
            "--output-tokens-mean",
            str(shape.output_tokens),
            "--output-tokens-stddev",
            "0",
        ]
    else:
        cmd += [
            "--input-file",
            input_file,
            "--custom-dataset-type",
            custom_dataset_type,
        ]

    cmd += [
        "--extra-inputs",
        f"temperature:{AA_TEMPERATURE}",
        "--extra-inputs",
        f"top_p:{AA_TOP_P}",
    ]
    if extra_output_controls:
        cmd += [
            "--extra-inputs",
            f"min_tokens:{shape.output_tokens}",
            "--extra-inputs",
            "ignore_eos:true",
        ]

    cmd += [
        "--concurrency",
        str(concurrency),
        "--request-count",
        str(request_count),
        "--output-artifact-dir",
        output_artifact_dir,
    ]
    return cmd


def normalize_outputs(
    cell: CellConfig,
    raw_dir: Path,
    cell_dir: Path,
    *,
    shape: AAShape,
    mode: str,
) -> tuple[list[AtlasCell], str]:
    """Parse per-concurrency AIPerf reports under ``raw_dir`` into AtlasCell rows.

    One AA shape maps to one cell (so the ``(cell_id, concurrency)`` atlas key
    stays unique); the shape + mode are recorded in each row's ``extra`` for
    provenance. Backend is recorded as ``aiperf`` (AA is AIPerf-driven) so the
    AtlasCell schema enum stays unchanged.
    """
    captured_at = utc_now_iso()
    rows: list[AtlasCell] = []
    measured: set[int] = set()

    if raw_dir.is_dir():
        for report_path in sorted(raw_dir.rglob("*.json")):
            parent = report_path.parent.name
            try:
                concurrency = int(parent.rsplit("c", 1)[-1])
            except (ValueError, IndexError):
                continue
            report = _parse_aiperf_json(report_path)
            if not report:
                continue
            metrics = _extract_metrics(report)
            if not metrics:
                continue
            tps_per_gpu = metrics["output_throughput"] / max(cell.tensor_parallel, 1)
            tps_per_user = metrics["output_throughput"] / max(concurrency, 1)
            rows.append(
                AtlasCell(
                    cell_id=cell.cell_id,
                    model=cell.model,
                    hardware=cell.hardware,
                    quant=cell.quant,
                    tensor_parallel=cell.tensor_parallel,
                    parallel_strategy=cell.parallel_strategy,
                    mtp=cell.mtp,
                    max_num_batched_tokens=cell.max_num_batched_tokens,
                    concurrency=concurrency,
                    status=STATUS_FULL,  # provisional; reconciled below
                    ttft_avg_ms=metrics["ttft_ms"],
                    ttfo_avg_ms=metrics.get("ttfo_ms"),
                    reasoning_token_count=metrics.get("reasoning_tokens"),
                    ttfo_coverage=metrics.get("ttfo_coverage"),
                    request_throughput_avg=metrics["request_throughput"],
                    output_tps_per_user=tps_per_user,
                    output_tps_per_gpu=tps_per_gpu,
                    # Promote the AA shape ISL/OSL to the typed analysis fields (not just
                    # extra), and tag the dataset, so the leaderboard/lake ground the
                    # ranking at the real AA shape instead of leaving dataset=unknown.
                    mean_input_tokens=float(shape.input_tokens),
                    mean_output_tokens=float(shape.output_tokens),
                    dataset="aa",
                    bench_backend="aiperf",
                    backend=BACKEND_AIPERF,
                    raw_path=str(report_path.relative_to(cell_dir.parent.parent)),
                    captured_at=captured_at,
                    notes=f"AA {shape.name} ({mode})",
                    extra={
                        "aa_shape": shape.name,
                        "aa_mode": mode,
                        "aa_input_tokens": shape.input_tokens,
                        "aa_output_tokens": shape.output_tokens,
                    },
                )
            )
            measured.add(concurrency)

    requested = set(cell.concurrencies)
    if not measured:
        return [], STATUS_FAILED
    status = STATUS_FULL if measured >= requested else STATUS_PARTIAL
    rows = [AtlasCell(**{**r.to_dict(), "status": status}) for r in rows]
    return rows, status
