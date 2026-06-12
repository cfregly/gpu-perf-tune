#!/usr/bin/env python3
"""Reproduce the Artificial Analysis (AA) language-model performance workload shapes.

https://artificialanalysis.ai/methodology/performance-benchmarking

Self-contained port of the original ``repro_artificialanalysis.sh``: no
profile_and_optimize package imports, so it runs on a bench pod or laptop with only
``aiperf`` (or ``uv``) on PATH. Defaults target W&B Inference's
OpenAI-compatible chat endpoint; override via env vars or flags.

Examples:

    # Faithful synthetic run (matches the original bash):
    WANDB_INFERENCE_API_KEY=... python3 repro_artificialanalysis.py

    # Against a self-hosted vLLM, no auth:
    MODEL=meta-llama/Llama-3.1-8B-Instruct URL=http://host:8000 \\
        python3 repro_artificialanalysis.py

    # Generate the reproducible replay dataset once, then replay it:
    python3 repro_artificialanalysis.py --generate-dataset-only
    python3 repro_artificialanalysis.py --mode dataset-replay

Set EXTRA_OUTPUT_CONTROLS=0 (or --no-extra-output-controls) if your provider
rejects vLLM-style ``min_tokens`` / ``ignore_eos``. Without those fields,
output length is a cap rather than an "at least N answer tokens" guarantee.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

# --- AA workload shapes (single source of truth; kept in sync with the
# package module tools/perf_tune_report/runners/aa_workload.py::AA_SHAPES via the
# test_aa_workload.py drift-guard). (input_tokens, output_tokens). ---
AA_SHAPES: dict[str, tuple[int, int]] = {
    "aa-1k": (1000, 1000),
    "aa-10k": (10000, 1500),
    "aa-100k": (100000, 2000),
}
AA_SHAPE_ORDER = ("aa-1k", "aa-10k", "aa-100k")

AA_TEMPERATURE = 0
AA_TOP_P = 1
O200K_ENCODING = "o200k_base"

_FILLER_WORDS = (
    "the quick brown fox jumps over the lazy dog while a curious cat watches "
    "from the windowsill as morning light spills across the wooden floor and "
    "somewhere distant a kettle begins to whistle softly in the quiet kitchen "
    "where yesterday's newspaper still lies folded beside an empty coffee cup "
).split()


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val not in ("0", "false", "False", "")


def resolve_aiperf_cmd(aiperf_bin: str | None) -> list[str]:
    if aiperf_bin:
        return shlex.split(aiperf_bin)
    from shutil import which

    if which("aiperf"):
        # Prefer a real aiperf on PATH (e.g. `uv tool install aiperf`).
        return ["aiperf"]
    if which("uv"):
        # Zero-install fallback: uv fetches aiperf ephemerally (`--with aiperf`).
        return ["uv", "run", "--with", "aiperf", "--python", "3.13", "aiperf"]
    print("error: could not find 'aiperf' or 'uv' on PATH", file=sys.stderr)
    print("Install AIPerf (PyPI package 'aiperf', Python >=3.10), e.g.:", file=sys.stderr)
    print("  uv tool install aiperf            # isolated global CLI", file=sys.stderr)
    print("  pip install aiperf                # inside an active venv", file=sys.stderr)
    print("Or point AIPERF_BIN at the command, for example:", file=sys.stderr)
    print(
        "  AIPERF_BIN='uv run --python 3.13 aiperf' python3 repro_artificialanalysis.py",
        file=sys.stderr,
    )
    sys.exit(127)


def validate_auth(url: str, api_key: str) -> None:
    if "api.inference.wandb.ai" in url and not api_key:
        print(
            "error: WANDB_INFERENCE_API_KEY is not set, and API_KEY was not provided",
            file=sys.stderr,
        )
        print(
            "Run with WANDB_INFERENCE_API_KEY exported or pass API_KEY explicitly.",
            file=sys.stderr,
        )
        sys.exit(2)


def _count_tokens(text: str):
    try:
        import tiktoken
    except ImportError:
        return None
    return len(tiktoken.get_encoding(O200K_ENCODING).encode(text))


def generate_prompt_text(target_tokens: int) -> tuple[str, bool]:
    """Deterministic filler text ~``target_tokens`` o200k_base tokens.

    Returns ``(text, used_tiktoken)``. Falls back to a ~0.75 tok/word
    heuristic (and warns) when tiktoken is unavailable.
    """
    if _count_tokens("warmup") is None:
        n_words = max(1, round(target_tokens / 0.75))
        return " ".join(_FILLER_WORDS[i % len(_FILLER_WORDS)] for i in range(n_words)), False
    words: list[str] = []
    i = 0
    while (_count_tokens(" ".join(words)) or 0) < target_tokens:
        words.extend(_FILLER_WORDS[(i + j) % len(_FILLER_WORDS)] for j in range(64))
        i += 64
    while words and (_count_tokens(" ".join(words)) or 0) > target_tokens:
        words.pop()
    return " ".join(words), True


def generate_dataset(shape: str, count: int, out_path: Path) -> dict:
    in_tokens, out_tokens = AA_SHAPES[shape]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text, used_tiktoken = generate_prompt_text(in_tokens)
    if not used_tiktoken:
        print(
            f"warning: tiktoken not installed; {shape} prompt sized by heuristic "
            "(~0.75 o200k_base tokens/word), not exact token count",
            file=sys.stderr,
        )
    with out_path.open("w", encoding="utf-8") as f:
        for _ in range(count):
            f.write(json.dumps({"text_input": text, "output_length": out_tokens}))
            f.write("\n")
    return {"shape": shape, "rows": count, "path": str(out_path), "used_tiktoken": used_tiktoken}


def build_command(cfg: argparse.Namespace, aiperf_cmd: list[str], shape: str) -> list[str]:
    in_tokens, out_tokens = AA_SHAPES[shape]
    artifact_dir = str(Path(cfg.artifact_root) / shape)
    cmd = list(aiperf_cmd) + [
        "profile",
        "--model",
        cfg.model,
        "--endpoint-type",
        cfg.endpoint_type,
        "--endpoint",
        cfg.endpoint,
        "--streaming",
        "--url",
        cfg.url,
    ]
    if cfg.api_key:
        cmd += ["--api-key", cfg.api_key]
    if cfg.tokenizer:
        cmd += ["--tokenizer", cfg.tokenizer]
        if cfg.tokenizer_trust_remote_code:
            cmd += ["--tokenizer-trust-remote-code"]

    if cfg.mode == "dataset-replay":
        input_file = str(Path(cfg.dataset_dir) / f"{shape}.jsonl")
        cmd += ["--input-file", input_file, "--custom-dataset-type", cfg.custom_dataset_type]
    else:
        cmd += [
            "--synthetic-input-tokens-mean",
            str(in_tokens),
            "--synthetic-input-tokens-stddev",
            "0",
            "--output-tokens-mean",
            str(out_tokens),
            "--output-tokens-stddev",
            "0",
        ]

    cmd += ["--extra-inputs", f"temperature:{AA_TEMPERATURE}", "--extra-inputs", f"top_p:{AA_TOP_P}"]
    if cfg.extra_output_controls:
        cmd += ["--extra-inputs", f"min_tokens:{out_tokens}", "--extra-inputs", "ignore_eos:true"]
    cmd += [
        "--concurrency",
        str(cfg.concurrency),
        "--request-count",
        str(cfg.request_count),
        "--output-artifact-dir",
        artifact_dir,
    ]
    return cmd


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.environ.get("MODEL", "moonshotai/Kimi-K2.6"))
    p.add_argument("--url", default=os.environ.get("URL", "https://api.inference.wandb.ai"))
    p.add_argument(
        "--api-key",
        default=os.environ.get("API_KEY", os.environ.get("WANDB_INFERENCE_API_KEY", "")),
    )
    p.add_argument("--endpoint", default=os.environ.get("ENDPOINT", "/v1/chat/completions"))
    p.add_argument("--endpoint-type", default=os.environ.get("ENDPOINT_TYPE", "chat"))
    p.add_argument("--concurrency", type=int, default=int(os.environ.get("CONCURRENCY", "1")))
    p.add_argument("--request-count", type=int, default=int(os.environ.get("REQUEST_COUNT", "10")))
    p.add_argument(
        "--artifact-root",
        default=os.environ.get("ARTIFACT_ROOT", "artifacts/artificial-analysis"),
    )
    p.add_argument("--tokenizer", default=os.environ.get("TOKENIZER", "") or None)
    p.add_argument("--aiperf-bin", default=os.environ.get("AIPERF_BIN", "") or None)
    p.add_argument(
        "--mode",
        choices=("synthetic", "dataset-replay"),
        default=os.environ.get("MODE", "synthetic"),
    )
    p.add_argument(
        "--custom-dataset-type",
        default=os.environ.get("CUSTOM_DATASET_TYPE", "mooncake_trace"),
    )
    p.add_argument(
        "--dataset-dir",
        default=os.environ.get("DATASET_DIR", "artifacts/aa-dataset"),
        help="Where dataset-replay JSONL files live / are generated.",
    )
    p.add_argument(
        "--shapes",
        default=",".join(AA_SHAPE_ORDER),
        help="Comma-separated subset of AA shapes to run (default: all three).",
    )
    p.add_argument(
        "--generate-dataset-only",
        action="store_true",
        help="Generate the replay JSONL files for --shapes and exit (no benchmark).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands; do not execute.")
    # tokenizer-trust-remote-code defaults on; expose an off switch.
    trc_default = _env_bool("TOKENIZER_TRUST_REMOTE_CODE", True)
    p.add_argument(
        "--no-tokenizer-trust-remote-code",
        dest="tokenizer_trust_remote_code",
        action="store_false",
        default=trc_default,
    )
    eoc_default = _env_bool("EXTRA_OUTPUT_CONTROLS", True)
    p.add_argument(
        "--no-extra-output-controls",
        dest="extra_output_controls",
        action="store_false",
        default=eoc_default,
    )
    p.add_argument(
        "--reasoning",
        action="store_true",
        default=_env_bool("REASONING", False),
        help="Reasoning-model mode (e.g. MiniMax-M2.7, DeepSeek-R1, Qwen3-thinking): "
        "surfaces the methodology guidance loudly. Synthetic filler is ill-posed (the "
        "model reasons about nonsense and emits ~0 answer tokens), so use a real-content "
        "--mode dataset-replay --input-file, and report TTFO (first answer token), not TTFT "
        "(first reasoning token). For the field-agnostic answer latency run assets/aa_ttfo_probe.py.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    cfg = build_arg_parser().parse_args(argv)
    shapes = [s.strip() for s in cfg.shapes.split(",") if s.strip()]
    for s in shapes:
        if s not in AA_SHAPES:
            print(f"error: unknown shape {s!r}; expected {sorted(AA_SHAPES)}", file=sys.stderr)
            return 2

    if cfg.reasoning:
        print(
            "warning: --reasoning set. AA synthetic-filler is ILL-POSED for a reasoning model: "
            "filler input makes it reason about nonsense and emit ~0 answer tokens (measured: "
            "MiniMax-M2.7 aa-1k = ~1064 reasoning tokens, 0 answer). Report TTFO (first answer "
            "token), NOT TTFT (first reasoning token). Use a real-content --mode dataset-replay "
            "--input-file, and run assets/aa_ttfo_probe.py for the field-agnostic answer latency. "
            "Note: MiniMax thinking cannot be disabled (vLLM #36778 / MiniMax #68).",
            file=sys.stderr,
        )
        if cfg.mode == "synthetic":
            print(
                "warning: --reasoning with synthetic mode -> the resulting TTFT will be the "
                "first-reasoning-token time, which is NOT the answer latency.",
                file=sys.stderr,
            )

    if cfg.generate_dataset_only or cfg.mode == "dataset-replay":
        for s in shapes:
            out = Path(cfg.dataset_dir) / f"{s}.jsonl"
            if cfg.dry_run:
                print(f"[dry-run] would generate {cfg.request_count} rows -> {out}")
            else:
                info = generate_dataset(s, cfg.request_count, out)
                print(f"generated {info['rows']} rows -> {info['path']}")
        if cfg.generate_dataset_only:
            return 0

    aiperf_cmd = resolve_aiperf_cmd(cfg.aiperf_bin)
    validate_auth(cfg.url, cfg.api_key)

    rc = 0
    for s in shapes:
        cmd = build_command(cfg, aiperf_cmd, s)
        if cfg.dry_run:
            print(shlex.join(cmd))
            continue
        print(f"=== running {s} (mode={cfg.mode}, concurrency={cfg.concurrency}) ===")
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            print(f"warning: {s} exited {proc.returncode}", file=sys.stderr)
            rc = proc.returncode
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
