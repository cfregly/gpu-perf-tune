#!/usr/bin/env python3
"""Dependency-free AA-shape streaming probe: AIPerf-faithful TTFT vs TTFO for reasoning models.

For a reasoning model (e.g. MiniMax-M2.7 with --reasoning-parser, DeepSeek-R1, Qwen3) the
model thinks before it answers. AIPerf reports two distinct first-token metrics, and capturing
only TTFT under-reports answer latency by the entire think phase:

  TTFT = time to the first token of ANY type, INCLUDING reasoning. (docs.nvidia.com/aiperf)
  TTFO = time to the first NON-reasoning (answer/content) token.
  For a non-reasoning model, TTFT == TTFO.

This probe measures both directly off the raw SSE stream, FIELD-AGNOSTICALLY: it counts a
reasoning token from `delta.reasoning` OR `delta.reasoning_content` (the field name differs by
vLLM version: 0.20 uses reasoning_content, 0.22 uses reasoning; Dynamo exposes neither), and an
answer token from `delta.content`. That is the only cross-stack-comparable way to get the real
answer latency on a reasoning model, because AIPerf's own TTFO split depends on whether the stack
surfaces a reasoning field it recognizes.

Why this exists: AA's synthetic-filler + min_tokens shape is ill-posed for a reasoning model
(filler input makes it reason for the whole budget and emit 0 answer tokens). For reasoning models
prefer dataset-replay (real-content) prompts AND report TTFO, not TTFT. See the skill's
"Reasoning models" section.

No PyPI / uv / aiperf needed (urllib only), so it runs from inside a vllm pod:
  python3 aa_ttfo_probe.py --url http://localhost:8000/v1/chat/completions \
      --model <served-name> --shape aa-1k --count 5
"""
import argparse, json, statistics, sys, time, urllib.request

# (input_tokens, min_answer_tokens) per the AA "Workload Types" table.
SHAPES = {"aa-1k": (1000, 1000), "aa-10k": (10000, 1500), "aa-100k": (100000, 2000)}
FILLER = (
    "the quick brown fox jumps over the lazy dog while a curious cat watches "
    "from the windowsill as morning light spills across the wooden floor and "
    "somewhere distant a kettle begins to whistle softly in the quiet kitchen "
    "where yesterday's newspaper still lies folded beside an empty coffee cup "
).split()


def prompt_for(in_tokens):
    n = max(1, round(in_tokens / 0.75))  # ~0.75 o200k_base tok/word
    return " ".join(FILLER[i % len(FILLER)] for i in range(n))


def run_one(url, model, prompt, min_tokens, extra_inputs=None):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "top_p": 1, "stream": True,
        "max_tokens": min_tokens + 64, "min_tokens": min_tokens, "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }
    if extra_inputs:
        body.update(extra_inputs)
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    ttft = ttfo = None
    rtok = ctok = 0
    usage = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            now = time.perf_counter()
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            # FIELD-AGNOSTIC: reasoning streams as delta.reasoning (vllm 0.22) OR
            # delta.reasoning_content (vllm 0.20); the answer is always delta.content.
            rc = delta.get("reasoning") or delta.get("reasoning_content")
            cc = delta.get("content")
            if ttft is None and (rc or cc):
                ttft = now - t0
            if rc:
                rtok += 1
            if cc:
                if ttfo is None:
                    ttfo = now - t0
                ctok += 1
    total = time.perf_counter() - t0
    return {
        "ttft_ms": round((ttft or 0) * 1000, 1),
        "ttfo_ms": round((ttfo or 0) * 1000, 1) if ttfo is not None else None,
        "reasoning_tokens": rtok,
        "content_tokens": ctok,
        "total_s": round(total, 2),
        "usage": usage,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--model", required=True, help="served-model-name")
    ap.add_argument("--shape", default="aa-1k", choices=list(SHAPES))
    ap.add_argument("--count", type=int, default=5)
    a = ap.parse_args()
    in_tok, min_out = SHAPES[a.shape]
    prompt = prompt_for(in_tok)
    rows = []
    for i in range(a.count):
        r = run_one(a.url, a.model, prompt, min_out)
        rows.append(r)
        print(f"[{a.shape} #{i + 1}] " + json.dumps(r), flush=True)
    fin = [r for r in rows if r["ttft_ms"] > 0]

    def p50(key):
        vals = [r[key] for r in fin if r.get(key) is not None]
        return round(statistics.median(vals), 1) if vals else None

    decode = [(r["content_tokens"] + r["reasoning_tokens"]) / r["total_s"]
              for r in fin if r["total_s"] > 0] or [0]
    summary = {
        "shape": a.shape, "n": len(fin),
        "ttft_p50_ms": p50("ttft_ms"),
        "ttfo_p50_ms": p50("ttfo_ms"),  # None if the model never emitted an answer token (all-reasoning)
        "reasoning_tokens_p50": p50("reasoning_tokens"),
        "content_tokens_p50": p50("content_tokens"),
        "decode_tok_s_per_user_mean": round(statistics.mean(decode), 1),
        "ttfo_minus_ttft_p50_ms": (round(p50("ttfo_ms") - p50("ttft_ms"), 1)
                                   if p50("ttfo_ms") and p50("ttft_ms") else None),
    }
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    sys.exit(main())
