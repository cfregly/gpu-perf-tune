#!/usr/bin/env python3
"""WS-C1 workload profiler.

Sample request traffic into a token/shape distribution profile artifact. This is what
makes the adaptive loop "adaptive": the draft is trained against the CUSTOMER's
distribution (the documented source of Fireworks FireOptimizer's "higher hit-rate"),
not a generic UltraChat+ShareGPT mix.

Input: an OpenAI-style access JSONL, one request/response per line. Tolerant of two
common shapes:
  (a) {"messages":[{"role","content"},...], "completion":"...text..."}
  (b) {"prompt_tokens":N, "completion_tokens":M, "content_class":"..."}  (pre-counted)
Lines that have neither are skipped (counted in `skipped`).

Output: workload-profile.json with input/output length distributions, a content-class
mix (chat/code/structured-rewrite heuristic), ISL/OSL shape buckets for the downstream
bench, and a spec-decode method recommendation.

Privacy: this reads potentially-sensitive traffic. By default it stores ONLY
aggregate statistics + class counts, never raw prompt/response text. Pass
--keep-samples N to retain N redacted exemplars for the corpus builder (off by default).

Usage:
  ./workload-profile.py --in access.jsonl --out workload-profile.json [--tokenizer <hf-id>] [--keep-samples 0]
"""
import argparse
import json
import math
import re
import sys
from collections import Counter


def approx_token_count(text: str) -> int:
    """Cheap tokenizer-free estimate (~4 chars/token for English-ish text). Replaced by
    a real HF tokenizer when --tokenizer is given."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def classify(text: str) -> str:
    """Heuristic content class so the corpus builder can match the mix."""
    if not text:
        return "chat"
    code_markers = ("```", "def ", "function ", "class ", "import ", "#include",
                    "{", "};", "</", "SELECT ", "=>")
    n_code = sum(text.count(m) for m in code_markers)
    # structured-rewrite: a request that echoes a large block it wants lightly edited
    rewrite_markers = ("rewrite", "edit", "change every", "replace all", "apply this diff",
                        "modify the following", "refactor")
    is_rewrite = any(m in text.lower() for m in rewrite_markers) and len(text) > 400
    if is_rewrite:
        return "structured-rewrite"
    if n_code >= 3:
        return "code"
    return "chat"


def percentiles(xs, ps=(50, 90, 99)):
    if not xs:
        return {f"p{p}": None for p in ps}
    s = sorted(xs)
    out = {}
    for p in ps:
        k = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
        out[f"p{p}"] = s[k]
    return out


def bucket(n, edges=(256, 1024, 4096, 16384, 65536)):
    for e in edges:
        if n <= e:
            return f"<= {e}"
    return f"> {edges[-1]}"


def recommend_method(class_mix: dict, out_p50: int | None) -> dict:
    """Map the profile to a spec-decode recommendation (the adaptive choice)."""
    total = sum(class_mix.values()) or 1
    rewrite_frac = class_mix.get("structured-rewrite", 0) / total
    code_frac = class_mix.get("code", 0) / total
    if rewrite_frac >= 0.30:
        method = "predicted_outputs"
        why = (f"{rewrite_frac:.0%} structured-rewrite traffic -> output is largely "
               "known a priori; Predicted Outputs (WS-B) gives the biggest win.")
    elif code_frac >= 0.40:
        method = "eagle3"
        why = (f"{code_frac:.0%} code traffic -> high local predictability; a draft head "
               "trained on the code-weighted corpus (EAGLE3) should accept well.")
    else:
        method = "eagle3"
        why = ("general chat-dominant -> EAGLE3 draft head on the profile-matched corpus; "
               "fall back to MTP if the model ships a built-in head.")
    return {"method": method, "rationale": why}


# Artificial Analysis (AA) text shapes: (input_tokens, min_answer_tokens). The three
# canonical AA shapes (see the inference-aa-workload skill). Used as a synthetic
# "profile" when there is no access log -- e.g. to match a draft to AA-style traffic.
AA_SHAPES = [(1024, 1024), (10240, 1536), (102400, 2048)]


def build_aa_profile() -> dict:
    """Synthesize a workload-profile.json from the AA shapes (equal weight). AA is
    long-context chat/reasoning; the matched corpus should weight toward the longest
    trainable sequences (training MAX_LENGTH caps actual train length -- see
    profile-to-corpus.py / run-offline.sh)."""
    in_lens = [s[0] for s in AA_SHAPES]
    out_lens = [s[1] for s in AA_SHAPES]
    n = len(AA_SHAPES)
    # AA text shapes are long-context chat/reasoning, not code/rewrite.
    class_mix = {"chat": n}
    isl_buckets, osl_buckets = Counter(), Counter()
    for il, ol in AA_SHAPES:
        isl_buckets[bucket(il)] += 1
        osl_buckets[bucket(ol)] += 1
    return {
        "n_requests": n,
        "skipped": 0,
        "profile_source": "aa-shapes",
        "tokenizer": "aa-shape-definition",
        "input_tokens": {"mean": sum(in_lens) / n, **percentiles(in_lens)},
        "output_tokens": {"mean": sum(out_lens) / n, **percentiles(out_lens)},
        "content_class_mix": class_mix,
        "isl_buckets": dict(isl_buckets),
        "osl_buckets": dict(osl_buckets),
        "bench_shapes": [list(s) for s in AA_SHAPES],
        "recommended_spec_decode": {
            "method": "eagle3",
            "rationale": "AA long-context chat/reasoning shapes -> EAGLE3 draft head on "
                         "an AA-length-weighted corpus; fall back to the model's built-in "
                         "MTP if a trained head loses the acceptance A/B.",
        },
        "redacted_samples": {},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=None,
                    help="OpenAI-style access JSONL (omit when using --aa-shapes)")
    ap.add_argument("--aa-shapes", action="store_true",
                    help="synthesize the profile from the Artificial Analysis 1k/10k/100k "
                         "shapes instead of an access log")
    ap.add_argument("--out", default="workload-profile.json")
    ap.add_argument("--tokenizer", default=None,
                    help="HF tokenizer id for exact token counts (else ~4 chars/token)")
    ap.add_argument("--keep-samples", type=int, default=0,
                    help="retain N redacted exemplars per class for the corpus builder")
    args = ap.parse_args()

    if args.aa_shapes:
        profile = build_aa_profile()
        with open(args.out, "w") as f:
            json.dump(profile, f, indent=2, default=list)
        print(f"== wrote {args.out} (profile_source=aa-shapes) ==")
        print(json.dumps({k: profile[k] for k in
                          ("bench_shapes", "recommended_spec_decode")}, indent=2))
        print("\nNext: profile-to-corpus.py --profile %s -> AA-length-weighted corpus." % args.out)
        return

    if not args.inp:
        print("FATAL: provide --in <access.jsonl> or --aa-shapes", file=sys.stderr)
        sys.exit(2)

    tok = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.tokenizer)
        except Exception as e:
            print(f"[warn] tokenizer load failed ({e}); using char estimate", file=sys.stderr)

    def count(text):
        if tok is not None and text:
            return len(tok.encode(text))
        return approx_token_count(text)

    in_lens, out_lens = [], []
    class_mix = Counter()
    isl_buckets, osl_buckets = Counter(), Counter()
    samples = {"chat": [], "code": [], "structured-rewrite": []}
    n, skipped = 0, 0

    with open(args.inp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                skipped += 1
                continue
            # shape (b): pre-counted
            if "prompt_tokens" in rec or "completion_tokens" in rec:
                il = int(rec.get("prompt_tokens", 0))
                ol = int(rec.get("completion_tokens", 0))
                cls = rec.get("content_class") or "chat"
            elif "messages" in rec:
                in_text = "\n".join(
                    m.get("content", "") if isinstance(m.get("content"), str) else ""
                    for m in rec["messages"]
                )
                out_text = rec.get("completion") or rec.get("response") or ""
                il, ol = count(in_text), count(out_text)
                cls = classify(in_text)
                if args.keep_samples and len(samples.get(cls, [])) < args.keep_samples:
                    # redact long digit/identifier runs; keep only short exemplars
                    red = re.sub(r"\d{4,}", "<NUM>", in_text[:1200])
                    samples.setdefault(cls, []).append(red)
            else:
                skipped += 1
                continue
            n += 1
            in_lens.append(il)
            out_lens.append(ol)
            class_mix[cls] += 1
            isl_buckets[bucket(il)] += 1
            osl_buckets[bucket(ol)] += 1

    if n == 0:
        print("FATAL: no usable records (need `messages`/`completion` or "
              "`prompt_tokens`/`completion_tokens`)", file=sys.stderr)
        sys.exit(2)

    out_p50 = percentiles(out_lens)["p50"]
    profile = {
        "n_requests": n,
        "skipped": skipped,
        "tokenizer": args.tokenizer or "char-estimate(~4/tok)",
        "input_tokens": {"mean": sum(in_lens) / n, **percentiles(in_lens)},
        "output_tokens": {"mean": sum(out_lens) / n, **percentiles(out_lens)},
        "content_class_mix": dict(class_mix),
        "isl_buckets": dict(isl_buckets),
        "osl_buckets": dict(osl_buckets),
        "bench_shapes": sorted(
            {(percentiles(in_lens)["p50"], percentiles(out_lens)["p50"]),
             (percentiles(in_lens)["p90"], percentiles(out_lens)["p90"])}
        ),
        "recommended_spec_decode": recommend_method(class_mix, out_p50),
        "redacted_samples": samples if args.keep_samples else {},
    }
    with open(args.out, "w") as f:
        json.dump(profile, f, indent=2, default=list)
    print(f"== wrote {args.out} ==")
    print(json.dumps({k: profile[k] for k in
                      ("n_requests", "content_class_mix", "recommended_spec_decode")},
                     indent=2))
    print("\nNext: profile-to-corpus.py --profile %s -> hit-rate-matched corpus." % args.out)


if __name__ == "__main__":
    main()
