#!/usr/bin/env python3
"""WS-C2 corpus builder.

Turn a WS-C1 workload-profile.json into a SpecForge-`conversations` training corpus
matched to the customer's content-class mix + length distribution -- replacing the
generic 40K-UltraChat + 14K-ShareGPT blend with a profile-matched one. This is the
"profile-driven customization" half of the FireOptimizer analog (the documented source
of the higher draft hit-rate).

Output schema (exactly what SpecForge's prepare_data.py emits and run-offline.sh
consumes as DATA_PATH):
    {"id": str, "conversations": [{"role": str, "content": str}, ...]}

Two modes:
  (b) --traffic <redacted-traffic.jsonl>   DIRECT: convert real (redacted) traffic to
      conversations jsonl. Runnable here; no network. Preferred when the operator can
      supply a redacted sample (highest hit-rate, true distribution).
  (a) default (no --traffic)               WEIGHTED-DATASET PLAN: compute per-dataset
      sample sizes proportional to the profile's content-class mix and emit the
      SpecForge prepare_data.py invocations (run on the cluster in the SpecForge env)
      that build the matched mix from the public datasets. Maps classes -> datasets:
        chat -> ultrachat ; code -> opencodeinstruct/codealpaca ; structured-rewrite -> opc

Usage:
  ./profile-to-corpus.py --profile workload-profile.json --out-plan corpus-plan.sh [--total 54000]
  ./profile-to-corpus.py --profile workload-profile.json --traffic redacted.jsonl --out corpus.jsonl
"""
import argparse
import json
import sys

# content-class -> SpecForge prepare_data.py --dataset choices (see that script's
# choices list). Multiple datasets per class are blended evenly.
CLASS_TO_DATASETS = {
    "chat": ["ultrachat", "sharegpt"],
    "code": ["opencodeinstruct", "codealpaca-20k"],
    "structured-rewrite": ["opc"],          # opc = code-edit/diff-flavored
}


def emit_weighted_plan(profile: dict, total: int, out_plan: str, specforge_dir: str,
                       out_data: str):
    mix = profile.get("content_class_mix", {})
    grand = sum(mix.values()) or 1
    lines = [
        "#!/usr/bin/env bash",
        "# corpus-plan.sh -- WS-C2 weighted-dataset plan (auto-generated from a "
        "workload profile).",
        "# Run inside the SpecForge env (e.g. via b1-prepare-data.sbatch's container).",
        "# Produces a profile-matched conversations.jsonl to use as run-offline.sh DATA_PATH.",
        "set -euo pipefail",
        f"SPECFORGE_DIR=${{SPECFORGE_DIR:-{specforge_dir}}}",
        f"OUT=${{OUT:-$(dirname {out_data})}}",
        'mkdir -p "$OUT"',
        'PIP="pip install --no-cache-dir --break-system-packages"',
        '$PIP --no-deps -e "$SPECFORGE_DIR" >/dev/null 2>&1 || true',
        'cd "$SPECFORGE_DIR"',
        "PARTS=()",
    ]
    for cls, count in sorted(mix.items()):
        frac = count / grand
        cls_total = int(round(frac * total))
        datasets = CLASS_TO_DATASETS.get(cls, ["ultrachat"])
        per = max(1, cls_total // len(datasets))
        lines.append(f'echo "[corpus] class={cls} frac={frac:.2f} n~={cls_total} '
                     f'via {",".join(datasets)}"')
        for ds in datasets:
            tag = f"{cls}_{ds}".replace("-", "_")
            lines.append(
                f'python scripts/prepare_data.py --dataset {ds} '
                f'--sample-size {per} --output-path "$OUT" || true'
            )
            # prepare_data.py writes <dataset>_train.jsonl; collect them
            lines.append(f'PARTS+=("$OUT/{ds}_train.jsonl")')
    lines += [
        f'cat "${{PARTS[@]}}" > "{out_data}"',
        f'echo "[corpus] wrote {out_data} lines=$(wc -l < {out_data})"',
    ]
    with open(out_plan, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"== wrote weighted-dataset plan {out_plan} ==")
    print("Profile-matched mix:")
    for cls, count in sorted(mix.items()):
        print(f"  {cls:<20} {count/grand:6.1%}  -> {CLASS_TO_DATASETS.get(cls,['ultrachat'])}")
    print(f"\nRun it in the SpecForge container (cluster). DATA_PATH = {out_data}")


def convert_traffic(profile: dict, traffic: str, out_data: str):
    """Mode (b): direct redacted-traffic -> conversations jsonl."""
    n, skipped = 0, 0
    with open(traffic) as fin, open(out_data, "w") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                skipped += 1
                continue
            convs = []
            if "messages" in rec:
                for m in rec["messages"]:
                    c = m.get("content")
                    if isinstance(c, str) and c:
                        convs.append({"role": m.get("role", "user"), "content": c})
                comp = rec.get("completion") or rec.get("response")
                if isinstance(comp, str) and comp:
                    convs.append({"role": "assistant", "content": comp})
            elif "conversations" in rec:   # already in schema
                convs = rec["conversations"]
            if len(convs) < 2:             # need at least user+assistant to supervise
                skipped += 1
                continue
            fout.write(json.dumps({"id": rec.get("id", f"traffic-{i}"),
                                   "conversations": convs}) + "\n")
            n += 1
    print(f"== wrote {out_data}: {n} conversations ({skipped} skipped) ==")
    if n == 0:
        sys.exit(2)
    print("This is a true profile-matched corpus (real traffic). Use as run-offline.sh "
          "DATA_PATH. NOTE: ensure the traffic was redacted upstream (PII / secrets).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--traffic", default=None, help="redacted traffic jsonl (mode b)")
    ap.add_argument("--out", dest="out_data", default="corpus.jsonl",
                    help="output conversations jsonl (mode b) or final concat path (mode a)")
    ap.add_argument("--out-plan", default="corpus-plan.sh",
                    help="weighted-dataset plan script (mode a)")
    ap.add_argument("--total", type=int, default=54000, help="target corpus size (mode a)")
    ap.add_argument("--specforge-dir", default="/mnt/data/eagle3-train/SpecForge")
    args = ap.parse_args()

    with open(args.profile) as f:
        profile = json.load(f)

    if args.traffic:
        convert_traffic(profile, args.traffic, args.out_data)
    else:
        emit_weighted_plan(profile, args.total, args.out_plan, args.specforge_dir,
                           args.out_data)


if __name__ == "__main__":
    main()
