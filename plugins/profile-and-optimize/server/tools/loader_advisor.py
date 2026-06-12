#!/usr/bin/env python3
"""Loader auto-select resolver: given a serving tier's constraints, recommend the vLLM weight
loader (hf-pull | runai) + rationale.

This is the SINGLE SOURCE OF TRUTH for the loader decision, shared by:
  - (B) the inference-model-optimize / model-onboarding advisor step (emits LOADER.md + JSON), and
  - (C) the scaffold-model-bringup.sh `--loader auto` resolver (renders the chosen fragment).

Pure-Python, no cluster calls (decisions are structural). Decision tree (kept in lockstep with
docs/inference-fast-model-loading.md "Loader selection for a serving tier"):

  not MTP + runai-image + S3      -> runai       clean-win  (~10min, S3 byte-identical, no HF dep / emptyDir / FUSE)
  not MTP + HF-egress (else)         -> hf-pull     ok         (~12-13min via xet; needs HF egress + 700Gi emptyDir)
  not MTP + S3 only (no runai)    -> none        (s3fs fallback retired 2026-06-09; provide a runai image)
  MTP + HF-egress                    -> hf-pull     clean-win  (MTP-native; no double-stream)
  MTP + runai-image + S3 (no HF)  -> runai+patch tradeoff   (~16min double-stream; S3/no-HF over speed)
  MTP + S3 only (no runai)        -> none        (s3fs fallback retired 2026-06-09; provide a runai image)
  (none of the above)                -> none        (error: provide --hf-egress yes or --s3 yes + a runai image)

Measured basis (GLM-5.1-NVFP4, GB300 TP4, 2026-06-09): hf-pull+xet ~12-13min;
RunAI streamer ~10min (streams S3 at ~2 GB/s); RunAI + MTP breaks without the drafter patch
and then double-streams (~16min). boto3 parallel-multipart (~0.5 GB/s) is dominated by RunAI.
s3fs FUSE was RETIRED 2026-06-09 (~50min single-stream, ~4x slower than hf-pull): the
S3-only + no-runai-image case now returns `none` instead of the slow s3fs fallback.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

# recommended loader -> the k8s deploy fragment the scaffolder renders as experiments/baseline.yaml
LOADER_FRAGMENT = {"hf-pull": "baseline", "runai": "baseline-runai"}


@dataclasses.dataclass
class Gate:
    name: str
    status: str  # pass | fail | info
    detail: str


@dataclasses.dataclass
class LoaderResult:
    recommended: str  # hf-pull | runai | none
    fragment_key: str  # baseline | baseline-runai | ""
    needs_mtp_patch: bool  # runai + MTP requires the runai-mtp-drafter speculative.py patch
    tier: str  # clean-win | ok | tradeoff | fallback | none
    gates: list[Gate]
    reasons: list[str]


def resolve(
    *,
    mtp: bool,
    hf_egress_ok: bool,
    image_has_runai: bool,
    s3_available: bool,
) -> LoaderResult:
    """Encode the loader decision tree. Inputs are booleans knowable from the tier's serve
    config (mtp) + cluster/image facts (the other three)."""
    gates = [
        Gate("mtp", "info",
             "tier uses an MTP / in-checkpoint spec-decode drafter" if mtp
             else "no in-checkpoint spec-decode drafter"),
        Gate("hf_egress", "pass" if hf_egress_ok else "fail",
             "HuggingFace egress available at (re)start" if hf_egress_ok
             else "no HuggingFace egress (air-gap / policy)"),
        Gate("runai_image", "pass" if image_has_runai else "fail",
             "serve image carries runai_model_streamer" if image_has_runai
             else "serve image lacks runai_model_streamer"),
        Gate("s3", "pass" if s3_available else "fail",
             "S3 S3 mirror + creds available" if s3_available
             else "no S3 S3 mirror"),
    ]
    reasons: list[str] = []
    rec, tier, needs_patch = "none", "none", False

    if not mtp:
        if image_has_runai and s3_available:
            rec, tier = "runai", "clean-win"
            reasons.append(
                "non-MTP tier: RunAI streamer streams the S3 byte-identical weights to GPU "
                "(~10min, no HF dependency, no 700Gi emptyDir, no privileged FUSE) -- the clean win "
                "over hf-pull.")
        elif hf_egress_ok:
            rec, tier = "hf-pull", "ok"
            reasons.append(
                "non-MTP but RunAI is unavailable (no runai-capable image and/or no S3 mirror) "
                "-> hf-pull (~12-13min via xet; needs HF egress + a 700Gi emptyDir).")
        # (s3fs FUSE fallback retired 2026-06-09: S3-only + no runai image now -> none)
    else:  # MTP tier
        if hf_egress_ok:
            rec, tier = "hf-pull", "clean-win"
            reasons.append(
                "MTP tier: hf-pull is MTP-native (~12-13min). RunAI needs the drafter patch AND "
                "double-streams the checkpoint (~16min), so hf-pull is preferred when HF egress "
                "is available.")
        elif image_has_runai and s3_available:
            rec, tier, needs_patch = "runai", "tradeoff", True
            reasons.append(
                "MTP + no HF egress -> RunAI streamer + the MTP-drafter patch (S3, no HF "
                "dependency) but ~16min due to the drafter double-stream. Tradeoff: "
                "provenance/air-gap over cold-start speed.")
        # (s3fs FUSE fallback retired 2026-06-09: S3-only + no runai image now -> none)

    if rec == "none":
        reasons.append(
            "No viable loader for these constraints: need at least HF egress, or a S3 mirror "
            "plus a S3-capable loader. Provide --hf-egress yes, or --s3 yes (+ a runai image).")

    return LoaderResult(rec, LOADER_FRAGMENT.get(rec, ""), needs_patch, tier, gates, reasons)


def mtp_from_serve_args(serve_args: str) -> bool:
    """Detect an in-checkpoint spec-decode drafter from the vLLM serve args."""
    a = (serve_args or "").lower()
    return "speculative" in a and any(k in a for k in ("mtp", "eagle", "draft"))


def render_md(r: LoaderResult, mtp: bool) -> str:
    lines = ["# Loader selection", ""]
    if r.recommended == "none":
        lines.append("**RECOMMENDED LOADER: NONE -- no viable loader for these constraints.**")
    else:
        patch = " + MTP-drafter patch" if r.needs_mtp_patch else ""
        lines.append(
            f"**RECOMMENDED LOADER: `{r.recommended}`{patch}  [{r.tier}]**  "
            f"(deploy fragment: `{r.fragment_key}`)")
    lines += ["", "## Inputs / gates"]
    for g in r.gates:
        mark = {"pass": "[pass]", "fail": "[fail]", "info": "[info]"}[g.status]
        lines.append(f"- {mark} **{g.name}** -- {g.detail}")
    lines += ["", "## Rationale"]
    for x in r.reasons:
        lines.append(f"- {x}")
    if r.needs_mtp_patch:
        lines += ["", "## Note",
                  "- The runai+MTP path requires the speculative.py drafter patch "
                  "(the runai-mtp-drafter speculative.py patch)."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Recommend the vLLM weight loader for a serving tier.")
    p.add_argument("--mtp", dest="mtp", action="store_true", help="tier uses an MTP/spec-decode drafter")
    p.add_argument("--no-mtp", dest="mtp", action="store_false")
    p.set_defaults(mtp=None)
    p.add_argument("--serve-args", default="",
                   help="vLLM serve args; auto-detects MTP when --mtp/--no-mtp not given")
    p.add_argument("--hf-egress", choices=["yes", "no"], default="yes")
    p.add_argument("--image-has-runai", choices=["yes", "no"], default="yes")
    p.add_argument("--s3", choices=["yes", "no"], default="yes")
    p.add_argument("--emit", default="", help="directory to write loader_advisor.json + LOADER.md")
    p.add_argument("--print-fragment", action="store_true",
                   help="print only the fragment_key (for the scaffolder) and exit")
    args = p.parse_args(argv)

    mtp = args.mtp if args.mtp is not None else mtp_from_serve_args(args.serve_args)
    r = resolve(
        mtp=mtp,
        hf_egress_ok=args.hf_egress == "yes",
        image_has_runai=args.image_has_runai == "yes",
        s3_available=args.s3 == "yes",
    )

    if args.print_fragment:
        print(r.fragment_key or "NONE")
        return 0 if r.recommended != "none" else 2

    md = render_md(r, mtp)
    sys.stdout.write(md)
    if args.emit:
        d = Path(args.emit)
        d.mkdir(parents=True, exist_ok=True)
        payload = dataclasses.asdict(r)
        payload["mtp"] = mtp
        (d / "loader_advisor.json").write_text(json.dumps(payload, indent=2) + "\n")
        (d / "LOADER.md").write_text(md)
        sys.stdout.write(f"\nwrote {d / 'loader_advisor.json'} + {d / 'LOADER.md'}\n")
    return 0 if r.recommended != "none" else 2


if __name__ == "__main__":
    raise SystemExit(main())
