#!/bin/bash
# perf-lake teardown gate (beforeShellExecution) -- SMART.
# Asks ONLY when an experiment teardown would destroy an UNPUBLISHED bundle's evidence.
# For a matched teardown (kubectl delete with an experiment= label / experiment-named
# pods, scancel, helm uninstall) it resolves the owning evidence bundle and either:
#   - ALLOWS silently if the bundle is already PUBLISHED to the perf-lake or carries a
#     `perf-lake: intentional-gap` / `perf-lake: published` marker (safe to tear down), OR
#   - ASKS (fail-safe) if the bundle is genuinely unpublished, cannot be resolved to a
#     single bundle, or the teardown has no resolvable experiment key (bare helm
#     uninstall / scancel).
# Local + fast: matches the recorded `experiment=<slug>` label (or the exact run-id dir
# name) in SOURCE.md/summary.md, OR the deploy manifest (commands/*.yaml `experiment:` /
# `experiment=<slug>`), so a deploy whose label != its evidence-bundle run-id still resolves
# to its owning bundle; never hits the network. (Canonical grammar still wants the deploy
# `experiment=` label == the run-id, recorded as lineage.cluster_objects[] in SOURCE.md --
# see docs/METHODOLOGY.md; the manifest resolution here is the safety net
# for when they diverge, e.g. ffg-qwen330b-* vs qwen3-30b-bf16-roofline-*.) The slug is read
# from the command text including the `experiment="$VAR"` / `$VAR` / `${VAR}` idiom (the
# VAR=<literal> assignment is resolved from the same command block). The sessionEnd reminder +
# perf-lake-coverage-audit.sh remain the authoritative S3 backstop.
# Fail-OPEN on a non-teardown / parse error -> allow (never wedge normal work);
# fail-SAFE on an unresolvable teardown -> ask (never silently destroy evidence).
# bash 3.2-compatible (macOS /bin/bash): no mapfile / ${var,,} / associative arrays.
set -uo pipefail

input="$(cat)"
cmd="$(printf '%s' "$input" | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("command","") or "")
except Exception: print("")' 2>/dev/null)"

# Not a destructive teardown of an EXPERIMENT? allow immediately.
# Triggers: scancel | helm uninstall/delete | kubectl delete ... (experiment= | -l experiment | glm51-e3bs).
if ! printf '%s' "$cmd" | grep -qiE '(\bscancel\b)|(helm[[:space:]]+(uninstall|delete)\b)|(kubectl[^|]*\bdelete\b[^|]*(experiment=|glm51-e3bs|-l[[:space:]]+experiment))'; then
  echo '{"permission":"allow"}'
  exit 0
fi

# ask <user_message> <agent_message>  (args must be pre-sanitized of " and \).
ask() {
  printf '{"permission":"ask","user_message":"%s","agent_message":"%s"}\n' "$1" "$2"
  exit 0
}

# ---- extract the experiment slug (label value, lowercased) ----------------------------
# Resolve the slug from the raw command text (the hook never expands the shell). Three
# forms, in priority order, via python3 (already a hard dep above; avoids bash 3.2 ERE
# quoting traps around " ' $ { }):
#   (1) literal          -l experiment=<slug>
#   (2) shell variable   -l experiment="$VAR" | $VAR | ${VAR}  -> resolve a VAR=<literal>
#                        assignment in the SAME command block
#   (3) glm51-e3bs       serve-name fallback
# Unresolvable (e.g. $VAR assigned outside the block) -> empty -> fail-safe ask below.
slug="$(printf '%s' "$cmd" | python3 -c '
import sys, re
cmd = sys.stdin.read()
slug = ""
m = re.search(r"experiment=([A-Za-z0-9._-]+)", cmd)
if m:
    slug = m.group(1)
else:
    m = re.search(r"experiment=\W*\$\{?([A-Za-z_][A-Za-z0-9_]*)", cmd)
    if m:
        var = m.group(1)
        m2 = re.search(r"(?:^|\W)" + re.escape(var) + r"=\W*([A-Za-z0-9._-]+)", cmd)
        if m2:
            slug = m2.group(1)
    if not slug:
        m = re.search(r"(glm51-e3bs[A-Za-z0-9._-]*)", cmd)
        if m:
            slug = m.group(1)
print(slug.lower())
' 2>/dev/null)"

if [[ -z "$slug" ]]; then
  ask "perf-lake gate: experiment teardown detected, but there is no experiment= key to confirm it is published. Verify it is published (./perf-lake-coverage-audit.sh) or intentional-gap-marked before destroying emptyDir evidence." \
      "perf-lake teardown gate: no resolvable experiment slug (bare helm uninstall / scancel). Fail-safe ask; confirm the bundle is published or intentional-gap-marked, then proceed."
fi

# ---- bundle roots (only those that exist) ---------------------------------------------
ROOTS=()
# Search each deploy-bundle repo ROOT recursively. The resolvers below scope rg/find to
# SOURCE.md / *.yaml, so the recursion finds bundles under ANY layout -- experiments/artifacts,
# cluster-probes, the perf-tune-report/campaigns mirror, AND non-standard topic dirs like
# perf-tune-deepseek-v4-flash/vllm-v4-patch, spec-decode/, quantize/. Keying on the
# {experiments,cluster-probes} subdirs alone missed those (the dsv4f vllm-v4-patch hangs).
for base in "$PWD"/campaigns; do
  [[ -d "$base" ]] && ROOTS+=("$base")
done

ASK_UNRES_U="perf-lake gate: experiment teardown of '$slug' -- could not confirm a single published bundle. Verify it is published (./perf-lake-coverage-audit.sh) or intentional-gap-marked before destroying emptyDir evidence."
ASK_UNRES_A="perf-lake teardown gate: experiment=$slug did not resolve to exactly one owning bundle (not found / ambiguous / search unavailable). Fail-safe ask; confirm published or intentional-gap-marked."

if [[ "${#ROOTS[@]}" -eq 0 ]]; then
  ask "$ASK_UNRES_U" "$ASK_UNRES_A"
fi

# ---- search backends (rg preferred; grep/find fallback; short timeout) -----------------
TO=()
if command -v timeout >/dev/null 2>&1; then TO=(timeout 5)
elif command -v gtimeout >/dev/null 2>&1; then TO=(gtimeout 5); fi

have_rg=0; command -v rg >/dev/null 2>&1 && have_rg=1

find_label() {  # files (SOURCE.md/summary.md) that DECLARE experiment=<slug>
  if [[ "$have_rg" -eq 1 ]]; then
    ${TO[@]+"${TO[@]}"} rg -l -i -F "experiment=$slug" --no-ignore -g 'SOURCE.md' -g 'summary.md' "${ROOTS[@]}" 2>/dev/null
  else
    ${TO[@]+"${TO[@]}"} grep -RIliF --include=SOURCE.md --include=summary.md -e "experiment=$slug" "${ROOTS[@]}" 2>/dev/null
  fi
}
list_sourcemd() {  # every SOURCE.md under the roots (for the exact run-id dir-name match)
  if [[ "$have_rg" -eq 1 ]]; then
    ${TO[@]+"${TO[@]}"} rg --files --no-ignore -g 'SOURCE.md' "${ROOTS[@]}" 2>/dev/null
  else
    ${TO[@]+"${TO[@]}"} find "${ROOTS[@]}" -type f -name SOURCE.md 2>/dev/null
  fi
}
find_label_yaml() {  # deploy manifests (commands/*.yaml) that DECLARE experiment:/experiment=<slug>
  if [[ "$have_rg" -eq 1 ]]; then
    ${TO[@]+"${TO[@]}"} rg -l -i -e "experiment:[[:space:]]*$slug" -e "experiment=$slug" --no-ignore -g '*.yaml' -g '*.yml' "${ROOTS[@]}" 2>/dev/null
  else
    ${TO[@]+"${TO[@]}"} grep -RIli --include=*.yaml --include=*.yml -e "experiment:[[:space:]]*$slug" -e "experiment=$slug" "${ROOTS[@]}" 2>/dev/null
  fi
}

# ---- resolve owning bundle dir(s) -----------------------------------------------------
cands=()
# (1) the bundle that DECLARES `experiment=<slug>` (precise: the label assignment, not a
#     prose / path cross-reference).
while IFS= read -r f; do
  [[ -n "$f" ]] && cands+=("$(dirname "$f")")
done < <(find_label)
# (2) the bundle whose dir basename == slug (full run-id slugs whose SOURCE.md may not write
#     the label literally). Pure-bash basename + case-insensitive compare (shopt nocasematch):
#     avoids spawning dirname/basename/tr PER FILE, which is ~9s over the ~900 SOURCE.md the
#     repo-root ROOTS now enumerate (slug is already lowercased; basenames are mixed-case).
shopt -s nocasematch
while IFS= read -r f; do
  [[ -n "$f" ]] || continue
  d="${f%/*}"          # dirname
  bl="${d##*/}"        # basename
  [[ "$bl" == "$slug" ]] && cands+=("$d")
done < <(list_sourcemd)
shopt -u nocasematch
# (3) the bundle whose deploy manifest (commands/*.yaml) DECLARES experiment:/experiment=<slug>
#     -- resolves a deploy whose label != its evidence-bundle run-id (the canonical isolation
#     grammar wants them equal, but this is the safety net when they diverge). Map each yaml
#     back to its owning bundle = the nearest ancestor dir that contains a SOURCE.md.
while IFS= read -r f; do
  [[ -n "$f" ]] || continue
  d="$(dirname "$f")"
  while [[ "$d" != "/" && ! -f "$d/SOURCE.md" ]]; do d="$(dirname "$d")"; done
  [[ -f "$d/SOURCE.md" ]] && cands+=("$d")
done < <(find_label_yaml)

# distinct bundle dirs
bundles=()
while IFS= read -r line; do
  [[ -n "$line" ]] && bundles+=("$line")
done < <(printf '%s\n' ${cands[@]+"${cands[@]}"} | awk 'NF' | sort -u)

# ---- safe-marker check (mirrors perf-lake-coverage-audit.py + organic publish evidence) -
# intentional-gap | explicit published marker | s3://perf-lake/ path | campaign=<run-id stamp>
# | published_at_utc -- all are written ONLY by publish_to_lake AFTER a real publish.
# A run-id legitimately maps to >1 dir: the evidence bundle AND its
# perf-tune-report/campaigns/<run-id> mirror (created by campaign_init / publish_to_lake). So do
# NOT bail on multiplicity -- classify each resolved bundle independently (like the audit's
# per-bundle pass) and ALLOW if ANY is published/gap-marked (the lake then holds the
# evidence -> safe to tear down). Only a wholly unmarked set falls through to ask.
MARKER_RE='perf-lake:[[:space:]]*intentional-gap|perf-lake:[[:space:]]*published|s3://perf-lake/|campaign=[A-Za-z0-9._-]*[0-9]{8}T[0-9]{6}Z|published_at_utc'

# 0 resolved -> genuinely unresolvable -> fail-safe ask.
if [[ "${#bundles[@]}" -eq 0 ]]; then
  ask "$ASK_UNRES_U" "$ASK_UNRES_A"
fi

# ---- per-arm roofline check (additive; runs BEFORE the publish-marker allow) ----------
# Even a PUBLISHED campaign can have a variant whose roofline was never captured
# (the per-arm publish gate bites only on re-render / new publishes, so the
# already-published fleet may be per-arm-incomplete). Tearing down the live
# deploy is the only chance to capture that arm's roofline before its emptyDir
# evidence is destroyed. So if a resolved bundle's perf-report campaign has
# sol_per_arm_complete=false and an uncovered arm is NOT declared in the
# campaign config.yaml roofline_gap_arms allowlist, ASK. Fail-OPEN on any
# missing-file / parse error (the python exits 0 -> empty -> no ask) so a normal
# teardown is never wedged; declaring roofline_gap_arms (B200/MTP/overlay-gone)
# silences it for genuine gaps.
CAMPAIGNS_DIR="./campaigns"
for b in "${bundles[@]}"; do
  rs=""
  if [[ -f "$b/report_status.json" ]]; then
    rs="$b/report_status.json"
  elif [[ -f "$CAMPAIGNS_DIR/$(basename "$b")/report_status.json" ]]; then
    rs="$CAMPAIGNS_DIR/$(basename "$b")/report_status.json"
  fi
  [[ -n "$rs" ]] || continue
  missing="$(RS="$rs" python3 -c '
import json, os, sys
try:
    rs = os.environ["RS"]
    d = json.load(open(rs))
except Exception:
    sys.exit(0)  # fail-open: cannot read status -> no per-arm ask
# per-arm roofline coverage applies ONLY to throughput/mixed campaigns (sol-rigor rule #6,
# matching publish_to_lake --strict scope); accuracy/latency campaigns (e.g. gpqa eval cells)
# have no per-arm rooflines -> never gate their teardown here.
if str(d.get("focus", "")).strip().lower() not in ("throughput", "mixed"):
    sys.exit(0)
if d.get("sol_per_arm_complete", True):
    sys.exit(0)
unc = d.get("arms_uncovered") or []
gap = set()
cfg = os.path.join(os.path.dirname(rs), "config.yaml")
try:
    import yaml
    y = yaml.safe_load(open(cfg).read()) or {}
    blk = y.get("roofline_gap_arms")
    if isinstance(blk, dict):
        gap = {str(k) for k in blk}
    elif isinstance(blk, (list, tuple)):
        gap = {str(a) for a in blk}
except Exception:
    gap = set()  # cannot read opt-out -> treat undeclared arms as missing (fail-safe ask)
miss = [str(a) for a in unc if str(a) not in gap]
print(",".join(miss))
' 2>/dev/null)"
  if [[ -n "$missing" ]]; then
    bsan="${b//\\/}"; bsan="${bsan//\"/}"
    msan="${missing//\\/}"; msan="${msan//\"/}"
    ask "perf-lake gate: experiment teardown of '$bsan' -- its perf-report campaign has arm(s) with NO roofline ($msan) and this live deploy is the only chance to capture them. Run perftune-specdec/profiling/roofline-sweep.sh + perftunereport import_roofline_sweep for each, then re-render -- OR declare each genuinely un-capturable arm in the campaign config.yaml roofline_gap_arms: {<arm>: <reason>}, then re-run." \
        "per-arm roofline gate: campaign for $bsan is per-arm-INCOMPLETE (arms missing a roofline: $msan, not in roofline_gap_arms). Capture them before teardown, or declare them as roofline_gap_arms (B200 / MTP-engine-blocked / overlay-gone)."
  fi
done

# ANY resolved bundle carries a publish/gap marker -> evidence is in the lake -> allow.
for b in "${bundles[@]}"; do
  if grep -qiE "$MARKER_RE" "$b/SOURCE.md" "$b/summary.md" 2>/dev/null; then
    echo '{"permission":"allow"}'
    exit 0
  fi
done

# ---- resolved but NONE published/gap-marked -> ask ------------------------------------
if [[ "${#bundles[@]}" -eq 1 ]]; then
  bundle="${bundles[0]}"
  bsan="${bundle//\\/}"; bsan="${bsan//\"/}"
  ask "perf-lake gate: experiment teardown of '$bsan' -- this bundle has NO perf-lake publish or intentional-gap marker and its emptyDir evidence is lost on delete. Publish it (perftunereport publish_to_lake; ./perf-lake-coverage-audit.sh to check) OR add 'perf-lake: intentional-gap - <reason>' to its SOURCE.md, then re-run." \
      "perf-lake teardown gate: bundle $bsan is UNPUBLISHED (no s3://perf-lake/, campaign=<runid>, 'perf-lake: published', or 'perf-lake: intentional-gap' marker in SOURCE.md/summary.md). Publish or gap-mark before tearing down. Resolved from experiment=$slug."
else
  ask "$ASK_UNRES_U" "$ASK_UNRES_A"
fi
