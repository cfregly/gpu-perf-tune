#!/bin/bash
# provenance commit gate (beforeShellExecution) -- Phase 2 source-attribution gate.
#
# On a `git commit`, if a staged experiment-bundle SOURCE.md lacks a VALID
# ```provenance``` block (missing / malformed / a verdict with an unpinned or
# unpushed/dirty source commit), surface it before the result is committed -- the
# commit-time analog of the publish_to_lake --strict source gate.
#
# PHASED ENFORCEMENT: controlled by $PROVENANCE_COMMIT_GATE (off|ask|deny).
#   off  (DEFAULT, Phase 1): inert -- always allow. Flip to `ask`/`deny` AFTER the
#        existing-bundle backlog is backfilled (provenance-audit.sh --backfill).
#   ask  : surface the violation on a native approval card (human decides).
#   deny : refuse the commit.
#
# Fail-OPEN on any parse error / non-commit / tooling-absent -> allow (a hook bug
# must never wedge normal commits). bash 3.2-compatible (macOS /bin/bash).
set -uo pipefail

input="$(cat)"
cmd="$(printf '%s' "$input" | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("command","") or "")
except Exception: print("")' 2>/dev/null)"

allow() { echo '{"permission":"allow"}'; exit 0; }

# Only real `git commit` invocations.
printf '%s' "$cmd" | grep -qE '(^|[^[:alnum:]])git([[:space:]]|[[:space:]].*[[:space:]])commit([[:space:]]|$)' || allow

MODE="${PROVENANCE_COMMIT_GATE:-off}"
[[ "$MODE" == "off" ]] && allow

viol=0
for repo in "$HOME"/the external workspace; do
  git -C "$repo" rev-parse --git-dir >/dev/null 2>&1 || continue
  audit="$repo/provenance-audit.py"
  [[ -f "$audit" ]] || continue
  staged="$(git -C "$repo" diff --cached --name-only --diff-filter=ACM 2>/dev/null \
            | grep -E '(experiments|cluster-probes).*/SOURCE\.md$' || true)"
  [[ -n "$staged" ]] || continue
  dirs="$(printf '%s\n' "$staged" | sed -E 's#/SOURCE\.md$##' | sort -u)"
  # shellcheck disable=SC2086
  if ! python3 "$audit" --repo-root "$repo" --gate --changed-only $dirs >/dev/null 2>&1; then
    viol=1
  fi
done

[[ "$viol" -eq 0 ]] && allow

MSG="provenance gate: a staged experiment bundle has a missing/invalid source-attribution block. Run '*-deploy/capture-provenance.sh <bundle> --write' and pin (commit+push) the source, or set verdict tier=draft. (provenance-audit.sh --gate)"
if [[ "$MODE" == "deny" ]]; then
  printf '{"permission":"deny","user_message":"%s","agent_message":"%s"}\n' "$MSG" "$MSG"
else
  printf '{"permission":"ask","user_message":"%s","agent_message":"%s"}\n' "$MSG" "$MSG"
fi
exit 0
