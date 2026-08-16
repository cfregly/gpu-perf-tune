#!/bin/bash
# Source attribution gate for staged experiment SOURCE.md files.
#
# PROVENANCE_COMMIT_GATE controls enforcement:
#   off  default, always allow
#   ask  request human approval when validation cannot pass
#   deny block the commit when validation cannot pass
#
# An enabled gate audits one checkout. PROVENANCE_REPO_ROOT overrides the
# hook payload cwd. PROVENANCE_AUDIT can override the bundled validator.
# Invalid hook input, missing tooling, and audit errors fail closed whenever
# enforcement is enabled. Non-commit commands still allow immediately.
# Compatible with macOS bash 3.2.
set -uo pipefail

allow() {
  printf '%s\n' '{"permission":"allow"}'
  exit 0
}

deny_invalid_mode() {
  printf '%s\n' '{"permission":"deny","user_message":"provenance gate has an invalid PROVENANCE_COMMIT_GATE value.","agent_message":"Set PROVENANCE_COMMIT_GATE to off, ask, or deny."}'
  exit 0
}

MODE="${PROVENANCE_COMMIT_GATE:-off}"
case "$MODE" in
  off) allow ;;
  ask|deny) ;;
  *) deny_invalid_mode ;;
esac

gate_failure() {
  local user_message="provenance gate could not validate staged experiment source attribution. Add a valid experiment_provenance_v1 block with pinned source commits, or keep the verdict at draft."
  local agent_message="The enabled provenance gate failed closed. Check the hook input, repository target, validator dependencies, and staged SOURCE.md provenance block."
  if [ "$MODE" = "deny" ]; then
    printf '{"permission":"deny","user_message":"%s","agent_message":"%s"}\n' "$user_message" "$agent_message"
  else
    printf '{"permission":"ask","user_message":"%s","agent_message":"%s"}\n' "$user_message" "$agent_message"
  fi
  exit 0
}

command -v python3 >/dev/null 2>&1 || gate_failure
command -v git >/dev/null 2>&1 || gate_failure

input=""
IFS= read -r -d '' input || true

if ! printf '%s' "$input" | python3 -c '
import json
import sys

value = json.load(sys.stdin)
if not isinstance(value, dict):
    raise TypeError("hook input must be an object")
command = value.get("command")
cwd = value.get("cwd", "")
if not isinstance(command, str) or not command:
    raise TypeError("command must be a nonempty string")
if not isinstance(cwd, str):
    raise TypeError("cwd must be a string")
' >/dev/null 2>&1; then
  gate_failure
fi

cmd="$(printf '%s' "$input" | python3 -c 'import json,sys; print(json.load(sys.stdin)["command"], end="")' 2>/dev/null)" || gate_failure
cwd="$(printf '%s' "$input" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("cwd", ""), end="")' 2>/dev/null)" || gate_failure

# Only actual git commit commands need the staged-file audit.
printf '%s' "$cmd" | grep -qE '(^|[^[:alnum:]])git([[:space:]]|[[:space:]].*[[:space:]])commit([[:space:]]|$)' || allow

target="${PROVENANCE_REPO_ROOT:-${cwd:-$PWD}}"
repo="$(git -C "$target" rev-parse --show-toplevel 2>/dev/null)" || gate_failure

if ! staged="$(git -C "$repo" diff --cached --name-only --diff-filter=ACM 2>/dev/null)"; then
  gate_failure
fi

dirs=()
while IFS= read -r path; do
  [ -n "$path" ] || continue
  if [[ "$path" =~ (^|/)(experiments|cluster-probes)/.*/SOURCE\.md$ ]]; then
    dirs+=("${path%/SOURCE.md}")
  fi
done <<<"$staged"

[ "${#dirs[@]}" -gt 0 ] || allow

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || gate_failure
audit="${PROVENANCE_AUDIT:-$script_dir/provenance-audit.py}"
[ -f "$audit" ] && [ -r "$audit" ] || gate_failure
audit_python="${PROFILE_AND_OPTIMIZE_PYTHON:-$script_dir/../server/.venv/bin/python}"
if [ ! -x "$audit_python" ]; then
  audit_python="$(command -v python3)" || gate_failure
fi

if ! "$audit_python" "$audit" --repo-root "$repo" --gate --changed-only "${dirs[@]}" >/dev/null 2>&1; then
  gate_failure
fi

allow
