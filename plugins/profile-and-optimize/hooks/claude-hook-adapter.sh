#!/usr/bin/env bash
# claude-hook-adapter.sh - run a Cursor-native beforeShellExecution guard as a
# Claude Code PreToolUse(Bash) hook.
#
# The profile-and-optimize guards (perflake-teardown-gate, provenance-commit-gate)
# speak the runtime-agnostic contract the
# hooks/README.md documents: stdin {"command":...[,"cwd":...]} -> stdout
# {"permission":"allow|deny|ask"[,"user_message","agent_message"]}. Claude Code's
# PreToolUse hook instead sends {"tool_input":{"command":...},"hook_event_name":
# "PreToolUse","cwd":...} and consumes {"hookSpecificOutput":{"hookEventName",
# "permissionDecision","permissionDecisionReason"}}.
#
# This adapter bridges the two without touching the guards: it projects the Claude
# payload to the Cursor shape, runs the guard unchanged, and translates the
# guard's verdict back to Claude's schema (permission -> permissionDecision;
# allow/deny/ask map 1:1). Fail-closed: a missing or erroring guard denies.
#
# Usage (from hooks/hooks.json): claude-hook-adapter.sh <path-to-guard.sh>
set -uo pipefail

real="${1:-}"
payload="$(cat)"

cmd="$(jq -r '.tool_input.command // .command // ""' <<<"$payload" 2>/dev/null || echo "")"
cwd="$(jq -r '.cwd // .tool_input.cwd // ""' <<<"$payload" 2>/dev/null || echo "")"
event="$(jq -r '.hook_event_name // "PreToolUse"' <<<"$payload" 2>/dev/null || echo "PreToolUse")"

# Emit a Claude PreToolUse verdict. $1=allow|deny|ask  $2=reason.
claude_verdict() {
  jq -nc --arg e "$event" --arg d "$1" --arg r "$2" \
    '{hookSpecificOutput:{hookEventName:$e,permissionDecision:$d,permissionDecisionReason:$r}}' 2>/dev/null \
    || printf '{"hookSpecificOutput":{"hookEventName":"%s","permissionDecision":"%s","permissionDecisionReason":"%s"}}\n' "$event" "$1" "$2"
}

# Fail-closed: a missing/unrunnable guard denies.
if [ -z "$real" ] || [ ! -f "$real" ]; then
  claude_verdict deny "profile-and-optimize guard adapter: guard script not found ($real); failing closed."
  exit 0
fi

cursor_in="$(jq -nc --arg c "$cmd" --arg w "$cwd" '{command:$c,cwd:$w}' 2>/dev/null || printf '{"command":"%s"}' "$cmd")"

# Run the guard with the synthesized Cursor input. Fail-closed on guard error.
if ! verdict="$(printf '%s' "$cursor_in" | bash "$real" 2>/dev/null)"; then
  claude_verdict deny "profile-and-optimize guard ($(basename "$real")) errored; failing closed."
  exit 0
fi

perm="$(jq -r '.permission // "allow"' <<<"$verdict" 2>/dev/null || echo "allow")"
reason="$(jq -r '.agent_message // .user_message // ""' <<<"$verdict" 2>/dev/null || echo "")"
claude_verdict "$perm" "$reason"
exit 0
