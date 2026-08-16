#!/usr/bin/env bash
# claude-hook-adapter.sh - run a Cursor-native beforeShellExecution guard as a
# Claude Code PreToolUse(Bash) hook.
#
# The profile-and-optimize provenance guard speaks the runtime-agnostic contract the
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
payload=""
IFS= read -r -d '' payload || true

# The only automatically registered guard is inert by default. Its off mode is
# a fixed allow decision, so it does not need JSON parsing or jq.
if [[ "${real##*/}" == "provenance-commit-gate.sh" && "${PROVENANCE_COMMIT_GATE:-off}" == "off" ]]; then
  printf '%s\n' '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow","permissionDecisionReason":"provenance commit gate is disabled."}}'
  exit 0
fi

# This fallback contains no caller-controlled text, so it stays valid JSON even
# when jq is unavailable or errors while rendering a verdict.
hard_deny() {
  printf '%s\n' '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"profile-and-optimize guard adapter could not validate hook input or output. Failing closed."}}'
  exit 0
}

command -v jq >/dev/null 2>&1 || hard_deny

event="PreToolUse"

# Emit a Claude PreToolUse verdict. $1=allow|deny|ask  $2=reason.
claude_verdict() {
  local rendered
  if ! rendered="$(jq -nc --arg e "$event" --arg d "$1" --arg r "$2" \
    '{hookSpecificOutput:{hookEventName:$e,permissionDecision:$d,permissionDecisionReason:$r}}' 2>/dev/null)"; then
    hard_deny
  fi
  printf '%s\n' "$rendered"
}

# Claude Bash hooks must provide one JSON object with a nonempty string
# command. Invalid input is not treated as an empty safe command.
if ! jq -e -s '
  length == 1 and
  (.[0] | type == "object") and
  (.[0] as $p |
    (($p.tool_input.command? // $p.command?) | type == "string" and length > 0) and
    (($p.cwd? // $p.tool_input.cwd? // "") | type == "string") and
    (($p.hook_event_name? // "PreToolUse") | type == "string" and length > 0))
' >/dev/null 2>&1 <<<"$payload"; then
  claude_verdict deny "profile-and-optimize guard adapter received invalid hook JSON. Failing closed."
  exit 0
fi

cmd="$(jq -er '.tool_input.command // .command' <<<"$payload" 2>/dev/null)" || hard_deny
cwd="$(jq -er '.cwd // .tool_input.cwd // ""' <<<"$payload" 2>/dev/null)" || hard_deny
event="$(jq -er '.hook_event_name // "PreToolUse"' <<<"$payload" 2>/dev/null)" || hard_deny

# Fail-closed: a missing/unrunnable guard denies.
if [ -z "$real" ] || [ ! -f "$real" ] || [ ! -r "$real" ]; then
  claude_verdict deny "profile-and-optimize guard adapter: guard script not found ($real); failing closed."
  exit 0
fi

cursor_in="$(jq -nc --arg c "$cmd" --arg w "$cwd" '{command:$c,cwd:$w}' 2>/dev/null)" || hard_deny

# Run the guard with the synthesized Cursor input. Fail-closed on guard error.
if ! verdict="$(printf '%s' "$cursor_in" | "$BASH" "$real" 2>/dev/null)"; then
  claude_verdict deny "profile-and-optimize guard (${real##*/}) errored. Failing closed."
  exit 0
fi

# A guard must emit exactly one JSON object with a recognized permission.
# Missing, malformed, or extra output is a denial.
if ! normalized="$(jq -ce -s '
  if length == 1 and
     (.[0] | type == "object") and
     (.[0] as $v |
       (["allow", "deny", "ask"] | index($v.permission)) != null and
       ((($v | has("agent_message")) | not) or ($v.agent_message | type == "string")) and
       ((($v | has("user_message")) | not) or ($v.user_message | type == "string")))
  then .[0]
  else error("invalid guard verdict")
  end
' <<<"$verdict" 2>/dev/null)"; then
  claude_verdict deny "profile-and-optimize guard returned invalid JSON or permission. Failing closed."
  exit 0
fi

perm="$(jq -er '.permission' <<<"$normalized" 2>/dev/null)" || hard_deny
reason="$(jq -er '.agent_message // .user_message // ""' <<<"$normalized" 2>/dev/null)" || hard_deny
claude_verdict "$perm" "$reason"
exit 0
