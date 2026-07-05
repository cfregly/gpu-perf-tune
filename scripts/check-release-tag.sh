#!/usr/bin/env bash
# Tagging-rigidity pre-push gate (CLAUDE.md "Git identity + release tagging").
#
# Wired in .pre-commit-config.yaml as a `stages: [pre-push]` local hook; enable with
#   pre-commit install --hook-type pre-push
#
# A release == a plugin.json version bump. This blocks a push when the current
# plugin.json version has no matching annotated vX.Y.Z tag. Non-release commits
# (version unchanged) pass, because the prior release's tag already exists.
set -euo pipefail

PLUGIN_JSON="plugins/profile-and-optimize/.claude-plugin/plugin.json"
[ -f "$PLUGIN_JSON" ] || { echo "check-release-tag: $PLUGIN_JSON not found" >&2; exit 1; }

ver="$(python3 -c "import json,sys; print(json.load(open('$PLUGIN_JSON'))['version'])")"

if git rev-parse -q --verify "refs/tags/v${ver}" >/dev/null 2>&1; then
  exit 0
fi

cat >&2 <<EOF
RELEASE-TAG GATE: plugin.json version is ${ver} but no annotated tag v${ver} exists.
Tagging rigidity (CLAUDE.md "Git identity + release tagging"): every release MUST be
tagged. Create + push it (or run 'make release'):
  git tag -a v${ver} -m "v${ver}: <summary>" && git push origin v${ver}
EOF
exit 1
