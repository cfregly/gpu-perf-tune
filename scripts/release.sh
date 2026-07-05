#!/usr/bin/env bash
# `make release` helper (CLAUDE.md "Git identity + release tagging").
#
# Tags the CURRENT release commit and pushes main + the tag atomically, so the
# vX.Y.Z tag can never be forgotten. Run AFTER the release commit (plugin.json
# version bump + dated CHANGELOG entry) is committed on main.
#
# Identity is the repo-local security@example.com (do NOT override).
set -euo pipefail

PLUGIN_JSON="plugins/profile-and-optimize/.claude-plugin/plugin.json"
ver="$(python3 -c "import json; print(json.load(open('$PLUGIN_JSON'))['version'])")"
tag="v${ver}"

# Gate: version headers consistent before we tag/push.
python3 scripts/lint-versions.py

if [ -n "$(git status --porcelain)" ]; then
  echo "release: working tree not clean -- commit the release (plugin.json bump + CHANGELOG) first." >&2
  git status --short >&2
  exit 1
fi

if git rev-parse -q --verify "refs/tags/${tag}" >/dev/null 2>&1; then
  echo "release: tag ${tag} already exists locally; will push main + existing tag."
else
  echo "release: creating annotated tag ${tag} at $(git rev-parse --short HEAD)"
  git tag -a "${tag}" -m "${tag}: $(git log -1 --format=%s | sed "s/^${tag}: //")"
fi

echo "release: pushing main + ${tag} ..."
git push origin main
git push origin "${tag}"
echo "release: published ${tag}."
