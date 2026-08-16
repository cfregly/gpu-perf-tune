#!/usr/bin/env bash
# Verify that one exact release commit has its matching annotated version tag.
# This is an explicit release assertion. It is not a pre-push policy hook.
set -euo pipefail

if [[ "${1:-}" == "--require" && "$#" -eq 1 ]]; then
  expected_ref="HEAD"
elif [[ "${1:-}" == "--require-at" && "$#" -eq 2 ]]; then
  expected_ref="$2"
else
  echo "usage: scripts/check-release-tag.sh --require | --require-at COMMIT" >&2
  exit 2
fi

VERSION_FILE="VERSION"
expected_commit="$(git rev-parse "${expected_ref}^{commit}")"
ver="$(git show "${expected_commit}:${VERSION_FILE}")"
tag="v${ver}"

if ! git rev-parse -q --verify "refs/tags/${tag}" >/dev/null 2>&1; then
  echo "RELEASE-TAG GATE: ${tag} is missing for ${expected_commit}." >&2
  exit 1
fi
if [[ "$(git cat-file -t "refs/tags/${tag}")" != "tag" ]]; then
  echo "RELEASE-TAG GATE: ${tag} exists but is not an annotated tag." >&2
  exit 1
fi

tag_commit="$(git rev-parse "refs/tags/${tag}^{}")"
if [[ "${tag_commit}" != "${expected_commit}" ]]; then
  echo "RELEASE-TAG GATE: ${tag} points to ${tag_commit}, expected ${expected_commit}." >&2
  exit 1
fi

tagged_ver="$(git show "${tag_commit}:${VERSION_FILE}")"
if [[ "${tagged_ver}" != "${ver}" ]]; then
  echo "RELEASE-TAG GATE: ${tag} contains VERSION ${tagged_ver}, expected ${ver}." >&2
  exit 1
fi

echo "[ok] annotated ${tag} points to ${expected_commit}"
