#!/usr/bin/env bash
# Manual recovery helper for an unpublished, already-green main version bump.
# Normal publication is owned by .github/workflows/release.yml after CI passes.
set -euo pipefail

python3 scripts/lint-versions.py

version="$(<VERSION)"
tag="v${version}"
remote_tag_ref="refs/release-check/${tag}"
notes_file=""

cleanup() {
  git update-ref -d "${remote_tag_ref}" >/dev/null 2>&1 || true
  if [[ -n "${notes_file}" ]]; then
    rm -f "${notes_file}"
  fi
}
trap cleanup EXIT

if ! command -v gh >/dev/null 2>&1; then
  echo "release: gh is required to publish the GitHub Release." >&2
  exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
  echo "release: gh is not authenticated." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "release: the worktree is not clean. Commit the version bump and changelog first." >&2
  git status --short >&2
  exit 1
fi
if [[ "$(git branch --show-current)" != "main" ]]; then
  echo "release: the current branch is not main." >&2
  exit 1
fi

git fetch --quiet origin main
head_sha="$(git rev-parse HEAD)"
remote_main_sha="$(git rev-parse origin/main)"
if [[ "${head_sha}" != "${remote_main_sha}" ]]; then
  echo "release: local main must exactly match origin/main." >&2
  exit 1
fi

python3 scripts/check-version-transition.py --ref HEAD --require-increase

gh run list \
  --workflow ci.yml \
  --branch main \
  --commit "${head_sha}" \
  --event push \
  --limit 20 \
  --json workflowName,event,headBranch,headSha,status,conclusion,url | \
  python3 scripts/check-ci-status.py "${head_sha}"

assert_tag_ref() {
  local ref="$1"
  if [[ "$(git cat-file -t "${ref}")" != "tag" ]]; then
    echo "release: ${tag} exists but is not annotated." >&2
    return 1
  fi
  if [[ "$(git rev-parse "${ref}^{}")" != "${head_sha}" ]]; then
    echo "release: ${tag} does not point to ${head_sha}." >&2
    return 1
  fi
  if [[ "$(git show "${ref}^{}:VERSION")" != "${version}" ]]; then
    echo "release: ${tag} does not contain VERSION ${version}." >&2
    return 1
  fi
}

fetch_remote_tag() {
  git update-ref -d "${remote_tag_ref}" >/dev/null 2>&1 || true
  if ! git ls-remote --exit-code --tags origin "refs/tags/${tag}" >/dev/null 2>&1; then
    return 1
  fi
  git fetch --quiet --force origin "refs/tags/${tag}:${remote_tag_ref}"
}

sync_local_tag_from_remote() {
  assert_tag_ref "${remote_tag_ref}"
  git update-ref "refs/tags/${tag}" "$(git rev-parse "${remote_tag_ref}")"
  bash scripts/check-release-tag.sh --require
}

if fetch_remote_tag; then
  sync_local_tag_from_remote
  echo "release: verified existing remote tag ${tag}."
else
  if git rev-parse -q --verify "refs/tags/${tag}" >/dev/null 2>&1; then
    assert_tag_ref "refs/tags/${tag}"
  else
    echo "release: creating annotated tag ${tag} at $(git rev-parse --short HEAD)"
    git tag -a "${tag}" -m "${tag}: $(git log -1 --format=%s | sed "s/^${tag}: //")"
  fi
  bash scripts/check-release-tag.sh --require
  if ! git push origin "refs/tags/${tag}"; then
    fetch_remote_tag
    sync_local_tag_from_remote
  fi
  fetch_remote_tag
  sync_local_tag_from_remote
fi

assert_stable_release() {
  local state
  state="$(gh release view "${tag}" \
    --json isDraft,isPrerelease \
    --jq '"\(.isDraft) \(.isPrerelease)"')" || return 1
  if [[ "${state}" != "false false" ]]; then
    echo "release: GitHub Release ${tag} is a draft or prerelease." >&2
    return 1
  fi
}

if gh release view "${tag}" >/dev/null 2>&1; then
  assert_stable_release
else
  notes_file="$(mktemp)"
  make release-notes VERSION="${tag}" >"${notes_file}"
  test -s "${notes_file}"
  if ! gh release create "${tag}" --verify-tag --title "${tag}" \
    --notes-file "${notes_file}"; then
    assert_stable_release
  fi
  assert_stable_release
fi

echo "release: verified stable ${tag} publication."
