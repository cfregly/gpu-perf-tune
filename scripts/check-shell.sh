#!/usr/bin/env bash
# Run ShellCheck with the contributor venv when available.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_SHELLCHECK="${REPO_ROOT}/plugins/profile-and-optimize/server/.venv/bin/shellcheck"

if [[ -x "${VENV_SHELLCHECK}" ]]; then
  checker="${VENV_SHELLCHECK}"
elif command -v shellcheck >/dev/null 2>&1; then
  checker="$(command -v shellcheck)"
else
  printf '%s\n' \
    '[FAIL] ShellCheck is not installed. Run: bash plugins/profile-and-optimize/server/install.sh --with-dev' >&2
  exit 2
fi

if (( $# > 0 )); then
  exec "${checker}" "$@"
fi

files=()
while IFS= read -r -d '' path; do
  files+=("${path}")
done < <(git -C "${REPO_ROOT}" ls-files -z '*.sh')

if (( ${#files[@]} == 0 )); then
  exit 0
fi

cd "${REPO_ROOT}"
exec "${checker}" "${files[@]}"
