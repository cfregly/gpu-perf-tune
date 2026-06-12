#!/usr/bin/env bash
# Operator convenience: symlink each plugin skill into ~/.cursor/skills/
# so Cursor sessions auto-load the same SKILL.md files as Claude Code.
#
# This is needed because Cursor does not yet support the Claude Code plugin
# marketplace install path. The SKILL.md files themselves are cross-tool
# compatible (open Agent Skills standard); only the wrapping is different.
#
# Idempotent: re-runs replace existing symlinks; does NOT touch any
# non-symlink directory in ~/.cursor/skills/ (so it won't clobber a hand-
# authored skill the operator already has).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SRC_SKILLS="${REPO_ROOT}/plugins/profile-and-optimize/skills"
DEST_SKILLS="${HOME}/.cursor/skills"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/install-skills-into-cursor.sh [options]

Options:
  --skills-dir PATH    Target Cursor skills directory. Default: ~/.cursor/skills
  --dry-run            Print actions without writing.
  -h, --help           Show this help.

Symlinks each plugin skill (plugins/profile-and-optimize/skills/<name>) into
~/.cursor/skills/<name>. Restart Cursor after running so the new skills
are discovered.

Removes pre-existing symlinks pointing into this repo before re-linking
(idempotent). Refuses to touch non-symlink directories (so a hand-authored
skill with the same name is preserved; you get a warning).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skills-dir) DEST_SKILLS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown arg: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

# Sanity check.
if [[ ! -d "${SRC_SKILLS}" ]]; then
  printf 'FATAL: source skills directory not found: %s\n' "${SRC_SKILLS}" >&2
  exit 2
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf '[dry-run] would ensure %s exists\n' "${DEST_SKILLS}"
else
  mkdir -p "${DEST_SKILLS}"
fi

LINKED=0
SKIPPED=0
FAILED=0

for skill_dir in "${SRC_SKILLS}"/*/; do
  skill_name="$(basename "${skill_dir%/}")"
  dest="${DEST_SKILLS}/${skill_name}"

  if [[ -e "${dest}" || -L "${dest}" ]]; then
    if [[ -L "${dest}" ]]; then
      target="$(readlink "${dest}")"
      if [[ "${target}" == "${skill_dir%/}" ]]; then
        # Already correctly linked.
        printf '  [skip] %s already linked\n' "${skill_name}"
        SKIPPED=$((SKIPPED + 1))
        continue
      else
        # Wrong-target symlink; replace it.
        if [[ "${DRY_RUN}" -eq 1 ]]; then
          printf '  [dry-run] would re-link %s -> %s\n' "${skill_name}" "${skill_dir%/}"
        else
          rm "${dest}"
          ln -s "${skill_dir%/}" "${dest}"
          printf '  [relink] %s\n' "${skill_name}"
          LINKED=$((LINKED + 1))
        fi
        continue
      fi
    else
      # Non-symlink directory or file at that name -- do NOT touch.
      printf '  [WARN] %s exists and is not a symlink (refusing to clobber)\n' "${dest}" >&2
      FAILED=$((FAILED + 1))
      continue
    fi
  fi

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '  [dry-run] would link %s -> %s\n' "${skill_name}" "${skill_dir%/}"
  else
    ln -s "${skill_dir%/}" "${dest}"
    printf '  [link]  %s\n' "${skill_name}"
    LINKED=$((LINKED + 1))
  fi
done

printf '\nSummary: %d linked, %d already-linked (skipped), %d refused (non-symlink conflict)\n' \
  "${LINKED}" "${SKIPPED}" "${FAILED}"

if [[ "${DRY_RUN}" -eq 0 ]]; then
  printf '\nRestart Cursor to discover the new skills.\n'
  printf 'Verify in a Cursor session: the skills appear in the Skills panel.\n'
fi

if [[ "${FAILED}" -gt 0 ]]; then
  printf '\n[WARN] %d skill(s) skipped due to non-symlink conflict. Inspect %s manually.\n' \
    "${FAILED}" "${DEST_SKILLS}" >&2
  exit 1
fi
