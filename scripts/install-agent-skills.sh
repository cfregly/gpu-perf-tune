#!/usr/bin/env bash
# Install the shared Agent Skills surface for a supported client.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SRC_SKILLS="${REPO_ROOT}/plugins/profile-and-optimize/skills"
CLIENT=""
DEST_SKILLS=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/install-agent-skills.sh --client codex|cursor [options]

Options:
  --client NAME        Required. Install for codex or cursor.
  --skills-dir PATH    Override the client skills directory.
  --dry-run            Print actions without writing.
  -h, --help           Show this help.

Default destinations:
  codex   ~/.agents/skills
  cursor  ~/.cursor/skills

Each maintained skill is symlinked from this checkout. A rerun keeps correct
links and replaces same-name symlinks that point elsewhere. The installer
refuses to replace a regular file or directory.
EOF
}

require_value() {
  if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
    printf 'missing value for %s\n' "$1" >&2
    usage >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --client) require_value "$@"; CLIENT="$2"; shift 2 ;;
    --skills-dir) require_value "$@"; DEST_SKILLS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown arg: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$CLIENT" in
  codex|cursor) ;;
  "") printf '%s\n' '--client is required' >&2; usage >&2; exit 2 ;;
  *) printf 'invalid client: %s\n' "$CLIENT" >&2; usage >&2; exit 2 ;;
esac

if [[ -z "$DEST_SKILLS" ]]; then
  if [[ -z "${HOME:-}" ]]; then
    printf 'HOME is unset. Pass --skills-dir explicitly.\n' >&2
    exit 2
  fi
  if [[ "$CLIENT" == "codex" ]]; then
    DEST_SKILLS="${HOME}/.agents/skills"
  else
    DEST_SKILLS="${HOME}/.cursor/skills"
  fi
fi

if [[ ! -d "$SRC_SKILLS" ]]; then
  printf 'FATAL: source skills directory not found: %s\n' "$SRC_SKILLS" >&2
  exit 2
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '[dry-run] would ensure %s exists\n' "$DEST_SKILLS"
else
  mkdir -p "$DEST_SKILLS"
fi

LINKED=0
SKIPPED=0
FAILED=0

for skill_dir in "$SRC_SKILLS"/*/; do
  [[ -f "${skill_dir}SKILL.md" ]] || continue
  skill_name="$(basename "${skill_dir%/}")"
  dest="${DEST_SKILLS}/${skill_name}"

  if [[ -e "$dest" || -L "$dest" ]]; then
    if [[ -L "$dest" ]]; then
      target="$(readlink "$dest")"
      if [[ "$target" == "${skill_dir%/}" ]]; then
        printf '  [skip] %s already linked\n' "$skill_name"
        SKIPPED=$((SKIPPED + 1))
        continue
      fi
      if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '  [dry-run] would re-link %s -> %s\n' "$skill_name" "${skill_dir%/}"
      else
        rm "$dest"
        ln -s "${skill_dir%/}" "$dest"
        printf '  [relink] %s\n' "$skill_name"
        LINKED=$((LINKED + 1))
      fi
      continue
    fi

    printf '  [WARN] %s exists and is not a symlink (refusing to replace)\n' "$dest" >&2
    FAILED=$((FAILED + 1))
    continue
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '  [dry-run] would link %s -> %s\n' "$skill_name" "${skill_dir%/}"
  else
    ln -s "${skill_dir%/}" "$dest"
    printf '  [link]  %s\n' "$skill_name"
    LINKED=$((LINKED + 1))
  fi
done

printf '\nSummary: %d linked, %d already-linked, %d refused\n' \
  "$LINKED" "$SKIPPED" "$FAILED"

if [[ "$DRY_RUN" -eq 0 ]]; then
  if [[ "$CLIENT" == "codex" ]]; then
    printf '\nStart a new Codex session to discover the skills.\n'
  else
    printf '\nRestart Cursor to discover the skills.\n'
  fi
fi

if [[ "$FAILED" -gt 0 ]]; then
  printf '\n[WARN] %d skill conflict(s) require manual review in %s.\n' \
    "$FAILED" "$DEST_SKILLS" >&2
  exit 1
fi
