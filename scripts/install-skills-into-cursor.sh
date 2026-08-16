#!/usr/bin/env bash
# Compatibility wrapper for the client-neutral Agent Skills installer.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/install-agent-skills.sh" --client cursor "$@"
