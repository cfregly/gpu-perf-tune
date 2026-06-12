"""Known-good config registry CLI surface.

Re-exports `CONTRACT`, `build_parser`, `main`, and `parse_args` from the
canonical implementation at
[`tools/known_good_config/known_good_config_cli.py`](../tools/known_good_config/known_good_config_cli.py).

Added in profile-and-optimize v1.68.0. Backs the
[`inference-known-good-config`](../skills/inference-known-good-config/SKILL.md) skill.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.known_good_config.known_good_config_cli import (  # noqa: E402
    CONTRACT,
    build_parser,
    main,
    parse_args,
)

__all__ = ["CONTRACT", "build_parser", "main", "parse_args"]


if __name__ == "__main__":  # direct invocation: fail loud, never silently no-op
    raise SystemExit(main())
