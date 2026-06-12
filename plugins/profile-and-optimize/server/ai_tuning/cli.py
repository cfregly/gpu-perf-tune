"""AI-tuning CLI surface.

Re-exports `CONTRACT`, `build_parser`, and `main` from the canonical
implementation at
[`tools/ai_tuning/ai_tuning.py`](../tools/ai_tuning/ai_tuning.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ai_tuning.ai_tuning import (  # noqa: E402
    CONTRACT,
    build_parser,
    main,
)

__all__ = ["CONTRACT", "build_parser", "main"]


if __name__ == "__main__":  # direct invocation: fail loud, never silently no-op
    raise SystemExit(main())
