"""findings CLI surface.

Re-exports from [`tools/findings/findings_cli.py`](../tools/findings/findings_cli.py).

Added in profile-and-optimize v0.9.0.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.findings.findings_cli import (  # noqa: E402
    CONTRACT,
    build_parser,
    main,
)


if __name__ == "__main__":
    sys.exit(main())
