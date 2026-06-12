"""Slurm CLI surface.

Re-exports `CONTRACT`, `build_parser`, and `main` from the canonical
implementation at [`tools/slurm/slurm_cli.py`](../tools/slurm/slurm_cli.py).

Added in profile-and-optimize v0.5.0.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.slurm.slurm_cli import (  # noqa: E402
    CONTRACT,
    build_parser,
    main,
    parse_args,
)

__all__ = ["CONTRACT", "build_parser", "main", "parse_args"]


if __name__ == "__main__":  # direct invocation: fail loud, never silently no-op
    raise SystemExit(main())
