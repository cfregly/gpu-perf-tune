"""Perf-report CLI surface.

Re-exports `CONTRACT`, `build_parser`, `main`, and `parse_args` from the
canonical implementation at
[`tools/perf_tune_report/perf_tune_report_cli.py`](../tools/perf_tune_report/perf_tune_report_cli.py).

Added in profile-and-optimize v1.10.0. Backs the
[`inference-perf-tune-report`](../skills/inference-perf-tune-report/SKILL.md) skill.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.perf_tune_report.perf_tune_report_cli import (  # noqa: E402
    CONTRACT,
    build_parser,
    main,
    parse_args,
)

__all__ = ["CONTRACT", "build_parser", "main", "parse_args"]


if __name__ == "__main__":  # direct invocation: fail loud, never silently no-op
    raise SystemExit(main())
