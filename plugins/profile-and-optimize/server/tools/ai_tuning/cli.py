"""CLI entrypoints for the AI tuner."""
# PROFILE_AND_OPTIMIZE_OPT_OUT: import facade for tools/ai_tuning/ai_tuning.py.

from __future__ import annotations

try:  # Package import, e.g. python -m tools.ai_tuning.cli.
    from .ai_tuning import build_parser, main
except ImportError:  # Direct script import from tools/ai_tuning.
    from ai_tuning import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
