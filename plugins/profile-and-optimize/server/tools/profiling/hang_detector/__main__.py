"""Make ``python -m tools.profiling.hang_detector`` work."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
