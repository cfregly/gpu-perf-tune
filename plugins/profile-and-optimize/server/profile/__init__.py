"""Operator-facing profiling package.

This package is a thin shim. The real implementation lives at
[`tools/pipeline/submission/profile/profile_cli.py`](../tools/pipeline/submission/profile/profile_cli.py),
which umbrellas `host_overhead.py` (py-spy CPU sampler) and
`profile_diff.py` (nsys-stats delta tables). The shim exists so the
MCP surface in [`mcp_surface.py`](../mcp_surface.py) can introspect
the profile CLI the same way it introspects launcher / selector /
validator / contention via `<repo_root>/<library>/cli.py`.
"""

from __future__ import annotations

from .cli import CONTRACT, build_parser, main

__all__ = ["CONTRACT", "build_parser", "main"]
