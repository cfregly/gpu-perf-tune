"""Operator-facing AI-assisted tuning package.

This package is a thin shim. The real implementation lives at
[`tools/ai_tuning/ai_tuning.py`](../tools/ai_tuning/ai_tuning.py).
The shim exists so the MCP surface in [`mcp_surface.py`](../mcp_surface.py)
can introspect the AI-tuning CLI the same way it introspects launcher /
selector / validator / contention via `<repo_root>/<library>/cli.py`.
"""

from __future__ import annotations

from .cli import CONTRACT, build_parser, main

__all__ = ["CONTRACT", "build_parser", "main"]
