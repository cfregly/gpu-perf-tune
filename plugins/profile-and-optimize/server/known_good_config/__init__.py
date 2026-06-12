"""Known-good per-model serving-config registry CLI.

This package is a thin shim. The real implementation lives at
[`tools/known_good_config/known_good_config_cli.py`](../tools/known_good_config/known_good_config_cli.py).
The shim exists so the MCP surface in [`mcp_surface.py`](../mcp_surface.py)
can introspect this library the same way it introspects the others via
`<repo_root>/<library>/cli.py`.

Added in profile-and-optimize v1.68.0.
"""

from __future__ import annotations

from .cli import CONTRACT, build_parser, main

__all__ = ["CONTRACT", "build_parser", "main"]
