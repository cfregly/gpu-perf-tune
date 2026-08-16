"""Reproducibility-grade evidence scaffolder CLI.

This package is a thin shim. The real implementation lives at
[`tools/evidence/evidence_cli.py`](../tools/evidence/evidence_cli.py).
The shim exists so the MCP surface in [`mcp_surface.py`](../mcp_surface.py)
can introspect this library like the other contract-driven libraries via
`<repo_root>/<library>/cli.py`.

Backs the
[`evidence-bundle-init`](../../skills/evidence-bundle-init/SKILL.md) skill.
"""

from __future__ import annotations

from .cli import CONTRACT, build_parser, main

__all__ = ["CONTRACT", "build_parser", "main"]
