"""Workload-agnostic Slurm job triage CLI.

This package is a thin shim. The real implementation lives at
[`tools/slurm/slurm_cli.py`](../tools/slurm/slurm_cli.py).
The shim exists so the MCP surface in [`mcp_surface.py`](../mcp_surface.py)
can introspect this library alongside every package in
`mcp_surface.LIBRARIES` via `<repo_root>/<library>/cli.py`.

Cluster-level findings can hand off to
[`k8s-troubleshooting`](../../skills/k8s-troubleshooting/SKILL.md).
"""

from __future__ import annotations

from .cli import CONTRACT, build_parser, main

__all__ = ["CONTRACT", "build_parser", "main"]
