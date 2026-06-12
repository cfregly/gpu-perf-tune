"""Workload-agnostic Slurm job triage CLI.

This package is a thin shim. The real implementation lives at
[`tools/slurm/slurm_cli.py`](../tools/slurm/slurm_cli.py).
The shim exists so the MCP surface in [`mcp_surface.py`](../mcp_surface.py)
can introspect this library the same way it introspects the MLPerf libraries
via `<repo_root>/<library>/cli.py`.

Added in profile-and-optimize v0.5.0. Backs the
[`slurm-job-triage-generic`](../skills/slurm-job-triage-generic/SKILL.md) skill.
"""

from __future__ import annotations

from .cli import CONTRACT, build_parser, main

__all__ = ["CONTRACT", "build_parser", "main"]
