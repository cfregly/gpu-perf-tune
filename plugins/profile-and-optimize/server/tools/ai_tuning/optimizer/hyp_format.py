"""Parser for hypertune `.hyp` template files.

A `.hyp` is a bash sbatch script with hypertune markers:

    {{ hypertune.range=[1,8] }}        -> integer range
    {{ hypertune.array=[True, False] }} -> categorical
    {{ hypertune.flag=--bf16 }}         -> presence/absence boolean

The parser extracts a parameter manifest the optimizer can consume.
It does not interpret the script; it just reports the variables and
their search space so they can be promoted to a tuning-space manifest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_RANGE_RE = re.compile(r"\{\{\s*hypertune\.range\s*=\s*\[\s*([^\]]+)\s*\]\s*\}\}")
_ARRAY_RE = re.compile(r"\{\{\s*hypertune\.array\s*=\s*\[\s*([^\]]+)\s*\]\s*\}\}")
_FLAG_RE = re.compile(r"\{\{\s*hypertune\.flag\s*=\s*([^\s}]+)\s*\}\}")
_EXPORT_RE = re.compile(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")


@dataclass
class HypParameter:
    name: str
    kind: str  # integer | array | flag
    values: list[str] | None = None
    minimum: int | None = None
    maximum: int | None = None
    flag_token: str | None = None
    raw_line: str | None = None


def _split_array(payload: str) -> list[str]:
    items = [item.strip() for item in payload.split(",") if item.strip()]
    return items


def parse_text(text: str) -> list[HypParameter]:
    """Return parameters extracted from a `.hyp` script body."""

    parameters: list[HypParameter] = []

    for line in text.splitlines():
        export_match = _EXPORT_RE.match(line)
        if not export_match:
            # Capture flag-only lines, even if they aren't exports.
            flag_match = _FLAG_RE.search(line)
            if flag_match:
                token = flag_match.group(1)
                # Strip leading dashes for the parameter name.
                name = token.lstrip("-").replace("-", "_") or token
                parameters.append(
                    HypParameter(
                        name=name,
                        kind="flag",
                        values=["0", "1"],
                        flag_token=token,
                        raw_line=line,
                    )
                )
            continue

        name, value = export_match.group(1), export_match.group(2).strip()
        range_match = _RANGE_RE.search(value)
        array_match = _ARRAY_RE.search(value)
        flag_match = _FLAG_RE.search(value)
        if range_match:
            bounds = _split_array(range_match.group(1))
            if len(bounds) != 2:
                continue
            try:
                lo = int(bounds[0])
                hi = int(bounds[1])
            except ValueError:
                continue
            parameters.append(
                HypParameter(
                    name=name,
                    kind="integer",
                    minimum=min(lo, hi),
                    maximum=max(lo, hi),
                    raw_line=line,
                )
            )
        elif array_match:
            values = _split_array(array_match.group(1))
            parameters.append(
                HypParameter(
                    name=name,
                    kind="array",
                    values=values,
                    raw_line=line,
                )
            )
        elif flag_match:
            token = flag_match.group(1)
            parameters.append(
                HypParameter(
                    name=name,
                    kind="flag",
                    values=["0", "1"],
                    flag_token=token,
                    raw_line=line,
                )
            )
    return parameters


def parse_file(path: Path) -> list[HypParameter]:
    return parse_text(path.read_text(encoding="utf-8", errors="replace"))


def to_manifest_parameters(parameters: list[HypParameter]) -> list[dict[str, Any]]:
    """Convert parsed parameters to manifest-shape parameter entries."""

    out: list[dict[str, Any]] = []
    for param in parameters:
        if param.kind == "integer":
            out.append(
                {
                    "name": param.name,
                    "category": "imported",
                    "kind": "integer",
                    "minimum": param.minimum,
                    "maximum": param.maximum,
                    "values": [str(v) for v in range(param.minimum or 0, (param.maximum or 0) + 1)],
                    "wire": "imported from .hyp",
                    "description": f"Imported from hypertune .hyp template: range[{param.minimum},{param.maximum}]",
                }
            )
        elif param.kind == "array":
            out.append(
                {
                    "name": param.name,
                    "category": "imported",
                    "kind": "enum",
                    "values": list(param.values or []),
                    "wire": "imported from .hyp",
                    "description": "Imported from hypertune .hyp template: array",
                }
            )
        elif param.kind == "flag":
            out.append(
                {
                    "name": param.name,
                    "category": "imported",
                    "kind": "boolean",
                    "values": ["0", "1"],
                    "wire": f"imported from .hyp (flag {param.flag_token})",
                    "description": "Imported from hypertune .hyp template: flag",
                    "flag_token": param.flag_token,
                }
            )
    return out


__all__ = ["HypParameter", "parse_file", "parse_text", "to_manifest_parameters"]
