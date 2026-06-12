"""Importer for hypertune session directories.

A hypertune session typically lives at `~/.hypertune/<session-id>/` and
contains:

- `session.yaml` describing the session, objective, and trial states
- one `<trial-id>.sbatch` per trial
- one `<trial-id>.out` per finished trial with the score on a known line

This module imports a session by:

1. Reading `session.yaml` for objective and trial metadata
2. For each trial, reading the `.sbatch` to recover parameter values
   (looking for `export NAME=VALUE` lines for parameters in the active
   tuning space)
3. Parsing the `.out` file for the score (defaults to lines containing
   `:::MLLOG` or a known `score=<float>` marker)
4. Emitting one ledger record per recoverable trial with
   `provenance=hyp_import`.

The implementation tolerates missing fields and skips trials whose
parameters cannot be matched against the active tuning space.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_EXPORT_RE = re.compile(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")
_SCORE_PATTERNS = (
    re.compile(r":::MLLOG.*\"key\":\s*\"final_loss\".*\"value\":\s*(-?\d+(?:\.\d+)?)"),
    re.compile(r":::MLLOG.*\"key\":\s*\"eval_accuracy\".*\"value\":\s*(-?\d+(?:\.\d+)?)"),
    re.compile(r"^\s*score\s*=\s*(-?\d+(?:\.\d+)?)\s*$", re.MULTILINE),
    re.compile(r"^\s*HYPERTUNE_SCORE\s+(-?\d+(?:\.\d+)?)\s*$", re.MULTILINE),
)


@dataclass
class HypTrial:
    trial_id: str
    parameters: dict[str, str]
    score: float | None
    status: str
    sbatch_file: str | None = None
    log_file: str | None = None


def _parse_yaml_lines(text: str) -> dict[str, Any]:
    """Tiny yaml-ish parser limited to the session.yaml shapes hypertune emits.

    Supports flat scalars, list-of-dicts under top-level keys, and
    nested mappings to one level. Avoids a yaml dependency so the repo
    stays self-contained.
    """

    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(0, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        # Pop stack to current indent level.
        while len(stack) > 1 and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if line.startswith("- "):
            entry = line[2:].strip()
            container = parent if isinstance(parent, list) else None
            if container is None:
                # Last key on parent expects a list.
                if isinstance(parent, dict):
                    last_key = next(reversed(parent))
                    if not isinstance(parent[last_key], list):
                        parent[last_key] = []
                    container = parent[last_key]
                else:
                    continue
            if ":" in entry:
                key, _, value = entry.partition(":")
                element: dict[str, Any] = {key.strip(): value.strip()} if value.strip() else {key.strip(): None}
                container.append(element)
                stack.append((indent, element))
            else:
                container.append(entry)
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if not value:
                if isinstance(parent, dict):
                    parent[key] = {}
                    stack.append((indent, parent[key]))
            else:
                if isinstance(parent, dict):
                    parent[key] = value
    return root


def _coerce_value(value: Any) -> Any:
    if isinstance(value, str):
        if value.lower() in {"true", "yes"}:
            return True
        if value.lower() in {"false", "no"}:
            return False
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value
    return value


def _read_score(log_path: Path) -> float | None:
    if not log_path.is_file():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for pattern in _SCORE_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def _read_sbatch_parameters(sbatch_path: Path, parameter_names: set[str]) -> dict[str, str]:
    if not sbatch_path.is_file():
        return {}
    parameters: dict[str, str] = {}
    for line in sbatch_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _EXPORT_RE.match(line)
        if not match:
            continue
        name, value = match.group(1), match.group(2).strip()
        if name not in parameter_names:
            continue
        parameters[name] = value.strip("\"'")
    return parameters


@dataclass
class HypSession:
    session_id: str | None
    objective: str | None
    trials: list[HypTrial]


def import_session(
    session_dir: Path,
    parameter_names: set[str],
) -> HypSession:
    yaml_path = session_dir / "session.yaml"
    yaml_data: dict[str, Any] = {}
    if yaml_path.is_file():
        yaml_data = _parse_yaml_lines(yaml_path.read_text(encoding="utf-8", errors="replace"))
    objective = None
    if isinstance(yaml_data.get("objective"), str):
        objective = yaml_data["objective"]
    elif isinstance(yaml_data.get("results"), dict):
        # Fall back to no objective; hypertune sessions sometimes name it elsewhere.
        objective = None
    session_id = yaml_data.get("id") if isinstance(yaml_data.get("id"), str) else None

    trials: list[HypTrial] = []
    for sbatch in sorted(session_dir.glob("*.sbatch")):
        trial_id = sbatch.stem
        params = _read_sbatch_parameters(sbatch, parameter_names)
        if not params:
            continue
        log_path = session_dir / f"{trial_id}.out"
        score = _read_score(log_path)
        status = "succeeded" if score is not None else "failed"
        trials.append(
            HypTrial(
                trial_id=trial_id,
                parameters=params,
                score=score,
                status=status,
                sbatch_file=str(sbatch),
                log_file=str(log_path) if log_path.is_file() else None,
            )
        )
    return HypSession(session_id=session_id, objective=objective, trials=trials)


__all__ = ["HypSession", "HypTrial", "import_session"]
