"""Shared optimizer data types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    """One objective observation at an encoded parameter vector."""

    vector: list[float]
    value: float


__all__ = ["Observation"]
