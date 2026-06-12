"""Parameter space encoding for optimizer engines.

Each parameter from a tuning space manifest becomes a `Dimension`. The
`Space` wrapper exposes:

- `encode(parameters)` to map a parameter dict to a numeric vector
- `decode(vector)` to map a numeric vector back to manifest values
- `random(rng)` to sample a uniform random parameter dict

Numeric encoding stays in [0, 1] across kinds so that the GP and TPE
implementations can treat all dimensions uniformly without scipy/numpy.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


def _coerce_numeric(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, str):
        return int(value)
    return int(value)


@dataclass
class Dimension:
    """One parameter dimension."""

    name: str
    kind: str  # categorical | integer | continuous | boolean
    values: list[str] | None = None
    minimum: float | None = None
    maximum: float | None = None

    def encode(self, value: Any) -> float:
        if self.kind == "categorical":
            options = self.values or []
            try:
                index = options.index(str(value))
            except ValueError as exc:
                raise ValueError(
                    f"value {value!r} not in categorical dimension {self.name!r} options {options}"
                ) from exc
            if len(options) <= 1:
                return 0.0
            return index / (len(options) - 1)
        if self.kind == "boolean":
            normalized = str(value).strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return 1.0
            if normalized in {"0", "false", "no", "off"}:
                return 0.0
            raise ValueError(f"value {value!r} is not boolean for {self.name!r}")
        if self.kind == "integer":
            number = _coerce_int(value)
            if self.minimum is None or self.maximum is None:
                raise ValueError(f"integer dimension {self.name!r} missing min/max")
            if self.maximum == self.minimum:
                return 0.0
            return (number - self.minimum) / (self.maximum - self.minimum)
        if self.kind == "continuous":
            number = _coerce_numeric(value)
            if self.minimum is None or self.maximum is None:
                raise ValueError(f"continuous dimension {self.name!r} missing min/max")
            if self.maximum == self.minimum:
                return 0.0
            return (number - self.minimum) / (self.maximum - self.minimum)
        raise ValueError(f"unsupported dimension kind {self.kind!r}")

    def decode(self, scalar: float) -> str:
        clamped = max(0.0, min(1.0, scalar))
        if self.kind == "categorical":
            options = self.values or []
            if not options:
                return ""
            if len(options) == 1:
                return options[0]
            index = round(clamped * (len(options) - 1))
            index = max(0, min(len(options) - 1, index))
            return options[index]
        if self.kind == "boolean":
            return "1" if clamped >= 0.5 else "0"
        if self.kind == "integer":
            assert self.minimum is not None and self.maximum is not None
            number = round(self.minimum + clamped * (self.maximum - self.minimum))
            return str(int(number))
        if self.kind == "continuous":
            assert self.minimum is not None and self.maximum is not None
            number = self.minimum + clamped * (self.maximum - self.minimum)
            return repr(number)
        raise ValueError(f"unsupported dimension kind {self.kind!r}")

    def cardinality(self) -> int | None:
        if self.kind == "categorical":
            return len(self.values or [])
        if self.kind == "boolean":
            return 2
        if self.kind == "integer" and self.minimum is not None and self.maximum is not None:
            return int(self.maximum - self.minimum + 1)
        return None


def dimension_from_param(param: dict[str, Any]) -> Dimension:
    name = str(param["name"])
    kind = str(param.get("kind", "string"))
    values = param.get("values")
    if values:
        return Dimension(name=name, kind="categorical", values=[str(v) for v in values])
    if kind == "boolean":
        return Dimension(name=name, kind="boolean", values=["0", "1"])
    if kind in {"integer", "number"}:
        minimum = param.get("minimum")
        maximum = param.get("maximum")
        if minimum is None or maximum is None:
            raise ValueError(f"parameter {name!r} kind {kind!r} requires minimum and maximum")
        return Dimension(
            name=name,
            kind="integer" if kind == "integer" else "continuous",
            minimum=float(minimum),
            maximum=float(maximum),
        )
    if kind in {"enum", "string"}:
        raise ValueError(f"parameter {name!r} kind {kind!r} requires explicit `values` to be a finite domain")
    raise ValueError(f"unsupported parameter kind {kind!r} for {name!r}")


@dataclass
class Space:
    """Optimizer-facing view of a tuning space's finite/searchable dimensions."""

    dimensions: list[Dimension]

    @classmethod
    def from_manifest(cls, space: dict[str, Any], parameter_names: Iterable[str] | None = None) -> Space:
        names = set(parameter_names) if parameter_names else None
        dimensions: list[Dimension] = []
        for param in space.get("parameters", []):
            if not isinstance(param, dict) or "name" not in param:
                continue
            if names is not None and param["name"] not in names:
                continue
            try:
                dimensions.append(dimension_from_param(param))
            except ValueError:
                # Parameters without a finite/searchable encoding are skipped silently;
                # callers can still validate them via the higher-level proposal validator.
                continue
        if not dimensions:
            raise ValueError("optimizer space requires at least one finite/searchable dimension")
        return cls(dimensions=dimensions)

    def names(self) -> list[str]:
        return [dim.name for dim in self.dimensions]

    def encode(self, parameters: dict[str, Any]) -> list[float]:
        encoded: list[float] = []
        for dim in self.dimensions:
            if dim.name not in parameters:
                raise ValueError(f"missing parameter {dim.name!r} when encoding")
            encoded.append(dim.encode(parameters[dim.name]))
        return encoded

    def decode(self, vector: list[float]) -> dict[str, str]:
        if len(vector) != len(self.dimensions):
            raise ValueError(
                f"decode expected {len(self.dimensions)} values, got {len(vector)}"
            )
        return {dim.name: dim.decode(vector[index]) for index, dim in enumerate(self.dimensions)}

    def random(self, rng) -> dict[str, str]:
        vector = [rng.random() for _ in self.dimensions]
        return self.decode(vector)

    def total_finite(self) -> int | None:
        total = 1
        for dim in self.dimensions:
            card = dim.cardinality()
            if card is None:
                return None
            total *= card
        return total

    def euclidean(self, a: list[float], b: list[float]) -> float:
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


__all__ = ["Dimension", "Space", "dimension_from_param"]
