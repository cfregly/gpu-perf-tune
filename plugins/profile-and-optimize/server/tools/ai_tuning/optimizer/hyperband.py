"""Hyperband and BOHB multi-fidelity optimization.

Hyperband structures search into brackets of successive halving:

  s_max = floor(log_eta(max_budget / min_budget))
  For s in s_max .. 0:
    n_s = ceil((s_max + 1) / (s + 1)) * eta^s
    For rung i in 0 .. s:
      n_i = floor(n_s / eta^i)
      budget_i = min_budget * eta^i
      run n_i candidates at budget_i, advance top n_i / eta

BOHB seeds each bracket's candidates with TPE instead of uniform random.

The optimizer does not execute trials. Each `propose` call emits the
candidates owed by the current bracket state, encoded with their
assigned fidelity, and returns updated optimizer state. The caller
runs them via the experiment ledger flow and feeds completed
observations back on the next propose call.
"""

from __future__ import annotations

import math
import random as _random
from collections.abc import Iterable
from dataclasses import dataclass, field

from .space import Space
from .tpe import propose as tpe_propose
from .types import Observation


@dataclass
class HyperbandConfig:
    eta: int = 3
    min_budget: float = 1.0
    max_budget: float = 27.0


@dataclass
class Rung:
    bracket_index: int
    rung_index: int
    budget: float
    n_candidates: int


@dataclass
class Bracket:
    s: int
    rungs: list[Rung] = field(default_factory=list)


def _log(value: float, base: float) -> float:
    if value <= 0 or base <= 1:
        return 0.0
    return math.log(value) / math.log(base)


def plan(config: HyperbandConfig) -> list[Bracket]:
    eta = max(2, int(config.eta))
    min_budget = max(1e-9, float(config.min_budget))
    max_budget = max(min_budget * eta, float(config.max_budget))
    s_max = max(0, int(math.floor(_log(max_budget / min_budget, eta))))
    brackets: list[Bracket] = []
    for s_index, s in enumerate(range(s_max, -1, -1)):
        n_s = int(math.ceil((s_max + 1) / (s + 1) * (eta ** s)))
        n_s = max(1, n_s)
        rungs: list[Rung] = []
        for i in range(s + 1):
            n_i = max(1, int(math.floor(n_s / (eta ** i))))
            budget_i = min_budget * (eta ** i)
            if budget_i > max_budget:
                budget_i = max_budget
            rungs.append(
                Rung(
                    bracket_index=s_index,
                    rung_index=i,
                    budget=budget_i,
                    n_candidates=n_i,
                )
            )
        brackets.append(Bracket(s=s, rungs=rungs))
    return brackets


def _trial_key(parameters: dict[str, str]) -> str:
    return "|".join(f"{name}={parameters[name]}" for name in sorted(parameters))


@dataclass
class HyperbandState:
    bracket_index: int = 0
    rung_index: int = 0
    bracket_keys: list[list[str]] = field(default_factory=list)
    completed: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "bracket_index": self.bracket_index,
            "rung_index": self.rung_index,
            "bracket_keys": self.bracket_keys,
            "completed": self.completed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object] | None) -> HyperbandState:
        if not isinstance(data, dict):
            return cls()
        return cls(
            bracket_index=int(data.get("bracket_index", 0) or 0),
            rung_index=int(data.get("rung_index", 0) or 0),
            bracket_keys=[
                [str(key) for key in row]
                for row in data.get("bracket_keys", []) or []
            ],
            completed=bool(data.get("completed", False)),
        )


def _select_top(
    candidates: list[tuple[str, float]],
    keep: int,
    direction: str,
) -> list[str]:
    sorted_keys = sorted(
        candidates,
        key=lambda item: item[1],
        reverse=(direction == "maximize"),
    )
    return [key for key, _value in sorted_keys[:keep]]


def propose(
    space: Space,
    observations: Iterable[Observation],
    direction: str,
    rng: _random.Random,
    *,
    config: HyperbandConfig,
    variant: str = "hyperband",
    state_in: dict[str, object] | None = None,
    skip_keys: set[str] | None = None,
    tpe_min_observations: int = 4,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Emit the candidates owed by the current bracket/rung.

    Returns `(candidates, state_out)`. Each candidate is a dict with
    `parameters` (decoded values) and `fidelity` (assigned budget).
    """

    obs = [o for o in observations if isinstance(o.value, (int, float)) and math.isfinite(o.value)]
    state = HyperbandState.from_dict(state_in)
    brackets = plan(config)
    skip = skip_keys or set()
    out_state: dict[str, object] = {
        "strategy": "multifidelity",
        "variant": variant,
        "eta": config.eta,
        "min_budget": config.min_budget,
        "max_budget": config.max_budget,
        "bracket_count": len(brackets),
    }

    if state.bracket_index >= len(brackets):
        out_state.update(state.to_dict())
        out_state["completed"] = True
        return [], out_state

    current_bracket = brackets[state.bracket_index]
    if state.rung_index >= len(current_bracket.rungs):
        # Advance to next bracket.
        state.bracket_index += 1
        state.rung_index = 0
        state.bracket_keys = []
        if state.bracket_index >= len(brackets):
            out_state.update(state.to_dict())
            out_state["completed"] = True
            return [], out_state
        current_bracket = brackets[state.bracket_index]

    rung = current_bracket.rungs[state.rung_index]

    # Determine eligible candidate keys for this rung.
    if state.rung_index == 0 or not state.bracket_keys:
        # Brand new bracket: sample n candidates.
        sampled: list[dict[str, str]] = []
        sampled_keys: list[str] = []
        if variant == "bohb" and len(obs) >= tpe_min_observations:
            tpe_candidates, _ = tpe_propose(
                space,
                obs,
                direction,
                rng,
                limit=rung.n_candidates,
                candidate_pool=max(rung.n_candidates * 4, 16),
                min_observations=tpe_min_observations,
                skip_keys=set(skip),
            )
            sampled = tpe_candidates
        # Fill remainder (or full set for plain Hyperband) with uniform random.
        attempts = 0
        while len(sampled) < rung.n_candidates and attempts < rung.n_candidates * 8:
            attempts += 1
            params = space.random(rng)
            key = _trial_key(params)
            if key in skip or key in {_trial_key(p) for p in sampled}:
                continue
            sampled.append(params)
        sampled_keys = [_trial_key(params) for params in sampled]
        state.bracket_keys = [sampled_keys]
        out_candidates = [
            {"parameters": params, "fidelity": rung.budget}
            for params in sampled
        ]
        out_state.update(state.to_dict())
        out_state.update(
            {
                "current_bracket_s": current_bracket.s,
                "current_rung": rung.rung_index,
                "current_budget": rung.budget,
                "current_n": rung.n_candidates,
                "emitted_count": len(out_candidates),
            }
        )
        # Caller advances rung once trials complete via `advance_rung`.
        return out_candidates, out_state

    # Subsequent rung: advance from previous rung's survivors using observations.
    previous_keys = state.bracket_keys[-1]
    observed_pairs: list[tuple[str, float]] = []
    for o in obs:
        decoded = space.decode(o.vector)
        key = _trial_key({k: decoded[k] for k in decoded})
        if key in previous_keys:
            observed_pairs.append((key, float(o.value)))
    needed = max(1, int(math.floor(len(previous_keys) / max(2, config.eta))))
    survivors = _select_top(observed_pairs, needed, direction)
    if not survivors:
        # Without observed evidence, repeat the previous rung's keys at the new budget.
        survivors = previous_keys[:needed]
    state.bracket_keys.append(survivors)
    survivor_params = []
    for key in survivors:
        # Decode keys back into parameter dicts via known sample order from previous_keys.
        # Because keys carry sorted name=value pairs, parse them directly.
        params: dict[str, str] = {}
        for chunk in key.split("|"):
            if "=" not in chunk:
                continue
            name, _, value = chunk.partition("=")
            params[name] = value
        survivor_params.append(params)
    out_candidates = [
        {"parameters": params, "fidelity": rung.budget}
        for params in survivor_params
    ]
    out_state.update(state.to_dict())
    out_state.update(
        {
            "current_bracket_s": current_bracket.s,
            "current_rung": rung.rung_index,
            "current_budget": rung.budget,
            "current_n": len(survivors),
            "emitted_count": len(out_candidates),
        }
    )
    return out_candidates, out_state


def advance_rung(state_in: dict[str, object] | None) -> dict[str, object]:
    """Mark the current rung as completed so the next propose moves to the next rung."""

    state = HyperbandState.from_dict(state_in)
    state.rung_index += 1
    return state.to_dict()


__all__ = [
    "Bracket",
    "HyperbandConfig",
    "HyperbandState",
    "Rung",
    "advance_rung",
    "plan",
    "propose",
]
