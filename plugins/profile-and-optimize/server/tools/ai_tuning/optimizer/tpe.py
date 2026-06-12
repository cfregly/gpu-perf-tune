"""Tree-structured Parzen Estimator (TPE) Bayesian optimization.

This module implements a pure-Python TPE without scipy/numpy. The
algorithm follows the classic Bergstra et al. formulation:

1. Sort observations by objective value (according to `direction`).
2. Split at quantile `gamma` into a `good` set and a `bad` set.
3. Model l(x) over good and g(x) over bad as Parzen-window mixtures
   (Gaussian for numeric dims, smoothed count ratios for categoricals).
4. For a pool of random candidates, score by `l(x) / g(x)` and return
   the top picks.

When fewer than `min_observations` observations exist, fall back to
uniform random sampling and record `cold_start=true` in optimizer state.
"""

from __future__ import annotations

import math
import random as _random
from collections.abc import Iterable
from typing import NamedTuple

from .space import Space
from .types import Observation


class _ScoredCandidate(NamedTuple):
    score: float
    parameters: dict[str, str]


def _split_good_bad(
    observations: list[Observation],
    direction: str,
    gamma: float,
) -> tuple[list[Observation], list[Observation]]:
    if not observations:
        return [], []
    if direction == "maximize":
        sorted_obs = sorted(observations, key=lambda obs: obs.value, reverse=True)
    else:
        sorted_obs = sorted(observations, key=lambda obs: obs.value)
    n = len(sorted_obs)
    n_good = max(1, int(math.ceil(gamma * n)))
    n_good = min(n_good, n - 1) if n > 1 else 1
    good = sorted_obs[:n_good]
    bad = sorted_obs[n_good:] or [sorted_obs[-1]]
    return good, bad


def _gaussian_pdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 1e-12:
        sigma = 1e-12
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _bandwidth(values: list[float]) -> float:
    if len(values) < 2:
        return 0.25
    sorted_values = sorted(values)
    diffs = [
        max(sorted_values[idx + 1] - sorted_values[idx], 0.0)
        for idx in range(len(sorted_values) - 1)
    ]
    width = max(diffs) if diffs else 0.0
    span = sorted_values[-1] - sorted_values[0] if sorted_values else 0.0
    return max(width, span / max(len(values), 1), 0.05)


def _kde_score(value: float, samples: list[float]) -> float:
    if not samples:
        return 1e-6
    sigma = _bandwidth(samples)
    score = sum(_gaussian_pdf(value, sample, sigma) for sample in samples) / len(samples)
    return max(score, 1e-12)


def score_candidate(
    space: Space,
    candidate: list[float],
    good: list[Observation],
    bad: list[Observation],
) -> float:
    score = 0.0
    for index, _dim in enumerate(space.dimensions):
        good_samples = [obs.vector[index] for obs in good]
        bad_samples = [obs.vector[index] for obs in bad]
        l_score = _kde_score(candidate[index], good_samples)
        g_score = _kde_score(candidate[index], bad_samples)
        score += math.log(l_score) - math.log(g_score)
    return score


def propose(
    space: Space,
    observations: Iterable[Observation],
    direction: str,
    rng: _random.Random,
    *,
    gamma: float = 0.25,
    candidate_pool: int = 32,
    limit: int = 1,
    min_observations: int = 8,
    skip_keys: set[str] | None = None,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Return up to `limit` decoded candidate parameter dicts plus state metadata."""

    obs_list = [
        obs
        for obs in observations
        if isinstance(obs.value, (int, float)) and math.isfinite(obs.value)
    ]
    state: dict[str, object] = {
        "strategy": "bayesian",
        "variant": "tpe",
        "gamma": gamma,
        "candidate_pool": candidate_pool,
        "min_observations": min_observations,
        "observation_count": len(obs_list),
        "direction": direction,
    }
    skip = skip_keys or set()

    if len(obs_list) < min_observations:
        state["cold_start"] = True
        decoded: list[dict[str, str]] = []
        attempts = 0
        while len(decoded) < limit and attempts < candidate_pool * 4:
            params = space.random(rng)
            attempts += 1
            key = "|".join(f"{name}={params[name]}" for name in sorted(params))
            if key in skip:
                continue
            skip.add(key)
            decoded.append(params)
        return decoded, state

    state["cold_start"] = False
    good, bad = _split_good_bad(obs_list, direction, gamma)
    state["good_observations"] = len(good)
    state["bad_observations"] = len(bad)

    # Sample a pool, score, sort by score descending, return top non-duplicates.
    pool: list[tuple[float, list[float]]] = []
    for _ in range(candidate_pool):
        vector = [rng.random() for _ in space.dimensions]
        pool.append((score_candidate(space, vector, good, bad), vector))
    pool.sort(key=lambda item: item[0], reverse=True)

    scored_candidates: list[_ScoredCandidate] = []
    for score, vector in pool:
        params = space.decode(vector)
        key = "|".join(f"{name}={params[name]}" for name in sorted(params))
        if key in skip:
            continue
        skip.add(key)
        scored_candidates.append(_ScoredCandidate(score=score, parameters=params))
        if len(scored_candidates) >= limit:
            break

    state["candidate_scores"] = [candidate.score for candidate in scored_candidates]
    return [candidate.parameters for candidate in scored_candidates], state


__all__ = ["Observation", "propose", "score_candidate"]
