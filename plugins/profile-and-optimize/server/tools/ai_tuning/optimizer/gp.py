"""Pure-Python Gaussian Process Bayesian optimization.

This module provides a minimal but correct GP regressor with an RBF
kernel, Cholesky-based posterior, Expected Improvement acquisition,
and a small length-scale grid search via type-II marginal likelihood.

The implementation is deliberately scipy/numpy free so the repository
stays self-contained. Acceptable performance bound: typical MLPerf
trial counts are tens to low hundreds, so an O(n^3) Cholesky on small
n is fine.
"""

from __future__ import annotations

import math
import random as _random
from collections.abc import Iterable
from dataclasses import dataclass

from .space import Space
from .types import Observation

JITTER = 1e-6


def _matrix_zeros(n: int) -> list[list[float]]:
    return [[0.0 for _ in range(n)] for _ in range(n)]


def _rbf(a: list[float], b: list[float], length_scale: float, signal_variance: float) -> float:
    sq = 0.0
    for ai, bi in zip(a, b):
        diff = ai - bi
        sq += diff * diff
    return signal_variance * math.exp(-sq / max(2.0 * length_scale * length_scale, 1e-12))


def _gram(
    vectors: list[list[float]],
    length_scale: float,
    signal_variance: float,
    noise_variance: float,
) -> list[list[float]]:
    n = len(vectors)
    matrix = _matrix_zeros(n)
    for i in range(n):
        for j in range(i, n):
            value = _rbf(vectors[i], vectors[j], length_scale, signal_variance)
            if i == j:
                value += noise_variance + JITTER
            matrix[i][j] = value
            matrix[j][i] = value
    return matrix


def _cholesky(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    L = _matrix_zeros(n)
    for i in range(n):
        for j in range(i + 1):
            s = matrix[i][j]
            for k in range(j):
                s -= L[i][k] * L[j][k]
            if i == j:
                if s <= 0.0:
                    s = JITTER
                L[i][j] = math.sqrt(s)
            else:
                if L[j][j] <= 0.0:
                    L[i][j] = 0.0
                else:
                    L[i][j] = s / L[j][j]
    return L


def _solve_lower(L: list[list[float]], b: list[float]) -> list[float]:
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        s = b[i]
        for k in range(i):
            s -= L[i][k] * y[k]
        y[i] = s / L[i][i] if abs(L[i][i]) > 1e-12 else 0.0
    return y


def _solve_upper(L: list[list[float]], b: list[float]) -> list[float]:
    """Solve L^T x = b given lower-triangular L."""
    n = len(L)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = b[i]
        for k in range(i + 1, n):
            s -= L[k][i] * x[k]
        x[i] = s / L[i][i] if abs(L[i][i]) > 1e-12 else 0.0
    return x


def _log_marginal_likelihood(
    vectors: list[list[float]],
    targets: list[float],
    length_scale: float,
    signal_variance: float,
    noise_variance: float,
) -> float:
    matrix = _gram(vectors, length_scale, signal_variance, noise_variance)
    L = _cholesky(matrix)
    alpha_step1 = _solve_lower(L, targets)
    alpha = _solve_upper(L, alpha_step1)
    n = len(targets)
    log_det = 0.0
    for index in range(n):
        diag = max(L[index][index], 1e-12)
        log_det += math.log(diag)
    log_det *= 2.0
    data_fit = sum(t * a for t, a in zip(targets, alpha))
    return -0.5 * data_fit - 0.5 * log_det - 0.5 * n * math.log(2.0 * math.pi)


@dataclass
class GpModel:
    vectors: list[list[float]]
    targets: list[float]
    length_scale: float
    signal_variance: float
    noise_variance: float
    L: list[list[float]]
    alpha: list[float]


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def fit(
    vectors: list[list[float]],
    targets: list[float],
    *,
    length_scale_candidates: Iterable[float] = (0.1, 0.3, 0.5, 1.0, 2.0),
    signal_variance: float = 1.0,
    noise_variance: float = 1e-3,
) -> GpModel:
    if not vectors:
        raise ValueError("GP requires at least one observation")
    if any(len(row) != len(vectors[0]) for row in vectors):
        raise ValueError("inconsistent vector dimensions")
    best_ls = None
    best_lml = -math.inf
    for ls in length_scale_candidates:
        try:
            lml = _log_marginal_likelihood(vectors, targets, ls, signal_variance, noise_variance)
        except (ValueError, ZeroDivisionError):
            continue
        if lml > best_lml:
            best_lml = lml
            best_ls = ls
    chosen_ls = best_ls if best_ls is not None else 0.5
    matrix = _gram(vectors, chosen_ls, signal_variance, noise_variance)
    L = _cholesky(matrix)
    alpha_step1 = _solve_lower(L, targets)
    alpha = _solve_upper(L, alpha_step1)
    return GpModel(
        vectors=vectors,
        targets=targets,
        length_scale=chosen_ls,
        signal_variance=signal_variance,
        noise_variance=noise_variance,
        L=L,
        alpha=alpha,
    )


def predict(model: GpModel, query: list[float]) -> tuple[float, float]:
    if not model.vectors:
        raise ValueError("GP model has no training vectors")
    expected_dim = len(model.vectors[0])
    if len(query) != expected_dim:
        raise ValueError(f"query dimension mismatch: expected {expected_dim}, got {len(query)}")
    k_star = [
        _rbf(train, query, model.length_scale, model.signal_variance)
        for train in model.vectors
    ]
    mean = sum(k_star[i] * model.alpha[i] for i in range(len(k_star)))
    v = _solve_lower(model.L, k_star)
    k_self = _rbf(query, query, model.length_scale, model.signal_variance) + JITTER
    variance = max(k_self - sum(value * value for value in v), 0.0)
    return mean, variance


def expected_improvement(
    mean: float,
    variance: float,
    best: float,
    direction: str,
    xi: float = 0.01,
) -> float:
    sigma = math.sqrt(max(variance, 0.0))
    if sigma <= 1e-12:
        return 0.0
    if direction == "minimize":
        improvement = best - mean - xi
    else:
        improvement = mean - best - xi
    z = improvement / sigma
    return improvement * _normal_cdf(z) + sigma * _normal_pdf(z)


def propose(
    space: Space,
    observations: Iterable[Observation],
    direction: str,
    rng: _random.Random,
    *,
    candidate_pool: int = 64,
    limit: int = 1,
    min_observations: int = 4,
    exploration_xi: float = 0.01,
    skip_keys: set[str] | None = None,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    obs = [o for o in observations if isinstance(o.value, (int, float)) and math.isfinite(o.value)]
    state: dict[str, object] = {
        "strategy": "bayesian",
        "variant": "gp",
        "candidate_pool": candidate_pool,
        "min_observations": min_observations,
        "observation_count": len(obs),
        "direction": direction,
        "exploration_xi": exploration_xi,
    }
    skip = skip_keys or set()

    if len(obs) < min_observations:
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
    vectors = [list(o.vector) for o in obs]
    targets = [float(o.value) for o in obs]
    model = fit(vectors, targets)
    state["length_scale"] = model.length_scale
    state["signal_variance"] = model.signal_variance
    state["noise_variance"] = model.noise_variance

    if direction == "minimize":
        best = min(targets)
    else:
        best = max(targets)
    state["best_observed"] = best

    pool: list[tuple[float, list[float], float, float]] = []
    for _ in range(candidate_pool):
        vector = [rng.random() for _ in space.dimensions]
        mean, variance = predict(model, vector)
        ei = expected_improvement(mean, variance, best, direction, exploration_xi)
        pool.append((ei, vector, mean, variance))
    pool.sort(key=lambda item: item[0], reverse=True)

    decoded = []
    acquisition_scores = []
    means = []
    variances = []
    for ei, vector, mean, variance in pool:
        params = space.decode(vector)
        key = "|".join(f"{name}={params[name]}" for name in sorted(params))
        if key in skip:
            continue
        skip.add(key)
        decoded.append(params)
        acquisition_scores.append(ei)
        means.append(mean)
        variances.append(variance)
        if len(decoded) >= limit:
            break

    state["acquisition_scores"] = acquisition_scores
    state["posterior_means"] = means
    state["posterior_variances"] = variances
    return decoded, state


__all__ = ["GpModel", "fit", "predict", "expected_improvement", "propose"]
