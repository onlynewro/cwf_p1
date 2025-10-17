"""Numerical covariance estimation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass
class CovarianceResult:
    """Container for a covariance estimate."""

    matrix: np.ndarray
    correlation: np.ndarray
    errors: dict
    condition_number: float


class CovarianceComputationError(RuntimeError):
    """Raised when the numerical covariance estimation fails."""


def _build_step_sizes(theta: Sequence[float], bounds):
    theta = np.asarray(theta, dtype=float)
    n_params = theta.size
    steps = np.maximum(np.abs(theta) * 1e-4, 1e-6)

    if bounds is None:
        return steps

    adjusted_steps = steps.copy()
    for i, (low, high) in enumerate(bounds):
        lower_gap = np.inf if low is None else max(theta[i] - low, 0.0)
        upper_gap = np.inf if high is None else max(high - theta[i], 0.0)

        max_allowed = min(lower_gap, upper_gap)
        if not np.isfinite(max_allowed):
            max_allowed = max(lower_gap, upper_gap)

        if max_allowed <= 0.0:
            raise CovarianceComputationError(
                f"parameter '{i}' is pinned to its boundary, cannot estimate curvature"
            )

        adjusted_steps[i] = min(adjusted_steps[i], max_allowed * 0.5)
        adjusted_steps[i] = max(adjusted_steps[i], 1e-8)

    return adjusted_steps


def _numerical_hessian(func: Callable[[np.ndarray], float], theta: Sequence[float], steps):
    theta = np.asarray(theta, dtype=float)
    steps = np.asarray(steps, dtype=float)
    n_params = theta.size

    cache = {}

    def evaluate(point):
        key = tuple(point.tolist())
        if key in cache:
            return cache[key]
        value = float(func(point))
        if not np.isfinite(value):
            raise CovarianceComputationError(
                "objective returned a non-finite value during Hessian evaluation"
            )
        cache[key] = value
        return value

    f0 = evaluate(theta)
    hessian = np.zeros((n_params, n_params), dtype=float)

    for i in range(n_params):
        ei = np.zeros(n_params, dtype=float)
        ei[i] = steps[i]

        f_plus = evaluate(theta + ei)
        f_minus = evaluate(theta - ei)
        hessian[i, i] = (f_plus - 2.0 * f0 + f_minus) / (steps[i] ** 2)

        for j in range(i + 1, n_params):
            ej = np.zeros(n_params, dtype=float)
            ej[j] = steps[j]

            f_pp = evaluate(theta + ei + ej)
            f_pm = evaluate(theta + ei - ej)
            f_mp = evaluate(theta - ei + ej)
            f_mm = evaluate(theta - ei - ej)

            value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * steps[i] * steps[j])
            hessian[i, j] = value
            hessian[j, i] = value

    return hessian, f0


def estimate_covariance(func: Callable[[np.ndarray], float], theta, param_names, bounds, dof):
    """Estimate the covariance matrix at the best-fit point ``theta``."""
    if dof <= 0:
        raise CovarianceComputationError("non-positive degrees of freedom")

    steps = _build_step_sizes(theta, bounds)
    hessian, _ = _numerical_hessian(func, theta, steps)

    cond_number = np.linalg.cond(hessian)
    if not np.isfinite(cond_number):
        cond_number = np.inf

    eye = np.eye(len(theta))
    covariance = None
    regularisation_used = None

    for lam in (0.0, 1e-8, 1e-6, 1e-4, 1e-2):
        try:
            candidate = np.linalg.inv(hessian + lam * eye)
            covariance = 2.0 * candidate
            regularisation_used = lam
            break
        except np.linalg.LinAlgError:
            continue

    if covariance is None:
        candidate = np.linalg.pinv(hessian)
        covariance = 2.0 * candidate
        regularisation_used = None

    covariance = 0.5 * (covariance + covariance.T)

    diag = np.diag(covariance)
    errors = {}
    for name, value in zip(param_names, diag):
        errors[name] = float(np.sqrt(value)) if value >= 0 else None
        if name == "h" and errors[name] is not None:
            errors["H0"] = float(errors[name] * 100.0)

    with np.errstate(invalid="ignore"):
        correlation = covariance / np.sqrt(np.outer(diag, diag))

    return CovarianceResult(
        matrix=covariance,
        correlation=correlation,
        errors=errors,
        condition_number=float(cond_number),
    )


__all__ = ["CovarianceComputationError", "CovarianceResult", "estimate_covariance"]
