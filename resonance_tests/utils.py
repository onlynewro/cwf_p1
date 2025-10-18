"""Utility helpers for the resonance pre-test pipeline."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class RationalApproximation:
    """Representation of a best-fit rational approximation.

    Attributes
    ----------
    numerator:
        Numerator of the approximation.
    denominator:
        Denominator of the approximation. Always positive.
    value:
        Rational value ``numerator / denominator``.
    error:
        Absolute difference between the approximation and the target number.
    target:
        The target floating point value that was approximated.
    """

    numerator: int
    denominator: int
    value: float
    error: float
    target: float

    def as_string(self) -> str:
        """Return a human readable string for logs and summaries."""

        return f"{self.numerator}/{self.denominator} (Δ={self.error:.3g})"


def ensure_directory(path: Path | str) -> Path:
    """Create ``path`` and parents when necessary.

    Parameters
    ----------
    path:
        Directory that should exist.

    Returns
    -------
    pathlib.Path
        The normalised directory path.
    """

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def weighted_mean(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Compute a weighted mean with sensible fall-backs."""

    if weights is None:
        return float(np.mean(values))
    weights = np.asarray(weights)
    denom = np.sum(weights)
    if denom == 0:
        return float(np.mean(values))
    return float(np.sum(weights * values) / denom)


def weighted_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Perform a weighted linear regression.

    The model is ``y = beta0 + beta1 * x``.

    Parameters
    ----------
    x, y:
        One-dimensional data arrays of matching length.
    weights:
        Optional weights applied to each sample.

    Returns
    -------
    tuple
        ``(params, covariance, chi2)`` where ``params`` contains ``beta0`` and
        ``beta1``.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    if weights is None:
        W = np.ones_like(x)
    else:
        W = np.asarray(weights, dtype=float)
        if W.shape != x.shape:
            raise ValueError("weights must match x")
    W = np.clip(W, 0.0, np.inf)

    X = np.column_stack([np.ones_like(x), x])
    WX = X * np.sqrt(W)[:, None]
    Wy = y * np.sqrt(W)
    params, residuals, rank, s = np.linalg.lstsq(WX, Wy, rcond=None)
    if residuals.size == 0:
        residuals = np.array([np.sum((Wy - WX @ params) ** 2)])
    dof = max(len(x) - len(params), 1)
    chi2 = float(residuals[0])
    sigma2 = chi2 / dof
    cov = np.linalg.pinv(WX.T @ WX)
    cov = cov * sigma2
    return params, cov, chi2


def continued_fraction(value: float, max_terms: int = 12) -> List[int]:
    """Return the continued fraction coefficients for ``value``."""

    if math.isnan(value) or math.isinf(value):
        raise ValueError("continued_fraction requires a finite value")
    coefficients: List[int] = []
    x = value
    for _ in range(max_terms):
        a = math.floor(x)
        coefficients.append(int(a))
        frac = x - a
        if frac < 1e-12:
            break
        x = 1.0 / frac
    return coefficients


def convergents(cf: Sequence[int]) -> Iterable[Tuple[int, int]]:
    """Yield convergents produced by ``cf`` coefficients."""

    num1, num2 = 1, 0
    den1, den2 = 0, 1
    for a in cf:
        num = a * num1 + num2
        den = a * den1 + den2
        yield num, den
        num2, num1 = num1, num
        den2, den1 = den1, den


def best_rational_approximation(
    value: float,
    max_denominator: int,
    tolerance: float,
    max_terms: int = 12,
) -> RationalApproximation | None:
    """Return the best rational approximation within ``tolerance``.

    The search is based on truncating the continued fraction representation and
    checking each convergent.
    """

    cf = continued_fraction(value, max_terms=max_terms)
    best: RationalApproximation | None = None
    for num, den in convergents(cf):
        if den > max_denominator or den == 0:
            continue
        approx = num / den
        error = abs(value - approx)
        if error <= tolerance:
            candidate = RationalApproximation(num, den, approx, error, value)
            if best is None or candidate.error < best.error:
                best = candidate
    return best


def lomb_scargle_power(
    t: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    angular_frequencies: np.ndarray,
) -> np.ndarray:
    """Compute a weighted Lomb–Scargle periodogram.

    This implementation follows the normalised variant described by Scargle
    (1982) with the addition of sample weights.
    """

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)
    if t.shape != y.shape or t.shape != w.shape:
        raise ValueError("Input arrays must share the same shape")
    if np.any(~np.isfinite(t)) or np.any(~np.isfinite(y)):
        raise ValueError("Inputs must be finite")

    y_mean = weighted_mean(y, w)
    y = y - y_mean

    freq = np.asarray(angular_frequencies, dtype=float)
    power = np.zeros_like(freq)
    for i, omega in enumerate(freq):
        wt = omega * t
        cos_wt = np.cos(wt)
        sin_wt = np.sin(wt)
        two_omega_t = 2.0 * wt
        tan2wt_num = np.sum(w * np.sin(two_omega_t))
        tan2wt_den = np.sum(w * np.cos(two_omega_t))
        tau = 0.5 * math.atan2(tan2wt_num, tan2wt_den) / omega if omega != 0 else 0.0
        cos_term = np.cos(omega * (t - tau))
        sin_term = np.sin(omega * (t - tau))
        C = np.sum(w * cos_term * y)
        S = np.sum(w * sin_term * y)
        CC = np.sum(w * cos_term * cos_term)
        SS = np.sum(w * sin_term * sin_term)
        if CC == 0 or SS == 0:
            power[i] = 0.0
        else:
            power[i] = 0.5 * (C * C / CC + S * S / SS)
    return power


def estimate_fap(power: np.ndarray, trials: int) -> np.ndarray:
    """Estimate the false-alarm probability for observed powers."""

    power = np.asarray(power, dtype=float)
    if trials <= 0:
        raise ValueError("trials must be positive")
    return 1.0 - (1.0 - np.exp(-power)) ** trials


def false_alarm_threshold(alpha: float, trials: int) -> float:
    """Return the Lomb–Scargle power threshold for a global false alarm ``alpha``.

    The returned power corresponds to the value whose false-alarm probability,
    evaluated assuming ``trials`` independent frequencies, equals ``alpha``.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie within (0, 1)")
    if trials <= 0:
        raise ValueError("trials must be positive")
    # Invert ``alpha = 1 - (1 - exp(-z)) ** trials`` with respect to ``z``.
    base = 1.0 - alpha
    exponent = base ** (1.0 / trials)
    inner = 1.0 - exponent
    if inner <= 0.0:
        return float("inf")
    return float(-math.log(inner))
