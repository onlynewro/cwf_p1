"""Diophantine-style diagnostics for β slopes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .utils import (
    RationalApproximation,
    convergents,
    continued_fraction,
    ensure_directory,
)

@dataclass
class DiophantineResult:
    """Summary of the rationality analysis for β₁/(2π)."""

    slopes: np.ndarray
    ratios: np.ndarray
    approximations: List[RationalApproximation | None]
    thresholds: np.ndarray
    scaled_errors: np.ndarray
    artifact_path: Path

    def violators(self) -> List[int]:
        """Return indices whose approximation fell within the tolerance."""

        return [idx for idx, scale in enumerate(self.scaled_errors) if scale <= 1.0]


def _analyse_ratio(
    ratio: float,
    max_denominator: int,
    base_tolerance: float,
) -> Tuple[RationalApproximation | None, float, float]:
    """Return the best convergent and scaled error for ``ratio``."""

    cf = continued_fraction(ratio)
    best: RationalApproximation | None = None
    best_threshold = base_tolerance / max(max_denominator, 1)
    best_scaled = float("inf")
    for numerator, denominator in convergents(cf):
        if denominator == 0 or denominator > max_denominator:
            continue
        approx_value = numerator / denominator
        error = abs(ratio - approx_value)
        threshold = base_tolerance / max(denominator, 1)
        candidate = RationalApproximation(numerator, denominator, approx_value, error, ratio)
        scaled_error = error / threshold if threshold > 0 else float("inf")
        if best is None or error < best.error:
            best = candidate
            best_threshold = threshold
            best_scaled = scaled_error
        if error <= threshold:
            return candidate, threshold, scaled_error
    return best, best_threshold, best_scaled


def run_diophantine(
    beta1_values: Iterable[float],
    *,
    max_denominator: int = 32,
    tolerance: float = 1e-3,
    output_dir: Path | str = Path("resonance_artifacts"),
) -> DiophantineResult:
    """Analyse β₁ values for proximity to low-order rational ratios.

    Parameters
    ----------
    beta1_values:
        Iterable containing the β₁ slopes produced by :func:`run_beta_stability`.
    max_denominator:
        Largest denominator considered during the continued fraction search.
    tolerance:
        Maximum accepted deviation between the continuous value and the rational
        approximation.
    output_dir:
        Directory used for the diagnostic bar chart.
    """

    slopes = np.asarray(list(beta1_values), dtype=float)
    ratios = slopes / (2.0 * np.pi)
    approximations: List[RationalApproximation | None] = []
    thresholds: List[float] = []
    scaled_errors: List[float] = []
    for ratio in ratios:
        approx, threshold, scaled = _analyse_ratio(ratio, max_denominator, tolerance)
        approximations.append(approx)
        thresholds.append(threshold)
        scaled_errors.append(scaled)

    output_path = ensure_directory(output_dir) / "diophantine_alerts.png"

    fig, ax = plt.subplots(figsize=(8, 4))
    indices = np.arange(len(slopes))
    display = np.array(scaled_errors, dtype=float)
    display = np.nan_to_num(display, nan=2.0, posinf=2.0)
    colors = ["tab:red" if scale <= 1.0 else "tab:blue" for scale in display]
    ax.bar(indices, display, color=colors)
    ax.set_xticks(indices)
    ax.set_xticklabels([str(i) for i in indices])
    ax.set_ylabel("|β₁/(2π) - p/q| / (τ/q)")
    ax.set_title("Scaled Diophantine proximity of β₁ slopes")
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label="threshold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return DiophantineResult(
        slopes=slopes,
        ratios=ratios,
        approximations=approximations,
        thresholds=np.asarray(thresholds, dtype=float),
        scaled_errors=np.asarray(scaled_errors, dtype=float),
        artifact_path=output_path,
    )
