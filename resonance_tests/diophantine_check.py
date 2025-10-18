"""Diophantine-style diagnostics for β slopes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .utils import RationalApproximation, best_rational_approximation, ensure_directory

@dataclass
class DiophantineResult:
    """Summary of the rationality analysis for β₁/(2π)."""

    slopes: np.ndarray
    ratios: np.ndarray
    approximations: List[RationalApproximation | None]
    artifact_path: Path

    def violators(self) -> List[int]:
        """Return indices whose approximation fell within the tolerance."""

        return [idx for idx, approx in enumerate(self.approximations) if approx is not None]


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
    approximations = [
        best_rational_approximation(ratio, max_denominator, tolerance)
        for ratio in ratios
    ]

    output_path = ensure_directory(output_dir) / "diophantine_alerts.png"

    fig, ax = plt.subplots(figsize=(8, 4))
    distances = [approx.error if approx is not None else tolerance for approx in approximations]
    colors = ["tab:red" if approx is not None else "tab:blue" for approx in approximations]
    indices = np.arange(len(slopes))
    ax.bar(indices, distances, color=colors)
    ax.set_xticks(indices)
    ax.set_xticklabels([str(i) for i in indices])
    ax.set_ylabel("|β₁/(2π) - p/q|")
    ax.set_title("Diophantine proximity of β₁ slopes")
    ax.axhline(tolerance, color="k", linestyle="--", linewidth=1, label="tolerance")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return DiophantineResult(slopes=slopes, ratios=ratios, approximations=approximations, artifact_path=output_path)
