"""Refitting and stability diagnostics for β parameters."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .utils import (
    RationalApproximation,
    best_rational_approximation,
    ensure_directory,
    weighted_linear_regression,
)

@dataclass
class BetaBinResult:
    """Result of the β stability refit for a single redshift bin."""

    bin_label: str
    beta0: float
    beta1: float
    beta0_uncertainty: float
    beta1_uncertainty: float
    chi2: float
    approximation: Optional[RationalApproximation]


@dataclass
class BetaStabilityResult:
    """Aggregate results for β stability."""

    bins: List[BetaBinResult]
    artifact_path: Path
    table_path: Path
    beta1_uncertainty_total: float
    ratio_lock_indices: List[int]

    def rational_alerts(self) -> List[BetaBinResult]:
        """Return bins that triggered a low-order rational approximation alert."""

        return [bin_result for bin_result in self.bins if bin_result.approximation is not None]

    def ratio_lock_alerts(self) -> List[str]:
        """Return the labels of bins exhibiting a rational ratio lock."""

        return [self.bins[idx].bin_label for idx in self.ratio_lock_indices]


def _format_bin_label(index: int, centers: np.ndarray, *, edges: Optional[np.ndarray] = None) -> str:
    if edges is not None and index < len(edges) - 1:
        return f"z∈[{edges[index]:.3g}, {edges[index + 1]:.3g})"
    label = centers[index]
    if isinstance(label, (int, np.integer)):
        return f"bin-{int(label)}"
    return f"bin-{index}"


def run_beta_stability(
    z: Iterable[float],
    sigma_a: Iterable[float],
    beta_predictions: Iterable[float],
    weights: Optional[Iterable[float]] = None,
    *,
    bin_edges: Optional[Iterable[float]] = None,
    max_denominator: int = 16,
    approximation_tolerance: float = 5e-3,
    output_dir: Path | str = Path("resonance_artifacts"),
) -> BetaStabilityResult:
    """Refit β(σ(a)) relations inside each redshift bin.

    Parameters
    ----------
    z:
        Redshift measurements for each sample.
    sigma_a:
        Predicted σ(a) values corresponding to ``z``.
    beta_predictions:
        Model β predictions to be refit as ``β = β0 + β1 σ(a)``.
    weights:
        Optional per-sample weights.
    bin_edges:
        Optional redshift bin edges. When omitted the routine groups by unique
        ``z`` values.
    max_denominator:
        Maximum denominator searched when looking for rational approximations of
        ``β1 / (2π)``.
    approximation_tolerance:
        Maximum absolute error tolerated when matching to a rational number.
    output_dir:
        Directory for the generated plot and summary table.
    """

    z = np.asarray(list(z), dtype=float)
    sigma_a = np.asarray(list(sigma_a), dtype=float)
    beta_predictions = np.asarray(list(beta_predictions), dtype=float)
    if not (z.shape == sigma_a.shape == beta_predictions.shape):
        raise ValueError("Inputs must share the same shape")

    if weights is None:
        weights_arr = np.ones_like(z)
    else:
        weights_arr = np.asarray(list(weights), dtype=float)
        if weights_arr.shape != z.shape:
            raise ValueError("weights must match z")
        weights_arr = np.clip(weights_arr, 0.0, np.inf)

    if bin_edges is not None:
        edges = np.asarray(list(bin_edges), dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("bin_edges must provide at least two edges")
        bin_indices = np.digitize(z, edges) - 1
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
    else:
        edges = None
        bin_centers, bin_indices = np.unique(z, return_inverse=True)

    bin_results: List[BetaBinResult] = []
    slopes: List[float] = []
    bin_labels: List[str] = []

    for idx in range(len(bin_centers)):
        mask = bin_indices == idx
        if not np.any(mask):
            continue
        x = sigma_a[mask]
        y = beta_predictions[mask]
        w = weights_arr[mask]
        params, cov, chi2 = weighted_linear_regression(x, y, w)
        beta0, beta1 = params
        beta0_unc = math.sqrt(max(cov[0, 0], 0.0))
        beta1_unc = math.sqrt(max(cov[1, 1], 0.0))
        ratio = beta1 / (2.0 * math.pi) if beta1 != 0 else 0.0
        approximation = best_rational_approximation(ratio, max_denominator, approximation_tolerance)
        label = _format_bin_label(idx, bin_centers, edges=edges)
        bin_labels.append(label)
        slopes.append(beta1)
        bin_results.append(
            BetaBinResult(
                bin_label=label,
                beta0=float(beta0),
                beta1=float(beta1),
                beta0_uncertainty=float(beta0_unc),
                beta1_uncertainty=float(beta1_unc),
                chi2=float(chi2),
                approximation=approximation,
            )
        )

    output_path = ensure_directory(output_dir) / "beta_slopes.png"
    table_path = ensure_directory(output_dir) / "beta_slopes.csv"

    if not bin_results:
        raise ValueError("No populated bins available for β stability analysis")

    ratio_lock_indices = [idx for idx, result in enumerate(bin_results) if result.approximation is not None]
    beta1_uncertainty_total = math.sqrt(
        sum(result.beta1_uncertainty ** 2 for result in bin_results)
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(
        np.arange(len(slopes)),
        slopes,
        yerr=[result.beta1_uncertainty for result in bin_results],
        fmt="o",
        label="β₁",
    )
    for result in bin_results:
        if result.approximation is None:
            continue
        idx = bin_labels.index(result.bin_label)
        ax.annotate(
            result.approximation.as_string(),
            (idx, result.beta1),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(np.arange(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=45, ha="right")
    ax.set_ylabel("β₁")
    ax.set_title("β stability across redshift bins")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    import csv

    with table_path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "bin",
            "beta0",
            "beta0_unc",
            "beta1",
            "beta1_unc",
            "chi2",
            "rational",
        ])
        for result in bin_results:
            writer.writerow(
                [
                    result.bin_label,
                    f"{result.beta0:.6g}",
                    f"{result.beta0_uncertainty:.6g}",
                    f"{result.beta1:.6g}",
                    f"{result.beta1_uncertainty:.6g}",
                    f"{result.chi2:.6g}",
                    result.approximation.as_string() if result.approximation else "",
                ]
            )

    return BetaStabilityResult(
        bins=bin_results,
        artifact_path=output_path,
        table_path=table_path,
        beta1_uncertainty_total=float(beta1_uncertainty_total),
        ratio_lock_indices=ratio_lock_indices,
    )
