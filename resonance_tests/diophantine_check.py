"""Diophantine diagnostics for β₁/(2π)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .utils import ensure_directory


@dataclass
class DioResult:
    """Summary statistics for Diophantine analysis."""

    x: float
    delta_min: float
    delta_min_scaled: float
    viol: list[Tuple[int, int, float]]
    records: list[Tuple[int, int, float, float]]
    barplot_path: str | None


def _continued_fraction_convergents(x: float, Qmax: int = 20) -> List[Tuple[int, int]]:
    """Generate continued-fraction convergents with denominator ≤ ``Qmax``."""

    conv: List[Tuple[int, int]] = []
    h_2, h_1 = 0, 1
    k_2, k_1 = 1, 0
    xi = x
    for _ in range(128):
        ai = int(np.floor(xi))
        h = ai * h_1 + h_2
        k = ai * k_1 + k_2
        if k != 0:
            conv.append((h, k))
            if k > Qmax:
                break
        frac = xi - ai
        if frac == 0:
            break
        xi = 1.0 / frac
        h_2, h_1 = h_1, h
        k_2, k_1 = k_1, k
    return [(p, q) for (p, q) in conv if q <= Qmax]


def run_diophantine(
    beta1: float,
    sigma_beta1: float | None = None,
    *,
    Q: int = 20,
    c: float = 5e-3,
    k_sigma: float = 2.0,
    make_plot: Optional[Callable[[Sequence[Tuple[int, int, float, float]], float, float, float], str | None]] = None,
) -> DioResult:
    """Analyse ``β₁/(2π)`` against low-order rationals using convergents."""

    x = beta1 / (2.0 * np.pi)
    sigma_x = 0.0 if sigma_beta1 is None else sigma_beta1 / (2.0 * np.pi)
    convergents = _continued_fraction_convergents(float(x), Qmax=Q)
    if not convergents:
        convergents = [(0, 1)]

    records: List[Tuple[int, int, float, float]] = []
    viol: List[Tuple[int, int, float]] = []
    delta_min = np.inf
    delta_min_scaled = np.inf

    for p, q in convergents:
        delta = abs(x - p / q)
        thr = c / (q * q) + k_sigma * sigma_x
        scaled = delta * (q * q)
        records.append((p, q, delta, scaled))
        if delta < delta_min:
            delta_min = delta
        if scaled < delta_min_scaled:
            delta_min_scaled = scaled
        if delta <= thr:
            viol.append((p, q, delta))

    barplot_path: str | None = None
    if callable(make_plot):
        barplot_path = make_plot(records, c, k_sigma, sigma_x)

    return DioResult(
        x=float(x),
        delta_min=float(delta_min),
        delta_min_scaled=float(delta_min_scaled),
        viol=viol,
        records=records,
        barplot_path=barplot_path,
    )


def plot_diophantine(
    records: Sequence[Tuple[int, int, float, float]],
    c: float,
    k_sigma: float,
    sigma_x: float,
    *,
    out: str | Path = "cwf_resonance_results/diophantine_scaled.png",
) -> str:
    """Create a diagnostic bar plot for ``|x-p/q| q²``."""

    out_path = Path(out)
    ensure_directory(out_path.parent)
    qs = [q for (_, q, _, _) in records]
    y_scaled = [scaled for (*_, scaled) in records]
    thr_scaled = [c + (k_sigma * sigma_x) * (q * q) for q in qs]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(qs)), y_scaled, width=0.8, label=r"$|x-p/q|q^2$")
    ax.plot(range(len(qs)), thr_scaled, "k--", label="threshold")
    ax.set_ylabel(r"$|x-p/q|\,q^2$")
    ax.set_xlabel("Convergents (ascending q)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return str(out_path)
