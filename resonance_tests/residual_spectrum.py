"""Residual spectrum analysis for resonance detection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .utils import ensure_directory


@dataclass
class SpecResult:
    """Container for Lomb–Scargle peak metadata."""

    peak_found: bool
    peaks: list[tuple[float, float, float, float]]
    figure_path: str | None


def _lomb_scargle_basic(z: np.ndarray, r: np.ndarray, freqs: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
    """Compute a simple (weighted) Lomb–Scargle periodogram."""

    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    r = r - np.average(r, weights=None if weights is None else weights)
    powers = []
    if weights is None:
        w = np.ones_like(z)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != z.shape:
            raise ValueError("weights must match z")
        w = np.clip(w, 0.0, np.inf)
    sqrt_w = np.sqrt(w)
    for omega in 2.0 * np.pi * freqs:
        cos_term = np.cos(omega * z)
        sin_term = np.sin(omega * z)
        c_proj = np.dot(r * sqrt_w, cos_term * sqrt_w)
        s_proj = np.dot(r * sqrt_w, sin_term * sqrt_w)
        c_norm = np.dot(cos_term * sqrt_w, cos_term * sqrt_w) + 1e-15
        s_norm = np.dot(sin_term * sqrt_w, sin_term * sqrt_w) + 1e-15
        powers.append(0.5 * (c_proj ** 2 / c_norm + s_proj ** 2 / s_norm))
    return np.asarray(powers)


def _global_fap_from_power(power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate local and global false-alarm probabilities."""

    M = len(power)
    scale = np.median(power)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = np.mean(np.abs(power))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    z = power / scale
    p_local = np.exp(-z)
    p_local = np.clip(p_local, 1e-300, 1.0)
    p_global = 1.0 - (1.0 - p_local) ** M
    return p_local, p_global


def run_residual_spectrum(
    z: Iterable[float],
    residuals: Iterable[float],
    weights: Optional[Iterable[float]] = None,
    *,
    alpha: float = 0.05,
    n_freq: int = 2000,
    freq_min: Optional[float] = None,
    freq_max: Optional[float] = None,
    make_plot: Optional[Callable[[np.ndarray, np.ndarray, float], str | None]] = None,
) -> SpecResult:
    """Compute a Lomb–Scargle spectrum and summarise peak statistics."""

    z = np.asarray(list(z), dtype=float)
    r = np.asarray(list(residuals), dtype=float)
    if z.shape != r.shape:
        raise ValueError("z and residuals must match in shape")
    if weights is None:
        w = None
    else:
        w_arr = np.asarray(list(weights), dtype=float)
        if w_arr.shape != z.shape:
            raise ValueError("weights must match z")
        w = w_arr

    baseline = z.max() - z.min()
    if baseline <= 0:
        raise ValueError("z must span a finite range")
    if freq_min is None:
        freq_min = 1.0 / (baseline * 2.0)
    if freq_max is None:
        spacing = np.median(np.diff(np.sort(z)))
        if not np.isfinite(spacing) or spacing <= 0:
            spacing = baseline / max(len(z) - 1, 1)
        freq_max = 0.5 / spacing
    if freq_min <= 0 or freq_max <= freq_min:
        raise ValueError("Invalid frequency bounds")

    freqs = np.linspace(freq_min, freq_max, n_freq)
    power = _lomb_scargle_basic(z, r, freqs, w)
    p_local, p_global = _global_fap_from_power(power)
    order = np.argsort(power)[::-1]
    top = [
        (float(freqs[i]), float(power[i]), float(p_local[i]), float(p_global[i]))
        for i in order[:3]
    ]
    peak_found = any(pg < alpha for (_, _, _, pg) in top)

    fig_path: str | None = None
    if callable(make_plot):
        fig_path = make_plot(freqs, power, alpha)

    return SpecResult(peak_found=peak_found, peaks=top, figure_path=fig_path)


def plot_spectrum(
    freqs: np.ndarray,
    power: np.ndarray,
    alpha: float = 0.05,
    *,
    out: str | Path = "cwf_resonance_results/residual_spectrum.png",
) -> str:
    """Render and persist the Lomb–Scargle spectrum with global thresholds."""

    out_path = Path(out)
    ensure_directory(out_path.parent)
    M = len(power)
    scale = np.median(power)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.mean(np.abs(power))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    level_05 = -np.log(1 - (1 - alpha) ** (1.0 / max(M, 1))) * scale
    level_01 = -np.log(1 - (1 - 0.01) ** (1.0 / max(M, 1))) * scale

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(freqs, power, label="Lomb–Scargle")
    ax.axhline(level_05, ls="--", lw=1, color="k", label="global α=0.05")
    ax.axhline(level_01, ls=":", lw=1, color="k", label="global α=0.01")
    ax.set_xlabel("Frequency [cycles per z]")
    ax.set_ylabel("Lomb–Scargle Power")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return str(out_path)
