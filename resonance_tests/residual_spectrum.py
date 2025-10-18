"""Residual spectrum analysis for resonance detection."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .utils import ensure_directory, estimate_fap, lomb_scargle_power, weighted_linear_regression

@dataclass
class SpectrumPeak:
    """Container holding information about a detected Lomb–Scargle peak."""

    frequency: float
    power: float
    false_alarm_probability: float


@dataclass
class ResidualSpectrumResult:
    """Structured output from :func:`run_residual_spectrum`."""

    frequencies: np.ndarray
    powers: Dict[str, np.ndarray]
    peaks: Dict[str, List[SpectrumPeak]]
    alpha: float
    artifact_path: Path

    def significant_series(self) -> Dict[str, List[SpectrumPeak]]:
        """Return the peaks that lie below the configured false alarm level."""

        return {
            name: [peak for peak in peaks if peak.false_alarm_probability <= self.alpha]
            for name, peaks in self.peaks.items()
        }


def _detrend(series: np.ndarray, sample_points: np.ndarray, weights: np.ndarray, order: int) -> np.ndarray:
    """Detrend ``series`` with a weighted polynomial fit."""

    if order <= 0:
        return series
    x = np.asarray(sample_points, dtype=float)
    y = np.asarray(series, dtype=float)
    W = np.asarray(weights, dtype=float)
    if order == 1:
        params, _, _ = weighted_linear_regression(x, y, W)
        model = params[0] + params[1] * x
    else:
        V = np.vander(x, N=order + 1, increasing=True)
        WV = V * np.sqrt(W)[:, None]
        Wy = y * np.sqrt(W)
        params, *_ = np.linalg.lstsq(WV, Wy, rcond=None)
        model = V @ params
    return y - model


def _prepare_frequency_grid(z: np.ndarray, freq_min: Optional[float], freq_max: Optional[float], num: int) -> np.ndarray:
    """Return a uniformly sampled frequency grid based on ``z`` spacing."""

    z = np.asarray(z, dtype=float)
    baseline = z.max() - z.min()
    if baseline <= 0:
        raise ValueError("z must span a finite range")
    if freq_min is None:
        freq_min = 1.0 / (baseline * 10.0)
    if freq_max is None:
        spacing = np.median(np.diff(np.sort(z)))
        freq_max = 0.5 / max(spacing, 1e-6)
    if freq_min <= 0 or freq_max <= freq_min:
        raise ValueError("Invalid frequency limits")
    return np.linspace(freq_min, freq_max, num=num)


def _compute_series(
    label: str,
    z: np.ndarray,
    series: np.ndarray,
    weights: np.ndarray,
    angular_frequencies: np.ndarray,
    top_n: int,
    alpha: float,
) -> List[SpectrumPeak]:
    """Compute and rank Lomb–Scargle peaks for the provided ``series``."""

    power = lomb_scargle_power(z, series, weights, angular_frequencies)
    trials = len(angular_frequencies)
    fap = estimate_fap(power, trials)
    n_peaks = min(top_n, len(power))
    top_indices = np.argpartition(power, -n_peaks)[-n_peaks:]
    ordering = top_indices[np.argsort(power[top_indices])[::-1]]
    peaks = [
        SpectrumPeak(
            frequency=angular_frequencies[idx] / (2.0 * math.pi),
            power=float(power[idx]),
            false_alarm_probability=float(fap[idx]),
        )
        for idx in ordering
    ]
    return peaks


def run_residual_spectrum(
    z: Iterable[float],
    lcdm_residuals: Iterable[float],
    weights: Optional[Iterable[float]] = None,
    *,
    sigma_residuals: Optional[Iterable[float]] = None,
    freq_min: Optional[float] = None,
    freq_max: Optional[float] = None,
    num_frequencies: int = 1024,
    top_n: int = 5,
    alpha: float = 0.01,
    detrend_order: int = 1,
    output_dir: Path | str = Path("resonance_artifacts"),
) -> ResidualSpectrumResult:
    """Analyse residuals with a weighted Lomb–Scargle spectrum.

    Parameters
    ----------
    z:
        Sample locations, typically redshifts.
    lcdm_residuals:
        Residuals for the ΛCDM prediction.
    weights:
        Optional sample weights.
    sigma_residuals:
        Residuals for the σ(a) extension. When provided a second spectrum is
        produced.
    freq_min, freq_max:
        Lower/upper limits of the scanned frequency interval in cycles per unit
        ``z``.
    num_frequencies:
        Number of frequency samples.
    top_n:
        Number of peaks recorded in the metadata for each residual series.
    alpha:
        False-alarm probability threshold used for significance classification.
    detrend_order:
        Order of the polynomial trend removed prior to computing the spectrum.
    output_dir:
        Directory used for storing the diagnostic plot.

    Returns
    -------
    ResidualSpectrumResult
        Structured container with peak metadata and the plot location.
    """

    z = np.asarray(list(z), dtype=float)
    lcdm_residuals = np.asarray(list(lcdm_residuals), dtype=float)
    if z.shape != lcdm_residuals.shape:
        raise ValueError("z and lcdm_residuals must match in shape")
    if weights is None:
        weights_arr = np.ones_like(z)
    else:
        weights_arr = np.asarray(list(weights), dtype=float)
        if weights_arr.shape != z.shape:
            raise ValueError("weights must match z")
        weights_arr = np.clip(weights_arr, 0.0, np.inf)

    sigma_array: Optional[np.ndarray]
    if sigma_residuals is not None:
        sigma_array = np.asarray(list(sigma_residuals), dtype=float)
        if sigma_array.shape != z.shape:
            raise ValueError("sigma_residuals must match z")
    else:
        sigma_array = None

    freq_grid = _prepare_frequency_grid(z, freq_min, freq_max, num=num_frequencies)
    angular = 2.0 * math.pi * freq_grid

    series_dict = {"lcdm": lcdm_residuals}
    if sigma_array is not None:
        series_dict["sigma"] = sigma_array

    detrended = {
        name: _detrend(series, z, weights_arr, detrend_order)
        for name, series in series_dict.items()
    }

    power_dict: Dict[str, np.ndarray] = {}
    peaks_dict: Dict[str, List[SpectrumPeak]] = {}

    for name, series in detrended.items():
        power = lomb_scargle_power(z, series, weights_arr, angular)
        power_dict[name] = power
        peaks = _compute_series(name, z, series, weights_arr, angular, top_n, alpha)
        peaks_dict[name] = peaks

    output_path = ensure_directory(output_dir) / "residual_spectrum.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, power in power_dict.items():
        ax.plot(freq_grid, power, label=name.upper())
    ax.set_xlabel("Frequency [cycles per z]")
    ax.set_ylabel("Lomb–Scargle Power")
    ax.set_title("Residual Lomb–Scargle Spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return ResidualSpectrumResult(
        frequencies=freq_grid,
        powers=power_dict,
        peaks=peaks_dict,
        alpha=alpha,
        artifact_path=output_path,
    )
