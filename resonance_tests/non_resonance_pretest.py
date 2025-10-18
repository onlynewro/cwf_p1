"""Top level orchestration for the resonance pre-test."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from .beta_stability import BetaStabilityResult, run_beta_stability
from .diophantine_check import DioResult, plot_diophantine, run_diophantine
from .residual_spectrum import SpecResult, plot_spectrum, run_residual_spectrum
from .utils import ensure_directory


@dataclass
class PretestConfiguration:
    """Tunables for :func:`run_non_resonance_pretest`."""

    alpha: float = 0.05
    n_freq: int = 2000
    freq_min: Optional[float] = None
    freq_max: Optional[float] = None
    beta_max_denominator: int = 16
    beta_tolerance: float = 5e-3
    ratio_q_max: int = 10
    ratio_tau: float = 2.0
    ratio_min_consecutive: int = 2
    diophantine_Q: int = 20
    diophantine_c: float = 5e-3
    diophantine_k_sigma: float = 2.0
    output_dir: Path | str = Path("resonance_artifacts")


@dataclass
class PretestResult:
    """Combined output from the three resonance tests."""

    spectrum: SpecResult
    beta_stability: BetaStabilityResult
    diophantine: DioResult
    decision: str
    summary: str
    artifacts: Dict[str, str]


def _combine_slopes(beta_result: BetaStabilityResult) -> tuple[float, float | None]:
    slopes = np.array([bin_result.beta1 for bin_result in beta_result.bins], dtype=float)
    uncertainties = np.array([bin_result.beta1_uncertainty for bin_result in beta_result.bins], dtype=float)
    if not len(slopes):
        return float("nan"), None
    valid = uncertainties > 0
    if not np.any(valid):
        return float(np.mean(slopes)), None
    weights = 1.0 / (uncertainties[valid] ** 2)
    mean = float(np.average(slopes[valid], weights=weights))
    sigma = float(np.sqrt(1.0 / np.sum(weights)))
    return mean, sigma


def run_non_resonance_pretest(
    z: Iterable[float],
    lcdm_residuals: Iterable[float],
    sigma_a: Iterable[float],
    beta_predictions: Iterable[float],
    weights: Optional[Iterable[float]] = None,
    *,
    config: Optional[PretestConfiguration] = None,
) -> PretestResult:
    """Execute the resonance pre-test pipeline."""

    if config is None:
        config = PretestConfiguration()

    output_dir = ensure_directory(config.output_dir)

    spectrum_plot_path = output_dir / "residual_spectrum.png"
    spectrum = run_residual_spectrum(
        z,
        lcdm_residuals,
        weights,
        alpha=config.alpha,
        n_freq=config.n_freq,
        freq_min=config.freq_min,
        freq_max=config.freq_max,
        make_plot=lambda freqs, power, alpha: plot_spectrum(freqs, power, alpha, out=spectrum_plot_path),
    )

    beta_result = run_beta_stability(
        z,
        sigma_a,
        beta_predictions,
        weights,
        max_denominator=config.beta_max_denominator,
        approximation_tolerance=config.beta_tolerance,
        output_dir=output_dir,
        ratio_q_max=config.ratio_q_max,
        ratio_tau=config.ratio_tau,
        ratio_min_consecutive=config.ratio_min_consecutive,
    )

    beta_mean, beta_sigma = _combine_slopes(beta_result)
    diophantine_plot_path = output_dir / "diophantine_scaled.png"
    diophantine = run_diophantine(
        beta_mean,
        beta_sigma,
        Q=config.diophantine_Q,
        c=config.diophantine_c,
        k_sigma=config.diophantine_k_sigma,
        make_plot=lambda records, c, k_sigma, sigma_x: plot_diophantine(
            records,
            c,
            k_sigma,
            sigma_x,
            out=diophantine_plot_path,
        ),
    )

    summary_lines = []
    if spectrum.peaks and spectrum.peak_found:
        flagged = [peak for peak in spectrum.peaks if peak[3] < config.alpha]
        if flagged:
            peak_text = ", ".join(
                f"f={f:.3g} (p_global={pg:.2e})" for (f, _p, _pl, pg) in flagged
            )
            summary_lines.append(f"Spectrum flagged peaks: {peak_text}")
        else:
            summary_lines.append("Spectrum shows peaks but none cross α globally.")
    else:
        summary_lines.append("Residual spectrum shows no significant peaks.")

    beta_rational = beta_result.rational_alerts()
    beta_lock = beta_result.ratio_lock_alerts()
    if beta_rational:
        summary_lines.append(
            "β stability rational approximations: "
            + ", ".join(f"{b.bin_label}≈{b.approximation.as_string()}" for b in beta_rational)
        )
    else:
        summary_lines.append("β stability detected no direct rational approximations.")
    if beta_lock:
        summary_lines.append(
            "β stability ratio lock alerts: " + ", ".join(b.bin_label for b in beta_lock)
        )

    if diophantine.viol:
        viol_text = ", ".join(f"{p}/{q}" for (p, q, _delta) in diophantine.viol)
        summary_lines.append(f"Diophantine check violations: {viol_text}")
    else:
        summary_lines.append("Diophantine check detected no near-resonant ratios.")

    decision = "pass"
    if (spectrum.peak_found and spectrum.peaks and any(pg < config.alpha for *_, pg in spectrum.peaks)) or beta_rational or beta_lock or diophantine.viol:
        decision = "review"

    artifacts: Dict[str, str] = {
        "spectrum_plot": spectrum.figure_path or str(spectrum_plot_path),
        "beta_plot": str(beta_result.artifact_path),
        "beta_table": str(beta_result.table_path),
        "diophantine_plot": diophantine.barplot_path or str(diophantine_plot_path),
    }

    return PretestResult(
        spectrum=spectrum,
        beta_stability=beta_result,
        diophantine=diophantine,
        decision=decision,
        summary="\n".join(summary_lines),
        artifacts=artifacts,
    )
