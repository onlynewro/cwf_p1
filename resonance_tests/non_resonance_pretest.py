"""Top level orchestration for the resonance pre-test."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .beta_stability import BetaStabilityResult, run_beta_stability
from .diophantine_check import DiophantineResult, run_diophantine
from .residual_spectrum import ResidualSpectrumResult, run_residual_spectrum
from .utils import ensure_directory


@dataclass
class PretestConfiguration:
    """Tunables for :func:`run_non_resonance_pretest`."""

    alpha: float = 0.01
    top_n: int = 5
    num_frequencies: int = 1024
    freq_min: Optional[float] = None
    freq_max: Optional[float] = None
    detrend_order: int = 1
    beta_max_denominator: int = 16
    beta_tolerance: float = 5e-3
    diophantine_max_denominator: int = 32
    diophantine_tolerance: float = 1e-3
    output_dir: Path | str = Path("resonance_artifacts")


@dataclass
class PretestResult:
    """Combined output from the three resonance tests."""

    spectrum: ResidualSpectrumResult
    beta_stability: BetaStabilityResult
    diophantine: DiophantineResult
    decision: str
    summary: str
    artifacts: dict


def run_non_resonance_pretest(
    z: Iterable[float],
    lcdm_residuals: Iterable[float],
    sigma_a: Iterable[float],
    beta_predictions: Iterable[float],
    weights: Optional[Iterable[float]] = None,
    *,
    sigma_residuals: Optional[Iterable[float]] = None,
    config: Optional[PretestConfiguration] = None,
) -> PretestResult:
    """Execute the resonance pre-test pipeline.

    Parameters
    ----------
    z, lcdm_residuals, sigma_a, beta_predictions:
        Core observational inputs shared across the three routines.
    weights:
        Optional per-sample weights.
    sigma_residuals:
        Optional σ(a) residuals used for the secondary spectrum.
    config:
        Optional :class:`PretestConfiguration` overriding defaults.
    """

    if config is None:
        config = PretestConfiguration()

    output_dir = ensure_directory(config.output_dir)

    spectrum = run_residual_spectrum(
        z,
        lcdm_residuals,
        weights,
        sigma_residuals=sigma_residuals,
        freq_min=config.freq_min,
        freq_max=config.freq_max,
        num_frequencies=config.num_frequencies,
        top_n=config.top_n,
        alpha=config.alpha,
        detrend_order=config.detrend_order,
        output_dir=output_dir,
    )

    beta_result = run_beta_stability(
        z,
        sigma_a,
        beta_predictions,
        weights,
        max_denominator=config.beta_max_denominator,
        approximation_tolerance=config.beta_tolerance,
        output_dir=output_dir,
    )

    diophantine = run_diophantine(
        [result.beta1 for result in beta_result.bins],
        max_denominator=config.diophantine_max_denominator,
        tolerance=config.diophantine_tolerance,
        output_dir=output_dir,
    )

    significant_peaks = {
        name: peaks for name, peaks in spectrum.significant_series().items() if peaks
    }
    beta_alerts = beta_result.rational_alerts()
    diophantine_alerts = diophantine.violators()

    summary_lines = []
    if significant_peaks:
        for name, peaks in significant_peaks.items():
            peak_info = ", ".join(
                f"f={peak.frequency:.3g} (FAP={peak.false_alarm_probability:.2e})"
                for peak in peaks
            )
            summary_lines.append(f"{name.upper()} spectrum flagged peaks: {peak_info}")
    else:
        summary_lines.append("Residual spectrum shows no significant peaks.")

    if beta_alerts:
        summary_lines.append(
            "β stability identified rational ratios: "
            + ", ".join(
                f"{result.bin_label}≈{result.approximation.as_string()}"
                for result in beta_alerts
            )
        )
    else:
        summary_lines.append("β stability detected no rational-ratio concerns.")

    if diophantine_alerts:
        summary_lines.append(
            "Diophantine check flagged bins: " + ", ".join(map(str, diophantine_alerts))
        )
    else:
        summary_lines.append("Diophantine check detected no near-resonant ratios.")

    decision = "pass"
    if significant_peaks or beta_alerts or diophantine_alerts:
        decision = "review"

    artifacts = {
        "spectrum_plot": str(spectrum.artifact_path),
        "beta_plot": str(beta_result.artifact_path),
        "beta_table": str(beta_result.table_path),
        "diophantine_plot": str(diophantine.artifact_path),
    }

    return PretestResult(
        spectrum=spectrum,
        beta_stability=beta_result,
        diophantine=diophantine,
        decision=decision,
        summary="\n".join(summary_lines),
        artifacts=artifacts,
    )
