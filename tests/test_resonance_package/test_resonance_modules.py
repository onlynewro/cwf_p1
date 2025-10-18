import math
from pathlib import Path

import numpy as np

from resonance_tests import (
    PretestConfiguration,
    run_beta_stability,
    run_diophantine,
    run_non_resonance_pretest,
    run_residual_spectrum,
    plot_diophantine,
    plot_spectrum,
)


def test_residual_spectrum_identifies_injected_frequency(tmp_path: Path):
    rng = np.random.default_rng(42)
    z = np.linspace(0.0, 5.0, 200)
    signal_freq = 0.4
    lcdm = 0.1 * np.sin(2.0 * np.pi * signal_freq * z)
    lcdm += rng.normal(scale=0.02, size=z.shape)
    figure_path = tmp_path / "spectrum.png"
    result = run_residual_spectrum(
        z,
        lcdm,
        alpha=0.05,
        n_freq=512,
        freq_min=0.1,
        freq_max=1.0,
        make_plot=lambda freqs, power, alpha: plot_spectrum(freqs, power, alpha, out=figure_path),
    )
    peak_freq = result.peaks[0][0]
    assert math.isclose(peak_freq, signal_freq, rel_tol=0.1, abs_tol=0.05)
    assert result.peak_found
    assert result.figure_path is not None and Path(result.figure_path).exists()


def test_beta_stability_recovers_linear_relationship(tmp_path: Path):
    rng = np.random.default_rng(123)
    z = np.repeat([0.3, 0.6, 0.9], 50)
    sigma = np.concatenate([
        np.linspace(0.7, 1.0, 50),
        np.linspace(0.6, 0.9, 50),
        np.linspace(0.5, 0.8, 50),
    ])
    beta = np.concatenate([
        0.8 + 1.5 * sigma[:50],
        1.0 + 0.9 * sigma[50:100],
        0.6 + 1.2 * sigma[100:],
    ])
    beta += rng.normal(scale=0.02, size=beta.shape)
    result = run_beta_stability(
        z,
        sigma,
        beta,
        output_dir=tmp_path,
        approximation_tolerance=1e-2,
    )
    slopes = [bin_result.beta1 for bin_result in result.bins]
    assert any(abs(slope - 1.5) < 0.1 for slope in slopes)
    assert any(abs(slope - 0.9) < 0.1 for slope in slopes)
    assert any(abs(slope - 1.2) < 0.1 for slope in slopes)
    assert Path(result.artifact_path).exists()
    assert Path(result.table_path).exists()


def test_diophantine_flags_low_order_rationals(tmp_path: Path):
    beta1 = 2.0 * math.pi * (3 / 5 + 1e-5)
    sigma_beta1 = 2.0 * math.pi * 5e-6
    plot_path = tmp_path / "dio.png"
    result = run_diophantine(
        beta1=beta1,
        sigma_beta1=sigma_beta1,
        Q=10,
        c=5e-3,
        k_sigma=2.0,
        make_plot=lambda records, c, k_sigma, sigma_x: plot_diophantine(
            records,
            c,
            k_sigma,
            sigma_x,
            out=plot_path,
        ),
    )
    assert any(q == 5 for (_p, q, _delta) in result.viol)
    assert result.barplot_path is not None and Path(result.barplot_path).exists()


def test_pretest_summary_and_decision(tmp_path: Path):
    rng = np.random.default_rng(7)
    z = np.linspace(0.0, 4.0, 120)
    sigma = np.linspace(0.7, 0.9, 120)
    beta = 1.0 + 1.0 * sigma + rng.normal(scale=0.005, size=sigma.shape)
    residuals = 0.05 * np.sin(2.0 * np.pi * 0.5 * z) + rng.normal(scale=0.01, size=z.shape)

    config = PretestConfiguration(
        alpha=0.05,
        n_freq=256,
        freq_min=0.1,
        freq_max=1.0,
        output_dir=tmp_path,
        beta_tolerance=1e-2,
        diophantine_Q=12,
    )

    result = run_non_resonance_pretest(z, residuals, sigma, beta, config=config)
    assert "spectrum" in result.summary.lower()
    assert result.decision in {"pass", "review"}
    for path in result.artifacts.values():
        assert Path(path).exists()
