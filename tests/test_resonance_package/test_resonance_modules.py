import math
from pathlib import Path

import numpy as np

from resonance_tests import (
    PretestConfiguration,
    run_beta_stability,
    run_diophantine,
    run_non_resonance_pretest,
    run_residual_spectrum,
)


def test_residual_spectrum_identifies_injected_frequency(tmp_path: Path):
    rng = np.random.default_rng(42)
    z = np.linspace(0.0, 5.0, 200)
    signal_freq = 0.4
    lcdm = 0.1 * np.sin(2.0 * np.pi * signal_freq * z)
    lcdm += rng.normal(scale=0.02, size=z.shape)
    result = run_residual_spectrum(
        z,
        lcdm,
        None,
        freq_min=0.1,
        freq_max=1.0,
        num_frequencies=256,
        top_n=3,
        output_dir=tmp_path,
    )
    peak_freq = result.peaks["lcdm"][0].frequency
    assert math.isclose(peak_freq, signal_freq, rel_tol=0.1, abs_tol=0.05)
    assert Path(result.artifact_path).exists()


def test_beta_stability_recovers_linear_relationship(tmp_path: Path):
    rng = np.random.default_rng(123)
    z = np.repeat([0.3, 0.6], 50)
    sigma = np.concatenate([
        np.linspace(0.7, 1.0, 50),
        np.linspace(0.6, 0.9, 50),
    ])
    beta = np.concatenate([
        0.8 + 1.5 * sigma[:50],
        1.0 + 0.9 * sigma[50:],
    ])
    beta += rng.normal(scale=0.02, size=beta.shape)
    result = run_beta_stability(z, sigma, beta, output_dir=tmp_path, approximation_tolerance=1e-2)
    slopes = [bin_result.beta1 for bin_result in result.bins]
    assert any(abs(slope - 1.5) < 0.1 for slope in slopes)
    assert any(abs(slope - 0.9) < 0.1 for slope in slopes)
    assert Path(result.artifact_path).exists()
    assert Path(result.table_path).exists()


def test_diophantine_flags_low_order_rationals(tmp_path: Path):
    slopes = [2.0 * math.pi * (3 / 5 + 5e-4), 2.0 * math.pi * (1 / 7 + 1e-2)]
    result = run_diophantine(slopes, max_denominator=10, tolerance=1e-3, output_dir=tmp_path)
    assert result.violators() == [0]
    assert Path(result.artifact_path).exists()


def test_pretest_summary_and_decision(tmp_path: Path):
    rng = np.random.default_rng(7)
    z = np.linspace(0.0, 4.0, 120)
    sigma = np.linspace(0.7, 0.9, 120)
    beta = 1.0 + 1.0 * sigma + rng.normal(scale=0.005, size=sigma.shape)
    residuals = 0.05 * np.sin(2.0 * np.pi * 0.5 * z) + rng.normal(scale=0.01, size=z.shape)

    config = PretestConfiguration(
        freq_min=0.1,
        freq_max=1.0,
        num_frequencies=128,
        top_n=2,
        output_dir=tmp_path,
        beta_tolerance=1e-2,
        diophantine_tolerance=1e-3,
    )

    result = run_non_resonance_pretest(z, residuals, sigma, beta, config=config)
    assert "spectrum" in result.summary.lower()
    assert result.decision in {"pass", "review"}
    for path in result.artifacts.values():
        assert Path(path).exists()
