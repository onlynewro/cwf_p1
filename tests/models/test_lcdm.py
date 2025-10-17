import pytest

from src.models.lcdm import LCDMModel
from src.utils.cosmology import omega_radiation_fraction


def test_E_matches_standard_evolution():
    model = LCDMModel()
    params = (0.7, 0.3)
    z = 1.5
    expected_or = omega_radiation_fraction(params[0])
    expected = (
        params[1] * (1 + z) ** 3
        + expected_or * (1 + z) ** 4
        + (1.0 - params[1] - expected_or)
    ) ** 0.5

    assert model.E(z, params) == pytest.approx(expected)


def test_Hz_scales_with_normalised_E():
    model = LCDMModel()
    params = (0.67, 0.31)
    z = 0.8
    expected = params[0] * 100.0 * model.E(z, params)

    assert model.Hz(z, params) == pytest.approx(expected)
