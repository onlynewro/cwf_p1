"""Tests for the SevenDModel linear deviation implementation."""

import numpy as np
import pytest

from src.models.seven_d import SevenDModel


def test_sigma_linear_form():
    model = SevenDModel()
    assert model.sigma(1.0, 0.01, -0.02) == pytest.approx(0.01)
    assert model.sigma(0.5, 0.01, -0.02) == pytest.approx(0.0)
    assert model.sigma(0.0, 0.01, -0.02) == pytest.approx(-0.01)


def test_E_invalid_sigma_returns_inf():
    model = SevenDModel(sigma_eps=0.1)
    valid_params = (0.7, 0.3, 0.02, -0.01)
    invalid_params = (0.7, 0.3, 0.2, 0.0)

    assert np.isfinite(model.E(0.0, valid_params))
    assert np.isinf(model.E(0.0, invalid_params))


def test_regularization_enforces_hard_limit():
    model = SevenDModel(sigma_eps=0.05)
    params_within = (0.7, 0.3, 0.04, 0.0)
    params_exceed = (0.7, 0.3, 0.06, 0.0)

    assert np.isfinite(model.regularization(params_within, {}))
    assert np.isinf(model.regularization(params_exceed, {}))


def test_describe_sigma_reports_stability():
    model = SevenDModel(sigma_eps=0.05)
    params = (0.7, 0.3, 0.01, 0.0)
    diagnostics = model.describe_sigma(params, [0.0, 1.0])

    assert diagnostics[0.0]['stable']
    assert diagnostics[1.0]['stable']
    assert all('sigma' in entry for entry in diagnostics.values())


def test_from_config_overrides_bounds_and_limits():
    cfg = {
        'beta0_bounds': (-0.02, 0.02),
        'beta1_bounds': (-0.03, 0.03),
        'sigma_eps': 0.02,
        'sigma_soft_limit': 0.01,
    }
    model = SevenDModel.from_config(cfg)

    assert model.bounds[2] == (-0.02, 0.02)
    assert model.bounds[3] == (-0.03, 0.03)
    assert model.sigma_eps == 0.02
    assert model.sigma_soft_limit == 0.01
