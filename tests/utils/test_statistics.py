import numpy as np
import pytest

from joint_fit_multiproc import (
    CovarianceComputationError,
    calculate_statistics,
    estimate_covariance,
)


def test_estimate_covariance_returns_identity_for_quadratic():
    optimum = np.array([1.0, -2.0])

    def quadratic(point):
        diff = point - optimum
        return float(diff @ diff)

    result = estimate_covariance(
        quadratic,
        theta=optimum,
        param_names=['a', 'b'],
        bounds=[(-10.0, 10.0), (-10.0, 10.0)],
        dof=5,
    )

    assert result.matrix == pytest.approx(np.eye(2))
    assert result.errors['a'] == pytest.approx(1.0)
    assert result.errors['b'] == pytest.approx(1.0)
    assert result.errors['H0'] is None


def test_estimate_covariance_requires_positive_dof():
    with pytest.raises(CovarianceComputationError):
        estimate_covariance(
            lambda theta: np.sum(theta**2),
            theta=np.array([0.0]),
            param_names=['x'],
            bounds=[(-1.0, 1.0)],
            dof=0,
        )


def test_calculate_statistics_reports_information():
    stats = calculate_statistics(chi2=10.0, n_data=5, n_params=2)

    assert stats['chi2'] == pytest.approx(10.0)
    assert stats['dof'] == 3
    assert stats['chi2_red'] == pytest.approx(10.0 / 3.0)
    assert stats['aic'] == pytest.approx(10.0 + 4.0)
    assert stats['bic'] == pytest.approx(10.0 + 2.0 * np.log(5))


