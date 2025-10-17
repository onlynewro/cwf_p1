"""CMB distance prior loader."""
from __future__ import annotations

import numpy as np

from src.utils.cosmology import (
    C_LIGHT,
    comoving_distance,
    hu_sugiyama_z_star,
    sound_horizon_at_z,
)


class CMBData:
    """Planck 2018 CMB distance prior handler."""

    PLANCK2018_MEAN = np.array([1.7502, 301.471, 0.02236])
    PLANCK2018_COV = np.array([
        [1.7996e-05, 7.2910e-03, -1.1244e-05],
        [7.2910e-03, 3.6510e+00, -2.1980e-03],
        [-1.1244e-05, -2.1980e-03, 4.1600e-06]
    ])

    def __init__(self, mean_vector=None, covariance=None):
        self.mean = np.array(mean_vector if mean_vector is not None else self.PLANCK2018_MEAN, dtype=float)
        self.cov = np.array(covariance if covariance is not None else self.PLANCK2018_COV, dtype=float)
        if self.mean.shape != (3,):
            raise ValueError("CMB mean vector must have three elements (R, l_A, ω_b)")
        if self.cov.shape != (3, 3):
            raise ValueError("CMB covariance must be 3x3")
        self.cov_inv = np.linalg.inv(self.cov)
        self.dimension = 3
        self.last_theory = None
        self.last_chi2 = None

    def _extract_omega_b_h2(self, model, params):
        if hasattr(model, 'omega_b_h2_from_params') and callable(model.omega_b_h2_from_params):
            return model.omega_b_h2_from_params(params)
        if hasattr(model, 'omega_b_h2'):
            omega_method = getattr(model, 'omega_b_h2')
            if callable(omega_method):
                return omega_method(params)
            return omega_method
        return 0.02237

    def _theory_vector(self, model, params):
        h = params[0]
        Om = params[1]
        H0 = h * 100.0
        omega_b_h2 = self._extract_omega_b_h2(model, params)
        omega_m_h2 = Om * h ** 2

        z_star = hu_sugiyama_z_star(omega_b_h2, omega_m_h2)
        if not np.isfinite(z_star) or z_star <= 0:
            return None

        dm_zstar = comoving_distance(z_star, model, params)
        if not np.isfinite(dm_zstar) or dm_zstar <= 0:
            return None

        r_s = sound_horizon_at_z(model, params, z_star, omega_b_h2)
        if not np.isfinite(r_s) or r_s <= 0:
            return None

        R = np.sqrt(Om) * H0 * dm_zstar / C_LIGHT
        l_A = np.pi * dm_zstar / r_s

        return np.array([R, l_A, omega_b_h2], dtype=float)

    def chi2(self, model, params):
        """Calculate chi-squared for CMB data."""
        theory = self._theory_vector(model, params)
        if theory is None or not np.all(np.isfinite(theory)):
            return np.inf
        self.last_theory = theory
        diff = theory - self.mean
        chi2_val = float(diff @ (self.cov_inv @ diff))
        self.last_chi2 = chi2_val
        return chi2_val

    def count_observables(self):
        return self.dimension
