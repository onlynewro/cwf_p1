"""CWF cosmology model definition."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import expit

from src.utils.cosmology import omega_radiation_fraction


@dataclass
class CWFModel:
    """CWF extended cosmology model with deviation parameters."""

    name: str = "CWF"
    param_names: tuple[str, ...] = ("h", "Om", "x0", "x1")
    bounds: tuple[tuple[float, float], ...] = (
        (0.55, 0.85),
        (0.1, 0.5),
        (-8.0, 8.0),
        (-8.0, 0.0),
    )
    s_max: float = 0.1
    sigma_soft_limit: float = 0.05
    sigma_prior_width: float = 0.02
    x1_prior_mu: float = -2.0
    x1_prior_sigma: float = 2.0
    omega_b_h2_default: float = 0.02237

    def sigma(self, a: float, x0: float, x1: float) -> float:
        """Deviation function σ(a) = s_max · sigmoid(x0 + x1 (1-a))."""
        return self.s_max * expit(x0 + x1 * (1 - a))

    def E(self, z: float, params: Sequence[float]):
        """Modified Hubble parameter for the CWF model."""
        h, Om, x0, x1 = params
        Or = omega_radiation_fraction(h)
        OL = 1.0 - Om - Or

        a = 1.0 / (1 + z)
        sig = self.sigma(a, x0, x1)

        if 1 - sig <= 0:
            return np.inf

        E2 = (Om * (1 + z) ** 3 + Or * (1 + z) ** 4) / (1 - sig) + OL

        if E2 < 0:
            return np.inf

        return np.sqrt(E2)

    def Hz(self, z, params):
        """Hubble parameter at redshift z."""
        h = params[0]
        H0 = h * 100
        return H0 * self.E(z, params)

    def regularization(self, params, datasets):
        """Soft prior to keep |σ(a)| ≲ 0.05 across sampled redshifts."""
        _, _, x0, x1 = params

        scale = self.sigma_soft_limit
        width = self.sigma_prior_width
        chi2_reg = 0.0

        if self.x1_prior_sigma is not None and self.x1_prior_sigma > 0:
            delta = (x1 - self.x1_prior_mu) / self.x1_prior_sigma
            chi2_reg += float(delta ** 2)

        a_samples = []
        bao_data = datasets.get('bao') if isinstance(datasets, dict) else None
        if bao_data is not None:
            for point in bao_data.data:
                z = point.get('z')
                if z is None or not np.isfinite(z):
                    continue
                a_samples.append(1.0 / (1.0 + z))

        if not a_samples:
            a_samples = np.linspace(0.25, 1.0, 10)

        sigma_vals = [self.sigma(a, x0, x1) for a in a_samples]
        max_sigma = float(np.max(np.abs(sigma_vals)))

        if max_sigma > scale and width > 0:
            excess = max_sigma - scale
            chi2_reg += float((excess / width) ** 2)

        return chi2_reg

    def describe_sigma(self, params, z_values):
        """Return σ(a) evaluated at specified redshifts for diagnostics."""
        _, _, x0, x1 = params
        sigma_map = {}
        for z in z_values:
            a = 1.0 / (1.0 + z)
            sigma_map[z] = self.sigma(a, x0, x1)
        return sigma_map

    def omega_b_h2(self, params):
        return self.omega_b_h2_default
