"""Seven-dimensional cosmology model definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Mapping, Sequence

import numpy as np

from src.utils.cosmology import omega_radiation_fraction


@dataclass
class SevenDModel:
    """7D extended cosmology model with linear deviation parameters."""

    name: str = "7D"
    param_names: tuple[str, ...] = ("h", "Om", "beta0", "beta1")
    default_param_bounds: ClassVar[tuple[tuple[float, float], ...]] = (
        (0.55, 0.85),
        (0.1, 0.5),
        (-0.05, 0.05),
        (-0.05, 0.05),
    )
    bounds: tuple[tuple[float, float], ...] = default_param_bounds
    sigma_eps: float = 0.05
    sigma_soft_limit: float = 0.05
    sigma_prior_width: float = 0.02
    beta1_prior_mu: float = 0.0
    beta1_prior_sigma: float | None = None
    omega_b_h2_default: float = 0.02237

    def sigma(self, a: float, beta0: float, beta1: float) -> float:
        """Deviation function σ(a) = β₀ + β₁ · (1 - a)."""
        return beta0 + beta1 * (1 - a)

    @classmethod
    def from_config(cls, config: Mapping[str, object] | None = None) -> "SevenDModel":
        """Instantiate the model with overrides from a configuration mapping."""

        if not config:
            return cls()

        config_copy = dict(config)
        bounds = list(cls.default_param_bounds)

        beta0_bounds = config_copy.pop('beta0_bounds', None)
        if beta0_bounds is not None:
            if not isinstance(beta0_bounds, (list, tuple)) or len(beta0_bounds) != 2:
                raise ValueError("beta0_bounds must be an iterable of length two")
            bounds[2] = (float(beta0_bounds[0]), float(beta0_bounds[1]))

        beta1_bounds = config_copy.pop('beta1_bounds', None)
        if beta1_bounds is not None:
            if not isinstance(beta1_bounds, (list, tuple)) or len(beta1_bounds) != 2:
                raise ValueError("beta1_bounds must be an iterable of length two")
            bounds[3] = (float(beta1_bounds[0]), float(beta1_bounds[1]))

        config_copy['bounds'] = tuple((float(low), float(high)) for low, high in bounds)

        return cls(**config_copy)

    def _is_sigma_within_limits(self, sigma_value: float) -> bool:
        """Return ``True`` when σ(a) respects stability constraints."""

        if not np.isfinite(sigma_value):
            return False

        if 1.0 - sigma_value <= 0.0:
            return False

        eps = self.sigma_eps
        if eps is not None and eps > 0 and abs(sigma_value) > eps:
            return False

        return True

    def E(self, z: float, params: Sequence[float]):
        """Modified Hubble parameter for 7D model."""
        h, Om, beta0, beta1 = params
        Or = omega_radiation_fraction(h)
        OL = 1.0 - Om - Or

        a = 1.0 / (1 + z)
        sig = self.sigma(a, beta0, beta1)

        if not self._is_sigma_within_limits(sig):
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
        """Soft prior to keep |σ(a)| within the configured envelope."""
        _, _, beta0, beta1 = params

        scale = self.sigma_soft_limit
        width = self.sigma_prior_width
        chi2_reg = 0.0

        if self.beta1_prior_sigma is not None and self.beta1_prior_sigma > 0:
            delta = (beta1 - self.beta1_prior_mu) / self.beta1_prior_sigma
            chi2_reg += float(delta**2)

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

        sigma_vals = [self.sigma(a, beta0, beta1) for a in a_samples]

        hard_limit = self.sigma_eps
        if hard_limit is not None and hard_limit > 0:
            if any(abs(value) > hard_limit or not self._is_sigma_within_limits(value) for value in sigma_vals):
                return np.inf

        max_sigma = float(np.max(np.abs(sigma_vals)))

        if max_sigma > scale and width > 0:
            excess = max_sigma - scale
            chi2_reg += float((excess / width) ** 2)

        return chi2_reg

    def describe_sigma(self, params, z_values):
        """Return σ(a) diagnostics evaluated at specified redshifts."""
        _, _, beta0, beta1 = params
        sigma_map = {}
        for z in z_values:
            a = 1.0 / (1.0 + z)
            sigma_val = self.sigma(a, beta0, beta1)
            sigma_map[z] = {
                'sigma': sigma_val,
                'stable': self._is_sigma_within_limits(sigma_val),
                'within_soft_limit': (
                    abs(sigma_val) <= self.sigma_soft_limit
                    if self.sigma_soft_limit is not None and self.sigma_soft_limit > 0
                    else True
                ),
            }
        return sigma_map

    def omega_b_h2(self, params):
        return self.omega_b_h2_default
