"""ΛCDM cosmology model definition."""
from __future__ import annotations

from dataclasses import dataclass

from src.utils.cosmology import omega_radiation_fraction


@dataclass
class LCDMModel:
    """Standard ΛCDM cosmology model."""

    name: str = "LCDM"
    param_names: tuple[str, ...] = ("h", "Om")
    bounds: tuple[tuple[float, float], ...] = ((0.55, 0.85), (0.1, 0.5))
    omega_b_h2_default: float = 0.02237

    def E(self, z, params):
        """Hubble parameter normalized by H0."""
        h, Om = params
        Or = omega_radiation_fraction(h)
        OL = 1.0 - Om - Or
        return (Om * (1 + z) ** 3 + Or * (1 + z) ** 4 + OL) ** 0.5

    def Hz(self, z, params):
        """Hubble parameter at redshift z."""
        h = params[0]
        H0 = h * 100  # km/s/Mpc
        return H0 * self.E(z, params)

    def regularization(self, params, datasets):
        return 0.0

    def omega_b_h2(self, params):
        return self.omega_b_h2_default
