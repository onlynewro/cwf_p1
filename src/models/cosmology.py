"""Cosmology models and shared helper functions."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy.integrate import quad
from scipy.special import expit

C_LIGHT = 299_792.458  # km/s
T_CMB = 2.7255  # K
# Ω_γ h^2 for the CMB temperature above (Planck 2018 conventions)
OMEGA_GAMMA_H2 = 2.469e-5 * (T_CMB / 2.7255) ** 4
N_EFF = 3.046


def omega_radiation_fraction(h: float) -> float:
    """Return Ω_r for a given reduced Hubble constant ``h``."""
    if h <= 0:
        return 0.0
    omega_r = OMEGA_GAMMA_H2 * (1.0 + 0.22710731766 * N_EFF)
    return omega_r / (h ** 2)


def comoving_distance(z: float, model: "BaseModel", params: Sequence[float]) -> float:
    """Comoving line-of-sight distance :math:`D_M(z)` in Mpc."""
    if z < 0:
        return float("nan")

    h = params[0]
    H0 = h * 100.0

    def integrand(zp: float) -> float:
        e_val = model.E(zp, params)
        if not np.isfinite(e_val) or e_val <= 0:
            return 0.0
        return 1.0 / e_val

    try:
        integral, _ = quad(integrand, 0.0, z, limit=500)
    except Exception:  # pragma: no cover - defensive programming
        return float("nan")
    return C_LIGHT / H0 * integral


def luminosity_distance(z: float, model: "BaseModel", params: Sequence[float]) -> float:
    """Luminosity distance :math:`D_L(z)` in Mpc."""
    dm = comoving_distance(z, model, params)
    if not np.isfinite(dm):
        return float("nan")
    return (1.0 + z) * dm


def angular_diameter_distance(z: float, model: "BaseModel", params: Sequence[float]) -> float:
    """Angular diameter distance :math:`D_A(z)` in Mpc."""
    dm = comoving_distance(z, model, params)
    if not np.isfinite(dm) or (1.0 + z) == 0:
        return float("nan")
    return dm / (1.0 + z)


def hu_sugiyama_z_star(omega_b_h2: float, omega_m_h2: float) -> float:
    """Hu & Sugiyama fitting formula for the photon decoupling redshift."""
    if omega_b_h2 <= 0 or omega_m_h2 <= 0:
        return float("nan")
    g1 = 0.0783 * omega_b_h2 ** (-0.238) / (1.0 + 39.5 * omega_b_h2 ** 0.763)
    g2 = 0.560 / (1.0 + 21.1 * omega_b_h2 ** 1.81)
    return 1048.0 * (1.0 + 0.00124 * omega_b_h2 ** (-0.738)) * (1.0 + g1 * omega_m_h2 ** g2)


def sound_horizon_at_z(model: "BaseModel", params: Sequence[float], z: float, omega_b_h2: float) -> float:
    """Comoving sound horizon at redshift ``z``."""
    if omega_b_h2 <= 0:
        return float("nan")

    def integrand(zp: float) -> float:
        hz = model.Hz(zp, params)
        if not np.isfinite(hz) or hz <= 0:
            return 0.0
        r_ratio = 3.0 * omega_b_h2 / (4.0 * OMEGA_GAMMA_H2) * 1.0 / (1.0 + zp)
        return 1.0 / (hz * np.sqrt(3.0 * (1.0 + r_ratio)))

    try:
        integral, _ = quad(integrand, z, np.inf, limit=800)
    except Exception:  # pragma: no cover - defensive programming
        return float("nan")
    return C_LIGHT * integral


class BaseModel:
    """Protocol-like base class for cosmological models."""

    name: str
    param_names: List[str]
    bounds: Sequence[Tuple[float, float]]

    def E(self, z: float, params: Sequence[float]) -> float:
        raise NotImplementedError

    def Hz(self, z: float, params: Sequence[float]) -> float:
        raise NotImplementedError

    def regularization(self, params: Sequence[float], datasets) -> float:  # pragma: no cover - interface
        return 0.0

    def omega_b_h2(self, params: Sequence[float]) -> float:  # pragma: no cover - interface
        return 0.02237


class LCDMModel(BaseModel):
    """Standard ΛCDM cosmology model."""

    def __init__(self):
        self.name = "LCDM"
        self.param_names = ["h", "Om"]
        self.bounds = [(0.55, 0.85), (0.1, 0.5)]
        self.omega_b_h2_default = 0.02237

    def E(self, z: float, params: Sequence[float]) -> float:
        h, Om = params
        Or = omega_radiation_fraction(h)
        Ol = 1.0 - Om - Or
        return np.sqrt(Om * (1 + z) ** 3 + Or * (1 + z) ** 4 + Ol)

    def Hz(self, z: float, params: Sequence[float]) -> float:
        h = params[0]
        H0 = h * 100  # km/s/Mpc
        return H0 * self.E(z, params)

    def regularization(self, params: Sequence[float], datasets) -> float:
        return 0.0

    def omega_b_h2(self, params: Sequence[float]) -> float:
        return self.omega_b_h2_default


class SevenDModel(BaseModel):
    """7D extended cosmology model with deviation parameters."""

    def __init__(self):
        self.name = "7D"
        self.param_names = ["h", "Om", "x0", "x1"]
        self.bounds = [(0.55, 0.85), (0.1, 0.5), (-8.0, 8.0), (-8.0, 0.0)]
        self.s_max = 0.1
        self.sigma_soft_limit = 0.05
        self.sigma_prior_width = 0.02
        self.x1_prior_mu = -2.0
        self.x1_prior_sigma = 2.0
        self.omega_b_h2_default = 0.02237

    def sigma(self, a: float, x0: float, x1: float) -> float:
        return self.s_max * expit(x0 + x1 * (1 - a))

    def E(self, z: float, params: Sequence[float]) -> float:
        h, Om, x0, x1 = params
        Or = omega_radiation_fraction(h)
        Ol = 1.0 - Om - Or

        a = 1.0 / (1 + z)
        sig = self.sigma(a, x0, x1)

        if 1 - sig <= 0:
            return float("inf")

        e_squared = (Om * (1 + z) ** 3 + Or * (1 + z) ** 4) / (1 - sig) + Ol
        if e_squared < 0:
            return float("inf")
        return np.sqrt(e_squared)

    def Hz(self, z: float, params: Sequence[float]) -> float:
        h = params[0]
        H0 = h * 100
        return H0 * self.E(z, params)

    def regularization(self, params: Sequence[float], datasets) -> float:
        _, _, x0, x1 = params

        scale = self.sigma_soft_limit
        width = self.sigma_prior_width
        chi2_reg = 0.0

        if self.x1_prior_sigma is not None and self.x1_prior_sigma > 0:
            delta = (x1 - self.x1_prior_mu) / self.x1_prior_sigma
            chi2_reg += float(delta ** 2)

        a_samples: Iterable[float] = []
        bao_data = datasets.get("bao") if isinstance(datasets, dict) else None
        if bao_data is not None:
            samples: List[float] = []
            for point in bao_data.data:
                z = point.get("z")
                if z is None or not np.isfinite(z):
                    continue
                samples.append(1.0 / (1.0 + z))
            if samples:
                a_samples = samples

        if not a_samples:
            a_samples = np.linspace(0.25, 1.0, 10)

        sigma_vals = [self.sigma(a, x0, x1) for a in a_samples]
        max_sigma = float(np.max(np.abs(sigma_vals)))

        if max_sigma > scale and width > 0:
            excess = max_sigma - scale
            chi2_reg += float((excess / width) ** 2)

        return chi2_reg

    def describe_sigma(self, params: Sequence[float], z_values: Iterable[float]):
        _, _, x0, x1 = params
        sigma_map = {}
        for z in z_values:
            a = 1.0 / (1.0 + z)
            sigma_map[z] = self.sigma(a, x0, x1)
        return sigma_map

    def omega_b_h2(self, params: Sequence[float]) -> float:
        return self.omega_b_h2_default


class RDFitWrapper(BaseModel):
    """Wrapper that augments a cosmology model with a free ``r_d`` parameter."""

    def __init__(self, base_model: BaseModel, rd_bounds: Tuple[float, float] = (120.0, 170.0)):
        self.base_model = base_model
        self.param_names = list(base_model.param_names) + ["rd"]
        self.bounds = list(base_model.bounds) + [rd_bounds]
        self.name = f"{base_model.name}+rd"

    def __getattr__(self, item):  # pragma: no cover - passthrough
        if item.startswith("__") or item.startswith("_"):
            raise AttributeError(item)

        base_model = object.__getattribute__(self, "base_model")
        if hasattr(base_model, item):
            return getattr(base_model, item)
        raise AttributeError(item)

    def __getstate__(self):  # pragma: no cover - multiprocessing support
        return {
            "base_model": self.base_model,
            "param_names": self.param_names,
            "bounds": self.bounds,
            "name": self.name,
        }

    def __setstate__(self, state):  # pragma: no cover - multiprocessing support
        self.base_model = state["base_model"]
        self.param_names = state["param_names"]
        self.bounds = state["bounds"]
        self.name = state["name"]

    def _split(self, params: Sequence[float]) -> Tuple[Sequence[float], float]:
        core_len = len(self.base_model.param_names)
        core = params[:core_len]
        rd = params[core_len]
        return core, rd

    def E(self, z: float, params: Sequence[float]) -> float:
        core, _ = self._split(params)
        return self.base_model.E(z, core)

    def Hz(self, z: float, params: Sequence[float]) -> float:
        core, _ = self._split(params)
        return self.base_model.Hz(z, core)

    def regularization(self, params: Sequence[float], datasets) -> float:
        if hasattr(self.base_model, "regularization"):
            core, _ = self._split(params)
            return self.base_model.regularization(core, datasets)
        return 0.0

    def omega_b_h2(self, params: Sequence[float]) -> float:
        if hasattr(self.base_model, "omega_b_h2"):
            core, _ = self._split(params)
            return self.base_model.omega_b_h2(core)
        return 0.02237


__all__ = [
    "C_LIGHT",
    "T_CMB",
    "OMEGA_GAMMA_H2",
    "N_EFF",
    "omega_radiation_fraction",
    "comoving_distance",
    "luminosity_distance",
    "angular_diameter_distance",
    "hu_sugiyama_z_star",
    "sound_horizon_at_z",
    "LCDMModel",
    "SevenDModel",
    "RDFitWrapper",
]
