"""Common cosmology constants and helper functions.

This module centralises the physical constants used throughout the analysis so
that downstream JSON artefacts remain reproducible even if upstream defaults
change.  The ``COSMO_CONSTANTS_VERSION`` flag should be bumped whenever any of
the numerical values below are updated.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad

COSMO_CONSTANTS_VERSION = "2024-05-planck2018"
C_LIGHT = 299792.458  # km/s, IAU 2015 resolution B2
DEFAULT_RD_MPC = 147.0  # Mpc, DESI DR2 fiducial sound horizon
T_CMB = 2.7255  # K, COBE/FIRAS with Planck 2018 convention
# Ω_γ h^2 for the CMB temperature above (Planck 2018 conventions)
OMEGA_GAMMA_H2 = 2.469e-5 * (T_CMB / 2.7255) ** 4
OMEGA_RADIATION_NEUTRINO_FACTOR = 0.22710731766  # Hu & Sugiyama 1996 form
N_EFF = 3.046


def omega_radiation_fraction(h: float) -> float:
    """Return Ω_r for a given reduced Hubble constant h."""
    if h <= 0:
        return 0.0
    omega_r = OMEGA_GAMMA_H2 * (1.0 + OMEGA_RADIATION_NEUTRINO_FACTOR * N_EFF)
    return omega_r / (h ** 2)


def comoving_distance(z: float, model, params) -> float:
    """Comoving line-of-sight distance DM(z) in Mpc."""
    if z < 0:
        return np.nan

    h = params[0]
    H0 = h * 100.0

    def integrand(zp):
        E_val = model.E(zp, params)
        if not np.isfinite(E_val) or E_val <= 0:
            return 0.0
        return 1.0 / E_val

    try:
        integral, _ = quad(integrand, 0.0, z, limit=500)
    except Exception:
        return np.nan
    return C_LIGHT / H0 * integral


def luminosity_distance(z: float, model, params) -> float:
    """Luminosity distance DL(z) in Mpc."""
    dm = comoving_distance(z, model, params)
    if not np.isfinite(dm):
        return np.nan
    return (1.0 + z) * dm


def angular_diameter_distance(z: float, model, params) -> float:
    """Angular diameter distance DA(z) in Mpc."""
    dm = comoving_distance(z, model, params)
    if not np.isfinite(dm) or (1.0 + z) == 0:
        return np.nan
    return dm / (1.0 + z)


def hu_sugiyama_z_star(omega_b_h2: float, omega_m_h2: float) -> float:
    """Hu & Sugiyama fitting formula for photon decoupling redshift."""
    if omega_b_h2 <= 0 or omega_m_h2 <= 0:
        return np.nan
    g1 = 0.0783 * omega_b_h2 ** (-0.238) / (1.0 + 39.5 * omega_b_h2 ** 0.763)
    g2 = 0.560 / (1.0 + 21.1 * omega_b_h2 ** 1.81)
    return 1048.0 * (1.0 + 0.00124 * omega_b_h2 ** (-0.738)) * (1.0 + g1 * omega_m_h2 ** g2)


def sound_horizon_at_z(model, params, z: float, omega_b_h2: float) -> float:
    """Comoving sound horizon at redshift z."""
    if omega_b_h2 <= 0:
        return np.nan

    def integrand(zp):
        Hz = model.Hz(zp, params)
        if not np.isfinite(Hz) or Hz <= 0:
            return 0.0
        R = 3.0 * omega_b_h2 / (4.0 * OMEGA_GAMMA_H2) * 1.0 / (1.0 + zp)
        return 1.0 / (Hz * np.sqrt(3.0 * (1.0 + R)))

    try:
        integral, _ = quad(integrand, z, np.inf, limit=800)
    except Exception:
        return np.nan
    return C_LIGHT * integral
