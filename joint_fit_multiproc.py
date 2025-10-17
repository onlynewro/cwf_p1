#!/usr/bin/env python3
"""
joint_fit_patched_multiproc5.py
7차원 우주론 모델과 ΛCDM 모델의 BAO, SNe, CMB 데이터 합동 피팅
멀티프로세싱 지원 및 오류 처리 강화 버전
"""

import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype
import argparse
import json
import sys
import os
import copy
from functools import partial
from dataclasses import dataclass
from contextlib import contextmanager
from scipy.optimize import differential_evolution
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import expit
from scipy.linalg import cho_factor, cho_solve
import warnings
warnings.filterwarnings('ignore')


class CovarianceComputationError(RuntimeError):
    """Raised when the numerical covariance estimation fails."""


@dataclass
class CovarianceResult:
    matrix: np.ndarray
    correlation: np.ndarray
    errors: dict
    condition_number: float

C_LIGHT = 299792.458  # km/s
T_CMB = 2.7255  # K
# Ω_γ h^2 for the CMB temperature above (Planck 2018 conventions)
OMEGA_GAMMA_H2 = 2.469e-5 * (T_CMB / 2.7255) ** 4
N_EFF = 3.046


def omega_radiation_fraction(h):
    """Return Ω_r for a given reduced Hubble constant h."""
    if h <= 0:
        return 0.0
    omega_r = OMEGA_GAMMA_H2 * (1.0 + 0.22710731766 * N_EFF)
    return omega_r / (h ** 2)


def comoving_distance(z, model, params):
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


def luminosity_distance(z, model, params):
    """Luminosity distance DL(z) in Mpc."""
    dm = comoving_distance(z, model, params)
    if not np.isfinite(dm):
        return np.nan
    return (1.0 + z) * dm


def angular_diameter_distance(z, model, params):
    """Angular diameter distance DA(z) in Mpc."""
    dm = comoving_distance(z, model, params)
    if not np.isfinite(dm) or (1.0 + z) == 0:
        return np.nan
    return dm / (1.0 + z)


def hu_sugiyama_z_star(omega_b_h2, omega_m_h2):
    """Hu & Sugiyama fitting formula for photon decoupling redshift."""
    if omega_b_h2 <= 0 or omega_m_h2 <= 0:
        return np.nan
    g1 = 0.0783 * omega_b_h2 ** (-0.238) / (1.0 + 39.5 * omega_b_h2 ** 0.763)
    g2 = 0.560 / (1.0 + 21.1 * omega_b_h2 ** 1.81)
    return 1048.0 * (1.0 + 0.00124 * omega_b_h2 ** (-0.738)) * (1.0 + g1 * omega_m_h2 ** g2)


def sound_horizon_at_z(model, params, z, omega_b_h2):
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

# 멀티프로세싱 관련 설정
import multiprocessing as mp
if sys.platform == "darwin":  # macOS
    mp.set_start_method('fork', force=True)

# ==================== 우주론 모델 정의 ====================

class LCDMModel:
    """Standard ΛCDM cosmology model"""
    def __init__(self):
        self.name = "LCDM"
        self.param_names = ['h', 'Om']
        # Allow a wider and more physical parameter volume to prevent
        # artificial boundary sticking during optimization.
        self.bounds = [(0.55, 0.85), (0.1, 0.5)]
        self.omega_b_h2_default = 0.02237

    def E(self, z, params):
        """Hubble parameter normalized by H0"""
        h, Om = params
        Or = omega_radiation_fraction(h)
        OL = 1.0 - Om - Or
        return np.sqrt(Om * (1 + z)**3 + Or * (1 + z)**4 + OL)
    
    def Hz(self, z, params):
        """Hubble parameter at redshift z"""
        h = params[0]
        H0 = h * 100  # km/s/Mpc
        return H0 * self.E(z, params)

    def regularization(self, params, datasets):
        return 0.0

    def omega_b_h2(self, params):
        return self.omega_b_h2_default

class SevenDModel:
    """7D extended cosmology model with deviation parameters"""
    def __init__(self):
        self.name = "7D"
        self.param_names = ['h', 'Om', 'x0', 'x1']
        # Allow wider bounds on the re-parameterised logistic coefficients.
        # Enforce a non-positive slope (x1 ≤ 0) so σ(a) monotonically
        # decreases as the scale factor grows, matching the physical prior
        # used in the 7D analysis.
        self.bounds = [(0.55, 0.85), (0.1, 0.5), (-8.0, 8.0), (-8.0, 0.0)]
        self.s_max = 0.1
        self.sigma_soft_limit = 0.05
        # A slightly wider prior keeps σ(a) physical without over-constraining
        # the optimizer, preventing boundary sticking while honouring |σ|≲0.05.
        self.sigma_prior_width = 0.02
        # Mild Gaussian prior on the slope term to discourage sticking to the
        # hard boundary while keeping the fit flexible. The width can be tuned
        # alongside the data volume if needed.
        self.x1_prior_mu = -2.0
        self.x1_prior_sigma = 2.0
        self.omega_b_h2_default = 0.02237

    def sigma(self, a, x0, x1):
        """Deviation function σ(a) = s_max · sigmoid(x0 + x1 (1-a))."""
        return self.s_max * expit(x0 + x1 * (1 - a))
    
    def E(self, z, params):
        """Modified Hubble parameter for 7D model"""
        h, Om, x0, x1 = params
        Or = omega_radiation_fraction(h)
        OL = 1.0 - Om - Or

        a = 1.0 / (1 + z)
        sig = self.sigma(a, x0, x1)

        # Ensure stability: 1 - σ(a) > 0
        if 1 - sig <= 0:
            return np.inf

        E2 = (Om * (1 + z)**3 + Or * (1 + z)**4) / (1 - sig) + OL
        
        if E2 < 0:
            return np.inf
        
        return np.sqrt(E2)
    
    def Hz(self, z, params):
        """Hubble parameter at redshift z"""
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


class RDFitWrapper:
    """Wrapper that augments a cosmology model with a free r_d parameter."""

    def __init__(self, base_model, rd_bounds=(120.0, 170.0)):
        self.base_model = base_model
        self.param_names = list(base_model.param_names) + ['rd']
        self.bounds = list(base_model.bounds) + [rd_bounds]
        self.name = f"{base_model.name}+rd"

    def __getattr__(self, item):
        if item.startswith('__') or item.startswith('_'):
            raise AttributeError(item)

        try:
            base_model = object.__getattribute__(self, 'base_model')
        except AttributeError:
            raise AttributeError(item) from None

        if hasattr(base_model, item):
            return getattr(base_model, item)

        raise AttributeError(item)

    def __getstate__(self):
        return {
            'base_model': self.base_model,
            'param_names': self.param_names,
            'bounds': self.bounds,
            'name': self.name,
        }

    def __setstate__(self, state):
        self.base_model = state['base_model']
        self.param_names = state['param_names']
        self.bounds = state['bounds']
        self.name = state['name']

    def _split(self, params):
        core_len = len(self.base_model.param_names)
        core = params[:core_len]
        rd = params[core_len]
        return core, rd

    def E(self, z, params):
        core, _ = self._split(params)
        return self.base_model.E(z, core)

    def Hz(self, z, params):
        core, _ = self._split(params)
        return self.base_model.Hz(z, core)

    def regularization(self, params, datasets):
        if hasattr(self.base_model, 'regularization'):
            core, _ = self._split(params)
            return self.base_model.regularization(core, datasets)
        return 0.0

    def omega_b_h2(self, params):
        if hasattr(self.base_model, 'omega_b_h2'):
            core, _ = self._split(params)
            return self.base_model.omega_b_h2(core)
        return 0.02237

# ==================== 데이터 로더 ====================

class BAOData:
    """BAO data handler"""

    OFFICIAL_DATASETS = [
        {
            'name': 'DESI LRG GCcomb z=0.4-0.6',
            'mean_file': 'desi_2024_gaussian_bao_LRG_GCcomb_z0.4-0.6_mean.txt',
            'cov_file': 'desi_2024_gaussian_bao_LRG_GCcomb_z0.4-0.6_cov.txt'
        },
        {
            'name': 'DESI LRG GCcomb z=0.6-0.8',
            'mean_file': 'desi_2024_gaussian_bao_LRG_GCcomb_z0.6-0.8_mean.txt',
            'cov_file': 'desi_2024_gaussian_bao_LRG_GCcomb_z0.6-0.8_cov.txt'
        },
        {
            'name': 'DESI Lyα GCcomb',
            'mean_file': 'desi_2024_gaussian_bao_Lya_GCcomb_mean.txt',
            'cov_file': 'desi_2024_gaussian_bao_Lya_GCcomb_cov.txt'
        }
    ]

    def __init__(self, filename=None, use_official_covariance=True, include_proxy=True):
        self.data = []
        self._base_data = []
        self._covariance_used_last = False
        self._default_use_covariance = use_official_covariance
        self._include_proxy = include_proxy
        if filename and os.path.exists(filename):
            self.load_from_file(filename)
        else:
            self.load_default(use_official_covariance=use_official_covariance,
                               include_proxy=include_proxy)

    def load_default(self, use_official_covariance=True, include_proxy=True):
        """Load official DESI BAO data with optional covariance usage."""
        try:
            entries = []
            for spec in self.OFFICIAL_DATASETS:
                entry = self._load_official_entry(spec, use_official_covariance)
                entries.append(entry)

            if include_proxy:
                qso_fallback = {
                    'name': 'Legacy QSO proxy',
                    'z': 1.48,
                    'DM_over_rd': 26.07,
                    'err_DM': 0.67,
                    'DH_over_rd': None,
                    'err_DH': None
                }
                entries.append(self._normalize_entry(qso_fallback))

            self.data = entries
            self._base_data = copy.deepcopy(self.data)
        except FileNotFoundError:
            # Fallback to legacy static table if official files are unavailable.
            default_entries = [
                {'name': 'LRG_0.6', 'z': 0.51, 'DM_over_rd': 13.62, 'err_DM': 0.25,
                 'DH_over_rd': None, 'err_DH': None},
                {'name': 'LRG_0.8', 'z': 0.71, 'DM_over_rd': 16.85, 'err_DM': 0.33,
                 'DH_over_rd': None, 'err_DH': None},
                {'name': 'QSO', 'z': 1.48, 'DM_over_rd': 26.07, 'err_DM': 0.67,
                 'DH_over_rd': None, 'err_DH': None},
                {'name': 'Lya_1', 'z': 2.33, 'DM_over_rd': 37.41, 'err_DM': 1.86,
                 'DH_over_rd': 9.08, 'err_DH': 0.34},
            ]
            self.data = [self._normalize_entry(entry) for entry in default_entries]
            self._base_data = copy.deepcopy(self.data)

    def load_from_file(self, filename):
        """Load BAO data from JSON file"""
        try:
            with open(filename, 'r') as f:
                raw_data = json.load(f)
                self.data = [self._normalize_entry(entry) for entry in raw_data]
                self._base_data = copy.deepcopy(self.data)
        except Exception as e:
            print(f"Warning: Could not load BAO data from {filename}: {e}")
            print("Using default BAO data...")
            self.load_default(use_official_covariance=self._default_use_covariance)

    def _load_official_entry(self, spec, use_official_covariance):
        mean_path = spec['mean_file']
        cov_path = spec.get('cov_file')

        if not os.path.exists(mean_path):
            raise FileNotFoundError(mean_path)

        raw_order = []
        values = {}
        z_ref = None

        with open(mean_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                z_val = float(parts[0])
                quantity_value = float(parts[1])
                quantity_name = parts[2]

                norm_name = quantity_name.replace('rs', 'rd')
                raw_order.append(norm_name)
                values[norm_name] = quantity_value
                z_ref = z_val

        entry = {
            'name': spec['name'],
            'z': z_ref,
            'DM_over_rd': values.get('DM_over_rd'),
            'DH_over_rd': values.get('DH_over_rd'),
            'err_DM': None,
            'err_DH': None
        }

        target_order = []
        if entry['DM_over_rd'] is not None:
            target_order.append('DM_over_rd')
        if entry['DH_over_rd'] is not None:
            target_order.append('DH_over_rd')

        if use_official_covariance and cov_path and os.path.exists(cov_path):
            cov = np.loadtxt(cov_path)
            cov = np.array(cov, dtype=float)
            if cov.ndim == 0:
                cov = cov.reshape(1, 1)

            perm = [raw_order.index(q) for q in target_order]
            cov = cov[np.ix_(perm, perm)]

            if cov.shape[0] != len(target_order):
                raise ValueError(
                    f"Covariance/order mismatch for {spec['name']} ({cov.shape} vs {len(target_order)})"
                )

            if np.any(np.diag(cov) <= 0):
                raise ValueError(f"Non-positive diagonal in covariance for {spec['name']}")

            entry['cov_matrix'] = cov
            diag = np.sqrt(np.diag(cov))
            for idx, key in enumerate(target_order):
                if key == 'DM_over_rd':
                    entry['err_DM'] = float(diag[idx])
                elif key == 'DH_over_rd':
                    entry['err_DH'] = float(diag[idx])
        else:
            # Fall back to diagonal uncertainties when covariance is not applied.
            if cov_path and os.path.exists(cov_path):
                diag_cov = np.loadtxt(cov_path)
                diag_cov = np.array(diag_cov, dtype=float)
                diag_cov = np.atleast_2d(diag_cov)
                diag_vals = np.diag(diag_cov)
            else:
                diag_vals = None

            if diag_vals is not None and diag_vals.size >= len(target_order):
                for idx, key in enumerate(target_order):
                    err = float(np.sqrt(diag_vals[idx]))
                    if key == 'DM_over_rd':
                        entry['err_DM'] = err
                    elif key == 'DH_over_rd':
                        entry['err_DH'] = err

        entry['_observable_order'] = target_order
        return self._normalize_entry(entry)

    @staticmethod
    def _normalize_entry(entry):
        """Normalize legacy field names to the current convention."""
        normalized = dict(entry)

        if 'err' in normalized and 'err_DM' not in normalized:
            normalized['err_DM'] = normalized.pop('err')
        if 'err_Hz' in normalized and 'err_DH' not in normalized:
            normalized['err_DH'] = normalized.pop('err_Hz')
        if 'Hz_rd' in normalized and 'DH_over_rd' not in normalized:
            normalized['DH_over_rd'] = normalized.pop('Hz_rd')

        # Ensure missing keys exist explicitly
        normalized.setdefault('DM_over_rd', None)
        normalized.setdefault('DH_over_rd', None)
        normalized.setdefault('err_DM', None)
        normalized.setdefault('err_DH', None)
        normalized.setdefault('name', 'BAO_point')
        normalized.setdefault('z', np.nan)
        normalized.setdefault('_observable_order', [])

        return normalized

    def restore_original_configuration(self):
        """Restore the BAO data to the initially loaded state."""
        if self._base_data:
            self.data = copy.deepcopy(self._base_data)

    def remove_all_covariances(self):
        """Convert all covariance matrices to diagonal variances only."""
        for point in self.data:
            cov = point.get('cov_matrix')
            if cov is None:
                continue
            cov = np.array(cov, dtype=float)
            variances = np.diag(cov)
            point['cov_matrix'] = np.diag(variances)

    def drop_observable(self, point_name, quantity):
        """Drop a specific observable from a BAO entry (e.g. Lyα DH/rd)."""
        quantity_key = None
        err_key = None
        if quantity.lower().startswith('dm'):
            quantity_key = 'DM_over_rd'
            err_key = 'err_DM'
        elif quantity.lower().startswith('dh'):
            quantity_key = 'DH_over_rd'
            err_key = 'err_DH'
        else:
            raise ValueError(f"Unsupported quantity '{quantity}' for dropout")

        for point in self.data:
            if point.get('name') != point_name:
                continue

            if point.get(quantity_key) is None:
                return False

            point[quantity_key] = None
            point[err_key] = None

            order = list(point.get('_observable_order', []))
            if quantity_key in order:
                idx = order.index(quantity_key)
                order.pop(idx)
                cov = point.get('cov_matrix')
                if cov is not None:
                    cov = np.array(cov, dtype=float)
                    cov = np.delete(np.delete(cov, idx, axis=0), idx, axis=1)
                    if cov.size == 0:
                        cov = None
                    else:
                        cov = np.atleast_2d(cov)
                point['cov_matrix'] = cov

                # Update the stored order and refresh remaining errors.
                point['_observable_order'] = order
                if cov is not None and cov.size:
                    diag = np.sqrt(np.diag(cov))
                    for i, key in enumerate(order):
                        if key == 'DM_over_rd':
                            point['err_DM'] = float(diag[i])
                        elif key == 'DH_over_rd':
                            point['err_DH'] = float(diag[i])
            return True

        return False

    @contextmanager
    def temporarily_drop(self, point_name, quantity):
        """Context manager to temporarily remove an observable and auto-restore."""
        backup = copy.deepcopy(self.data)
        try:
            dropped = self.drop_observable(point_name, quantity)
            yield dropped
        finally:
            self.data = backup

    def _collect_observables(self, point, model, params, rd_value):
        """Prepare observation/theory vectors and covariance for a BAO point."""
        obs = []
        theory = []
        variances = []

        z = point['z']

        DM_obs = point.get('DM_over_rd')
        err_DM = point.get('err_DM')
        if DM_obs is not None and err_DM is not None:
            if err_DM <= 0:
                raise ValueError(f"Non-positive DM error for BAO entry '{point.get('name', 'BAO_point')}'")
            DM_theory = self.compute_DM(z, model, params) / rd_value
            if not np.isfinite(DM_theory):
                return None, None, None
            obs.append(DM_obs)
            theory.append(DM_theory)
            variances.append(err_DM ** 2)

        DH_obs = point.get('DH_over_rd')
        err_DH = point.get('err_DH')
        if DH_obs is not None and err_DH is not None:
            if err_DH <= 0:
                raise ValueError(f"Non-positive DH error for BAO entry '{point.get('name', 'BAO_point')}'")
            DH_theory = self.compute_DH_over_rd(z, model, params, rd_value)
            if not np.isfinite(DH_theory):
                return None, None, None
            obs.append(DH_obs)
            theory.append(DH_theory)
            variances.append(err_DH ** 2)

        if not obs:
            return np.array([], dtype=float), np.array([], dtype=float), None

        obs_vec = np.array(obs, dtype=float)
        theory_vec = np.array(theory, dtype=float)

        # Attempt to load a covariance matrix if supplied.
        cov_matrix = point.get('cov_matrix')
        if cov_matrix is None and point.get('cov') is not None:
            cov_matrix = point['cov']

        if cov_matrix is not None:
            cov = np.array(cov_matrix, dtype=float)
            if cov.shape != (len(obs), len(obs)):
                raise ValueError(
                    f"Invalid covariance shape {cov.shape} for BAO entry '{point.get('name', 'BAO_point')}'"
                )
            if np.any(np.diag(cov) <= 0):
                raise ValueError(
                    f"Covariance has non-positive variance terms for '{point.get('name', 'BAO_point')}'"
                )
            self._covariance_used_last = True
            return obs_vec, theory_vec, cov

        return obs_vec, theory_vec, np.diag(variances)

    def chi2(self, model, params, rd_value=147.0):
        """Calculate chi-squared for BAO data including covariance if available."""
        chi2_total = 0.0
        self._covariance_used_last = False

        for point in self.data:
            try:
                obs_vec, theory_vec, cov = self._collect_observables(point, model, params, rd_value)
            except ValueError:
                return np.inf

            if obs_vec is None:
                return np.inf
            if obs_vec.size == 0:
                continue

            diff = obs_vec - theory_vec

            try:
                solved = np.linalg.solve(cov, diff)
            except np.linalg.LinAlgError:
                return np.inf

            chi2_total += float(diff.T @ solved)

        return chi2_total

    def covariance_entry_count(self):
        return sum(
            1
            for point in self.data
            if point.get('cov_matrix') is not None or point.get('cov') is not None
        )

    def used_covariance_last_call(self):
        return self._covariance_used_last

    def compute_DM(self, z, model, params):
        """Compute comoving distance DM"""
        dm = comoving_distance(z, model, params)
        if not np.isfinite(dm):
            return np.inf
        return dm

    def compute_DH_over_rd(self, z, model, params, rd_value):
        """Compute the radial BAO observable DH/rd = c/[H(z) rd]."""
        Hz = model.Hz(z, params)
        if Hz <= 0 or not np.isfinite(Hz):
            return np.inf
        return C_LIGHT / (Hz * rd_value)

    def count_observables(self):
        """Count the number of independent BAO observables."""
        count = 0
        for point in self.data:
            dm_valid = point.get('DM_over_rd') is not None and point.get('err_DM') is not None
            dh_valid = point.get('DH_over_rd') is not None and point.get('err_DH') is not None
            if dm_valid:
                count += 1
            if dh_valid:
                count += 1
        return count

    def print_residual_table(self, model, params, rd_value=147.0, title=None):
        """Print a table comparing observations and theory for BAO data."""
        rows = []
        max_abs_pull = 0.0
        for point in self.data:
            z = point['z']
            row = {
                'Name': point.get('name', ''),
                'z': z
            }

            order = point.get('_observable_order', [])
            obs_vec, theory_vec, cov = self._collect_observables(point, model, params, rd_value)
            if obs_vec is None:
                continue
            residual = None if obs_vec.size == 0 else obs_vec - theory_vec
            contributions = None

            # Map residual entries back to observables for printing
            label_map = {
                'DM_over_rd': 'DM/rd',
                'DH_over_rd': 'DH/rd'
            }

            if residual is not None and residual.size:
                if cov is not None:
                    cov_to_use = np.array(cov, dtype=float)
                else:
                    diag_terms = []
                    for key in order:
                        if key == 'DM_over_rd':
                            err_sq = (point.get('err_DM') or 0.0) ** 2
                        elif key == 'DH_over_rd':
                            err_sq = (point.get('err_DH') or 0.0) ** 2
                        else:
                            err_sq = 0.0
                        diag_terms.append(err_sq)
                    cov_to_use = np.diag(diag_terms) if diag_terms else np.zeros((0, 0))

                try:
                    solved = np.linalg.solve(cov_to_use, residual)
                except np.linalg.LinAlgError:
                    solved = None

                if solved is not None:
                    contributions = residual * solved
                else:
                    contributions = np.zeros_like(residual)

            for idx, key in enumerate(order):
                label = label_map.get(key, key)
                obs_key = f'{label}_obs'
                theory_key = f'{label}_theory'
                residual_key = f'{label}_residual'
                pull_key = f'{label}_pull_corr'
                pull_naive_key = f'{label}_pull_naive'

                if key == 'DM_over_rd':
                    obs_val = point.get('DM_over_rd')
                    err_val = point.get('err_DM')
                    theory_val = self.compute_DM(z, model, params) / rd_value
                elif key == 'DH_over_rd':
                    obs_val = point.get('DH_over_rd')
                    err_val = point.get('err_DH')
                    theory_val = self.compute_DH_over_rd(z, model, params, rd_value)
                else:
                    obs_val = None
                    err_val = None
                    theory_val = None

                row[obs_key] = obs_val if obs_val is not None else np.nan
                row[theory_key] = theory_val if theory_val is not None else np.nan

                if obs_val is None or theory_val is None or not np.isfinite(theory_val):
                    row[residual_key] = np.nan
                    row[pull_key] = np.nan
                    row[pull_naive_key] = np.nan
                    continue

                res_val = obs_val - theory_val
                row[residual_key] = res_val

                naive_pull = np.nan
                if err_val is not None and err_val > 0:
                    naive_pull = res_val / err_val
                    row[pull_naive_key] = naive_pull
                else:
                    row[pull_naive_key] = np.nan

                corr_pull = None
                if residual is not None and residual.size > idx and contributions is not None:
                    contrib_val = contributions[idx]
                    if np.isfinite(contrib_val) and contrib_val >= 0:
                        corr_pull = np.sign(res_val) * np.sqrt(contrib_val)
                    elif np.isfinite(contrib_val):
                        corr_pull = np.sign(res_val) * np.sqrt(np.abs(contrib_val))

                if corr_pull is None or not np.isfinite(corr_pull):
                    corr_pull = naive_pull if np.isfinite(naive_pull) else np.nan

                row[pull_key] = corr_pull
                if np.isfinite(corr_pull):
                    max_abs_pull = max(max_abs_pull, abs(corr_pull))

            rows.append(row)

        if not rows:
            print("(No BAO observables to display)")
            return 0.0

        df = pd.DataFrame(rows)
        if title:
            print(title)
        with pd.option_context('display.float_format', '{:,.3f}'.format):
            print(df.fillna('-').to_string(index=False))
        print(f"Maximum |pull| = {max_abs_pull:.3f}")
        print()

        return max_abs_pull

class SNData:
    """Supernova distance modulus data handler."""

    def __init__(self, filename=None, marginalize_m=True):
        self.data = []
        self.z = np.array([])
        self.mu_obs = np.array([])
        self.mu_err = np.array([])
        self.cov = None
        self._cov_factor = None
        self._cov_inv = None
        self._cov_inv_ones = None
        self._alpha = None
        self.marginalize_m = marginalize_m
        self._last_best_M = None
        self.source_file = filename
        self._column_map = {'z': None, 'mu': None, 'sigma': None}
        self._cov_source = None
        self._cov_rank = 0
        if filename and os.path.exists(filename):
            self.load_from_file(filename)

    @staticmethod
    def _normalize_column_name(name):
        """Return a case-insensitive, whitespace/punctuation agnostic key."""
        if name is None:
            return None
        name = str(name).strip()
        # Remove Unicode BOM markers that sometimes appear in CSV headers.
        if name.startswith("\ufeff"):
            name = name.lstrip("\ufeff")
        # Keep alphanumeric characters and underscores; collapse everything else.
        cleaned = ''.join(ch for ch in name if ch.isalnum() or ch == '_')
        return cleaned.lower()

    def _build_column_lookup(self, df):
        lookup = {}
        for col in df.columns:
            col_str = str(col)
            variants = {
                col_str,
                col_str.strip(),
                col_str.lower(),
                col_str.strip().lower(),
                col_str.replace(' ', ''),
                col_str.replace(' ', '').lower(),
                col_str.replace(' ', '_'),
                col_str.replace(' ', '_').lower(),
                self._normalize_column_name(col_str),
            }
            for key in variants:
                if key:
                    lookup.setdefault(key, col_str)
        return lookup

    def _extract_column(self, df, candidates, lookup):
        for cand in candidates:
            keys = {
                cand,
                cand.strip() if isinstance(cand, str) else cand,
                cand.lower() if isinstance(cand, str) else cand,
            }
            norm = self._normalize_column_name(cand)
            if norm:
                keys.add(norm)
            for key in list(keys):
                if isinstance(key, str):
                    keys.add(key.replace(' ', ''))
                    keys.add(key.replace(' ', '').lower())
                    keys.add(key.replace(' ', '_'))
                    keys.add(key.replace(' ', '_').lower())
            for key in keys:
                if not isinstance(key, str):
                    continue
                actual = lookup.get(key)
                if actual and actual in df.columns:
                    series = pd.to_numeric(df[actual], errors='coerce')
                    if series.notna().any():
                        return series, actual
        return None, None

    def _infer_covariance_file(self, filename):
        base, _ = os.path.splitext(filename)
        candidates = [
            base + suffix
            for suffix in [
                '.cov',
                '_cov.txt',
                '_COV.txt',
                '_STAT+SYS.cov',
                '_STATONLY.cov'
            ]
        ]
        for cand in candidates:
            if os.path.exists(cand):
                return cand
        return None

    def _load_covariance_matrix(self, filename, n_expected):
        try:
            with open(filename, 'r') as f:
                first = f.readline().strip()
                try:
                    size = int(first)
                except ValueError:
                    size = None
                if size is not None and size != n_expected:
                    if size <= 0:
                        raise ValueError("Invalid covariance size header")
                    n_expected = size
                else:
                    # rewind if header was not an integer
                    if size is None:
                        f.seek(0)
                data = np.fromfile(f, sep=' ')
            if data.size == 0:
                raise ValueError("Empty covariance file")
            total = data.size
            matrix_size = int(round(np.sqrt(total)))
            if matrix_size * matrix_size != total:
                raise ValueError("Covariance file does not contain a square matrix")
            if size is not None and matrix_size != size:
                raise ValueError("Covariance size header does not match data length")
            cov = data.reshape((matrix_size, matrix_size))
            cov = 0.5 * (cov + cov.T)
            return cov
        except Exception as exc:
            print(f"Warning: Failed to load SN covariance from {filename}: {exc}")
            return None

    def _prepare_covariance(self):
        if self.cov is None:
            return
        try:
            self._cov_factor = cho_factor(self.cov, lower=True, check_finite=False)
            ones = np.ones(self.cov.shape[0])
            self._cov_inv_ones = cho_solve(self._cov_factor, ones)
            self._alpha = float(ones @ self._cov_inv_ones)
        except Exception:
            self._cov_factor = None
            try:
                self._cov_inv = np.linalg.pinv(self.cov)
                ones = np.ones(self.cov.shape[0])
                self._cov_inv_ones = self._cov_inv @ ones
                self._alpha = float(ones @ self._cov_inv_ones)
            except Exception as exc:
                print(f"Warning: SN covariance inversion failed: {exc}")
                self._cov_inv = None
                self._cov_inv_ones = None
                self._alpha = None

    def load_from_file(self, filename):
        """Load SN data from a text (or Pantheon+) file."""
        self.source_file = filename
        parse_attempts = [
            {"comment": '#', "sep": None, "engine": 'python'},
            {"comment": None, "sep": None, "engine": 'python'},
            {"comment": None, "sep": ',', "engine": 'python'},
        ]

        df = None
        z_series = mu_series = err_series = None
        z_col = mu_col = err_col = None
        parse_errors = []

        for options in parse_attempts:
            try:
                read_kwargs = dict(options)
                df_candidate = pd.read_csv(
                    filename,
                    skipinitialspace=True,
                    **read_kwargs,
                )
            except Exception as exc:
                parse_errors.append(f"{options}: {exc}")
                continue

            # Ensure all columns are treated as strings when matching names and
            # strip potential BOM markers that sometimes sneak into headers.
            df_candidate.columns = [str(col).lstrip("\ufeff") for col in df_candidate.columns]

            for col in df_candidate.columns:
                if is_categorical_dtype(df_candidate[col]):
                    df_candidate[col] = df_candidate[col].astype(str)

            column_lookup = self._build_column_lookup(df_candidate)

            z_series_candidate, z_col_candidate = self._extract_column(
                df_candidate,
                ['z', 'zHD', 'zcmb', 'zCMB', 'z_hel', 'zHEL'],
                column_lookup,
            )
            mu_series_candidate, mu_col_candidate = self._extract_column(
                df_candidate,
                ['mu', 'mu_obs', 'muSN', 'MU_SH0ES', 'm_b_corr', 'mb'],
                column_lookup,
            )
            err_series_candidate, err_col_candidate = self._extract_column(
                df_candidate,
                [
                    'sigma_mu', 'sigma', 'mu_err', 'dmu', 'mu_error',
                    'MU_SH0ES_ERR_DIAG', 'm_b_corr_err_DIAG', 'sigma_mu_tot', 'muerr'
                ],
                column_lookup,
            )

            if z_series_candidate is not None and mu_series_candidate is not None:
                df = df_candidate
                z_series, z_col = z_series_candidate, z_col_candidate
                mu_series, mu_col = mu_series_candidate, mu_col_candidate
                err_series, err_col = err_series_candidate, err_col_candidate
                break
            else:
                parse_errors.append(
                    f"{options}: missing required columns (available={list(df_candidate.columns)})"
                )

        if df is None or z_series is None or mu_series is None:
            details = '; '.join(parse_errors) if parse_errors else 'unknown format'
            raise RuntimeError(
                "SN file does not contain recognizable redshift/mu columns"
                f" (attempts: {details})"
            )

        self._column_map = {
            'z': z_col,
            'mu': mu_col,
            'sigma': err_col
        }

        mask = np.isfinite(z_series) & np.isfinite(mu_series)
        if err_series is not None:
            mask &= np.isfinite(err_series)

        z = z_series.to_numpy(dtype=float)[mask]
        mu = mu_series.to_numpy(dtype=float)[mask]
        if err_series is not None:
            mu_err = err_series.to_numpy(dtype=float)[mask]
        else:
            mu_err = np.full_like(z, np.nan)

        if z.size == 0:
            raise RuntimeError("No valid SN entries found after cleaning")

        self.z = z
        self.mu_obs = mu
        self.mu_err = mu_err
        self.data = [
            {
                'z': float(zz),
                'mu': float(mm),
                'sigma_mu': float(se) if np.isfinite(se) else None
            }
            for zz, mm, se in zip(z, mu, mu_err)
        ]

        cov_file = self._infer_covariance_file(filename)
        cov_matrix = None
        if cov_file:
            cov_matrix = self._load_covariance_matrix(cov_file, len(self.z))

        if cov_matrix is None:
            if np.all(np.isfinite(self.mu_err)):
                cov_matrix = np.diag(self.mu_err ** 2)
                self._cov_source = 'diagonal_from_sigma'
            else:
                raise RuntimeError("SN uncertainties missing and covariance file not found")
        else:
            self._cov_source = cov_file

        self.cov = np.array(cov_matrix, dtype=float)
        if self.cov.shape[0] != len(self.z):
            raise RuntimeError("SN covariance dimension does not match number of supernovae")
        self._cov_rank = int(np.linalg.matrix_rank(self.cov))
        self._prepare_covariance()

    def chi2(self, model, params):
        """Calculate chi-squared for SN data with optional M marginalization."""
        if self.z.size == 0:
            return 0.0

        mu_theory = []
        for z in self.z:
            dl = luminosity_distance(z, model, params)
            if not np.isfinite(dl) or dl <= 0:
                return np.inf
            mu_theory.append(5.0 * np.log10(dl) + 25.0)
        mu_theory = np.asarray(mu_theory)

        diff = self.mu_obs - mu_theory

        if self._cov_factor is not None:
            cov_inv_diff = cho_solve(self._cov_factor, diff)
        elif self._cov_inv is not None:
            cov_inv_diff = self._cov_inv @ diff
        else:
            raise RuntimeError("SN covariance not prepared")

        chi2_val = float(diff @ cov_inv_diff)

        if self.marginalize_m and self._cov_inv_ones is not None and self._alpha and self._alpha > 0:
            beta = float(diff @ self._cov_inv_ones)
            chi2_val -= beta ** 2 / self._alpha
            self._last_best_M = beta / self._alpha
        else:
            self._last_best_M = 0.0

        return chi2_val

    def count_points(self):
        return int(self.z.size)

    def covariance_rank(self):
        return self._cov_rank

    def column_mapping(self):
        return dict(self._column_map)

    def covariance_source(self):
        return self._cov_source

    def summary(self):
        return {
            'file': self.source_file,
            'count': self.count_points(),
            'cov_rank': self.covariance_rank(),
            'cov_source': self.covariance_source(),
            'columns': self.column_mapping(),
        }

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

# ==================== 최적화 함수 ====================

def total_chi2(params, datasets, model, rd_mode='fixed'):
    """Calculate total chi-squared for all datasets"""
    chi2_total = 0.0

    def _as_positive_chi2(value):
        if value is None:
            return 0.0
        try:
            value = float(value)
        except (TypeError, ValueError):
            return np.inf
        if not np.isfinite(value):
            return value
        if value < 0:
            return -2.0 * value
        return value

    # Check parameter bounds
    for i, (val, (low, high)) in enumerate(zip(params, model.bounds)):
        if val < low or val > high:
            return 1e10

    # BAO contribution
    if 'bao' in datasets and datasets['bao'] is not None:
        rd_value = 147.0
        if rd_mode == 'fit' and getattr(model, 'param_names', None):
            if model.param_names[-1] == 'rd':
                rd_value = params[-1]

        chi2_bao = datasets['bao'].chi2(model, params, rd_value)
        chi2_bao = _as_positive_chi2(chi2_bao)
        chi2_total += chi2_bao

    # SN contribution
    if 'sn' in datasets and datasets['sn'] is not None:
        chi2_sn = datasets['sn'].chi2(model, params)
        chi2_sn = _as_positive_chi2(chi2_sn)
        chi2_total += chi2_sn

    # CMB contribution
    if 'cmb' in datasets and datasets['cmb'] is not None:
        chi2_cmb = datasets['cmb'].chi2(model, params)
        chi2_cmb = _as_positive_chi2(chi2_cmb)
        if hasattr(datasets['cmb'], 'last_chi2'):
            datasets['cmb'].last_chi2 = chi2_cmb
        chi2_total += chi2_cmb

    regularization = 0.0
    if hasattr(model, 'regularization'):
        regularization = model.regularization(params, datasets)
    chi2_total += regularization

    return chi2_total

def _finite_float_or_none(value):
    """Return a finite float from *value* or ``None`` when not finite."""
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(value):
        return None

    return value


def _build_step_sizes(theta, bounds):
    """Return parameter-wise step sizes adapted to ``theta`` and ``bounds``."""
    theta = np.asarray(theta, dtype=float)
    n_params = theta.size
    steps = np.maximum(np.abs(theta) * 1e-4, 1e-6)

    if bounds is None:
        return steps

    adjusted_steps = steps.copy()
    for i, (low, high) in enumerate(bounds):
        lower_gap = np.inf if low is None else max(theta[i] - low, 0.0)
        upper_gap = np.inf if high is None else max(high - theta[i], 0.0)

        max_allowed = min(lower_gap, upper_gap)
        if not np.isfinite(max_allowed):
            max_allowed = max(lower_gap, upper_gap)

        if max_allowed <= 0.0:
            raise CovarianceComputationError(
                f"parameter '{i}' is pinned to its boundary, cannot estimate curvature"
            )

        adjusted_steps[i] = min(adjusted_steps[i], max_allowed * 0.5)
        adjusted_steps[i] = max(adjusted_steps[i], 1e-8)

    return adjusted_steps


def _numerical_hessian(func, theta, steps):
    """Compute a numerical Hessian of ``func`` at ``theta`` using central differences."""
    theta = np.asarray(theta, dtype=float)
    steps = np.asarray(steps, dtype=float)
    n_params = theta.size

    cache = {}

    def evaluate(point):
        key = tuple(point.tolist())
        if key in cache:
            return cache[key]
        value = float(func(point))
        if not np.isfinite(value):
            raise CovarianceComputationError(
                "objective returned a non-finite value during Hessian evaluation"
            )
        cache[key] = value
        return value

    f0 = evaluate(theta)
    hessian = np.zeros((n_params, n_params), dtype=float)

    for i in range(n_params):
        ei = np.zeros(n_params, dtype=float)
        ei[i] = steps[i]

        f_plus = evaluate(theta + ei)
        f_minus = evaluate(theta - ei)
        hessian[i, i] = (f_plus - 2.0 * f0 + f_minus) / (steps[i] ** 2)

        for j in range(i + 1, n_params):
            ej = np.zeros(n_params, dtype=float)
            ej[j] = steps[j]

            f_pp = evaluate(theta + ei + ej)
            f_pm = evaluate(theta + ei - ej)
            f_mp = evaluate(theta - ei + ej)
            f_mm = evaluate(theta - ei - ej)

            value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * steps[i] * steps[j])
            hessian[i, j] = value
            hessian[j, i] = value

    return hessian, f0


def estimate_covariance(func, theta, param_names, bounds, dof):
    """Estimate the covariance matrix at the best-fit point ``theta``."""
    if dof <= 0:
        raise CovarianceComputationError("non-positive degrees of freedom")

    steps = _build_step_sizes(theta, bounds)
    hessian, _ = _numerical_hessian(func, theta, steps)

    cond_number = np.linalg.cond(hessian)
    if not np.isfinite(cond_number):
        cond_number = np.inf

    eye = np.eye(len(theta))
    covariance = None
    regularisation_used = None

    for lam in (0.0, 1e-8, 1e-6, 1e-4, 1e-2):
        try:
            candidate = np.linalg.inv(hessian + lam * eye)
            covariance = 2.0 * candidate
            regularisation_used = lam
            break
        except np.linalg.LinAlgError:
            continue

    if covariance is None:
        candidate = np.linalg.pinv(hessian)
        covariance = 2.0 * candidate
        regularisation_used = None

    covariance = 0.5 * (covariance + covariance.T)

    diag = np.diag(covariance)
    errors = {}
    for name, variance in zip(param_names, diag):
        if variance > 0.0:
            errors[name] = float(np.sqrt(variance))
        else:
            errors[name] = None

    if 'h' in param_names:
        sigma_h = errors.get('h')
        errors['H0'] = float(100.0 * sigma_h) if sigma_h is not None else None
    else:
        errors['H0'] = None

    std = np.sqrt(np.maximum(diag, 0.0))
    with np.errstate(invalid='ignore', divide='ignore'):
        denom = np.outer(std, std)
        correlation = np.divide(
            covariance,
            denom,
            out=np.zeros_like(covariance),
            where=denom > 0.0
        )

    result = CovarianceResult(
        matrix=covariance,
        correlation=correlation,
        errors=errors,
        condition_number=cond_number,
    )

    if regularisation_used not in (None, 0.0):
        print(f"[COV] Applied ridge regularisation λ={regularisation_used:.1e} to stabilise Hessian")

    print(f"[COV] Hessian built at MLE, cond={cond_number:.3e}")
    if errors.get('h') is not None:
        print(f"[COV] sigma(h)={errors['h']:.4e}, sigma(H0)={errors['H0']:.4e}")

    return result


def calculate_statistics(chi2, n_data, n_params):
    """Calculate AIC, BIC, and other statistics using JSON-safe numbers."""
    if n_data <= 0:
        raise ValueError("Number of data points must be positive for statistics calculation")

    chi2_value = _finite_float_or_none(chi2)
    dof = int(n_data - n_params)

    chi2_red = None
    if chi2_value is not None and dof > 0:
        chi2_red = chi2_value / dof

    aic = None
    bic = None
    if chi2_value is not None:
        aic = chi2_value + 2 * int(n_params)
        bic = chi2_value + int(n_params) * np.log(n_data)

    return {
        'chi2': chi2_value,
        'dof': dof,
        'chi2_red': chi2_red,
        'aic': aic,
        'bic': bic,
        'n_data': int(n_data),
        'n_params': int(n_params)
    }

# ==================== Main 함수 ====================

def main():
    parser = argparse.ArgumentParser(description='7D Cosmology Joint Fitting')
    parser.add_argument('--model', choices=['LCDM', '7D', 'both'], default='both',
                        help='Model to fit')
    parser.add_argument('--bao-file', type=str, default=None,
                        help='BAO data file (JSON)')
    parser.add_argument('--sn-file', type=str, default=None,
                        help='Supernova data file')
    parser.add_argument('--use-cmb', action='store_true',
                        help='Include CMB constraints')
    parser.add_argument('--use-default-bao', action='store_true',
                        help='Load the packaged DESI BAO catalogue when no file is supplied')
    parser.add_argument('--rd-mode', choices=['fixed', 'fit'], default='fixed',
                        help='Sound horizon mode')
    parser.add_argument('--disable-bao-cov', action='store_true',
                        help='Ignore supplied BAO covariance matrices')
    parser.add_argument('--no-proxy', action='store_true',
                        help='Omit the legacy QSO proxy point from the default BAO catalogue')
    parser.add_argument('--drop-lya-dh', action='store_true',
                        help='Remove the Lyα DH/rd point from the final fit')
    parser.add_argument('--diagnose-lya-dh', action='store_true',
                        help='Temporarily inspect residuals without the Lyα DH/rd point before fitting')
    parser.add_argument('--workers', type=int, default=-1,
                        help='Number of workers for parallelization (-1 for all CPUs)')
    parser.add_argument('--maxiter', type=int, default=1000,
                        help='Maximum iterations for optimization')
    parser.add_argument('--output', type=str, default='fit_results.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    # Set number of workers
    if args.workers == -1:
        args.workers = mp.cpu_count()
    
    print(f"=== 7D Cosmology Joint Fitting ===")
    print(f"Using {args.workers} workers for parallel computation")
    
    # Load datasets
    print("\nLoading datasets...")
    datasets = {'bao': None, 'sn': None, 'cmb': None}

    # BAO data
    use_covariance = not args.disable_bao_cov
    bao_requested = bool(args.bao_file) or args.use_default_bao
    if bao_requested:
        bao_source = args.bao_file if args.bao_file else 'default DESI catalogue'
        bao_data = BAOData(
            args.bao_file if args.bao_file else None,
            use_official_covariance=use_covariance,
            include_proxy=not args.no_proxy
        )
        if args.disable_bao_cov:
            bao_data.remove_all_covariances()
        datasets['bao'] = bao_data
        print(f"  BAO: {bao_data.count_observables()} observables from {len(bao_data.data)} entries loaded")
        print(f"      source: {bao_source}")
        cov_entries = bao_data.covariance_entry_count()
        if cov_entries:
            print(f"      covariance provided for {cov_entries} BAO entries")
    else:
        print("  BAO: skipped (no file provided)")

    # SN data
    if args.sn_file:
        try:
            sn_data = SNData(args.sn_file)
            datasets['sn'] = sn_data
            sn_summary = sn_data.summary()
            print(f"  SN: {sn_summary['count']} points loaded (cov rank = {sn_summary['cov_rank']})")
            print(f"      file: {sn_summary['file']}")
            columns = sn_summary['columns']
            print(f"      columns: z='{columns.get('z')}', mu='{columns.get('mu')}', sigma='{columns.get('sigma')}'")
            cov_source = sn_summary['cov_source']
            if cov_source:
                print(f"      covariance: {cov_source}")
            else:
                print("      covariance: none")
        except Exception as exc:
            print(f"  Warning: failed to load SN data ({exc})")
            datasets['sn'] = None
    else:
        datasets['sn'] = None
    
    # CMB data
    if args.use_cmb:
        cmb_data = CMBData()
        datasets['cmb'] = cmb_data
        print("  CMB: constraints loaded")
    else:
        datasets['cmb'] = None

    # Fiducial sanity check for BAO definitions
    if datasets['bao'] is not None and datasets['bao'].count_observables() > 0:
        fid_lcdm = LCDMModel()
        fid_params = [0.67, 0.31]
        print("\nFiducial BAO sanity check (ΛCDM: h=0.67, Ωm=0.31):")
        fid_max_pull = datasets['bao'].print_residual_table(fid_lcdm, fid_params, rd_value=147.0)
        print(f"  => Fiducial maximum |pull| = {fid_max_pull:.3f}")

        # Evaluate fractional deviations to guard against definition mismatches.
        print("Fiducial fractional differences (|Δ|/obs):")
        max_frac = 0.0
        for point in datasets['bao'].data:
            obs_vec, theory_vec, _ = datasets['bao']._collect_observables(point, fid_lcdm, fid_params, 147.0)
            if obs_vec is None or obs_vec.size == 0:
                continue
            frac_diff = np.abs(obs_vec - theory_vec) / obs_vec
            if frac_diff.size:
                max_frac = max(max_frac, float(np.max(frac_diff)))
            if np.any(frac_diff > 0.01):
                print(f"  Warning: {point.get('name', 'BAO_point')} exceeds 1%: {frac_diff}")
        if max_frac <= 0.01:
            print("  All fiducial predictions agree within 1% of observations")

        if args.diagnose_lya_dh:
            diagnostic_flag = False
            with datasets['bao'].temporarily_drop('DESI Lyα GCcomb', 'DH_over_rd') as dropped:
                diagnostic_flag = dropped
                if dropped:
                    print("\nDiagnostic BAO residuals without Lyα DH/rd:")
                    diag_pull = datasets['bao'].print_residual_table(
                        fid_lcdm, fid_params, rd_value=147.0,
                        title='(Lyα DH/rd temporarily removed)'
                    )
                    print(f"  => Diagnostic maximum |pull| (Lyα DH removed) = {diag_pull:.3f}")
                else:
                    print("\nDiagnostic request: Lyα DH/rd observable unavailable for removal.")
            if diagnostic_flag:
                print("  Lyα DH/rd observable restored for covariance-weighted fitting.")

    if args.drop_lya_dh:
        if datasets['bao'] is not None:
            dropped_final = datasets['bao'].drop_observable('DESI Lyα GCcomb', 'DH_over_rd')
            if dropped_final:
                print("\n  BAO: Lyα DH/rd observable removed from the final fit")
            else:
                print("\n  Warning: Lyα DH/rd observable not found or already removed")
        else:
            print("\n  Warning: cannot drop Lyα DH/rd because BAO data are not loaded")

    final_bao_count = datasets['bao'].count_observables() if datasets['bao'] is not None else 0
    print(f"  BAO (final): {final_bao_count} observables from {len(datasets['bao'].data) if datasets['bao'] else 0} entries in use")
    final_cov_entries = datasets['bao'].covariance_entry_count() if datasets['bao'] is not None else 0
    if final_cov_entries:
        print(f"    Covariance applied to {final_cov_entries} BAO entries")

    # Count total data points with the finalized configuration
    n_data = 0
    if datasets['bao'] is not None:
        n_data += datasets['bao'].count_observables()
    if datasets['sn']:
        n_data += datasets['sn'].count_points()
    if datasets['cmb']:
        n_data += datasets['cmb'].count_observables()

    # Models to fit
    models_to_fit = []
    if args.model in ['LCDM', 'both']:
        models_to_fit.append(LCDMModel())
    if args.model in ['7D', 'both']:
        models_to_fit.append(SevenDModel())

    if args.rd_mode == 'fit':
        models_to_fit = [RDFitWrapper(model) for model in models_to_fit]
    
    results = {}
    
    # Fit each model
    for model in models_to_fit:
        print(f"\n{'='*50}")
        print(f"Fitting {model.name} model...")
        print(f"Parameters: {model.param_names}")
        print(f"Bounds: {model.bounds}")
        
        # Create objective function
        obj = partial(total_chi2, 
                     datasets=datasets, 
                     model=model, 
                     rd_mode=args.rd_mode)
        
        # Run differential evolution
        try:
            de_res = differential_evolution(
                obj,
                bounds=model.bounds,
                workers=args.workers,
                updating='deferred',
                polish=True,
                disp=True,
                maxiter=args.maxiter,
                seed=42,  # For reproducibility
                tol=1e-6,
                atol=1e-8
            )
            
            theta_hat = np.asarray(de_res.x, dtype=float)
            param_names = list(model.param_names)
            dof_candidate = int(n_data - len(param_names))

            covariance_payload = None
            correlation_payload = None
            errors_payload = None

            if not de_res.success:
                print("[WARN] Skipping covariance estimation because the optimizer did not converge.")
            elif dof_candidate <= 0:
                print("[WARN] Skipping covariance estimation because degrees of freedom are non-positive.")
            else:
                try:
                    cov_result = estimate_covariance(
                        func=obj,
                        theta=theta_hat,
                        param_names=param_names,
                        bounds=getattr(model, 'bounds', None),
                        dof=dof_candidate,
                    )
                    covariance_payload = {
                        'param_names': param_names,
                        'matrix': cov_result.matrix.tolist(),
                        'condition_number': float(cov_result.condition_number),
                    }
                    correlation_payload = {
                        'param_names': param_names,
                        'matrix': cov_result.correlation.tolist(),
                    }
                    errors_payload = {
                        key: _finite_float_or_none(value)
                        for key, value in cov_result.errors.items()
                    }
                except CovarianceComputationError as exc:
                    print(f"[WARN] covariance estimation failed: {exc}")
                except Exception as exc:  # pylint: disable=broad-except
                    print(f"[WARN] covariance estimation raised an unexpected error: {exc}")

            # Calculate statistics
            stats = calculate_statistics(de_res.fun, n_data, len(model.param_names))

            # Prepare best-fit parameters with native Python numbers for JSON serialization
            best_fit_parameters = {
                name: float(value)
                for name, value in zip(model.param_names, de_res.x)
            }

            # Promote commonly used statistics to the top-level result while keeping a
            # dedicated statistics dictionary. This guarantees that χ², dof, AIC, BIC
            # and related values are always present in the saved JSON output.  Every
            # value is converted through ``_finite_float_or_none`` so that the JSON
            # file never contains NaN/Inf tokens that could break strict parsers.
            stats_native = {
                'chi2': _finite_float_or_none(stats.get('chi2')),
                'dof': int(stats.get('dof', 0)),
                'chi2_red': _finite_float_or_none(stats.get('chi2_red')),
                'aic': _finite_float_or_none(stats.get('aic')),
                'bic': _finite_float_or_none(stats.get('bic')),
                'n_data': int(stats.get('n_data', n_data)),
                'n_params': int(stats.get('n_params', len(model.param_names)))
            }

            # Store results
            results[model.name] = {
                'success': bool(de_res.success),
                'bestfit': best_fit_parameters,
                'parameters': best_fit_parameters.copy(),
                'chi2': stats_native['chi2'],
                'dof': stats_native['dof'],
                'chi2_red': stats_native['chi2_red'],
                'aic': stats_native['aic'],
                'bic': stats_native['bic'],
                'n_data': stats_native['n_data'],
                'n_params': stats_native['n_params'],
                'statistics': stats_native,
                'message': str(de_res.message),
                'nfev': int(de_res.nfev),
                'errors': errors_payload,
                'covariance': covariance_payload,
                'correlation': correlation_payload
            }

            # Print results
            print(f"\nOptimization {'succeeded' if de_res.success else 'failed'}")
            print(f"Message: {de_res.message}")
            print(f"Function evaluations: {de_res.nfev}")
            print(f"\nBest-fit parameters:")
            for name, value in zip(model.param_names, de_res.x):
                print(f"  {name:8s} = {value:.6f}")

            # Warn if parameters hit boundaries
            for (name, value), (low, high) in zip(zip(model.param_names, de_res.x), model.bounds):
                span = high - low
                if span <= 0:
                    continue
                lower_frac = (value - low) / span
                upper_frac = (high - value) / span
                if lower_frac < 0.05 or upper_frac < 0.05:
                    print(f"  Warning: parameter '{name}' is within 5% of its bounds ({low}, {high})")

            print(f"\nStatistics:")
            chi2_disp = stats.get('chi2')
            chi2_red_disp = stats.get('chi2_red')
            aic_disp = stats.get('aic')
            bic_disp = stats.get('bic')

            chi2_text = f"{chi2_disp:.3f}" if chi2_disp is not None else "undefined"
            chi2_red_text = f"{chi2_red_disp:.3f}" if chi2_red_disp is not None else "undefined"
            aic_text = f"{aic_disp:.3f}" if aic_disp is not None else "undefined"
            bic_text = f"{bic_disp:.3f}" if bic_disp is not None else "undefined"

            print(f"  χ² = {chi2_text}")
            print(f"  dof = {stats['dof']}")
            print(f"  χ²/dof = {chi2_red_text}")
            print(f"  AIC = {aic_text}")
            print(f"  BIC = {bic_text}")

            # BAO residuals at best fit
            if datasets['bao'] is not None and datasets['bao'].count_observables() > 0:
                print(f"BAO residuals at best-fit {model.name}:")
                rd_for_residuals = 147.0
                if args.rd_mode == 'fit' and 'rd' in model.param_names:
                    rd_index = model.param_names.index('rd')
                    rd_candidate = de_res.x[rd_index]
                    if np.isfinite(rd_candidate) and rd_candidate > 0:
                        rd_for_residuals = rd_candidate
                best_pull = datasets['bao'].print_residual_table(model, de_res.x, rd_value=rd_for_residuals)
                print(f"  => Maximum |pull| at best-fit {model.name}: {best_pull:.3f}")

        except Exception as e:
            print(f"ERROR during optimization: {e}")
            results[model.name] = {
                'success': False,
                'error': str(e)
            }
    
    # Compare models if both were fitted
    if 'LCDM' in results and '7D' in results:
        if results['LCDM']['success'] and results['7D']['success']:
            print(f"\n{'='*50}")
            print("Model Comparison:")
            print(f"{'Model':<10} {'χ²':<10} {'AIC':<10} {'BIC':<10}")
            print("-" * 40)
            for model_name in ['LCDM', '7D']:
                stats = results[model_name]['statistics']
                print(f"{model_name:<10} {stats['chi2']:<10.3f} "
                      f"{stats['aic']:<10.3f} {stats['bic']:<10.3f}")
            
            # Determine preferred model
            delta_aic = results['7D']['statistics']['aic'] - results['LCDM']['statistics']['aic']
            delta_bic = results['7D']['statistics']['bic'] - results['LCDM']['statistics']['bic']
            
            print(f"\nΔAIC (7D - ΛCDM) = {delta_aic:.3f}")
            print(f"ΔBIC (7D - ΛCDM) = {delta_bic:.3f}")
            
            if delta_aic < -2:
                print("AIC strongly prefers 7D model")
            elif delta_aic < 0:
                print("AIC slightly prefers 7D model")
            elif delta_aic < 2:
                print("AIC slightly prefers ΛCDM model")
            else:
                print("AIC strongly prefers ΛCDM model")
    
    # Save results
    try:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    except Exception as e:
        print(f"Warning: Could not save results: {e}")
    
    return results

if __name__ == "__main__":
    # Handle multiprocessing on Windows
    if sys.platform.startswith('win'):
        mp.freeze_support()
    
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

