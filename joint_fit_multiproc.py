#!/usr/bin/env python3
"""
joint_fit_patched_multiproc5.py
7차원 우주론 모델과 ΛCDM 모델의 BAO, SNe, CMB 데이터 합동 피팅
멀티프로세싱 지원 및 오류 처리 강화 버전
"""

import multiprocessing as mp
import sys
import warnings
from dataclasses import dataclass

import numpy as np

from src.utils.cosmology import (
    C_LIGHT,
    DEFAULT_RD_MPC,
    angular_diameter_distance,
    comoving_distance,
    hu_sugiyama_z_star,
    luminosity_distance,
    sound_horizon_at_z,
)

warnings.filterwarnings('ignore')


class CovarianceComputationError(RuntimeError):
    """Raised when the numerical covariance estimation fails."""


@dataclass
class CovarianceResult:
    matrix: np.ndarray
    correlation: np.ndarray
    errors: dict
    condition_number: float

# 멀티프로세싱 관련 설정
if sys.platform == "darwin":  # macOS
    mp.set_start_method('fork', force=True)

# ==================== 우주론 모델 정의 ====================

# ==================== 데이터 로더 ====================

# ==================== 최적화 함수 ====================

def total_chi2(params, datasets, model, rd_mode='fixed', return_components=False):
    """Calculate total chi-squared for all datasets.

    When ``return_components`` is ``True`` a tuple ``(total, components)`` is
    returned where *components* maps dataset names to their individual χ²
    contributions. Missing datasets are reported with ``None``.
    """
    chi2_total = 0.0
    chi2_components = {
        'bao': None,
        'sn': None,
        'cmb': None,
        'regularization': None,
    }

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
        rd_value = DEFAULT_RD_MPC
        if rd_mode == 'fit' and getattr(model, 'param_names', None):
            if model.param_names[-1] == 'rd':
                rd_value = params[-1]

        chi2_bao = datasets['bao'].chi2(model, params, rd_value)
        chi2_bao = _as_positive_chi2(chi2_bao)
        chi2_total += chi2_bao
        chi2_components['bao'] = chi2_bao
        if hasattr(datasets['bao'], 'last_chi2'):
            datasets['bao'].last_chi2 = chi2_bao

    # SN contribution
    if 'sn' in datasets and datasets['sn'] is not None:
        chi2_sn = datasets['sn'].chi2(model, params)
        chi2_sn = _as_positive_chi2(chi2_sn)
        chi2_total += chi2_sn
        chi2_components['sn'] = chi2_sn
        if hasattr(datasets['sn'], 'last_chi2'):
            datasets['sn'].last_chi2 = chi2_sn

    # CMB contribution
    if 'cmb' in datasets and datasets['cmb'] is not None:
        chi2_cmb = datasets['cmb'].chi2(model, params)
        chi2_cmb = _as_positive_chi2(chi2_cmb)
        chi2_total += chi2_cmb
        chi2_components['cmb'] = chi2_cmb
        if hasattr(datasets['cmb'], 'last_chi2'):
            datasets['cmb'].last_chi2 = chi2_cmb

    regularization = 0.0
    if hasattr(model, 'regularization'):
        regularization = model.regularization(params, datasets)
    chi2_total += regularization
    chi2_components['regularization'] = regularization

    if return_components:
        return chi2_total, chi2_components
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

def main():
    from src.main import main as entry_main
    return entry_main()


if __name__ == "__main__":
    from src.main import cli

    cli()
