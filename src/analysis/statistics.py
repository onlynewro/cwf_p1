"""Statistical helpers for reporting fit quality."""

from __future__ import annotations

import numpy as np


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
        "chi2": chi2_value,
        "dof": dof,
        "chi2_red": chi2_red,
        "aic": aic,
        "bic": bic,
        "n_data": int(n_data),
        "n_params": int(n_params),
    }


__all__ = ["_finite_float_or_none", "calculate_statistics"]
