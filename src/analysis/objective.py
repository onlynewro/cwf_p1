"""Objective functions for cosmological parameter estimation."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def total_chi2(params: Sequence[float], datasets: Dict[str, object], model, rd_mode: str = "fixed") -> float:
    """Calculate total chi-squared for all datasets."""
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

    for val, (low, high) in zip(params, model.bounds):
        if val < low or val > high:
            return 1e10

    if "bao" in datasets and datasets["bao"] is not None:
        rd_value = 147.0
        if rd_mode == "fit" and getattr(model, "param_names", None):
            if model.param_names[-1] == "rd":
                rd_value = params[-1]

        chi2_bao = datasets["bao"].chi2(model, params, rd_value)
        chi2_bao = _as_positive_chi2(chi2_bao)
        chi2_total += chi2_bao

    if "sn" in datasets and datasets["sn"] is not None:
        chi2_sn = datasets["sn"].chi2(model, params)
        chi2_sn = _as_positive_chi2(chi2_sn)
        chi2_total += chi2_sn

    if "cmb" in datasets and datasets["cmb"] is not None:
        chi2_cmb = datasets["cmb"].chi2(model, params)
        chi2_cmb = _as_positive_chi2(chi2_cmb)
        if hasattr(datasets["cmb"], "last_chi2"):
            datasets["cmb"].last_chi2 = chi2_cmb
        chi2_total += chi2_cmb

    regularization = 0.0
    if hasattr(model, "regularization"):
        regularization = model.regularization(params, datasets)
    chi2_total += regularization

    return chi2_total


__all__ = ["total_chi2"]
