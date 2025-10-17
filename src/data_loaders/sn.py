"""Supernova distance modulus data loader."""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype
from scipy.linalg import cho_factor, cho_solve

from src.models import luminosity_distance


class SNData:
    """Supernova distance modulus data handler."""

    def __init__(self, filename: Optional[str] = None, marginalize_m: bool = True):
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
        self._column_map = {"z": None, "mu": None, "sigma": None}
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
        if name.startswith("\ufeff"):
            name = name.lstrip("\ufeff")
        cleaned = "".join(ch for ch in name if ch.isalnum() or ch == "_")
        return cleaned.lower()

    def _build_column_lookup(self, df: pd.DataFrame):
        lookup: Dict[str, str] = {}
        for col in df.columns:
            col_str = str(col)
            variants = {
                col_str,
                col_str.strip(),
                col_str.lower(),
                col_str.strip().lower(),
                col_str.replace(" ", ""),
                col_str.replace(" ", "").lower(),
                col_str.replace(" ", "_"),
                col_str.replace(" ", "_").lower(),
                self._normalize_column_name(col_str),
            }
            for key in variants:
                if key:
                    lookup.setdefault(key, col_str)
        return lookup

    def _extract_column(self, df: pd.DataFrame, candidates: Iterable[str], lookup: Dict[str, str]):
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
                    keys.add(key.replace(" ", ""))
                    keys.add(key.replace(" ", "").lower())
                    keys.add(key.replace(" ", "_"))
                    keys.add(key.replace(" ", "_").lower())
            for key in keys:
                if not isinstance(key, str):
                    continue
                actual = lookup.get(key)
                if actual and actual in df.columns:
                    series = pd.to_numeric(df[actual], errors="coerce")
                    if series.notna().any():
                        return series, actual
        return None, None

    def _infer_covariance_file(self, filename: str):
        base, _ = os.path.splitext(filename)
        candidates = [
            base + suffix
            for suffix in [
                ".cov",
                "_cov.txt",
                "_COV.txt",
                "_STAT+SYS.cov",
                "_STATONLY.cov",
            ]
        ]
        for cand in candidates:
            if os.path.exists(cand):
                return cand
        return None

    def _load_covariance_matrix(self, filename: str, n_expected: int):
        try:
            with open(filename, "r", encoding="utf8") as handle:
                first = handle.readline().strip()
                try:
                    size = int(first)
                except ValueError:
                    size = None
                if size is not None and size != n_expected:
                    if size <= 0:
                        raise ValueError("Invalid covariance size header")
                    n_expected = size
                else:
                    if size is None:
                        handle.seek(0)
                data = np.fromfile(handle, sep=" ")
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
        except Exception as exc:  # pragma: no cover - user feedback
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

    def load_from_file(self, filename: str):
        """Load SN data from a text (or Pantheon+) file."""
        self.source_file = filename
        parse_attempts = [
            {"comment": "#", "sep": None, "engine": "python"},
            {"comment": None, "sep": None, "engine": "python"},
            {"comment": None, "sep": ",", "engine": "python"},
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

            df_candidate.columns = [str(col).lstrip("\ufeff") for col in df_candidate.columns]

            for col in df_candidate.columns:
                if is_categorical_dtype(df_candidate[col]):
                    df_candidate[col] = df_candidate[col].astype(str)

            column_lookup = self._build_column_lookup(df_candidate)

            z_series_candidate, z_col_candidate = self._extract_column(
                df_candidate,
                ["z", "zHD", "zcmb", "zCMB", "z_hel", "zHEL"],
                column_lookup,
            )
            mu_series_candidate, mu_col_candidate = self._extract_column(
                df_candidate,
                ["mu", "mu_obs", "muSN", "MU_SH0ES", "m_b_corr", "mb"],
                column_lookup,
            )
            err_series_candidate, err_col_candidate = self._extract_column(
                df_candidate,
                [
                    "sigma_mu",
                    "sigma",
                    "mu_err",
                    "dmu",
                    "mu_error",
                    "MU_SH0ES_ERR_DIAG",
                    "m_b_corr_err_DIAG",
                    "sigma_mu_tot",
                    "muerr",
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
            details = "; ".join(parse_errors) if parse_errors else "unknown format"
            raise RuntimeError(
                "SN file does not contain recognizable redshift/mu columns"
                f" (attempts: {details})"
            )

        self._column_map = {
            "z": z_col,
            "mu": mu_col,
            "sigma": err_col,
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
                "z": float(zz),
                "mu": float(mm),
                "sigma_mu": float(se) if np.isfinite(se) else None,
            }
            for zz, mm, se in zip(z, mu, mu_err)
        ]

        cov_file = self._infer_covariance_file(filename)
        cov_matrix = None
        if cov_file:
            cov_matrix = self._load_covariance_matrix(cov_file, len(self.data))
            if cov_matrix is not None:
                self._cov_source = cov_file

        if cov_matrix is not None:
            self.cov = np.asarray(cov_matrix, dtype=float)
            self._cov_rank = int(np.linalg.matrix_rank(self.cov))
        else:
            diag = np.where(np.isfinite(mu_err) & (mu_err > 0), mu_err ** 2, np.nan)
            if np.all(np.isnan(diag)):
                self.cov = None
                self._cov_rank = 0
            else:
                self.cov = np.diag(np.nan_to_num(diag, nan=0.0))
                self._cov_rank = int(np.linalg.matrix_rank(self.cov))

        self._prepare_covariance()

    def summary(self) -> Dict[str, object]:
        return {
            "count": int(len(self.data)),
            "file": self.source_file,
            "columns": self._column_map,
            "cov_source": self._cov_source,
            "cov_rank": int(self._cov_rank),
        }

    def count_points(self) -> int:
        return int(len(self.data))

    def chi2(self, model, params):
        if self.cov is None or self.cov.size == 0:
            residuals = self._distance_modulus_residuals(model, params)
            if residuals is None:
                return float("inf")
            if np.any(self.mu_err <= 0):
                return float("inf")
            return float(np.sum((residuals / self.mu_err) ** 2))

        residuals = self._distance_modulus_residuals(model, params)
        if residuals is None:
            return float("inf")

        if self._cov_factor is not None:
            y = residuals.reshape(-1, 1)
            solved = cho_solve(self._cov_factor, y, check_finite=False)
            return float(residuals @ solved.ravel())

        if self._cov_inv is not None:
            return float(residuals @ (self._cov_inv @ residuals))

        return float("inf")

    def _distance_modulus_residuals(self, model, params):
        if self.z.size == 0:
            return None

        mu_theory = []
        for z in self.z:
            dl = luminosity_distance(z, model, params)
            if not np.isfinite(dl) or dl <= 0:
                return None
            mu_theory.append(5 * np.log10(dl) + 25)

        mu_theory = np.array(mu_theory, dtype=float)
        residuals = self.mu_obs - mu_theory

        if not self.marginalize_m or self.cov is None:
            return residuals

        if self._alpha is None or self._cov_inv_ones is None:
            return residuals

        beta = float(residuals @ self._cov_inv_ones)
        m_best = beta / self._alpha if self._alpha else 0.0
        self._last_best_M = m_best
        return residuals - m_best


__all__ = ["SNData"]
