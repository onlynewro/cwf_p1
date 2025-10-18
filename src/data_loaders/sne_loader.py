"""Supernova data loader and utilities."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype
from scipy.linalg import cho_factor, cho_solve

from src.utils.cosmology import luminosity_distance
from src.utils.validation import ConfigValidationError, require_existing_file


class SNData:
    """Supernova distance modulus data handler."""

    def __init__(self, filename=None, marginalize_m=None, config=None):
        self.config = config or {}
        self._config_base = Path(self.config.get('base_dir', '.'))
        if marginalize_m is None:
            marginalize_m = self.config.get('marginalize_m', True)
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
        data_file = filename or self.config.get('file')
        if data_file:
            resolved = require_existing_file(
                data_file,
                base_dir=self._config_base,
                description='SN data file'
            )
            self.load_from_file(resolved)

    @staticmethod
    def _normalize_column_name(name):
        """Return a case-insensitive, whitespace/punctuation agnostic key."""
        if name is None:
            return None
        name = str(name).strip()
        if name.startswith("\ufeff"):
            name = name.lstrip("\ufeff")
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
            with open(filename, 'r', encoding='utf-8') as f:
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
        except Exception as exc:  # pylint: disable=broad-except
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
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Warning: SN covariance inversion failed: {exc}")
                self._cov_inv = None
                self._cov_inv_ones = None
                self._alpha = None

    def load_from_file(self, filename):
        """Load SN data from a text (or Pantheon+) file."""
        resolved = require_existing_file(
            filename,
            base_dir=self._config_base,
            description='SN data file'
        )
        self.source_file = resolved
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
                    resolved,
                    skipinitialspace=True,
                    **read_kwargs,
                )
            except Exception as exc:
                parse_errors.append(f"{options}: {exc}")
                continue

            if df_candidate.empty:
                parse_errors.append(f"{options}: file parsed but empty")
                continue

            lookup = self._build_column_lookup(df_candidate)

            z_series, z_col = self._extract_column(
                df_candidate,
                ['zHD', 'z', 'redshift', 'zhel', 'zcmb', 'zcmbhel'],
                lookup,
            )
            mu_series, mu_col = self._extract_column(
                df_candidate,
                [
                    'mu',
                    'm-m',
                    'distance_modulus',
                    'modulus',
                    'mu_sh0es',
                    'm_b_corr',
                    'mB',
                    'm_B',
                ],
                lookup,
            )
            err_series, err_col = self._extract_column(
                df_candidate,
                [
                    'sigma',
                    'mu_err',
                    'm_err',
                    'sig',
                    'err',
                    'mu_sigma',
                    'mu_sh0es_err',
                    'mu_sh0es_err_diag',
                    'm_b_corr_err_diag',
                    'mB_err',
                    'm_B_err',
                ],
                lookup,
            )

            if z_series is not None and mu_series is not None:
                df = df_candidate
                break

            parse_errors.append(f"{options}: required columns not found")

        if df is None:
            raise ValueError(
                "Failed to parse SN data file. Attempts:\n" + "\n".join(parse_errors)
            )

        self.data = df
        self._column_map = {'z': z_col, 'mu': mu_col, 'sigma': err_col}

        if is_categorical_dtype(df[z_col]):
            z_values = df[z_col].cat.codes.to_numpy(dtype=float)
        else:
            z_values = pd.to_numeric(df[z_col], errors='coerce').to_numpy(dtype=float)

        mu_values = pd.to_numeric(df[mu_col], errors='coerce').to_numpy(dtype=float)
        if err_series is not None:
            mu_err_values = err_series.to_numpy(dtype=float)
        else:
            mu_err_values = np.full_like(mu_values, np.nan)

        mask = np.isfinite(z_values) & np.isfinite(mu_values)
        z_values = z_values[mask]
        mu_values = mu_values[mask]
        mu_err_values = mu_err_values[mask]

        if np.any(~np.isfinite(mu_err_values)):
            raise ConfigValidationError(
                "SN data catalogue contains non-finite distance modulus uncertainties"
            )

        self.z = z_values
        self.mu_obs = mu_values
        self.mu_err = mu_err_values

        cov_file = self._infer_covariance_file(filename)
        if cov_file:
            cov = self._load_covariance_matrix(cov_file, len(self.z))
            if cov is not None:
                self.cov = cov
                self._cov_source = cov_file
                self._cov_rank = int(np.linalg.matrix_rank(cov))
        else:
            self.cov = None
            self._cov_source = None
            self._cov_rank = 0

        self._prepare_covariance()

    def _distance_modulus(self, z, model, params):
        dl = luminosity_distance(z, model, params)
        if not np.isfinite(dl):
            return np.nan
        return 5.0 * np.log10(dl) + 25.0

    def _chi2_matrix(self, residual):
        if self.cov is not None and self._cov_factor is not None:
            solved = cho_solve(self._cov_factor, residual)
            return float(residual.T @ solved)

        if self.cov is not None and self._cov_inv is not None:
            solved = self._cov_inv @ residual
            return float(residual.T @ solved)

        variance = self.mu_err ** 2
        if np.any(~np.isfinite(variance)):
            return np.inf
        if np.any(variance <= 0):
            return np.inf
        return float(np.sum(residual ** 2 / variance))

    def chi2(self, model, params):
        if self.z.size == 0:
            return 0.0

        mu_theory = np.array([
            self._distance_modulus(z, model, params)
            for z in self.z
        ], dtype=float)

        if np.any(~np.isfinite(mu_theory)):
            return np.inf

        residual = self.mu_obs - mu_theory

        if not self.marginalize_m:
            return self._chi2_matrix(residual)

        if self._cov_factor is not None:
            ones = np.ones_like(self.mu_obs)
            solved_res = cho_solve(self._cov_factor, residual)
            solved_ones = self._cov_inv_ones
        elif self._cov_inv is not None and self._cov_inv_ones is not None:
            ones = np.ones_like(self.mu_obs)
            solved_res = self._cov_inv @ residual
            solved_ones = self._cov_inv_ones
        else:
            variance = self.mu_err ** 2
            if np.any(~np.isfinite(variance)):
                return np.inf
            if np.any(variance <= 0):
                return np.inf
            weights = 1.0 / variance
            if np.any(~np.isfinite(weights)):
                return np.inf
            alpha = np.sum(weights)
            beta = np.sum(weights * residual)
            M_best = beta / alpha
            self._last_best_M = M_best
            return float(np.sum(weights * (residual - M_best) ** 2))

        beta = float(ones @ solved_res)
        alpha = float(ones @ solved_ones)
        if alpha <= 0:
            return np.inf

        M_best = beta / alpha
        self._last_best_M = M_best
        adjusted = residual - M_best

        if self._cov_factor is not None:
            solved = cho_solve(self._cov_factor, adjusted)
            return float(adjusted.T @ solved)

        if self._cov_inv is not None:
            solved = self._cov_inv @ adjusted
            return float(adjusted.T @ solved)

        variance = self.mu_err ** 2
        if np.any(~np.isfinite(variance)):
            return np.inf
        if np.any(variance <= 0):
            return np.inf
        return float(np.sum(adjusted ** 2 / variance))

    def last_best_M(self) -> Optional[float]:
        return self._last_best_M

    def count_points(self) -> int:
        return int(self.z.size)

    def covariance_rank(self) -> int:
        return int(self._cov_rank)

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

    def describe(self):
        info = {
            'n_sne': int(self.z.size),
            'source_file': self.source_file,
            'covariance_file': self._cov_source,
            'covariance_rank': int(self._cov_rank) if self.cov is not None else None,
        }
        return info
