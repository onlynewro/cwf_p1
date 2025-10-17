"""BAO data loading and manipulation utilities."""
from __future__ import annotations

import copy
import json
import os
from contextlib import contextmanager

import numpy as np
import pandas as pd

from src.utils.cosmology import C_LIGHT, comoving_distance


class BAOData:
    """BAO data handler."""

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
        """Load BAO data from JSON file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                self.data = [self._normalize_entry(entry) for entry in raw_data]
                self._base_data = copy.deepcopy(self.data)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Warning: Could not load BAO data from {filename}: {exc}")
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

        with open(mean_path, 'r', encoding='utf-8') as f:
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
        """Compute comoving distance DM."""
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
