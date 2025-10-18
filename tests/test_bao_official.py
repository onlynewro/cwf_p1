import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from joint_fit_multiproc import BAOData


def _read_raw_order(mean_path):
    raw_order = []
    with open(mean_path, 'r') as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            quantity_name = parts[2]
            raw_order.append(quantity_name.replace('rs', 'rd'))
    return raw_order


def test_official_bao_diagonal_errors_match_covariance():
    dataset = 'DESI LRG GCcomb z=0.4-0.6'
    mean_path = 'desi_2024_gaussian_bao_LRG_GCcomb_z0.4-0.6_mean.txt'
    cov_path = 'desi_2024_gaussian_bao_LRG_GCcomb_z0.4-0.6_cov.txt'

    bao = BAOData(use_official_covariance=False, include_proxy=False)
    entry = next(point for point in bao.data if point['name'] == dataset)

    raw_order = _read_raw_order(mean_path)

    target_order = []
    if entry['DM_over_rd'] is not None:
        target_order.append('DM_over_rd')
    if entry['DH_over_rd'] is not None:
        target_order.append('DH_over_rd')

    cov = np.loadtxt(cov_path, dtype=float)
    cov = np.array(cov, dtype=float)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)

    perm = [raw_order.index(q) for q in target_order]
    diag = np.diag(cov)[perm]
    expected_err = np.sqrt(diag)

    order_index = {name: idx for idx, name in enumerate(target_order)}

    assert entry['err_DM'] is not None
    assert np.isclose(entry['err_DM'], expected_err[order_index['DM_over_rd']])

    assert entry['err_DH'] is not None
    assert np.isclose(entry['err_DH'], expected_err[order_index['DH_over_rd']])
