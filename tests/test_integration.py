from pathlib import Path

import math

import numpy as np
import pytest

from joint_fit_multiproc import calculate_statistics, total_chi2
from src.data_loaders.bao_loader import BAOData
from src.data_loaders.sne_loader import SNData
from src.models.lcdm import LCDMModel


@pytest.fixture(scope="module")
def fixture_paths():
    base = Path(__file__).resolve().parent.parent / "data" / "processed"
    return {
        'bao': base / "bao_mock.json",
        'sn': base / "sn_mock.csv",
    }


def test_mock_dataset_fit_is_deterministic(fixture_paths):
    bao = BAOData(filename=fixture_paths['bao'])
    sn = SNData(filename=fixture_paths['sn'], marginalize_m=False)

    datasets = {'bao': bao, 'sn': sn}
    params = np.array([0.69, 0.31])
    model = LCDMModel()

    chi2 = total_chi2(params, datasets, model)

    assert np.isfinite(chi2)
    assert chi2 == pytest.approx(1.6332213273, rel=5e-2, abs=1e-2)
    assert bao.count_observables() == 4
    assert sn.count_points() == 3


def test_total_chi2_breakdown_matches_component_chi2(fixture_paths):
    bao = BAOData(filename=fixture_paths['bao'])
    sn = SNData(filename=fixture_paths['sn'], marginalize_m=False)

    datasets = {'bao': bao, 'sn': sn}
    params = np.array([0.69, 0.31])
    model = LCDMModel()

    chi2_total, components = total_chi2(
        params,
        datasets,
        model,
        return_components=True,
    )

    component_sum = sum(value for value in components.values() if value is not None)

    assert chi2_total == pytest.approx(component_sum)
    assert components['bao'] == pytest.approx(bao.last_chi2)
    assert components['sn'] == pytest.approx(sn.last_chi2)
    assert components['cmb'] is None
    assert math.isfinite(chi2_total)


def test_calculate_statistics_reports_counts():
    stats = calculate_statistics(chi2=10.0, n_data=20, n_params=3)

    assert stats['chi2'] == 10.0
    assert stats['dof'] == 17
    assert stats['n_data'] == 20
    assert stats['n_params'] == 3
    assert stats['chi2_red'] == pytest.approx(10.0 / 17)
    assert stats['aic'] == pytest.approx(10.0 + 2 * 3)
    assert stats['bic'] == pytest.approx(10.0 + 3 * math.log(20))
