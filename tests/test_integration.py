from pathlib import Path

import numpy as np
import pytest

from joint_fit_multiproc import total_chi2
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
