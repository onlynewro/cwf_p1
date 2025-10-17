import numpy as np
import pytest

from src.data_loaders.bao_loader import BAOData


class DummyModel:
    def E(self, z, params):
        return 1.0

    def Hz(self, z, params):
        return 100.0


def test_bao_chi2_uses_covariance(monkeypatch):
    bao = BAOData()
    bao.data = [
        {
            'name': 'test',
            'z': 0.5,
            'DM_over_rd': 10.0,
            'DH_over_rd': 20.0,
            'err_DM': 0.5,
            'err_DH': 0.4,
            'cov_matrix': np.array([[0.25, 0.05], [0.05, 0.16]]),
            '_observable_order': ['DM_over_rd', 'DH_over_rd'],
        }
    ]

    def fake_dm(z, model, params):
        return 9.6

    def fake_dh(z, model, params, rd_value):
        return 19.5

    monkeypatch.setattr(bao, 'compute_DM', fake_dm)
    monkeypatch.setattr(bao, 'compute_DH_over_rd', fake_dh)

    diff = np.array([10.0 - 9.6, 20.0 - 19.5])
    cov = np.array([[0.25, 0.05], [0.05, 0.16]])
    expected = float(diff.T @ np.linalg.solve(cov, diff))

    chi2 = bao.chi2(DummyModel(), params=(0.7, 0.3), rd_value=1.0)

    assert chi2 == pytest.approx(expected)
    assert bao.used_covariance_last_call()
