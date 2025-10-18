"""Tests for the supernova data loader."""

import pytest

from src.data_loaders.sne_loader import SNData
from src.utils.validation import ConfigValidationError


def test_pantheon_shoes_file_loads_all_entries(tmp_path):
    """Ensure the packaged Pantheon+SH0ES catalogue is ingested fully."""

    # Load using the project root (``tmp_path`` is unused but keeps the fixture available
    # for future extensions that might need a writable directory).
    sn_data = SNData('Pantheon+SH0ES.dat', config={'base_dir': '.'})

    assert sn_data.count_points() == 1701
    assert sn_data.column_mapping()['z'] == 'zHD'
    assert sn_data.column_mapping()['mu'] == 'MU_SH0ES'
    assert sn_data.mu_obs[0] == sn_data.data.loc[sn_data.data.index[0], 'MU_SH0ES']


def test_loader_errors_without_uncertainty_column(tmp_path):
    """Ensure files missing an uncertainty column raise a validation error."""

    catalogue = tmp_path / 'sne_catalogue.csv'
    catalogue.write_text("z,mu\n0.1,35.0\n0.2,36.5\n", encoding='utf-8')

    with pytest.raises(ConfigValidationError):
        SNData(catalogue, config={'base_dir': '.'})
