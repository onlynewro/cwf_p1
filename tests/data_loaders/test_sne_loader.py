"""Tests for the supernova data loader."""

from src.data_loaders.sne_loader import SNData


def test_pantheon_shoes_file_loads_all_entries(tmp_path):
    """Ensure the packaged Pantheon+SH0ES catalogue is ingested fully."""

    # Load using the project root (``tmp_path`` is unused but keeps the fixture available
    # for future extensions that might need a writable directory).
    sn_data = SNData('Pantheon+SH0ES.dat', config={'base_dir': '.'})

    assert sn_data.count_points() == 1701
    assert sn_data.column_mapping()['z'] == 'zHD'
    assert sn_data.column_mapping()['mu'] == 'MU_SH0ES'
    assert sn_data.mu_obs[0] == sn_data.data.loc[sn_data.data.index[0], 'MU_SH0ES']
