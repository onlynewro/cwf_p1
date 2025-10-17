# BAO data

This repo contains BAO data of
- DESI DR2 ([arXiv:2503.14738](https://arxiv.org/abs/2503.14738) and [arXiv:2503.14739](https://arxiv.org/abs/2503.14739))
- DESI DR1 ([arXiv:2404.03000](https://arxiv.org/abs/2404.03000), [arXiv:2404.03001](https://arxiv.org/abs/2404.03001) and [arXiv:2404.03002](https://arxiv.org/abs/2404.03002))
- eBOSS DR16 ([arXiv:2007.08991](https://arxiv.org/pdf/2007.08991.pdf)), SDSS DR7 MGS ([arXiv:1409.3242](https://arxiv.org/abs/1409.3242)) and SDSS DR12 ([arXiv:1607.03155](https://arxiv.org/abs/1607.03155)) as originally distributed with [CosmoMC](https://github.com/cmbant/CosmoMC).

## Requirements

- Python 3.10 or newer.
- Linux (x86_64) or macOS (Apple Silicon or Intel). Windows users are encouraged to work inside [WSL2](https://learn.microsoft.com/windows/wsl/) for compatibility with the data-loading scripts.

## Installation

The repository ships with reproducible environment specifications for both `pip` and Conda users. Choose the workflow that best matches your toolchain.

### pip / virtualenv

1. Create and activate a Python 3.10+ virtual environment.
2. Install the pinned runtime dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Conda / mamba

1. Create the environment from the provided specification:

   ```bash
   conda env create -f environment.yml
   ```

2. Activate the environment (Conda):

   ```bash
   conda activate bao-analysis
   ```

   or, if you use [mamba](https://mamba.readthedocs.io/en/latest/), replace `conda` with `mamba` in the commands above.

## Container builds

The Docker build context now includes both `requirements.txt` and `environment.yml`, enabling containerized workflows to reuse the same pinned dependency set.

