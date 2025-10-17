# Cosmology Workflow Project

This repository bundles public DESI, SDSS/BOSS and Pantheon+SH0ES catalogs together with a
Python pipeline for performing joint BAO, SN and CMB fits.  The codebase has been reorganised
around a `src/` package so the command line interface and reusable modules can be imported
cleanly.

## Project layout

```
├── data/
│   └── raw/
│       ├── desi/          # Official DESI releases (means, covariances, DR2 bundle)
│       ├── sdss/          # SDSS/BOSS consensus tables
│       ├── pantheon/      # Pantheon+SH0ES catalogue
│       └── test/          # Miscellaneous helper data for debugging
├── scripts/
│   └── run_analysis.py    # Thin wrapper that launches the CLI from the repo root
├── src/
│   ├── analysis/          # Objective, covariance and statistics helpers
│   ├── data_loaders/      # DESI/SDSS/Pantheon+ loaders
│   ├── models/            # Cosmological models and background functions
│   └── main.py            # Command line entry point used by the wrappers above
└── joint_fit_multiproc.py # Backwards-compatible shim that delegates to src.main
```

All raw datasets live under `data/raw/...` so the repository root only contains code.
Empty directories are kept in version control via `.gitkeep` files.

## Running the analysis

From the project root you can launch the CLI through the helper script:

```bash
python scripts/run_analysis.py --help
```

Any arguments understood by the legacy `joint_fit_multiproc.py` script remain valid.  The
wrapper still exists for backwards compatibility so existing automation can keep calling it:

```bash
python joint_fit_multiproc.py --model 7D --use-default-bao --use-cmb
```

The CLI writes results to `fit_results.json` by default.  You can override the output path with
`--output`.

## Working with the code

* Cosmology models (`LCDM`, `7D`, and the optional `r_d` wrapper) live in
  `src/models/cosmology.py`.
* Dataset readers for DESI BAO, Pantheon+ supernovae and Planck distance priors are grouped in
  `src/data_loaders/`.
* Shared analysis helpers (objective function, Hessian-based covariance estimator and statistics)
  reside in `src/analysis/`.

The modular structure makes it possible to reuse the analysis logic in notebooks or other
applications by importing from the `src` package.
