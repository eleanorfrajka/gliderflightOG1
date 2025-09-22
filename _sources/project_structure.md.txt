# What's in gliderflightOG1?

Below is an overview of the files and folders you'll find in the `gliderflightOG1`, along with what they do and why they're useful. If you're new to GitHub or Python packaging, this is your orientation.

---

## 🔍 Project Structure Overview

📷 *This is what the package looks like when you clone or fork it:*

# 📁 `gliderflightOG1` File Structure

A minimal, modular Python project structure for collaborative research and reproducible workflows.

```
gliderflightOG1/
├── gliderflightOG1               # [core] Main Python package with scientific code
│   ├── __init__.py               # [core] Makes this a Python package
│   ├── plotters.py               # [core] Functions to plot data
│   ├── readers.py                # [core] Functions to read raw data into xarray datasets
│   ├── flight.py                 # [core] Functions for glider flight
│   ├── writers.py                # [core] Functions to write data (e.g., to NetCDF)
│   ├── tools.py                  # [core] Utilities for unit conversion, calculations, etc.
│   ├── logger.py                 # [core] Structured logging configuration for reproducible runs
│   ├── gliderflightOG1.mplstyle  # [core] Default plotting parameters
│   └── utilities.py              # [core] Helper functions (e.g., file download or parsing)
│
├── tests/                        # [test] Unit tests using pytest
│   ├── test_readers.py           # [test] Test functions in readers.py
│   ├── test_tools.py             # [test] Test functions in tools.py
│   ├── test_utilities.py         # [test] Test functions in utilities.py
│   └── ...
│
├── docs/                         # [docs]
│   ├── source/                   # [docs] Sphinx documentation source files
│   │   ├── conf.py               # [docs] Setup for documentation
│   │   ├── index.rst             # [docs] Main page with menus in *.rst
│   │   ├── setup.md              # [docs] One of the documentation pages in *.md
│   │   ├── gliderflightOG1.rst   # [docs] The file to create the API based on docstrings
│   │   ├── ...                   # [docs] More *.md or *.rst linked in index.rst
│   │   └── _static               # [docs] Figures
│   │       ├── css/custom.css    # [docs, style] Custom style sheet for docs
│   │       └── logo.png          # [docs] logo for top left of docs/
│   └── Makefile                  # [docs] Build the docs
│
├── notebooks/                    # [demo] Example notebooks
│   ├── demo.ipynb                # [demo] Also run in docs.yml to appear in docs
│   └── ...
│
├── data/                         # [data]
│   └── temp.nc                   # [data] Example data file used
│
├── logs/                         # [core] Log output from structured logging
│   └── *.log                     # [core]
│
├── .github/                      # [ci] GitHub-specific workflows (e.g., Actions)
│   ├── workflows/
│   │   ├── docs.yml              # [ci] Test build documents on *pull-request*
│   │   ├── docs_deploy.yml       # [ci] Build and deploy documents on "merge"
│   │   ├── pypi.yml              # [ci] Package and release on GitHub.com "release"
│   │   └── test.yml              # [ci] Run pytest on tests/test_<name>.py on *pull-request*
│   ├── ISSUE_TEMPLATE.md         # [ci, meta] Template for issues on Github
│   └── PULL_REQUEST_TEMPLATE.md  # [ci, meta] Template for pull requests on Github
│
├── .gitignore                    # [meta] Exclude build files, logs, data, etc.
├── requirements.txt              # [meta] Pip requirements
├── requirements-dev.txt          # [meta] Pip requirements for development (docs, tests, linting)
├── .pre-commit-config.yaml       # [style] Instructions for pre-commits to run (linting)
├── pyproject.toml                # [ci, meta, style] Build system and config linters
├── CITATION.cff                  # [meta] So Github can populate the "cite" button
├── README.md                     # [meta] Project overview and getting started
└── LICENSE                       # [meta] Open source license (e.g., MIT as default)
```

The tags above give an indication of what parts of this project are used for what purposes, where:
- `# [core]` – Scientific core logic or core functions used across the project.
<!--- `# [api]` – Public-facing functions or modules users are expected to import and use.-->
- `# [docs]` – Documentation sources, configs, and assets for building project docs.
- `# [test]` – Automated tests for validating functionality.
- `# [demo]` – Notebooks and minimal working examples for demos or tutorials.
- `# [data]` – Sample or test data files.
- `# [ci]` – Continuous integration setup (GitHub Actions).
- `# [style]` – Configuration for code style, linting, and formatting.
- `# [meta]` – Project metadata (e.g., citation info, license, README).

**Note:** There are also files that you may end up generating but which don't necessarily appear in the project on GitHub.com (due to being ignored by your `.gitignore`).  These may include your environment (`venv/`, if you use pip and virtual environments), distribution files `dist/` for building packages to deploy on http://pypi.org, `htmlcov/` for coverage reports for tests, `gliderflightOG1.egg-info` for editable installs (e.g., `pip install -e .`).

