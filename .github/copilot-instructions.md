# Copilot Instructions for gliderflightOG1

## Project Overview

This is a scientific Python package for running glider flight models on OG1 format oceanographic data. The core functionality involves physical modeling of autonomous underwater glider dynamics through buoyancy, pitch, and drag coefficient calculations.

## Architecture & Data Flow

**Core Components:**
- `seaglider.py`: Main flight model implementation with `flightvec()` for velocity calculations and `regress_all_vec()` for parameter optimization
- `tools.py`: Unit conversions, calculations (especially `calc_w_meas()` for vertical velocities)
- `readers.py`/`writers.py`: Data I/O for xarray datasets
- `plotters.py`: Scientific plotting functions
- `utilities.py`: Variable validation via `_check_necessary_variables()`

**Data Pipeline:**
1. OG1 format NetCDF → xarray Dataset (via `readers.py`)
2. Physical parameter regression → optimized coefficients (`regress_all_vec()`)
3. Flight model calculation → velocities/angles (`flightvec_ds()`)
4. Visualization → scientific plots (`plotters.py`)

## OG1 Data Format Specifics

This package expects OG1 (OceanGliders) format following CF-1.10 conventions. Key variables:
- **Required for flight models**: `VBD`, `C_VBD`, `PRES`/`DEPTH`, `TEMP`, `PITCH`, `UPDN`
- **Model outputs**: `GLIDER_VERT_VELO_MODEL`, `GLIDER_HORZ_VELO_MODEL`, `GLIDE_ANGLE`
- **Attributes needed**: `hd_a`, `hd_b`, `hd_c`, `vbdbias`, `mass`, `rho0`, compression/expansion coefficients

Use `tools.calc_w_meas()` to calculate vertical speeds from pressure gradients.

## Development Conventions

**Function Naming:**
- `plot_*()` for visualization functions
- `calc_*()` for calculations  
- `compute_*()` for complex computations
- `regress_*()` for optimization/fitting

**Input/Output Patterns:**
- Primary data format: xarray Datasets with CF-compliant metadata
- Physical parameters stored in dataset `.attrs` dictionary
- Use `gliderflightOG1.utilities._check_necessary_variables()` for input validation
- Follow numpy docstring format

**Testing Strategy:**
- Tests mirror package structure: `test_seaglider.py` tests `seaglider.py`
- Use sample data from `data/` directory for integration tests
- Focus on numerical accuracy and shape validation for scientific functions

## Key Workflows

**Development Setup:**
```bash
pip install -r requirements-dev.txt
pip install -e .
```

**Quality Checks (use VS Code tasks or run manually):**
```bash
pytest -v                    # Run tests
ruff check --fix .           # Auto-fix linting
black .                      # Format code
pre-commit run --all-files   # Run all hooks
```

**Typical Flight Model Workflow:**
1. Load OG1 data: `glider = xr.open_dataset('file.nc')`
2. Add required variables if missing (see `notebooks/demo.ipynb`)
3. Set physical parameters in `glider.attrs`
4. Optimize parameters: `regressout = seaglider.regress_all_vec()`
5. Calculate flight: `result = seaglider.flightvec_ds(glider)`

## Integration Points

**External Dependencies:**
- `gsw`: TEOS-10 seawater calculations (density, etc.)
- `xarray`/`netcdf4`: CF-compliant data handling
- `scipy`: Optimization and numerical integration
- `matplotlib`: Scientific plotting with custom style in `flightmodels.mplstyle`

**Data Sources:**
- Sample datasets in `data/`: `sg014_*.nc` (Seaglider), `moc_transports.nc`
- OG1 configuration: https://github.com/ocean-uhh/seagliderOG1/tree/main/seagliderOG1/config

When implementing new features, follow the scientific computing patterns established in the existing codebase, ensure proper error handling for missing oceanographic variables, and maintain compatibility with the OG1 data standard.