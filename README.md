# gliderflightOG1

This is a starting point for thinking about running glider flight models on data in OG1 format.

📘 Documentation is available at:
👉 https://eleanorfrajka.github.io/gliderflightOG1/

Originating from conversations associated with the "vertical velocities" group at [OceanGlidersCommunity](https://github.com/OceanGlidersCommunity/Vertical_Velocities_SOP).  If/when this package is in good shape, it could be transferred to http://github.com/OceanGlidersCommunity or some other relevant organisation.

---

## 🚀 What's Included

- ✅ Python package layout: `gliderflightOG1/*.py`
- 📓 Jupyter notebook demo: `notebooks/demo.ipynb`
- 📄 Markdown and Sphinx-based documentation in `docs/`
- 🔍 Tests with `pytest` in `tests/`, CI with GitHub Actions
- 🎨 Code style via `black`, `ruff`, `pre-commit`
- 📦 Package config via `pyproject.toml` + optional PyPI release workflow
- 🧾 Machine-readable citation: `CITATION.cff`

---

## 🔧 Quickstart

Install in development mode:

```bash
git clone https://github.com/eleanorfrajka/gliderflightOG1.git
cd gliderflightOG1
python -m venv venv       # if you manage environments with venv
source venv/bin/activate  # if you manage environments with venv
pip install -r requirements-dev.txt
pip install -e .
```

To run tests:

```bash
pytest
```

To build the documentation locally:

```bash
cd docs
make html
```

---

## 🤝 Contributing

Contributions are welcome!  Please also consider adding an [issue](https://github.com/eleanorfrajka/flightmodels/issues) when something isn't clear.

---

## Future plans

Incorporate flight models based on FW2011 (to be added, matlab based) and the [Seaglider basestation](https://github.com/iop-apl-uw/basestation3/blob/master/FlightModel.py) designed to run on Seaglider data, and from Lucas Merckelbach's [gliderflight](https://gliderflight.readthedocs.io/en/latest/using_gliderflight.html) designed to run on Slocum data.

