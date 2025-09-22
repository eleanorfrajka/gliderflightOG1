import xarray as xr
from pandas.io.formats.style import Styler
from pandas import DataFrame

from gliderflightOG1 import plotters


def test_show_variables_returns_styler():
    # Create a dummy dataset
    ds = xr.Dataset(
        {
            "moc_mar_hc10": (
                ["TIME"],
                [1.0, 2.0, 3.0],
                {
                    "units": "Sv",
                    "comment": "Test variable",
                    "standard_name": "ocean_mass_transport",
                },
            )
        },
        coords={"TIME": ["2000-01-01", "2000-01-02", "2000-01-03"]},
    )

    styled = plotters.show_variables(ds)

    assert isinstance(styled, Styler)


def test_show_attributes_returns_styler():
    # Create a dummy dataset
    ds = xr.Dataset(
        {
            "moc_mar_hc10": (
                ["TIME"],
                [1.0, 2.0, 3.0],
                {
                    "units": "Sv",
                    "comment": "Test variable",
                    "standard_name": "ocean_mass_transport",
                },
            )
        },
        coords={"TIME": ["2000-01-01", "2000-01-02", "2000-01-03"]},
    )

    # Add global attributes
    ds.attrs["title"] = "Test Dataset"
    ds.attrs["institution"] = "Example Institute"

    df = plotters.show_attributes(ds)

    assert isinstance(df, DataFrame)
