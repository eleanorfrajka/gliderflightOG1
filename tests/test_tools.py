import numpy as np
import xarray as xr

from flightmodels import tools


def test_ml_bl_coord():
    """
    Unit test for seaglider.ml_coord() and seaglider.bl_coord().

    Creates synthetic glider profile data and verifies basic functionality.
    """

    # Create synthetic dataset
    nprofiles = 5
    ndepths = 20
    pgrid = np.linspace(0, 1000, ndepths)  # Pressure grid (dbar)

    # Synthetic measured and modeled vertical speeds
    np.random.seed(42)
    wg = np.random.normal(loc=0.0, scale=2.0, size=(nprofiles, ndepths))  # measured w
    wspdg = wg + np.random.normal(
        loc=0.0, scale=0.5, size=(nprofiles, ndepths)
    )  # modeled w slightly different

    # Synthetic mixed layer depths (some shallow, some deep)
    mld = np.array([50, 150, 300, 20, 80])

    # --- Test ml_coord ---
    wmean_ml, wsqr_ml, h_ml, hgrid_ml = tools.ml_coord(wg, wspdg, pgrid, mld, minmld=40)

    print("ml_coord test results:")
    print(f"wmean shape: {wmean_ml.shape}")
    print(f"wsqr shape: {wsqr_ml.shape}")
    print(f"h (ML depth) shape: {h_ml.shape}")
    print(f"hgrid (normalized) shape: {hgrid_ml.shape}")

    assert wmean_ml.shape == (nprofiles,), "wmean_ml shape mismatch"
    assert wsqr_ml.shape == (nprofiles,), "wsqr_ml shape mismatch"
    assert h_ml.shape == (nprofiles,), "h_ml shape mismatch"
    assert np.all(np.isfinite(hgrid_ml)), "hgrid_ml contains NaNs"

    # --- Test bl_coord ---
    wmean_bl = tools.bl_coord(wg, wspdg, pgrid, mld)

    print("bl_coord test results:")
    print(f"wmean_bl shape: {wmean_bl.shape}")

    assert wmean_bl.shape == (nprofiles,), "wmean_bl shape mismatch"

    print("✅ test_ml_bl_coord passed successfully.")


def test_reformat_units_var_sv_conversion():
    # Create a fake transport DataArray with units in m3/s
    ds = xr.Dataset(
        {
            "transport": xr.DataArray(
                data=np.array([1.0e6, 2.0e6]),
                dims=["time"],
                attrs={"units": "m^3/s", "long_name": "Volume transport"},
            ),
            "velocity": xr.DataArray(
                data=np.array([100.0, 200.0]),
                dims=["time"],
                attrs={"units": "cm/s", "long_name": "Flow velocity"},
            ),
        }
    )

    new_unit = tools.reformat_units_var(ds, "transport")
    # Units should now be Sv
    assert new_unit == "Sv"

    new_unit = tools.reformat_units_var(ds, "velocity")
    assert new_unit == "cm s-1"


def test_convert_units_var():
    var_values = 100
    current_units = "cm/s"
    new_units = "m/s"
    converted_values = tools.convert_units_var(var_values, current_units, new_units)
    assert converted_values == 1.0
