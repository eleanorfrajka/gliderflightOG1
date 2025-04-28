import numpy as np
import pytest

from flightmodels import seaglider

import numpy as np
from flightmodels import seaglider

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
    wspdg = wg + np.random.normal(loc=0.0, scale=0.5, size=(nprofiles, ndepths))  # modeled w slightly different

    # Synthetic mixed layer depths (some shallow, some deep)
    mld = np.array([50, 150, 300, 20, 80])

    # --- Test ml_coord ---
    wmean_ml, wsqr_ml, h_ml, hgrid_ml = seaglider.ml_coord(wg, wspdg, pgrid, mld, minmld=40)

    print(f"ml_coord test results:")
    print(f"wmean shape: {wmean_ml.shape}")
    print(f"wsqr shape: {wsqr_ml.shape}")
    print(f"h (ML depth) shape: {h_ml.shape}")
    print(f"hgrid (normalized) shape: {hgrid_ml.shape}")

    assert wmean_ml.shape == (nprofiles,), "wmean_ml shape mismatch"
    assert wsqr_ml.shape == (nprofiles,), "wsqr_ml shape mismatch"
    assert h_ml.shape == (nprofiles,), "h_ml shape mismatch"
    assert np.all(np.isfinite(hgrid_ml)), "hgrid_ml contains NaNs"

    # --- Test bl_coord ---
    wmean_bl = seaglider.bl_coord(wg, wspdg, pgrid, mld)

    print("bl_coord test results:")
    print(f"wmean_bl shape: {wmean_bl.shape}")

    assert wmean_bl.shape == (nprofiles,), "wmean_bl shape mismatch"

    print("✅ test_ml_bl_coord passed successfully.")



def test_flightvec_basic():
    # Simple synthetic test data
    buoy = np.array([10, -20, 30])   # buoyancy in grams
    pitch = np.array([10, -15, 20])  # pitch angles in degrees
    xl = 0.5                         # length scale (meters)
    hd_a = 1.0                       # hydrodynamic coefficient a
    hd_b = 0.5                       # hydrodynamic coefficient b
    hd_c = 0.2                       # hydrodynamic coefficient c
    rho0 = 1025.0                    # seawater density (kg/m³)

    # Call the function
    umag, thdeg = seaglider.flightvec(buoy, pitch, xl, hd_a, hd_b, hd_c, rho0)

    # Check output shapes match input
    assert umag.shape == buoy.shape
    assert thdeg.shape == buoy.shape

    # Check that outputs are finite numbers
    assert np.all(np.isfinite(umag))
    assert np.all(np.isfinite(thdeg))

    # Optional: check that speeds are non-negative
    assert np.all(umag >= 0)

    # Optional: check that glide angles are reasonable
    assert np.all(np.abs(thdeg) <= 90)

def test_flightvec_unstdy():
    """
    Simple test for flightvec_unstdy function.
    """

    time = np.linspace(0, 1000, 100)  # seconds
    buoy = np.ones(100) * 0.05  # arbitrary buoyancy
    pitch = np.linspace(-30, 30, 100)  # glide from dive to climb
    xl = 1.8  # meters
    hd_a = 0.0036
    hd_b = 0.0098
    hd_c = 0.0010
    rho0 = 1025  # kg/m³

    spd, ang = seaglider.flightvec_unstdy(time, buoy, pitch, xl, hd_a, hd_b, hd_c, rho0)

    print("Results:")
    for key, val in spd.items():
        print(f"{key}: mean {np.mean(val):.2f}, min {np.min(val):.2f}, max {np.max(val):.2f}")

    for key, val in ang.items():
        print(f"{key}: mean {np.mean(val):.2f} deg")

