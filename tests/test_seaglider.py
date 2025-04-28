import numpy as np

from flightmodels import seaglider


def test_flightvec_basic():
    # Simple synthetic test data
    buoy = np.array([10, -20, 30])  # buoyancy in grams
    pitch = np.array([10, -15, 20])  # pitch angles in degrees
    xl = 0.5  # length scale (meters)
    hd_a = 1.0  # hydrodynamic coefficient a
    hd_b = 0.5  # hydrodynamic coefficient b
    hd_c = 0.2  # hydrodynamic coefficient c
    rho0 = 1025.0  # seawater density (kg/m³)

    # Call the function
    umag, thdeg = seaglider.flightvec(buoy, pitch, xl, hd_a, hd_b, hd_c, rho0)
    thdeg = thdeg.astype(float)  # Ensure thdeg is a float array to handle NaN values

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
        print(
            f"{key}: mean {np.mean(val):.2f}, min {np.min(val):.2f}, max {np.max(val):.2f}"
        )

    for key, val in ang.items():
        print(f"{key}: mean {np.mean(val):.2f} deg")
