import logging


import numpy as np
import gsw
from flightmodels import utilities

# Initialize logging
_log = logging.getLogger(__name__)
# Various conversions from the key to units_name with the multiplicative conversion factor
unit_conversion = {
    "cm/s": {"units_name": "m/s", "factor": 0.01},
    "cm s-1": {"units_name": "m s-1", "factor": 0.01},
    "m/s": {"units_name": "cm/s", "factor": 100},
    "m s-1": {"units_name": "cm s-1", "factor": 100},
    "S/m": {"units_name": "mS/cm", "factor": 0.1},
    "S m-1": {"units_name": "mS cm-1", "factor": 0.1},
    "mS/cm": {"units_name": "S/m", "factor": 10},
    "mS cm-1": {"units_name": "S m-1", "factor": 10},
    "dbar": {"units_name": "Pa", "factor": 10000},
    "Pa": {"units_name": "dbar", "factor": 0.0001},
    "degrees_Celsius": {"units_name": "Celsius", "factor": 1},
    "Celsius": {"units_name": "degrees_Celsius", "factor": 1},
    "m": {"units_name": "cm", "factor": 100},
    "cm": {"units_name": "m", "factor": 0.01},
    "km": {"units_name": "m", "factor": 1000},
    "g m-3": {"units_name": "kg m-3", "factor": 0.001},
    "kg m-3": {"units_name": "g m-3", "factor": 1000},
}

# Specify the preferred units, and it will convert if the conversion is available in unit_conversion
preferred_units = ["m s-1", "dbar", "S m-1"]

# String formats for units.  The key is the original, the value is the desired format
unit_str_format = {
    "m/s": "m s-1",
    "cm/s": "cm s-1",
    "S/m": "S m-1",
    "meters": "m",
    "degrees_Celsius": "Celsius",
    "g/m^3": "g m-3",
    "m^3/s": "Sv",
}


def compute_insitu_density(
    salinity, temperature, pressure, longitude=None, latitude=None
):
    """
    Compute in-situ density from salinity, temperature, and pressure using GSW.

    Parameters
    ----------
    salinity : array-like
        Practical salinity (PSU).
    temperature : array-like
        In-situ temperature (°C).
    pressure : array-like
        Pressure (dbar).
    longitude : array-like, optional
        Longitude (degrees East). If not provided, defaults to -40°.
    latitude : array-like, optional
        Latitude (degrees North). If not provided, defaults to 30°.

    Returns
    -------
    density : array-like
        In-situ density (kg/m³).
    """

    # If longitude and latitude are not provided, use dummy values
    if longitude is None:
        longitude = np.full_like(pressure, -40.0)
    if latitude is None:
        latitude = np.full_like(pressure, 60.0)

    # Step 1: Convert Practical Salinity to Absolute Salinity
    SA = gsw.SA_from_SP(salinity, pressure, longitude, latitude)

    # Step 2: Convert in-situ Temperature to Conservative Temperature
    CT = gsw.CT_from_t(SA, temperature, pressure)

    # Step 3: Compute in-situ density
    density = gsw.rho(SA, CT, pressure)

    return density


def calc_w_meas(ds):
    """
    Calculates the vertical velocity of a glider using changes in pressure with time.

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset containing **DEPTH** and **TIME**.

    Returns
    -------
    ds: xarray.Dataset
        Containing the new variable **GLIDER_VERT_VELO_DZDT** (array-like), with vertical velocities calculated from dz/dt

    Notes
    ------
    Original Author: Eleanor Frajka-Williams
    """
    utilities._check_necessary_variables(ds, ["TIME"])
    # Ensure inputs are numpy arrays
    time = ds.TIME.values
    if "DEPTH_Z" not in ds.variables and all(
        var in ds.variables for var in ["PRES", "LATITUDE", "LONGITUDE"]
    ):
        ds = utilities.calc_DEPTH_Z(ds)
    depth = ds.DEPTH_Z.values

    # Calculate the centered differences in pressure and time, i.e. instead of using neighboring points,
    # use the points two steps away.  This has a couple of advantages: one being a slight smoothing of the
    # differences, and the other that the calculated speed will be the speed at the midpoint of the two
    # points.
    # For data which are evenly spaced in time, this will be equivalent to a centered difference.
    # For data which are not evenly spaced in time, i.e. when a Seaglider sample rate changes from 5
    # seconds to 10 seconds, there may be some uneven weighting of the differences.
    delta_z_meters = depth[2:] - depth[:-2]
    delta_time_datetime64ns = time[2:] - time[:-2]
    delta_time_sec = delta_time_datetime64ns / np.timedelta64(
        1, "s"
    )  # Convert to seconds

    # Calculate vertical velocity (rate of change of pressure with time)
    vertical_velocity = delta_z_meters / delta_time_sec

    # Pad the result to match the original array length
    vertical_velocity = np.pad(vertical_velocity, (1, 1), "edge")

    # No - Convert vertical velocity from m/s to cm/s
    vertical_velocity = vertical_velocity

    # Add vertical velocity to the dataset
    ds = ds.assign(
        GLIDER_VERT_VELO_DZDT=(
            ("N_MEASUREMENTS"),
            vertical_velocity,
            {"long_name": "glider_vertical_speed_from_pressure", "units": "m s-1"},
        )
    )

    return ds


def gridthem(w_measured, w_model, time, divenum, updn, press, pgrid):
    """
    Bin average vertical speeds into pressure bins.

    Parameters
    ----------
    w_measured : array
    w_model : array
    time : array
    divenum : array
    updn : array
    press : array
    pgrid : array
        Pressure grid.

    Returns
    -------
    wg : array
        Binned measured w.
    wspdg : array
        Binned modeled w.
    timeg : array
        Binned times.
    divenumg : array
        Binned dive numbers.
    updng : array
        Binned up/down flags.
    """

    def bin_avg(segP, segD, pgrid):
        dgrid = np.full_like(pgrid[:-1], np.nan, dtype=float)
        phalf = np.zeros(len(pgrid) + 1)
        phalf[1:-1] = (pgrid[:-1] + pgrid[1:]) / 2
        phalf[0] = 0
        phalf[-1] = pgrid[-1] + 1

        for pdo in range(len(pgrid) - 1):
            plim = (phalf[pdo], phalf[pdo + 1])
            idx = np.where((segP >= plim[0]) & (segP < plim[1]))[0]
            if idx.size > 0:
                dgrid[pdo] = np.nanmean(segD[idx])

        return dgrid

    wg = bin_avg(press, w_measured, pgrid)
    wspdg = bin_avg(press, w_model, pgrid)
    timeg = bin_avg(press, time, pgrid)
    divenumg = bin_avg(press, divenum, pgrid)
    updng = bin_avg(press, updn, pgrid)

    return wg, wspdg, timeg, divenumg, updng


def ml_coord(wg, wspdg, pgrid, mld, minmld=40):
    """
    Compute means and variances for mixed-layer segments of profiles.

    Parameters
    ----------
    wg : ndarray
        Measured vertical speeds gridded (profiles x pressure levels).
    wspdg : ndarray
        Modeled vertical speeds gridded (profiles x pressure levels).
    pgrid : ndarray
        Pressure grid [dbar].
    mld : ndarray
        Mixed layer depth estimates [dbar] (one per profile).
    minmld : float, optional
        Minimum mixed layer depth to use [dbar] (default 40 dbar).

    Returns
    -------
    wmean : ndarray
        Mean vertical speed difference within ML for each profile.
    wsqr : ndarray
        RMS vertical speed difference within ML for each profile.
    h : ndarray
        Mixed layer depth used per profile [dbar].
    hgrid : ndarray
        Scaled pressure bin array (zgrid) [0-1].
    """

    nprofiles, nlevels = wg.shape
    wmean = np.full(nprofiles, np.nan)
    wsqr = np.full(nprofiles, np.nan)

    hgrid = pgrid[:-1] / 1000  # Convert dbar to km for plotting
    h = np.full(nprofiles, np.nan)

    for p in range(nprofiles):
        if np.isnan(mld[p]):
            mld[p] = 1001  # Fill missing MLD with deep dummy value

        mask = pgrid[:-1] <= max(minmld, mld[p])
        if np.any(mask):
            diff = wspdg[p, mask] - wg[p, mask]
            wmean[p] = np.nanmean(diff)
            wsqr[p] = np.nanmean(diff**2)
            h[p] = np.nanmean(pgrid[:-1][mask])

    return wmean, wsqr, h, hgrid


def bl_coord(wg, wspdg, pgrid, mld):
    """
    Compute means for below-mixed-layer segments of profiles.

    Parameters
    ----------
    wg : ndarray
        Measured vertical speeds gridded (profiles x pressure levels).
    wspdg : ndarray
        Modeled vertical speeds gridded (profiles x pressure levels).
    pgrid : ndarray
        Pressure grid [dbar].
    mld : ndarray
        Mixed layer depth estimates [dbar] (one per profile).

    Returns
    -------
    wmean : ndarray
        Mean difference (wspdg - wg) below mixed layer, per profile.
    """

    nprofiles, nlevels = wg.shape
    wmean = np.full(nprofiles, np.nan)

    for p in range(nprofiles):
        if np.isnan(mld[p]):
            mld[p] = 1001  # Fill missing MLD

        mask = pgrid[:-1] > mld[p]
        if np.any(mask):
            diff = wspdg[p, mask] - wg[p, mask]
            wmean[p] = np.nanmean(diff)

    return wmean


def choose_min_prof(
    wg, wspdg, timeg, mld, divenumg, updng, pgrid, whichone, plotflag=0
):
    """
    Choose minimization metric based on gridded profiles.

    Parameters
    ----------
    wg : ndarray
        Measured vertical speeds binned on grid.
    wspdg : ndarray
        Modeled vertical speeds binned on grid.
    timeg : ndarray
        Gridded time.
    mld : ndarray
        Mixed layer depth estimates.
    divenumg : ndarray
        Dive number array corresponding to grid.
    updng : ndarray
        Up/down indicator corresponding to grid.
    pgrid : ndarray
        Pressure bin grid [dbar].
    whichone : int
        Metric selector (5–8).
    plotflag : int, optional
        If >0, makes a plot.

    Returns
    -------
    allmin : dict
        Dictionary of minimization metric(s).
    tmp : array
        Profile-wise differences (typically dive-climb).
    """

    w_water = wspdg - wg
    allmin = {}

    updn1 = np.nanmean(updng, axis=1)
    idive = np.where(updn1 < -0.5)[0]
    iclimb = np.where(updn1 > 0.5)[0]

    if whichone == 5:
        # Mean w = 0 in mixed layer
        minmld = 40
        wmeanD, wsqrD, hgridD, _ = ml_coord(
            wg[idive, :], wspdg[idive, :], pgrid, mld[idive], minmld
        )
        wmeanC, wsqrC, hgridC, _ = ml_coord(
            wg[iclimb, :], wspdg[iclimb, :], pgrid, mld[iclimb], minmld
        )

        wskew = np.nanmean((wmeanD - np.nanmean(wmeanD)) ** 2) + np.nanmean(
            (wmeanC - np.nanmean(wmeanC)) ** 2
        )
        w_rms = np.nanmean(wmeanD**2 + wmeanC**2)

        allmin["mlprofdiff"] = w_rms
        tmp = np.abs(wmeanD - wmeanC)

    elif whichone == 6:
        # Full-depth dive-climb profile difference
        minmld = 50
        mld1000 = 1000 * np.ones_like(mld)
        wmeanD, wsqrD, hgridD, _ = ml_coord(
            wg[idive, :], wspdg[idive, :], pgrid, mld1000[idive], minmld
        )
        wmeanC, wsqrC, hgridC, _ = ml_coord(
            wg[iclimb, :], wspdg[iclimb, :], pgrid, mld1000[iclimb], minmld
        )

        tmp = np.abs(wmeanD - wmeanC)
        allmin["flprofdiff"] = np.nanmean(tmp)

        if plotflag:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(wmeanD, 1000 * hgridD, label="Dive")
            plt.plot(wmeanC, 1000 * hgridC, label="Climb")
            plt.axvline(0, color="k", linestyle="--")
            plt.gca().invert_yaxis()
            plt.xlabel("Vertical speed (cm/s)")
            plt.ylabel("Depth (m)")
            plt.legend()
            plt.title("Dive vs Climb Vertical Speeds (Full Depth)")
            plt.tight_layout()
            plt.show()

    elif whichone == 7:
        # Mean w = 0 for full profiles
        minmld = 0
        mld1000 = 1000 * np.ones_like(mld)
        wmeanD, wsqrD, hgridD, _ = ml_coord(
            wg[idive, :], wspdg[idive, :], pgrid, mld1000[idive], minmld
        )
        wmeanC, wsqrC, hgridC, _ = ml_coord(
            wg[iclimb, :], wspdg[iclimb, :], pgrid, mld1000[iclimb], minmld
        )

        if len(wmeanC) == 0:
            wmeanC = np.zeros_like(wmeanD)
        if len(wmeanD) == 0:
            wmeanD = np.zeros_like(wmeanC)

        wskew = np.nanmean((wmeanD - np.nanmean(wmeanD)) ** 2) + np.nanmean(
            (wmeanC - np.nanmean(wmeanC)) ** 2
        )
        w_rms = np.nanmean(wmeanD**2 + wmeanC**2) + wskew

        allmin["meanw"] = w_rms
        tmp = wmeanD - wmeanC

    elif whichone == 8:
        # Mean w = 0 below ML only
        wmeanD = bl_coord(wg[idive, :], wspdg[idive, :], pgrid, mld[idive])
        wmeanC = bl_coord(wg[iclimb, :], wspdg[iclimb, :], pgrid, mld[iclimb])

        if len(wmeanC) == 0:
            wmeanC = np.zeros_like(wmeanD)
        if len(wmeanD) == 0:
            wmeanD = np.zeros_like(wmeanC)

        wskew = np.nanmean((wmeanD - np.nanmean(wmeanD)) ** 2) + np.nanmean(
            (wmeanC - np.nanmean(wmeanC)) ** 2
        )
        w_rms = np.nanmean(wmeanD**2 + wmeanC**2) + wskew

        allmin["meanwbl"] = w_rms
        tmp = wmeanD - wmeanC

    else:
        raise ValueError(f"Unknown whichone={whichone} in choose_min_prof")

    return allmin, tmp


def choose_min_long(w_measured, w_model, time, divenum, updn, press, whichone):
    """
    Compute different minimization metrics for long time-series.

    Parameters
    ----------
    w_measured : array-like
        Measured vertical speeds [cm/s].
    w_model : array-like
        Modeled vertical speeds [cm/s].
    time : array-like
        Time array [seconds or days].
    divenum : array-like
        Dive number array.
    updn : array-like
        Up or down flag.
    press : array-like
        Pressure measurements [dbar].
    whichone : int
        Choice of metric.

    Returns
    -------
    allmin : dict
        Dictionary of metrics.
    """
    w_water = w_model - w_measured
    allmin = {}

    if whichone == 1:
        # Charlie's standard: RMS penalty
        w_penalty = 5
        w_s = np.sum(w_water**2)
        w_rms_v = np.sqrt(w_s / len(w_water))
        w_rms = np.sqrt(
            w_s + (len(w_water) - len(w_water)) * w_penalty**2 / len(w_water)
        )
        allmin["charlie"] = w_rms
        allmin["charlie_v"] = w_rms_v

    elif whichone == 3:
        # Full-depth minimizing dive-climb diff
        intwmc = []
        intwmd = []
        dives = np.unique(divenum)

        for dive in dives:
            idive = np.where((divenum == dive) & (updn == -1))[0]
            iclimb = np.where((divenum == dive) & (updn == 1))[0]

            if idive.size > 1:
                intwmd.append(np.mean(np.trapezoid(w_water[idive], time[idive])))
            if iclimb.size > 1:
                intwmc.append(np.mean(np.trapezoid(w_water[iclimb], time[iclimb])))

        N = min(len(intwmc), len(intwmd))
        intwmc = np.array(intwmc[:N])
        intwmd = np.array(intwmd[:N])

        w_rms = np.mean(np.abs(intwmc - intwmd))
        allmin["intdiff"] = w_rms

    else:
        raise ValueError(f"choose_min_long not implemented for whichone = {whichone}")

    return allmin


def ramsey_binavg(pressure, ww, zgrid, dall=None):
    """
    Perform bin-averaging of vertical speed offsets over pressure bins.

    Parameters
    ----------
    pressure : array-like
        Pressure measurements [dbar].
    ww : array-like
        Vertical speed difference (measured - modeled) [cm/s].
    zgrid : array-like
        Grid of pressure bin edges [dbar].
    dall : array-like, optional
        Dive number array to account for degrees of freedom (default None).

    Returns
    -------
    meanw : np.ndarray
        Mean vertical speed in each bin [cm/s].
    zbin : np.ndarray
        Midpoint pressure for each bin [dbar].
    NNz : np.ndarray
        Number of points in each bin.
    CIpm : np.ndarray, optional
        Confidence interval around mean (NaN if dall not provided).
    """

    nbin = len(zgrid) - 1
    meanw = np.full(nbin, np.nan)
    NNz = np.zeros(nbin, dtype=int)
    CIpm = np.full(nbin, np.nan)

    z1 = zgrid[:-1]
    z2 = zgrid[1:]

    p = 95  # Confidence interval percentage

    for zdo in range(nbin):
        # Find points within current pressure bin
        ifind = np.where((pressure > z1[zdo]) & (pressure <= z2[zdo]))[0]

        if len(ifind) > 0:
            meanw[zdo] = np.nanmean(ww[ifind])
            NNz[zdo] = len(ifind)

            if dall is not None:
                # Estimate confidence interval
                unique_dives = np.unique(dall[ifind])
                ndof = len(unique_dives) - 1

                if ndof > 0:
                    s_est = np.nanstd(ww[ifind] - meanw[zdo])
                    tval = cum_ttest(ndof, p)
                    CIpm[zdo] = tval * s_est / np.sqrt(ndof)

    zbin = (z1 + z2) / 2

    return meanw, zbin, NNz, CIpm


def cum_ttest(ndof, p):
    """
    Approximate cumulative Student's t-test value.

    Parameters
    ----------
    ndof : int
        Degrees of freedom.
    p : float
        Confidence level (e.g., 95).

    Returns
    -------
    tval : float
        t-distribution value corresponding to p and ndof.
    """
    from scipy.stats import t

    alpha = 1 - p / 100
    return t.ppf(1 - alpha / 2, ndof)


def ramsey_offset(w_measured, w_model, updn, pressure, zgrid):
    """
    Calculate Ramsey offset misfit between measured and modeled vertical speeds.

    Parameters
    ----------
    w_measured : array-like
        Measured vertical speeds [cm/s].
    w_model : array-like
        Modeled vertical speeds [cm/s].
    updn : array-like
        Up/down flag (positive = climb, negative = dive).
    pressure : array-like
        Pressure [dbar] corresponding to each measurement.
    zgrid : array-like
        Pressure bin edges for averaging [dbar].

    Returns
    -------
    allmin : dict
        Dictionary containing 'ramsey' WRMS value.
    tmp : array-like
        Difference between mean dive and mean climb profiles [cm/s].
    """

    # Find indices of climbing and diving parts
    iup = np.where(updn > 0)[0]
    idn = np.where(updn < 0)[0]

    # Difference between measured and modeled vertical speeds
    ww = w_measured - w_model

    # Ramsey bin averages for climb and dive separately
    meanclimb, _, _, _ = ramsey_binavg(pressure[iup], ww[iup], zgrid)
    meandive, _, _, _ = ramsey_binavg(pressure[idn], ww[idn], zgrid)

    # Weighted RMS (wrms) of dive and climb offsets
    wrms = np.nanmean(meandive**2 + meanclimb**2)

    allmin = {"ramsey": wrms}
    tmp = meandive - meanclimb

    return allmin, tmp


def reformat_units_var(ds, var_name, unit_format=unit_str_format):
    """
    Renames units in the dataset based on the provided dictionary for OG1.

    Parameters
    ----------
    ds (xarray.Dataset): The input dataset containing variables with units to be renamed.
    unit_format (dict): A dictionary mapping old unit strings to new formatted unit strings.

    Returns
    -------
    xarray.Dataset: The dataset with renamed units.
    """
    old_unit = ds[var_name].attrs["units"]
    if old_unit in unit_format:
        new_unit = unit_format[old_unit]
    else:
        new_unit = old_unit
    return new_unit


def convert_units_var(
    var_values, current_unit, new_unit, unit_conversion=unit_conversion
):
    """
    Convert the units of variables in an xarray Dataset to preferred units.  This is useful, for instance, to convert cm/s to m/s.

    Parameters
    ----------
    ds (xarray.Dataset): The dataset containing variables to convert.
    preferred_units (list): A list of strings representing the preferred units.
    unit_conversion (dict): A dictionary mapping current units to conversion information.
    Each key is a unit string, and each value is a dictionary with:
        - 'factor': The factor to multiply the variable by to convert it.
        - 'units_name': The new unit name after conversion.

    Returns
    -------
    xarray.Dataset: The dataset with converted units.
    """
    if (
        current_unit in unit_conversion
        and new_unit in unit_conversion[current_unit]["units_name"]
    ):
        conversion_factor = unit_conversion[current_unit]["factor"]
        new_values = var_values * conversion_factor
    else:
        new_values = var_values
        print(f"No conversion information found for {current_unit} to {new_unit}")
    #        raise ValueError(f"No conversion information found for {current_unit} to {new_unit}")
    return new_values
