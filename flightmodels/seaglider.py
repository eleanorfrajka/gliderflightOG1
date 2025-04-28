import numpy as np
import scipy.sparse as sp
import xarray as xr
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.sparse.linalg import splu

from flightmodels import tools, utilities

PROFDIFF = None


def flightvec_ds(ds: xr.Dataset, xl: float, hd_a: float, hd_b: float, hd_c: float) -> xr.Dataset:
    """
    Run flightvec on an OG1 xarray Dataset.
    """
    umag, thdeg = flightvec(ds['buoyancy'].values, ds['pitch'].values, xl, hd_a, hd_b, hd_c, ds['rho0'].values)
    ds = ds.assign(umag=(('time',), umag), thdeg=(('time',), thdeg))
    return ds


def regress_all_vec(
    whichpar: list,
    glider: xr.Dataset,
    whichone: int,
    ensmat: list,
    plotflag: bool = True,
    unstdyflag: int = 0,
) -> tuple[np.ndarray, float]:
    """
    Solve for glider flight model parameters via minimization.

    Parameters
    ----------
    whichpar : list of int
        List indicating which parameters to optimize (1 = optimize, 0 = hold fixed).
        Order corresponds to [hd_a, hd_b, vbdbias, abs_compress, therm_expan, hd_c].
    glider : xarray.Dataset
        Glider dataset containing:
        - Attributes: 'hd_a', 'hd_b', 'vbdbias', 'abs_compress', 'therm_expan', 'hd_c' (floats).
        - Data variables (dimensioned along 'TIME'): 'PRES', 'TEMP',
          'PSAL', 'PITCH', 'GLIDER_SPEED', 'TIME', 'VERTICAL_SPEED', 'DIVENUM', 'UPDN'.
    whichone : int
        Selector for which misfit function to minimize (e.g., 10 for Ramsey bin averaging).
    ensmat : list or array-like
        List of dive numbers to use for the ensemble selection.
    plotflag : bool, optional
        Whether to generate plots (default is True).
    unstdyflag : int, optional
        Flag for unsteady flight adjustment (default is 0, meaning steady flight).

    Returns
    -------
    regressout : np.ndarray
        Optimized parameter values.
    allwrms : float
        Final minimized value of the misfit function.
    """
    print("Doing regress_all_vec")

    # Extract initial glider flight parameters from attributes
    hd_a = glider.attrs['hd_a']
    hd_b = glider.attrs['hd_b']
    vbdbias = glider.attrs['vbdbias']
    abs_compress = glider.attrs['abs_compress']
    therm_expan = glider.attrs['therm_expan']
    hd_c = glider.attrs['hd_c']

    # Create initial guess vector (scaling to match original code)
    x_0 = np.array([
        hd_a * 1e3,
        hd_b * 1e3,
        vbdbias,
        abs_compress * 1e6,
        therm_expan * 1e5,
        hd_c * 1e5
    ])

    # Find matching indices for selected dives
    is_selected = np.isin(glider['DIVENUM'].values, ensmat)
    glider0 = glider.sel(N_MEASUREMENTS=is_selected)

    # Build initial guess subset according to whichpar
    x_1 = x_0[np.where(np.array(whichpar) == 1)]

    # Perform minimization
    result = minimize(
        f_misfit_all,
        x_1,
        args=(whichpar, glider0, whichone, unstdyflag),
        method='Nelder-Mead',
        options={
            'disp': True,
            'xatol': 0.1,
            'fatol': 0.01,
            'maxfev': 250,
        }
    )

    regressout = result.x
    allwrms = result.fun

    return regressout, allwrms



# Subfunction: Acceleration equations for glide
def glide_acc(t, V, t_grid, buoy, pitch, xl, hd_a, hd_b, hd_c, rho0, enclosed_mass):
    """
    Right-hand side of the unsteady flight ODE system.

    Parameters
    ----------
    t : float
        Current time.
    V : array-like
        Current velocity components [Vx, Vz].
    (Other parameters describe glider and environmental conditions.)

    Returns
    -------
    dV_dt : list
        Derivatives [dVx/dt, dVz/dt].
    """
    idx = np.searchsorted(t_grid, t, side="right") - 1
    idx = np.clip(idx, 0, len(t_grid) - 1)

    buoyancy = buoy[idx]
    pitch_angle = pitch[idx] * np.pi / 180

    umag = np.sqrt(V[0]**2 + V[1]**2)  # speed magnitude
    q = 0.5 * rho0 * umag**2  # dynamic pressure

    # Aerodynamic forces
    lift = hd_a * q * np.sin(2 * pitch_angle)
    drag = hd_b * q
    added_mass = hd_c * rho0 * xl**3

    # Force components
    Fx = -drag * np.cos(pitch_angle) + lift * np.sin(pitch_angle)
    Fz = buoyancy - (drag * np.sin(pitch_angle) + lift * np.cos(pitch_angle))

    dVx_dt = Fx / (enclosed_mass + added_mass)
    dVz_dt = Fz / (enclosed_mass + added_mass)

    return [dVx_dt, dVz_dt]

# Main function
def flightvec_unstdy(time, buoy, pitch, xl, hd_a, hd_b, hd_c, rho0, tau0=20, odeFLAG=1):
    """
    Solve for glider speed and glide angle considering unsteady effects.

    Parameters
    ----------
    time : array-like
        Time series [seconds].
    buoy : array-like
        Buoyancy force [arbitrary units].
    pitch : array-like
        Pitch angle [degrees].
    xl : float
        Reference glider length [meters].
    hd_a, hd_b, hd_c : float
        Hydrodynamic coefficients for lift, drag, and added mass.
    rho0 : float
        Reference density of seawater [kg/m^3].
    tau0 : float, optional
        Lag time constant [seconds], default 20s.
    odeFLAG : int, optional
        Method flag: 1 = solve ODE system, <2 = lag model from steady flight.

    Returns
    -------
    spd : dict
        Various computed speeds (steady, unsteady-ODE, unsteady-lag).
    ang : dict
        Various computed glide angles (steady, unsteady-ODE, unsteady-lag).
    """
    mytol = 1.2

    gravity = 9.82  # m/s^2, acceleration due to gravity
    enclosed_mass = rho0 * 60e-3  # kg, volume of water enclosed in hull (60 liters)

    time = np.asarray(time)
    buoy = np.asarray(buoy)
    pitch = np.asarray(pitch)

    mp = len(time)
    spd = {}
    ang = {}

    if odeFLAG >= 1:
        # Solve unsteady glide ODE system
        sol = solve_ivp(
            glide_acc,
            (time[0], time[-1]),
            [0, 0],  # Initial horizontal and vertical velocities
            t_eval=time,
            args=(time, buoy, pitch, xl, hd_a, hd_b, hd_c, rho0, enclosed_mass),
            method='RK23',
            rtol=mytol * 1e-3,
            atol=mytol * 1e-5,
        )

        V = sol.y.T  # Velocities over time
        spd_unstdy = np.sqrt(V[:, 0]**2 + V[:, 1]**2)
        glideangle_unstdy = np.degrees(np.arctan2(V[:, 1], V[:, 0]))

        # Invalidate unrealistic results
        invalid = np.logical_or(np.iscomplex(spd_unstdy), spd_unstdy > 100)
        spd_unstdy[invalid] = 0
        glideangle_unstdy[invalid] = 0

        spd_unstdy *= 100  # Convert to cm/s

        w_unstdy = spd_unstdy * np.sin(np.radians(glideangle_unstdy))

        # Save unsteady ODE outputs
        spd['unstdy_ode'] = spd_unstdy
        spd['w_unstdy_ode'] = w_unstdy
        ang['unstdy_ode'] = glideangle_unstdy

    if odeFLAG < 2:
        # Otherwise, fallback to simple lag model based on steady flight
        from .seaglider import flightvec  # assumed your flightvec0 is already translated

        spd_stdy, glideangle_stdy = flightvec(buoy, pitch, xl, hd_a, hd_b, hd_c, rho0)
        spd_stdy = np.asarray(spd_stdy)
        glideangle_stdy = np.asarray(glideangle_stdy)

        invalid = np.logical_or(np.iscomplex(spd_stdy), spd_stdy > 100)
        spd_stdy[invalid] = 0
        glideangle_stdy[invalid] = 0

        hspd_stdy = spd_stdy * np.cos(np.radians(glideangle_stdy))
        w_stdy = spd_stdy * np.sin(np.radians(glideangle_stdy))

        spd['stdy'] = spd_stdy
        spd['h_stdy'] = hspd_stdy
        spd['w_stdy'] = w_stdy
        ang['stdy'] = glideangle_stdy

        # Lag model
        tau_i = tau0
        coef_d = np.zeros(mp)
        coef_d[0] = tau_i / (time[1] - time[0])
        coef_d[1:-1] = tau_i / (time[2:] - time[:-2])
        coef_d[-1] = tau_i / (time[-1] - time[-2])

        # Build sparse matrix GI
        GI = (
            sp.diags([-coef_d[1:], 1 + np.zeros(mp), coef_d[:-1]], [-1, 0, 1], shape=(mp, mp))
            .tocsc()
        )
        solver = splu(GI)

        hspd_unstdy = solver.solve(hspd_stdy)
        w_unstdy = solver.solve(w_stdy)

        spd_unstdy = np.sqrt(hspd_unstdy**2 + w_unstdy**2)
        glideangle_unstdy = np.degrees(np.arctan2(w_unstdy, hspd_unstdy))

        # Save lagged flight results
        spd['unstdy_lag'] = spd_unstdy
        spd['w_unstdy_lag'] = w_unstdy
        spd['h_unstdy_lag'] = hspd_unstdy
        ang['unstdy_lag'] = glideangle_unstdy

    return spd, ang

def flightvec(
    buoy: np.ndarray,
    pitch: np.ndarray,
    xl: float = 1.8,
    hd_a: float = 0.004,
    hd_b: float = 0.01,
    hd_c: float = 5.7e-05,
    rho0: float = 1027.5,
    tol: float = 0.001,
    max_iter: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve unaccelerated flight equations iteratively for steady-state speed and glide angle.

    Parameters
    ----------
    buoy : np.ndarray
        Buoyancy (units: grams).
    pitch : np.ndarray
        Pitch angle (degrees).
    xl : float
        Characteristic length scale (meters).
    hd_a : float
        Hydrodynamic coefficient a.
    hd_b : float
        Hydrodynamic coefficient b.
    hd_c : float
        Hydrodynamic coefficient c.
    rho0 : float
        Water density (kg/m³).
    tol : float, optional
        Tolerance for iteration convergence. Default is 0.001.
    max_iter : int, optional
        Maximum number of iterations. Default is 15.

    Returns
    -------
    umag : np.ndarray
        Steady-state speed magnitude (cm/s).
    thdeg : np.ndarray
        Glide angle (degrees).
    """
    # Constants
    gravity = 9.82  # gravitational acceleration (m/s²)
    # Initial estimate for glide angle (radians):
    # +45° for positive buoyancy, -45° for negative buoyancy
    th = (np.pi / 4) * np.sign(buoy)
    # Convert buoyancy into a force (Newtons)
    buoyforce = 0.001 * gravity * buoy

    # Estimate dynamic pressure (q) assuming vertical motion
    q = (np.sign(buoy) * buoyforce / (xl * xl * hd_b)) ** (4 / 3)
    # Initialize attack angle and parameter arrays
    alpha = np.zeros_like(buoy)

    # Initialize arrays for previous q (for convergence checking) and parameter
    q_old = np.zeros_like(buoy)
    param = np.ones_like(buoy)

    # Initialize outputs
    umag = np.zeros_like(buoy)    # steady speed (cm/s)
    thdeg = np.zeros_like(buoy)   # glide angle (degrees)

    # Mask of valid entries: buoyancy ≠ 0 and pitch matches sign of buoyancy
    valid = (buoy != 0) & (np.sign(buoy) * np.sign(pitch) > 0)

    j = 0  # Iteration counter
    while np.any(np.abs((q[valid] - q_old[valid]) / q[valid]) > tol) and j <= max_iter:
        # Save current q to check for convergence
        q_old = q.copy()

        # Calculate inverse of aerodynamic parameter
        param_inv = hd_a * hd_a * np.tan(th) ** 2 * q ** 0.25 / (4 * hd_b * hd_c)

        # Update valid points (param_inv must be > 1 and pitch sign must match buoyancy)
        valid = (param_inv > 1) & (np.sign(buoy) * np.sign(pitch) > 0)

        # Calculate flight parameters for valid entries
        param[valid] = 4 * hd_b * hd_c / (hd_a * hd_a * np.tan(th[valid]) ** 2 * q[valid] ** 0.25)

        # Update dynamic pressure q for valid entries
        q[valid] = (buoyforce[valid] * np.sin(th[valid]) / (2 * xl * xl * hd_b * q[valid] ** -0.25)) * (
            1 + np.sqrt(1 - param[valid])
        )

        # Calculate attack angle alpha
        alpha[valid] = (-hd_a * np.tan(th[valid]) / (2 * hd_c)) * (1 - np.sqrt(1 - param[valid]))

        # Update glide angle thdeg (degrees) if valid points exist
        if valid.any():
            thdeg[valid] = pitch[valid] - alpha[valid]
        else:
            thdeg[valid] = np.nan

        # Handle stalled solutions:
        # Stall occurs when param_inv <= 1 or pitch is opposite buoyancy
        stall = (param_inv <= 1) | (np.sign(buoy) * np.sign(pitch) < 0)

        # For stalled cases, set q and thdeg to 0
        q[stall] = 0.0
        thdeg[stall] = 0.0

        # Update the glide angle (in radians) for next iteration
        th = np.deg2rad(thdeg)

        j += 1  # Increment iteration counter

        # Compute steady-state speed from dynamic pressure
        umag = 100 * np.sqrt(2 * q / rho0)  # output in cm/s

    return umag, thdeg




def f_misfit_all(x, whichpar, glider: xr.Dataset, whichone: int, unstdyflag: int):
    """
    Misfit function for glider flight model parameter optimization.

    Parameters
    ----------
    x : array-like
        Current set of free parameters being optimized.
    whichpar : list of int
        Which parameters are being optimized (1 = optimize, 0 = hold fixed).
    glider : xarray.Dataset
        Subselected glider data.
    whichone : int
        Which minimization target to use (e.g., 10 for Ramsey bin averaging).
    unstdyflag : int
        Whether to use steady (0) or unsteady (1) flight model.

    Returns
    -------
    wrms : float
        Weighted root mean square (or other diagnostic) to be minimized.
    """
    global PROFDIFF

    # Start with original glider parameters
    hd_a = glider.attrs['hd_a']
    hd_b = glider.attrs['hd_b']
    vbdbias = glider.attrs['vbdbias']
    abs_compress = glider.attrs['abs_compress']
    therm_expan = glider.attrs['therm_expan']
    hd_c = glider.attrs['hd_c']


    # Apply updated values from optimization vector
    x_full = np.zeros(6)
    qdo = 0
    for wdo in range(6):
        if whichpar[wdo] == 1:
            x_full[wdo] = x[qdo]
            qdo += 1

    if x_full[0] != 0:
        hd_a = x_full[0] / 1e3
    if x_full[1] != 0:
        hd_b = x_full[1] / 1e3
    if x_full[2] != 0:
        vbdbias = x_full[2]
    if x_full[3] != 0:
        abs_compress = x_full[3] / 1e6
    if x_full[4] != 0:
        therm_expan = x_full[4] / 1e5
    if x_full[5] != 0:
        hd_c = x_full[5] / 1e5

    # Pull out data arrays
    vbd = glider['VBD'].values
    c_vbd = glider['C_VBD'].values
    press = glider['PRES'].values
    temp = glider['TEMP'].values
    salin = glider['PSAL'].values
    pitch = glider['PITCH'].values
    speed = glider['GLIDE_SPEED'].values
    time = glider['TIME'].values
    w_measured = glider['GLIDER_VERT_VELO_DZDT'].values
    divenum = glider['DIVENUM'].values
    updn = glider['UPDN'].values
    lat = glider['LATITUDE'].values.mean()
    lon = glider['LONGITUDE'].values.mean()

    # Constants
    vbd_min_cnts = glider.attrs['vbd_min_cnts']
    vbd_cnts_per_cc = glider.attrs['vbd_cnts_per_cc']
    temp_ref = glider.attrs['temp_ref']
    volmax = glider.attrs['volmax']
    mass = glider.attrs['mass']
    rho0 = glider.attrs['rho0']

    # Recalculate volume and buoyancy
    vol1 = vbd + volmax + (c_vbd - vbd_min_cnts) / vbd_cnts_per_cc
    compr_factor = np.exp(-abs_compress * press + therm_expan * (temp - temp_ref))
    density_insitu = tools.compute_insitu_density(salin, temp, press, lat, lon)

    vbdc = vbd - vbdbias
    vol = (vol1 - vbdbias) * compr_factor
    buoy = 1000 * (-mass + density_insitu * vol * 1e-6)

    # Identify valid steady gliding conditions
    valid = np.where((vbdc * pitch > 0) & (speed > 0))[0]

    if unstdyflag == 0:
        spd_stdy, glideangle_stdy = flightvec(buoy[valid], pitch[valid], 1.8, hd_a, hd_b, hd_c, rho0)
        w_model = spd_stdy * np.sin(np.radians(glideangle_stdy))
    else:
        tau0 = 12
        spd = flightvec_unstdy(time[valid], buoy[valid], pitch[valid], 1.8, hd_a, hd_b, hd_c, rho0, tau0, 0)
        w_model = spd['w_unstdy_lag']

    # Remove invalid entries
    valid2 = np.isreal(w_model)
    w_model = w_model[valid2]
    w_measured = w_measured[valid][valid2]
    time = time[valid][valid2]
    divenum = divenum[valid][valid2]
    updn = updn[valid][valid2]
    press = press[valid][valid2]

    # Minimize appropriate quantity
    if whichone < 5:
        allmin = tools.choose_min_long(w_measured, w_model, time, divenum, updn, whichone)
    elif whichone < 10:
        if 'PGRID' in glider:
            pgrid = glider['PGRID'].values
        else:
            pgrid = np.arange(0, 1000, 10)
        wg, wspdg, timeg, divenumg, updng = tools.gridthem(w_measured, w_model, time, divenum, updn, press, pgrid)
        allmin, profdiff = tools.choose_min_prof(wg, wspdg, timeg, divenumg, updng, pgrid, whichone)
        PROFDIFF = profdiff
    else:
        if 'PGRID' in glider:
            pgrid = glider['PGRID'].values
        else:
            pgrid = np.arange(0, 1000, 10)
        allmin, profdiff = tools.ramsey_offset(w_measured, w_model, updn, press, pgrid)
        PROFDIFF = profdiff

    # Extract wrms based on whichone
    if whichone == 1:
        wrms = allmin['charlie']
    elif whichone == 2:
        wrms = allmin['meanbl']
    elif whichone == 3:
        wrms = allmin['intdiff']
    elif whichone == 4:
        wrms = allmin['intdiffbl'] + allmin['intdiffbl2'] / 5000
    elif whichone == 5:
        wrms = allmin['mlprofdiff']
    elif whichone == 6:
        wrms = allmin['flprofdiff']
    elif whichone == 7:
        wrms = allmin['meanw']
    elif whichone == 8:
        wrms = allmin['meanwbl']
    elif whichone == 10:
        wrms = allmin['ramsey']
    else:
        raise ValueError(f"Unknown minimization mode: {whichone}")

    oneline = [wrms, hd_a, hd_b, vbdbias, abs_compress, therm_expan, hd_c]
    print(" ".join(f"{v:.4g}" for v in oneline))

    return wrms


