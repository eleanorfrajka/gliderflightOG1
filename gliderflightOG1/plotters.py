from pandas import DataFrame
import matplotlib.pyplot as plt
import xarray as xr


def plot_model_comparisons(
    glider: xr.Dataset,
    models: dict,
    x_axis: str = "TIME",
    select_dives: list = None,
    select_timerange: tuple = None,
    variable: str = "VERTICAL_SPEED",
    label_variable: str = "Measured Vertical Speed",
    model_labels: list = None,
    title: str = "",
):
    """Plot measured vs modeled glider speeds for multiple models.

    Parameters
    ----------
    glider : xarray.Dataset
        Dataset containing glider measurements.
    models : dict
        Dictionary of model outputs, e.g., {'steady': speed_array, 'unsteady': speed_array}.
    x_axis : str, optional
        Which x-axis to use ('TIME' or 'DIVENUM').
    select_dives : list of int, optional
        List of dive numbers to plot.
    select_timerange : tuple of floats, optional
        (start_time, end_time) in same units as glider['TIME'].
    variable : str, optional
        Measured variable name (default 'VERTICAL_SPEED').
    label_variable : str, optional
        Label for measured variable.
    model_labels : list of str, optional
        Labels for model results; if None, keys of models dict are used.
    title : str, optional
        Plot title.

    """
    # Select subset
    if select_dives is not None:
        selection = glider.where(glider["DIVENUM"].isin(select_dives), drop=True)
    elif select_timerange is not None:
        start, end = select_timerange
        selection = glider.where(
            (glider["TIME"] >= start) & (glider["TIME"] <= end), drop=True
        )
    else:
        selection = glider

    x = selection[x_axis].values
    y_measured = selection[variable].values

    plt.figure(figsize=(10, 6))
    plt.plot(
        x, y_measured, label=label_variable, linestyle="-", marker="o", markersize=3
    )

    if model_labels is None:
        model_labels = list(models.keys())

    for idx, (model_name, model_data) in enumerate(models.items()):
        plt.plot(x, model_data, label=model_labels[idx])

    plt.xlabel(x_axis)
    plt.ylabel("Vertical Speed (cm/s)")
    plt.legend()
    plt.grid(True)
    plt.title(title or "Measured vs Modeled Glider Speeds")
    plt.tight_layout()
    plt.show()


def show_variables(data):
    """Processes an xarray Dataset or a netCDF file, extracts variable information,
    and returns a styled DataFrame with details about the variables.

    Parameters
    ----------
    data (str or xr.Dataset): The input data, either a file path to a netCDF file or an xarray Dataset.

    Returns
    -------
    pandas.io.formats.style.Styler: A styled DataFrame containing the following columns:
        - dims: The dimension of the variable (or "string" if it is a string type).
        - name: The name of the variable.
        - units: The units of the variable (if available).
        - comment: Any additional comments about the variable (if available).

    """
    if isinstance(data, str):
        print("information is based on file: {}".format(data))
        dataset = xr.Dataset(data)
        variables = dataset.variables
    elif isinstance(data, xr.Dataset):
        print("information is based on xarray Dataset")
        variables = data.variables
    else:
        raise TypeError("Input data must be a file path (str) or an xarray Dataset")

    info = {}
    for i, key in enumerate(variables):
        var = variables[key]
        if isinstance(data, str):
            dims = var.dimensions[0] if len(var.dimensions) == 1 else "string"
            units = "" if not hasattr(var, "units") else var.units
            comment = "" if not hasattr(var, "comment") else var.comment
        else:
            dims = var.dims[0] if len(var.dims) == 1 else "string"
            units = var.attrs.get("units", "")
            comment = var.attrs.get("comment", "")

        info[i] = {
            "name": key,
            "dims": dims,
            "units": units,
            "comment": comment,
            "standard_name": var.attrs.get("standard_name", ""),
            "dtype": str(var.dtype) if isinstance(data, str) else str(var.data.dtype),
        }

    vars = DataFrame(info).T

    dim = vars.dims
    dim[dim.str.startswith("str")] = "string"
    vars["dims"] = dim

    vars = (
        vars.sort_values(["dims", "name"])
        .reset_index(drop=True)
        .loc[:, ["dims", "name", "units", "comment", "standard_name", "dtype"]]
        .set_index("name")
        .style
    )

    return vars


def show_attributes(data):
    """Processes an xarray Dataset or a netCDF file, extracts attribute information,
    and returns a DataFrame with details about the attributes.

    Parameters
    ----------
    data (str or xr.Dataset): The input data, either a file path to a netCDF file or an xarray Dataset.

    Returns
    -------
    pandas.DataFrame: A DataFrame containing the following columns:
        - Attribute: The name of the attribute.
        - Value: The value of the attribute.

    """
    from netCDF4 import Dataset

    if isinstance(data, str):
        print("information is based on file: {}".format(data))
        rootgrp = Dataset(data, "r", format="NETCDF4")
        attributes = rootgrp.ncattrs()
        get_attr = lambda key: getattr(rootgrp, key)
    elif isinstance(data, xr.Dataset):
        print("information is based on xarray Dataset")
        attributes = data.attrs.keys()
        get_attr = lambda key: data.attrs[key]
    else:
        raise TypeError("Input data must be a file path (str) or an xarray Dataset")

    info = {}
    for i, key in enumerate(attributes):
        dtype = type(get_attr(key)).__name__
        info[i] = {"Attribute": key, "Value": get_attr(key), "DType": dtype}

    attrs = DataFrame(info).T

    return attrs
