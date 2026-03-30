# ======================================================================
# Imports
# ======================================================================

import numpy as np
import xarray as xr


# ======================================================================
# Xarray ↔ NumPy helpers
# ======================================================================

def xr2np(da: xr.DataArray, mask: xr.DataArray, var='tp'):
    """
    Convert an xarray DataArray to a 2D NumPy array (time × land points).

    Parameters
    ----------
    da : xr.DataArray
        Input dataset containing the variable of interest
    mask : xr.DataArray
        Land–sea mask (non-null values indicate land)
    var : str
        Variable name inside `da`

    Returns
    -------
    np.ndarray
        Array of shape (time, n_land_points)
    """

    # Boolean land mask (True = land, False = ocean)
    land_mask = mask.notnull().values.ravel()

    # Extract variable and flatten spatial dimensions
    anom = da[var]
    anomalies_all = anom.values.reshape(anom.shape[0], -1)

    # Select land pixels only
    anomalies = anomalies_all[:, land_mask]

    return anomalies


def np2D_to_np3D(array: np.ndarray, mask: xr.DataArray):
    """
    Reconstruct a 3D (time, lat, lon) array from a 2D land-only array.

    Parameters
    ----------
    array : np.ndarray
        Array of shape (time, n_land_points)
    mask : xr.DataArray
        Land–sea mask defining the original spatial grid

    Returns
    -------
    np.ndarray
        Array of shape (time, lat, lon) with NaNs over ocean
    """

    # Boolean land mask (flattened)
    land_mask = mask.notnull().values.ravel()

    # Allocate full array with NaNs
    full = np.full((array.shape[0], land_mask.size), np.nan)

    # Insert land values
    full[:, land_mask] = array

    # Reshape back to (time, lat, lon)
    reconstructed = full.reshape(
        array.shape[0],
        mask.values.shape[0],
        mask.values.shape[1]
    )

    return reconstructed
