# =============================================================================
# utils.py — Helper functions for GPCA-ENSO/NDVI analysis
# =============================================================================

import os
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from tqdm import tqdm
from pyproj import Transformer
from scipy.stats import t as student_t
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score


# =============================================================================
# Köppen-Geiger classification lookup (Beck 2018)
# =============================================================================

KOPPEN_NAMES = {
    1: "Af", 2: "Am", 3: "Aw",
    4: "BWh", 5: "BWk", 6: "BSh", 7: "BSk",
    8: "Csa", 9: "Csb", 10: "Csc",
    11: "Cwa", 12: "Cwb", 13: "Cwc",
    14: "Cfa", 15: "Cfb", 16: "Cfc",
    17: "Dsa", 18: "Dsb", 19: "Dsc", 20: "Dsd",
    21: "Dwa", 22: "Dwb", 23: "Dwc", 24: "Dwd",
    25: "Dfa", 26: "Dfb", 27: "Dfc", 28: "Dfd",
    29: "ET", 30: "EF",
}


# =============================================================================
# Data utilities
# =============================================================================

def generate_past_data(X, maxlag=1, inflag=0):
    """
    Generate lagged predictor matrix from a time series.

    Parameters
    ----------
    X : np.ndarray, shape (T,) or (T, d)
    maxlag : int
    inflag : int  — initial lag offset

    Returns
    -------
    X_past    : np.ndarray, shape (T - maxlag, maxlag * d)
    X_present : np.ndarray, shape (T - maxlag, d)
    """
    if X.ndim == 1:
        X = X[:, None]
    X_past = np.array([
        X[i + inflag:i + maxlag, :].flatten()
        for i in range(len(X) - maxlag)
    ])
    return X_past, X[maxlag:]


def compute_aic_bic(X, y, model):
    """Return AIC and BIC for a fitted linear regression model."""
    model.fit(X, y)
    n, k = len(y), X.shape[1]
    sse = np.sum((y - model.predict(X)) ** 2)
    aic = n * np.log(sse / n) + 2 * k
    bic = n * np.log(sse / n) + k * np.log(n)
    return aic, bic


def explained_variance_ratio_fast(X, w, center=True):
    """
    Fraction of total variance in X captured by projection onto w.

    Parameters
    ----------
    X : np.ndarray, shape (T, p)
    w : np.ndarray, shape (p,)

    Returns
    -------
    float
    """
    X = np.asarray(X)
    w = np.asarray(w)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    w = w / np.linalg.norm(w)
    proj = X @ w
    var_w = (proj @ proj) / X.shape[0]
    total_var = np.sum(X * X) / X.shape[0]
    return var_w / total_var


def get_weights(anomalies_pca, anomalies, algo, idx=0, mode="correlation"):
    """
    Compute spatial weight map from a fitted teleconnection algorithm.

    Parameters
    ----------
    anomalies_pca : np.ndarray, shape (T, n_pca)
    anomalies     : np.ndarray, shape (T, n_pixels)
    algo          : fitted model with `.transform()` method
    idx           : int — component index
    mode          : "correlation" | "regression"

    Returns
    -------
    correlation mode → (corr, significance)  both shape (n_pixels,)
    regression  mode → coef                  shape (n_pixels,)
    """
    cp = algo.transform(anomalies_pca)
    if cp.ndim > 1 and cp.shape[1] > 1:
        cp = cp[:, idx]

    if mode == "regression":
        lr = LinearRegression()
        lr.fit(anomalies, cp)
        return lr.coef_

    elif mode == "correlation":
        cp = np.asarray(cp).ravel()
        n = cp.size
        cp_c = cp - cp.mean()
        anom_c = anomalies - anomalies.mean(axis=0)
        cp_norm = np.sqrt(np.sum(cp_c ** 2))
        anom_norm = np.sqrt(np.sum(anom_c ** 2, axis=0))
        cov = np.sum(anom_c * cp_c[:, None], axis=0)
        corr = cov / (cp_norm * anom_norm)
        t_stat = corr * np.sqrt((n - 2) / (1 - corr ** 2))
        p_values = 2 * (1 - student_t.cdf(np.abs(t_stat), n - 2))
        significance = p_values < 0.05
        return corr, significance

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'regression' or 'correlation'.")


# =============================================================================
# Lag analysis utilities
# =============================================================================

def lag_analysis(X, Y, n_splits=10, maxlags=range(1, 21),
                 model=None):
    """
    Cross-validated predictive skill and information criteria vs lag.

    Evaluates three configurations:
      - past Y → Y
      - past X → Y
      - past (X, Y) → Y
    """
    if model is None:
        model = LinearRegression()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    for lag in tqdm(maxlags):
        Y_past, Y_present = generate_past_data(Y, lag)
        X_past, _ = generate_past_data(X, lag)
        combined = np.hstack([Y_past, X_past])

        configs = {
            "Y past → Y":  Y_past,
            "X past → Y":  X_past,
            "XY past → Y": combined,
        }
        for name, X_data in configs.items():
            r2_scores = -cross_val_score(model, X_data, Y_present,
                                         cv=kf, scoring="r2")
            aic, bic = compute_aic_bic(X_data, Y_present, model)
            results.append({
                "lag": lag, "config": name,
                "r2_median": np.median(r2_scores),
                "r2_q25": np.percentile(r2_scores, 25),
                "r2_q75": np.percentile(r2_scores, 75),
                "AIC": aic, "BIC": bic,
            })
    return pd.DataFrame(results)


def find_optimal_mu_idx(scores):
    """
    Select the μ index that maximises EVR among admissible points
    (F_stat > F_crit).  Falls back to argmax(F_stat) if none is admissible.

    Parameters
    ----------
    scores : list of (mu, EVR, F_stat, F_crit) tuples

    Returns
    -------
    int — row index into `scores`
    """
    df = pd.DataFrame(scores, columns=["mu", "EVR", "F_stat", "F_crit"])
    admissible = df[df["F_stat"] > df["F_crit"]]
    if len(admissible) > 0:
        mu_star = admissible.iloc[admissible["EVR"].argmax()]["mu"]
    else:
        mu_star = df.iloc[df["F_stat"].argmax()]["mu"]
    return df[df["mu"] == mu_star].index[0]


# =============================================================================
# Geospatial utilities
# =============================================================================

def build_lonlat_grid(easting, northing):
    """
    Convert EPSG:6933 easting/northing vectors to WGS84 lon/lat 2-D grids.

    Returns
    -------
    lon2d, lat2d : np.ndarray, both shape (len(northing), len(easting))
    """
    transformer = Transformer.from_crs("EPSG:6933", "EPSG:4326", always_xy=True)
    e2d, n2d = np.meshgrid(easting, northing, indexing="ij")
    lon2d, lat2d = transformer.transform(e2d, n2d)
    return lon2d.T, lat2d.T


def reconstruct_2d_field(flat_values, na_mask, fill=np.nan):
    """
    Unpack a flat pixel vector back to the 2-D spatial domain defined by na_mask.

    Parameters
    ----------
    flat_values : np.ndarray, shape (n_valid_pixels,)
    na_mask     : np.ndarray, shape (H, W)  — 1 where valid, 0 elsewhere
    fill        : value to assign to masked pixels

    Returns
    -------
    field2d : np.ndarray, shape (H, W)
    """
    field = np.full_like(na_mask, fill, dtype=float).T
    field[na_mask.T == 1] = flat_values
    return field.T


def sample_koppen_classes(lons, lats, climate_tif):
    """
    Look up the Köppen-Geiger class at each (lon, lat) point.

    Parameters
    ----------
    lons, lats   : 1-D arrays of coordinates (WGS84)
    climate_tif  : path to the raster (e.g. koppen_geiger_0p1.tif)

    Returns
    -------
    climate_classes : np.ndarray of floats (NaN where out-of-bounds)
    valid_mask      : boolean array
    """
    with rasterio.open(climate_tif) as src:
        climate = src.read(1).astype(float)
        transform = src.transform
        nodata = src.nodata
    if nodata is not None:
        climate[climate == nodata] = np.nan

    rows, cols = rasterio.transform.rowcol(transform, lons, lats)
    rows, cols = np.asarray(rows), np.asarray(cols)
    in_bounds = (
        (rows >= 0) & (rows < climate.shape[0]) &
        (cols >= 0) & (cols < climate.shape[1])
    )
    classes = np.full(len(lons), np.nan)
    classes[in_bounds] = climate[rows[in_bounds], cols[in_bounds]]
    valid = in_bounds & np.isfinite(classes)
    return classes, valid


def build_koppen_dataframe(corr_2d, sig_2d, lon2d, lat2d, climate_tif):
    """
    Build a tidy DataFrame with correlation values, climate zone, and hemisphere
    restricted to *significant* pixels.

    Returns
    -------
    pd.DataFrame with columns: correlation, latitude, climate_code, climate, hemisphere
    """
    valid_mask = np.isfinite(corr_2d) & sig_2d
    lons = lon2d[valid_mask]
    lats = lat2d[valid_mask]
    corr_vals = corr_2d[valid_mask]

    classes, valid = sample_koppen_classes(lons, lats, climate_tif)
    df = pd.DataFrame({
        "correlation": corr_vals[valid],
        "latitude": lats[valid],
        "climate_code": classes[valid].astype(int),
    })
    df["climate"] = df["climate_code"].map(
        lambda c: KOPPEN_NAMES.get(c, f"Unknown_{c}")
    )
    df["hemisphere"] = np.where(df["latitude"] >= 0, "North", "South")
    return df


# =============================================================================
# Results summary
# =============================================================================

def build_results_table(method_dict, gpca_maxlag):
    """
    Assemble a publication-ready summary DataFrame.

    Parameters
    ----------
    method_dict : dict  {name: (results_df, evr_scalar)}
    gpca_maxlag : int

    Returns
    -------
    pd.DataFrame
    """
    frames = []
    for method, (res_df, evr) in method_dict.items():
        row = res_df[res_df["lag"] == gpca_maxlag].copy()
        row = row.assign(EVR=evr, Method=method)
        frames.append(row)
    combined = pd.concat(frames).reset_index(drop=True)
    cols = ["Method", "EVR", "F_statistic", "p_value"]
    return combined[[c for c in cols if c in combined.columns]]
