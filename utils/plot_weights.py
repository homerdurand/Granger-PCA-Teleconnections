# ======================================================================
# Imports
# ======================================================================

import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from pyproj import Transformer
from shapely.geometry import Point
from shapely.prepared import prep
import shapely.ops as ops


# ======================================================================
# Custom colormap normalization
# ======================================================================

class MidpointNormalize(mcolors.Normalize):
    """
    Normalize a colormap so that a specified midpoint maps to 0.5.

    Useful for diverging colormaps centered on zero.
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)

        vmin = self.vmin
        vmax = self.vmax
        midpoint = self.midpoint

        rescaled = np.interp(
            result.data,
            [vmin, midpoint, vmax],
            [0, 0.5, 1]
        )

        return np.ma.array(rescaled, mask=result.mask, copy=False)


# ======================================================================
# Plotting routine
# ======================================================================

def plot_weight_maps(
    weights,
    variable,
    lon=None,
    lat=None,
    extent=None,
    easting=None,
    northing=None,
    save_path=None,
    na_mask=None
):
    """
    Plot horizontal weight maps for multiple methods (e.g. GPCA, regression).

    Supports:
    - Precipitation-type variables on a regular lon/lat grid
    - NDVI-type variables on a projected grid with a land mask
    """

    n_maps = len(weights)

    # ------------------------------------------------------------------
    # Layout: two rows, variable number of columns
    # ------------------------------------------------------------------

    n_rows = 2
    n_cols = math.ceil(n_maps / 2)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 4 * n_rows),
        subplot_kw={'projection': ccrs.PlateCarree()},
        constrained_layout=True
    )

    axes = np.array(axes).flatten()[:n_maps]

    # ------------------------------------------------------------------
    # Global normalization for consistent color scale
    # ------------------------------------------------------------------

    all_vals = np.concatenate([np.ravel(v['weights']) for v in weights.values()])
    all_vals = all_vals[np.isfinite(all_vals)]

    vmin, vmax = np.percentile(all_vals, [0.01, 99.99])
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    meshes = []

    # ------------------------------------------------------------------
    # Loop over methods / maps
    # ------------------------------------------------------------------

    for ax, (name, data) in zip(axes, weights.items()):

        w = np.array(data['weights'])
        pval = np.array(data['p-value'])

        if np.ndim(pval) > 0:
            pval = float(pval[0])

        title_name = f"{name} - p={pval:.3e}"

        # --------------------------------------------------------------
        # Precipitation-type variable (regular lat/lon grid)
        # --------------------------------------------------------------

        if variable.lower() == 'precipitation':

            # Reshape weights to 2D grid
            w2d = w.reshape(len(lat), len(lon))

            # Build land mask using Natural Earth
            land_50m = cfeature.NaturalEarthFeature(
                "physical", "land", "50m", facecolor="none"
            )

            land_mask = np.zeros_like(w2d, dtype=bool)

            land_geom = ops.unary_union(list(land_50m.geometries()))
            land_geom = prep(land_geom)

            for i in range(len(lat)):
                for j in range(len(lon)):
                    if land_geom.contains(Point(lon[j], lat[i])):
                        land_mask[i, j] = True

            # Mask ocean points
            w2d[~land_mask] = np.nan

            mesh = ax.pcolormesh(
                lon,
                lat,
                w2d,
                transform=ccrs.PlateCarree(),
                cmap="RdBu_r",
                norm=norm,
                shading="auto"
            )

            ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()])

        # --------------------------------------------------------------
        # NDVI-type variable (projected grid + NA mask)
        # --------------------------------------------------------------

        elif variable.lower() == 'ndvi':

            weight_map = np.zeros_like(na_mask, dtype=float).T
            weight_map[na_mask.T == 1] = w
            weight_map[na_mask.T == 0] = np.nan

            transformer = Transformer.from_crs(
                "EPSG:6933",
                "EPSG:4326",
                always_xy=True
            )

            easting_2d, northing_2d = np.meshgrid(
                easting,
                northing,
                indexing="ij"
            )

            lon, lat = transformer.transform(easting_2d, northing_2d)
            lon = lon.T
            lat = lat.T

            mesh = ax.pcolormesh(
                lon,
                lat,
                weight_map.T,
                cmap='RdBu_r',
                norm=norm,
                shading='auto',
                transform=ccrs.PlateCarree()
            )

            if extent:
                ax.set_extent(extent, crs=ccrs.PlateCarree())

        # --------------------------------------------------------------
        # Map decorations
        # --------------------------------------------------------------

        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.set_title(title_name, fontsize=13)

        meshes.append(mesh)

    # ------------------------------------------------------------------
    # Shared colorbar
    # ------------------------------------------------------------------

    cbar = fig.colorbar(
        meshes[-1],
        ax=axes,
        orientation='vertical',
        fraction=0.025,
        pad=0.02,
        label='Weight value',
        shrink=0.7
    )
    cbar.ax.tick_params(labelsize=10)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


# ======================================================================
# Utility
# ======================================================================

def normalise(w):
    """Standard-score normalization."""
    return (w - w.mean()) / w.std()
