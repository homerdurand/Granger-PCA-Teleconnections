import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from pyproj import Transformer
from shapely.geometry import Point
from shapely.prepared import prep
import shapely.ops as ops


# ==========================================================
# Publication-quality global styling
# ==========================================================

mpl.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 24,
    "axes.linewidth": 1.2,
})


# ==========================================================
# Plot function
# ==========================================================

def plot_weight_maps(
    weights,
    variable,
    lon=None,
    lat=None,
    extent=None,
    easting=None,
    northing=None,
    save_path=None,
    na_mask=None,
    cmap='RdBu_r',
    n_rows = 2
):

    n_maps = len(weights)
    n_cols = math.ceil(n_maps / 2)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8 * n_cols, 6 * n_rows),
        subplot_kw={'projection': ccrs.PlateCarree()},
        constrained_layout=True
    )

    axes = np.array(axes).flatten()[:n_maps]

    # ------------------------------------------------------
    # Global normalization
    # ------------------------------------------------------

    all_vals = np.concatenate([np.ravel(v['weights']) for v in weights.values()])
    all_vals = all_vals[np.isfinite(all_vals)]

    vmin, vmax = np.percentile(all_vals, [1, 99])
    vmax = np.max([np.abs(vmin), np.abs(vmax)])
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    meshes = []

    for ax, (name, data) in zip(axes, weights.items()):

        w = np.array(data['weights'])
        pval = np.array(data['p-value'])

        if np.ndim(pval) > 0:
            pval = float(pval[0])

        title_name = f"{name}   (p = {pval:.2e})"

        # ==================================================
        # PRECIPITATION
        # ==================================================

        if variable.lower() == 'precipitation':

            w2d = w.reshape(len(lat), len(lon))

            sig = None
            if 'significances' in data:
                sig = np.array(data['significances'], dtype=bool)
                sig = sig.reshape(len(lat), len(lon))

            land_50m = cfeature.NaturalEarthFeature(
                "physical", "land", "50m", facecolor="none"
            )

            land_mask = np.zeros_like(w2d, dtype=bool)
            land_geom = prep(ops.unary_union(list(land_50m.geometries())))

            for i in range(len(lat)):
                for j in range(len(lon)):
                    if land_geom.contains(Point(lon[j], lat[i])):
                        land_mask[i, j] = True

            w2d[~land_mask] = np.nan

            mesh = ax.pcolormesh(
                lon,
                lat,
                w2d,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                norm=norm,
                shading="auto"
            )

            if sig is not None:
                nonsig = (~sig) & land_mask
                if np.any(nonsig):
                    lon2d, lat2d = np.meshgrid(lon, lat)
                    ax.scatter(
                        lon2d[nonsig],
                        lat2d[nonsig],
                        s=12,
                        color="black",
                        alpha=0.7,
                        linewidth=0,
                        transform=ccrs.PlateCarree(),
                        zorder=4
                    )

            ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()])

        # ==================================================
        # NDVI
        # ==================================================

        elif variable.lower() == 'ndvi':

            # --------------------------------------------------
            # Build 2D weight field
            # --------------------------------------------------

            weight_map = np.zeros_like(na_mask, dtype=float).T
            weight_map[na_mask.T == 1] = w
            weight_map[na_mask.T == 0] = np.nan

            # --------------------------------------------------
            # Significance handling (same logic as precipitation)
            # --------------------------------------------------

            sig = None
            if 'significances' in data:
                sig = np.array(data['significances'], dtype=bool)
                sig_map = np.zeros_like(na_mask, dtype=bool).T
                sig_map[na_mask.T == 1] = sig
                sig_map[na_mask.T == 0] = False
            else:
                sig_map = None

            # --------------------------------------------------
            # Transform projection (EPSG:6933 → WGS84)
            # --------------------------------------------------

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

            lon2d, lat2d = transformer.transform(easting_2d, northing_2d)

            lon2d = lon2d.T
            lat2d = lat2d.T
            weight_map = weight_map.T

            # --------------------------------------------------
            # Plot field
            # --------------------------------------------------

            mesh = ax.pcolormesh(
                lon2d,
                lat2d,
                weight_map,
                cmap=cmap,
                norm=norm,
                shading='auto',
                transform=ccrs.PlateCarree(),
                zorder=1
            )

            # --------------------------------------------------
            # Overlay non-significant points
            # --------------------------------------------------

            if sig_map is not None:
                nonsig = (~sig_map.T) & np.isfinite(weight_map)

                if np.any(nonsig):
                    ax.contourf(
                        lon2d,
                        lat2d,
                        nonsig,
                        levels=[0.5, 1],
                        colors='none',
                        hatches=['///'],   # try '...', 'xx', etc.
                        transform=ccrs.PlateCarree(),
                        zorder=3
                    )

            if extent:
                ax.set_extent(extent, crs=ccrs.PlateCarree())

        # --------------------------------------------------
        # Map styling
        # --------------------------------------------------

        ax.coastlines(linewidth=1.4)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1.1)
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

        ax.set_title(title_name, pad=14)

        meshes.append(mesh)

    # ------------------------------------------------------
    # Shared colorbar
    # ------------------------------------------------------

    cbar = fig.colorbar(
        meshes[-1],
        ax=axes,
        orientation='vertical',
        fraction=0.04,
        pad=0.03,
        shrink=0.9
    )

    cbar.set_label("Correlation", fontsize=22, labelpad=14)
    cbar.ax.tick_params(labelsize=18, width=1.2)

    # ------------------------------------------------------
    # Save high-resolution
    # ------------------------------------------------------

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=400)

    plt.show()