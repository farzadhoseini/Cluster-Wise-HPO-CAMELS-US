from __future__ import annotations

"""
Plot the 30 low-performing CAMELS-US basins (Cluster-wise model, NSE < 0.5)
on a CONUS map.

What the script does
--------------------
1) Loads all 531 basins from the master CAMELS attribute table.
2) Loads the low-performing basin table (bad_basins_clusterwise_nse_lt_0p5.csv).
3) Plots all basins in light gray / faded.
4) Highlights only the bad basins using a color scale based on NSE.
5) Optionally draws the USA outline if GeoPandas can access a country boundary.

Expected input files (default paths)
------------------------------------
- F:\Experiments\CAMELS_US\Clean_4_upload\camels_531_master_attributes_clusters.csv
- F:\Experiments\CAMELS_US\Clean_4_upload\bad_basins_clusterwise_nse_lt_0p5.csv

Output
------
- F:\Experiments\CAMELS_US\Clean_4_upload\cluster_performance_outputs\fig_bad_basins_clusterwise_nse_lt_0p5_map.png

Dependencies
------------
pip install pandas matplotlib geopandas openpyxl
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except Exception as e:  # pragma: no cover
    raise ImportError(
        "GeoPandas is required for this script. Install it with: pip install geopandas"
    ) from e

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_OK = True
except Exception:
    CARTOPY_OK = False

CONUS_EXTENT = (-125.2, -66.3, 24.0, 50.0)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(r"F:\Experiments\CAMELS_US\Clean_4_upload")
ATTR_FILE = PROJECT_ROOT / "camels_531_master_attributes_clusters.csv"
BAD_FILE = PROJECT_ROOT / "bad_basins_clusterwise_nse_lt_0p5.csv"
OUTPUT_DIR = PROJECT_ROOT / "cluster_performance_outputs"
OUTPUT_FIG = OUTPUT_DIR / "fig_bad_basins_clusterwise_nse_lt_0p5_map.png"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def zfill_basin(x) -> str:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s.endswith('.0'):
        s = s[:-2]
    digits = ''.join(ch for ch in s if ch.isdigit())
    return digits.zfill(8)


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def detect_basin_col(df: pd.DataFrame) -> str:
    col = find_col(df, ["basin", "gauge_id", "gaugeid", "station_id", "camels_id"])
    if col is None:
        raise ValueError(f"Could not detect basin column. Available columns: {list(df.columns)}")
    return col


def detect_latlon_cols(df: pd.DataFrame) -> tuple[str, str]:
    lat = find_col(df, ["gauge_lat", "lat", "latitude"])
    lon = find_col(df, ["gauge_lon", "lon", "longitude", "long"])
    if lat is None or lon is None:
        raise ValueError(f"Could not detect lat/lon columns. Available columns: {list(df.columns)}")
    return lat, lon


def load_us_boundary():
    """
    Return a GeoDataFrame for the USA outline if available.
    Tries a few common routes so the script works on more systems.
    """
    # Option 1: classic GeoPandas bundled dataset (older versions)
    try:
        path = gpd.datasets.get_path("naturalearth_lowres")
        world = gpd.read_file(path)
        name_col = "name" if "name" in world.columns else None
        if name_col is not None:
            usa = world.loc[world[name_col].isin(["United States of America", "United States"])]
            if len(usa) > 0:
                return usa
    except Exception:
        pass

    # Option 2: cartopy Natural Earth
    try:
        import cartopy.io.shapereader as shpreader
        shp = shpreader.natural_earth(
            resolution="110m", category="cultural", name="admin_0_countries"
        )
        world = gpd.read_file(shp)
        for col in ["ADMIN", "NAME", "name"]:
            if col in world.columns:
                usa = world.loc[world[col].isin(["United States of America", "United States"])]
                if len(usa) > 0:
                    return usa
    except Exception:
        pass

    warnings.warn(
        "Could not load a USA boundary automatically. The figure will still be produced, "
        "but without the country outline."
    )
    return None




def load_state_province_lines():
    """Return admin-1 state/province boundaries as a GeoDataFrame when available."""
    try:
        import cartopy.io.shapereader as shpreader
        shp = shpreader.natural_earth(
            resolution="50m", category="cultural", name="admin_1_states_provinces_lines"
        )
        gdf = gpd.read_file(shp)
        return gdf
    except Exception:
        return None

def load_all_basins(attr_file: Path) -> pd.DataFrame:
    df = pd.read_csv(attr_file)
    basin_col = detect_basin_col(df)
    lat_col, lon_col = detect_latlon_cols(df)

    out = df.copy()
    out[basin_col] = out[basin_col].map(zfill_basin)
    out = out.rename(columns={basin_col: "basin", lat_col: "lat", lon_col: "lon"})

    keep_cols = ["basin", "lat", "lon"]
    for extra in ["gauge_name", "cluster", "cluster_label"]:
        if extra in out.columns:
            keep_cols.append(extra)

    out = out[keep_cols].dropna(subset=["basin", "lat", "lon"]).drop_duplicates(subset=["basin"])
    return out


def load_bad_basins(bad_file: Path, all_basins: pd.DataFrame) -> pd.DataFrame:
    if bad_file.suffix.lower() == ".xlsx":
        df = pd.read_excel(bad_file)
    else:
        df = pd.read_csv(bad_file)

    basin_col = detect_basin_col(df)
    df[basin_col] = df[basin_col].map(zfill_basin)
    df = df.rename(columns={basin_col: "basin"})

    if "NSE" not in df.columns:
        raise ValueError("The bad-basin file must contain an 'NSE' column.")

    # If coordinates are already present, keep them; otherwise merge from master table.
    if {"gauge_lat", "gauge_lon"}.issubset(df.columns):
        df = df.rename(columns={"gauge_lat": "lat", "gauge_lon": "lon"})
    elif not {"lat", "lon"}.issubset(df.columns):
        df = df.merge(all_basins[["basin", "lat", "lon"]], on="basin", how="left")

    keep_cols = ["basin", "NSE", "lat", "lon"]
    for extra in ["gauge_name", "station_name", "cluster", "cluster_label"]:
        if extra in df.columns:
            keep_cols.append(extra)

    out = df[keep_cols].copy()
    out = out.dropna(subset=["basin", "NSE", "lat", "lon"]).drop_duplicates(subset=["basin"])
    out = out.sort_values("NSE", ascending=True).reset_index(drop=True)
    return out


def to_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )


def plot_bad_basins_map(
    all_basins_df: pd.DataFrame,
    bad_basins_df: pd.DataFrame,
    save_path: Path,
    figsize=(13.8, 7.7),
    background_alpha: float = 0.20,
    background_size: int = 18,
    highlight_size: int = 76,
    cmap: str = "cividis",
):
    ensure_dir(save_path.parent)

    gdf_all = to_gdf(all_basins_df)
    gdf_bad = to_gdf(bad_basins_df)

    if CARTOPY_OK:
        map_crs = ccrs.LambertConformal(central_longitude=-96, central_latitude=39, standard_parallels=(33, 45))
        data_crs = ccrs.PlateCarree()
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=map_crs)
        ax.set_extent(CONUS_EXTENT, crs=data_crs)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f6f6f6", zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#efefef", zorder=0)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=1.1, edgecolor="#333333", zorder=1.0)
        states = cfeature.NaturalEarthFeature("cultural", "admin_1_states_provinces_lines", "50m", facecolor="none")
        ax.add_feature(states, edgecolor="#9a9a9a", linewidth=0.6, zorder=1.0)
        ax.coastlines(resolution="50m", linewidth=0.9, color="#444444", zorder=1.0)
        gl = ax.gridlines(draw_labels=False, linestyle="--", linewidth=0.45, color="#bcbcbc", alpha=0.8)

        ax.scatter(gdf_all["lon"], gdf_all["lat"], s=background_size, c="lightgray", alpha=background_alpha,
                   edgecolors="none", transform=data_crs, zorder=2, label="Other CAMELS-US basins")
        sc = ax.scatter(gdf_bad["lon"], gdf_bad["lat"], c=gdf_bad["NSE"], s=highlight_size, cmap=cmap,
                        vmin=float(gdf_bad["NSE"].min()), vmax=0.5, edgecolors="black", linewidths=0.45,
                        transform=data_crs, zorder=4, label="Basins with NSE < 0.5")
    else:
        usa = load_us_boundary()
        states = load_state_province_lines()
        fig, ax = plt.subplots(figsize=figsize)
        if usa is not None and len(usa) > 0:
            try:
                usa = usa.to_crs(gdf_all.crs)
                usa.boundary.plot(ax=ax, color="black", linewidth=1.15, alpha=0.88, zorder=1)
            except Exception:
                pass
        if states is not None and len(states) > 0:
            try:
                states = states.to_crs(gdf_all.crs)
                states.plot(ax=ax, color="none", edgecolor="0.55", linewidth=0.55, alpha=0.85, zorder=1.5)
            except Exception:
                try:
                    states.boundary.plot(ax=ax, color="0.55", linewidth=0.55, alpha=0.85, zorder=1.5)
                except Exception:
                    pass
        ax.scatter(gdf_all["lon"], gdf_all["lat"], s=background_size, c="lightgray", alpha=background_alpha, edgecolors="none", zorder=2, label="Other CAMELS-US basins")
        sc = ax.scatter(gdf_bad["lon"], gdf_bad["lat"], c=gdf_bad["NSE"], s=highlight_size, cmap=cmap,
                        vmin=float(gdf_bad["NSE"].min()), vmax=0.5, edgecolors="black", linewidths=0.45, zorder=4, label="Basins with NSE < 0.5")
        ax.set_xlim(-125.2, -66.3)
        ax.set_ylim(24.0, 50.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Longitude", fontsize=18)
        ax.set_ylabel("Latitude", fontsize=18)
        ax.grid(alpha=0.18, linewidth=0.6)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.92, pad=0.02)
    cbar.set_label("NSE (Cluster-wise model; bad basins only)", fontsize=15)
    cbar.ax.tick_params(labelsize=13)

    ax.set_title(
        "Geographic distribution of the lower-skill basins in the cluster-wise family benchmark\n"
        "Basins with NSE < 0.5 are highlighted; the remaining CAMELS-US basins are shown in gray",
        fontsize=18.5, fontweight="bold", pad=12,
    )

    n_bad = len(gdf_bad)
    n_total = len(gdf_all)
    ax.text(0.985, 0.02, f"Highlighted basins: {n_bad} / {n_total}", transform=ax.transAxes, fontsize=12,
            va="bottom", ha="right", bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.88, edgecolor="0.6"))

    leg = ax.legend(loc="upper left", frameon=True, fontsize=13)
    handles = getattr(leg, "legendHandles", None) or getattr(leg, "legend_handles", None)
    if handles is not None:
        for h in handles:
            try:
                h.set_alpha(1.0)
            except Exception:
                pass

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    if not ATTR_FILE.exists():
        raise FileNotFoundError(f"Missing attributes file: {ATTR_FILE}")
    if not BAD_FILE.exists():
        raise FileNotFoundError(f"Missing bad-basin file: {BAD_FILE}")

    all_basins = load_all_basins(ATTR_FILE)
    bad_basins = load_bad_basins(BAD_FILE, all_basins)

    # Safety check: in the current workflow this should be the 30 basins with NSE < 0.5
    print(f"[INFO] Loaded {len(all_basins)} total basins.")
    print(f"[INFO] Loaded {len(bad_basins)} highlighted low-performing basins.")
    print(f"[INFO] NSE range among highlighted basins: {bad_basins['NSE'].min():.3f} to {bad_basins['NSE'].max():.3f}")

    plot_bad_basins_map(all_basins, bad_basins, OUTPUT_FIG)
    print(f"[OK] Figure saved to: {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
