from __future__ import annotations

"""
Professional CONUS map for CAMELS-US hydrologic clusters.

What this script does
---------------------
- plots the 531 CAMELS-US basins on a cartographic map of CONUS
- shows a terrain-like background (online terrain tiles when available;
  otherwise falls back to a clean cartographic base)
- overlays U.S. national border and state boundaries
- preserves a standard map aspect/projection
- shows longitude/latitude grid labels
- adds a scale bar and north arrow
- labels cluster centroids with cluster ID and basin count

Typical usage
-------------
python plot_camels_clusters_pro_map.py \
    --input "F:/Experiments/CAMELS_US/Clean_4_upload/camels_531_master_attributes_clusters.csv" \
    --output "F:/Experiments/CAMELS_US/Clean_4_upload/results/fig_camels_clusters_map_pro.png"

Requirements
------------
pip install pandas matplotlib cartopy pyproj openpyxl

Optional (only if you want local raster relief background instead of tiles):
pip install rasterio

Notes
-----
1) If internet is available, the script will try to use an open terrain tile
   background. If that fails, it falls back automatically to a clean vector map.
2) The script accepts either:
   - camels_531_master_attributes_clusters.csv, or
   - CAMELS_US_Clusters.xlsx + a separate attributes table containing lat/lon.
3) Cluster numbering is standardized to 1..6. If your input uses 0..5, it will
   be converted automatically.
"""

from pathlib import Path
import argparse
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle
from matplotlib import patheffects as pe

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.img_tiles as cimgt
except Exception as e:
    raise ImportError(
        "This script requires cartopy. Install it with: pip install cartopy"
    ) from e

try:
    from pyproj import Geod
except Exception as e:
    raise ImportError(
        "This script requires pyproj. Install it with: pip install pyproj"
    ) from e

# Optional local-raster support
try:
    import rasterio
    _HAS_RASTERIO = True
except Exception:
    _HAS_RASTERIO = False

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
CONUS_EXTENT = (-125.2, -66.3, 24.0, 50.0)  # lon_min, lon_max, lat_min, lat_max

# Colors deliberately kept consistent with the figure you were already using.
CLUSTER_COLORS = {
    1: "#1f77b4",  # blue
    2: "#ff7f0e",  # orange
    3: "#2ca02c",  # green
    4: "#d62728",  # red
    5: "#9467bd",  # purple
    6: "#8c564b",  # brown
}

# Projected CRS for a visually standard CONUS map.
# Lambert Conformal is widely used for U.S. thematic mapping and avoids the
# "stretched" look of a simple lon/lat scatter plot.
MAP_CRS = ccrs.LambertConformal(
    central_longitude=-96,
    central_latitude=39,
    standard_parallels=(33, 45),
)
DATA_CRS = ccrs.PlateCarree()


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def zfill_basin(x) -> str:
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.zfill(8)


def normalize_clusters(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    unique_vals = sorted(v for v in s.dropna().unique().tolist())
    if unique_vals == [0, 1, 2, 3, 4, 5]:
        s = s + 1
    return s.astype("Int64")


def load_input_table(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    elif input_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path)
    else:
        raise ValueError("Input file must be CSV or Excel.")

    cols = {c.lower(): c for c in df.columns}

    def find_col(candidates):
        for cand in candidates:
            if cand.lower() in cols:
                return cols[cand.lower()]
        # normalized fallback
        norm = {c.lower().replace("_", "").replace("-", "").replace(" ", ""): c for c in df.columns}
        for cand in candidates:
            key = cand.lower().replace("_", "").replace("-", "").replace(" ", "")
            if key in norm:
                return norm[key]
        return None

    basin_col = find_col(["gauge_id", "basin", "basin_id", "station_id"])
    lat_col = find_col(["gauge_lat", "lat", "latitude"])
    lon_col = find_col(["gauge_lon", "lon", "longitude"])
    cluster_col = find_col(["cluster", "cluster_raw", "cluster_id"])

    missing = []
    if basin_col is None:
        missing.append("basin/gauge_id")
    if lat_col is None:
        missing.append("lat/gauge_lat")
    if lon_col is None:
        missing.append("lon/gauge_lon")
    if cluster_col is None:
        missing.append("cluster")
    if missing:
        raise KeyError(
            f"Missing required columns in {input_path.name}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    out = df[[basin_col, lat_col, lon_col, cluster_col]].copy()
    out.columns = ["basin", "lat", "lon", "cluster"]
    out["basin"] = out["basin"].map(zfill_basin)
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out["cluster"] = normalize_clusters(out["cluster"])
    out = out.dropna(subset=["lat", "lon", "cluster"]).copy()
    out["cluster"] = out["cluster"].astype(int)
    out["cluster_label"] = out["cluster"].map(lambda x: f"Cluster {x}")
    return out.drop_duplicates(subset=["basin"]).reset_index(drop=True)


def add_base_features(ax):
    """Add a clean fallback base map."""
    land = cfeature.NaturalEarthFeature(
        "physical", "land", "50m", facecolor="#f3f1ea", edgecolor="none"
    )
    ocean = cfeature.NaturalEarthFeature(
        "physical", "ocean", "50m", facecolor="#eef3f6", edgecolor="none"
    )
    lakes = cfeature.NaturalEarthFeature(
        "physical", "lakes", "50m", facecolor="#dcecf7", edgecolor="none"
    )
    states = cfeature.NaturalEarthFeature(
        "cultural",
        "admin_1_states_provinces_lines",
        "50m",
        facecolor="none",
        edgecolor="#808080",
    )
    countries = cfeature.NaturalEarthFeature(
        "cultural",
        "admin_0_countries",
        "50m",
        facecolor="none",
        edgecolor="#444444",
    )

    ax.add_feature(ocean, zorder=0)
    ax.add_feature(land, zorder=0.1)
    ax.add_feature(lakes, zorder=0.2)
    ax.add_feature(states, linewidth=0.70, alpha=0.72, zorder=1)
    ax.add_feature(countries, linewidth=1.35, alpha=0.98, zorder=1.1)
    ax.coastlines(resolution="50m", linewidth=0.95, color="#444444", zorder=1.2)


class OpenTerrainTiles(cimgt.GoogleTiles):
    """
    Open terrain-like background using Stadia/OSM tile endpoints.
    This may require internet access and can fail on some systems/network setups.
    """
    def _image_url(self, tile):
        x, y, z = tile
        return f"https://tiles.stadiamaps.com/tiles/stamen_terrain_background/{z}/{x}/{y}.png"


def add_relief_background(ax, zoom=5, relief_alpha=0.42, local_raster: str | None = None):
    """
    Try, in order:
      1) a local georeferenced raster relief if provided,
      2) open terrain tiles (internet),
      3) a clean vector fallback.
    """
    done = False

    # Option 1: local georeferenced raster (GeoTIFF) if user has one.
    if local_raster is not None and _HAS_RASTERIO:
        raster_path = Path(local_raster)
        if raster_path.exists():
            try:
                with rasterio.open(raster_path) as src:
                    img = src.read()
                    # Convert from bands-first to rows/cols/bands.
                    if img.shape[0] >= 3:
                        img = img[:3, :, :].transpose(1, 2, 0)
                    else:
                        img = img[0, :, :]
                    bounds = src.bounds
                    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
                    ax.imshow(
                        img,
                        origin="upper",
                        extent=extent,
                        transform=DATA_CRS,
                        alpha=relief_alpha,
                        zorder=0,
                    )
                done = True
            except Exception as e:
                warnings.warn(f"Could not read local relief raster: {e}")

    # Option 2: open terrain tiles.
    if not done:
        try:
            tiler = OpenTerrainTiles()
            ax.add_image(tiler, zoom, alpha=relief_alpha, interpolation="bilinear", zorder=0)
            done = True
        except Exception as e:
            warnings.warn(f"Open terrain tiles not available. Falling back to a clean base map. Details: {e}")

    # Option 3: fallback.
    if not done:
        add_base_features(ax)
    else:
        # Even with relief tiles, still overlay borders cleanly.
        states = cfeature.NaturalEarthFeature(
            "cultural", "admin_1_states_provinces_lines", "50m",
            facecolor="none", edgecolor="#676767"
        )
        countries = cfeature.NaturalEarthFeature(
            "cultural", "admin_0_countries", "50m",
            facecolor="none", edgecolor="#303030"
        )
        ax.add_feature(states, linewidth=0.42, alpha=0.8, zorder=1)
        ax.add_feature(countries, linewidth=0.95, alpha=0.95, zorder=1.1)
        ax.coastlines(resolution="50m", linewidth=0.6, color="#444444", zorder=1.2)


def add_lonlat_grid(ax):
    gl = ax.gridlines(
        crs=DATA_CRS,
        draw_labels=True,
        linewidth=0.5,
        color="#9a9a9a",
        alpha=0.55,
        linestyle="--",
        zorder=2,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 10}
    gl.ylabel_style = {"size": 10}
    try:
        import matplotlib.ticker as mticker
        gl.xlocator = mticker.FixedLocator([-120, -110, -100, -90, -80, -70])
        gl.ylocator = mticker.FixedLocator([25, 30, 35, 40, 45, 50])
    except Exception:
        pass


def add_cluster_labels(ax, df: pd.DataFrame):
    grouped = df.groupby("cluster", as_index=False).agg(
        lon=("lon", "median"),
        lat=("lat", "median"),
        n=("basin", "count"),
    )

    # Small manual nudges for cleaner placement, matching the visual layout.
    nudges = {
        1: (1.5, 0.4),
        2: (1.0, -0.1),
        3: (0.2, -1.0),
        4: (-0.4, 0.0),
        5: (-0.2, 0.2),
        6: (0.5, -0.4),
    }

    for _, row in grouped.iterrows():
        c = int(row["cluster"])
        dx, dy = nudges.get(c, (0, 0))
        txt = ax.text(
            row["lon"] + dx,
            row["lat"] + dy,
            f"C{c}",
            transform=DATA_CRS,
            ha="center",
            va="center",
            fontsize=12.5,
            fontweight="bold",
            color="#222222",
            zorder=5,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="none",
                alpha=0.62,
            ),
        )
        txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white", alpha=0.75)])


def add_north_arrow(ax, location=(0.965, 0.09), size=0.06):
    x, y = location
    ax.annotate(
        "N",
        xy=(x, y + size),
        xytext=(x, y),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#222222"),
        zorder=10,
    )


def add_scalebar(ax, length_km=500, location=(0.12, 0.08), n_segments=2, bar_height_deg=0.22):
    """Professional segmented scale bar kept well inside the map frame."""
    geod = Geod(ellps="WGS84")
    lon_min, lon_max, lat_min, lat_max = ax.get_extent(crs=DATA_CRS)
    start_lon = lon_min + (lon_max - lon_min) * location[0]
    start_lat = lat_min + (lat_max - lat_min) * location[1]
    seg_km = length_km / n_segments

    current_lon, current_lat = start_lon, start_lat
    xs = [start_lon]
    ys = [start_lat]
    for _ in range(n_segments):
        next_lon, next_lat, _ = geod.fwd(current_lon, current_lat, 90, seg_km * 1000.0)
        xs.append(next_lon)
        ys.append(next_lat)
        current_lon, current_lat = next_lon, next_lat

    # alternating black-white segments as a clean professional scale bar
    for i in range(n_segments):
        x0, x1 = xs[i], xs[i+1]
        y0 = ys[i]
        color = "#222222" if i % 2 == 0 else "white"
        ax.add_patch(Rectangle((x0, y0), x1 - x0, bar_height_deg, transform=DATA_CRS,
                               facecolor=color, edgecolor="#222222", linewidth=1.0, zorder=8))
    # labels
    for i, lab in enumerate([0, int(seg_km), int(length_km)]):
        x = xs[i] if i < len(xs) else xs[-1]
        ax.text(x, start_lat - bar_height_deg * 0.55, f"{lab}", transform=DATA_CRS,
                ha="center", va="top", fontsize=10.5, fontweight="bold", color="#222222", zorder=9,
                bbox=dict(boxstyle="round,pad=0.05", facecolor="white", edgecolor="none", alpha=0.65))
    ax.text((xs[0] + xs[-1]) / 2, start_lat + bar_height_deg * 1.55, "km", transform=DATA_CRS,
            ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#222222", zorder=9,
            bbox=dict(boxstyle="round,pad=0.05", facecolor="white", edgecolor="none", alpha=0.65))


def plot_clusters_map(df: pd.DataFrame, output_path: Path, title: str, local_relief: str | None = None):
    fig = plt.figure(figsize=(14.8, 8.6), dpi=300)
    ax = plt.axes(projection=MAP_CRS)
    ax.set_extent(CONUS_EXTENT, crs=DATA_CRS)

    add_relief_background(ax, zoom=5, relief_alpha=0.42, local_raster=local_relief)
    add_lonlat_grid(ax)

    # Plot basins cluster-by-cluster for a clean legend.
    for cluster_id in sorted(df["cluster"].unique()):
        sub = df.loc[df["cluster"] == cluster_id]
        ax.scatter(
            sub["lon"],
            sub["lat"],
            s=48,
            c=CLUSTER_COLORS.get(cluster_id, "tab:blue"),
            label=f"Cluster {cluster_id}",
            transform=DATA_CRS,
            alpha=0.88,
            edgecolors="black",
            linewidths=0.35,
            zorder=5,
        )

    add_cluster_labels(ax, df)
    add_scalebar(ax, length_km=500, location=(0.10, 0.07), n_segments=2, bar_height_deg=0.18)
    add_north_arrow(ax)

    # Legend
    legend_handles = [
        mlines.Line2D(
            [], [],
            marker="o", linestyle="None",
            markersize=8.5,
            markerfacecolor=CLUSTER_COLORS[c],
            markeredgecolor="black",
            markeredgewidth=0.45,
            label=f"Cluster {c} (n={int((df['cluster']==c).sum())})"
        )
        for c in sorted(df["cluster"].unique())
    ]
    leg = ax.legend(
        handles=legend_handles,
        title="Hydrologic clusters",
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#b0b0b0",
        fontsize=13.5,
        title_fontsize=14.5,
    )
    leg.set_zorder(10)

    # Title and subtitle
    ax.set_title(
        title,
        fontsize=22,
        fontweight="bold",
        pad=14,
    )

    # Clean frame
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#666666")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Professional CAMELS-US cluster map")
    parser.add_argument(
        "--input",
        type=str,
        default="camels_531_master_attributes_clusters.csv",
        help="CSV/Excel table with basin, lat, lon, and cluster columns.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="fig_camels_clusters_map_pro.png",
        help="Output PNG file.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="CAMELS-US study basins grouped into six hydrologically informed clusters",
        help="Figure title.",
    )
    parser.add_argument(
        "--local-relief",
        type=str,
        default=None,
        help=(
            "Optional path to a local georeferenced relief raster (GeoTIFF). "
            "If provided, the script will use it as the topographic background."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = load_input_table(input_path)

    print(f"[INFO] Loaded {len(df)} basins from: {input_path}")
    print(f"[INFO] Cluster counts:\n{df['cluster'].value_counts().sort_index().to_string()}")

    plot_clusters_map(
        df=df,
        output_path=output_path,
        title=args.title,
        local_relief=args.local_relief,
    )

    print(f"[OK] Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
