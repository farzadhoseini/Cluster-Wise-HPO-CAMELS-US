from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

from camels_utils import (
    get_project_root,
    get_pickles_dir,
    load_pickle,
    safe_numeric,
    flatten_metrics_all_test_period,
)

try:
    import geopandas as gpd
except Exception:
    gpd = None


# =========================================================
# SETTINGS
# =========================================================
PRIMARY_METRIC = "NSE"
OUTPUT_SUBDIR = "Results/Tables_cluster"

MODEL_ORDER = ["Benchmark", "Top10", "Cluster-wise"]
CLUSTER_COLORS = {
    1: "#1f77b4",  # blue
    2: "#ff7f0e",  # orange
    3: "#2ca02c",  # green
    4: "#d62728",  # red
    5: "#9467bd",  # purple
    6: "#8c564b",  # brown
}


# =========================================================
# HELPERS
# =========================================================
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_text(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("_", "").replace("-", "").replace(" ", "")
    return s


def zfill_basin(x) -> str:
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    return s.zfill(8)


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm_map = {normalize_text(c): c for c in df.columns}
    for cand in candidates:
        nc = normalize_text(cand)
        if nc in norm_map:
            return norm_map[nc]
    return None


def detect_basin_col(df: pd.DataFrame) -> str:
    c = find_column(
        df,
        ["basin", "gauge_id", "gaugeid", "basin_id", "basinid", "station_id", "camels_id"],
    )
    if c is None:
        raise ValueError(f"Could not detect basin column. Available columns: {list(df.columns)}")
    return c


def detect_cluster_col(df: pd.DataFrame) -> str:
    c = find_column(df, ["cluster", "clusters", "cluster_id", "clusterid"])
    if c is None:
        raise ValueError(f"Could not detect cluster column. Available columns: {list(df.columns)}")
    return c


def canonicalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    metric_aliases = {
        "NSE": ["NSE", "nse"],
        "KGE": ["KGE", "kge"],
        "Pearson-r": ["Pearson-r", "Pearson_r", "pearson_r", "pearsonr", "corr", "correlation"],
        "RMSE": ["RMSE", "rmse"],
        "Peak-timing": ["Peak-timing", "peak_timing", "peaktiming", "peak_time", "peaktime"],
        "Peak-MAPE": ["Peak-MAPE", "peak_mape", "peakmape", "mape_peak", "peak_mape_percent"],
        "Missed-peaks": ["Missed-peaks", "missed_peaks", "missedpeaks", "missed_peak_ratio"],
    }
    rename_map = {}
    for canonical, aliases in metric_aliases.items():
        col = find_column(df, aliases)
        if col is not None:
            rename_map[col] = canonical
    return df.rename(columns=rename_map)


# =========================================================
# LOADERS
# =========================================================
def load_clusters(project_root: Path) -> pd.DataFrame:
    cluster_file = project_root / "CAMELS_US_Clusters.xlsx"
    if not cluster_file.exists():
        raise FileNotFoundError(f"Cluster file not found: {cluster_file}")

    df = pd.read_excel(cluster_file)
    basin_col = detect_basin_col(df)
    cluster_col = detect_cluster_col(df)

    out = df[[basin_col, cluster_col]].copy()
    out.columns = ["basin", "cluster_raw"]
    out["basin"] = out["basin"].map(zfill_basin)
    out["cluster_raw"] = pd.to_numeric(out["cluster_raw"], errors="coerce").astype("Int64")

    vals = sorted(v for v in out["cluster_raw"].dropna().unique().tolist())
    if vals == [0, 1, 2, 3, 4, 5]:
        out["cluster"] = out["cluster_raw"] + 1
    else:
        out["cluster"] = out["cluster_raw"]

    out = out.dropna(subset=["cluster"]).copy()
    out["cluster"] = out["cluster"].astype(int)
    out["cluster_label"] = out["cluster"].map(lambda x: f"Cluster {x}")

    return out[["basin", "cluster", "cluster_label"]].drop_duplicates()


def load_all_test_period_metrics(pickles_dir: Path) -> pd.DataFrame:
    p = pickles_dir / "metrics_all_test_period.p"
    if not p.exists():
        raise FileNotFoundError(f"Missing pickle: {p}")

    obj = load_pickle(p)
    df = flatten_metrics_all_test_period(obj)
    df = canonicalize_metric_columns(df)

    basin_col = detect_basin_col(df)
    if basin_col != "basin":
        df = df.rename(columns={basin_col: "basin"})
    df["basin"] = df["basin"].map(zfill_basin)

    required = ["basin", "seed", "model", "metric", "value"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in flattened metrics table: {missing}")

    df["value"] = safe_numeric(df["value"])
    return df


def load_basin_list(basin_list_path: Path) -> list[str]:
    with open(basin_list_path, "r", encoding="utf-8") as f:
        basins = [line.strip() for line in f if line.strip()]
    return [zfill_basin(b) for b in basins]


def load_camels_attributes(attributes_dir: Path, basins: list[str] | None = None) -> pd.DataFrame:
    txt_files = sorted(attributes_dir.glob("camels_*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No CAMELS attribute files found in {attributes_dir}")

    dfs = []
    for txt_file in txt_files:
        if txt_file.name.lower() == "camels_attributes_v2.0.pdf":
            continue
        if txt_file.suffix.lower() != ".txt":
            continue
        df = pd.read_csv(txt_file, sep=";", dtype={"gauge_id": str})
        if "gauge_id" not in df.columns:
            continue
        df["gauge_id"] = df["gauge_id"].astype(str).str.zfill(8)
        df = df.set_index("gauge_id")
        dfs.append(df)

    out = pd.concat(dfs, axis=1)

    if basins is not None:
        basins = [zfill_basin(b) for b in basins]
        out = out.loc[out.index.intersection(basins)].copy()
        out = out.reindex(basins)

    return out


def load_basin_geo(project_root: Path) -> pd.DataFrame:
    """
    Priority:
      1) existing master file if available
      2) rebuild from 531_basin_list.txt + Train_Data/camels_attributes_v2.0
    """
    attr_file = project_root / "camels_531_master_attributes_clusters.csv"

    if attr_file.exists():
        df = pd.read_csv(attr_file)
        basin_col = detect_basin_col(df)
        lat_col = find_column(df, ["gauge_lat", "lat", "latitude"])
        lon_col = find_column(df, ["gauge_lon", "lon", "longitude"])
        if lat_col is not None and lon_col is not None:
            out = df[[basin_col, lat_col, lon_col]].copy()
            out.columns = ["basin", "lat", "lon"]
            out["basin"] = out["basin"].map(zfill_basin)
            out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
            out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
            out = out.dropna(subset=["lat", "lon"]).drop_duplicates("basin")
            if not out.empty:
                return out

    basin_list_path = project_root / "531_basin_list.txt"
    attributes_dir = project_root / "Train_Data" / "camels_attributes_v2.0"

    if not basin_list_path.exists():
        raise FileNotFoundError(f"Missing basin list file: {basin_list_path}")
    if not attributes_dir.exists():
        raise FileNotFoundError(f"Missing CAMELS attributes folder: {attributes_dir}")

    basins = load_basin_list(basin_list_path)
    attrs = load_camels_attributes(attributes_dir, basins=basins)
    attrs = attrs.reset_index().rename(columns={"index": "basin", "gauge_id": "basin"})

    lat_col = find_column(attrs, ["gauge_lat", "lat", "latitude"])
    lon_col = find_column(attrs, ["gauge_lon", "lon", "longitude"])
    basin_col = detect_basin_col(attrs)

    if lat_col is None or lon_col is None:
        raise ValueError(
            f"Could not detect latitude/longitude columns. Available columns: {list(attrs.columns)}"
        )

    out = attrs[[basin_col, lat_col, lon_col]].copy()
    out.columns = ["basin", "lat", "lon"]
    out["basin"] = out["basin"].map(zfill_basin)
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna(subset=["lat", "lon"]).drop_duplicates("basin")
    return out


# =========================================================
# DATA PREP
# =========================================================
def prepare_clusterwise_nse_df(project_root: Path, pickles_dir: Path) -> pd.DataFrame:
    clusters = load_clusters(project_root)
    metrics_df = load_all_test_period_metrics(pickles_dir)
    basin_geo = load_basin_geo(project_root)

    merged = metrics_df.merge(clusters, on="basin", how="inner")

    dd = merged[(merged["model"] == "Cluster-wise") & (merged["metric"] == PRIMARY_METRIC)].copy()
    if dd.empty:
        raise ValueError("No Cluster-wise NSE rows found.")

    basin_seedmean = (
        dd.groupby(["basin", "cluster", "cluster_label"], dropna=False)["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": PRIMARY_METRIC})
    )

    plot_df = basin_seedmean.merge(basin_geo, on="basin", how="left")
    plot_df = plot_df.dropna(subset=["lat", "lon", PRIMARY_METRIC]).copy()

    if plot_df.empty:
        raise ValueError("No valid rows after merging NSE and basin coordinates.")

    return plot_df


# =========================================================
# MAP HELPERS
# =========================================================
def load_us_boundary():
    if gpd is None:
        return None

    # geopandas built-in dataset
    try:
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        usa = world[world["name"] == "United States of America"].copy()
        if not usa.empty:
            return usa
    except Exception:
        pass

    return None


def iqr_inlier_mask(series: pd.Series, whisker: float = 1.5) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return pd.Series(True, index=series.index)
    lower = q1 - whisker * iqr
    upper = q3 + whisker * iqr
    return (series >= lower) & (series <= upper)


def get_local_zoom_df(sub: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      sub_in: inlier points used for local zoom / local display
      sub_out: excluded outlier points
    """
    mask_lon = iqr_inlier_mask(sub["lon"], whisker=1.5)
    mask_lat = iqr_inlier_mask(sub["lat"], whisker=1.5)
    mask = mask_lon & mask_lat

    sub_in = sub.loc[mask].copy()
    sub_out = sub.loc[~mask].copy()

    # fallback if too aggressive
    if len(sub_in) < max(8, int(0.6 * len(sub))):
        sub_in = sub.copy()
        sub_out = sub.iloc[0:0].copy()

    return sub_in, sub_out


def padded_extent(df: pd.DataFrame, lon_pad_frac: float = 0.10, lat_pad_frac: float = 0.10):
    lon_min = df["lon"].min()
    lon_max = df["lon"].max()
    lat_min = df["lat"].min()
    lat_max = df["lat"].max()

    dx = lon_max - lon_min
    dy = lat_max - lat_min

    dx = max(dx, 1.5)
    dy = max(dy, 1.2)

    lon_pad = dx * lon_pad_frac
    lat_pad = dy * lat_pad_frac

    return (
        lon_min - lon_pad,
        lon_max + lon_pad,
        lat_min - lat_pad,
        lat_max + lat_pad,
    )


def add_panel_frame(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("0.35")


def add_card_frame(ax, pad=0.008, rounding=0.018, facecolor="#fcfcfc", edgecolor="0.60", lw=1.1):
    rect = FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes,
                          boxstyle=f"round,pad={pad},rounding_size={rounding}",
                          facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=-10)
    ax.add_patch(rect)


# =========================================================
# FIGURE
# =========================================================
def make_clusterwise_geo_figure(plot_df: pd.DataFrame, output_path: Path) -> None:
    usa = load_us_boundary()

    nse_vmin = float(plot_df[PRIMARY_METRIC].min())
    nse_vmax = float(plot_df[PRIMARY_METRIC].max())
    if np.isclose(nse_vmin, nse_vmax):
        nse_vmax = nse_vmin + 1e-6

    fig = plt.figure(figsize=(15.2, 19.2))
    gs = fig.add_gridspec(6, 2, height_ratios=[0.62, 2.45, 0.62, 2.45, 0.62, 2.45],
                          hspace=0.28, wspace=0.14, left=0.06, right=0.90, top=0.955, bottom=0.05)

    table_axes = [fig.add_subplot(gs[r, :]) for r in [0, 2, 4]]
    for a in table_axes:
        a.set_axis_off()
        add_card_frame(a, pad=0.006, rounding=0.018, facecolor="#f7f8fa", edgecolor="0.60", lw=1.0)
    map_axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
                fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1]),
                fig.add_subplot(gs[5, 0]), fig.add_subplot(gs[5, 1])]

    all_points = plot_df.copy()
    last_sc = None
    total_excluded = 0
    summaries = []

    for cl, ax in zip(sorted(plot_df["cluster"].unique()), map_axes):
        sub = plot_df[plot_df["cluster"] == cl].copy()
        sub_in, sub_out = get_local_zoom_df(sub)
        total_excluded += len(sub_out)
        ex0, ex1, ey0, ey1 = padded_extent(sub_in, lon_pad_frac=0.12, lat_pad_frac=0.12)

        if usa is not None:
            try:
                usa.boundary.plot(ax=ax, color="0.72", linewidth=0.50, zorder=0)
            except Exception:
                pass

        ax.scatter(all_points["lon"], all_points["lat"], s=6, color="0.88", edgecolors="none", alpha=0.65, zorder=1)
        sc = ax.scatter(sub_in["lon"], sub_in["lat"], c=sub_in[PRIMARY_METRIC], cmap="cividis",
                        vmin=nse_vmin, vmax=nse_vmax, s=28, edgecolors="black", linewidths=0.26, alpha=0.96, zorder=4)
        last_sc = sc

        med = float(sub[PRIMARY_METRIC].median())
        mean_ = float(sub[PRIMARY_METRIC].mean())
        n = len(sub)
        summaries.append((cl, n, med, mean_))

        ax.set_xlim(ex0, ex1)
        ax.set_ylim(ey0, ey1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        add_card_frame(ax, pad=0.005, rounding=0.015, facecolor="#fcfcfc", edgecolor="0.60", lw=1.0)
        add_panel_frame(ax)
        ax.set_title(f"Cluster {cl}", fontsize=16, fontweight="bold", pad=5)

    def add_summary_table(ax_tbl, rows):
        ax_tbl.axis("off")
        col_labels = [f"Cluster {r[0]}" for r in rows]
        cell_text = [
            [f"n = {r[1]}" for r in rows],
            [f"median = {r[2]:.3f}" for r in rows],
            [f"mean = {r[3]:.3f}" for r in rows],
        ]
        row_labels = ["Sample", "Median NSE", "Mean NSE"]
        tbl = ax_tbl.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellLoc="center",
            rowLoc="center",
            bbox=[0.20, 0.05, 0.60, 0.90],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(15.5)
        tbl.scale(1.0, 1.34)

        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("0.35")
            cell.set_linewidth(0.9)
            if r == 0:  # column headers
                cell.set_facecolor("#e8ecef")
                cell.get_text().set_fontweight("bold")
            elif c == -1:  # row-label column
                cell.set_facecolor("#f1f3f5")
                cell.get_text().set_fontweight("bold")
            else:
                if r % 2 == 1:
                    cell.set_facecolor("#fbfbfb")
                else:
                    cell.set_facecolor("white")
        return tbl

    add_summary_table(table_axes[0], summaries[0:2])
    add_summary_table(table_axes[1], summaries[2:4])
    add_summary_table(table_axes[2], summaries[4:6])

    cax = fig.add_axes([0.92, 0.24, 0.018, 0.50])
    cbar = fig.colorbar(last_sc, cax=cax)
    cbar.set_label(f"{PRIMARY_METRIC} (basin seed-mean)", fontsize=14, fontweight="bold")
    cbar.ax.tick_params(labelsize=13)

    fig.suptitle(f"Geographic distribution of basin-wise {PRIMARY_METRIC} for the cluster-wise family benchmark",
                 fontsize=20.5, fontweight="bold", y=0.983)
    fig.text(0.06, 0.02,
             f"The six panels show local zooms colored by basin-wise seed-averaged NSE; isolated spatial outliers were excluded only from the display windows (excluded basins across clusters: n = {total_excluded}).",
             fontsize=11.0, ha="left")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# MAIN
# =========================================================
def main():
    project_root = get_project_root()
    pickles_dir = get_pickles_dir()
    output_dir = ensure_dir(project_root / OUTPUT_SUBDIR)

    plot_df = prepare_clusterwise_nse_df(project_root, pickles_dir)

    out_png = output_dir / "fig_clusterwise_geo_nse_journal.png"
    make_clusterwise_geo_figure(plot_df, out_png)

    print(f"[OK] Saved figure to: {out_png}")


if __name__ == "__main__":
    main()