from __future__ import annotations

"""
cluster_performance_summary_v4.py

Cluster-wise performance summary aligned with the existing CAMELS-US codebase.

New in v4:
- keeps the cluster-summary tables and heatmaps from v3
- adds a geographic visualization of basin-wise NSE for the Cluster-wise model
- the geographic figure is a 2x3 panel map (one panel per cluster)
  showing basin locations colored by seed-mean NSE

It assumes the standard project structure already used by:
    make_07_summary_tables.py
and the helper functions available in camels_utils.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
TARGET_METRICS = [
    "NSE",
    "KGE",
    "Pearson-r",
    "RMSE",
    "Peak-timing",
    "Peak-MAPE",
    "Missed-peaks",
]

SUMMARY_FUNCS = {
    "mean": np.mean,
    "median": np.median,
    "q25": lambda x: np.percentile(x, 25),
    "q75": lambda x: np.percentile(x, 75),
    "min": np.min,
    "max": np.max,
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
    Load basin coordinates using the same logic as the reference hydrology-summary code.
    Priority:
      1) existing master file if already created
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
        raise FileNotFoundError(
            f"Missing basin list file: {basin_list_path}. "
            "This file is needed to restrict attributes to the 531 study basins."
        )
    if not attributes_dir.exists():
        raise FileNotFoundError(
            f"Missing CAMELS attributes folder: {attributes_dir}."
        )

    basins = load_basin_list(basin_list_path)
    attrs = load_camels_attributes(attributes_dir, basins=basins)
    attrs = attrs.reset_index().rename(columns={"index": "basin", "gauge_id": "basin"})

    lat_col = find_column(attrs, ["gauge_lat", "lat", "latitude"])
    lon_col = find_column(attrs, ["gauge_lon", "lon", "longitude"])
    basin_col = detect_basin_col(attrs)

    if lat_col is None or lon_col is None:
        raise ValueError(
            f"Could not detect latitude/longitude columns in CAMELS attributes. Available columns: {list(attrs.columns)}"
        )

    out = attrs[[basin_col, lat_col, lon_col]].copy()
    out.columns = ["basin", "lat", "lon"]
    out["basin"] = out["basin"].map(zfill_basin)
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna(subset=["lat", "lon"]).drop_duplicates("basin")
    return out


# =========================================================
# SUMMARIES
# =========================================================
def summarize_by_cluster(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, cluster, cluster_label, metric), grp in long_df.groupby(
        ["model", "cluster", "cluster_label", "metric"], dropna=False
    ):
        x = grp["value"].dropna().to_numpy(dtype=float)
        row = {
            "model": model,
            "cluster": cluster,
            "cluster_label": cluster_label,
            "metric": metric,
            "n_rows": len(x),
            "n_basins": int(grp["basin"].nunique()),
        }
        if len(x) == 0:
            for stat in SUMMARY_FUNCS:
                row[stat] = np.nan
        else:
            for stat_name, fn in SUMMARY_FUNCS.items():
                row[stat_name] = float(fn(x))
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["metric", "cluster", "model"]).reset_index(drop=True)


def summarize_basin_seedmean_by_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """
    First average over seeds within each basin, then summarize across basins.
    This is usually the best main-text summary for regional comparisons.
    """
    basin_seedmean = (
        df.groupby(["basin", "model", "metric", "cluster", "cluster_label"], dropna=False)["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "basin_seedmean"})
    )

    rows = []
    for (model, cluster, cluster_label, metric), grp in basin_seedmean.groupby(
        ["model", "cluster", "cluster_label", "metric"], dropna=False
    ):
        x = grp["basin_seedmean"].dropna().to_numpy(dtype=float)
        row = {
            "model": model,
            "cluster": cluster,
            "cluster_label": cluster_label,
            "metric": metric,
            "n_basins": int(grp["basin"].nunique()),
        }
        if len(x) == 0:
            for stat in SUMMARY_FUNCS:
                row[stat] = np.nan
        else:
            for stat_name, fn in SUMMARY_FUNCS.items():
                row[stat_name] = float(fn(x))
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["metric", "cluster", "model"]).reset_index(drop=True)


# =========================================================
# FIGURES
# =========================================================
def make_nse_boxplot(df: pd.DataFrame, output_dir: Path) -> None:
    dd = df[df["metric"] == PRIMARY_METRIC].copy()
    if dd.empty:
        return

    basin_seedmean = (
        dd.groupby(["basin", "model", "cluster"], dropna=False)["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "basin_seedmean"})
    )

    clusters = sorted(basin_seedmean["cluster"].unique().tolist())
    models = [m for m in ["Benchmark", "Top10", "Cluster-wise"] if m in basin_seedmean["model"].unique()]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes = axes.flatten()

    for idx, cl in enumerate(clusters):
        ax = axes[idx]
        sub = basin_seedmean[basin_seedmean["cluster"] == cl]
        data = [sub.loc[sub["model"] == m, "basin_seedmean"].dropna().to_numpy() for m in models]
        ax.boxplot(data, tick_labels=models, showfliers=False, widths=0.5)
        ax.set_title(f"Cluster {cl}")
        ax.set_ylabel(PRIMARY_METRIC)
        ax.grid(True, axis="y", alpha=0.3)
        for t in ax.get_xticklabels():
            t.set_rotation(15)

    for j in range(len(clusters), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{PRIMARY_METRIC} across basins within each cluster (mean over seeds)", fontsize=16)
    fig.savefig(output_dir / "fig_cluster_nse_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_nse_mean_median_plot(summary_df: pd.DataFrame, output_dir: Path) -> None:
    dd = summary_df[summary_df["metric"] == PRIMARY_METRIC].copy()
    if dd.empty:
        return

    models = [m for m in ["Benchmark", "Top10", "Cluster-wise"] if m in dd["model"].unique()]
    clusters = sorted(dd["cluster"].unique().tolist())
    x = np.arange(len(clusters))
    width = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for ax, stat in zip(axes, ["median", "mean"]):
        for i, m in enumerate(models):
            sub = dd[dd["model"] == m].sort_values("cluster")
            ax.bar(x + (i - 1) * width, sub[stat].to_numpy(), width=width, label=m)
        ax.set_xticks(x)
        ax.set_xticklabels([f"C{c}" for c in clusters])
        ax.set_title(f"{PRIMARY_METRIC} {stat} by cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel(PRIMARY_METRIC)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()

    fig.savefig(output_dir / "fig_cluster_nse_median_mean.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _get_heatmap_models(dd: pd.DataFrame) -> list[str]:
    return [m for m in ["Benchmark", "Top10", "Cluster-wise"] if m in dd["model"].unique()]


def _build_heatmap_pivot(summary_df: pd.DataFrame, metric: str, stat: str) -> pd.DataFrame:
    dd = summary_df[summary_df["metric"] == metric].copy()
    models = _get_heatmap_models(dd)
    clusters = sorted(dd["cluster"].unique().tolist())
    return (
        dd.pivot(index="cluster_label", columns="model", values=stat)
        .reindex(index=[f"Cluster {c}" for c in clusters], columns=models)
    )


def _adaptive_limits(mat: np.ndarray) -> tuple[float, float]:
    vals = mat[np.isfinite(mat)]
    if vals.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if np.isclose(vmin, vmax):
        eps = max(1e-6, abs(vmin) * 0.05 + 1e-6)
        vmin -= eps
        vmax += eps
    return vmin, vmax


def _annot_color(val: float, vmin: float, vmax: float) -> str:
    if not np.isfinite(val):
        return "black"
    mid = (vmin + vmax) / 2.0
    return "white" if val < mid else "black"


def make_metric_heatmap(summary_df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    dd = summary_df[summary_df["metric"] == metric].copy()
    if dd.empty:
        return

    pivot = _build_heatmap_pivot(summary_df, metric, "median")
    mat = pivot.to_numpy(dtype=float)
    vmin, vmax = _adaptive_limits(mat)

    fig, ax = plt.subplots(figsize=(6, 4.8), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax, cmap="cividis", interpolation="nearest")
    ax.grid(False)
    ax.tick_params(which="both", length=0)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=15)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"{metric} median by cluster and model")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=_annot_color(mat[i, j], vmin, vmax),
                )

    fig.colorbar(im, ax=ax, label=f"{metric} median")
    fig.savefig(output_dir / f"fig_cluster_metric_heatmap_{metric}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_metric_heatmap_2x2(summary_df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    dd = summary_df[summary_df["metric"] == metric].copy()
    if dd.empty:
        return

    stat_specs = [
        ("median", f"{metric} median"),
        ("mean", f"{metric} mean"),
        ("min", f"{metric} minimum"),
        ("max", f"{metric} maximum"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5), constrained_layout=True)
    axes = axes.flatten()

    for ax, (stat, title) in zip(axes, stat_specs):
        pivot = _build_heatmap_pivot(summary_df, metric, stat)
        mat = pivot.to_numpy(dtype=float)
        vmin, vmax = _adaptive_limits(mat)
        im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax, cmap="cividis", interpolation="nearest")
        ax.grid(False)
        ax.tick_params(which="both", length=0)

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=15, fontweight="bold")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(title, fontsize=18, fontweight="bold")

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{mat[i, j]:.3f}",
                        ha="center",
                        va="center",
                        fontsize=12.5,
                        color=_annot_color(mat[i, j], vmin, vmax),
                    )

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=11)
        ax.grid(color="0.80", linewidth=0.35, alpha=0.6)

    fig.suptitle(f"Cluster-level summary of basin-wise {metric} for the three benchmark frameworks", fontsize=19, fontweight="bold")
    fig.savefig(output_dir / f"fig_cluster_metric_heatmap_2x2_{metric}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _load_us_basemap() -> object | None:
    if gpd is None:
        return None
    try:
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        usa = world[world["name"] == "United States of America"].copy()
        return usa
    except Exception:
        return None


def _iqr_bounds(x: np.ndarray, k: float = 3.0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return -np.inf, np.inf
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    if np.isclose(iqr, 0):
        return float(np.min(x)), float(np.max(x))
    return float(q1 - k * iqr), float(q3 + k * iqr)



def _filter_spatial_outliers(df: pd.DataFrame, k: float = 3.0) -> pd.DataFrame:
    """
    Remove a very small number of geographically isolated basins within a cluster.
    The filtering is intentionally conservative and is only meant to discard clear
    spatial outliers that would otherwise force overly large empty panel extents.
    """
    if df.empty or len(df) < 8:
        return df.copy()

    lo_lon, hi_lon = _iqr_bounds(df["lon"].to_numpy(dtype=float), k=k)
    lo_lat, hi_lat = _iqr_bounds(df["lat"].to_numpy(dtype=float), k=k)
    mask = (
        df["lon"].between(lo_lon, hi_lon, inclusive="both")
        & df["lat"].between(lo_lat, hi_lat, inclusive="both")
    )

    # If the filter is too aggressive, progressively relax it.
    keep_rate = float(mask.mean()) if len(mask) else 1.0
    if keep_rate < 0.80:
        lo_lon, hi_lon = _iqr_bounds(df["lon"].to_numpy(dtype=float), k=4.5)
        lo_lat, hi_lat = _iqr_bounds(df["lat"].to_numpy(dtype=float), k=4.5)
        mask = (
            df["lon"].between(lo_lon, hi_lon, inclusive="both")
            & df["lat"].between(lo_lat, hi_lat, inclusive="both")
        )

    out = df.loc[mask].copy()
    return out if not out.empty else df.copy()



def _padded_extent(df: pd.DataFrame, pad_frac: float = 0.10, min_lon_pad: float = 1.0, min_lat_pad: float = 0.8) -> tuple[tuple[float, float], tuple[float, float]]:
    lon_min = float(df["lon"].min())
    lon_max = float(df["lon"].max())
    lat_min = float(df["lat"].min())
    lat_max = float(df["lat"].max())

    lon_span = max(lon_max - lon_min, 1e-6)
    lat_span = max(lat_max - lat_min, 1e-6)

    lon_pad = max(min_lon_pad, lon_span * pad_frac)
    lat_pad = max(min_lat_pad, lat_span * pad_frac)

    return (lon_min - lon_pad, lon_max + lon_pad), (lat_min - lat_pad, lat_max + lat_pad)



def make_clusterwise_geo_nse_map(merged: pd.DataFrame, basin_geo: pd.DataFrame, output_dir: Path) -> None:
    """
    Create a professional 7-panel geographic figure for the Cluster-wise model:
      - top row: one full-width CONUS map with all basins colored by cluster
      - lower rows: 6 cluster-specific subplots (3x2 layout), each zoomed to the
        spatial footprint of that cluster and colored by basin-wise seed-mean NSE.

    A small number of geographically isolated outliers are removed *within each cluster*
    before the local zoom is computed, which avoids excessively large empty panel areas.
    """
    dd = merged[(merged["model"] == "Cluster-wise") & (merged["metric"] == PRIMARY_METRIC)].copy()
    if dd.empty:
        return

    basin_seedmean = (
        dd.groupby(["basin", "cluster", "cluster_label"], dropna=False)["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": PRIMARY_METRIC})
    )

    plot_df = basin_seedmean.merge(basin_geo, on="basin", how="left")
    plot_df = plot_df.dropna(subset=["lat", "lon", PRIMARY_METRIC]).copy()
    if plot_df.empty:
        return

    usa = _load_us_basemap()
    clusters = sorted(plot_df["cluster"].unique().tolist())

    # Filter clearly isolated spatial outliers within each cluster.
    kept_parts = []
    removed_count = 0
    for cl in clusters:
        sub = plot_df[plot_df["cluster"] == cl].copy()
        sub_keep = _filter_spatial_outliers(sub, k=3.0)
        removed_count += max(0, len(sub) - len(sub_keep))
        kept_parts.append(sub_keep)

    plot_df = pd.concat(kept_parts, ignore_index=True)
    if plot_df.empty:
        return

    # Shared NSE range for the six lower panels.
    nse_vmin = float(plot_df[PRIMARY_METRIC].min())
    nse_vmax = float(plot_df[PRIMARY_METRIC].max())
    if np.isclose(nse_vmin, nse_vmax):
        nse_vmax = nse_vmin + 1e-6

    overall_xlim, overall_ylim = _padded_extent(
        plot_df,
        pad_frac=0.06,
        min_lon_pad=2.0,
        min_lat_pad=1.5,
    )

    cluster_cmap = plt.get_cmap("tab10")

    fig = plt.figure(figsize=(15.5, 13.5), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.25, 1.0, 1.0, 1.0])

    # -------------------------------------------------
    # Top full-width overview panel
    # -------------------------------------------------
    ax_top = fig.add_subplot(gs[0, :])
    if usa is not None:
        try:
            usa.boundary.plot(ax=ax_top, color="0.72", linewidth=0.8, zorder=1)
        except Exception:
            pass

    handles = []
    labels = []
    for idx, cl in enumerate(clusters):
        sub = plot_df[plot_df["cluster"] == cl].copy()
        color = cluster_cmap(idx % 10)
        sc = ax_top.scatter(
            sub["lon"],
            sub["lat"],
            color=color,
            s=22,
            edgecolors="black",
            linewidths=0.25,
            alpha=0.90,
            zorder=3,
        )
        handles.append(sc)
        labels.append(f"Cluster {cl}")

        # Centroid label for a cleaner high-level overview.
        cx = float(sub["lon"].median())
        cy = float(sub["lat"].median())
        ax_top.text(
            cx,
            cy,
            f"C{cl}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="0.55", alpha=0.85),
            zorder=4,
        )

    ax_top.set_xlim(*overall_xlim)
    ax_top.set_ylim(*overall_ylim)
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    for spine in ax_top.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("0.5")

    ax_top.set_title("CONUS overview: spatial distribution of the six basin clusters", fontsize=14, pad=8)
    ax_top.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.00),
        frameon=True,
        title="Clusters",
        borderaxespad=0.0,
    )

    # -------------------------------------------------
    # Lower 6 cluster panels (local zooms)
    # -------------------------------------------------
    bottom_axes = []
    last_sc = None
    all_xy = plot_df[["lon", "lat"]].copy()

    for idx, cl in enumerate(clusters[:6]):
        r = 1 + idx // 2
        c = idx % 2
        ax = fig.add_subplot(gs[r, c])
        bottom_axes.append(ax)

        sub = plot_df[plot_df["cluster"] == cl].copy()
        if sub.empty:
            ax.axis("off")
            continue

        local_xlim, local_ylim = _padded_extent(
            sub,
            pad_frac=0.12,
            min_lon_pad=0.6,
            min_lat_pad=0.5,
        )

        # Light context cloud from all kept basins, then highlight current cluster.
        ax.scatter(
            all_xy["lon"],
            all_xy["lat"],
            color="0.85",
            s=10,
            linewidths=0.0,
            alpha=0.35,
            zorder=1,
        )

        if usa is not None:
            try:
                usa.boundary.plot(ax=ax, color="0.82", linewidth=0.45, zorder=0)
            except Exception:
                pass

        sc = ax.scatter(
            sub["lon"],
            sub["lat"],
            c=sub[PRIMARY_METRIC],
            cmap="viridis",
            vmin=nse_vmin,
            vmax=nse_vmax,
            s=30,
            edgecolors="black",
            linewidths=0.28,
            alpha=0.98,
            zorder=3,
        )
        last_sc = sc

        med = float(sub[PRIMARY_METRIC].median())
        mean_ = float(sub[PRIMARY_METRIC].mean())
        n_b = int(sub["basin"].nunique())

        ax.set_title(
            f"Cluster {cl}  |  n = {n_b}  |  median = {med:.3f}  |  mean = {mean_:.3f}",
            fontsize=10.5,
            pad=4,
        )
        ax.set_xlim(*local_xlim)
        ax.set_ylim(*local_ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.65)
            spine.set_color("0.55")

    if last_sc is not None and bottom_axes:
        cbar = fig.colorbar(last_sc, ax=bottom_axes, fraction=0.025, pad=0.02, shrink=0.95)
        cbar.set_label(f"{PRIMARY_METRIC} (basin seed-mean)")

    fig.suptitle(
        f"Geographic distribution of basin-wise {PRIMARY_METRIC} for the Cluster-wise model",
        fontsize=18,
        y=0.995,
    )

    note = "Lower panels are locally zoomed to the cluster footprint after excluding a small number of isolated spatial outliers."
    if removed_count > 0:
        note += f" Excluded basins for local zooming: n = {removed_count}."
    fig.text(0.5, 0.006, note, ha="center", va="bottom", fontsize=9, color="0.35")

    fig.savefig(output_dir / "fig_clusterwise_geo_composite.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig_clusterwise_geo_nse.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig_clusterwise_geo_clusters.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# MAIN
# =========================================================
def main():
    project_root = get_project_root()
    pickles_dir = get_pickles_dir()
    output_dir = ensure_dir(project_root / "Results" / "Tables_cluster")

    clusters = load_clusters(project_root)
    metrics_df = load_all_test_period_metrics(pickles_dir)
    basin_geo = load_basin_geo(project_root)

    merged = metrics_df.merge(clusters, on="basin", how="inner")

    cluster_raw_summary = summarize_by_cluster(merged)
    cluster_basin_summary = summarize_basin_seedmean_by_cluster(merged)

    cluster_raw_summary.to_csv(output_dir / "cluster_metric_summary_raw.csv", index=False)
    cluster_basin_summary.to_csv(output_dir / "cluster_metric_summary_basin_seedmean.csv", index=False)

    wide_df = cluster_basin_summary.pivot_table(
        index=["cluster", "cluster_label", "metric"],
        columns="model",
        values=["mean", "median", "q25", "q75", "min", "max"],
    )
    wide_df.to_csv(output_dir / "cluster_metric_summary_basin_seedmean_wide.csv")

    with pd.ExcelWriter(output_dir / "cluster_metric_summary.xlsx") as writer:
        cluster_raw_summary.to_excel(writer, sheet_name="cluster_raw", index=False)
        cluster_basin_summary.to_excel(writer, sheet_name="cluster_basin_seedmean", index=False)
        wide_df.to_excel(writer, sheet_name="cluster_basin_seedmean_wide")

    make_nse_boxplot(merged, output_dir)
    make_nse_mean_median_plot(cluster_basin_summary, output_dir)
    make_metric_heatmap(cluster_basin_summary, PRIMARY_METRIC, output_dir)
    make_metric_heatmap_2x2(cluster_basin_summary, PRIMARY_METRIC, output_dir)
    make_clusterwise_geo_nse_map(merged, basin_geo, output_dir)

    print(f"[OK] Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
