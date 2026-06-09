from __future__ import annotations

"""
CAMELS-US hydrology summary builder for the paper section:
"Study area and dataset (CAMELS-US)"

What it does
------------
1) Loads the 531 study basins.
2) Merges CAMELS attributes v2.0 + cluster assignments.
3) Produces concise, hydrologically meaningful tables.
4) Produces professional maps and summary figures.
5) Writes all outputs to: path/CAMELS_US_hydrology

Expected folder structure
-------------------------
Notebook: path/codes
misc:     path/misc
clusters: path/CAMELS_US_Clusters.xlsx
basins:   path/531_basin_list.txt
attrs:    path/Train_Data/camels_attributes_v2.0

Usage
-----
Edit ROOT_PATH below or pass it to main(root_path=...).
Then run the full script.
"""

import os
import warnings
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_OK = True
except Exception:
    CARTOPY_OK = False

warnings.filterwarnings("ignore")

# Optional geospatial imports
try:
    import geopandas as gpd
    from shapely.geometry import box
    GEOPANDAS_OK = True
except Exception:
    GEOPANDAS_OK = False


# =========================
# User config
# =========================
ROOT_PATH = Path(r"F:\Experiments\CAMELS_US\Clean_4_upload")  # <-- CHANGE THIS
OUTPUT_FOLDER_NAME = "CAMELS_US_hydrology"


# =========================
# Hydrology-aware variable sets
# =========================
KEY_VARS_TABLE = [
    "area_gages2",          # catchment area [km2]
    "elev_mean",           # mean elevation [m]
    "slope_mean",          # mean slope [m/km or dataset native]
    "p_mean",              # mean precipitation
    "aridity",             # PET / P type index in CAMELS docs
    "frac_snow",           # snow fraction
    "q_mean",              # mean discharge [mm/day]
    "runoff_ratio",        # runoff / precip
    "baseflow_index",      # storage-groundwater signal
    "slope_fdc",           # flow-duration curve slope
    "high_q_freq",         # high-flow frequency
    "low_q_freq",          # low-flow frequency
]

PROFILE_VARS = [
    "p_mean",
    "aridity",
    "frac_snow",
    "elev_mean",
    "slope_mean",
    "area_gages2",
    "q_mean",
    "runoff_ratio",
    "baseflow_index",
    "slope_fdc",
    "high_q_freq",
    "low_q_freq",
]

DIST_VARS_HYDRO = [
    "runoff_ratio",
    "baseflow_index",
    "slope_fdc",
    "q_mean",
    "high_q_freq",
    "low_q_freq",
]

DIST_VARS_CLIMATE = [
    "p_mean",
    "aridity",
    "frac_snow",
    "elev_mean",
    "slope_mean",
    "area_gages2",
]

PRETTY_LABELS = {
    "area_gages2": "Area [km²]",
    "elev_mean": "Mean elevation [m]",
    "slope_mean": "Mean slope",
    "p_mean": "Mean precipitation",
    "aridity": "Aridity index",
    "frac_snow": "Snow fraction",
    "q_mean": "Mean discharge [mm d⁻¹]",
    "runoff_ratio": "Runoff ratio",
    "baseflow_index": "Baseflow index",
    "slope_fdc": "FDC slope",
    "high_q_freq": "High-flow freq.",
    "low_q_freq": "Low-flow freq.",
}


# =========================
# IO helpers
# =========================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_basin_list(basin_list_path: Path) -> List[str]:
    with open(basin_list_path, "r", encoding="utf-8") as f:
        basins = [line.strip() for line in f if line.strip()]
    basins = [b.zfill(8) for b in basins]
    return basins


def load_camels_attributes(attributes_dir: Path, basins: Optional[Iterable[str]] = None) -> pd.DataFrame:
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
        basins = [str(b).zfill(8) for b in basins]

        # Explicit filter: keep only the 531 study basins
        out = out.loc[out.index.intersection(basins)].copy()

        # Reorder strictly to the basin list order
        out = out.reindex(basins)

        # Hard safety check
        if len(out) != len(basins):
            raise ValueError(
                f"Filtered attributes dataframe has {len(out)} rows, but basin list has {len(basins)} basins."
            )

    if "huc_02" in out.columns:
        out["huc_02"] = pd.to_numeric(out["huc_02"], errors="coerce").astype("Int64")
        out["huc_str"] = out["huc_02"].astype(str).str.replace("<NA>", "", regex=False).str.zfill(2)

    return out


def load_clusters(cluster_xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(cluster_xlsx)
    basin_col = [c for c in df.columns if c.lower() in ["basin", "gauge_id", "gauge"]]
    cluster_col = [c for c in df.columns if c.lower() == "cluster"]
    if not basin_col or not cluster_col:
        raise ValueError("Cluster file must contain basin and cluster columns.")

    basin_col = basin_col[0]
    cluster_col = cluster_col[0]

    out = df[[basin_col, cluster_col]].copy()
    out.columns = ["gauge_id", "cluster_raw"]
    out["gauge_id"] = out["gauge_id"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)

    # Your file uses 0..5; convert to 1..6 for paper readability.
    out["cluster"] = out["cluster_raw"].astype(int) + 1
    return out[["gauge_id", "cluster"]]


def build_master_dataframe(root_path: Path) -> tuple[pd.DataFrame, Path]:
    output_dir = root_path / OUTPUT_FOLDER_NAME
    ensure_dir(output_dir)

    basin_list_path = root_path / "531_basin_list.txt"
    cluster_path = root_path / "CAMELS_US_Clusters.xlsx"
    attributes_dir = root_path / "Train_Data" / "camels_attributes_v2.0"
    name_file_path = attributes_dir / "camels_name.txt"

    basins = load_basin_list(basin_list_path)
    attrs = load_camels_attributes(attributes_dir, basins=basins)
    clusters = load_clusters(cluster_path)
    names = load_basin_names(name_file_path, basins=basins)

    df = attrs.reset_index().rename(columns={"index": "gauge_id"})
    df = df.merge(clusters, on="gauge_id", how="left")
    df = df.merge(names[["gauge_id", "station_name"]], on="gauge_id", how="left")

    # Hard guarantee: final dataframe is only the 531 study basins
    df = df[df["gauge_id"].isin(basins)].copy()
    df["gauge_id"] = pd.Categorical(df["gauge_id"], categories=basins, ordered=True)
    df = df.sort_values("gauge_id").reset_index(drop=True)
    df["gauge_id"] = df["gauge_id"].astype(str)

    if len(df) != len(basins):
        raise ValueError(f"Master dataframe has {len(df)} rows, expected {len(basins)} study basins.")

    if df["cluster"].isna().any():
        missing = df.loc[df["cluster"].isna(), "gauge_id"].tolist()
        raise ValueError(f"{len(missing)} study basins are missing cluster assignments: {missing[:10]}")

    if df["station_name"].isna().any():
        missing = df.loc[df["station_name"].isna(), "gauge_id"].tolist()
        print(f"Warning: {len(missing)} study basins are missing station names: {missing[:10]}")

    df["cluster"] = pd.to_numeric(df["cluster"], errors="raise").astype(int)
    df["cluster_label"] = df["cluster"].map(lambda x: f"Cluster {x}")

    if "area_gages2" in df.columns:
        df["log_area_gages2"] = np.log10(pd.to_numeric(df["area_gages2"], errors="coerce").clip(lower=1e-6))

    preferred_front = ["gauge_id", "station_name", "cluster", "cluster_label"]
    other_cols = [c for c in df.columns if c not in preferred_front]
    df = df[preferred_front + other_cols]

    df.to_csv(output_dir / "camels_531_master_attributes_clusters.csv", index=False)
    return df, output_dir


# =========================
# Tables
# =========================
def summarize_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return pd.Series({
        "n": s.notna().sum(),
        "mean": s.mean(),
        "median": s.median(),
        "std": s.std(),
        "q25": s.quantile(0.25),
        "q75": s.quantile(0.75),
        "min": s.min(),
        "max": s.max(),
    })


def make_overall_summary_table(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for var in KEY_VARS_TABLE:
        tmp = summarize_series(df[var])
        tmp.name = var
        rows.append(tmp)
    out = pd.DataFrame(rows)
    out.index.name = "variable"
    out.insert(0, "label", [PRETTY_LABELS.get(v, v) for v in out.index])
    out = out.reset_index()
    out.to_csv(output_dir / "table_overall_hydrology_summary.csv", index=False)
    return out


def make_cluster_summary_table(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    pieces = []
    for var in KEY_VARS_TABLE:
        g = df.groupby("cluster")[var].apply(summarize_series).unstack()
        g.insert(0, "variable", var)
        g.insert(1, "label", PRETTY_LABELS.get(var, var))
        pieces.append(g.reset_index())
    out = pd.concat(pieces, axis=0, ignore_index=True)
    out.to_csv(output_dir / "table_cluster_hydrology_summary_long.csv", index=False)
    return out


def make_cluster_core_table(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Cleaner compact table for the paper: medians only, one row per cluster.
    """
    core_vars = [
        "area_gages2", "elev_mean", "p_mean", "aridity",
        "frac_snow", "q_mean", "runoff_ratio", "baseflow_index",
        "slope_fdc", "high_q_freq", "low_q_freq"
    ]
    out = df.groupby("cluster")[core_vars].median().round(2)
    out.insert(0, "n_basins", df.groupby("cluster").size())
    out = out.reset_index()
    out.to_csv(output_dir / "table_cluster_core_medians.csv", index=False)
    return out


def make_cluster_categorical_table(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for cluster, sub in df.groupby("cluster"):
        row = {
            "cluster": int(cluster),
            "n_basins": len(sub),
            "dominant_geology": sub["geol_1st_class"].mode(dropna=True).iloc[0] if sub["geol_1st_class"].notna().any() else np.nan,
            "dominant_land_cover": sub["dom_land_cover"].mode(dropna=True).iloc[0] if sub["dom_land_cover"].notna().any() else np.nan,
            "dominant_huc02": ", ".join(sub["huc_str"].value_counts().head(3).index.tolist()) if "huc_str" in sub else np.nan,
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "table_cluster_categorical_context.csv", index=False)
    return out


# =========================
# Plot helpers
# =========================
def setup_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 15,
        "font.family": "sans-serif",
        "font.sans-serif": ["Calibri", "Arial", "Liberation Sans", "DejaVu Sans"],
    })


def get_cluster_palette(df: pd.DataFrame, as_str: bool = False):
    clusters = sorted([int(c) for c in pd.to_numeric(df["cluster"], errors="coerce").dropna().astype(int).unique()])
    colors = sns.color_palette("tab10", n_colors=max(6, len(clusters)))
    if as_str:
        return {str(c): colors[i] for i, c in enumerate(clusters)}
    return {c: colors[i] for i, c in enumerate(clusters)}


def load_huc02_boundaries(root_path: Path):
    if not GEOPANDAS_OK:
        return None

    gdb_path = root_path / "misc" / "shapefiles" / "huc_02" / "huc_02_simplified.gdb"
    if not gdb_path.exists():
        return None

    try:
        import fiona
        layers = fiona.listlayers(gdb_path)
        layer = layers[0]
        gdf = gpd.read_file(gdb_path, layer=layer)
        return gdf.to_crs(epsg=4326)
    except Exception as e:
        print(f"Could not read HUC02 boundaries: {e}")
        return None


def plot_cluster_map(df: pd.DataFrame, root_path: Path, output_dir: Path) -> None:
    setup_plot_style()
    palette = get_cluster_palette(df)
    plot_base = df.dropna(subset=["gauge_lon", "gauge_lat", "cluster"]).copy()

    if CARTOPY_OK:
        data_crs = ccrs.PlateCarree()
        map_crs = ccrs.LambertConformal(central_longitude=-96, central_latitude=39, standard_parallels=(33, 45))
        fig = plt.figure(figsize=(15.8, 9.7))
        ax = plt.axes(projection=map_crs)
        ax.set_extent((-125.2, -66.3, 24.0, 50.0), crs=data_crs)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f6f6f6", zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#efefef", zorder=0)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=1.1, edgecolor="#333333", zorder=1)
        states = cfeature.NaturalEarthFeature("cultural", "admin_1_states_provinces_lines", "50m", facecolor="none")
        ax.add_feature(states, edgecolor="#9a9a9a", linewidth=0.6, zorder=1)
        ax.coastlines(resolution="50m", linewidth=0.9, color="#444444", zorder=1)
        ax.gridlines(draw_labels=False, linestyle="--", linewidth=0.45, color="#bcbcbc", alpha=0.8)

        for cluster, sub in plot_base.groupby("cluster"):
            ax.scatter(sub["gauge_lon"], sub["gauge_lat"], s=54, color=palette[int(cluster)],
                       label=f"Cluster {int(cluster)} (n={len(sub)})", alpha=0.92, edgecolors="black",
                       linewidths=0.35, zorder=3, transform=data_crs)

        cent = plot_base.groupby("cluster")[["gauge_lon", "gauge_lat"]].median()
        for cluster, row in cent.iterrows():
            ax.text(row["gauge_lon"], row["gauge_lat"], f"C{int(cluster)}", transform=data_crs,
                    fontsize=14, weight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.65", alpha=0.82), zorder=5)

    else:
        fig, ax = plt.subplots(figsize=(15.5, 9.3))
        huc = load_huc02_boundaries(root_path)
        if huc is not None:
            try:
                huc.boundary.plot(ax=ax, color="0.62", linewidth=0.95, zorder=0)
            except Exception:
                pass
        for cluster, sub in plot_base.groupby("cluster"):
            ax.scatter(sub["gauge_lon"], sub["gauge_lat"], s=56, color=palette[int(cluster)],
                       label=f"Cluster {int(cluster)} (n={len(sub)})", alpha=0.92, edgecolors="black", linewidths=0.35, zorder=3)
        cent = plot_base.groupby("cluster")[["gauge_lon", "gauge_lat"]].median()
        for cluster, row in cent.iterrows():
            ax.text(row["gauge_lon"], row["gauge_lat"], f"C{int(cluster)}", fontsize=13, weight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.65", alpha=0.80), zorder=5)
        ax.set_xlim(-126, -66)
        ax.set_ylim(23, 52)
        ax.set_xlabel("Longitude", fontsize=18)
        ax.set_ylabel("Latitude", fontsize=18)

    ax.set_title("CAMELS-US study basins grouped into six hydrologically informed clusters", fontsize=22, fontweight="bold")
    leg = ax.legend(loc="lower right", frameon=True, title="Hydrologic clusters", title_fontsize=15, fontsize=14)
    leg.get_frame().set_alpha(0.96)
    plt.tight_layout()
    plt.savefig(output_dir / "fig01_map_clusters.png", bbox_inches="tight")
    plt.close(fig)


def plot_distributions(df: pd.DataFrame, vars_list: List[str], title: str, out_name: str, output_dir: Path) -> None:
    setup_plot_style()

    plot_df = df.copy()
    plot_df["cluster"] = pd.to_numeric(plot_df["cluster"], errors="coerce")
    plot_df = plot_df[plot_df["cluster"].notna()].copy()
    plot_df["cluster"] = plot_df["cluster"].astype(int)
    plot_df["cluster_plot"] = plot_df["cluster"].astype(str)

    order_int = sorted(plot_df["cluster"].unique().tolist())
    order = [str(c) for c in order_int]
    palette = get_cluster_palette(plot_df, as_str=True)

    fig, axes = plt.subplots(2, 3, figsize=(18.4, 10.8))
    axes = axes.flatten()

    for ax, var in zip(axes, vars_list):
        sub = plot_df[["cluster", "cluster_plot", var]].dropna().copy()
        sns.boxplot(
            data=sub,
            x="cluster_plot",
            y=var,
            order=order,
            ax=ax,
            palette=palette,
            showfliers=False,
            linewidth=0.9,
            saturation=1.0,
            boxprops=dict(edgecolor="0.20"),
            whiskerprops=dict(color="0.20", linewidth=0.9),
            capprops=dict(color="0.20", linewidth=0.9),
            medianprops=dict(color="black", linewidth=1.3),
        )

        strip_sub = sub.sample(min(len(sub), 400), random_state=42)
        sns.stripplot(
            data=strip_sub,
            x="cluster_plot",
            y=var,
            order=order,
            ax=ax,
            color="black",
            size=1.1,
            alpha=0.18,
            jitter=0.18,
        )
        ax.set_xlabel("Cluster", fontsize=17, fontweight="bold")
        ax.set_ylabel(PRETTY_LABELS.get(var, var), fontsize=17, fontweight="bold")
        ax.set_title(PRETTY_LABELS.get(var, var), fontsize=18, fontweight="bold")
        ax.tick_params(axis="both", labelsize=15)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontweight("bold")

    # Hide any unused axes, if present
    for ax in axes[len(vars_list):]:
        ax.axis("off")

    fig.suptitle(title, y=1.02, fontsize=20, weight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / out_name, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_profile_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    setup_plot_style()
    prof = df.groupby("cluster")[PROFILE_VARS].median()
    z = (prof - prof.mean(axis=0)) / prof.std(axis=0)
    z = z.rename(columns=PRETTY_LABELS)

    fig, ax = plt.subplots(figsize=(16, 5.5))
    sns.heatmap(z.T, cmap="cividis", center=None, annot=prof.rename(columns=PRETTY_LABELS).T.round(2), fmt="",
                cbar_kws={"label": "Cluster-median standardized anomaly"}, ax=ax,
                linewidths=0.0, linecolor=None, annot_kws={"fontsize": 14, "fontweight": "bold"})
    ax.grid(False)
    ax.set_title("Hydro-climatic profile of each cluster\n(values in cells = raw cluster medians; colors = standardized anomalies)")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=15)
    for lab in ax.get_xticklabels():
        lab.set_fontweight("bold")
    plt.tight_layout()
    plt.savefig(output_dir / "fig04_cluster_profile_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def plot_pca_view(df: pd.DataFrame, output_dir: Path) -> None:
    setup_plot_style()
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except Exception:
        print("scikit-learn not found; skipping PCA figure.")
        return

    use = df[["cluster"] + PROFILE_VARS].dropna().copy()
    X = use[PROFILE_VARS].values
    y = use["cluster"].astype(int).values

    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    PCs = pca.fit_transform(Xs)

    plot_df = pd.DataFrame({"PC1": PCs[:, 0], "PC2": PCs[:, 1], "cluster": y})
    palette = get_cluster_palette(df)

    fig, ax = plt.subplots(figsize=(11, 8))
    for c, sub in plot_df.groupby("cluster"):
        ax.scatter(sub["PC1"], sub["PC2"], s=45, alpha=0.7, color=palette[int(c)], label=f"Cluster {int(c)}")

    # variable loading arrows
    loadings = pca.components_.T
    scale = 3.0
    for i, var in enumerate(PROFILE_VARS):
        ax.arrow(0, 0, loadings[i, 0] * scale, loadings[i, 1] * scale,
                 head_width=0.08, alpha=0.75, color="black", length_includes_head=True)
        ax.text(loadings[i, 0] * scale * 1.08, loadings[i, 1] * scale * 1.08,
                PRETTY_LABELS.get(var, var), fontsize=11.5, fontweight="bold")

    ax.axhline(0, color="0.45", lw=2.0)
    ax.axvline(0, color="0.45", lw=2.0)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=17, fontweight="bold")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=17, fontweight="bold")
    ax.set_title("Attribute-space separation of the six clusters", fontsize=21, fontweight="bold")
    ax.legend(frameon=True, fontsize=14, title="Clusters", title_fontsize=15)
    ax.tick_params(axis="both", labelsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "fig05_pca_cluster_attribute_space.png", bbox_inches="tight")
    plt.close(fig)


HUC02_NAMES = {
    "01": "01 New England",
    "02": "02 Mid-Atlantic",
    "03": "03 South Atlantic-Gulf",
    "04": "04 Great Lakes",
    "05": "05 Ohio",
    "06": "06 Tennessee",
    "07": "07 Upper Mississippi",
    "08": "08 Lower Mississippi",
    "09": "09 Souris-Red-Rainy",
    "10": "10 Missouri",
    "11": "11 Arkansas-White-Red",
    "12": "12 Texas-Gulf",
    "13": "13 Rio Grande",
    "14": "14 Upper Colorado",
    "15": "15 Lower Colorado",
    "16": "16 Great Basin",
    "17": "17 Pacific Northwest",
    "18": "18 California",
}

def plot_cluster_huc_composition(df: pd.DataFrame, output_dir: Path) -> None:
    if "huc_str" not in df.columns:
        return

    setup_plot_style()

    plot_df = df.dropna(subset=["cluster", "huc_str"]).copy()
    plot_df["huc_str"] = plot_df["huc_str"].astype(str).str.zfill(2)
    comp = pd.crosstab(plot_df["cluster"], plot_df["huc_str"], normalize="index") * 100
    sorted_hucs = sorted(comp.columns.tolist(), key=lambda x: int(x))
    comp = comp.reindex(columns=sorted_hucs)

    # Javi-inspired extended palette by cycling the four approved tones.
    base = ["#56ae6c", "#8960b3", "#b0923b", "#ba495b", "#7fb3d5", "#c39bd3", "#f7dc6f", "#76d7c4",
            "#f8c471", "#a3e4d7", "#d7bde2", "#f5b7b1", "#82e0aa", "#85c1e9", "#f9e79f", "#aab7b8", "#58d68d", "#d2d57e"]
    colors = base[:len(sorted_hucs)]

    fig, ax = plt.subplots(figsize=(16.4, 7.2))
    comp.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.62, edgecolor="0.25", linewidth=0.25)

    handles = [Patch(facecolor=colors[i], edgecolor="0.25", label=HUC02_NAMES.get(h, h)) for i, h in enumerate(sorted_hucs)]
    pretty_labels = [HUC02_NAMES.get(h, h) for h in sorted_hucs]

    ax.set_title("Composition of each cluster by USGS 2-digit hydrologic region (HUC-02)", fontsize=22, fontweight="bold")
    ax.set_xlabel("Cluster", fontsize=18, fontweight="bold")
    ax.set_ylabel("Share of basins [%]", fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", labelsize=15)
    ax.set_xticklabels([str(x) for x in comp.index], rotation=0, fontweight="bold")
    for lab in ax.get_yticklabels():
        lab.set_fontweight("bold")

    ax.legend(
        handles,
        pretty_labels,
        title="USGS hydrologic region (HUC-02)",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        ncol=1,
        frameon=True,
        fontsize=12.5,
        title_fontsize=15
    )

    plt.tight_layout()
    plt.savefig(output_dir / "fig06_cluster_huc02_composition.png", bbox_inches="tight")
    plt.close(fig)


def load_basin_names(name_file_path: Path, basins: Optional[Iterable[str]] = None) -> pd.DataFrame:
    rows = []
    with open(name_file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(";", 2)
            if len(parts) < 3:
                continue

            gauge_id = parts[0].strip().zfill(8)
            huc02_from_namefile = parts[1].strip()
            station_name = parts[2].strip()

            # keep only proper gauge IDs
            if not gauge_id.isdigit():
                continue

            rows.append({
                "gauge_id": gauge_id,
                "station_name": station_name,
                "huc02_namefile": huc02_from_namefile,
            })

    names = pd.DataFrame(rows).drop_duplicates(subset="gauge_id", keep="first")

    if basins is not None:
        basins = [str(b).zfill(8) for b in basins]
        names = names[names["gauge_id"].isin(basins)].copy()

    return names

# =========================
# Main runner
# =========================
def run_all(root_path: Path = ROOT_PATH) -> pd.DataFrame:
    root_path = Path(root_path)
    setup_plot_style()

    df, output_dir = build_master_dataframe(root_path)

    expected_n = len(load_basin_list(root_path / "531_basin_list.txt"))
    if len(df) != expected_n:
        raise ValueError(f"Final dataframe has {len(df)} rows, expected {expected_n}.")
    print(f"Confirmed: all outputs are based on the {expected_n} study basins only.")

    # Robust cluster typing across the full workflow
    df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype("Int64")
    df["cluster_label"] = df["cluster"].apply(lambda x: f"Cluster {int(x)}" if pd.notna(x) else "Missing")

    # Tables
    make_overall_summary_table(df, output_dir)
    make_cluster_summary_table(df, output_dir)
    make_cluster_core_table(df, output_dir)
    make_cluster_categorical_table(df, output_dir)

    # Figures
    plot_cluster_map(df, root_path, output_dir)
    plot_distributions(
        df,
        vars_list=DIST_VARS_HYDRO,
        title="Hydrological signatures across the six clusters",
        out_name="fig02_hydrological_signature_distributions.png",
        output_dir=output_dir,
    )
    plot_distributions(
        df,
        vars_list=DIST_VARS_CLIMATE,
        title="Climatic and topographic gradients across the six clusters",
        out_name="fig03_climate_topography_distributions.png",
        output_dir=output_dir,
    )
    plot_cluster_profile_heatmap(df, output_dir)
    plot_pca_view(df, output_dir)
    plot_cluster_huc_composition(df, output_dir)

    # Helpful text summary for drafting the subsection
    summary_txt = output_dir / "draft_notes_study_area_dataset.txt"
    cluster_counts = df.groupby("cluster").size().to_dict()
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("CAMELS-US study-area quick notes\n")
        f.write("================================\n\n")
        f.write(f"Total basins in analysis: {df['gauge_id'].nunique()}\n")
        f.write(f"Cluster sizes: {cluster_counts}\n\n")
        f.write("Suggested reading of the figures:\n")
        f.write("- Fig. 1: spatial organization of the six clusters across CONUS.\n")
        f.write("- Fig. 2: cluster differences in key hydrological signatures.\n")
        f.write("- Fig. 3: cluster contrasts in climate/topography drivers.\n")
        f.write("- Fig. 4: concise hydro-climatic cluster profile heatmap.\n")
        f.write("- Fig. 5: optional attribute-space separation figure.\n")
        f.write("- Fig. 6: regional composition by HUC-02.\n")

    print("Done. Outputs written to:")
    print(output_dir)
    return df


if __name__ == "__main__":
    run_all(ROOT_PATH)
