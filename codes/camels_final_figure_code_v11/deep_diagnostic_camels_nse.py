from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from camels_diagnostics_utils import (
    ensure_dir,
    resolve_existing_file,
    load_pickle,
    flatten_metrics_all_test_period,
    flatten_metrics_by_water_year,
    flatten_metrics_by_year_season,
    load_master_attributes,
    load_train_data,
    extract_train_quality,
    zfill_basin,
    seasonal_order_key,
    summarize_group_numeric,
    safe_effect_size,
    _grouped_row_nanmean,
    _safe_row_nanmean,
    _safe_row_nanstd,
    _safe_row_nanmin,
    _safe_row_nanmax,
    _safe_ratio,
)

TARGET_MODEL = "Cluster-wise"
TARGET_METRIC = "NSE"
SEASONS = ["JFM", "AMJ", "JAS", "OND"]
HIGH_THRESHOLDS = [0.8, 0.9]
ATTRIBUTE_FOCUS = [
    "p_mean", "pet_mean", "frac_snow", "aridity", "q_mean", "runoff_ratio",
    "slope_fdc", "baseflow_index", "high_q_freq", "low_q_freq", "zero_q_freq",
    "elev_mean", "slope_mean", "area_gages2", "geol_permeability", "soil_conductivity",
]
QUALITY_FOCUS = [
    "qobs_missing_frac", "prcp_missing_frac", "temp_missing_frac", "qobs_zero_frac",
    "qobs_cv_train", "qobs_nonmissing_days", "prcp_zero_frac"
]
REGIME_FOCUS = [
    "wy_q_cv_train", "wy_p_cv_train", "wy_rr_cv_train",
    "q_seasonality_amp_train", "p_seasonality_amp_train",
    "jfm_q_mean_train", "amj_q_mean_train", "jas_q_mean_train", "ond_q_mean_train",
    "jfm_rr_train", "amj_rr_train", "jas_rr_train", "ond_rr_train",
    "jfm_zero_q_frac_train", "amj_zero_q_frac_train", "jas_zero_q_frac_train", "ond_zero_q_frac_train",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Deep diagnostic analysis for CAMELS-US benchmark results (NSE only)."
    )
    p.add_argument("--project-root", type=str, required=True,
                   help="Root folder like F:/Experiments/CAMELS_US/Clean_4_upload")
    p.add_argument("--output-subdir", type=str, default="deep_diagnostic_camels_nse")
    p.add_argument("--bad-threshold", type=float, default=0.5)
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()



def extract_regime_features(train_data: dict) -> pd.DataFrame:
    basin_obj = train_data["coords"]["basin"]
    date_obj = train_data["coords"]["date"]

    basin_vals = np.asarray(basin_obj["data"] if isinstance(basin_obj, dict) else basin_obj).tolist()
    date_vals = pd.to_datetime(np.asarray(date_obj["data"] if isinstance(date_obj, dict) else date_obj))

    basins = [zfill_basin(x) for x in basin_vals]

    def arr_first(names):
        for name in names:
            if name in train_data["data_vars"]:
                return np.asarray(train_data["data_vars"][name]["data"], dtype=float)
        return None

    q = arr_first(["QObs(mm/d)"])
    p = arr_first(["PRCP(mm/day)_maurer", "PRCP(mm/day)_nldas"])

    out = pd.DataFrame({"basin": basins})
    if q is None or p is None:
        return out

    months = pd.Index(date_vals.month).to_numpy()
    years = pd.Index(date_vals.year).to_numpy()
    water_year = np.where(months >= 10, years + 1, years)
    season = np.where(
        np.isin(months, [1, 2, 3]), "JFM",
        np.where(np.isin(months, [4, 5, 6]), "AMJ",
                 np.where(np.isin(months, [7, 8, 9]), "JAS", "OND"))
    )

    # Interannual variability proxies
    wy_order = np.unique(water_year)
    q_wy, _ = _grouped_row_nanmean(q, water_year, wy_order)
    p_wy, _ = _grouped_row_nanmean(p, water_year, wy_order)
    rr_wy = _safe_ratio(q_wy, p_wy)

    out["wy_q_cv_train"] = _safe_ratio(_safe_row_nanstd(q_wy), _safe_row_nanmean(q_wy))
    out["wy_p_cv_train"] = _safe_ratio(_safe_row_nanstd(p_wy), _safe_row_nanmean(p_wy))
    out["wy_rr_cv_train"] = _safe_ratio(_safe_row_nanstd(rr_wy), _safe_row_nanmean(rr_wy))

    # Seasonal climatology
    season_order = ["JFM", "AMJ", "JAS", "OND"]
    q_season, _ = _grouped_row_nanmean(q, season, season_order)
    p_season, _ = _grouped_row_nanmean(p, season, season_order)
    zero_q = np.where(np.isfinite(q), np.where(q == 0.0, 1.0, 0.0), np.nan)
    zero_q_season, _ = _grouped_row_nanmean(zero_q, season, season_order)

    for i, s in enumerate(season_order):
        sl = s.lower()
        out[f"{sl}_q_mean_train"] = q_season[:, i]
        out[f"{sl}_rr_train"] = _safe_ratio(q_season[:, i], p_season[:, i])
        out[f"{sl}_zero_q_frac_train"] = zero_q_season[:, i]

    # Monthly climatology and seasonality amplitude
    month_order = np.arange(1, 13)
    q_month, _ = _grouped_row_nanmean(q, months, month_order)
    p_month, _ = _grouped_row_nanmean(p, months, month_order)

    q_month_mean = _safe_row_nanmean(q_month)
    p_month_mean = _safe_row_nanmean(p_month)

    out["q_seasonality_amp_train"] = _safe_ratio(_safe_row_nanmax(q_month) - _safe_row_nanmin(q_month), q_month_mean)
    out["p_seasonality_amp_train"] = _safe_ratio(_safe_row_nanmax(p_month) - _safe_row_nanmin(p_month), p_month_mean)

    # Diagnostics on whether regime features were computable
    regime_cols = [c for c in out.columns if c != "basin"]
    out["n_missing_regime_features"] = out[regime_cols].isna().sum(axis=1)
    return out


def load_inputs(project_root: Path):
    pickles_dir = project_root / "Pickles"
    train_dir = project_root / "Train_Data" / "train_data"

    p_overall = resolve_existing_file(pickles_dir, ["metrics_all_test_period"])
    p_wy = resolve_existing_file(pickles_dir, ["metrics_by_water_year"])
    p_season = resolve_existing_file(pickles_dir, ["metrics_by_year_season"])
    p_train = resolve_existing_file(train_dir, ["train_data"])

    overall = flatten_metrics_all_test_period(load_pickle(p_overall))
    wy = flatten_metrics_by_water_year(load_pickle(p_wy))
    season = flatten_metrics_by_year_season(load_pickle(p_season))

    attrs = load_master_attributes(project_root)
    train_data = load_train_data(p_train)
    quality = extract_train_quality(train_data)
    regime = extract_regime_features(train_data)

    return overall, wy, season, attrs, quality, regime


def seed_mean_basin_metric(df: pd.DataFrame, metric: str = TARGET_METRIC, model: str = TARGET_MODEL) -> pd.DataFrame:
    d = df[(df["metric"] == metric) & (df["model"] == model)].copy()
    return (
        d.groupby("basin", as_index=False)["value"]
        .mean()
        .rename(columns={"value": metric})
    )


def seed_mean_basin_year_metric(df: pd.DataFrame, metric: str = TARGET_METRIC, model: str = TARGET_MODEL) -> pd.DataFrame:
    d = df[(df["metric"] == metric) & (df["model"] == model)].copy()
    return (
        d.groupby(["water_year", "basin"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": metric})
    )


def seed_mean_basin_season_metric(df: pd.DataFrame, metric: str = TARGET_METRIC, model: str = TARGET_MODEL) -> pd.DataFrame:
    d = df[(df["metric"] == metric) & (df["model"] == model)].copy()
    return (
        d.groupby(["water_year", "season", "basin"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": metric})
    )


def build_master_table(overall_df, attrs, quality, regime):
    basin_nse = seed_mean_basin_metric(overall_df)
    m = basin_nse.merge(attrs, on="basin", how="left").merge(quality, on="basin", how="left").merge(regime, on="basin", how="left")

    # Normalize common optional station-name columns if present
    if "station_name" in m.columns and "gauge_name" not in m.columns:
        m["gauge_name"] = m["station_name"]
    return m


def export_excel_sheets(path: Path, sheets: dict[str, pd.DataFrame]):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)


def plot_bar_counts(df_counts: pd.DataFrame, x: str, y: str, title: str, ylabel: str, save_path: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.bar(df_counts[x].astype(str), df_counts[y].values)
    ax.set_title(title, fontsize=19, fontweight="bold")
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_stacked(counts: pd.DataFrame, category_col: str, cluster_col: str, value_col: str, title: str, save_path: Path, dpi: int):
    piv = counts.pivot_table(index=category_col, columns=cluster_col, values=value_col, fill_value=0)
    piv = piv.sort_index()
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    bottom = np.zeros(len(piv))
    for col in piv.columns:
        vals = piv[col].values
        ax.bar(piv.index.astype(str), vals, bottom=bottom, label=f"Cluster {col}")
        bottom += vals
    ax.set_title(title, fontsize=19, fontweight="bold")
    ax.set_xlabel(category_col.replace("_", " ").title())
    if category_col == "water_year":
        ax.set_ylabel("Number of lower-skill basin-year cases")
    elif category_col == "season":
        ax.set_ylabel("Number of lower-skill basin-season cases")
    else:
        ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
        fontsize=9,
        title="Clusters",
        title_fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 0.84, 1])
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_effect_sizes(effect_df: pd.DataFrame, title: str, save_path: Path, dpi: int):
    d = effect_df.dropna(subset=["effect_size"]).copy().sort_values("effect_size")
    fig, ax = plt.subplots(figsize=(9.0, max(4.8, 0.35 * len(d))))
    ax.barh(d["variable"], d["effect_size"])
    ax.axvline(0, color="k", lw=1)
    ax.set_title(title, fontsize=19, fontweight="bold")
    ax.set_xlabel("Standardized mean difference\n(NSE < threshold minus remaining basins)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.tick_params(axis="both", labelsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_year_ranking(df_year: pd.DataFrame, save_path: Path, dpi: int):
    d = df_year.copy()
    d["water_year"] = pd.to_numeric(d["water_year"], errors="coerce")
    d = d.sort_values("water_year").copy()
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.bar(d["water_year"].astype(int).astype(str), d["median_nse"].values, color="#56ae6c", edgecolor="0.20", linewidth=0.5)
    ax.set_title("Cluster-wise model: median basin-wise NSE by water year", fontsize=19, fontweight="bold")
    ax.set_xlabel("Water year")
    ax.set_ylabel("Median NSE")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=45)
    ax.tick_params(axis="both", labelsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)



def basin_seasonality_link(master: pd.DataFrame, season_df: pd.DataFrame):
    d = seed_mean_basin_season_metric(season_df)
    season_pivot = d.pivot_table(index="basin", columns="season", values=TARGET_METRIC, aggfunc="median")
    for s in SEASONS:
        if s not in season_pivot.columns:
            season_pivot[s] = np.nan
    season_pivot = season_pivot[SEASONS].reset_index()
    other_mean = season_pivot[["JFM", "AMJ", "OND"]].mean(axis=1)
    season_pivot["jas_penalty"] = other_mean - season_pivot["JAS"]
    season_pivot["seasonal_nse_range"] = season_pivot[SEASONS].max(axis=1) - season_pivot[SEASONS].min(axis=1)
    out = season_pivot.merge(master, on="basin", how="left")
    return out

def basin_wy_instability_link(master: pd.DataFrame, wy_df: pd.DataFrame):
    d = seed_mean_basin_year_metric(wy_df)
    basin_var = d.groupby("basin")[TARGET_METRIC].agg(
        wy_n='count',
        wy_nse_mean='mean',
        wy_nse_median='median',
        wy_nse_std='std',
        wy_nse_min='min',
        wy_nse_max='max',
    ).reset_index()
    basin_var["wy_nse_range"] = basin_var["wy_nse_max"] - basin_var["wy_nse_min"]
    out = basin_var.merge(master, on="basin", how="left")
    return out

def seasonal_diagnostics(master: pd.DataFrame, season_df: pd.DataFrame, outdir: Path, bad_threshold: float, dpi: int):
    d = seed_mean_basin_season_metric(season_df)
    keep_cols = ["basin", "cluster", "cluster_label"] + [c for c in ATTRIBUTE_FOCUS + QUALITY_FOCUS + REGIME_FOCUS if c in master.columns]
    d = d.merge(master[keep_cols], on="basin", how="left")

    seasonal_summary = (
        d.groupby("season")[TARGET_METRIC]
        .agg(
            n_basin_year="count",
            mean_nse="mean",
            median_nse="median",
            min_nse="min",
            max_nse="max",
            n_lt_0p5=lambda s: (s < bad_threshold).sum(),
            n_ge_0p8=lambda s: (s >= 0.8).sum(),
            n_ge_0p9=lambda s: (s >= 0.9).sum(),
        )
        .reset_index()
    )
    seasonal_summary["season_order"] = seasonal_summary["season"].map(seasonal_order_key)
    seasonal_summary = seasonal_summary.sort_values("season_order").drop(columns="season_order")

    # JAS vs rest attribute and quality comparison
    d["is_jas_bad"] = (d["season"] == "JAS") & (d[TARGET_METRIC] < bad_threshold)
    jas_bad = d[d["is_jas_bad"]].copy()
    rest = d[~d["is_jas_bad"]].copy()

    effect_rows = []
    for col in [c for c in ATTRIBUTE_FOCUS + QUALITY_FOCUS if c in d.columns]:
        effect_rows.append({
            "variable": col,
            "effect_size": safe_effect_size(jas_bad[col], rest[col]),
            "bad_mean": pd.to_numeric(jas_bad[col], errors="coerce").mean(),
            "rest_mean": pd.to_numeric(rest[col], errors="coerce").mean(),
            "bad_median": pd.to_numeric(jas_bad[col], errors="coerce").median(),
            "rest_median": pd.to_numeric(rest[col], errors="coerce").median(),
        })
    jas_effects = pd.DataFrame(effect_rows)

    seasonal_lt_0p5_cluster = (
        d.assign(is_bad=d[TARGET_METRIC] < bad_threshold)
         .query("is_bad")
         .groupby(["season", "cluster"], as_index=False)
         .size()
         .rename(columns={"size": "n_bad"})
    )

    plot_bar_counts(
        seasonal_summary,
        "season",
        "median_nse",
        "Cluster-wise model: median NSE by season",
        "Median NSE",
        outdir / "seasonal_median_nse.png",
        dpi,
    )
    if len(seasonal_lt_0p5_cluster):
        plot_cluster_stacked(
            seasonal_lt_0p5_cluster,
            "season",
            "cluster",
            "n_bad",
            f"Counts of basin-season cases with NSE < {bad_threshold} by season and cluster",
            outdir / "seasonal_nse_lt_0p5_counts_by_cluster.png",
            dpi,
        )
    if len(jas_effects.dropna(subset=["effect_size"])):
        plot_effect_sizes(
            jas_effects,
            "JAS basin-season cases with NSE < threshold vs remaining basin-season cases",
            outdir / "jas_lt_0p5_effect_sizes.png",
            dpi,
        )

    # Basin-level seasonal sensitivity link
    season_link = basin_seasonality_link(master, season_df)
    q75 = season_link["jas_penalty"].quantile(0.75)
    season_link["high_jas_penalty"] = season_link["jas_penalty"] >= q75
    penalty_high = season_link[season_link["high_jas_penalty"]]
    penalty_rest = season_link[~season_link["high_jas_penalty"]]
    penalty_effects = []
    for col in [c for c in REGIME_FOCUS + ATTRIBUTE_FOCUS + QUALITY_FOCUS if c in season_link.columns]:
        penalty_effects.append({
            "variable": col,
            "effect_size": safe_effect_size(penalty_high[col], penalty_rest[col]),
            "high_penalty_mean": pd.to_numeric(penalty_high[col], errors="coerce").mean(),
            "rest_mean": pd.to_numeric(penalty_rest[col], errors="coerce").mean(),
        })
    penalty_effects = pd.DataFrame(penalty_effects)
    if len(penalty_effects.dropna(subset=["effect_size"])):
        plot_effect_sizes(
            penalty_effects,
            "Hydro-climatic signature of basins with pronounced JAS performance loss",
            outdir / "seasonality_link_effect_sizes.png",
            dpi,
        )

    return {
        "seasonal_summary": seasonal_summary,
        "jas_lt_0p5_basin_year_rows": jas_bad.sort_values(TARGET_METRIC),
        "jas_lt_0p5_effect_sizes": jas_effects.sort_values("effect_size"),
        "seasonal_lt_0p5_cluster": seasonal_lt_0p5_cluster,
        "basin_seasonality_link": season_link.sort_values("jas_penalty", ascending=False),
        "seasonality_link_effect_sizes": penalty_effects.sort_values("effect_size"),
    }


def wateryear_diagnostics(master: pd.DataFrame, wy_df: pd.DataFrame, outdir: Path, bad_threshold: float, dpi: int):
    d = seed_mean_basin_year_metric(wy_df)
    d = d.merge(master[["basin", "cluster", "cluster_label"] + [c for c in ATTRIBUTE_FOCUS + QUALITY_FOCUS + REGIME_FOCUS if c in master.columns]], on="basin", how="left")

    year_rank = (
        d.groupby("water_year", as_index=False)[TARGET_METRIC]
         .agg(median_nse="median", mean_nse="mean", n_lt_0p5=lambda s: (s < bad_threshold).sum(), n_ge_0p8=lambda s: (s >= 0.8).sum())
         .sort_values("median_nse")
    )
    weak_years = year_rank.head(3)["water_year"].tolist()

    weak_rows = d[d["water_year"].isin(weak_years)].copy()
    weak_cluster = (
        weak_rows.assign(is_bad=weak_rows[TARGET_METRIC] < bad_threshold)
        .query("is_bad")
        .groupby(["water_year", "cluster"], as_index=False)
        .size()
        .rename(columns={"size": "n_bad"})
    )

    effect_rows = []
    weak_bad = weak_rows[weak_rows[TARGET_METRIC] < bad_threshold]
    others = d[~((d["water_year"].isin(weak_years)) & (d[TARGET_METRIC] < bad_threshold))]
    for col in [c for c in ATTRIBUTE_FOCUS + QUALITY_FOCUS if c in d.columns]:
        effect_rows.append({
            "variable": col,
            "effect_size": safe_effect_size(weak_bad[col], others[col]),
            "weak_bad_mean": pd.to_numeric(weak_bad[col], errors="coerce").mean(),
            "others_mean": pd.to_numeric(others[col], errors="coerce").mean(),
        })
    weak_effects = pd.DataFrame(effect_rows)

    plot_year_ranking(year_rank, outdir / "water_year_median_nse_ranking.png", dpi)
    if len(weak_cluster):
        plot_cluster_stacked(weak_cluster, "water_year", "cluster", "n_bad",
                             f"Counts of basin-year cases with NSE < threshold in weaker water years",
                             outdir / "water_year_nse_lt_0p5_counts_by_cluster.png", dpi)
    plot_effect_sizes(weak_effects, "Lower-skill basin-year cases in weaker water years vs remaining basin-year cases", outdir / "weak_year_effect_sizes.png", dpi)

    # Basin-level interannual instability link
    wy_link = basin_wy_instability_link(master, wy_df)
    q75 = wy_link["wy_nse_std"].quantile(0.75)
    wy_link["high_wy_instability"] = wy_link["wy_nse_std"] >= q75
    inst_high = wy_link[wy_link["high_wy_instability"]]
    inst_rest = wy_link[~wy_link["high_wy_instability"]]
    inst_effects = []
    for col in [c for c in REGIME_FOCUS + ATTRIBUTE_FOCUS + QUALITY_FOCUS if c in wy_link.columns]:
        inst_effects.append({
            "variable": col,
            "effect_size": safe_effect_size(inst_high[col], inst_rest[col]),
            "high_instability_mean": pd.to_numeric(inst_high[col], errors="coerce").mean(),
            "rest_mean": pd.to_numeric(inst_rest[col], errors="coerce").mean(),
        })
    inst_effects = pd.DataFrame(inst_effects)
    if len(inst_effects.dropna(subset=["effect_size"])):
        plot_effect_sizes(inst_effects, "High interannual-instability basins vs remaining basins",
                          outdir / "wy_instability_link_effect_sizes.png", dpi)

    return {
        "water_year_ranking": year_rank,
        "weak_water_years": pd.DataFrame({"water_year": weak_years}),
        "weak_year_lt_0p5_rows": weak_bad.sort_values(["water_year", TARGET_METRIC]),
        "weak_year_lt_0p5_cluster": weak_cluster,
        "weak_year_effect_sizes": weak_effects.sort_values("effect_size"),
        "basin_wy_instability_link": wy_link.sort_values("wy_nse_std", ascending=False),
        "wy_instability_link_effect_sizes": inst_effects.sort_values("effect_size"),
    }


def threshold_diagnostics(master: pd.DataFrame, outdir: Path, dpi: int):
    outputs = {}
    for thr in HIGH_THRESHOLDS:
        tag = f"ge_{str(thr).replace('.', 'p')}"
        sub = master[master[TARGET_METRIC] >= thr].copy().sort_values(TARGET_METRIC, ascending=False)
        outputs[f"basins_{tag}"] = sub
        outputs[f"summary_{tag}"] = pd.DataFrame([{
            "threshold": thr,
            "n_basins": len(sub),
            "share_pct": 100 * len(sub) / len(master),
            "median_nse": sub[TARGET_METRIC].median() if len(sub) else np.nan,
            "mean_nse": sub[TARGET_METRIC].mean() if len(sub) else np.nan,
        }])
        if len(sub):
            cluster_counts = sub.groupby(["cluster", "cluster_label"], as_index=False).size().rename(columns={"size": "n_basins"})
            outputs[f"cluster_{tag}"] = cluster_counts
            plot_bar_counts(cluster_counts, "cluster_label", "n_basins",
                            f"Counts of basins with NSE ≥ {thr}", "Count", outdir / f"highskill_cluster_counts_{tag}.png", dpi)

            attr_summary = summarize_group_numeric(sub, "variable", [c for c in ATTRIBUTE_FOCUS if c in sub.columns])
            outputs[f"attributes_{tag}"] = attr_summary
    return outputs


def runoff_ratio_gt1_diagnostics(master: pd.DataFrame):
    sub = master[pd.to_numeric(master["runoff_ratio"], errors="coerce") > 1].copy()
    cols = [
        "basin", "gauge_name", "cluster", "cluster_label", TARGET_METRIC,
        "runoff_ratio", "q_mean", "p_mean", "aridity", "frac_snow",
        "baseflow_index", "slope_fdc", "zero_q_freq"
    ]
    cols = [c for c in cols if c in sub.columns]
    return {"runoff_ratio_gt1_basins": sub[cols].sort_values("runoff_ratio", ascending=False)}



def low_skill_threshold_diagnostics(master: pd.DataFrame, outdir: Path, bad_threshold: float, dpi: int):
    low_skill = master[master[TARGET_METRIC] < bad_threshold].copy().sort_values(TARGET_METRIC)
    remaining = master[master[TARGET_METRIC] >= bad_threshold].copy()

    effect_rows = []
    for col in [c for c in ATTRIBUTE_FOCUS + QUALITY_FOCUS + REGIME_FOCUS if c in master.columns]:
        effect_rows.append({
            "variable": col,
            "effect_size": safe_effect_size(low_skill[col], remaining[col]),
            "low_skill_mean": pd.to_numeric(low_skill[col], errors="coerce").mean(),
            "remaining_mean": pd.to_numeric(remaining[col], errors="coerce").mean(),
            "low_skill_median": pd.to_numeric(low_skill[col], errors="coerce").median(),
            "remaining_median": pd.to_numeric(remaining[col], errors="coerce").median(),
        })
    effects = pd.DataFrame(effect_rows).sort_values("effect_size")

    cluster_counts = (
        low_skill.groupby(["cluster", "cluster_label"], as_index=False)
        .size()
        .rename(columns={"size": "n_lt_0p5_basins"})
    )
    plot_bar_counts(
        cluster_counts,
        "cluster_label",
        "n_lt_0p5_basins",
        f"Counts of basins with NSE < {bad_threshold} by cluster",
        "Count",
        outdir / "nse_lt_0p5_cluster_counts.png",
        dpi,
    )
    plot_effect_sizes(
        effects,
        "Basins with NSE < threshold vs remaining basins",
        outdir / "nse_lt_0p5_effect_sizes.png",
        dpi,
    )

    wide_cols = [
        "basin", "gauge_name", "cluster", "cluster_label", TARGET_METRIC,
        "p_mean", "q_mean", "runoff_ratio", "frac_snow", "aridity",
        "baseflow_index", "slope_fdc", "high_q_freq", "low_q_freq", "zero_q_freq",
        "gauge_lat", "gauge_lon", "elev_mean", "slope_mean", "area_gages2",
        "qobs_missing_frac", "prcp_missing_frac", "temp_missing_frac",
        "qobs_zero_frac", "qobs_cv_train", "qobs_nonmissing_days", "prcp_zero_frac",
        "wy_q_cv_train", "wy_p_cv_train", "wy_rr_cv_train",
        "q_seasonality_amp_train", "p_seasonality_amp_train",
        "jfm_q_mean_train", "amj_q_mean_train", "jas_q_mean_train", "ond_q_mean_train",
        "jfm_rr_train", "amj_rr_train", "jas_rr_train", "ond_rr_train",
        "jfm_zero_q_frac_train", "amj_zero_q_frac_train", "jas_zero_q_frac_train", "ond_zero_q_frac_train",
        "qobs_allnan_flag", "prcp_allnan_flag", "tmax_allnan_flag", "tmin_allnan_flag",
        "n_allnan_core_vars", "n_missing_regime_features",
    ]
    wide_cols = [c for c in wide_cols if c in low_skill.columns]

    quality_summary = summarize_group_numeric(
        low_skill,
        "variable",
        [c for c in QUALITY_FOCUS + ["qobs_allnan_flag", "prcp_allnan_flag", "tmax_allnan_flag", "tmin_allnan_flag", "n_allnan_core_vars", "n_missing_regime_features"] if c in low_skill.columns],
    )
    attr_summary = summarize_group_numeric(
        low_skill,
        "variable",
        [c for c in ATTRIBUTE_FOCUS + REGIME_FOCUS if c in low_skill.columns],
    )

    summary_table = pd.DataFrame([{
        "threshold": bad_threshold,
        "n_basins_lt_threshold": len(low_skill),
        "share_pct": 100 * len(low_skill) / len(master) if len(master) else np.nan,
        "median_nse_lt_threshold": low_skill[TARGET_METRIC].median() if len(low_skill) else np.nan,
        "mean_nse_lt_threshold": low_skill[TARGET_METRIC].mean() if len(low_skill) else np.nan,
        "min_nse_lt_threshold": low_skill[TARGET_METRIC].min() if len(low_skill) else np.nan,
        "max_nse_lt_threshold": low_skill[TARGET_METRIC].max() if len(low_skill) else np.nan,
    }])

    return {
        "nse_lt_0p5_summary": summary_table,
        "nse_lt_0p5_basins_full": low_skill[wide_cols],
        "nse_lt_0p5_cluster_counts": cluster_counts,
        "nse_lt_0p5_effect_sizes": effects,
        "nse_lt_0p5_quality_summary": quality_summary,
        "nse_lt_0p5_attribute_summary": attr_summary,
    }


def write_readme(outdir: Path, bad_threshold: float):
    txt = f"""
    CAMELS-US deep diagnostic outputs (NSE only; threshold subset defined as NSE < user-specified cutoff, default 0.5)
    ============================================

    Core questions addressed
    ------------------------
    1. Why did the model perform poorly in summer (JAS)?
    2. Why were some water years weaker than others?
    3. How many basins reached NSE >= 0.80 and NSE >= 0.90, and what attributes did they have?
    4. How did the model perform in basins with runoff_ratio > 1?
    5. What hydrologic and data-quality characteristics distinguish the subset of basins with NSE < {bad_threshold}?
6. Do summer failures relate to basin seasonal regime structure?
7. Do basins with more variable year-to-year hydrology also show less uniform model skill across water years?

    Main outputs
    ------------
    - diagnostics_tables.xlsx : multi-sheet export of all main tables
    - seasonal_median_nse.png
    - seasonal_nse_lt_0p5_counts_by_cluster.png
    - jas_lt_0p5_effect_sizes.png
    - water_year_median_nse_ranking.png
    - water_year_nse_lt_0p5_counts_by_cluster.png
    - weak_year_effect_sizes.png
    - highskill_cluster_counts_ge_0p8.png
    - highskill_cluster_counts_ge_0p9.png
    - nse_lt_0p5_cluster_counts.png
    - nse_lt_0p5_effect_sizes.png
- seasonality_link_effect_sizes.png
- wy_instability_link_effect_sizes.png

    Interpretation note
    -------------------
    Effect-size plots show standardized mean differences for the NSE < threshold subset relative to the remaining basins/cases.
    Positive values mean the variable tends to be larger in the lower-skill subset.
    Negative values mean the variable tends to be smaller in the lower-skill subset.
    """
    (outdir / "README.txt").write_text(textwrap.dedent(txt).strip() + "\n", encoding="utf-8")


def main():
    args = parse_args()
    project_root = Path(args.project_root)
    outdir = ensure_dir(project_root / "Results" / args.output_subdir)

    overall_df, wy_df, season_df, attrs, quality, regime = load_inputs(project_root)
    master = build_master_table(overall_df, attrs, quality, regime)

    sheets = {}
    sheets["overall_master"] = master.sort_values(TARGET_METRIC, ascending=False)

    seasonal_out = seasonal_diagnostics(master, season_df, outdir, args.bad_threshold, args.dpi)
    wateryear_out = wateryear_diagnostics(master, wy_df, outdir, args.bad_threshold, args.dpi)
    threshold_out = threshold_diagnostics(master, outdir, args.dpi)
    runoff_out = runoff_ratio_gt1_diagnostics(master)
    bad_out = low_skill_threshold_diagnostics(master, outdir, args.bad_threshold, args.dpi)

    for d in [seasonal_out, wateryear_out, threshold_out, runoff_out, bad_out]:
        sheets.update(d)

    export_excel_sheets(outdir / "diagnostics_tables.xlsx", sheets)
    for name, df in sheets.items():
        df.to_csv(outdir / f"{name}.csv", index=False)

    write_readme(outdir, args.bad_threshold)

    print(f"[OK] Diagnostic package written to: {outdir}")


if __name__ == "__main__":
    main()
