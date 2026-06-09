from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

from camels_utils import (
    get_project_root,
    get_pickles_dir,
    load_pickle,
    safe_numeric,
    flatten_metrics_all_test_period,
)

THRESHOLD = 0.5
TARGET_MODEL = "Cluster-wise"
TARGET_METRIC = "NSE"


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


def load_basin_list(basin_list_path: Path) -> list[str]:
    with open(basin_list_path, "r", encoding="utf-8") as f:
        basins = [line.strip() for line in f if line.strip()]
    return [zfill_basin(b) for b in basins]


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


def load_basin_names(project_root: Path, basins: list[str]) -> pd.DataFrame:
    candidate_files = [
        project_root / "camels_name.txt",
        project_root / "camels_names.txt",
        project_root / "gauge_information.txt",
        project_root / "gauge_information.csv",
    ]

    for f in candidate_files:
        if not f.exists():
            continue

        if f.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(f, dtype=str)
                basin_col = find_column(df, ["gauge_id", "basin", "station_id"])
                name_col = find_column(df, ["station_name", "gauge_name", "name"])
                if basin_col and name_col:
                    out = df[[basin_col, name_col]].copy()
                    out.columns = ["basin", "station_name"]
                    out["basin"] = out["basin"].map(zfill_basin)
                    return out.drop_duplicates("basin")
            except Exception:
                pass
        else:
            rows = []
            try:
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        m = re.match(r"^(\d{8})\s+(.+)$", line)
                        if m:
                            rows.append({"basin": m.group(1), "station_name": m.group(2).strip()})
                if rows:
                    out = pd.DataFrame(rows)
                    out["basin"] = out["basin"].map(zfill_basin)
                    return out.drop_duplicates("basin")
            except Exception:
                pass

    return pd.DataFrame({"basin": basins, "station_name": [""] * len(basins)})


def main():
    project_root = get_project_root()
    pickles_dir = get_pickles_dir()
    output_dir = ensure_dir(project_root / "Results" / "Tables_cluster")

    basin_list_path = project_root / "531_basin_list.txt"
    attributes_dir = project_root / "Train_Data" / "camels_attributes_v2.0"

    basins = load_basin_list(basin_list_path)
    clusters = load_clusters(project_root)
    names = load_basin_names(project_root, basins)
    attrs = load_camels_attributes(attributes_dir, basins=basins).reset_index().rename(
        columns={"gauge_id": "basin", "index": "basin"}
    )
    attrs["basin"] = attrs["basin"].map(zfill_basin)

    metrics = load_all_test_period_metrics(pickles_dir)

    # seed-mean metric table at basin level for all metrics of the Cluster-wise model
    model_df = metrics[metrics["model"] == TARGET_MODEL].copy()

    basin_metric_mean = (
        model_df.groupby(["basin", "metric"], dropna=False)["value"]
        .mean()
        .reset_index()
    )

    basin_metric_wide = (
        basin_metric_mean.pivot(index="basin", columns="metric", values="value")
        .reset_index()
    )

    # bad basins based on NSE < threshold
    if TARGET_METRIC not in basin_metric_wide.columns:
        raise KeyError(f"{TARGET_METRIC} not found in basin-level metric table.")

    bad = basin_metric_wide[basin_metric_wide[TARGET_METRIC] < THRESHOLD].copy()
    bad = bad.sort_values(TARGET_METRIC, ascending=True).reset_index(drop=True)

    bad = bad.merge(clusters, on="basin", how="left")
    bad = bad.merge(names, on="basin", how="left")
    bad = bad.merge(attrs, on="basin", how="left")

    counts_by_cluster = (
        bad.groupby(["cluster", "cluster_label"], dropna=False)
        .size()
        .reset_index(name="n_bad_basins")
        .sort_values(["cluster"])
    )

    selected_cols_front = [
        "basin", "station_name", "cluster", "cluster_label",
        "NSE", "KGE", "Pearson-r", "RMSE", "Peak-timing", "Peak-MAPE", "Missed-peaks"
    ]
    selected_cols_front = [c for c in selected_cols_front if c in bad.columns]
    other_cols = [c for c in bad.columns if c not in selected_cols_front]
    bad = bad[selected_cols_front + other_cols]

    summary = pd.DataFrame({
        "threshold": [THRESHOLD],
        "model": [TARGET_MODEL],
        "metric_for_filter": [TARGET_METRIC],
        "n_bad_basins": [len(bad)],
        "n_total_basins": [len(basin_metric_wide)],
        "fraction_bad": [len(bad) / len(basin_metric_wide) if len(basin_metric_wide) else np.nan],
        "worst_nse": [bad["NSE"].min() if len(bad) and "NSE" in bad.columns else np.nan],
        "best_within_bad": [bad["NSE"].max() if len(bad) and "NSE" in bad.columns else np.nan],
    })

    base = f"bad_basins_clusterwise_nse_lt_{str(THRESHOLD).replace('.', 'p')}"
    csv_path = output_dir / f"{base}.csv"
    count_csv_path = output_dir / f"{base}_counts_by_cluster.csv"
    xlsx_path = output_dir / f"{base}.xlsx"

    bad.to_csv(csv_path, index=False)
    counts_by_cluster.to_csv(count_csv_path, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        counts_by_cluster.to_excel(writer, sheet_name="counts_by_cluster", index=False)
        bad.to_excel(writer, sheet_name="bad_basins_full", index=False)

        compact_cols = [
            c for c in [
                "basin", "station_name", "cluster", "cluster_label",
                "NSE", "KGE", "Pearson-r", "RMSE"
            ] if c in bad.columns
        ]
        bad[compact_cols].to_excel(writer, sheet_name="bad_basins_compact", index=False)

    print(f"[OK] threshold = {THRESHOLD}")
    print(f"[OK] cluster ids converted to 1..6 when source file uses 0..5")
    print(f"[OK] number of bad basins (NSE < {THRESHOLD}): {len(bad)}")
    print(f"[OK] saved:")
    print(f"  - {csv_path}")
    print(f"  - {count_csv_path}")
    print(f"  - {xlsx_path}")


if __name__ == "__main__":
    main()