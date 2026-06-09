# make_07_summary_tables.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from camels_utils import (
    get_pickles_dir,
    get_project_root,   # <-- use this, not get_results_dir()
    load_pickle,
    safe_numeric,
    flatten_metrics_all_test_period,
    flatten_metrics_by_water_year,
    flatten_metrics_by_year_season,
    flatten_highflow_allbasins,
    flatten_highflow_eventwise,
    flatten_highflow_normalized_bins,  # returns (df_norm_metrics, df_bins)
)

# ---------------------------
# Helpers
# ---------------------------
def _save(df: pd.DataFrame, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{stem}.csv"
    df.to_csv(outpath, index=False)
    print(f"[OK] {outpath}")


def _ensure_event_id(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    if "event_id" in df.columns:
        return df
    if "event_key" in df.columns:
        d = df.copy()
        d["event_id"] = d["event_key"]
        return d
    for alt in ["event", "eid", "event_name", "event_idx"]:
        if alt in df.columns:
            d = df.copy()
            d["event_id"] = d[alt]
            return d
    return df


def _ensure_metric_col(df: pd.DataFrame) -> pd.DataFrame:
    """Bins often use 'stat' instead of 'metric'."""
    if df is None or len(df) == 0:
        return df
    if "metric" in df.columns:
        return df
    if "stat" in df.columns:
        d = df.copy()
        d["metric"] = d["stat"]
        return d
    for alt in ["met", "name", "measure"]:
        if alt in df.columns:
            d = df.copy()
            d["metric"] = d[alt]
            return d
    return df


def _summarize(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df is None or len(df) == 0:
        cols = list(group_cols) + [
            "model", "metric", "N_basins",
            "mean_of_seedmean", "median_of_seedmean",
            "min_of_seedmean", "max_of_seedmean",
            "p05", "p25", "p75", "p95", "std"
        ]
        return pd.DataFrame(columns=cols)

    d = df.copy()
    d = _ensure_metric_col(d)

    if "event_id" in group_cols and "event_id" not in d.columns:
        d = _ensure_event_id(d)

    # validate
    missing = [c for c in group_cols if c not in d.columns]
    if missing:
        raise KeyError(f"Missing group columns: {missing}. Available: {list(d.columns)}")

    required = ["basin", "seed", "model", "metric", "value"]
    missing_req = [c for c in required if c not in d.columns]
    if missing_req:
        raise KeyError(f"Missing required columns: {missing_req}. Available: {list(d.columns)}")

    d["value"] = safe_numeric(d["value"])

    # mean over seeds first
    seed_mean = (
        d.groupby(group_cols + ["basin", "model", "metric"], dropna=False)["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "seed_mean"})
    )

    def stats(x: pd.Series) -> pd.Series:
        x = x.dropna()
        if len(x) == 0:
            return pd.Series({
                "N_basins": 0,
                "mean_of_seedmean": np.nan,
                "median_of_seedmean": np.nan,
                "min_of_seedmean": np.nan,
                "max_of_seedmean": np.nan,
                "p05": np.nan, "p25": np.nan, "p75": np.nan, "p95": np.nan,
                "std": np.nan
            })
        return pd.Series({
            "N_basins": int(x.shape[0]),
            "mean_of_seedmean": float(x.mean()),
            "median_of_seedmean": float(x.median()),
            "min_of_seedmean": float(x.min()),
            "max_of_seedmean": float(x.max()),
            "p05": float(np.percentile(x, 5)),
            "p25": float(np.percentile(x, 25)),
            "p75": float(np.percentile(x, 75)),
            "p95": float(np.percentile(x, 95)),
            "std": float(x.std(ddof=0)),
        })

    out = (
        seed_mean.groupby(group_cols + ["model", "metric"], dropna=False)["seed_mean"]
        .apply(stats)
        .reset_index()
    )

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if c[1] == "" else c[1] for c in out.columns]

    return out


def _load_if_exists(p: Path) -> Optional[object]:
    if not p.exists():
        print(f"[WARN] Missing pickle: {p}")
        return None
    return load_pickle(p)


# ---------------------------
# Main
# ---------------------------
def main():
    pick = get_pickles_dir()

    # ✅ fixed forever: don’t call get_results_dir() (it needs script_name)
    outdir = get_project_root() / "Results" / "Tables"
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Whole hydrograph
    obj = _load_if_exists(pick / "metrics_all_test_period.p")
    if obj is not None:
        df = flatten_metrics_all_test_period(obj)
        _save(_summarize(df, group_cols=[]), outdir, "summary__whole_hydrograph__all_metrics")

    # 2) Water years
    obj = _load_if_exists(pick / "metrics_by_water_year.p")
    if obj is not None:
        df = flatten_metrics_by_water_year(obj)
        _save(_summarize(df, group_cols=[]), outdir, "summary__water_year_slices__all_metrics")
        if "water_year" in df.columns:
            _save(_summarize(df, group_cols=["water_year"]), outdir, "summary__water_year_by_year__all_metrics")

    # 3) Seasons
    obj = _load_if_exists(pick / "metrics_by_year_season.p")
    if obj is not None:
        df = flatten_metrics_by_year_season(obj)
        _save(_summarize(df, group_cols=[]), outdir, "summary__season_slices__all_metrics")
        if "season" in df.columns:
            _save(_summarize(df, group_cols=["season"]), outdir, "summary__season_by_season__all_metrics")

    # 4) Highflow allbasins
    obj = _load_if_exists(pick / "metrics_highflow_events_allbasins.p")
    if obj is not None:
        df = flatten_highflow_allbasins(obj)
        _save(_summarize(df, group_cols=[]), outdir, "summary__highflow_allbasins__all_metrics")

    # 5) Highflow eventwise
    obj = _load_if_exists(pick / "metrics_highflow_events_eventwise.p")
    if obj is not None:
        df_ev = flatten_highflow_eventwise(obj)
        _save(_summarize(df_ev, group_cols=[]), outdir, "summary__highflow_eventwise__all_metrics")

        df_ev2 = _ensure_event_id(df_ev)
        if df_ev2 is not None and len(df_ev2) > 0 and "event_id" in df_ev2.columns:
            _save(_summarize(df_ev2, group_cols=["event_id"]), outdir, "summary__highflow_eventwise_by_event__all_metrics")
        else:
            print("[WARN] Skipped eventwise-by-event table: no event_id/event_key found.")

    # 6) Normalized events + bins (tuple)
    obj = _load_if_exists(pick / "metrics_highflow_events_normalized_bins.p")
    if obj is not None:
        df_norm_metrics, df_bins = flatten_highflow_normalized_bins(obj)

        _save(_summarize(df_norm_metrics, group_cols=[]), outdir, "summary__highflow_normalized_events__all_metrics")
        if df_norm_metrics is not None and len(df_norm_metrics) > 0 and "event_key" in df_norm_metrics.columns:
            _save(_summarize(df_norm_metrics, group_cols=["event_key"]), outdir, "summary__highflow_normalized_events_by_event__all_metrics")

        _save(_summarize(df_bins, group_cols=[]), outdir, "summary__highflow_normalized_bins__all_metrics")
        if df_bins is not None and len(df_bins) > 0 and "bin" in df_bins.columns:
            _save(_summarize(df_bins, group_cols=["bin"]), outdir, "summary__highflow_normalized_bins_by_bin__all_metrics")
        else:
            print("[WARN] Skipped bins-by-bin table: no 'bin' column found.")

    print(f"\n[OK] All tables saved under: {outdir}")


if __name__ == "__main__":
    main()
