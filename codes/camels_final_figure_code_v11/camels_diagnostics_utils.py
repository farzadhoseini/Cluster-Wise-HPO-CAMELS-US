from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def zfill_basin(x) -> str:
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    return s.zfill(8)


def normalize_model_name(name: str) -> str:
    s = str(name).strip().lower()
    if s.startswith("bench"):
        return "Benchmark"
    if "cluster" in s:
        return "Cluster-wise"
    if "top" in s:
        return "Top10"
    return str(name)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_existing_file(base_dir: Path, stem_candidates: Iterable[str]) -> Path:
    exts = ["", ".p", ".pickle", ".pkl", ".csv", ".xlsx", ".xls"]
    for stem in stem_candidates:
        p0 = base_dir / stem
        if p0.exists():
            return p0
        for ext in exts:
            p = base_dir / f"{stem}{ext}"
            if p.exists():
                return p
    raise FileNotFoundError(
        f"Could not find any of these files in {base_dir}: {list(stem_candidates)}"
    )


def flatten_metrics_all_test_period(d: dict) -> pd.DataFrame:
    rows = []
    for basin, d_seed in d.items():
        for seed, d_model in d_seed.items():
            for model, d_met in d_model.items():
                model2 = normalize_model_name(model)
                for metric, value in d_met.items():
                    rows.append((zfill_basin(basin), str(seed), model2, metric, pd.to_numeric(value, errors="coerce")))
    return pd.DataFrame(rows, columns=["basin", "seed", "model", "metric", "value"])


def flatten_metrics_by_water_year(d: dict) -> pd.DataFrame:
    rows = []
    for wy, d_basin in d.items():
        for basin, d_seed in d_basin.items():
            for seed, d_model in d_seed.items():
                for model, d_met in d_model.items():
                    model2 = normalize_model_name(model)
                    for metric, value in d_met.items():
                        rows.append((int(wy), zfill_basin(basin), str(seed), model2, metric, pd.to_numeric(value, errors="coerce")))
    return pd.DataFrame(rows, columns=["water_year", "basin", "seed", "model", "metric", "value"])


def flatten_metrics_by_year_season(d: dict) -> pd.DataFrame:
    rows = []
    for wy, d_season in d.items():
        for season, d_basin in d_season.items():
            for basin, d_seed in d_basin.items():
                for seed, d_model in d_seed.items():
                    for model, d_met in d_model.items():
                        model2 = normalize_model_name(model)
                        for metric, value in d_met.items():
                            rows.append((int(wy), str(season), zfill_basin(basin), str(seed), model2, metric, pd.to_numeric(value, errors="coerce")))
    return pd.DataFrame(rows, columns=["water_year", "season", "basin", "seed", "model", "metric", "value"])


def load_master_attributes(project_root: Path) -> pd.DataFrame:
    candidates = [
        "camels_531_master_attributes_clusters.csv",
        "camels_531_master_attributes.csv",
    ]
    p = resolve_existing_file(project_root, candidates)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        df = pd.read_excel(p)
    if "gauge_id" not in df.columns:
        # try a softer detection
        for c in df.columns:
            if c.lower().replace("_", "") in {"gaugeid", "basin", "basinid"}:
                df = df.rename(columns={c: "gauge_id"})
                break
    if "cluster" not in df.columns:
        raise KeyError(f"cluster column not found in {p}")
    df = df.copy()
    df["basin"] = df["gauge_id"].map(zfill_basin)
    cl = pd.to_numeric(df["cluster"], errors="coerce")
    vals = sorted(v for v in cl.dropna().unique().tolist())
    if vals == [0, 1, 2, 3, 4, 5]:
        cl = cl + 1
    df["cluster"] = cl.astype("Int64")
    df["cluster_label"] = df["cluster"].map(lambda x: f"Cluster {int(x)}" if pd.notna(x) else np.nan)
    return df


def load_train_data(train_data_path: Path) -> dict:
    return load_pickle(train_data_path)


def _basin_coord_values(train_data: dict, coord_name: str):
    obj = train_data["coords"][coord_name]
    if isinstance(obj, dict) and "data" in obj:
        return np.asarray(obj["data"])
    return np.asarray(obj)



def _safe_row_nanmean(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    out = np.full(arr.shape[0], np.nan, dtype=float)
    valid_rows = np.isfinite(arr).any(axis=1)
    if np.any(valid_rows):
        with np.errstate(invalid="ignore"):
            out[valid_rows] = np.nanmean(arr[valid_rows], axis=1)
    return out


def _safe_row_nanstd(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    out = np.full(arr.shape[0], np.nan, dtype=float)
    valid_rows = np.isfinite(arr).sum(axis=1) > 1
    if np.any(valid_rows):
        with np.errstate(invalid="ignore"):
            out[valid_rows] = np.nanstd(arr[valid_rows], axis=1)
    return out


def _safe_row_nanmin(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    out = np.full(arr.shape[0], np.nan, dtype=float)
    valid_rows = np.isfinite(arr).any(axis=1)
    if np.any(valid_rows):
        with np.errstate(invalid="ignore"):
            out[valid_rows] = np.nanmin(arr[valid_rows], axis=1)
    return out


def _safe_row_nanmax(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    out = np.full(arr.shape[0], np.nan, dtype=float)
    valid_rows = np.isfinite(arr).any(axis=1)
    if np.any(valid_rows):
        with np.errstate(invalid="ignore"):
            out[valid_rows] = np.nanmax(arr[valid_rows], axis=1)
    return out


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(num.shape, np.nan, dtype=float)
    valid = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out[valid] = num[valid] / den[valid]
    return out


def _grouped_row_nanmean(arr: np.ndarray, labels, order=None):
    arr = np.asarray(arr, dtype=float)
    labels = np.asarray(labels)
    if order is None:
        order = pd.unique(labels)

    mats = []
    names = []
    for lab in order:
        mask = labels == lab
        if not np.any(mask):
            mats.append(np.full(arr.shape[0], np.nan, dtype=float))
            names.append(lab)
            continue
        mats.append(_safe_row_nanmean(arr[:, mask]))
        names.append(lab)
    return np.column_stack(mats), list(names)


def extract_train_quality(train_data: dict) -> pd.DataFrame:
    basins = [zfill_basin(x) for x in _basin_coord_values(train_data, "basin").tolist()]
    out = pd.DataFrame({"basin": basins})

    def arr(name: str):
        if name not in train_data["data_vars"]:
            return None
        return np.asarray(train_data["data_vars"][name]["data"], dtype=float)

    def arr_first(names):
        for name in names:
            value = arr(name)
            if value is not None:
                return value
        return None

    q = arr("QObs(mm/d)")
    p = arr_first(["PRCP(mm/day)_maurer", "PRCP(mm/day)_nldas"])
    tmax = arr_first(["Tmax(C)_maurer", "Tmax(C)_nldas"])
    tmin = arr_first(["Tmin(C)_maurer", "Tmin(C)_nldas"])

    if q is not None:
        q_valid = np.isfinite(q)
        out["qobs_missing_frac"] = 1.0 - q_valid.mean(axis=1)
        q_zero = np.where(q_valid, np.where(q == 0.0, 1.0, 0.0), np.nan)
        out["qobs_zero_frac"] = _safe_row_nanmean(q_zero)
        out["qobs_mean_train"] = _safe_row_nanmean(q)
        out["qobs_std_train"] = _safe_row_nanstd(q)
        out["qobs_cv_train"] = out["qobs_std_train"] / out["qobs_mean_train"].replace(0, np.nan)
        out["qobs_nonmissing_days"] = np.sum(q_valid, axis=1)
        out["qobs_allnan_flag"] = (~q_valid.any(axis=1)).astype(int)

    if p is not None:
        p_valid = np.isfinite(p)
        out["prcp_missing_frac"] = 1.0 - p_valid.mean(axis=1)
        p_zero = np.where(p_valid, np.where(p == 0.0, 1.0, 0.0), np.nan)
        out["prcp_zero_frac"] = _safe_row_nanmean(p_zero)
        out["prcp_mean_train"] = _safe_row_nanmean(p)
        out["prcp_std_train"] = _safe_row_nanstd(p)
        out["prcp_allnan_flag"] = (~p_valid.any(axis=1)).astype(int)

    if tmax is not None:
        tmax_valid = np.isfinite(tmax)
        out["tmax_missing_frac"] = 1.0 - tmax_valid.mean(axis=1)
        out["tmax_allnan_flag"] = (~tmax_valid.any(axis=1)).astype(int)
    if tmin is not None:
        tmin_valid = np.isfinite(tmin)
        out["tmin_missing_frac"] = 1.0 - tmin_valid.mean(axis=1)
        out["tmin_allnan_flag"] = (~tmin_valid.any(axis=1)).astype(int)

    temp_cols = [c for c in ["tmax_missing_frac", "tmin_missing_frac"] if c in out.columns]
    if temp_cols:
        out["temp_missing_frac"] = out[temp_cols].mean(axis=1)
    flag_cols = [c for c in ["qobs_allnan_flag", "prcp_allnan_flag", "tmax_allnan_flag", "tmin_allnan_flag"] if c in out.columns]
    if flag_cols:
        out["n_allnan_core_vars"] = out[flag_cols].sum(axis=1)

    return out


def seasonal_order_key(season: str) -> int:
    season = str(season).upper()
    order = {"JFM": 1, "AMJ": 2, "JAS": 3, "OND": 4}
    return order.get(season, 99)


def summarize_group_numeric(df: pd.DataFrame, group_name: str, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        rows.append({
            group_name: c,
            "n": int(s.notna().sum()),
            "mean": float(s.mean()) if s.notna().any() else np.nan,
            "median": float(s.median()) if s.notna().any() else np.nan,
            "q25": float(s.quantile(0.25)) if s.notna().any() else np.nan,
            "q75": float(s.quantile(0.75)) if s.notna().any() else np.nan,
            "min": float(s.min()) if s.notna().any() else np.nan,
            "max": float(s.max()) if s.notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def safe_effect_size(bad: pd.Series, rest: pd.Series) -> float:
    bad = pd.to_numeric(bad, errors="coerce").dropna()
    rest = pd.to_numeric(rest, errors="coerce").dropna()
    if len(bad) < 2 or len(rest) < 2:
        return np.nan
    pooled = np.sqrt(((bad.var(ddof=1) + rest.var(ddof=1)) / 2.0))
    if pooled == 0 or not np.isfinite(pooled):
        return np.nan
    return (bad.mean() - rest.mean()) / pooled
