# camels_utils.py
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Project paths (script is in path/codes/*.py)
# -----------------------------
def get_project_root() -> Path:
    env_root = os.environ.get('CAMELS_PROJECT_ROOT')
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def get_pickles_dir() -> Path:
    return get_project_root() / "Pickles"


def get_results_dir(script_name: str) -> Path:
    out = get_project_root() / "Results" / script_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def metric_dir(script_outdir: Path, metric: str) -> Path:
    """Creates/returns: Results/<script>/<metric>/"""
    d = script_outdir / str(metric)
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_basin_list(txt_path: Optional[Path] = None) -> List[str]:
    if txt_path is None:
        txt_path = get_project_root() / "531_basin_list.txt"
    return [b.strip() for b in txt_path.read_text().splitlines() if b.strip()]


def load_pickle(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)


# -----------------------------
# Metrics configuration
# -----------------------------
ALL_METRICS = [
    "NSE", "MSE", "RMSE", "KGE", "Alpha-NSE", "Pearson-r",
    "Beta-KGE", "Beta-NSE", "FHV", "FMS", "FLV",
    "Peak-MAPE", "MAPE", "Peak-Timing", "Missed-Peaks"
]

DEFAULT_MODELS = ["Benchmark", "Cluster-wise", "Top10"]


def model_display_name(m: str) -> str:
    ml = str(m).lower()
    if ml.startswith("bench"):
        return "Benchmark"
    if "cluster" in ml:
        return "Cluster-wise"
    if "top" in ml:
        return "Top10"
    return str(m)


def parse_models_arg(models: str) -> List[str]:
    ms = [model_display_name(x.strip()) for x in str(models).split(",") if x.strip()]
    out: List[str] = []
    for m in ms:
        if m not in out:
            out.append(m)
    return out


def parse_metrics_arg(metrics: Optional[str], all_metrics: bool) -> List[str]:
    if all_metrics or (metrics is None) or (str(metrics).strip().lower() == "all"):
        return list(ALL_METRICS)
    items = [m.strip() for m in str(metrics).split(",") if m.strip()]
    keep = [m for m in ALL_METRICS if m in set(items)]
    missing = [m for m in items if m not in set(ALL_METRICS)]
    if missing:
        print(f"[WARN] Unknown metrics ignored: {missing}")
    return keep


# -----------------------------
# Protocol (acceptance based on NSE only)
# -----------------------------
def nse_accept_threshold(context: str) -> float:
    """Acceptance threshold used throughout the paper (based on mean NSE over seeds)."""
    c = context.lower()
    if "event" in c or "highflow" in c or "normalized" in c:
        return 0.3
    return 0.5


def compute_nse_acceptance(
    df: pd.DataFrame,
    group_cols: List[str],
    threshold: float,
    metric_col: str = "metric",
    value_col: str = "value",
) -> pd.DataFrame:
    """
    Returns a table with:
      group_cols + ['nse_mean', 'accepted']
    Acceptance rule: mean(NSE over seeds) >= threshold
    """
    d = df[df[metric_col] == "NSE"].copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    out = (
        d.groupby(group_cols, dropna=False)[value_col]
         .mean()
         .reset_index()
         .rename(columns={value_col: "nse_mean"})
    )
    out["accepted"] = out["nse_mean"] >= float(threshold)
    return out


def filter_by_acceptance(
    df: pd.DataFrame,
    accept_df: pd.DataFrame,
    on_cols: List[str],
) -> pd.DataFrame:
    """Inner-join to accepted=True rows only."""
    keep = accept_df.loc[accept_df["accepted"] == True, on_cols].drop_duplicates()
    return df.merge(keep.assign(_keep=1), on=on_cols, how="inner").drop(columns=["_keep"])


# -----------------------------
# Metric-specific plotting rules
# -----------------------------
# Directions:
#   higher: larger is better
#   lower: smaller is better
#   abs0: closer to 0 is better
#
# For boxplots / comparisons (visual clarity): we drop *clearly bad* values
# based on your plotting notes. We NEVER clip; we filter.
MetricRule = Dict[str, object]


def _between(a: float, b: float) -> Callable[[pd.Series], pd.Series]:
    lo, hi = (a, b) if a <= b else (b, a)
    return lambda s: (s >= lo) & (s <= hi)


METRIC_RULES: Dict[str, MetricRule] = {
    "NSE":         dict(direction="higher", xlim=(-0.35, 1.0),  split=0.5,  keep=lambda s: s >= 0.0),
    "KGE":         dict(direction="higher", xlim=(-1.5, 1.0),   split=-0.3, keep=lambda s: s >= -0.3),
    "MSE":         dict(direction="lower",  xlim=(0.0, 30.0),   split=15.0, keep=lambda s: s <= 15.0),
    "RMSE":        dict(direction="lower",  xlim=(0.0, 6.0),    split=4.0,  keep=lambda s: s <= 4.0),
    "Alpha-NSE":   dict(direction="abs0",   xlim=(0.0, 1.5),    split=1.2,  keep=lambda s: s <= 1.2),
    "Pearson-r":   dict(direction="higher", xlim=(0.0, 1.0),    split=0.5,  keep=lambda s: s >= 0.0),
    "Beta-KGE":    dict(direction="lower",  xlim=(0.35, 3.5),   split=1.2,  keep=lambda s: s <= 1.2),
    "Beta-NSE":    dict(direction="abs0",   xlim=(-0.5, 0.2),   split=0.2,  keep=_between(-0.2, 0.2)),
    "FHV":         dict(direction="abs0",   xlim=(-90.0, 30.0), split=40.0, keep=lambda s: s >= -40.0),
    "FMS":         dict(direction="abs0",   xlim=(-90.0, 90.0), split=60.0, keep=_between(-60.0, 60.0)),
    "FLV":         dict(direction="abs0",   xlim=(-1.5, 100.0), split=20.0, keep=lambda s: s <= 100.0),
    "Peak-MAPE":   dict(direction="lower",  xlim=(10.0, 100.0), split=50.0, keep=lambda s: (s >= 10.0) & (s <= 100.0)),
    "MAPE":        dict(direction="lower",  xlim=(0.0, 100.0),  split=50.0, keep=lambda s: s <= 50.0),
    "Peak-Timing": dict(direction="lower",  xlim=(0.0, 2.0),    split=1.5,  keep=lambda s: s <= 1.5),
    "Missed-Peaks":dict(direction="lower",  xlim=(0.0, 1.0),    split=0.9,  keep=lambda s: s <= 0.9),
}


def metric_rule(metric: str) -> MetricRule:
    return METRIC_RULES.get(metric, dict(direction="higher", xlim=None, split=None, keep=lambda s: s.notna()))


def filter_metric_for_plot(df: pd.DataFrame, metric: str, value_col: str = "value") -> pd.DataFrame:
    """Apply the metric-specific keep-mask (visualization protocol)."""
    rule = metric_rule(metric)
    s = pd.to_numeric(df[value_col], errors="coerce")
    mask_fn = rule.get("keep", lambda x: x.notna())
    try:
        mask = mask_fn(s)
    except Exception:
        mask = s.notna()
    return df.loc[mask].copy()


# -----------------------------
# Plot helpers
# -----------------------------
def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([]), np.array([])
    xs = np.sort(x)
    ys = np.arange(1, xs.size + 1) / xs.size
    return xs, ys


def save_fig(fig: plt.Figure, outpath: Path, dpi: int = 200, tight_rect: tuple | None = None):
    """Save figure safely and create both PNG and PDF outputs.

    The publication runner monkey-patches Matplotlib as a second safety net,
    but this helper also saves a sibling PDF for scripts that use camels_utils.
    """
    try:
        from camels_publication_style import save_publication_figure, polish_figure
        if tight_rect is not None:
            try:
                l, b, r, t = tight_rect
                fig.subplots_adjust(left=l, bottom=b, right=r, top=t)
            except Exception:
                pass
        polish_figure(fig)
        save_publication_figure(fig, outpath, dpi=dpi, close=True)
        return
    except Exception:
        pass

    outpath.parent.mkdir(parents=True, exist_ok=True)
    if tight_rect is not None:
        try:
            l, b, r, t = tight_rect
            fig.subplots_adjust(left=l, bottom=b, right=r, top=t)
        except Exception:
            pass
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    if str(outpath).lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
        fig.savefig(Path(outpath).with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)



def annotate_box_stats(ax, data_list: List[np.ndarray], labels: List[str], fmt="{:.3f}"):
    """Write median and IQR inside the axes (safe for layout)."""
    y0, y1 = ax.get_ylim()
    yr = y1 - y0 if np.isfinite(y1 - y0) and (y1 - y0) > 0 else 1.0
    y_text = y1 - 0.06 * yr

    for i, (lab, arr) in enumerate(zip(labels, data_list), start=1):
        arr = np.asarray(arr)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        med = np.median(arr)
        q25 = np.quantile(arr, 0.25)
        q75 = np.quantile(arr, 0.75)
        txt = f"med={fmt.format(med)}\nIQR=[{fmt.format(q25)}, {fmt.format(q75)}]"
        ax.text(
            i, y_text, txt,
            ha="center", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", alpha=0.12, pad=0.2),
            clip_on=True,
        )



def boxplot_with_stats_panel(
    data_list: List[np.ndarray],
    labels: List[str],
    title: str,
    ylabel: str,
    figsize: tuple[float, float] = (8.0, 5.6),
    showfliers: bool = False,
    fmt: str = "{:.3f}",
):
    """Create a clean boxplot with a separate stats panel (no text clutter on the plot).

    Top panel: a small table of summary stats per group.
    Bottom panel: the boxplot.

    Stats shown: N, mean, median, q25, q75.
    """
    import matplotlib.gridspec as gridspec

    # Prepare stats
    rows = []
    for lab, arr in zip(labels, data_list):
        a = np.asarray(arr)
        a = a[np.isfinite(a)]
        if a.size == 0:
            rows.append([lab, "0", "nan", "nan", "nan", "nan"])
            continue
        mean = float(np.mean(a))
        med = float(np.median(a))
        q25 = float(np.quantile(a, 0.25))
        q75 = float(np.quantile(a, 0.75))
        rows.append([lab, str(int(a.size)), fmt.format(mean), fmt.format(med), fmt.format(q25), fmt.format(q75)])

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.0, 4.5], hspace=0.05)

    ax_tbl = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])

    # Table panel
    ax_tbl.axis("off")
    col_labels = ["Model", "N", "Mean", "Median", "Q25", "Q75"]
    table = ax_tbl.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.15)

    # Boxplot panel
    ax.boxplot(data_list, tick_labels=labels, showfliers=showfliers)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)

    return fig, ax


def agg_iqr(df: pd.DataFrame, group_cols: List[str], value_col: str) -> pd.DataFrame:
    """Aggregate mean + (Q25,Q75) + median for convenience.

    Returns columns: mean, median, q25, q75, n
    """
    out = (
        df.groupby(group_cols, dropna=False)
          .agg(
              n=(value_col, "size"),
              mean=(value_col, "mean"),
              median=(value_col, "median"),
              q25=(value_col, lambda s: s.quantile(0.25)),
              q75=(value_col, lambda s: s.quantile(0.75)),
          )
          .reset_index()
    )
    return out


def _split_masks(x: np.ndarray, direction: str, split: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return boolean masks (left, right) for the split-ECDF."""
    if direction == "lower":
        # smaller is better -> "bad" is above split
        m_left = x > split
        m_right = x <= split
    elif direction == "abs0":
        # closer to 0 is better -> left = outside [-split, +split]
        m_left = np.abs(x) > split
        m_right = np.abs(x) <= split
    else:
        # higher is better
        m_left = x < split
        m_right = x >= split
    return m_left, m_right


def plot_split_ecdf_metric(fig_title: str, series_by_model: Dict[str, np.ndarray], metric: str, rule: dict | None = None, xlabel: str | None = None, outpath: Path | None = None, dpi: int = 200, **kwargs):
    # Backward/forward compatible wrapper: ignore unused kwargs from older scripts
    if rule is None:
        rule = metric_rule(metric)
    if xlabel is None:
        xlabel = metric
    """Metric-aware ECDF plot with robust layout.

    - Legend + counts are placed INSIDE the canvas (no negative y).
    - If one side (bad/good) is empty for ALL models, we auto-switch to a single panel.
    - X-lims are tightened to actual data (bounded by the metric rule xlim).
    """
    rule = metric_rule(metric)
    if xlabel is None:
        xlabel = metric
    direction = str(rule.get("direction", "higher"))
    split = float(rule.get("split", 0.5) if rule.get("split", None) is not None else 0.5)
    xlim_rule = rule.get("xlim", None)

    def side_labels() -> tuple[str, str]:
        if direction == "lower":
            return f"> {split}", f"≤ {split}"
        if direction == "abs0":
            return f"|x| > {split}", f"|x| ≤ {split}"
        return f"< {split}", f"≥ {split}"

    prepared: dict[str, dict[str, np.ndarray]] = {}
    plotted_any_left = False
    plotted_any_right = False
    count_lines: list[str] = []

    for mname, x in series_by_model.items():
        x = np.asarray(x)
        x = x[np.isfinite(x)]
        if x.size == 0:
            prepared[mname] = dict(all_=x, left=np.array([]), right=np.array([]))
            count_lines.append(f"{mname}: N=0")
            continue

        m_left, m_right = _split_masks(x, direction, split)
        xl = x[m_left]
        xr = x[m_right]
        prepared[mname] = dict(all_=x, left=xl, right=xr)

        count_lines.append(f"{mname}: bad={int(xl.size)}  good={int(xr.size)}  (N={int(x.size)})")
        plotted_any_left = plotted_any_left or (xl.size > 0)
        plotted_any_right = plotted_any_right or (xr.size > 0)

    def _tight_xlim(arrays: list[np.ndarray], fallback: tuple[float, float] | None):
        arrays = [a for a in arrays if a is not None and len(a)]
        if not arrays:
            return fallback
        xs = np.concatenate([a[np.isfinite(a)] for a in arrays], axis=0)
        if xs.size == 0:
            return fallback
        lo = float(xs.min())
        hi = float(xs.max())
        pad = 0.03 * (hi - lo) if hi > lo else 0.5
        lo2, hi2 = lo - pad, hi + pad
        if fallback is None:
            return (lo2, hi2)
        return (max(lo2, fallback[0]), min(hi2, fallback[1]))

    use_two = plotted_any_left and plotted_any_right

    if use_two:
        fig = plt.figure(figsize=(12.0, 5.2))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.2], wspace=0.15)
        axL = fig.add_subplot(gs[0, 0])
        axR = fig.add_subplot(gs[0, 1])

        for ax in (axL, axR):
            ax.grid(True, alpha=0.3)
            ax.set_xlabel(xlabel)
        axL.set_ylabel("ECDF")

        left_lab, right_lab = side_labels()
        axL.set_title(f"{fig_title}\nBad: {left_lab}")
        axR.set_title(f"{fig_title}\nGood: {right_lab}")

        for mname, dd in prepared.items():
            if dd["left"].size:
                xs, ys = ecdf(dd["left"])
                if xs.size:
                    axL.plot(xs, ys, label=mname)
            if dd["right"].size:
                xs, ys = ecdf(dd["right"])
                if xs.size:
                    axR.plot(xs, ys, label=mname)

        # dynamic x-lims with split-aware bounds
        if xlim_rule is not None and direction == "higher":
            xl = _tight_xlim([prepared[m]["left"] for m in prepared], (xlim_rule[0], split))
            xr = _tight_xlim([prepared[m]["right"] for m in prepared], (split, xlim_rule[1]))
        elif xlim_rule is not None and direction == "lower":
            xl = _tight_xlim([prepared[m]["left"] for m in prepared], (split, xlim_rule[1]))
            xr = _tight_xlim([prepared[m]["right"] for m in prepared], (xlim_rule[0], split))
        else:
            xl = _tight_xlim([prepared[m]["left"] for m in prepared], xlim_rule)
            xr = _tight_xlim([prepared[m]["right"] for m in prepared], xlim_rule)

        if xl is not None: axL.set_xlim(*xl)
        if xr is not None: axR.set_xlim(*xr)

        handles, labels = axR.get_legend_handles_labels()
        if not handles:
            handles, labels = axL.get_legend_handles_labels()

        fig.subplots_adjust(bottom=0.18)
        if handles:
            fig.legend(handles, labels, loc="lower center",
                       ncol=max(1, len(labels)), frameon=False, bbox_to_anchor=(0.5, 0.06))
        fig.text(0.01, 0.015, " | ".join(count_lines), ha="left", va="bottom", fontsize=9)

        save_fig(fig, outpath, dpi=dpi, tight_rect=(0, 0.10, 1, 1))

    else:
        fig = plt.figure(figsize=(9.2, 5.2))
        ax = fig.add_subplot(111)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ECDF")

        side = "right" if plotted_any_right else "left"
        side_lab = side_labels()[1] if side == "right" else side_labels()[0]
        ax.set_title(f"{fig_title}\nShowing: {side_lab}")

        all_arrays: list[np.ndarray] = []
        for mname, dd in prepared.items():
            x = dd[side]
            if x.size:
                xs, ys = ecdf(x)
                if xs.size:
                    ax.plot(xs, ys, label=mname)
                all_arrays.append(x)

        xlim = _tight_xlim(all_arrays, xlim_rule)
        if xlim is not None:
            ax.set_xlim(*xlim)

        handles, labels = ax.get_legend_handles_labels()
        fig.subplots_adjust(bottom=0.18)
        if handles:
            fig.legend(handles, labels, loc="lower center",
                       ncol=max(1, len(labels)), frameon=False, bbox_to_anchor=(0.5, 0.06))
        fig.text(0.01, 0.015, " | ".join(count_lines), ha="left", va="bottom", fontsize=9)
        save_fig(fig, outpath, dpi=dpi, tight_rect=(0, 0.10, 1, 1))


# -----------------------------
# CLI builder
# -----------------------------
def build_common_argparser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--metrics", type=str, default="NSE", help="Comma-separated metrics to plot, or 'all'.")
    p.add_argument("--all-metrics", action="store_true", help="If set, plot all metrics regardless of --metrics.")
    p.add_argument("--models", type=str, default="Benchmark,Cluster-wise,Top10",
                   help="Comma-separated model names (default: Benchmark,Cluster-wise,Top10)")
    p.add_argument("--dpi", type=int, default=200)
    return p


# -----------------------------
# Flatteners for your pickle trees
# -----------------------------
def flatten_metrics_all_test_period(d: dict) -> pd.DataFrame:
    rows = []
    for basin, d_seed in d.items():
        for seed, d_model in d_seed.items():
            for model, d_met in d_model.items():
                model2 = model_display_name(model)
                for metric, value in d_met.items():
                    rows.append((basin, str(seed), model2, metric, value))
    df = pd.DataFrame(rows, columns=["basin", "seed", "model", "metric", "value"])
    df["value"] = safe_numeric(df["value"])
    return df


def flatten_metrics_by_water_year(d: dict) -> pd.DataFrame:
    rows = []
    for wy, d_basin in d.items():
        for basin, d_seed in d_basin.items():
            for seed, d_model in d_seed.items():
                for model, d_met in d_model.items():
                    model2 = model_display_name(model)
                    for metric, value in d_met.items():
                        rows.append((int(wy), basin, str(seed), model2, metric, value))
    df = pd.DataFrame(rows, columns=["water_year", "basin", "seed", "model", "metric", "value"])
    df["value"] = safe_numeric(df["value"])
    return df


def flatten_metrics_by_year_season(d: dict) -> pd.DataFrame:
    rows = []
    for wy, d_season in d.items():
        for season, d_basin in d_season.items():
            for basin, d_seed in d_basin.items():
                for seed, d_model in d_seed.items():
                    for model, d_met in d_model.items():
                        model2 = model_display_name(model)
                        for metric, value in d_met.items():
                            rows.append((int(wy), str(season), basin, str(seed), model2, metric, value))
    df = pd.DataFrame(rows, columns=["water_year", "season", "basin", "seed", "model", "metric", "value"])
    df["value"] = safe_numeric(df["value"])
    return df


def flatten_highflow_eventwise(d: dict) -> pd.DataFrame:
    rows = []
    for basin, d_event in d.items():
        for event_key, d_seed in d_event.items():
            for seed, d_model in d_seed.items():
                for model, d_met in d_model.items():
                    model2 = model_display_name(model)
                    for metric, value in d_met.items():
                        rows.append((basin, event_key, str(seed), model2, metric, value))
    df = pd.DataFrame(rows, columns=["basin", "event_key", "seed", "model", "metric", "value"])
    df["value"] = safe_numeric(df["value"])
    return df


def flatten_highflow_allbasins(d: dict) -> pd.DataFrame:
    return flatten_metrics_all_test_period(d)


BIN_ORDER = (
    [f"L{str(i).zfill(2)}" for i in range(10, 0, -1)] +
    ["P00"] +
    [f"R{str(i).zfill(2)}" for i in range(1, 11)]
)
BIN_TO_X = {b: (-(10 - i)) for i, b in enumerate([f"L{str(i).zfill(2)}" for i in range(10, 0, -1)], start=0)}
BIN_TO_X["P00"] = 0
for i in range(1, 11):
    BIN_TO_X[f"R{str(i).zfill(2)}"] = i


def flatten_highflow_normalized_bins(d: dict, stat_filter: Optional[set[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows_metrics = []
    rows_bins = []

    for event_key, d_basinwrap in d.items():
        for basin, d_seed in d_basinwrap.items():
            for seed, d_model in d_seed.items():
                for model, d_payload in d_model.items():
                    model2 = model_display_name(model)

                    mets = d_payload.get("metrics", {})
                    for met_name, met_obj in mets.items():
                        if isinstance(met_obj, dict) and "value" in met_obj:
                            val = met_obj.get("value", np.nan)
                        else:
                            val = met_obj
                        rows_metrics.append((event_key, basin, str(seed), model2, met_name, val))

                    bins = d_payload.get("bins", {})
                    for bin_name, bin_stats in bins.items():
                        if not isinstance(bin_stats, dict):
                            continue
                        for stat_name, stat_val in bin_stats.items():
                            if stat_filter is not None and stat_name not in stat_filter:
                                continue
                            rows_bins.append((event_key, basin, str(seed), model2, bin_name, stat_name, stat_val))

    df_metrics = pd.DataFrame.from_records(rows_metrics, columns=["event_key", "basin", "seed", "model", "metric", "value"])
    df_metrics["value"] = safe_numeric(df_metrics["value"])

    df_bins = pd.DataFrame.from_records(rows_bins, columns=["event_key", "basin", "seed", "model", "bin", "stat", "value"])
    df_bins["value"] = safe_numeric(df_bins["value"])

    return df_metrics, df_bins


def parse_event_start_date(event_key: str) -> Optional[pd.Timestamp]:
    try:
        parts = str(event_key).split("__")
        return pd.to_datetime(parts[2])
    except Exception:
        return None


def water_year_from_date(dt: pd.Timestamp) -> int:
    return int(dt.year + 1) if dt.month >= 10 else int(dt.year)

def season_from_date(dt: Optional[pd.Timestamp]) -> Optional[str]:
    """Return fixed 3-month seasons: JFM, AMJ, JAS, OND."""
    try:
        if dt is None or pd.isna(dt):
            return None
        m = int(pd.Timestamp(dt).month)
        if m in (1, 2, 3):
            return "JFM"
        if m in (4, 5, 6):
            return "AMJ"
        if m in (7, 8, 9):
            return "JAS"
        return "OND"
    except Exception:
        return None

