
# plot_02_water_year_v03.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from camels_publication_style import get_model_color, get_model_linestyle

from camels_utils import (
    get_pickles_dir, get_results_dir, metric_dir, load_pickle,
    flatten_metrics_by_water_year,
    build_common_argparser, parse_metrics_arg, parse_models_arg,
    agg_iqr, save_fig, ecdf
)

MODEL_ORDER = ["Benchmark", "Cluster-wise", "Top10"]
TREND_ORDER = ["Benchmark", "Top10", "Cluster-wise"]

MODEL_DISPLAY = {
    "Benchmark": "Benchmark",
    "Cluster-wise": "Cluster-wise",
    "Top10": "Top10",
}

# Optional broad metric bounds, used only as soft clipping for heatmap color limits.
METRIC_BOUNDS = {
    "NSE": (-1.0, 1.0),
    "KGE": (-1.0, 1.0),
    "Pearson-r": (-1.0, 1.0),
    "Pearson_r": (-1.0, 1.0),
    "Pearson": (-1.0, 1.0),
}

def _ordered_models(models):
    order = [m for m in MODEL_ORDER if m in models]
    order += [m for m in models if m not in order]
    return order

def _summary_rows(data_arrays, labels):
    rows = []
    for lab, arr in zip(labels, data_arrays):
        x = np.asarray(arr, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            rows.append([lab, 0, np.nan, np.nan, np.nan, np.nan])
        else:
            rows.append([
                MODEL_DISPLAY.get(lab, lab),
                len(x),
                float(np.mean(x)),
                float(np.median(x)),
                float(np.percentile(x, 25)),
                float(np.percentile(x, 75)),
            ])
    return rows

def _add_summary_table(ax_tbl, data, labels, fmt="{:.3f}"):
    rows = _summary_rows(data, labels)
    header = ["Model", "N", "Mean", "Median", "Q25", "Q75"]
    cell_text = []
    numeric_cols = {2: [], 3: [], 4: [], 5: []}
    for r in rows:
        cell_text.append([
            r[0],
            f"{r[1]:d}",
            fmt.format(r[2]) if np.isfinite(r[2]) else "NA",
            fmt.format(r[3]) if np.isfinite(r[3]) else "NA",
            fmt.format(r[4]) if np.isfinite(r[4]) else "NA",
            fmt.format(r[5]) if np.isfinite(r[5]) else "NA",
        ])
        for j in numeric_cols:
            numeric_cols[j].append(r[j])
    max_idx = {}
    for j, vals in numeric_cols.items():
        arr = np.asarray(vals, dtype=float)
        arr[~np.isfinite(arr)] = -np.inf
        max_idx[j] = int(np.nanargmax(arr)) if np.isfinite(arr).any() else None
    ax_tbl.axis("off")
    tbl = ax_tbl.table(
        cellText=cell_text,
        colLabels=header,
        loc="center",
        cellLoc="center",
        bbox=[0.02, 0.02, 0.96, 0.90],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12.0)
    tbl.scale(1.0, 1.24)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("0.35")
        cell.set_linewidth(0.85)
        if r == 0:
            cell.set_facecolor("#e8ecef")
            cell.get_text().set_fontweight("bold")
        elif c == 0:
            cell.set_facecolor("#f1f3f5")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("white" if r % 2 == 0 else "#fbfbfb")
        if r > 0 and c in max_idx and max_idx[c] is not None and (r - 1) == max_idx[c]:
            cell.get_text().set_fontweight("bold")
    return tbl


def boxplot_with_stats_table_below(
    data, labels, title, ylabel, figsize=(8.8, 6.6), showfliers=False, fmt="{:.3f}"
):
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[4.0, 1.25], hspace=0.10)

    ax = fig.add_subplot(gs[0])
    ax_tbl = fig.add_subplot(gs[1])

    bp = ax.boxplot(
        data,
        labels=[MODEL_DISPLAY.get(l, l) for l in labels],
        showfliers=showfliers,
        widths=0.42,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.4),
        whiskerprops=dict(color="0.20", linewidth=1.0),
        capprops=dict(color="0.20", linewidth=1.0),
        boxprops=dict(edgecolor="0.20", linewidth=1.0),
    )
    for box, lab in zip(bp["boxes"], labels):
        box.set_facecolor(get_model_color(lab))
        box.set_alpha(0.88)
    ax.set_title(title, fontsize=20, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(axis="x", labelsize=13, pad=6)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    _add_summary_table(ax_tbl, data, labels, fmt=fmt)
    fig.subplots_adjust(top=0.91, bottom=0.06, left=0.10, right=0.98)
    return fig, ax

def _infer_metric_bounds(metric_name):
    return METRIC_BOUNDS.get(metric_name, None)

def _compute_heatmap_limits(mat, metric_name, mode="robust"):
    """
    Compute informative color limits for heatmaps.
    mode='robust' -> uses central spread of actual values, not the full generic metric range.
    """
    vals = mat[np.isfinite(mat)]
    if vals.size == 0:
        return None, None

    metric_bounds = _infer_metric_bounds(metric_name)

    if mode == "full" and metric_bounds is not None:
        return metric_bounds

    # Robust local range
    q_low, q_high = np.quantile(vals, [0.05, 0.95])
    data_min = float(vals.min())
    data_max = float(vals.max())

    # Build a slightly padded local range
    low = min(data_min, q_low)
    high = max(data_max, q_high)

    span = high - low
    if span < 1e-10:
        # fallback for near-constant matrix
        center = float(np.mean(vals))
        span = max(abs(center) * 0.02, 0.01)
        low = center - span
        high = center + span
    else:
        pad = 0.12 * span
        low -= pad
        high += pad

    # Soft clipping to known metric bounds if available
    if metric_bounds is not None:
        bound_low, bound_high = metric_bounds
        low = max(low, bound_low)
        high = min(high, bound_high)

    # Final safety
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        low = float(vals.min())
        high = float(vals.max())
        if np.isclose(low, high):
            low -= 0.01
            high += 0.01

    return low, high

def _annotate_heatmap(ax, mat, vmin, vmax, fmt="{:.3f}"):
    if vmin is None or vmax is None:
        return
    midpoint = (vmin + vmax) / 2.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                color = "white" if val < midpoint else "black"
                ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=9, color=color)

def main():
    parser = build_common_argparser("CAMELS-US: Water-year metrics (trend + distributions + heatmaps).")
    parser.add_argument("--year-min", type=int, default=None)
    parser.add_argument("--year-max", type=int, default=None)
    parser.add_argument("--heatmap-scale", type=str, default="robust", choices=["robust", "full"],
                        help="robust: adaptive local range (recommended); full: full metric bound range.")
    parser.add_argument("--annotate-heatmap", action="store_true",
                        help="Write values on heatmap cells.")
    parser.add_argument(
        "--seed-years",
        default="1992,1995,1998",
        help=(
            "Comma-separated water years for seed-robustness WY figures. "
            "Use 'all' to reproduce the old exploratory behavior. "
            "Default keeps only the final-paper appendix/reference figures."
        ),
    )
    args = parser.parse_args()

    if str(args.seed_years).strip().lower() == "all":
        selected_seed_years = None
    else:
        selected_seed_years = {int(x.strip()) for x in str(args.seed_years).split(",") if x.strip()}

    script_name = Path(__file__).stem
    outdir = get_results_dir(script_name)

    metrics = parse_metrics_arg(args.metrics, args.all_metrics)
    models = _ordered_models(parse_models_arg(args.models))

    d = load_pickle(get_pickles_dir() / "metrics_by_water_year.p")
    df = flatten_metrics_by_water_year(d)
    df = df[df["model"].isin(models)]

    if args.year_min is not None:
        df = df[df["water_year"] >= args.year_min]
    if args.year_max is not None:
        df = df[df["water_year"] <= args.year_max]

    years = sorted(df["water_year"].dropna().unique().tolist())

    # basin mean over seeds per year
    df_basin_mean = (
        df.groupby(["water_year", "basin", "model", "metric"], as_index=False)["value"]
          .mean()
          .rename(columns={"value": "basin_mean"})
    )

    # per year summary across basins: median + IQR of basin means
    df_year_summary = agg_iqr(df_basin_mean, ["water_year", "model", "metric"], "basin_mean")

    # seed mean over basins per year
    df_seed_year = (
        df.groupby(["water_year", "seed", "model", "metric"], as_index=False)["value"]
          .mean()
          .rename(columns={"value": "seed_mean"})
    )

    for met in metrics:
        outm = metric_dir(outdir, met)

        # ---- Trend across years with IQR ribbon
        fig = plt.figure(figsize=(10.0, 5.6))
        ax = fig.add_subplot(111)
        trend_models = [m for m in TREND_ORDER if m in models] + [m for m in models if m not in TREND_ORDER]
        for m in trend_models:
            dd = df_year_summary[(df_year_summary["metric"] == met) & (df_year_summary["model"] == m)].sort_values("water_year")
            ax.plot(dd["water_year"].to_numpy(), dd["median"].to_numpy(), label=MODEL_DISPLAY.get(m, m), linewidth=2.4, color=get_model_color(m), linestyle=get_model_linestyle(m))
            ax.fill_between(dd["water_year"].to_numpy(), dd["q25"].to_numpy(), dd["q75"].to_numpy(), alpha=0.16, color=get_model_color(m))
        ax.set_title(f"Water-year evolution of median basin-wise {met} across the 531 CAMELS-US basins", fontsize=20, fontweight="bold", pad=12)
        ax.set_xlabel("Water year", fontsize=15)
        ax.set_ylabel(met, fontsize=15)
        ax.tick_params(labelsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=3,
            frameon=True,
            fontsize=11.0,
            title=None,
        )
        fig.subplots_adjust(top=0.90, bottom=0.22, left=0.10, right=0.98)
        save_fig(fig, outm / f"Trend_water_year__{met}.png", dpi=args.dpi)

        # ---- Heatmap: years x models (median across basins of basin means)
        mat = np.full((len(years), len(models)), np.nan, dtype=float)
        for i, y in enumerate(years):
            for j, m in enumerate(models):
                v = df_year_summary[
                    (df_year_summary["metric"] == met) &
                    (df_year_summary["water_year"] == y) &
                    (df_year_summary["model"] == m)
                ]["median"]
                if len(v) > 0:
                    mat[i, j] = float(v.iloc[0])

        fig = plt.figure(figsize=(7.8, max(4.6, 0.38 * len(years))))
        ax = fig.add_subplot(111)

        vmin, vmax = _compute_heatmap_limits(mat, met, mode=args.heatmap_scale)
        im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax, cmap="cividis", interpolation="nearest")
        ax.grid(False)
        ax.tick_params(which="both", length=0)

        ax.set_title(f"Heatmap of median basin-wise {met} by water year and benchmark framework", fontsize=19, fontweight="bold", pad=12)
        ax.set_xlabel("Model", fontsize=14)
        ax.set_ylabel("Water year", fontsize=14)
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models], rotation=0, fontsize=12)
        ax.set_yticks(np.arange(len(years)))
        ax.set_yticklabels(years, fontsize=11)

        if args.annotate_heatmap:
            _annotate_heatmap(ax, mat, vmin, vmax, fmt="{:.3f}")

        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(labelsize=11)
        fig.subplots_adjust(top=0.90, bottom=0.10, left=0.16, right=0.94)
        save_fig(fig, outm / f"Heatmap_year_x_model__{met}.png", dpi=args.dpi)

        # ---- Box across basins per year (one figure per model)
        for m in models:
            dd = df_basin_mean[(df_basin_mean["metric"] == met) & (df_basin_mean["model"] == m)]
            groups = [dd.loc[dd["water_year"] == y, "basin_mean"].dropna().to_numpy() for y in years]
            if len(years) == 0:
                continue

            fig = plt.figure(figsize=(max(7.2, 0.28 * len(years)), 5.4))
            ax = fig.add_subplot(111)
            bp = ax.boxplot(groups, tick_labels=years, showfliers=False, widths=0.58, patch_artist=True,
                            medianprops=dict(color="black", linewidth=1.2),
                            whiskerprops=dict(color="0.20", linewidth=0.9),
                            capprops=dict(color="0.20", linewidth=0.9),
                            boxprops=dict(edgecolor="0.20", linewidth=0.9))
            for box in bp["boxes"]:
                box.set_facecolor(get_model_color(m))
                box.set_alpha(0.86)
            ax.set_title("", pad=2)
            ax.set_xlabel("Water year", fontsize=14)
            ax.set_ylabel(met, fontsize=14)
            ax.tick_params(axis="x", labelsize=10, rotation=90)
            ax.tick_params(axis="y", labelsize=12)
            ax.grid(True, axis="y", alpha=0.3)
            fig.subplots_adjust(top=0.89, bottom=0.24, left=0.09, right=0.98)
            save_fig(fig, outm / f"Box_basins_per_year__{m}__{met}.png", dpi=args.dpi)

        # ---- Seed robustness per year: box + CDF.
        # Final-paper mode: generate only the selected representative WY figures
        # listed in the mapping/reference archive, not one figure for every year.
        seed_years_to_plot = years if selected_seed_years is None else sorted([y for y in years if y in selected_seed_years])
        for y in seed_years_to_plot:
            dd = df_seed_year[(df_seed_year["metric"] == met) & (df_seed_year["water_year"] == y)]
            data = [dd.loc[dd["model"] == m, "seed_mean"].dropna().to_numpy() for m in models]

            fig, ax = boxplot_with_stats_table_below(
                data,
                models,
                title=f"Distribution of seed-level {met} values in WY{y} after averaging over the 531 CAMELS-US basins",
                ylabel=f"{met} (seed mean)",
                figsize=(8.8, 6.6),
                showfliers=False,
                fmt="{:.3f}",
            )
            save_fig(fig, outm / f"BOX_seeds_WY{y}__{met}.png", dpi=args.dpi)

            fig = plt.figure(figsize=(8.8, 5.9))
            ax = fig.add_subplot(111)
            for m in models:
                x = dd.loc[dd["model"] == m, "seed_mean"].to_numpy()
                x = x[np.isfinite(x)]
                xs, ys = ecdf(x)
                ax.plot(xs, ys, label=MODEL_DISPLAY.get(m, m), linewidth=2.3, color=get_model_color(m), linestyle=get_model_linestyle(m))
            ax.set_title(f"CDF of seed-level {met} values in WY{y} after averaging over the 531 CAMELS-US basins", fontsize=18, fontweight="bold", pad=12)
            ax.set_xlabel(f"{met} (seed mean)", fontsize=14)
            ax.set_ylabel("CDF", fontsize=14)
            ax.tick_params(labelsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(0.03, 0.98),
                ncol=1,
                frameon=True,
                fontsize=11.5,
                title="Model",
                title_fontsize=11.5,
            )
            fig.subplots_adjust(top=0.89, bottom=0.12, left=0.10, right=0.98)
            save_fig(fig, outm / f"CDF_seeds_WY{y}__{met}.png", dpi=args.dpi)

    print(f"[OK] Saved improved figures to: {outdir}")
    print("Recommended for heatmaps:")
    print("  --heatmap-scale robust --annotate-heatmap")
    print("This avoids washed-out colors and makes between-model differences visible.")

if __name__ == "__main__":
    main()
