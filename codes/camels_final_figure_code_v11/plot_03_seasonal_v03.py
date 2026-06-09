from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from camels_utils import (
    get_pickles_dir, get_results_dir, metric_dir, load_pickle,
    flatten_metrics_by_year_season,
    build_common_argparser, parse_metrics_arg, parse_models_arg,
    save_fig
)

SEASONS = ["JFM", "AMJ", "JAS", "OND"]
MODEL_ORDER = ["Benchmark", "Cluster-wise", "Top10"]
MODEL_DISPLAY = {
    "Benchmark": "Benchmark",
    "Cluster-wise": "Cluster-wise",
    "Top10": "Top10",
}


SEASON_CMAPS = {
    "JFM": "YlGn",
    "AMJ": "YlOrRd",
    "JAS": "plasma",
    "OND": "PuBuGn",
}

def _season_limits(values, pad_frac=0.08):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None
    vmin = float(vals.min())
    vmax = float(vals.max())
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.02, 0.01)
        return vmin - pad, vmax + pad
    pad = pad_frac * (vmax - vmin)
    return vmin - pad, vmax + pad



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
    for r in rows:
        cell_text.append([
            r[0],
            f"{r[1]:d}",
            fmt.format(r[2]) if np.isfinite(r[2]) else "NA",
            fmt.format(r[3]) if np.isfinite(r[3]) else "NA",
            fmt.format(r[4]) if np.isfinite(r[4]) else "NA",
            fmt.format(r[5]) if np.isfinite(r[5]) else "NA",
        ])
    ax_tbl.axis("off")
    tbl = ax_tbl.table(
        cellText=cell_text,
        colLabels=header,
        loc="center",
        cellLoc="center",
        bbox=[0.02, 0.06, 0.96, 0.88],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.15)
    return tbl


def boxplot_with_stats_table_below(
    data, labels, title, ylabel, figsize=(8.8, 6.6), showfliers=False, fmt="{:.3f}"
):
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[4.0, 1.25], hspace=0.08)

    ax = fig.add_subplot(gs[0])
    ax_tbl = fig.add_subplot(gs[1])

    ax.boxplot(
        data,
        tick_labels=[MODEL_DISPLAY.get(l, l) for l in labels],
        showfliers=showfliers,
        widths=0.36,
        patch_artist=False,
    )
    ax.set_title(title, fontsize=17, pad=10)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(axis="x", labelsize=13, pad=6)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    _add_summary_table(ax_tbl, data, labels, fmt=fmt)
    fig.subplots_adjust(top=0.91, bottom=0.06, left=0.10, right=0.98)
    return fig, ax


def _adaptive_limits(values, pad_frac=0.06):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None, None
    if vmin == vmax:
        pad = 0.02 * max(1.0, abs(vmin))
        return vmin - pad, vmax + pad

    pad = pad_frac * (vmax - vmin)
    return vmin - pad, vmax + pad


def _build_heatmap_summary_table(ax_tbl, mats, models, years, fmt="{:.3f}"):
    header = ["Model"] + SEASONS + ["Overall"]
    cell_text = []

    for m in models:
        mat = mats[m]
        season_medians = []
        for j in range(len(SEASONS)):
            col = mat[:, j]
            col = col[np.isfinite(col)]
            season_medians.append(float(np.median(col)) if col.size else np.nan)
        overall = np.asarray(season_medians, dtype=float)
        overall = overall[np.isfinite(overall)]
        overall = float(np.mean(overall)) if overall.size else np.nan
        row = [MODEL_DISPLAY.get(m, m)]
        row += [fmt.format(v) if np.isfinite(v) else "NA" for v in season_medians]
        row += [fmt.format(overall) if np.isfinite(overall) else "NA"]
        cell_text.append(row)

    ax_tbl.axis("off")
    tbl = ax_tbl.table(
        cellText=cell_text,
        colLabels=header,
        loc="center",
        cellLoc="center",
        bbox=[0.08, 0.02, 0.84, 0.92],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.22)
    return tbl


def main():
    parser = build_common_argparser("CAMELS-US: Seasonal metrics (pooled distributions + heatmaps).")
    parser.add_argument("--year-min", type=int, default=None)
    parser.add_argument("--year-max", type=int, default=None)
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Generate only final mapped seasonal figure(s), not exploratory pooled-season boxes or all-model heatmaps.",
    )
    args = parser.parse_args()

    script_name = Path(__file__).stem
    outdir = get_results_dir(script_name)
    metrics = parse_metrics_arg(args.metrics, args.all_metrics)
    models = _ordered_models(parse_models_arg(args.models))

    d = load_pickle(get_pickles_dir() / "metrics_by_year_season.p")
    df = flatten_metrics_by_year_season(d)
    df = df[df["model"].isin(models)]

    if args.year_min is not None:
        df = df[df["water_year"] >= args.year_min]
    if args.year_max is not None:
        df = df[df["water_year"] <= args.year_max]

    df_basin_mean = (
        df.groupby(["water_year", "season", "basin", "model", "metric"], as_index=False)["value"]
          .mean()
          .rename(columns={"value": "basin_mean"})
    )

    df_grid = (
        df_basin_mean.groupby(["water_year", "season", "model", "metric"], as_index=False)["basin_mean"]
                    .median()
                    .rename(columns={"basin_mean": "median_over_basins"})
    )

    years = sorted(df_grid["water_year"].dropna().unique().tolist())

    for met in metrics:
        outm = metric_dir(outdir, met)

        if not args.final_only:
            for season in SEASONS:
                dd = df_basin_mean[(df_basin_mean["metric"] == met) & (df_basin_mean["season"] == season)]
                data = [dd.loc[dd["model"] == m, "basin_mean"].dropna().to_numpy() for m in models]

                fig, ax = boxplot_with_stats_table_below(
                    data,
                    models,
                    title=f"{season}: across basins (mean over seeds, pooled years) — {met}",
                    ylabel=met,
                    figsize=(8.6, 6.4),
                    showfliers=False,
                    fmt="{:.3f}",
                )
                save_fig(fig, outm / f"BOX_{season}_pooledYears__{met}.png", dpi=args.dpi)

        ddh = df_grid[df_grid["metric"] == met].copy()

        mats = {}
        all_vals = []
        for m in models:
            mat = np.full((len(years), len(SEASONS)), np.nan, dtype=float)
            for i, y in enumerate(years):
                for j, seas in enumerate(SEASONS):
                    v = ddh[
                        (ddh["model"] == m) &
                        (ddh["water_year"] == y) &
                        (ddh["season"] == seas)
                    ]["median_over_basins"]
                    if len(v) > 0 and np.isfinite(v.iloc[0]):
                        mat[i, j] = float(v.iloc[0])
            mats[m] = mat
            all_vals.append(mat[np.isfinite(mat)])

        vv = np.concatenate([x for x in all_vals if x.size]) if any(x.size for x in all_vals) else np.array([])
        vmin, vmax = _adaptive_limits(vv, pad_frac=0.08)

        if not args.final_only:
            # 1) Main seasonal heatmap: 3 model panels + summary table below
            fig = plt.figure(figsize=(15.0, max(7.8, 0.36 * len(years) + 3.3)), constrained_layout=False)
            gs = fig.add_gridspec(
                2,
                len(models),
                height_ratios=[4.0, 1.45],
                hspace=0.12,
                wspace=0.10,
            )

            im = None
            for j, m in enumerate(models):
                ax = fig.add_subplot(gs[0, j])
                im = ax.imshow(mats[m], aspect="auto", vmin=vmin, vmax=vmax, cmap="cividis", interpolation="nearest")
                ax.grid(False)
                ax.tick_params(which="both", length=0)
                ax.set_title(MODEL_DISPLAY.get(m, m), fontsize=15, pad=8)
                ax.set_xlabel("Season", fontsize=12)
                ax.set_xticks(np.arange(len(SEASONS)))
                ax.set_xticklabels(SEASONS, fontsize=13, fontweight="bold")
                if j == 0:
                    ax.set_ylabel("Water year", fontsize=13)
                    ax.set_yticks(np.arange(len(years)))
                    ax.set_yticklabels([str(y) for y in years], fontsize=12)
                else:
                    ax.set_yticks([])

                midpoint = None if (vmin is None or vmax is None) else (vmin + vmax) / 2
                for i in range(len(years)):
                    for k in range(len(SEASONS)):
                        val = mats[m][i, k]
                        if np.isfinite(val):
                            txt_color = "white" if (midpoint is not None and val < midpoint) else "black"
                            ax.text(k, i, f"{val:.3f}", ha="center", va="center", fontsize=12.5, fontweight="bold", color=txt_color)

            cax = fig.add_axes([0.92, 0.33, 0.018, 0.48])
            cb = fig.colorbar(im, cax=cax, label=met)
            cb.ax.tick_params(labelsize=10)

            ax_tbl = fig.add_subplot(gs[1, :])
            _build_heatmap_summary_table(ax_tbl, mats, models, years, fmt="{:.3f}")

            fig.suptitle(f"Seasonal heatmap of median basin-wise {met} across water years", y=0.965, fontsize=17)
            fig.subplots_adjust(top=0.88, bottom=0.05, left=0.06, right=0.90)
            save_fig(fig, outm / f"Heatmap_ALLMODELS__{met}.png", dpi=args.dpi)

            # 2) New comparison figure: 4 subplots in a 2x2 grid, one per season, all models compared inside each season
            fig = plt.figure(figsize=(12.8, max(9.0, 0.55 * len(years) + 4.6)), constrained_layout=False)
            gs = fig.add_gridspec(3, 2, height_ratios=[3.4, 3.4, 1.45], hspace=0.34, wspace=0.20)

            season_overall_rows = []
            for j, seas in enumerate(SEASONS):
                ax = fig.add_subplot(gs[j // 2, j % 2])
                season_mat = np.full((len(years), len(models)), np.nan, dtype=float)
                for i, y in enumerate(years):
                    for k, m in enumerate(models):
                        season_mat[i, k] = mats[m][i, j]

                sval = season_mat[np.isfinite(season_mat)]
                svmin, svmax = _season_limits(sval, pad_frac=0.08)
                cmap = SEASON_CMAPS.get(seas, "viridis")
                im2 = ax.imshow(season_mat, aspect="auto", vmin=svmin, vmax=svmax, cmap=cmap, interpolation="nearest")
                ax.grid(False)
                ax.tick_params(which="both", length=0)
                ax.set_title(seas, fontsize=15, pad=8)
                ax.set_xticks(np.arange(len(models)))
                ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models], fontsize=10, rotation=20)
                if j % 2 == 0:
                    ax.set_ylabel("Water year", fontsize=13)
                    ax.set_yticks(np.arange(len(years)))
                    ax.set_yticklabels([str(y) for y in years], fontsize=12)
                else:
                    ax.set_yticks([])

                midpoint = None if (svmin is None or svmax is None) else (svmin + svmax) / 2
                for i in range(len(years)):
                    for k in range(len(models)):
                        val = season_mat[i, k]
                        if np.isfinite(val):
                            txt_color = "white" if (midpoint is not None and val < midpoint) else "black"
                            ax.text(k, i, f"{val:.3f}", ha="center", va="center", fontsize=12.0, fontweight="bold", color=txt_color)

                cbar = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.02)
                cbar.ax.tick_params(labelsize=11)

                season_meds = []
                for k in range(len(models)):
                    col = season_mat[:, k]
                    col = col[np.isfinite(col)]
                    season_meds.append(float(np.median(col)) if col.size else np.nan)
                season_overall_rows.append([seas] + season_meds)

            ax_tbl2 = fig.add_subplot(gs[2, :])
            ax_tbl2.axis("off")
            cell_text = []
            for row in season_overall_rows:
                cell_text.append([row[0]] + [f"{v:.3f}" if np.isfinite(v) else "NA" for v in row[1:]])
            tbl2 = ax_tbl2.table(
                cellText=cell_text,
                colLabels=["Season"] + [MODEL_DISPLAY.get(m, m) for m in models],
                loc="center",
                cellLoc="center",
                bbox=[0.20, 0.00, 0.60, 0.78],
            )
            tbl2.auto_set_font_size(False)
            tbl2.set_fontsize(12.0)
            tbl2.scale(1.0, 1.18)

            fig.suptitle(f"Season-by-season heatmaps across water years (median over basins) — {met}", y=0.975, fontsize=18)
            fig.subplots_adjust(top=0.89, bottom=0.04, left=0.06, right=0.98)
            save_fig(fig, outm / f"Heatmap_bySeason_ALLMODELS__{met}.png", dpi=args.dpi)

        # 3) Single-model seasonal heatmap for Cluster-wise only
        if "Cluster-wise" in mats:
            m = "Cluster-wise"
            fig = plt.figure(figsize=(7.2, max(7.3, 0.38 * len(years) + 2.5)), constrained_layout=False)
            gs = fig.add_gridspec(2, 1, height_ratios=[4.0, 1.20], hspace=0.22)

            ax = fig.add_subplot(gs[0, 0])
            cluster_mat = mats[m]
            cv = cluster_mat[np.isfinite(cluster_mat)]
            cvmin, cvmax = _adaptive_limits(cv, pad_frac=0.08)
            im3 = ax.imshow(cluster_mat, aspect="auto", vmin=cvmin, vmax=cvmax, cmap="cividis", interpolation="nearest")
            ax.grid(False)
            ax.tick_params(which="both", length=0)
            ax.set_title("", pad=2)
            ax.set_xlabel("Season", fontsize=14, fontweight="bold", labelpad=12)
            ax.set_ylabel("Water year", fontsize=15, fontweight="bold")
            ax.set_xticks(np.arange(len(SEASONS)))
            ax.set_xticklabels(SEASONS, fontsize=13, fontweight="bold")
            ax.set_yticks(np.arange(len(years)))
            ax.set_yticklabels([str(y) for y in years], fontsize=12)

            midpoint = None if (cvmin is None or cvmax is None) else (cvmin + cvmax) / 2
            for i in range(len(years)):
                for k in range(len(SEASONS)):
                    val = cluster_mat[i, k]
                    if np.isfinite(val):
                        txt_color = "white" if (midpoint is not None and val < midpoint) else "black"
                        ax.text(k, i, f"{val:.3f}", ha="center", va="center", fontsize=12.0, fontweight="bold", color=txt_color)

            cbar = fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.03)
            cbar.ax.tick_params(labelsize=11)
            cbar.set_label(met, fontsize=13, fontweight="bold")

            ax_tbl3 = fig.add_subplot(gs[1, 0])
            ax_tbl3.axis("off")
            season_vals = []
            for j, seas in enumerate(SEASONS):
                col = cluster_mat[:, j]
                col = col[np.isfinite(col)]
                season_vals.append(float(np.median(col)) if col.size else np.nan)
            overall = np.asarray(season_vals, dtype=float)
            overall = overall[np.isfinite(overall)]
            overall = float(np.mean(overall)) if overall.size else np.nan
            tbl3 = ax_tbl3.table(
                cellText=[["Cluster-wise"] + [f"{v:.3f}" if np.isfinite(v) else "NA" for v in season_vals] + [f"{overall:.3f}" if np.isfinite(overall) else "NA"]],
                colLabels=["Model"] + SEASONS + ["Overall"],
                loc="center",
                cellLoc="center",
                bbox=[0.08, 0.02, 0.84, 0.78],
            )
            tbl3.auto_set_font_size(False)
            tbl3.set_fontsize(12.5)
            tbl3.scale(1.0, 1.26)
            for (r, c), cell in tbl3.get_celld().items():
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

            fig.suptitle(f"Seasonal heatmap of median basin-wise {met} across water years\nfor the cluster-wise family benchmark", y=0.968, fontsize=16.5, fontweight="bold")
            fig.subplots_adjust(top=0.85, bottom=0.04, left=0.08, right=0.92)
            save_fig(fig, outm / f"Heatmap_CLUSTERWISE_ONLY__{met}.png", dpi=args.dpi)

    print(f"[OK] Saved improved figures to: {outdir}")


if __name__ == "__main__":
    main()
