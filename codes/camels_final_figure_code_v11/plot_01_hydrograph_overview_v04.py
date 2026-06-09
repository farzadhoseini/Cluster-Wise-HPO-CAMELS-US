from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from camels_publication_style import get_model_color, get_model_linestyle

from camels_utils import (
    get_pickles_dir, get_results_dir, metric_dir, load_pickle,
    flatten_metrics_all_test_period,
    build_common_argparser, parse_metrics_arg, parse_models_arg,
    metric_rule, save_fig, ecdf
)

# ---------------------------
# Styling helpers
# ---------------------------
MODEL_ORDER = ["Benchmark", "Cluster-wise", "Top10"]
MODEL_DISPLAY = {
    "Benchmark": "Benchmark",
    "Cluster-wise": "Cluster-wise",
    "Top10": "Top10",
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
    tbl.set_fontsize(12.5)
    tbl.scale(1.0, 1.28)

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


def boxplot_with_stats_panel_pro(
    data,
    labels,
    title,
    ylabel,
    figsize=(9.2, 7.2),
    showfliers=False,
    fmt="{:.3f}",
):
    # plot first, table below (not on top of plot)
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[4.0, 1.32],
        hspace=0.16,
    )

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
    ax.set_title(title, fontsize=20, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis="x", labelsize=16, pad=16)
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(True, axis="y", alpha=0.3)

    _add_summary_table(ax_tbl, data, labels, fmt=fmt)

    fig.subplots_adjust(top=0.92, bottom=0.06, left=0.10, right=0.98)
    return fig, ax


def _parse_metric_rule(rule_obj, metric_name):
    threshold = None
    higher_is_better = True

    if isinstance(rule_obj, dict):
        for key in ["threshold", "thr", "split", "cutoff", "good_threshold"]:
            if key in rule_obj:
                threshold = float(rule_obj[key])
                break
        for key in ["higher_is_better", "maximize", "greater_is_better"]:
            if key in rule_obj:
                higher_is_better = bool(rule_obj[key])
                break
    else:
        name = metric_name.lower()
        if name in {"nse", "kge", "pearson-r", "pearson_r", "alpha-nse", "beta-kge"}:
            threshold = 0.5
            higher_is_better = True
        elif name in {"rmse", "mse", "mape", "peak-mape", "peak-timing", "missed-peaks"}:
            threshold = None
            higher_is_better = False

    return threshold, higher_is_better


def plot_split_ecdf_metric_pro(fig_title, series_by_model, metric, outpath, dpi=200):
    models = _ordered_models(list(series_by_model.keys()))
    threshold, higher_is_better = _parse_metric_rule(metric_rule(metric), metric)

    if threshold is None:
        fig, ax = plt.subplots(figsize=(9.8, 6.0))
        for m in models:
            x = np.asarray(series_by_model[m], dtype=float)
            x = x[np.isfinite(x)]
            xs, ys = ecdf(x)
            ax.plot(xs, ys, label=MODEL_DISPLAY.get(m, m), linewidth=2.3, color=get_model_color(m), linestyle=get_model_linestyle(m))

        ax.set_title(fig_title, fontsize=18, pad=12)
        ax.set_xlabel(metric, fontsize=15)
        ax.set_ylabel("ECDF", fontsize=15)
        ax.tick_params(labelsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=len(models),
            frameon=True,
            fontsize=12,
            title="Model",
            title_fontsize=12,
        )
        fig.subplots_adjust(top=0.90, bottom=0.24, left=0.10, right=0.98)
        save_fig(fig, outpath, dpi=dpi)
        return

    fig = plt.figure(figsize=(13.6, 6.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.16)
    ax_bad = fig.add_subplot(gs[0, 0])
    ax_good = fig.add_subplot(gs[0, 1], sharey=ax_bad)
    summary_parts = []

    if higher_is_better:
        bad_label = f"Low performance: < {threshold:g}"
        good_label = f"Accepted performance: ≥ {threshold:g}"
    else:
        bad_label = f"Low performance: > {threshold:g}"
        good_label = f"Accepted performance: ≤ {threshold:g}"

    for m in models:
        x = np.asarray(series_by_model[m], dtype=float)
        x = x[np.isfinite(x)]

        if higher_is_better:
            bad = x[x < threshold]
            good = x[x >= threshold]
        else:
            bad = x[x > threshold]
            good = x[x <= threshold]

        if len(bad):
            xs, ys = ecdf(bad)
            ax_bad.plot(xs, ys, label=MODEL_DISPLAY.get(m, m), linewidth=2.3, color=get_model_color(m), linestyle=get_model_linestyle(m))
        if len(good):
            xs, ys = ecdf(good)
            ax_good.plot(xs, ys, label=MODEL_DISPLAY.get(m, m), linewidth=2.3, color=get_model_color(m), linestyle=get_model_linestyle(m))

        summary_parts.append(
            f"{MODEL_DISPLAY.get(m,m)}: low={len(bad)}, accepted={len(good)} (N={len(x)})"
        )

    fig.suptitle(fig_title, fontsize=21, fontweight="bold", y=0.985)
    fig.text(0.5, 0.895, " | ".join(summary_parts), ha="center", va="center", fontsize=11.5)

    ax_bad.set_title(bad_label, fontsize=18, fontweight="bold", pad=10)
    ax_good.set_title(good_label, fontsize=18, fontweight="bold", pad=10)

    for ax in [ax_bad, ax_good]:
        ax.set_xlabel(metric, fontsize=15)
        ax.tick_params(labelsize=13)
        ax.grid(True, alpha=0.3)
    ax_bad.set_ylabel("ECDF", fontsize=15)
    ax_good.set_ylabel("")
    plt.setp(ax_good.get_yticklabels(), visible=False)

    handles, labels = ax_bad.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=len(models),
        frameon=True,
        fontsize=12,
        title="Model",
        title_fontsize=12,
    )
    fig.subplots_adjust(top=0.81, bottom=0.18, left=0.07, right=0.98)
    save_fig(fig, outpath, dpi=dpi)


def plot_seed_cdf_pro(d2, models, met, outpath, dpi=200):
    fig, ax = plt.subplots(figsize=(9.8, 6.0))
    models = _ordered_models(models)
    for m in models:
        x = d2.loc[d2["model"] == m, "seed_mean"].to_numpy()
        x = x[np.isfinite(x)]
        xs, ys = ecdf(x)
        ax.plot(xs, ys, label=MODEL_DISPLAY.get(m, m), linewidth=2.3, color=get_model_color(m), linestyle=get_model_linestyle(m))

    ax.set_title(f"CDF of seed-level {met} values after averaging over the 531 CAMELS-US basins", fontsize=18, pad=12)
    ax.set_xlabel(f"{met} (seed mean)", fontsize=15)
    ax.set_ylabel("CDF", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.03, 0.98),
        ncol=1,
        frameon=True,
        fontsize=11.5,
        title="Model",
        title_fontsize=12,
    )
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.10, right=0.98)
    save_fig(fig, outpath, dpi=dpi)


def main():
    parser = build_common_argparser("CAMELS-US: Whole-hydrograph metrics overview.")
    args = parser.parse_args()

    script_name = Path(__file__).stem
    outdir = get_results_dir(script_name)

    metrics = parse_metrics_arg(args.metrics, args.all_metrics)
    models = parse_models_arg(args.models)
    models = _ordered_models(models)

    d = load_pickle(get_pickles_dir() / "metrics_all_test_period.p")
    df = flatten_metrics_all_test_period(d)
    df = df[df["model"].isin(models)]

    # basin mean over seeds
    df_basin_mean = (
        df.groupby(["basin", "model", "metric"], as_index=False)["value"]
          .mean()
          .rename(columns={"value": "basin_mean"})
    )

    # seed mean over basins
    df_seed_mean = (
        df.groupby(["seed", "model", "metric"], as_index=False)["value"]
          .mean()
          .rename(columns={"value": "seed_mean"})
    )

    for met in metrics:
        outm = metric_dir(outdir, met)

        d1 = df_basin_mean[df_basin_mean["metric"] == met]
        d2 = df_seed_mean[df_seed_mean["metric"] == met]

        # Split ECDF across basins
        series = {m: d1.loc[d1["model"] == m, "basin_mean"].to_numpy() for m in models}
        plot_split_ecdf_metric_pro(
            fig_title=f"ECDF of basin-wise test-period {met} across the 531 CAMELS-US basins",
            series_by_model=series,
            metric=met,
            outpath=outm / f"ECDF_split_basins__{met}.png",
            dpi=args.dpi,
        )

        # Box across basins + table below
        data = [d1.loc[d1["model"] == m, "basin_mean"].dropna().to_numpy() for m in models]
        fig, ax = boxplot_with_stats_panel_pro(
            data,
            models,
            title=f"Distribution of basin-wise test-period {met} across the 531 CAMELS-US basins",
            ylabel=met,
            figsize=(9.4, 7.2),
            showfliers=False,
            fmt="{:.3f}",
        )
        save_fig(fig, outm / f"BOX_basins__{met}.png", dpi=args.dpi)

        # Seed robustness: box + table below
        data = [d2.loc[d2["model"] == m, "seed_mean"].dropna().to_numpy() for m in models]
        fig, ax = boxplot_with_stats_panel_pro(
            data,
            models,
            title=f"Distribution of seed-level {met} values after averaging over the 531 CAMELS-US basins",
            ylabel=f"{met} (seed mean)",
            figsize=(9.4, 7.2),
            showfliers=False,
            fmt="{:.3f}",
        )
        save_fig(fig, outm / f"BOX_seed_robustness__{met}.png", dpi=args.dpi)

        # Seed robustness CDF
        plot_seed_cdf_pro(
            d2=d2,
            models=models,
            met=met,
            outpath=outm / f"CDF_seed_robustness__{met}.png",
            dpi=args.dpi,
        )

    print(f"[OK] Saved improved figures to: {outdir}")


if __name__ == "__main__":
    main()
