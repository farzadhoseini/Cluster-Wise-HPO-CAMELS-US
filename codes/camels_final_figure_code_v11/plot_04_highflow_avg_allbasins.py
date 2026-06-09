# plot_04_highflow_avg_allbasins.py
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

from camels_utils import (
    get_pickles_dir, get_results_dir, metric_dir, load_pickle,
    flatten_highflow_allbasins,
    build_common_argparser, parse_metrics_arg, parse_models_arg,
    plot_split_ecdf_metric, metric_rule, save_fig, boxplot_with_stats_panel, ecdf
)

def main():
    parser = build_common_argparser("High-flow events: average metrics per basin.")
    args = parser.parse_args()

    script_name = Path(__file__).stem
    outdir = get_results_dir(script_name)
    metrics = parse_metrics_arg(args.metrics, args.all_metrics)
    models = parse_models_arg(args.models)

    d = load_pickle(get_pickles_dir() / "metrics_highflow_events_allbasins.p")
    df = flatten_highflow_allbasins(d)
    df = df[df["model"].isin(models)]

    df_basin_med = (
        df.groupby(["basin", "model", "metric"], as_index=False)["value"]
          .mean()
          .rename(columns={"value": "basin_med"})
    )

    df_seed_med = (
        df.groupby(["seed", "model", "metric"], as_index=False)["value"]
          .mean()
          .rename(columns={"value": "seed_med"})
    )

    for met in metrics:
        outm = metric_dir(outdir, met)
        d1 = df_basin_med[df_basin_med["metric"] == met]
        d2 = df_seed_med[df_seed_med["metric"] == met]

        # split ECDF across basins
        series = {m: d1.loc[d1["model"] == m, "basin_med"].to_numpy() for m in models}
        plot_split_ecdf_metric(
            fig_title=f"High-flow avg: ECDF across basins (mean over seeds) — {met}",
            series_by_model=series,
            metric=met,
            rule=metric_rule(met),
            
            
            
            xlabel=met,
            outpath=outm / f"ECDF_split_highflowAvg__{met}.png",
            dpi=args.dpi
        )

        # box across basins (clean: separate stats panel)
        data = [d1.loc[d1["model"] == m, "basin_med"].dropna().to_numpy() for m in models]
        fig, _ax = boxplot_with_stats_panel(
            data_list=data,
            labels=models,
            title=f"High-flow avg: Box across basins (mean over seeds) — {met}",
            ylabel=met,
            figsize=(8.2, 5.8),
            showfliers=False,
            fmt="{:.3f}",
        )
        save_fig(fig, outm / f"BOX_highflowAvg_basins__{met}.png", dpi=args.dpi)

        # seed robustness box + CDF (clean: separate stats panel)
        data = [d2.loc[d2["model"] == m, "seed_med"].dropna().to_numpy() for m in models]
        fig, _ax = boxplot_with_stats_panel(
            data_list=data,
            labels=models,
            title=f"High-flow avg: Seed robustness (median over basins) — {met}",
            ylabel=f"{met} (seed median)",
            figsize=(8.2, 5.8),
            showfliers=False,
            fmt="{:.4f}",
        )
        save_fig(fig, outm / f"BOX_seed_robustness__{met}.png", dpi=args.dpi)

        fig = plt.figure(figsize=(7.5, 5))
        ax = fig.add_subplot(111)
        for m in models:
            x = d2.loc[d2["model"] == m, "seed_med"].to_numpy()
            xs, ys = ecdf(x)
            ax.plot(xs, ys, label=m)
        ax.set_title(f"High-flow avg: Seed robustness CDF — {met}")
        ax.set_xlabel(f"{met} (seed median)")
        ax.set_ylabel("CDF")
        ax.grid(True, alpha=0.3)
        ax.legend()
        save_fig(fig, outm / f"CDF_seed_robustness__{met}.png", dpi=args.dpi)

    print(f"[OK] Saved figures to: {outdir}")

if __name__ == "__main__":
    main()
