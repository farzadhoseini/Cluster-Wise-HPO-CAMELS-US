# plot_05_highflow_eventwise.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from camels_utils import (
    get_pickles_dir, get_results_dir, metric_dir, load_pickle,
    flatten_highflow_eventwise,
    build_common_argparser, parse_metrics_arg, parse_models_arg,
    plot_split_ecdf_metric, metric_rule, save_fig, parse_event_start_date, water_year_from_date
)

def main():
    parser = build_common_argparser("High-flow events (eventwise): split-ECDF + heatmap + win-rate.")
    parser.add_argument("--reference", type=str, default="Benchmark",
                        help="Reference model for win-rate comparisons (default Benchmark).")
    args = parser.parse_args()

    script_name = Path(__file__).stem
    outdir = get_results_dir(script_name)
    metrics = parse_metrics_arg(args.metrics, args.all_metrics)
    models = parse_models_arg(args.models)
    ref = args.reference

    d = load_pickle(get_pickles_dir() / "metrics_highflow_events_eventwise.p")
    df = flatten_highflow_eventwise(d)
    df = df[df["model"].isin(models)]

    # event mean over seeds for robustness
    df_event_med = (
        df.groupby(["basin", "event_key", "model", "metric"], as_index=False)["value"]
          .mean()
          .rename(columns={"value": "event_med"})
    )

    # derive event water year
    event_dates = df_event_med[["event_key"]].drop_duplicates().copy()
    event_dates["start_dt"] = event_dates["event_key"].apply(parse_event_start_date)
    event_dates["event_wy"] = event_dates["start_dt"].apply(lambda x: water_year_from_date(x) if x is not None else np.nan)

    df_event_med = df_event_med.merge(event_dates[["event_key", "event_wy"]], on="event_key", how="left")

    for met in metrics:
        outm = metric_dir(outdir, met)
        de = df_event_med[df_event_med["metric"] == met]

        # ---- split ECDF across events (pooled basins)
        series = {m: de.loc[de["model"] == m, "event_med"].to_numpy() for m in models}
        plot_split_ecdf_metric(
            fig_title=f"Eventwise high-flow: ECDF across events (mean over seeds) — {met}",
            series_by_model=series,
            metric=met,
            rule=metric_rule(met),
            
            
            
            xlabel=met,
            outpath=outm / f"ECDF_split_events__{met}.png",
            dpi=args.dpi
        )

        # ---- heatmap: event_wy x model, value = mean across events in that WY
        # (still robust because values already mean over seeds)
        dd = de.dropna(subset=["event_wy"]).copy()
        years = sorted(dd["event_wy"].dropna().unique().astype(int).tolist())

        mat = np.full((len(years), len(models)), np.nan, dtype=float)
        for i, y in enumerate(years):
            for j, m in enumerate(models):
                vals = dd[(dd["event_wy"] == y) & (dd["model"] == m)]["event_med"].to_numpy()
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    mat[i, j] = float(np.median(vals))

        fig = plt.figure(figsize=(8, max(4, 0.25 * len(years))))
        ax = fig.add_subplot(111)
        im = ax.imshow(mat, aspect="auto")
        ax.set_title(f"Heatmap: event water-year × model (mean across events) — {met}")
        ax.set_xlabel("Model")
        ax.set_ylabel("Event water year")
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models)
        ax.set_yticks(np.arange(len(years)))
        ax.set_yticklabels(years)
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        save_fig(fig, outm / f"Heatmap_eventWY_x_model__{met}.png", dpi=args.dpi)

        # ---- win-rate vs reference across (basin,event)
        lower_better = met in ["MSE", "RMSE", "MAPE", "Peak-MAPE", "Missed-Peaks", "Peak-Timing"]
        pivot = de.pivot_table(index=["basin", "event_key"], columns="model", values="event_med", aggfunc="first")
        if ref in pivot.columns:
            win_rates = {}
            for m in models:
                if m == ref or m not in pivot.columns:
                    continue
                a = pivot[m]
                b = pivot[ref]
                valid = a.notna() & b.notna()
                if valid.sum() == 0:
                    continue
                win = (a[valid] < b[valid]).mean() if lower_better else (a[valid] > b[valid]).mean()
                win_rates[m] = float(win)

            if win_rates:
                fig = plt.figure(figsize=(7.5, 5))
                ax = fig.add_subplot(111)
                labs = list(win_rates.keys())
                vals = [win_rates[k] for k in labs]
                ax.bar(labs, vals)
                ax.set_ylim(0, 1)
                ax.set_title(f"Win-rate vs {ref} across events — {met}")
                ax.set_ylabel("Fraction of (basin,event) wins")
                ax.grid(True, axis="y", alpha=0.3)
                save_fig(fig, outm / f"WinRate_vs_{ref}__{met}.png", dpi=args.dpi)

    print(f"[OK] Saved figures to: {outdir}")

if __name__ == "__main__":
    main()
