from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from camels_utils import (
    get_pickles_dir, get_results_dir, metric_dir, load_pickle,
    flatten_highflow_normalized_bins,
    build_common_argparser, parse_metrics_arg, parse_models_arg,
    BIN_ORDER, BIN_TO_X,
    compute_nse_acceptance, filter_by_acceptance, nse_accept_threshold,
    filter_metric_for_plot, plot_split_ecdf_metric,
    save_fig, agg_iqr
)

def main():
    parser = build_common_argparser("High-flow normalized events: global metrics + bin profiles + innovative 2D-PDF.")
    parser.add_argument("--bin-stat", type=str, default="ME",
                        help="Which bin stat to use for 2D-PDF plot (default ME). Options: ME, MAE, RMSE, MSE, Pearson-r")
    parser.add_argument("--bins-clip-low-q", type=float, default=0.01,
                        help="Quantile clip for bin-stat y-limits in density plots (display only).")
    parser.add_argument("--bins-clip-high-q", type=float, default=0.99,
                        help="Quantile clip for bin-stat y-limits in density plots (display only).")
    args = parser.parse_args()

    script_name = Path(__file__).stem
    outdir = get_results_dir(script_name)

    metrics = parse_metrics_arg(args.metrics, args.all_metrics)
    models = parse_models_arg(args.models)

    d = load_pickle(get_pickles_dir() / "metrics_highflow_events_normalized_bins.p")
    df_metrics, df_bins = flatten_highflow_normalized_bins(d)

    df_metrics = df_metrics[df_metrics["model"].isin(models)]
    df_bins = df_bins[df_bins["model"].isin(models)]

    # -----------------------------
    # NSE acceptance per (event_key, basin, model) on normalized events (threshold 0.3)
    # -----------------------------
    thr = nse_accept_threshold("normalized_events")
    acc = compute_nse_acceptance(df_metrics, group_cols=["event_key", "basin", "model"], threshold=thr)

    acc_out = outdir / "acceptance"
    acc_out.mkdir(parents=True, exist_ok=True)

    summary = (
        acc.groupby("model", as_index=False)
           .agg(n_total=("accepted", "size"), n_accepted=("accepted", "sum"))
    )
    summary["accept_rate"] = summary["n_accepted"] / summary["n_total"].replace(0, np.nan)
    summary["context"] = "highflow_normalized_bins"
    summary["nse_threshold"] = thr
    summary.to_csv(acc_out / "acceptance_summary_normalized_bins.csv", index=False)
    acc.rename(columns={"nse_median": "nse_median_over_seeds"}).to_csv(
        acc_out / "acceptance_by_basin_event_normalized.csv", index=False
    )

    # Filter metrics + bins to accepted basin-event per model
    dfM = filter_by_acceptance(df_metrics, acc, on_cols=["event_key", "basin", "model"])
    dfB = filter_by_acceptance(df_bins, acc, on_cols=["event_key", "basin", "model"])

    # ---------------- Global metrics: split ECDF across basin×event×seed (accepted only)
    for met in metrics:
        outm = metric_dir(outdir, met)
        dmet = dfM[dfM["metric"] == met].copy()
        dmet = filter_metric_for_plot(dmet, met, value_col="value")

        series = {m: dmet.loc[dmet["model"] == m, "value"].dropna().to_numpy() for m in models}
        plot_split_ecdf_metric(
            fig_title=f"Normalized events: ECDF across basin×event×seed (accepted) — {met}",
            series_by_model=series,
            metric=met,
            outpath=outm / f"ECDF_split_normEvents_basinEventSeed__{met}.png",
            dpi=args.dpi
        )

    # ---------------- Bin profiles (mean ± IQR across all accepted basin×event×seed samples)
    dfB["bin"] = pd.Categorical(dfB["bin"], categories=BIN_ORDER, ordered=True)
    dfB = dfB.dropna(subset=["bin", "value"])

    df_bin_iqr = agg_iqr(dfB, ["model", "bin", "stat"], "value")
    df_bin_iqr = df_bin_iqr.sort_values(["stat", "bin", "model"])

    xs = np.array([BIN_TO_X[b] for b in BIN_ORDER], dtype=float)
    out_bins = metric_dir(outdir, "BINS")

    for stat in sorted(df_bin_iqr["stat"].dropna().unique().tolist()):
        fig = plt.figure(figsize=(10.0, 5.4))
        ax = fig.add_subplot(111)
        for m in models:
            dd = df_bin_iqr[(df_bin_iqr["stat"] == stat) & (df_bin_iqr["model"] == m)].set_index("bin")
            y = np.array([dd.loc[b, "mean"] if b in dd.index else np.nan for b in BIN_ORDER], dtype=float)
            y25 = np.array([dd.loc[b, "q25"] if b in dd.index else np.nan for b in BIN_ORDER], dtype=float)
            y75 = np.array([dd.loc[b, "q75"] if b in dd.index else np.nan for b in BIN_ORDER], dtype=float)

            ax.plot(xs, y, label=m)
            ax.fill_between(xs, y25, y75, alpha=0.2)

        ax.set_title(f"Accepted normalized-event bin profile (mean±IQR) — {stat}")
        ax.set_xlabel("Relative bin position (L10..L01, P00, R01..R10)")
        ax.set_ylabel(stat)
        ax.grid(True, alpha=0.3)
        # legend outside
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=max(1, len(models)), frameon=False)
        fig.subplots_adjust(bottom=0.22)
        save_fig(fig, out_bins / f"BinProfile__{stat}.png", dpi=args.dpi)

        # --- Polar "circle map" view of the same bin profile (often clearer than a long x-axis)
        try:
            prof_stat = df_bin_iqr[df_bin_iqr["stat"] == stat].copy()
            theta = np.linspace(0, 2*np.pi, len(BIN_ORDER), endpoint=False)

            figp = plt.figure(figsize=(7.2, 7.2))
            axp = figp.add_subplot(111, projection="polar")
            axp.set_theta_zero_location("N")
            axp.set_theta_direction(-1)
            axp.set_xticks(theta)
            axp.set_xticklabels(BIN_ORDER, fontsize=9)

            for m in models:
                ddm = prof_stat[prof_stat["model"] == m].set_index("bin").reindex(BIN_ORDER)
                rr = ddm["mean"].to_numpy(dtype=float)

                # If gaps exist, interpolate for a continuous polar line (keeps plot non-empty)
                idx = np.arange(rr.size)
                finite = np.isfinite(rr)
                if finite.sum() == 0:
                    continue
                if finite.sum() < rr.size:
                    rr = np.interp(idx, idx[finite], rr[finite])

                th = np.r_[theta, theta[0]]
                rr2 = np.r_[rr, rr[0]]
                axp.plot(th, rr2, marker="o", linewidth=2, label=m)

            axp.set_title(f"Bin profile (polar; mean) — {stat}")
            axp.grid(True, alpha=0.3)
            axp.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=max(1, len(models)), frameon=False)
            figp.subplots_adjust(bottom=0.18)
            save_fig(figp, out_bins / f"BinProfilePolar__{stat}.png", dpi=args.dpi)
        except Exception:
            pass


    # ---------------- Innovative PDF plot: 2D density (bin position × stat value), all models in one PDF
    stat0 = args.bin_stat
    df_stat = dfB[dfB["stat"] == stat0].copy()
    df_stat = df_stat.dropna(subset=["bin", "value"])
    df_stat["x"] = df_stat["bin"].astype(str).map(BIN_TO_X)

    y_all = df_stat["value"].to_numpy()
    y_all = y_all[np.isfinite(y_all)]
    if y_all.size:
        ylo = float(np.quantile(y_all, args.bins_clip_low_q))
        yhi = float(np.quantile(y_all, args.bins_clip_high_q))
    else:
        ylo, yhi = -1, 1

    fig = plt.figure(figsize=(12.5, 8.5))
    gs = fig.add_gridspec(len(models), 1, hspace=0.18)

    mappable = None
    for i, m in enumerate(models):
        ax = fig.add_subplot(gs[i, 0])
        dd = df_stat[df_stat["model"] == m].dropna(subset=["x", "value"])
        x = dd["x"].to_numpy(dtype=float)
        y = dd["value"].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]

        h = ax.hist2d(
            x, y,
            bins=[len(BIN_ORDER), 70],
            range=[[xs.min(), xs.max()], [ylo, yhi]],
            density=True
        )
        mappable = h[3]
        ax.axvline(0.0, linestyle="--", linewidth=1)
        ax.set_xlim(xs.min(), xs.max())
        ax.set_ylim(ylo, yhi)
        ax.set_ylabel(m)
        ax.grid(True, alpha=0.15)
        if i < len(models) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Bin position (0 = peak-centered)")

    fig.suptitle(f"CAMELS-US normalized high-flow events — 2D PDF of {stat0}\n(accepted basin-events by NSE median ≥ {thr})")
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=fig.axes, fraction=0.02, pad=0.01)
        cbar.set_label("Probability density")

    out_pdf = out_bins / f"PDF2D_allModels_bin_x_{stat0}.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"[OK] Saved figures + PDF to: {outdir}")

if __name__ == "__main__":
    main()
