# CAMELS-US final figure pipeline — strict v3

This version was revised after comparing the final Excel mapping with the reference old plot set.

## Main rule

The final delivery folder is limited to the exact mapped/reference figures only:

- 30 final figure names: `fig01` to `fig30`
- one high-resolution `.png` and one `.pdf` for each figure
- output folder: `<project-root>/results/All_plots/`

No exploratory seed/year/season figures are kept in `results/All_plots`.

## Important correction from v2

The old water-year script originally generated seed-robustness figures for every water year.
In v3, final mode generates only the representative seed water years used in the final mapping/reference set:

- WY1992
- WY1995
- WY1998

The old seasonal script originally generated pooled seasonal boxplots and all-model seasonal heatmaps.
In v3, the runner calls it with `--final-only`, so it generates only the mapped Cluster-wise seasonal heatmap.

## Run

```bash
python run_all_camels_figures_final.py ^
  --project-root "F:/Experiments/CAMELS_US/Clean_4_upload" ^
  --metrics NSE ^
  --models "Benchmark,Cluster-wise,Top10" ^
  --dpi 300 ^
  --clean-outdir
```

Optional reproduction of old exploratory behavior:

```bash
python plot_02_water_year_v03.py --metrics NSE --models "Benchmark,Cluster-wise,Top10" --seed-years all
```

## Style

The Javi Del Ser-style color palette is retained and strengthened for readability:

- green: `#56ae6c`
- purple: `#8960b3`
- ochre: `#b0923b`
- red: `#ba495b`

The style module also increases line widths, marker visibility, font sizes, table readability, and black/white print robustness.

## Final reference list

See `FINAL_REFERENCE_FIGURES.txt`.
