
# CAMELS-US final figure code package — v2

This version was rebuilt after reviewing the full `All_old_codes.zip` archive.
It keeps the Javi Del Ser visual style and writes every final figure to one folder:

```text
<project-root>/results/All_plots/
```

For each final figure, the runner creates both:

```text
*.png   high-resolution raster figure
*.pdf   PDF version for manuscript/supplement use
```

## Run

From this folder:

```bash
python run_all_camels_figures_final.py ^
  --project-root "F:/Experiments/CAMELS_US/Clean_4_upload" ^
  --metrics NSE ^
  --models "Benchmark,Cluster-wise,Top10" ^
  --dpi 300 ^
  --clean-outdir
```

Linux/macOS style:

```bash
python run_all_camels_figures_final.py \
  --project-root "/path/to/CAMELS_US/Clean_4_upload" \
  --metrics NSE \
  --models "Benchmark,Cluster-wise,Top10" \
  --dpi 300 \
  --clean-outdir
```

## Style

The global plotting style is in:

```text
camels_publication_style.py
```

It uses the Javi Del Ser palette:

```text
#56ae6c, #8960b3, #b0923b, #ba495b
```

and applies larger serif fonts, thicker lines, larger markers, dark marker edges,
visible axis spines, and grayscale/black-and-white-safe line styles/hatches.

## Included legacy code

All old `.py` code files from `All_old_codes.zip` are included so the final runner
can call the original figure-generation logic. Some exploratory or non-final scripts
are kept as source/reference but are not run by default because they are not part of
the final fig01--fig30 mapping.

## Important note about fig01

The runner first generates the hydrology-summary map and then tries to run
`plot_camels_clusters_pro_map.py` to make a more professional Cartopy map with
U.S. national and state/province boundaries. If Cartopy is not installed, it falls
back automatically to the hydrology-summary map.

Install optional map dependency if needed:

```bash
pip install cartopy pyproj
```

## Expected project inputs

The runner assumes the same project structure used by the old scripts, including:

```text
531_basin_list.txt
CAMELS_US_Clusters.xlsx
Train_Data/camels_attributes_v2.0/
Pickles/metrics_all_test_period.p
Pickles/metrics_by_water_year.p
Pickles/metrics_by_year_season.p
```

The deep-diagnostic scripts also require their usual processed diagnostic inputs
already available in the project tree.
