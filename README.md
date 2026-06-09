\# Cluster-Wise Hyperparameter Optimization Enhances Regional Rainfall–Runoff Modeling with Deep Learning: A Robust CAMELS-US Benchmark



This repository contains the cleaned code used for the CAMELS-US cluster-wise hyperparameter optimization benchmark.



\## Study overview



The workflow supports:



1\. preparation of CAMELS-US basin/cluster metadata,

2\. random-search hyperparameter optimization of regional LSTM models,

3\. selection of best-performing models,

4\. construction of model ensembles,

5\. generation of final prediction and metric files used for paper figures and analysis.



\## Repository structure



```text

final\_codes/

├── codes/                         # Main modelling, HPO, ensemble and evaluation scripts/notebooks

├── misc/                          # Supporting utilities and metadata

├── 531\_basin\_list.txt             # List of CAMELS-US basins used in the benchmark

├── bad\_basins\_clusterwise\_nse\_lt\_0p5.csv

├── camels\_531\_master\_attributes\_clusters.csv

├── CAMELS\_US\_Clusters.xlsx

├── README.md

└── .gitignore

Data requirements

The repository does not include the full CAMELS-US dataset.
Users should download CAMELS-US from the official data source and place the required forcing, streamflow, and attribute files following the paths described in the notebooks/scripts.

Required input information includes:

CAMELS-US basin IDs,
meteorological forcing data,
observed streamflow data,
CAMELS catchment attributes,
cluster assignment file,
train/validation/test split configuration.
Workflow

The cleaned workflow starts from the HPO stage and continues to final prediction/metric pickle generation.

A typical run consists of:

run random-search HPO,
collect and rank trained configurations,
select best models per cluster/family,
build ensembles,
evaluate on the test period,
export final prediction and metric files.

Paper figures are generated separately using the final figure package.

Citation

If you use this repository, please cite the associated paper:

Cluster-Wise Hyperparameter Optimization Enhances Regional Rainfall–Runoff Modeling with Deep Learning: A Robust CAMELS-US Benchmark

Author

Farzad Hosseini Hossein Abadi
GitHub: farzadhoseini