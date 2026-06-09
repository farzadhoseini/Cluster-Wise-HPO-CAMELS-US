# CAMELS-US clean HPO reproducibility data checklist

## Required for full reproduction from scratch

1. `configRegional_Fred.yml`: original NeuralHydrology base YAML config.
2. CAMELS-US dataset in NeuralHydrology-compatible format.
3. `531Basins.csv`: one CAMELS-US gauge ID per row.
4. Six cluster split files: `cluster_0.txt` to `cluster_5.txt`.
5. `metrics.py` placed beside the notebook.
6. Python environment with pandas, numpy, xarray, scipy, scikit-learn, pyyaml, PyTorch/CUDA, and NeuralHydrology.

## If skipping HPO training

Provide finished HPO run folders under `runs/hpo_random_search/`, each with:

- `config.yml`
- `validation/model_epoch020/validation_results.p`
- `validation/model_epoch025/validation_results.p`
- `validation/model_epoch030/validation_results.p`

At minimum, epoch 30 is enough if you only want the original simple extraction.

## If skipping final retraining

Provide finished final run folders under:

- `runs/final_top10/`
- `runs/final_clusterwise/`

Each run folder must contain:

- `config.yml`
- `test/model_epochXXX/test_results.p`

## If skipping prediction extraction

Provide these under `outputs/pickles/`:

- `All_Top10_Configs_CAMELSUS_test.p`
- `All_Cluster_wise_Configs_CAMELSUS_test.p`
- `Obs_test.p`

## If skipping ensemble construction

Provide these under `outputs/pickles/`:

- `ensemble_Top10_CAMELSUS_test.p`
- `ensemble_Cluster_wise_CAMELSUS_test.p`
- `Obs_test.p`

## Main outputs

- `outputs/evaluations/generated_random_search_configs.csv`
- `outputs/evaluations/Ver_results_2000RS_validation_epochs_20_25_30.csv`
- `outputs/evaluations/selected_top10_models.csv`
- `outputs/evaluations/selected_clusterwise_models.csv`
- `outputs/pickles/All_Top10_Configs_CAMELSUS_test.p`
- `outputs/pickles/All_Cluster_wise_Configs_CAMELSUS_test.p`
- `outputs/pickles/Obs_test.p`
- `outputs/pickles/ensemble_Top10_CAMELSUS_test.p`
- `outputs/pickles/ensemble_Cluster_wise_CAMELSUS_test.p`
- `outputs/evaluations/test_metrics_ensembles_long.csv`
- `outputs/evaluations/test_metrics_ensembles_long_summary.csv`
