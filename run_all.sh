#!/usr/bin/env bash
# Reproduce todo el flujo, de los datos crudos a los envíos de Kaggle.
set -euo pipefail
cd "$(dirname "$0")/scripts"
python 01_download_osm.py     # requiere internet; se puede omitir si ya existe osm_pois.csv
python 02_build_features.py
python 03_naive_baselines.py
python 04_train_models.py
python 06_ensemble.py --number 30 --how median --models 17_boosting_full_coords_by_type_per_m2 04_elastic_net_full 18_random_forest_full_by_type_per_m2
python 06_ensemble.py --number 31 --how mean --models 17_boosting_full_coords_by_type_per_m2 04_elastic_net_full 18_random_forest_full_by_type_per_m2
python 07_results_table.py
