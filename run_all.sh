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
# Ronda 2: ensambles y versiones escaladas (calibración de nivel, fuera de la tabla principal).
python 06_ensemble.py --number 55 --how mean --models 54_en_boosting_core_coords_residual_q65 48_boosting_core_coords_by_type_level_trunc
python 06_ensemble.py --number 56 --how mean --models 54_en_boosting_core_coords_residual_q65 48_boosting_core_coords_by_type_level_trunc 04_elastic_net_full
python 09_level_calibration.py --number 57 --factors 1.10 1.15 1.20 --submission 54_en_boosting_core_coords_residual_q65
python 09_level_calibration.py --number 60 --factors 1.20 --model none --submission 10_boosting_full_coords
python 07_results_table.py
# Opcional y lento (~5 min por esquema): python 08_calibrate_holdout.py
