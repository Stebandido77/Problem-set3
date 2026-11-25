#!/usr/bin/env bash
# Ruta principal: de los datos crudos a los envíos finales y al material de las dos
# presentaciones. Reproduce los modelos ingenuos, el mejor modelo de cada familia, la mezcla
# final (203) y todas las figuras y tablas de document/. Los demás experimentos (rondas 1
# a 10) están en EXPERIMENTOS.md, con sus comandos y su tabla de resultados.
#
# Uso:  bash run_all.sh                    # usa las descargas externas ya versionadas
#       bash run_all.sh --download         # vuelve a descargar OSM y los datos de Bogotá
# El tiempo de una corrida completa está en el README ("Cómo reproducir").
set -euo pipefail
cd "$(dirname "$0")/scripts"
PYTHON="${PYTHON:-python}"

# 1. Fuentes externas. Sin --download se arman con lo que ya está en stores/processed.
if [[ "${1:-}" == "--download" ]]; then
    "$PYTHON" 01_download_osm.py       # OpenStreetMap (Overpass); requiere internet
    "$PYTHON" 11_download_bogota.py    # estrato y valor del terreno por manzana
    "$PYTHON" 16_download_lots.py      # lote, pisos y área construida del catastro
else
    "$PYTHON" 01_download_osm.py --cache-only
fi

# 2. Predictores, modelos ingenuos y embeddings del texto (los usa el bosque 114).
"$PYTHON" 02_build_features.py
"$PYTHON" 03_naive_baselines.py
"$PYTHON" 13_text_embeddings.py

# 3. Mejor especificación de cada familia: lineal 108, elastic net 104, CART 105,
#    bosque 114, boosting 83 y 125, red 148. La 127 es la 125 con la regla de anuncios
#    repetidos y la 152 mezcla la 127 (70 %) con la red 148 (30 %).
"$PYTHON" 04_train_models.py --specs 104 105 108 114 83 125 148 \
    --schemes random random_twins spatial northeast
"$PYTHON" 20_twin_rules.py --rules mix --numbers 127 \
    --model 125_boosting_pt_core_coords_by_type_level_q70_ridge_nocoords_s10
"$PYTHON" 06_ensemble.py --number 152 --how weighted --weights 0.7 0.3 --models \
    127_boosting_pt_core_coords_by_type_level_q70_ridge_nocoords_s10_twin_mix \
    148_quantile_net_ext_core_coords_by_type_q70_ridge_nocoords

# 3b. Comparaciones que usan las presentaciones: el kNN 185 y la 186, que mezcla la 181
#     (la 177 con pinball propia, más la regla de gemelos) con la red 148 y gana en la
#     espacial solo por bajar el nivel.
"$PYTHON" 04_train_models.py --specs 177 185 --schemes spatial northeast
"$PYTHON" 20_twin_rules.py --rules mix --numbers 181 \
    --model 177_boosting_pt_core_coords_by_type_pinball_ridge_nocoords_s10
"$PYTHON" 06_ensemble.py --number 186 --how weighted --weights 0.7 0.3 --models \
    181_boosting_pt_core_coords_by_type_pinball_ridge_nocoords_s10_twin_mix \
    148_quantile_net_ext_core_coords_by_type_q70_ridge_nocoords

# 3c. Mejor modelo en Kaggle: la 203 mezcla el boosting 198 (la 194 con la regla de
#     gemelos) y la red 196, los dos con el precio que predice BETO junto al ridge. BETO
#     corre en Kaggle con GPU (beto_kaggle.py); su salida está versionada en
#     stores/processed/beto_predictions.csv y 38 la verifica.
"$PYTHON" 38_beto_check.py
"$PYTHON" 04_train_models.py --specs 194 196 --schemes spatial northeast
"$PYTHON" 20_twin_rules.py --rules mix --numbers 198 \
    --model 194_boosting_pt_core_coords_by_type_level_q70_ridge_beto_nocoords_s10
"$PYTHON" 06_ensemble.py --number 203 --how weighted --weights 0.7 0.3 --models \
    198_boosting_pt_core_coords_by_type_level_q70_ridge_beto_nocoords_s10_twin_mix \
    196_quantile_net_ext_core_coords_by_type_q70_ridge_beto_nocoords

# 4. La receta de la 125 con cuantiles 0,20, 0,30 y 0,50 para la regla de compra.
"$PYTHON" 04_train_models.py --specs 190 191 192 --schemes spatial --no-submission
"$PYTHON" 07_results_table.py

# 5. Figuras y tablas de las presentaciones (views/) y las presentaciones (document/).
"$PYTHON" 31_data_overview.py
"$PYTHON" 33_validation_schemes.py
"$PYTHON" 32_algorithm_comparison.py
"$PYTHON" 34_best_model_errors.py
"$PYTHON" 35_purchase_rule.py
"$PYTHON" 41_beto_figure.py
"$PYTHON" 42_deck_checks.py
"$PYTHON" 36_build_presentations.py
