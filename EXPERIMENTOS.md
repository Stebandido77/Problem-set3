# Experimentos (rondas 1 a 10)

Registro de todo lo que se probó antes de llegar al modelo final. La ruta principal
(`run_all.sh` / `run_all.ps1`) solo reproduce los ingenuos, el mejor modelo de cada familia,
la mezcla 152 y el material de las presentaciones; lo que sigue se reproduce con los
comandos de abajo, desde `scripts/`. La tabla completa de resultados está en
`views/cv_results.csv` (en Markdown, `views/cv_results_table.md`, de `07_results_table.py`)
y los puntajes públicos en `views/kaggle_scores.csv`.

Las cifras de este archivo se escribieron al cerrar cada ronda y salen de esos archivos.
Algunas frases describen el estado del proyecto en esa ronda (por ejemplo, "el mejor modelo
nuevo es el 54"); el resultado final está en el README.

## Comandos de los experimentos

```bash
python 01_download_osm.py                 # una vez; requiere internet (la caché ya está en el repo)
python 01_download_osm.py --cache-only    # arma osm_pois.csv solo con lo ya descargado
python 02_build_features.py
python 03_naive_baselines.py
python 04_train_models.py                 # todas las especificaciones (~41 min en 16 núcleos)
python 04_train_models.py --specs 17 18   # solo algunos números de SPECS
python 05_tune_models.py --family cart --features full --n-iter 15 --n-jobs 4
python 06_ensemble.py --number 31 --how mean --models \
    17_boosting_full_coords_by_type_per_m2 04_elastic_net_full 18_random_forest_full_by_type_per_m2
# Ronda 4, en este orden:
python 16_download_lots.py                # lote, pisos y área construida; antes de 02 (internet)
python 13_text_embeddings.py              # embeddings del texto; antes de las filas 86-98
python 14_tune_text_ridge.py              # ridge de texto sintonizado; antes de la fila 89
python 15_optuna_search.py --spec 94 --sampler tpe cmaes --n-trials 60   # antes de la 98
python 15_optuna_search.py --spec 94 --confirm optuna_tpe
# Ronda 5 (la receta de la 83 en las demás familias), en este orden:
python 04_train_models.py --specs 99 100 101 102 --schemes northeast spatial chapinero
python 17_tune_families.py --spec 99 --n-trials 25    # bosque; ~25 min, se puede cortar y seguir
python 17_tune_families.py --spec 102 --n-trials 25   # CART
python 17_tune_families.py --spec 100 --n-trials 15   # elastic net
python 04_train_models.py --specs 103 104 105 --schemes northeast spatial chapinero
python 18_family_table.py
# Ronda 6 (el bosque contra el boosting), en este orden:
python 04_train_models.py --specs 106 107 108 109 110 --schemes northeast spatial chapinero random_twins
python 19_genetic_search.py --spec 107 --sampler nsga2 --n-trials 120   # ~60 min, reanudable
python 19_genetic_search.py --spec 107 --confirm
python 04_train_models.py --specs 112 113 114 115 --schemes northeast spatial chapinero
python 06_ensemble.py --number 116 --how weighted --weights 0.8 0.2 --models     83_boosting_ext_core_coords_by_type_level_q70_ridge_s10     114_random_forest_ext_core_coords_by_type_level_q70_ridge_net_nocoords   # y 117-119
# Ronda 8 (revisión de supuestos), en este orden:
python 23_error_anatomy.py
python 04_train_models.py --specs 154 155 156 157 158 159 160 161 162 163 164 165 --schemes northeast spatial --no-submission
python 24_assumptions.py --nocap          # reentrena la 82 sin el tope de 1.000 M (~1 min)
python 25_criteria_vs_kaggle.py
# Ronda 9 (redes y mezclas), en este orden:
python 04_train_models.py --specs 168 169 170 --schemes northeast spatial
python 17_tune_families.py --spec 169 --n-trials 20 --objective spatial   # ~17 min
python 04_train_models.py --specs 171 --schemes northeast spatial
python 06_ensemble.py --number 172 --how weighted --weights 0.7 0.3 --models     127_boosting_pt_core_coords_by_type_level_q70_ridge_nocoords_s10_twin_mix     171_quantile_net_pt_core_coords_by_type_q70_ridge_nocoords_tuned   # y 173-176
python 27_net_table.py --models 148 168 169 170 171 152 172 173 174 175 176
# Ronda 10 (monotonía, hojas lineales y kNN), en este orden:
python 04_train_models.py --specs 177 178 179 180 --schemes spatial northeast   # ~65 min
python 20_twin_rules.py --model 177_boosting_pt_core_coords_by_type_pinball_ridge_nocoords_s10 --rules mix --numbers 181   # y 178-180 -> 182-184
python 28_tune_knn.py                     # ~3 min
python 04_train_models.py --specs 185 --schemes spatial northeast
python 06_ensemble.py --number 186 --how weighted --weights 0.7 0.3 --models     181_boosting_pt_core_coords_by_type_pinball_ridge_nocoords_s10_twin_mix     148_quantile_net_ext_core_coords_by_type_q70_ridge_nocoords   # 187 y 188 suman la 185
python 30_probe_top.py --model 152_ensemble_w70_30_2 --number 189 --share 0.05 --factor 1.15
python 07_results_table.py
python 29_round10_table.py --models 127 152 177 178 179 180 181 182 183 184 185 186 187 188 189
# Ronda 11 (BETO). beto_predictions.csv viene de beto_kaggle.py en Kaggle con GPU, sobre la
# salida de 37_beto_input.py. Con poca memoria, PS3_MAX_JOBS=2 limita las particiones en paralelo.
python 38_beto_check.py
python 04_train_models.py --specs 193 194 195 196 199 200 --schemes spatial northeast
python 20_twin_rules.py --model 193_boosting_pt_core_coords_by_type_level_q70_beto_nocoords_s10 --rules mix --numbers 197
python 20_twin_rules.py --model 194_boosting_pt_core_coords_by_type_level_q70_ridge_beto_nocoords_s10 --rules mix --numbers 198
python 20_twin_rules.py --model 199_boosting_pt_core_coords_by_type_level_q65_ridge_beto_nocoords_s10 --rules mix --numbers 201
python 20_twin_rules.py --model 200_boosting_pt_core_coords_by_type_level_q75_ridge_beto_nocoords_s10 --rules mix --numbers 202
python 06_ensemble.py --number 203 --how weighted --weights 0.7 0.3 --models 198_boosting_pt_core_coords_by_type_level_q70_ridge_beto_nocoords_s10_twin_mix 196_quantile_net_ext_core_coords_by_type_q70_ridge_beto_nocoords
python 06_ensemble.py --number 204 --how weighted --weights 0.6 0.4 --models 198_boosting_pt_core_coords_by_type_level_q70_ridge_beto_nocoords_s10_twin_mix 196_quantile_net_ext_core_coords_by_type_q70_ridge_beto_nocoords
python 06_ensemble.py --number 205 --how weighted --weights 0.7 0.3 --models 198_boosting_pt_core_coords_by_type_level_q70_ridge_beto_nocoords_s10_twin_mix 148_quantile_net_ext_core_coords_by_type_q70_ridge_nocoords
python 40_clip_range.py --model 203_ensemble_w70_30_2 --number 206
python 40_clip_range.py --model 204_ensemble_w60_40_2 --number 207
python 40_clip_range.py --model 198_boosting_pt_core_coords_by_type_level_q70_ridge_beto_nocoords_s10_twin_mix --number 208
python 39_beto_table.py
# Diversidad de errores (Neislen). 43 reescribe en oof_predictions.csv las columnas de estas
# especificaciones; 45 escribe los envíos 209 a 216 desde test_predictions.csv.
python 43_rebuild_oof.py --specs 3 45 54 65 77 --schemes random_twins spatial chapinero northeast
python 44_analyze_oof.py
python 45_new_submissions.py
python 07_results_table.py                # tabla de resultados para el README
python 08_calibrate_holdout.py            # opcional: ~5 min por esquema candidato
python 09_level_calibration.py --number 57 --factors 1.10 1.15 1.20 \
    --submission 54_en_boosting_core_coords_residual_q65
# Ensambles y calibraciones de la ronda 2 (antes en run_all.sh):
python 06_ensemble.py --number 30 --how median --models 17_boosting_full_coords_by_type_per_m2 04_elastic_net_full 18_random_forest_full_by_type_per_m2
python 06_ensemble.py --number 55 --how mean --models 54_en_boosting_core_coords_residual_q65 48_boosting_core_coords_by_type_level_trunc
python 06_ensemble.py --number 56 --how mean --models 54_en_boosting_core_coords_residual_q65 48_boosting_core_coords_by_type_level_trunc 04_elastic_net_full
python 06_ensemble.py --number 63 --how mean --models 54_en_boosting_core_coords_residual_q65 62_boosting_core_coords_by_type_level_trunc_up
python 09_level_calibration.py --number 32 --factors 1.15 --model none --submission 10_boosting_full_coords
python 09_level_calibration.py --number 60 --factors 1.20 --model none --submission 10_boosting_full_coords
python 06_ensemble.py --number 84 --how weighted --weights 0.9 0.1 --models 83_boosting_ext_core_coords_by_type_level_q70_ridge_s10 78_quantile_net_core_coords_by_type_q70
python 06_ensemble.py --number 85 --how weighted --weights 0.8 0.2 --models 83_boosting_ext_core_coords_by_type_level_q70_ridge_s10 78_quantile_net_core_coords_by_type_q70
python 12_text_report.py --svd 20
```

El 32 y el 60 escalan `10_boosting_full_coords.csv`, el envío sin OSM que se subió a Kaggle;
en un clon sin envíos, 04 escribe ese nombre con la versión con OSM y estos dos cambian.
`05_tune_models.py` guarda los mejores hiperparámetros en `stores/processed/best_params.json`
(versionado); `04_train_models.py` los usa solo en las especificaciones marcadas `tuned=True`.

## Datos y predictores

| Bloque | Predictores | Fuente |
|---|---|---|
| Estructurales | tipo de inmueble, alcobas, baños, cuartos, área (y su logaritmo), indicador de área faltante, tendencia mensual | Properati |
| Texto (18) | parqueadero y número de parqueaderos, terraza, balcón, ascensor, depósito, chimenea, vigilancia, gimnasio, piscina, conjunto cerrado, remodelado, dúplex, penthouse, vista, estrato, piso, longitud de la descripción | título y descripción |
| Puntos de referencia (4) | distancia a Zona T, Parque de la 93, Centro Internacional y Plaza de Bolívar | coordenadas aproximadas |
| OpenStreetMap (16) | distancia al más cercano de 11 categorías (parque, estación de transporte, universidad, colegio, hospital, centro comercial, restaurante, bar, banco, supermercado, policía) y conteos a 500 m de 5 de ellas | Overpass API |
| Solo en `lux` (12) | área ampliada con más patrones del texto (`surface_ext`, recupera 1 punto de faltante) y señales de lujo: terraza privada, ascensor privado, club house, doble altura, vista a los cerros, penthouse estricto, estrato 6 explícito, 3 o más parqueaderos y su conteo | texto |
| Solo en `ext_core` (2) | estrato socioeconómico de la manzana (0 = sin uso residencial, como faltante) y valor de referencia catastral del terreno por m2 de la vigencia del año del anuncio | Secretaría Distrital de Planeación y Catastro Distrital (`11_download_bogota.py`) |
| Texto completo (dentro de cada partición) | TF-IDF de palabras y bigramas del título y la descripción, resumido como SVD (20 componentes) o como un ridge del log del precio fuera de muestra (`txt_ridge`) | título y descripción |
| Solo en `rich` (24) | 8 de texto (cuarto de servicio, salón de conductores, vestier, estudio, lujo, spa, exterior, antigüedad), distancia a la Carrera Séptima y a los cerros (trazados aproximados), conteos de OSM a 300 m y 1 km | texto, OSM |

El área falta en cerca del 80 % de los anuncios; se recupera de la descripción cuando aparece
(queda faltante en 39 % de train y 36 % de test). Baños faltantes también se completan con el texto.

Conjuntos de predictores usados en los modelos: `base` (estructurales), `base_text`,
`full` (todo menos lat/lon crudos, para que el modelo aprenda de distancias y no de
coordenadas que en Chapinero casi no vio), `full_coords` (agrega lat/lon) y `rich` /
`rich_coords` (agregan la segunda tanda; no mejoraron y solo se usan en las filas 19 y 20),
`lux` / `lux_coords` (`full` con el área ampliada y las señales de lujo; filas 35 y 36) y
`core` / `core_coords` (`full` sin OSM, como los envíos 01–11 que se subieron primero).

Datos de © OpenStreetMap contributors (ODbL). Estrato: "Estratificación Manzana Bogotá D.C."
(SDP); valor del terreno: "Valor de referencia por metro cuadrado de terreno" (UAECD), ambos de
los datos abiertos de Bogotá, consultados el 19 de noviembre de 2026.

## Validación

- **Aleatoria:** 5 particiones al azar.
- **Espacial:** k-means sobre lat/lon define 10 zonas; se dejan zonas completas por fuera
  (GroupKFold de 5). Imita el salto del resto de Bogotá a Chapinero.
- **Holdout tipo Chapinero (ronda 1):** se valida en los inmuebles de train a menos de 250 m
  de algún inmueble del test y se entrena con el resto, menos una franja de 400 m. Daba
  182 M para el envío 10 (180 M en Kaggle), pero no reproduce el orden de los puntajes: el
  ensamble 31 le ganaba al 10 por 18 M y en Kaggle perdió por 9 M.
- **Holdout nororiental (ronda 2, criterio actual):** misma validación (train a menos de
  250 m del test), sin franja, y el ajuste usa solo los inmuebles de hasta 1.000 M. Imita
  dos rasgos del problema real: el train está truncado en 1.650 M (ver abajo) y el test tiene
  inmuebles de train muy cerca (mediana de 96 m). De 17 esquemas probados con
  `08_calibrate_holdout.py`, es de los dos que reproducen el orden de los cinco puntajes
  públicos, y el que mejor reproduce las diferencias entre ellos (tabla abajo).

**El train está truncado.** Todos los precios están entre 300 M y 1.650 M, sin masa
apilada en el tope (106 anuncios en 1.650 M frente a 238 en 1.600 M): se quitaron los
anuncios por fuera de ese rango, no se recortaron. Por encima de 180 m2 el precio medio
deja de crecer con el área (1.111 M entre 180 y 250 m2, 1.151 M entre 250 y 400 m2 cerca del
test). Un árbol nunca predice por encima de lo que vio, así que en los inmuebles grandes y
de lujo del test se queda corto.

Por defecto los modelos se entrenan sobre log(precio) y predicen en pesos; las variantes
`_level` y `_per_m2` usan el precio en nivel y log(precio por m2) cuando hay área.
`_by_type` ajusta un modelo para apartamentos y otro para casas.
En la ronda 2 (filas 33 en adelante) hay además, todo dentro de `models.py` y calculado en
cada ajuste:
- `en_boosting` (`_residual`): elastic net sobre log(precio) con área, baños, alcobas,
  parqueaderos, estrato y distancias, y boosting sobre sus residuos. La parte lineal no tiene
  techo.
- `_q55`/`_q60`/`_q65`: boosting con pérdida cuantílica.
- `_l2`: pérdida cuadrática.
- `_pw3`: pesos de 1 a 3 por decil de precio.
- `_smoter`: sintéticos interpolados entre apartamentos caros del nororiente, solo con las
  filas de ajuste.
- `_nbr`: mediana del precio y del precio por m2 de los 20 vecinos, sin contarse a sí mismo.
- `_imputed`: área faltante imputada con un boosting.
- `_trunc`: corrección por truncamiento. Pasa de la mediana truncada a la mediana sin
  truncar, suponiendo log-precio normal, con la escala estimada fuera de muestra dentro del
  ajuste.

## Resultados actuales

Salen de `views/cv_results.csv` (tabla generada con `07_results_table.py`). Las filas 01–31
usan los 16 predictores de OSM; las `core` no. La tabla anterior, sin OSM, que corresponde a
los envíos 01–11, está en `views/cv_results_pre_osm.csv`. Las filas de la ronda 2 que no se
conservaron solo se evaluaron en el holdout nororiental (guion en las demás columnas).

Cifras en millones de pesos.

| Modelo | Predictores | MAE aleatoria | MAE espacial | MAE tipo Chapinero | MAE nororiental | MAE ponderado | Sesgo nororiental | % sobrepredicho (nororiental) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `01_linear_base` | 8 | 187.4 | 202.2 | 234.8 | 247.0 | 244.8 | -205.3 | 25% |
| `02_linear_base_text` | 26 | 173.3 | 186.0 | 211.3 | 221.7 | 219.7 | -177.8 | 28% |
| `03_linear_full` | 46 | 155.8 | 168.1 | 183.3 | 190.0 | 188.7 | -103.8 | 41% |
| `04_elastic_net_full` | 46 | 132.8 | 159.7 | 172.0 | 179.3 | 178.1 | -107.7 | 37% |
| `05_cart_full` | 46 | 130.6 | 169.9 | 195.6 | 200.9 | 199.1 | -132.4 | 33% |
| `06_random_forest_full` | 46 | 93.9 | 147.6 | 180.2 | 193.4 | 191.5 | -137.5 | 33% |
| `07_random_forest_full_coords` | 48 | 93.0 | 146.6 | 181.2 | 191.6 | 189.8 | -134.5 | 34% |
| `08_boosting_base_text` | 26 | 124.4 | 150.7 | 181.2 | 192.9 | 190.5 | -158.9 | 24% |
| `09_boosting_full` | 46 | 96.5 | 145.8 | 182.3 | 177.8 | 176.2 | -120.5 | 33% |
| `100_elastic_net_ext_core_coords_splines_by_type_level_q70_ridge` | 35 | – | 159.5 | 154.4 | 167.5 | 165.8 | -63.8 | 49% |
| `101_linear_ext_core_coords_splines_by_type_level_q70_ridge` | 35 | – | 163.9 | 157.7 | 167.7 | 166.0 | -64.7 | 49% |
| `102_cart_ext_core_coords_by_type_level_q70_ridge` | 35 | – | 164.0 | 171.8 | 174.0 | 172.4 | -56.8 | 48% |
| `103_random_forest_ext_core_coords_by_type_level_q70_ridge_tuned` | 35 | – | 147.4 | 149.9 | 161.8 | 160.2 | -40.8 | 53% |
| `104_elastic_net_ext_core_coords_splines_by_type_level_q70_ridge_tuned` | 35 | – | 158.5 | 155.1 | 167.5 | 165.8 | -65.7 | 49% |
| `105_cart_ext_core_coords_by_type_level_q70_ridge_tuned` | 35 | – | 159.8 | 169.3 | 170.3 | 168.7 | -59.4 | 48% |
| `106_random_forest_ext_core_coords_by_type_level_q70_net_tuned` | 803 | – | 151.8 | 154.6 | 165.5 | 163.9 | -45.2 | 53% |
| `107_random_forest_ext_core_coords_by_type_level_q70_ridge_net_tuned` | 803 | – | 145.7 | 147.9 | 161.7 | 160.0 | -42.5 | 53% |
| `108_linear_ext_core_coords_splines_by_type_level_q70_ridge_oof` | 35 | – | 164.1 | 157.8 | 167.6 | 166.0 | -64.6 | 49% |
| `109_elastic_net_ext_core_coords_splines_by_type_level_q70_ridge_tuned_oof` | 35 | – | 158.8 | 155.1 | 167.4 | 165.7 | -65.4 | 49% |
| `10_boosting_full_coords` | 48 | 95.6 | 145.4 | 179.2 | 174.7 | 173.1 | -113.8 | 35% |
| `110_cart_ext_core_coords_by_type_level_q70_ridge_tuned_oof` | 35 | – | 161.6 | 169.4 | 170.3 | 168.6 | -56.8 | 50% |
| `112_random_forest_ext_core_coords_by_type_level_q70_ridge_net_resid` | 803 | – | 145.3 | 153.3 | 161.0 | 159.4 | -32.3 | 55% |
| `113_random_forest_ext_core_coords_by_type_level_q70_ridge_net_extra` | 803 | – | 147.1 | 141.1 | 159.0 | 157.5 | -50.4 | 51% |
| `114_random_forest_ext_core_coords_by_type_level_q70_ridge_net_nocoords` | 803 | – | 143.8 | 147.5 | 161.3 | 159.6 | -45.3 | 52% |
| `115_random_forest_ext_core_coords_by_type_level_q70_ridge_net_extra_nocoords` | 803 | – | 146.5 | 140.4 | 159.2 | 157.6 | -51.9 | 51% |
| `116_ensemble_w80_20_2` | – | – | 132.4 | 142.2 | 156.8 | 155.1 | -53.4 | 51% |
| `117_ensemble_w70_30_2` | – | – | 133.2 | 142.4 | 157.1 | 155.3 | -52.4 | 51% |
| `118_ensemble_w60_40_2` | – | – | 134.1 | 142.6 | 157.4 | 155.7 | -51.4 | 51% |
| `119_ensemble_w50_50_2` | – | – | 135.3 | 143.1 | 157.9 | 156.2 | -50.4 | 52% |
| `11_neural_net_full` | 46 | 146.8 | 196.6 | 237.2 | 194.8 | 193.0 | -44.9 | 46% |
| `120_boosting_pt_core_coords_by_type_level_q70_ridge` | 42 | 95.2 | 129.2 | 140.4 | 157.8 | 155.9 | -53.6 | 51% |
| `121_boosting_ext_core_coords_by_type_level_q70_ridge_nocoords` | 35 | 98.6 | 130.7 | 143.1 | 158.0 | 156.4 | -58.7 | 49% |
| `122_boosting_ext_core_coords_by_type_level_q70_ridge_coords3` | 35 | 97.0 | 132.6 | 143.1 | 157.6 | 155.8 | -55.5 | 50% |
| `123_boosting_pt_core_coords_by_type_level_q70_ridge_nocoords` | 42 | 96.7 | 127.7 | 139.8 | 158.0 | 156.1 | -60.0 | 49% |
| `124_boosting_pt_core_coords_by_type_level_q70_ridge_s10` | 42 | 94.4 | 128.4 | 139.9 | 157.3 | 155.4 | -54.8 | 51% |
| `125_boosting_pt_core_coords_by_type_level_q70_ridge_nocoords_s10` | 42 | 95.9 | 127.1 | – | 157.6 | 155.7 | -60.9 | 49% |
| `126_boosting_pt_core_coords_by_type_level_q70_ridge_nocoords_s10_twin_median` | – | 82.9 | 127.1 | – | 157.5 | 155.5 | -62.2 | 48% |
| `127_boosting_pt_core_coords_by_type_level_q70_ridge_nocoords_s10_twin_mix` | – | 89.0 | 127.0 | – | 157.4 | 155.5 | -61.6 | 49% |
| `128_boosting_pt_core_coords_by_type_level_q70_ridge_nocoords_s10_twin_avg` | – | 96.7 | 127.5 | – | 157.9 | 156.0 | -60.9 | 49% |
| `129_boosting_pt_core_coords_by_type_level_q70_ridge_nocoords_s10_twin_mix_avg` | – | 89.4 | 127.2 | – | 157.6 | 155.7 | -61.6 | 49% |
| `12_boosting_full_coords_weighted` | 48 | 112.7 | 150.1 | 176.5 | 181.2 | 179.1 | -121.0 | 36% |
| `130_linear_ext_core_coords_by_type_level_q70_ridge_oof_nogeo` | 35 | 139.3 | 156.3 | 151.6 | 167.1 | 165.5 | -62.8 | 49% |
| `131_elastic_net_ext_core_coords_by_type_level_q70_ridge_tuned_oof_nogeo` | 35 | 139.9 | 156.8 | 151.3 | 167.1 | 165.5 | -62.8 | 49% |
| `132_boosting_ext_core_coords_by_type_level_q55_ridge_pw2` | 35 | – | – | – | 158.7 | 157.0 | -78.6 | 43% |
| `133_boosting_ext_core_coords_by_type_level_q60_ridge_pw2` | 35 | – | – | – | 158.4 | 156.7 | -68.7 | 47% |
| `134_boosting_ext_core_coords_by_type_level_q65_ridge_pw2` | 35 | – | 132.7 | – | 158.6 | 156.8 | -59.9 | 49% |
| `135_boosting_ext_core_coords_by_type_level_q70_ridge_pw2` | 35 | – | – | – | 160.2 | 158.5 | -48.6 | 52% |
| `136_boosting_ext_core_coords_by_type_level_q55_ridge_pw4` | 35 | – | – | – | 159.8 | 158.1 | -73.8 | 45% |
| `137_boosting_ext_core_coords_by_type_level_q60_ridge_pw4` | 35 | – | – | – | 160.1 | 158.4 | -61.7 | 49% |
| `138_boosting_ext_core_coords_by_type_level_q65_ridge_pw4` | 35 | – | 138.1 | – | 161.3 | 159.5 | -51.0 | 51% |
| `139_boosting_ext_core_coords_by_type_level_q70_ridge_pw4` | 35 | – | – | – | 163.3 | 161.5 | -40.3 | 54% |
| `13_random_forest_full_weighted` | 46 | 107.1 | 151.2 | 179.2 | 193.9 | 191.8 | -137.8 | 34% |
| `140_boosting_ext_core_coords_by_type_level_q55_ridge_os2` | 35 | – | – | – | 158.0 | 156.4 | -74.2 | 44% |
| `141_boosting_ext_core_coords_by_type_level_q60_ridge_os2` | 35 | – | 127.3 | – | 157.8 | 156.1 | -68.2 | 46% |
| `142_boosting_ext_core_coords_by_type_level_q65_ridge_os2` | 35 | – | – | – | 157.9 | 156.2 | -57.2 | 49% |
| `143_boosting_ext_core_coords_by_type_level_q70_ridge_os2` | 35 | – | – | – | 159.5 | 157.7 | -46.3 | 52% |
| `144_boosting_ext_core_coords_by_type_level_q55_ridge_os3` | 35 | – | 125.0 | – | 158.7 | 156.9 | -72.3 | 45% |
| `145_boosting_ext_core_coords_by_type_level_q60_ridge_os3` | 35 | – | – | – | 158.9 | 157.1 | -61.9 | 48% |
| `146_boosting_ext_core_coords_by_type_level_q65_ridge_os3` | 35 | – | – | – | 159.2 | 157.4 | -53.8 | 50% |
| `147_boosting_ext_core_coords_by_type_level_q70_ridge_os3` | 35 | – | – | – | 161.8 | 159.9 | -41.5 | 53% |
| `148_quantile_net_ext_core_coords_by_type_q70_ridge_nocoords` | 35 | 118.0 | 137.6 | 141.6 | 149.5 | 148.2 | -47.0 | 51% |
| `149_boosting_ext_core_coords_by_type_level_q70_ridge_tfidfnet` | 35 | – | – | – | 157.3 | 155.6 | -56.2 | 51% |
| `14_boosting_full_coords_by_type` | 48 | 90.5 | 142.7 | 177.6 | 175.6 | 174.1 | -116.5 | 35% |
| `150_ensemble_w70_30_2` | – | – | 131.1 | 139.3 | 153.1 | 151.5 | -52.9 | 51% |
| `151_ensemble_w50_50_2` | – | – | 132.0 | 138.6 | 151.4 | 149.9 | -51.2 | 51% |
| `152_ensemble_w70_30_2` | – | 95.5 | 127.6 | – | 153.5 | 151.7 | -57.2 | 50% |
| `153_ensemble_w50_50_2` | – | 101.1 | 129.4 | – | 151.6 | 150.0 | -54.3 | 50% |
| `154_boosting_ext_core_coords_by_type_level_q60_ridge` | 35 | – | 126.1 | – | 157.1 | 155.5 | -74.5 | 45% |
| `155_boosting_ext_core_coords_by_type_level_q65_ridge` | 35 | – | 126.7 | – | 156.6 | 154.9 | -65.8 | 47% |
| `156_boosting_ext_core_coords_by_type_level_q75_ridge` | 35 | – | 138.7 | – | 158.8 | 157.1 | -43.7 | 53% |
| `157_boosting_ext_core_coords_by_type_level_q80_ridge` | 35 | – | 147.6 | – | 162.1 | 160.3 | -25.8 | 58% |
| `158_boosting_ext_core_coords_by_type_level_q70_ridge_local_ne` | 35 | – | 160.6 | – | 165.0 | 163.3 | -60.4 | 48% |
| `159_boosting_ext_core_coords_by_type_level_q70_ridge_local_3km` | 35 | – | 191.7 | – | 169.0 | 167.2 | -65.9 | 48% |
| `15_random_forest_full_by_type` | 46 | 91.5 | 146.8 | 177.5 | 193.7 | 192.0 | -137.7 | 34% |
| `160_boosting_ext_core_coords_by_type_level_q70_ridge_local_5km` | 35 | – | 154.4 | – | 160.7 | 159.1 | -61.9 | 49% |
| `161_boosting_ext_core_coords_by_type_level_q70_ridge_local_8km` | 35 | – | 140.4 | – | 158.1 | 156.4 | -56.3 | 50% |
| `162_boosting_ext_core_coords_by_type_level_q70_ridge_notwins` | 35 | – | 132.7 | – | 157.9 | 156.2 | -54.6 | 50% |
| `163_boosting_ext_core_coords_by_type_level_q70_ridge_twinweight` | 35 | – | 131.7 | – | 157.3 | 155.6 | -57.3 | 50% |
| `164_boosting_ext_core_coords_by_type_level_q70_ridge_trim` | 35 | – | 133.0 | – | 158.7 | 157.0 | -58.8 | 49% |
| `165_boosting_pt_core_coords_by_type_per_m2_q70_ridge` | 42 | – | 126.7 | – | 159.3 | 157.4 | -59.9 | 49% |
| `166_stack_lineal_mediana_3` | – | – | – | – | – | 169.9 | – | – |
| `167_stack_lgbm_l1_3` | – | – | – | – | – | 178.0 | – | – |
| `168_quantile_net_ext_core_coords_by_type_q70_ridge_nocoords_s10` | 35 | – | 137.3 | – | 149.4 | 148.0 | -46.0 | 51% |
| `169_quantile_net_pt_core_coords_by_type_q70_ridge_nocoords` | 42 | – | 136.7 | – | 147.8 | 146.1 | -41.5 | 52% |
| `16_boosting_full_coords_by_type_level` | 48 | 90.8 | 144.5 | 171.2 | 173.0 | 171.5 | -109.8 | 36% |
| `170_quantile_net_ext_core_coords_by_type_q70_wide_embnet_nocoords` | 803 | – | 143.6 | – | 154.2 | 152.8 | -51.7 | 49% |
| `171_quantile_net_pt_core_coords_by_type_q70_ridge_nocoords_tuned` | 42 | – | 132.4 | – | 152.3 | 150.7 | -61.3 | 47% |
| `172_ensemble_w70_30_2` | – | – | 126.6 | – | 154.5 | 152.7 | -61.5 | 48% |
| `173_ensemble_w60_40_2` | – | – | 126.9 | – | 153.8 | 152.0 | -61.5 | 48% |
| `174_ensemble_w50_50_2` | – | – | 127.4 | – | 153.2 | 151.5 | -61.4 | 48% |
| `175_ensemble_w60_20_20_3` | – | – | 127.1 | – | 153.3 | 151.5 | -59.5 | 49% |
| `176_ensemble_w50_25_25_3` | – | – | 127.8 | – | 152.6 | 150.9 | -59.0 | 48% |
| `177_boosting_pt_core_coords_by_type_pinball_ridge_nocoords_s10` | 42 | – | 123.8 | – | 157.9 | 156.0 | -69.7 | 46% |
| `178_boosting_pt_core_coords_by_type_pinball_mono_ridge_nocoords_s10` | 42 | – | 127.4 | – | 154.4 | 152.8 | -61.5 | 48% |
| `179_boosting_pt_core_coords_by_type_pinball_mono_txt_ridge_nocoords_s10` | 42 | – | 133.0 | – | 154.7 | 153.3 | -58.2 | 48% |
| `17_boosting_full_coords_by_type_per_m2` | 48 | 90.7 | 142.6 | 169.9 | 174.9 | 173.3 | -116.2 | 35% |
| `180_boosting_pt_core_coords_by_type_pinball_linear_ridge_nocoords_s5` | 42 | – | 124.0 | – | 157.5 | 155.7 | -68.7 | 47% |
| `181_boosting_pt_core_coords_by_type_pinball_ridge_nocoords_s10_twin_mix` | – | – | 123.7 | – | 157.7 | 155.8 | -70.2 | 46% |
| `182_boosting_pt_core_coords_by_type_pinball_mono_ridge_nocoords_s10_twin_mix` | – | – | 127.3 | – | 154.2 | 152.6 | -61.9 | 48% |
| `183_boosting_pt_core_coords_by_type_pinball_mono_txt_ridge_nocoords_s10_twin_mix` | – | – | 132.9 | – | 154.5 | 153.1 | -58.6 | 48% |
| `184_boosting_pt_core_coords_by_type_pinball_linear_ridge_nocoords_s5_twin_mix` | – | – | 123.9 | – | 157.4 | 155.5 | -69.2 | 46% |
| `185_knn_pt_core_coords_by_type_q70_tuned` | 42 | – | 169.5 | – | 179.8 | 178.0 | -71.6 | 47% |
| `186_ensemble_w70_30_2` | – | – | 123.9 | – | 153.2 | 151.5 | -63.2 | 48% |
| `187_ensemble_w60_25_15_3` | – | – | 125.7 | – | 154.8 | 153.0 | -64.6 | 48% |
| `188_ensemble_w65_25_10_3` | – | – | 124.7 | – | 154.3 | 152.6 | -64.5 | 48% |
| `189_probe_ensemble_w70_30_2_top5_x115` | – | 103.0 | 132.7 | – | 148.3 | 146.6 | -49.8 | 50% |
| `18_random_forest_full_by_type_per_m2` | 46 | 91.7 | 146.8 | 172.2 | 189.9 | 188.3 | -134.0 | 34% |
| `19_boosting_rich_coords_by_type_per_m2` | 72 | 89.4 | 144.1 | 174.5 | 173.2 | 171.4 | -115.9 | 35% |
| `20_random_forest_rich_by_type_per_m2` | 70 | 89.7 | 150.4 | 177.3 | 186.1 | 184.4 | -129.0 | 35% |
| `21_cart_full_tuned` | 46 | 137.6 | 165.9 | 193.2 | 204.1 | 202.0 | -141.4 | 33% |
| `22_neural_net_full_tuned` | 46 | 129.8 | 170.6 | 212.0 | 186.8 | 185.6 | -108.1 | 36% |
| `23_boosting_full_coords_by_type_per_m2_tuned` | 48 | 95.7 | 143.8 | 174.0 | 176.5 | 175.0 | -118.0 | 35% |
| `24_random_forest_full_by_type_per_m2_tuned` | 46 | 96.0 | 147.2 | 172.7 | 189.9 | 188.3 | -133.8 | 34% |
| `30_ensemble_median_3` | – | 93.0 | 142.2 | 167.3 | 178.8 | – | -121.4 | 35% |
| `31_ensemble_mean_3` | – | 99.9 | 142.9 | 164.6 | 177.8 | – | -119.3 | 35% |
| `33_en_boosting_full_coords_residual` | 48 | 96.8 | 145.3 | 168.4 | 166.5 | 165.2 | -96.6 | 38% |
| `34_boosting_full_coords_by_type_level_l2` | 48 | – | – | – | 172.9 | 171.3 | -106.3 | 37% |
| `35_boosting_lux_coords_by_type_per_m2` | 57 | – | – | – | 175.0 | 173.5 | -118.0 | 34% |
| `36_boosting_lux_coords_by_type_per_m2_imputed` | 57 | – | – | – | 174.3 | 172.6 | -115.1 | 34% |
| `37_boosting_full_coords_by_type_per_m2_nbr` | 48 | – | – | – | 178.2 | 176.5 | -118.9 | 35% |
| `38_random_forest_full_by_type_per_m2_nbr` | 48 | – | – | – | 191.6 | 190.0 | -134.3 | 34% |
| `39_boosting_full_coords_by_type_level_pw3` | 48 | – | – | – | 172.2 | 170.6 | -97.7 | 39% |
| `40_random_forest_full_by_type_per_m2_pw3` | 46 | – | – | – | 187.6 | 185.9 | -121.8 | 37% |
| `41_boosting_full_coords_by_type_level_q55` | 48 | – | – | – | 171.0 | 169.5 | -99.3 | 39% |
| `42_boosting_full_coords_by_type_level_q60` | 48 | – | – | – | 168.6 | 166.9 | -88.4 | 41% |
| `43_boosting_full_coords_by_type_level_q65` | 48 | 94.9 | 149.9 | 160.1 | 167.8 | 166.1 | -76.8 | 45% |
| `44_boosting_full_coords_by_type_level_smoter` | 48 | – | – | – | 170.9 | 169.4 | -102.6 | 38% |
| `45_boosting_core_coords_by_type_level` | 32 | 93.0 | 135.8 | 174.1 | 168.5 | 167.0 | -106.5 | 36% |
| `46_boosting_full_coords_by_type_level_trunc` | 48 | 94.5 | 145.1 | 171.4 | 169.8 | 168.4 | -100.0 | 36% |
| `47_random_forest_full_by_type_per_m2_trunc` | 46 | – | – | – | 188.6 | 187.0 | -132.0 | 34% |
| `48_boosting_core_coords_by_type_level_trunc` | 32 | – | – | – | 168.1 | 166.9 | -88.3 | 38% |
| `49_en_boosting_core_coords_residual` | 32 | 98.3 | 137.4 | 173.2 | 163.6 | 162.3 | -93.1 | 39% |
| `50_en_boosting_full_coords_residual_q60` | 48 | – | – | – | 162.2 | 160.9 | -67.8 | 46% |
| `51_en_boosting_core_coords_residual_q60` | 32 | 99.4 | 139.5 | 165.5 | 159.2 | 157.9 | -69.2 | 45% |
| `52_en_boosting_core_coords_residual_trunc` | 32 | – | – | – | 219.7 | 217.1 | +0.2 | 44% |
| `53_boosting_core_coords_by_type_level_q65` | 32 | 97.3 | 142.5 | 162.0 | 162.7 | 161.1 | -69.6 | 46% |
| `54_en_boosting_core_coords_residual_q65` | 32 | 102.3 | 142.7 | 157.9 | 157.8 | 156.4 | -52.3 | 49% |
| `55_ensemble_mean_2` | – | – | – | – | 158.5 | – | -70.3 | 45% |
| `56_ensemble_mean_3` | – | – | – | – | 160.4 | – | -82.7 | 41% |
| `61_en_boosting_lux_core_coords_residual_q65` | 41 | – | – | – | 157.2 | 155.9 | -51.5 | 48% |
| `62_boosting_core_coords_by_type_level_trunc_up` | 32 | – | – | – | 164.7 | 163.4 | -92.7 | 38% |
| `63_ensemble_mean_2` | – | – | – | – | 157.7 | – | -72.5 | 44% |
| `65_boosting_core_coords_by_type_level_q70` | 32 | 101.6 | 148.9 | 159.0 | 162.9 | 161.3 | -57.9 | 49% |
| `66_boosting_core_coords_by_type_level_q75` | 32 | 108.2 | 153.1 | 159.7 | 164.0 | 162.4 | -45.0 | 52% |
| `67_boosting_core_coords_by_type_level_q70_bld` | 32 | 101.2 | 150.1 | 156.6 | 167.1 | 165.6 | -79.8 | 44% |
| `68_boosting_core_coords_by_type_level_q70_s10` | 32 | – | – | 159.0 | 162.4 | 160.8 | -58.3 | 49% |
| `70_boosting_core_coords_by_type_level_q70_ft` | 32 | – | 148.8 | 158.6 | 163.0 | 161.3 | -58.8 | 49% |
| `71_boosting_core_coords_by_type_level_q70_bld30` | 32 | – | 147.8 | 159.1 | 163.3 | 161.6 | -59.2 | 48% |
| `72_boosting_core_coords_by_type_level_q70_svd20` | 33 | – | 148.8 | 157.0 | 162.8 | 161.3 | -63.3 | 48% |
| `75_boosting_core_coords_by_type_level_q70_ridge` | 33 | – | 133.0 | 148.5 | 157.4 | 155.6 | -56.5 | 49% |
| `76_boosting_core_coords_by_type_level_q70_names` | 33 | – | 147.7 | 162.5 | 173.1 | 171.5 | -73.8 | 47% |
| `77_boosting_ext_core_coords_by_type_level_q70` | 34 | – | 146.0 | 151.2 | 162.7 | 161.1 | -55.0 | 50% |
| `78_quantile_net_core_coords_by_type_q70` | 32 | – | 164.0 | 156.0 | 156.6 | 155.3 | -46.4 | 49% |
| `79_quantile_net_core_coords_by_type_q70_smoter` | 32 | – | 167.0 | 156.4 | 157.0 | 155.7 | -28.4 | 54% |
| `80_boosting_core_coords_by_type_level_q70_tuned` | 32 | – | 146.9 | 161.7 | 161.8 | 160.2 | -61.0 | 48% |
| `82_boosting_ext_core_coords_by_type_level_q70_ridge` | 35 | – | 132.6 | 142.5 | 157.1 | 155.4 | -56.6 | 49% |
| `83_boosting_ext_core_coords_by_type_level_q70_ridge_s10` | 35 | – | 131.5 | 142.4 | 156.6 | 154.8 | -55.4 | 50% |
| `84_ensemble_w90_10_2` | – | – | 132.2 | 141.6 | 155.3 | 153.6 | -54.5 | 50% |
| `85_ensemble_w80_20_2` | – | – | 133.7 | 141.3 | 154.4 | 152.7 | -53.6 | 51% |
| `86_boosting_core_coords_by_type_level_q70_net` | 801 | – | 136.8 | – | 160.6 | 158.8 | -59.3 | 49% |
| `87_boosting_core_coords_by_type_level_q70_ridge_net` | 801 | – | 130.5 | – | 158.0 | 156.2 | -56.7 | 50% |
| `88_boosting_core_coords_by_type_level_q70_ridge_esvd20` | 801 | – | 136.5 | – | 158.5 | 156.7 | -60.8 | 48% |
| `89_boosting_core_coords_by_type_level_q70_ridge_tuned` | 33 | – | 134.5 | – | 156.7 | 154.9 | -57.1 | 49% |
| `90_boosting_core_coords_by_type_level_q70_ridge_type` | 33 | – | 133.4 | – | 156.8 | 155.1 | -54.7 | 50% |
| `91_boosting_core_coords_by_type_level_q70_ridge_title_desc` | 35 | – | 133.7 | – | 158.2 | 156.4 | -55.9 | 50% |
| `92_boosting_core_coords_by_type_level_q70_ridge_char` | 33 | – | 133.5 | – | 157.3 | 155.5 | -56.9 | 49% |
| `93_boosting_core_coords_by_type_level_q70_ridge_type_net` | 801 | – | 131.3 | – | 156.6 | 154.8 | -54.4 | 50% |
| `94_boosting_ext_core_coords_by_type_level_q70_ridge_type_net_s10` | 803 | – | 129.6 | – | 156.2 | 154.5 | -53.5 | 51% |
| `95_catboost_ext_core_coords_by_type_level_q70_ridge_type_net` | 803 | – | 132.3 | – | 158.9 | 157.2 | -67.4 | 48% |
| `96_xgboost_ext_core_coords_by_type_level_q70_ridge_type_net` | 803 | – | 133.0 | – | 157.1 | 155.4 | -52.4 | 51% |
| `97_boosting_lot_core_coords_by_type_level_q70_ridge_type_net_s10` | 806 | – | 128.5 | – | 156.1 | 154.4 | -57.1 | 50% |
| `98_boosting_ext_core_coords_by_type_level_q70_ridge_type_net_s10_tpe` | 803 | – | 128.6 | – | 154.6 | 152.9 | -53.8 | 51% |
| `99_random_forest_ext_core_coords_by_type_level_q70_ridge` | 35 | – | 147.8 | 148.4 | 161.7 | 160.1 | -41.1 | 53% |

"MAE ponderado" es el holdout nororiental ponderado como el test (ronda 3a): el error de los
inmuebles con anuncios del mismo edificio pesa lo que pesan en el test (24 %). La columna
`mae_random_twins` de `views/cv_results.csv` repite la validación aleatoria con los gemelos
(mismo inmueble o mismo texto) siempre en la misma partición: sube unos 15 M frente a la
aleatoria simple (65: de 101.6 a 118.5).

Ronda 3b (filas 72 a 85):

- Lo que ayudó: el texto completo como un ridge fuera de muestra (75: −4.7 M ponderado,
  −12 M tipo Chapinero, −16 M espacial frente al 65) y el estrato y valor del terreno de la
  manzana (77: −7.8 M tipo Chapinero, −2.9 M espacial, sin cambio en el nororiental). Juntos
  y con 10 semillas (83): 154.8 ponderado, 142.4 tipo Chapinero y 131.5 espacial.
- Lo que no: la SVD del texto (72), el nombre del barrio en el título (76: +10 M), la
  sintonización de la receta del 65 (80: −1.1 M ponderado, pero +2.7 M tipo Chapinero) y los
  sintéticos SMOTER en la red (79: suben el nivel, no el orden).
- La red cuantílica por tipo (78) gana en los holdouts y pierde 15 M en la espacial, el mismo
  patrón del híbrido 54, que ganaba en el holdout y perdió 12 M en Kaggle frente al 53.

Modelos ingenuos (`views/cv_results_naive.csv`): la mediana constante da 236.7 / 248.9 / 298.1
y la mediana por tipo y alcobas 217.7 / 242.6 / 289.2 (aleatoria / espacial / tipo Chapinero).

### Puntajes públicos de Kaggle

En `views/kaggle_scores.csv`. El holdout nororiental se calcula reentrenando cada envío tal
como se subió (los envíos 10 y 11, sin OSM), con `08_calibrate_holdout.py`
(`views/holdout_calibration.csv`).

| Envío | Kaggle público | Holdout nororiental | Holdout tipo Chapinero |
|---|---:|---:|---:|
| `00_naive_1_median` | 377.1 | 298.5 | 298.1 |
| `11_neural_net_full` | 232.4 | 188.9 | 201.0 |
| `31_ensemble_mean_3` | 189.3 | 177.8 | 164.6 |
| `10_boosting_full_coords` | 180.1 | 169.8 | 182.2 |
| `32_boosting_full_coords_x115` | 171.7 | 161.7 | 162.4 |
| `49_en_boosting_core_coords_residual` | 184.1 | 163.6 | 173.2 |
| `54_en_boosting_core_coords_residual_q65` | 173.3 | 157.8 | 157.9 |
| `53_boosting_core_coords_by_type_level_q65` | 161.7 | 162.7 | 162.0 |
| `65_boosting_core_coords_by_type_level_q70` | 160.9 | 162.9 | 159.0 |
| `75_boosting_core_coords_by_type_level_q70_ridge_v2` | 153.1 | 157.4 | 148.5 |
| `83_boosting_ext_core_coords_by_type_level_q70_ridge_s10_v2` | 152.1 | 156.6 | 142.4 |
| `98_boosting_ext_core_coords_by_type_level_q70_ridge_type_net_s10_tpe` | 153.4 | 154.6 | sin calcular |
| `114_random_forest_ext_core_coords_by_type_level_q70_ridge_net_nocoords` | 158.9 | 161.3 | 147.5 |
| `105_cart_ext_core_coords_by_type_level_q70_ridge_tuned` | 177.5 | 170.3 | 169.3 |
| `108_linear_ext_core_coords_splines_by_type_level_q70_ridge_oof` | 180.5 | 167.6 | 157.8 |
| `148_quantile_net_ext_core_coords_by_type_q70_ridge_nocoords` | 157.1 | 149.5 | 141.6 |
| `152_ensemble_w70_30_2` (127 al 70 %, red 148 al 30 %) | **148.2** | 153.5 | sin calcular |

**Mejor envío: `152_ensemble_w70_30_2`**, la mezcla del boosting 127 (70 %) y la red 148
(30 %), con 148,2 M en el tablero público, 3,9 M mejor que la 83 (152,1).

**Ponderado y espacial frente a Kaggle** (`25_criteria_vs_kaggle.py`,
`views/criteria_vs_kaggle.csv`), en los 12 envíos que tienen los tres números:

| Envío | Kaggle | Ponderado | Espacial |
|---|---:|---:|---:|
| 152 mezcla 127/148 | 148,2 | 151,7 | 127,6 |
| 83 boosting | 152,1 | 154,8 | 131,5 |
| 75 boosting | 153,1 | 155,6 | 133,0 |
| 98 boosting TPE | 153,4 | 152,9 | 128,6 |
| 148 red | 157,1 | 148,2 | 137,6 |
| 114 bosque | 158,9 | 159,6 | 143,8 |
| 65 boosting | 160,9 | 161,3 | 148,9 |
| 53 boosting | 161,7 | 161,1 | 142,5 |
| 54 híbrido | 173,3 | 156,4 | 142,7 |
| 105 CART | 177,5 | 168,7 | 159,8 |
| 108 lineal | 180,5 | 166,0 | 164,1 |
| 49 híbrido | 184,1 | 162,3 | 137,4 |

| Criterio | Spearman con Kaggle | Pares bien ordenados | Brecha media frente a Kaggle |
|---|---:|---:|---:|
| Ponderado | 0,83 | 80 % | −5,2 |
| Espacial | 0,73 | 80 % | −21,9 |

El ponderado ordena algo mejor y acierta el nivel; la espacial es optimista en unos 22 M.
Los dos ordenan bien el mismo 80 % de los pares. Los errores del ponderado son modelos que
estiran la cola alta: la red 148 (148,2 frente a 157,1) y los híbridos 54 y 49. El holdout
entrena solo con precios de hasta 1.000 M, y el 18 % de su validación pasa de ese tope.
Ahí gana el modelo que extrapola hacia arriba, una ventaja que en el test, con el
entrenamiento hasta 1.650 M, vale mucho menos. La 82 entrenada sin ese tope baja de 155,4 a
138,6 M en el ponderado (`views/round8_ceiling.csv`).

Con el holdout ponderado (`mae_weighted`) frente a Kaggle: boosting 83 154,8 contra 152,1 y
bosque 114 159,6 contra 158,9 (acierta); CART 105 168,7 contra 177,5 y lineal 108 166,0
contra 180,5 (optimista en 9 a 15 M). El lineal con splines de coordenadas viaja mal a
Chapinero.

El texto completo del anuncio (ridge sobre TF-IDF, envío 75) explica 7,9 de los 8,9 M que
gana el 83 frente al 65 en Kaggle. El estrato y el valor del terreno por manzana, junto con
las 10 semillas, suman 1,0 M, dentro del ruido del tablero público.

Los envíos 49 a 98 se evaluaron directamente en `views/cv_results.csv`, sin pasar por 08.
Con ellos el holdout nororiental ya no ordena bien: pone el 54 (157.8) por delante del 53
(162.7) y del 65 (162.9), y en Kaggle el 54 queda 12 M detrás. Diferencias de pocos millones
en el holdout no sirven para escoger envíos.

El holdout nororiental pone los cinco en el mismo orden que Kaggle (Spearman 1). También
reproduce las diferencias: el ×1,15 mejora 8,1 M (8,5 en Kaggle) y el ensamble 31 pierde
8,0 M frente al 10 (9,1 en Kaggle). Los niveles quedan unos 10 M por debajo para los
árboles y más abajo para el ingenuo y la red. El tipo Chapinero ponía el ensamble 31
18 M por delante del 10.

Lectura:

- El mejor modelo nuevo es `54_en_boosting_core_coords_residual_q65`, con 157.8 en el
  holdout nororiental frente a 174.7 del boosting 10 con OSM. Bootstrap por celdas de
  500 m: −16.9 M, error estándar 2.7. Combina tres cosas que funcionaron por separado:
  - un elastic net en área, baños, parqueaderos y ubicación, con boosting sobre sus
    residuos (−8.1 M);
  - quitar OSM (−3 a −4,5 M);
  - pérdida cuantílica en 0,65 (−5 a −6 M).
- Ningún árbol predice por encima de lo que vio: en el holdout, el 17,9 % de los precios
  pasa el tope de 1.000 M con que se entrena, pero los boosting predicen por encima de él en
  0,1–0,7 % de los casos. En el decil más caro los árboles y los híbridos subpredicen entre
  517 y 700 M (la red 11, 409 M). En el test, los árboles apenas pasan de 1.650 M (máximo
  1.789 M); la parte lineal (filas 33 y 49–54) y la corrección por truncamiento (46 y 48)
  llegan a 3.300–4.500 M.
- OSM empeora la validación espacial (boosting en nivel: 144.5 con OSM, 135.8 sin OSM) y el
  holdout nororiental. En la ronda 1 se adoptó por el holdout tipo Chapinero.
- Lo que no ayudó:
  - el precio por m2 (17 frente a 10: +0.2);
  - el área ampliada y las señales de lujo (35: +0.1 frente a 17; en la parte lineal del
    híbrido, 61: −0.6 frente a 54, dentro del ruido);
  - la imputación del área (36: −0.7);
  - el nivel de los vecinos (37: +3.3);
  - los pesos por decil (39: −0.8);
  - los ensambles 55, 56 y 63 (158.5, 160.4 y 157.7, sin mejora sobre el 54 solo).
- Advertencias:
  - (i) el 157.8 es doblemente optimista: el esquema del holdout se escogió entre 17 por
    reproducir los puntajes públicos, y después las ~25 variantes se escogieron con ese
    mismo holdout;
  - (ii) el MAE espacial de las filas `_tuned` es optimista, porque 05 sintoniza con esas
    particiones;
  - (iii) las filas 12 y 13 ponderan con un clasificador train-contra-test que usa las
    covariables del test, nunca el precio;
  - (iv) las redes neuronales (11, 22) dan predicciones de hasta 5.900 M;
  - (v) la pérdida cuantílica sube el nivel: es en parte lo mismo que escalar, y su
    validez depende de que el holdout mida bien el nivel (lo hace con el ×1,15). Además
    empuja a sobrepredecir (el 54 sobrepredice el 49 % de las veces), que es el error caro
    para comprar: el modelo para Kaggle y el de la regla de compra no tienen por qué ser el
    mismo;
  - (vi) el 32 % de las filas de train tiene un gemelo exacto (mismas coordenadas,
    estructura y precio). Eso hace optimista la validación aleatoria, y en `_nbr` el gemelo
    aporta su precio aunque la fila se excluya a sí misma;
  - (vii) las filas 46, 47, 48 y 52 usan la primera versión de la corrección por
    truncamiento, que manda las predicciones cercanas al piso o al techo a los bordes de su
    búsqueda (110 M y 4.481 M). La 62 usa la versión corregida, que solo corrige el techo:
    164.7 frente a 168.5 sin corrección (45);
  - (viii) ronda 4: la fila 98 y `views/optuna_*.csv` son optimistas frente a la 94,
    porque 15 busca, confirma y compara cuantiles con las mismas particiones (holdout
    nororiental y validación espacial). La espacial de la 89 también lo es un poco: 14
    escoge el ridge con las filas de ajuste del holdout, que son casi todo train.

  - (ix) ronda 5: las filas 103 a 105 son algo optimistas, porque 17 sintoniza con el
    holdout nororiental y la validación espacial. Las filas 99 a 105 no tienen validación
    aleatoria clásica; la aleatoria con gemelos se agregó en la ronda 6 (99 a 115).
  - (x) ronda 6: la columna `tuned` de `cv_results.csv` dice `False` en las filas 109, 110,
    112, 113 y 115, que reciben los hiperparámetros sintonizados por `moved_params`.

### Ronda 5: la receta del boosting en las demás familias

Para comparar algoritmos en igualdad de condiciones, cada familia recibe lo que funcionó en
la 83: un modelo por tipo, estrato y valor del terreno, el ridge del texto fuera de muestra
y el precio en niveles llevado al cuantil 0,70 con lo que la familia permite
(`models.QUANTILE_RECIPES`):

- random forest: bosque de regresión cuantílica (`quantile-forest`), percentil 70 de los
  precios de entrenamiento de las hojas;
- CART: árbol sobre el precio en niveles que predice el percentil 70 de cada hoja;
- elastic net y regresión lineal: splines del log del área, splines aditivos de latitud y
  longitud con su interacción, esos splines por el área y el área por los acabados del
  texto, el estrato y el ridge; el nivel se corre con el percentil 70 de los residuos del
  ajuste. La regresión cuantílica (`QuantileRegressor`) tardó 6 minutos por ajuste y dio
  una media negativa: la lineal es mínimos cuadrados con la misma corrección. El producto
  tensorial de splines de ubicación se probó y se descartó (espacial de 246 M).

`views/round5_families.csv` (de `18_family_table.py`), en millones:

| Familia | Versión | MAE ponderado | MAE espacial | Media en test |
|---|---|---:|---:|---:|
| Regresión lineal | 03 → 101 | 188.7 → 166.0 | 168.1 → 163.9 | 708 → 860 |
| Elastic net | 04 → 104 | 178.1 → 165.8 | 159.7 → 158.5 | 717 → 847 |
| CART | 21 → 105 | 202.0 → 168.7 | 165.9 → 159.8 | 679 → 829 |
| Random forest | 24 → 103 | 188.3 → 160.2 | 147.2 → 147.4 | 731 → 880 |
| Red neuronal | 11 → 78 (ya existía) | 193.0 → 155.3 | 196.6 → 164.0 | 830 → 838 |
| Boosting | 83 | 154.8 | 131.5 | 882 |

Todas mejoran en el holdout ponderado (de 12 a 33 M) y quedan detrás del boosting, sobre
todo en la validación espacial. Trazas de las búsquedas en `views/tuning_*_q70.csv`.

### Ronda 6: el bosque contra el boosting

Sobre el bosque cuantílico de la 103 (160,2 ponderado, 147,4 espacial), en millones:

| Fila | Cambio | MAE ponderado | MAE espacial | Media en test |
|---|---|---:|---:|---:|
| 106 | texto con la red de la ronda 4 en vez del ridge | 163,9 | 151,8 | 863 |
| 107 | ridge + red | 160,0 | 145,7 | 870 |
| 112 | bosque sobre residuos de un lineal | 159,4 | 145,3 | 908 |
| 113 | ExtraTrees cuantílico | 157,5 | 147,1 | 842 |
| 114 | sin latitud ni longitud crudas | 159,6 | 143,8 | 866 |
| 115 | 113 + 114 | 157,6 | 146,5 | 838 |
| 116–119 | 83 con 114, pesos 80/20 a 50/50 | 155,1–156,2 | 132,4–135,3 | 874–878 |
| 83 | boosting (referencia) | 154,8 | 131,5 | 882 |

Ninguna variante mejora a la 103 por más de 2 M en los dos criterios, y ninguna mezcla le
gana a la 83. El genético (NSGA-II, 20 × 6, `views/genetic_*.csv`) no supera a los
hiperparámetros de la 103 con 5 semillas (153,26 ± 0,27 frente a 152,92 ± 0,14 M). La
corrección de nivel con residuos fuera de muestra (108 a 110) no cambia las lineales ni el
CART. La correlación entre la 83 y la 114 en test es 0,975.

### Ronda 7: ideas sobre la receta de la 83 y familias que faltan en Kaggle

Cifras en millones, de `views/cv_results.csv` y `views/round7_summary.csv`. Media, mediana
y p95 son del envío en test. Las filas de una semilla se comparan con la 82 (la 83 con una
semilla).

| Fila | Cambio | Ponderado | Espacial | Media test | Mediana | p95 |
|---|---|---:|---:|---:|---:|---:|
| 83 | boosting de referencia (10 semillas) | 154,8 | 131,5 | 882 | 860 | 1.457 |
| 82 | la 83 con una semilla | 155,4 | 132,6 | 883 | 864 | 1.443 |
| 120 | A1: área y baños del mismo punto | 155,9 | 129,2 | 875 | 849 | 1.447 |
| 121 | A4: sin lat ni lon | 156,4 | 130,7 | 880 | 858 | 1.455 |
| 122 | A4: lat y lon a 3 decimales | 155,8 | 132,6 | 880 | 862 | 1.447 |
| 123 | A1 + A4 sin coordenadas | 156,1 | 127,7 | 876 | 853 | 1.445 |
| 124 | A1, 10 semillas | 155,4 | 128,4 | 879 | 859 | 1.446 |
| 125 | A1 + A4, 10 semillas | 155,7 | 127,1 | 879 | 857 | 1.443 |
| 126 | 125 + A2, mediana de los gemelos de train | 155,5 | 127,1 | 877 | 854 | 1.446 |
| 127 | 125 + A2, mezcla 50/50 | 155,5 | 127,0 | 878 | 854 | 1.443 |
| 128 | 125 + A3, promedio entre gemelos del test | 156,0 | 127,5 | 879 | 856 | 1.446 |
| 129 | 125 + A2 mezcla + A3 | 155,7 | 127,2 | 878 | 856 | 1.446 |

- **A1** (`02_build_features.add_same_point_structure`, conjunto `pt_core_coords`): para el
  inmueble sin área o sin baños, mediana de los anuncios de train y test en el mismo punto
  (coordenadas a 5 decimales), con igual número de alcobas y tipo. No usa precios. Recupera
  el área del 45,3 % de los inmuebles del test sin área (1.681 de 3.710) y del 34,1 % en
  train; los baños, del 24,7 % en test. En esas filas el error espacial baja de 145,8 a
  132,9 (`views/round7_subsets.csv`).
- **A2, regla de vecino más cercano sobre anuncios repetidos** (`20_twin_rules.py`): si un
  anuncio tiene exactamente la misma descripción (más de 30 caracteres) que anuncios de
  train, su predicción pasa a la mediana del precio de esos gemelos (126) o al promedio
  50/50 con el modelo (127). En el test toca 200 anuncios. En la validación aleatoria simple
  (gemelos sin agrupar) acierta mucho: en sus 11.098 filas con gemelo, de 60,3 a 15,2. Pero
  son casi siempre el mismo inmueble en el mismo sitio. Donde el gemelo está en otra zona,
  como en el test, el reemplazo no ayuda: en la espacial, sus 582 filas pasan de 159,9 a
  160,6. La mezcla sí, a 148,4. En el holdout nororiental (152 filas), de 179,2 a 174,1 con
  la mediana y a 172,4 con la mezcla.
- **A3** (promediar las predicciones de los gemelos del test, 2.993 anuncios): empeora en
  todos los esquemas, por ejemplo de 164,7 a 166,0 en las filas afectadas del nororiental.
- **A4:** cerca de un anuncio de ajuste (a menos de 30 m) el error sin coordenadas es igual
  (nororiental 124,1 frente a 124,7); lejos tampoco cambia. Lo que gana sin coordenadas sale
  en la espacial, en bloques sin vecinos.

**Pieza B: lineales y redes sin coordenadas crudas.**

| Fila | Modelo | Ponderado | Espacial | Media test | Mín. | Máx. | En 300 M | En 1.650 M |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 108 | lineal con splines de lat y lon (Kaggle 180,5) | 166,0 | 164,1 | 860 | 300 | 1.650 | 32 | 62 |
| 130 | lineal sin lat ni lon | 165,5 | 156,3 | 816 | 300 | 1.650 | 80 | 36 |
| 104 | elastic net con splines | 165,8 | 158,5 | 847 | 300 | 1.650 | 57 | 59 |
| 131 | elastic net sin lat ni lon | 165,5 | 156,8 | 817 | 300 | 1.650 | 82 | 34 |
| 78 | red cuantílica por tipo (ronda 3) | 155,3 | 164,0 | 838 | 330 | 2.003 | 0 | 50* |
| 148 | red con estrato, valor del terreno y ridge, sin lat ni lon | 148,2 | 137,6 | 875 | 333 | 2.002 | 0 | 167* |

\* La red no tiene tope: es el número de predicciones de 1.650 M o más.

Sin coordenadas, la lineal gana 7,8 M en la espacial, pero baja su media en test a 816 M y
pega 80 predicciones al piso de 300 M. La red 148 es la mejor fila de la ronda en el
ponderado: 6,6 M mejor que la 83. En la espacial queda 6,1 M peor que la 83, pero 26 M mejor
que la 78. Las redes ya ganaban en el holdout y perdían en la espacial (78, y el patrón del
54 en Kaggle), así que solo Kaggle dirá si el ponderado acierta con ella.

**Pieza C: balanceo hacia el lujo** (filas 132 a 147; `views/round7_deciles.csv`). Cuatro
variantes sobre la 82, cada una con cuantil 0,55 a 0,70. Se escoge el cuantil cuya media en
test queda más cerca de 880 M (la de la 83): es una calibración de nivel contra el test que
no usa sus precios.

| Fila | Variante | Ponderado | Espacial | Media test | MAE 30 % más caro | Sobrepredicho (30 % caro) | Spearman (30 % caro) |
|---|---|---:|---:|---:|---:|---:|---:|
| 82 | sin balanceo, q0,70 | 155,4 | 132,6 | 883 | 307,7 | 6,2 % | 0,248 |
| 134 | pesos 1 a 2, q0,65 | 156,8 | 132,7 | 872 | 315,5 | 5,6 % | 0,233 |
| 138 | pesos 1 a 4, q0,65 | 159,5 | 138,1 | 881 | 312,5 | 5,4 % | 0,232 |
| 141 | duplicado, q0,60 | 156,1 | 127,3 | 872 | 316,9 | 6,0 % | 0,236 |
| 144 | triplicado, q0,55 | 156,9 | 125,0 | 874 | 320,2 | 5,7 % | 0,210 |

El balanceo no mejora el orden de los caros: empeora su MAE y su correlación de rangos. El
sobremuestreo baja la espacial, porque reduce la sobrepredicción de los inmuebles baratos
(decil 1: sesgo de +101 a +79). Es nivel, no orden. En la 134 hubo empate a 8 M entre q0,65
(872) y q0,70 (888); se escogió q0,65 por su menor ponderado.

**Pieza D: la red frente al ridge en el texto** (`22_text_nets.py`, `views/text_nets.csv`).
Cada fuente se evalúa sola, fuera de muestra, como predictor del precio.

| Fuente | Ponderado | Espacial |
|---|---:|---:|
| ridge sobre TF-IDF (el de la 83) | 212,8 | 166,1 |
| red de la ronda 4 sobre embeddings e5 (cuantil 0,70) | 206,8 | 204,5 |
| red densa sobre el mismo TF-IDF | 216,1 | 171,0 |
| red sobre la SVD de 300 componentes del TF-IDF | 225,7 | 175,3 |
| la red TF-IDF sintonizada con NSGA-II (60 evaluaciones) | 213,9 | 165,5† |
| embeddings tipo fastText aprendidos desde cero | no terminó en 30 min | |

† Algo optimista: el genético se sintonizó con filas que la espacial usa para validar.

La red de la ronda 4 (`EmbeddingNet`: capas de 256 y 64, dropout 0,2, tasa 3e-3, sin
sintonizar) ve otra representación (embeddings de frases). El genético prefiere la red más
simple: embeddings de 32, sin capas ocultas, dropout 0,23 y regularización 2e-4. Dentro del
boosting, el ridge con la red sintonizada (149, solo holdout nororiental) da 155,6, frente a
155,4 de la 82. Con el mismo TF-IDF la red empata con el ridge, aunque se le dé más
capacidad. La diferencia entre la red de la ronda 4 y el ridge viene de la representación:
con embeddings la red gana en el nororiente y viaja mucho peor. torch-directml no se instaló:
bajaba torch de 2.15 a 2.4.1 en el entorno.

**Pieza E.** Ningún boosting de A o C supera a la 83 en los dos criterios, así que no se
repitió la mezcla con el bosque 114. Validación aleatoria con gemelos agregada: 83 111,0;
125 110,7; 78 140,2; 120 a 124, 130, 131, 134, 138, 141, 144 y 148 en
`views/round7_summary.csv`. Siguen sin ella las filas 72 a 98 que no son candidatas a envío
y las variantes de la C sin escoger. Los envíos vigentes se revisaron: todos tienen los ids y
el orden de la plantilla, sin faltantes y con precios positivos; no hubo que mover nada a
`obsoletos/`.

**Mezclas de boosting y red** (`06_ensemble.py`, pesos fijos boosting/red):

| Fila | Mezcla | Correlación en test | Ponderado | Espacial | Media test |
|---|---|---:|---:|---:|---:|
| 83 | boosting solo | | 154,8 | 131,5 | 882 |
| 148 | red sola | | 148,2 | 137,6 | 875 |
| 150 | 83 / 148, 70/30 | 0,969 | 151,5 | 131,1 | 879 |
| 151 | 83 / 148, 50/50 | 0,969 | 149,9 | 132,0 | 878 |
| 152 | 127 / 148, 70/30 | 0,962 | 151,7 | 127,6 | 877 |
| 153 | 127 / 148, 50/50 | 0,962 | 150,0 | 129,4 | 876 |

La 152 es la primera fila que supera a la 83 por más de 2 M en los dos criterios (−3,1 y
−3,9). La 153 también, por poco en la espacial (−2,1). Las dos dependen de que el ponderado
acierte con la red 148, que aún no tiene puntaje público.

**Conclusión antes de las mezclas:** nada supera a la 83 por más de 2 M en los dos criterios. La 127 gana 4,5 M en
la espacial y empata en el ponderado (+0,7). La red 148 gana 6,6 M en el ponderado y pierde
6,1 M en la espacial.

### Ronda 8: anatomía del error y revisión de supuestos

**Dónde está el error de la 83** (`23_error_anatomy.py`, `views/error_anatomy_*`). Aporte al
error absoluto total, ponderado / espacial:

| Segmento | Filas (peso) | Aporte | MAE | Sesgo |
|---|---:|---:|---:|---:|
| Decil 10 del precio observado | 10 % / 10 % | 35 % / 18 % | 562 / 246 | −562 / −215 |
| Casas | 13 % / 25 % | 22 % / 34 % | 268 / 183 | −154 / +46 |
| 200 m² o más | 12 % / 13 % | 25 % / 19 % | 332 / 193 | −260 / +42 |

La 127 repite el mismo patrón. Los 50 peores del ponderado pasan todos de 1.000 M, el tope
del ajuste del holdout. 22 de ellos son anuncios de Rentahouse, muchos con la descripción
repetida.

**Pruebas sobre la receta de la 83** (`views/round8_summary.csv`; las de una semilla se
comparan también con la 82). Diferencias en millones; menos de 2 M es ruido:

| Prueba | Fila | Ponderado | Espacial | Δ ponderado | Δ espacial |
|---|---|---:|---:|---:|---:|
| 83, referencia | 83 | 154,8 | 131,5 | | |
| 82, una semilla | 82 | 155,4 | 132,6 | +0,6 | +1,1 |
| a. cuantil 0,60 / 0,65 | 154 / 155 | 155,5 / 154,9 | 126,1 / 126,7 | +0,6 / +0,1 | −5,4 / −4,8 |
| a. cuantil 0,75 / 0,80 | 156 / 157 | 157,1 / 160,4 | 138,7 / 147,6 | +2,2 / +5,5 | +7,3 / +16,1 |
| a. cuantil por tercil de precio predicho | | 153,7 | 126,1 | −1,1 | −5,4 |
| a. cuantil por estrato | | 157,1 | 125,9 | +2,3 | −5,6 |
| b. apartamentos de estrato 5 y 6 del nororiente | 158 | 163,3 | 160,6 | +8,4 | +29,1 |
| b. radio de 3 / 5 / 8 km | 159–161 | 167,2 / 159,1 / 156,4 | 191,7 / 154,4 / 140,4 | +12,4 / +4,3 / +1,5 | +60,3 / +22,9 / +9,0 |
| c. sin gemelos / gemelos a peso 1/n | 162 / 163 | 156,2 / 155,6 | 132,7 / 131,7 | +1,4 / +0,7 | +1,2 / +0,2 |
| d. sin precio por m² fuera de p1–p99 | 164 | 157,0 | 133,0 | +2,2 | +1,5 |
| e. precio por m² con el área del mismo punto | 165 | 157,4 | 126,7 | +2,6 | −4,8 |
| f. mezcla 70/30, 83 y red 148 | 150 | 151,5 | 131,1 | −3,4 | −0,4 |
| f. apilamiento lineal, ajustado en la espacial | 166 | 169,9 | 119,5* | +15,1 | −11,9* |
| f. apilamiento LightGBM, ajustado en la espacial | 167 | 178,0 | 120,2* | +23,2 | −11,3* |

\* Espacial del apilamiento escogido con cuatro particiones espaciales y aplicado a la quinta.
Es algo optimista: sus entradas son predicciones espaciales de modelos que sí vieron los
precios de las otras particiones. Por eso no se guarda como `mae_spatial` de 166 y 167.

- **a. Nivel:** la espacial pide el cuantil 0,60 en casi todos los segmentos, y 0,65 en el
  estrato 6. El ponderado pide de 0,60 a 0,80, más alto en el tercil caro. Los dos criterios no se ponen de acuerdo sobre
  el nivel, y escoger el cuantil por segmento no resuelve ese desacuerdo.
- **b, c, d:** entrenar con toda Bogotá, con los gemelos y sin limpiar los extremos se
  sostiene.
- **e:** el precio por m² con el área recuperada baja la espacial porque baja el nivel
  (sesgo de +30 frente a +43), no porque ordene mejor.
- **f, apilamiento:** si el segundo nivel se ajusta dentro del holdout ponderado, gana entre
  9 y 15 M. Pero aprende a corregir el tope de 1.000 M del holdout (intercepto de −132 M), y
  eso no se puede llevar al test (los interceptos por zona van de −111 a −143 M,
  `views/round8_stacking.csv`). Ajustado en la espacial, que es lo único que se puede
  enviar, baja el nivel del test a 788 M y pierde 15 M en el ponderado. Los envíos 166 y 167
  quedan en `obsoletos/` y no se recomiendan.
- **g. Tope:** 88 predicciones de la 83 en el test quedan a menos de 5 % de 1.650 M (66 de
  la 127, 138 de la 152). En el holdout, el 1,7 % de las filas validadas tiene un precio de
  1.567 M o más.

### Ronda 9: redes para la mezcla

Criterio: la validación espacial, que acertó con la red 148 (Kaggle 157,1 frente a 137,6
espacial y 148,2 ponderado; el ponderado fue optimista en 9 M con ella). Cifras de
`views/round9_nets.csv` (`27_net_table.py`). Ninguna fila tiene la validación aleatoria.

| Fila | Red | Espacial | Ponderado | Media test | Corr. 148 | Corr. 127 |
|---|---|---:|---:|---:|---:|---:|
| 148 | referencia (5 semillas) | 137,6 | 148,2 | 875 | 1,000 | 0,962 |
| 168 | 1a. 10 semillas | 137,3 | 148,0 | 877 | 0,999 | 0,963 |
| 169 | 1b. área y baños del mismo punto | 136,7 | 146,1 | 872 | 0,991 | 0,966 |
| 171 | 1c. 169 sintonizada, 5 semillas | 132,4† | 150,7 | 858 | 0,987 | 0,968 |
| 170 | 2. una capa de 512, red de texto en lugar del ridge | 143,6 | 152,8 | 868 | 0,960 | 0,937 |

† La sintonización (20 evaluaciones con una semilla, Optuna TPE) escogió el mínimo de esas
mismas particiones espaciales: la cifra es optimista. Ganó una red de tres capas (512, 256 y
128), dropout 0,24, tasa 0,0094 y regularización 7e-5. Parte de la mejora es nivel: el
sesgo espacial baja de +59 a +44 y la media en test, de 875 a 858.

Mezclas con pesos fijos (`06_ensemble.py`), boosting 127 con la 171 y la 170:

| Fila | Pesos 127 / 171 / 170 | Espacial | Ponderado | Media test | Corr. entre componentes | Δ esp. frente a 152 |
|---|---|---:|---:|---:|---:|---:|
| 152 | 70 / 148 al 30 | 127,6 | 151,7 | 877 | 0,962 | |
| 172 | 70 / 30 / – | 126,6 | 152,7 | 872 | 0,968 | −1,0 |
| 173 | 60 / 40 / – | 126,9 | 152,0 | 870 | 0,968 | −0,8 |
| 174 | 50 / 50 / – | 127,4 | 151,5 | 868 | 0,968 | −0,3 |
| 175 | 60 / 20 / 20 | 127,1 | 151,5 | 872 | 0,953 | −0,6 |
| 176 | 50 / 25 / 25 | 127,8 | 150,9 | 870 | 0,953 | +0,1 |

Ninguna mezcla supera a la 152 en la espacial por más de 2 M. La red sintonizada mejora
sola, pero se parece más al boosting (correlación de 0,968 frente a 0,962) y la mezcla
gana poco. La segunda red sí es distinta (0,937 con el 127), pero es peor sola, y en la
mezcla a tres no mejora lo que logra la 172.

### Ronda 10: monotonía, hojas lineales y kNN

Criterio: validación espacial y después el ponderado; menos de 2 M es ruido. Cifras de
`views/round10_table.csv` (`29_round10_table.py`); ninguna fila tiene la validación
aleatoria. Las filas 181 a 184 son la 177 a la 180 con la regla `mix` de gemelos, como la
127 lo es de la 125.

| Fila | Cambio sobre la receta del 127 | Espacial | Ponderado | Media test | Máx. test | > 1.650 M |
|---|---|---:|---:|---:|---:|---:|
| 127 | referencia | 127,0 | 155,5 | 878 | 1.730 | 7 |
| 181 | pinball propia, sin restricciones (control) | 123,7 | 155,8 | 862 | 1.747 | 10 |
| 182 | pinball, monótona en área, baños, parqueaderos, estrato y terreno | 127,3 | 152,6 | 871 | 1.878 | 36 |
| 183 | la 182 también monótona en el ridge del texto | 132,9 | 153,1 | 878 | 1.921 | 107 |
| 184 | pinball con hojas lineales (lambda 100, 5 semillas) | 123,9 | 155,5 | 859 | 1.783 | 8 |
| 185 | kNN por tipo, percentil 70 de 40 vecinos† | 169,5† | 178,0 | 789 | 1.550 | 0 |

- **Por qué una pinball propia:** LightGBM no admite restricciones monótonas con su pérdida
  cuantílica. La 181 aísla el cambio de pérdida: baja la espacial 3,3 M, pero casi todo es
  nivel (sesgo espacial de +35 a +15, media en test de 878 a 862, sesgo nororiental de −62 a
  −70), y el ponderado no se mueve.
- **Monotonía:** frente a su control, gana 3,3 M en el ponderado y pierde 3,6 en la
  espacial. Con el texto también restringido pierde 9 M en la espacial.
- **Hojas lineales:** con la pérdida cuantílica de LightGBM no sirven (209 M en el
  nororiente con cualquier lambda y precios negativos); con la pinball propia empatan con el
  control. Ninguna predicción del test es absurda (mínimo 285 M), y no se recortó nada.
- **kNN:** † rejilla de k y pesos con la espacial como objetivo (optimista), y el óptimo
  quedó en el borde (k = 40, peso de ubicación 0,1). Su correlación en test es de 0,854 con
  la 181, 0,857 con la 127 y 0,848 con la red 148.

Mezclas con pesos fijos, contra la 152 (127 al 70 % y red 148 al 30 %):

| Fila | Pesos 181 / 148 / 185 | Espacial | Ponderado | Media test | Δ esp. 152 | Δ pond. 152 |
|---|---|---:|---:|---:|---:|---:|
| 152 | (127) 70 / 30 / – | 127,6 | 151,7 | 877 | | |
| 186 | 70 / 30 / – | 123,9 | 151,5 | 866 | −3,7 | −0,3 |
| 187 | 60 / 25 / 15 | 125,7 | 153,0 | 854 | −1,9 | +1,3 |
| 188 | 65 / 25 / 10 | 124,7 | 152,6 | 858 | −2,9 | +0,8 |

La 186 supera a la 152 en la espacial por más de 2 M y empata en el ponderado, pero la
mejora viene de bajar el nivel, como la red 171 en la ronda 9. El kNN no aporta a la mezcla.

**Sondeo, no es un modelo:** `189_probe_ensemble_w70_30_2_top5_x115` multiplica por 1,15 el
5 % más alto de las predicciones de la 152 (515 pasan de 1.650 M). Baja el ponderado de
151,7 a 146,6 y sube la espacial de 127,6 a 132,7: sirve para medir en Kaggle si el tramo
caro está subpredicho.

### Ronda 11: el texto con BETO

`txt_beto` es el log del precio que predice BETO afinado con el título y la descripción,
fuera de muestra con las particiones espaciales (los anuncios con la misma descripción,
juntos). Solo, da 148,5 M de MAE frente a 169,1 M del ridge en las mismas filas y
particiones; la correlación entre ambos, en log, es 0,83 (`views/beto_check.csv`). BETO
corrige mejor los extremos: en el decil más caro su error es 341 M y el del ridge 537 M
(`views/beto_deciles.csv`).

Cifras en millones, de `views/beto_table.csv` (`39_beto_table.py`):

| Fila | Qué es | Espacial | Ponderado | Media test | Máx. test | Δ esp. 152 | Δ pond. 152 |
|---|---|---:|---:|---:|---:|---:|---:|
| 127 | boosting, solo ridge (referencia) | 127,0 | 155,5 | 878 | 1.730 | −0,7 | +3,8 |
| 197 | boosting, BETO en lugar del ridge | 125,1 | 155,8 | 882 | 1.686 | −2,6 | +4,1 |
| 198 | boosting, BETO y ridge | 124,4 | 154,5 | 884 | 1.685 | −3,2 | +2,8 |
| 201 | la 198 con cuantil 0,65 | 120,7 | 153,9 | 863 | 1.701 | −6,9 | +2,2 |
| 202 | la 198 con cuantil 0,75 | 129,6 | 155,7 | 905 | 1.690 | +2,0 | +4,0 |
| 148 | red, solo ridge (referencia) | 137,6 | 148,2 | 875 | 2.002 | +10,0 | −3,5 |
| 195 | red, BETO en lugar del ridge | 135,0 | 147,6 | 887 | 2.059 | +7,4 | −4,1 |
| 196 | red, BETO y ridge | 133,7 | 146,3 | 875 | 1.981 | +6,0 | −5,5 |
| 152 | 127 / 148, 70/30 | 127,6 | 151,7 | 877 | 1.772 | | |
| 203 | 198 / 196, 70/30 | 124,8 | 150,5 | 881 | 1.745 | −2,8 | −1,2 |
| 204 | 198 / 196, 60/40 | 125,5 | 149,4 | 881 | 1.778 | −2,2 | −2,3 |
| 205 | 198 / 148, 70/30 | 125,5 | 150,9 | 881 | 1.753 | −2,1 | −0,8 |

Las 197, 198, 201 y 202 llevan la regla `mix` de gemelos, como la 127. El cuantil 0,65 baja
la espacial porque baja el nivel; interpolando entre 0,65 y 0,70 hasta la media de la 127
(878 M), la espacial de la receta con BETO queda cerca de 123,3. La correlación en test entre
los miembros de la 203 es 0,962, la misma que entre los de la 152: BETO mejora a los dos
modelos sin hacerlos más distintos. Recortar al rango del train (300 a 1.650 M) no cambia la
validación en más de 0,4 M; las 206 a 208 son la 203, la 204 y la 198 recortadas
(`40_clip_range.py`).

El `txt_beto` de cada fila viene de un modelo que no vio su precio ni el de sus anuncios de
descripción idéntica. 155 filas (37 grupos) tienen gemelos físicos con otra descripción
(`utils.twin_groups`) en otra partición de BETO.

- **Espacial:** las filas de ajuste traen un `txt_beto` de modelos que vieron precios de la
  validación. Es la limitación habitual de apilar con predicciones fuera de muestra fijas,
  de segundo orden. Además, las 361 filas que se movieron con sus gemelos de descripción
  hacen que algunos modelos de BETO hayan visto precios de la zona que predicen.
- **Holdout nororiental:** la limitación es de primer orden. Sus 6.860 filas caen casi todas
  en las particiones 0 y 2 de BETO, así que el `txt_beto` de cada fila de validación salió de
  un modelo entrenado con los precios de otras filas de esa misma validación, incluidas las
  de más de 1.000 M que el holdout deja fuera del ajuste.

Por eso la decisión se toma con la espacial y el ponderado solo se reporta.

### Diversidad de errores fuera de muestra (scripts 43 a 45)

Análisis de Neislen, hecho en paralelo a la ronda 8 (22 de noviembre). Se fusionó a la rama
principal el 25 de noviembre. Sus scripts y envíos se renumeraron para no chocar con los que
ya existían:
- los scripts pasaron de 23–25 a 43–45;
- los envíos pasaron de 154–161 a 209–216, porque las filas 154 a 161 de `cv_results.csv`
  son otros modelos.

No amplía la búsqueda de hiperparámetros. Reconstruye las predicciones fuera de muestra de
cinco modelos (03, 45, 54, 65 y 77) en cuatro esquemas, mide qué tan distintos son sus
errores y define mezclas con pesos fijados antes de mirar el tablero.

Métricas reconstruidas, en millones de pesos (`views/oof_model_metrics.csv`):

| Modelo | Aleatoria, repetidos juntos | Espacial | Tipo Chapinero | Nororiental |
|---|---:|---:|---:|---:|
| 03 lineal | 156,1 | 168,1 | 183,3 | 190,0 |
| 45 boosting | **108,4** | **135,9** | 174,1 | 168,5 |
| 54 híbrido q0,65 | 116,4 | 142,8 | 157,9 | **157,8** |
| 65 boosting q0,70 | 118,5 | 148,9 | 159,0 | 162,9 |
| 77 boosting externo q0,70 | 117,7 | 146,0 | **151,2** | 162,7 |

**Lectura:**
- **Ningún modelo gana en todos los esquemas.**
- **Lineal 03:** es el que más difiere de los árboles. La correlación de sus errores con el 77
  es 0,657 con los repetidos juntos y 0,683 en la espacial (`views/oof_error_correlations.csv`).
- **Mezclas fuera de muestra** (`views/oof_ensemble_candidates.csv`): `E05_45_77_03_w70_20_10`
  baja el MAE espacial del 45 de 135,9 a 134,1 M. En los holdouts siguen ganando los modelos
  solos.

**Envíos de `45_new_submissions.py`** (trazabilidad en `views/new_submissions_manifest.csv`;
no tienen puntaje registrado en `views/kaggle_scores.csv`):

| Envío | Mezcla |
|---:|---|
| 209 | 80 % de la 83 y 20 % de la 54 |
| 210 | 60 % de la 83, 20 % de la 54 y 20 % de la 45 |
| 211 | 70 % de la 83, 20 % de la 54 y 10 % de la 03 |
| 212 | 70 % de la 83, 15 % de la 54 y 15 % de la 77 |
| 213 | 60 % de la 83, 20 % de la 127 y 20 % de la red 148 |
| 214 | promedio simple de la 83, la 127 y la 148 |
| 215 | mediana de la 83, la 127 y la 148 |
| 216 | 70 % de la 83, 15 % del bosque 114 y 15 % de la red 148 |

### Calibración de nivel (no es un modelo)

Multiplicar un envío por un factor fijo no es un modelo: es calibrar con el leaderboard.
Estos envíos están en `views/level_calibration.csv` y no en la tabla principal.

| Envío | Base | Factor | Media en test | Holdout nororiental |
|---|---|---:|---:|---:|
| `32_boosting_full_coords_x115` | 10 (sin OSM) | 1,15 | 878 | 161.7 (Kaggle 171.7) |
| `60_boosting_full_coords_x120` | 10 (sin OSM) | 1,20 | 916 | sin calcular |
| `57_..._q65_x110` | 54 | 1,10 | 926 | 163.7 |
| `58_..._q65_x115` | 54 | 1,15 | 968 | 174.0 |
| `59_..._q65_x120` | 54 | 1,20 | 1.010 | 188.2 |

El holdout dice que el 54 ya está cerca del nivel óptimo (sesgo −52 M): escalarlo empeora.

