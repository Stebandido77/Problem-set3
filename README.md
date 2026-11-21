# Problem Set 3 — Making Money with ML?

**MECA 4107 · Big Data and Machine Learning para Economía Aplicada · Universidad de los Andes · 2026-20**

Equipo XX: _nombres de los integrantes_

Predicción de precios de vivienda en Chapinero (Bogotá) con datos de Properati. El modelo se
entrena con inmuebles del resto de la ciudad y debe viajar a una zona que casi no vio. La
métrica de Kaggle es el error absoluto medio (MAE), pero la pregunta de negocio es otra:
¿se puede confiar en el modelo lo suficiente para comprar, sabiendo que sobrepredecir
destruye capital?

## Estructura del repositorio

```
├── document/              presentaciones finales (best_equipo_XX.pdf, compare_equipo_XX.pdf)
├── scripts/
│   ├── utils.py               rutas, validación cruzada aleatoria y espacial, métricas, envíos
│   ├── models.py              una receta por familia de algoritmos + espacios de búsqueda
│   ├── 01_download_osm.py     descarga puntos de interés de OpenStreetMap (Overpass)
│   ├── 02_build_features.py   limpieza, predictores de texto y espaciales
│   ├── 03_naive_baselines.py  modelos ingenuos (mediana, media, mediana por celda)
│   ├── 04_train_models.py     seis familias, CV aleatoria, espacial y holdout tipo Chapinero, envíos
│   ├── 05_tune_models.py      búsqueda aleatoria con CV espacial o en el holdout ponderado
│   ├── 06_ensemble.py         promedio, mediana o pesos fijos de especificaciones ya entrenadas
│   ├── 07_results_table.py    tabla de resultados en Markdown y error por decil de precio
│   ├── 08_calibrate_holdout.py  compara esquemas de holdout contra los puntajes de Kaggle
│   ├── 09_level_calibration.py  versiones escaladas de un envío (x1,10, x1,20...)
│   ├── 10_same_building.py    error con y sin anuncios del mismo edificio (ronda 3a)
│   ├── 11_download_bogota.py  estrato y valor catastral del terreno por manzana (datos abiertos)
│   ├── 12_text_report.py      términos y componentes del texto que más pesan
│   ├── 13_text_embeddings.py  embeddings de frases del título y la descripción (ronda 4)
│   ├── 14_tune_text_ridge.py  alfa, min_df y n-gramas del ridge de texto (ronda 4)
│   ├── 15_optuna_search.py    Optuna TPE contra CMA-ES para el LightGBM (ronda 4)
│   ├── 16_download_lots.py    lote, pisos y área construida del catastro (ronda 4)
│   ├── 17_tune_families.py    Optuna TPE para bosque, elastic net y CART cuantílicos (ronda 5)
│   ├── 18_family_table.py     tabla por familia: versión anterior, nueva y boosting 83 (ronda 5)
│   └── 19_genetic_search.py   NSGA-II (genético) para el bosque cuantílico (ronda 6)
├── stores/
│   ├── raw/                   datos de Kaggle (train, test, plantilla de envío)
│   ├── external/              caché de descargas de Bogotá (no se versiona)
│   ├── processed/             bases con predictores, predicciones fuera de muestra
│   └── submissions/           un .csv por envío
├── views/                 tablas y figuras
├── requirements.txt
└── run_all.sh             corre todo el flujo de principio a fin
```

## Cómo reproducir

```bash
pip install -r requirements.txt
bash run_all.sh
```

O paso a paso desde `scripts/`:

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
python 07_results_table.py                # tabla de resultados para el README
python 08_calibrate_holdout.py            # opcional: ~5 min por esquema candidato
python 09_level_calibration.py --number 57 --factors 1.10 1.15 1.20 \
    --submission 54_en_boosting_core_coords_residual_q65
```

`05_tune_models.py` guarda los mejores hiperparámetros en `stores/processed/best_params.json`;
`04_train_models.py` los usa solo en las especificaciones marcadas `tuned=True`. Un envío
ya escrito nunca se sobrescribe: si su contenido cambia, se guarda como `_v2`, `_v3`...
Los envíos reemplazados por un `_v2`, y las filas 46 a 48 y 52, que tienen un error conocido,
están en `stores/submissions/obsoletos/`.

Estos archivos no se versionan y se regeneran con los scripts:

- `stores/processed/oof_predictions.csv`, con las predicciones fuera de muestra y del
  holdout de cada especificación. Lo escribe `04_train_models.py`: cada corrida agrega o
  reemplaza las columnas de las especificaciones y los esquemas que corre.
- `stores/processed/text_embeddings.npz`, con los embeddings de título y descripción. Lo
  escribe `13_text_embeddings.py` (unos 20 minutos en CPU; descarga
  `intfloat/multilingual-e5-small` de Hugging Face la primera vez).
- `stores/processed/text_cache/`, la caché de los predictores de texto ya ajustados
  (`models.fit_text_sources`). Su clave incluye las filas, los precios, las fuentes y una
  huella del código de esas clases; se puede borrar sin perder nada.
- `stores/processed/optuna_round5.db` y `optuna_round6.db`, los estudios de Optuna de
  `17_tune_families.py` y `19_genetic_search.py`. Permiten cortar una búsqueda y seguirla;
  si se borran, la búsqueda empieza de cero.

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
| `12_boosting_full_coords_weighted` | 48 | 112.7 | 150.1 | 176.5 | 181.2 | 179.1 | -121.0 | 36% |
| `13_random_forest_full_weighted` | 46 | 107.1 | 151.2 | 179.2 | 193.9 | 191.8 | -137.8 | 34% |
| `14_boosting_full_coords_by_type` | 48 | 90.5 | 142.7 | 177.6 | 175.6 | 174.1 | -116.5 | 35% |
| `15_random_forest_full_by_type` | 46 | 91.5 | 146.8 | 177.5 | 193.7 | 192.0 | -137.7 | 34% |
| `16_boosting_full_coords_by_type_level` | 48 | 90.8 | 144.5 | 171.2 | 173.0 | 171.5 | -109.8 | 36% |
| `17_boosting_full_coords_by_type_per_m2` | 48 | 90.7 | 142.6 | 169.9 | 174.9 | 173.3 | -116.2 | 35% |
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

**Conclusión:** nada supera a la 83 por más de 2 M en los dos criterios. La 127 gana 4,5 M en
la espacial y empata en el ponderado (+0,7). La red 148 gana 6,6 M en el ponderado y pierde
6,1 M en la espacial.

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

## Pendientes

- [x] Correr `01_download_osm.py` y regenerar predictores, modelos y envíos.
- [x] Registrar el puntaje público de Kaggle por envío en `views/kaggle_scores.csv` (se
  actualiza con cada envío nuevo).
- [x] Sintonizar la receta del 65 y el elastic net en el holdout ponderado
  (`05_tune_models.py --spec`; trazas en `views/tuning_*_holdout.csv`). El elastic net
  sintonizado no alcanza a reentrenarse en 40 minutos (fila 81 sin resultado).
- [ ] Comparar CV aleatoria, CV espacial y leaderboard público (Etapa 3); el material está en
  `views/holdout_calibration.csv`.
- [ ] Análisis de errores asimétricos y regla de compra (Etapa 4), a partir de
  `oof_predictions.csv` y `views/northeast_deciles.csv`.
- [ ] Figuras y tablas para las dos presentaciones.
