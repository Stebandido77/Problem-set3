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
│   ├── 05_tune_models.py      búsqueda aleatoria de hiperparámetros con CV espacial
│   ├── 06_ensemble.py         promedio o mediana de especificaciones ya entrenadas
│   └── 07_results_table.py    tabla de resultados en Markdown para este README
├── stores/
│   ├── raw/                   datos de Kaggle (train, test, plantilla de envío)
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
python 04_train_models.py                 # todas las especificaciones (~75 min en 16 núcleos)
python 04_train_models.py --specs 17 18   # solo algunos números de SPECS
python 05_tune_models.py --family cart --features full --n-iter 15 --n-jobs 4
python 06_ensemble.py --number 31 --how mean --models \
    17_boosting_full_coords_by_type_per_m2 04_elastic_net_full 18_random_forest_full_by_type_per_m2
python 07_results_table.py                # tabla de resultados para el README
```

`05_tune_models.py` guarda los mejores hiperparámetros en `stores/processed/best_params.json`;
`04_train_models.py` los usa solo en las especificaciones marcadas `tuned=True`. Un envío
ya escrito nunca se sobrescribe: si su contenido cambia, se guarda como `_v2`, `_v3`...

## Datos y predictores

| Bloque | Predictores | Fuente |
|---|---|---|
| Estructurales | tipo de inmueble, alcobas, baños, cuartos, área (y su logaritmo), indicador de área faltante, tendencia mensual | Properati |
| Texto (18) | parqueadero y número de parqueaderos, terraza, balcón, ascensor, depósito, chimenea, vigilancia, gimnasio, piscina, conjunto cerrado, remodelado, dúplex, penthouse, vista, estrato, piso, longitud de la descripción | título y descripción |
| Puntos de referencia (4) | distancia a Zona T, Parque de la 93, Centro Internacional y Plaza de Bolívar | coordenadas aproximadas |
| OpenStreetMap (16) | distancia al más cercano de 11 categorías (parque, estación de transporte, universidad, colegio, hospital, centro comercial, restaurante, bar, banco, supermercado, policía) y conteos a 500 m de 5 de ellas | Overpass API |
| Solo en `rich` (24) | 8 de texto (cuarto de servicio, salón de conductores, vestier, estudio, lujo, spa, exterior, antigüedad), distancia a la Carrera Séptima y a los cerros (trazados aproximados), conteos de OSM a 300 m y 1 km | texto, OSM |

El área falta en cerca del 80 % de los anuncios; se recupera de la descripción cuando aparece
(queda faltante en 39 % de train y 36 % de test). Baños faltantes también se completan con el texto.

Conjuntos de predictores usados en los modelos: `base` (estructurales), `base_text`,
`full` (todo menos lat/lon crudos, para que el modelo aprenda de distancias y no de
coordenadas que en Chapinero casi no vio), `full_coords` (agrega lat/lon) y `rich` /
`rich_coords` (agregan la segunda tanda; no mejoraron y solo se usan en las filas 19 y 20).

## Validación

- **Aleatoria:** 5 particiones al azar.
- **Espacial:** k-means sobre lat/lon define 10 zonas; se dejan zonas completas por fuera
  (GroupKFold de 5). Imita el salto del resto de Bogotá a Chapinero.
- **Holdout tipo Chapinero:** se valida en los inmuebles de train a menos de 250 m de algún
  inmueble del test y se entrena con el resto, menos una franja de 400 m alrededor. Radio y
  franja se escogieron para que el mejor envío ya subido (boosting con coordenadas, sin OSM)
  diera un MAE cercano a su puntaje público de 180 M: da 182 M, frente a 137 M de la
  validación espacial. Es el criterio principal para escoger modelos.

Por defecto los modelos se entrenan sobre log(precio) y predicen en pesos; las variantes
`_level` y `_per_m2` usan el precio en nivel y log(precio por m2) cuando hay área.
`_by_type` ajusta un modelo para apartamentos y otro para casas.

## Resultados actuales

Salen de `views/cv_results.csv` (tabla generada con `07_results_table.py`). Todas las filas
usan los 16 predictores de OSM. La tabla anterior, sin OSM, que corresponde a los envíos
01–11 ya subidos a Kaggle, está en `views/cv_results_pre_osm.csv`.

Cifras en millones de pesos.

| Modelo | Predictores | MAE aleatoria | MAE espacial | MAE tipo Chapinero | Sesgo tipo Chapinero | % sobrepredicho (Chapinero) |
|---|---:|---:|---:|---:|---:|---:|
| `01_linear_base` | 8 | 187.4 | 202.2 | 234.8 | -183.0 | 26% |
| `02_linear_base_text` | 26 | 173.3 | 186.0 | 211.3 | -156.5 | 30% |
| `03_linear_full` | 46 | 155.8 | 168.1 | 183.3 | -55.6 | 47% |
| `04_elastic_net_full` | 46 | 132.8 | 159.7 | 172.0 | -61.5 | 42% |
| `05_cart_full` | 46 | 130.6 | 169.9 | 195.6 | -103.6 | 36% |
| `06_random_forest_full` | 46 | 93.9 | 147.6 | 180.2 | -110.4 | 35% |
| `07_random_forest_full_coords` | 48 | 93.0 | 146.6 | 181.2 | -111.6 | 35% |
| `08_boosting_base_text` | 26 | 124.4 | 150.7 | 181.2 | -138.4 | 24% |
| `09_boosting_full` | 46 | 96.5 | 145.8 | 182.3 | -120.8 | 30% |
| `10_boosting_full_coords` | 48 | 95.6 | 145.4 | 179.2 | -115.8 | 31% |
| `11_neural_net_full` | 46 | 146.8 | 196.6 | 237.2 | +56.6 | 57% |
| `12_boosting_full_coords_weighted` | 48 | 112.7 | 150.1 | 176.5 | -117.3 | 31% |
| `13_random_forest_full_weighted` | 46 | 107.1 | 151.2 | 179.2 | -108.6 | 36% |
| `14_boosting_full_coords_by_type` | 48 | 90.5 | 142.7 | 177.6 | -114.2 | 31% |
| `15_random_forest_full_by_type` | 46 | 91.5 | 146.8 | 177.5 | -106.5 | 35% |
| `16_boosting_full_coords_by_type_level` | 48 | 90.8 | 144.5 | 171.2 | -97.3 | 34% |
| `17_boosting_full_coords_by_type_per_m2` | 48 | 90.7 | 142.6 | 169.9 | -102.4 | 32% |
| `18_random_forest_full_by_type_per_m2` | 46 | 91.7 | 146.8 | 172.2 | -98.9 | 37% |
| `19_boosting_rich_coords_by_type_per_m2` | 72 | 89.4 | 143.9 | 174.2 | -110.4 | 31% |
| `20_random_forest_rich_by_type_per_m2` | 70 | 89.6 | 150.5 | 177.5 | -110.2 | 33% |
| `21_cart_full_tuned` | 46 | 137.6 | 165.9 | 193.2 | -116.1 | 34% |
| `22_neural_net_full_tuned` | 46 | 129.8 | 170.6 | 212.0 | -62.2 | 38% |
| `23_boosting_full_coords_by_type_per_m2_tuned` | 48 | 95.7 | 143.8 | 174.0 | -108.2 | 31% |
| `24_random_forest_full_by_type_per_m2_tuned` | 46 | 96.0 | 147.2 | 172.7 | -102.3 | 35% |
| `30_ensemble_median_3` | – | 93.0 | 142.2 | 167.3 | -94.6 | 35% |
| `31_ensemble_mean_3` | – | 99.9 | 142.9 | 164.6 | -87.6 | 36% |

Modelos ingenuos (`views/cv_results_naive.csv`): la mediana constante da 236.7 / 248.9 / 298.1
y la mediana por tipo y alcobas 217.7 / 242.6 / 289.2 (aleatoria / espacial / tipo Chapinero).

Lectura:

- La validación aleatoria es muy optimista y la espacial también se queda corta: el holdout
  tipo Chapinero da entre 12 y 35 M más que la espacial según el modelo.
- El ranking cambia: en el holdout el elastic net (172.0) le gana al boosting y al bosque
  iniciales (179–181), aunque en la espacial pierde. Ningún esquema basta por sí solo.
- Lo que ayudó: un modelo aparte para apartamentos y el precio por m2 como objetivo
  (boosting de 179.2 a 169.9) y el promedio de boosting, elastic net y bosque (164.6).
- Lo que no: ponderar por parecido al test, la segunda tanda de predictores y sintonizar
  boosting y bosque sobre log(precio).
- Advertencias: (i) el MAE espacial de las filas `_tuned` es optimista, porque 05 sintoniza con
  esas mismas particiones; (ii) las filas 23 y 24 usan parámetros buscados sobre log(precio),
  no sobre precio por m2; (iii) los componentes de los ensambles y el radio y la franja del
  holdout se escogieron mirando el holdout y el leaderboard, así que su MAE también es algo
  optimista; (iv) las filas 12 y 13 ponderan con un clasificador train-contra-test que usa las
  covariables del test (nunca el precio), una excepción a la regla de no usar el test;
  (v) las redes neuronales (11, 22) dan predicciones de hasta 5.900 M, muy por encima del
  máximo de train (1.650 M).
- En el holdout todos los modelos subpredicen (sesgo de −55 a −180 M; solo un tercio de los
  inmuebles queda sobrepredicho). Para comprar, eso es el error barato.

## Pendientes

- [x] Correr `01_download_osm.py` y regenerar predictores, modelos y envíos.
- [ ] Sintonizar el elastic net (no cupo en 40 minutos con 8 iteraciones) y registrar el
  puntaje público de Kaggle por envío en `views/kaggle_scores.csv`.
- [ ] Comparar CV aleatoria, CV espacial y leaderboard público (Etapa 3).
- [ ] Análisis de errores asimétricos y regla de compra (Etapa 4), a partir de `oof_predictions.csv`.
- [ ] Figuras y tablas para las dos presentaciones.
