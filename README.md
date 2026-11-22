# Problem Set 3 — Making Money with ML?

**MECA 4107 · Big Data and Machine Learning para Economía Aplicada · Universidad de los Andes · 2026-20**

Equipo 06: _nombres de los integrantes_

## Qué responde el proyecto

Una empresa quiere comprar vivienda en Chapinero (Bogotá) usando un modelo que prediga el
precio de cada anuncio. Entrenamos con anuncios de Properati del resto de la ciudad, así que
el modelo tiene que viajar a una zona que casi no vio, más cara y distinta. Kaggle evalúa con
el error absoluto medio (MAE), pero la pregunta de negocio es otra: **¿se puede confiar en el
modelo lo suficiente para comprar, sabiendo que sobrepredecir cuesta más que subpredecir?**

## Resultados principales

Cifras en millones de pesos. Salen de `views/family_table.csv`, `views/validation_schemes.csv`
y `views/kaggle_scores.csv`.

**Mejor modelo: `152_ensemble_w70_30_2`**, la mezcla del boosting 127 (70 %) y la red
neuronal 148 (30 %), con **148,2 M en el tablero público** (148.209.395), frente a 377,1 M del
ingenuo (mediana constante) y 152,1 M del mejor boosting solo (83). Es el envío con el menor
puntaje público de los 17 registrados.

- **Boosting 127:** LightGBM con pérdida cuantílica en 0,70 sobre el precio en nivel, un
  modelo para apartamentos y otro para casas, con estrato y valor del terreno de la manzana,
  el texto completo del anuncio resumido en un ridge fuera de muestra, el área y los baños de
  los anuncios del mismo punto, sin latitud ni longitud crudas, 10 semillas y una regla que
  mezcla la predicción con el precio de anuncios de train con la misma descripción.
- **Red 148:** red cuantílica (0,70) por tipo con los mismos predictores de la manzana y el
  ridge del texto, sin coordenadas. Es la única familia que predice por encima del tope de
  precios del train (1.650 M), y por eso mejora la mezcla.

Una fila por familia (su mejor especificación con la receta común):

| Familia | Fila | Búsqueda de hiperparámetros | Aleatoria con gemelos | Espacial | Ponderado | Kaggle |
|---|---:|---|---:|---:|---:|---:|
| Regresión lineal | 108 | no aplica (mínimos cuadrados) | 144,1 | 164,1 | 166,0 | 180,5 |
| Elastic net | 104 | Optuna TPE, 15 evaluaciones | 144,5 | 158,5 | 165,8 | sin envío |
| CART | 105 | Optuna TPE, 25 evaluaciones | 141,3 | 159,8 | 168,7 | 177,5 |
| Random forest | 114 | Optuna TPE, 25; NSGA-II, 120 (no mejora) | 123,1 | 143,8 | 159,6 | 158,9 |
| Boosting | 127 | sin sintonizar (TPE y CMA-ES en la 98 no mejoraron en Kaggle) | 110,2 | 127,0 | 155,5 | sin envío |
| Boosting | 83 | igual que la 127 | 111,0 | 131,5 | 154,8 | 152,1 |
| Red neuronal | 148 | sin sintonizar (TPE en la 171: mejor espacial, peor ponderado) | 124,8 | 137,6 | 148,2 | 157,1 |
| **Mezcla** | **152** | pesos fijos 70/30 | 112,2 | 127,6 | 151,7 | **148,2** |

Los hiperparámetros de cada familia y el detalle de cada búsqueda están en
`views/family_table.csv` (columnas `hiperparametros` y `busqueda`). Ojo: la elastic net,
el CART y el bosque se sintonizaron con la media del ponderado y la espacial como objetivo, y
los pesos de la mezcla se escogieron entre 70/30 y 50/50 con esos mismos criterios; para
esas filas, esas dos columnas son algo optimistas y Kaggle es la única medida fuera de
muestra.

- **Validación** (`views/validation_rank_corr.csv`, sin el ingenuo): frente a Kaggle, la
  aleatoria simple apenas ordena los envíos (Spearman 0,30, n = 9); con los anuncios gemelos
  en la misma partición sube a 0,78; la espacial da 0,69 (n = 15) y el holdout nororiental
  ponderado 0,83 (n = 12). Todas son optimistas; la espacial, en unos 27 M. La espacial
  premia bajar el nivel de las predicciones y el tablero no.
- **Asimetría y regla de compra:** ver la sección "Del modelo a la decisión de compra".

## Del modelo a la decisión de compra

Con las predicciones fuera de muestra de la validación espacial (`34_best_model_errors.py`
y `35_purchase_rule.py`; tablas `views/best_errors_*.csv`, `views/asymmetric_ranking.csv` y
`views/purchase_rule*.csv`). Cifras en millones.

**Dónde se equivoca la 152.** Sobrepredice el 67,3 % de los inmuebles, con un sesgo de
+41,7 M; la sobrepredicción media es de 125,9 M y la subpredicción media de 131,3 M. Por
precio pedido, sobrepredice el 92 % de los de menos de 400 M y solo el 18 % de los de más
de 1.200 M (sesgo de −213 M). Por precio predicho, que es lo que ve el comprador, la
sobrepredicción media sube de 37 M a 214 M. El error es mayor en casas (MAE 181,3 frente a
110,3 en apartamentos) y cuando falta el área (145,6 frente a 119,0). En el boosting, el
ridge del texto concentra el 22 % de la ganancia en apartamentos y el 14 % en casas
(`views/best_errors_importance.csv`, un ajuste con una semilla).

**Pérdida asimétrica.** Si sobrepredecir cuesta 2 o 3 veces lo que subpredecir, el orden
entre familias no cambia, pero ganan los cuantiles bajos: con k = 3 el mejor es el boosting
en cuantil 0,20 (190, costo 174,0) y la 152 queda sexta (297,0). Es lo que predice la teoría:
con costo k veces mayor, el cuantil óptimo es 1/(1 + k).

**Regla de compra.** La empresa ofrece la predicción con un descuento y compra si la oferta
alcanza el precio pedido; la diferencia es sobrepago. La receta del boosting 127 se entrenó
con cuantiles 0,20, 0,30 y 0,50 (filas 190 a 192, con la misma regla de gemelos).

| Regla | Descuento | Compras cerradas | Sobrepago medio | Sobrepago / capital |
|---|---:|---:|---:|---:|
| 152 (Kaggle, cuantil 0,70) | 0 % | 67,3 % | 125,9 | 18,3 % |
| 152 (Kaggle, cuantil 0,70) | 10 % | 45,9 % | 98,3 | 15,5 % |
| 152 (Kaggle, cuantil 0,70) | 20 % | 25,1 % | 79,9 | 13,6 % |
| Cuantil 0,50 | 0 % | 52,7 % | 98,1 | 15,5 % |
| Cuantil 0,30 | 0 % | 40,5 % | 77,0 | 13,2 % |
| Cuantil 0,20 | 0 % | 33,0 % | 66,9 | 12,1 % |
| Cuantil 0,20 | 20 % | 5,5 % | 50,5 | 10,0 % |

Hay selección adversa: la oferta solo alcanza el precio cuando el modelo sobreestima, así
que toda compra cerrada es un sobrepago frente al precio pedido. A igual tasa de compra,
bajar el cuantil sobrepaga menos que descontar la predicción de Kaggle, y ninguna regla baja
del 10 % del capital.

**Recomendación.** El modelo no sirve para fijar el precio de compra, pero sí para escoger
candidatos. La regla que proponemos es ofrecer el cuantil 0,20 sin descuento, solo en
apartamentos (compra el 32,4 % con un sobrepago del 10,9 % del capital, frente a 14,9 % en
casas), y cerrar después de una tasación física. No lo usaríamos en casas ni con
predicciones por encima de 900 M, donde la sobrepredicción media pasa de 200 M. Riesgos:
(i) los precios son de oferta y no de cierre, así que el sobrepago real puede ser mayor;
(ii) falta el área en uno de cada cuatro anuncios y ahí el error es mayor; (iii) los datos
son de 2019 a 2021; (iv) cambio de zona: la espacial (127,6) es optimista frente a Kaggle
(148,2), y en el holdout junto a Chapinero el sesgo cambia de signo (−57,2), así que el
nivel correcto en Chapinero es incierto; (v) la regla (cuantil y segmento) se escogió con
las mismas predicciones espaciales con que se reporta, así que su sobrepago es algo
optimista. La validación espacial premia bajar el nivel (el
cuantil 0,50, fila 192, da 118,0 frente a 127,0 de la 127), pero en Kaggle el cuantil 0,70
le ganó al 0,65 (65 frente a 53), aunque por menos de 1 M.

## Cómo reproducir

```bash
pip install -r requirements.txt
bash run_all.sh            # Linux / macOS / Git Bash
.\run_all.ps1              # Windows PowerShell (usa .venv si existe)
```

`run_all` es la ruta principal: desde los datos crudos de `stores/raw` reproduce los modelos
ingenuos, la mejor especificación de cada familia (108, 104, 105, 114, 83 y 125 → 127, 148),
la mezcla final 152, dos comparaciones que usan las presentaciones (el kNN 185 y la mezcla
186, de la 177 → 181 con la red 148), los tres cuantiles de la regla de compra (190 a 192) y
todas las figuras y tablas de `views/` que usan las presentaciones, que se arman al final en
`document/`. PENDIENTE_TIEMPO

Notas:

- **Descargas externas.** Por defecto se usan las descargas ya versionadas
  (`stores/processed/osm_cache/`, `bogota_blocks.csv`, `bogota_lots.csv`); con
  `bash run_all.sh --download` (o `-Download`) se vuelven a bajar de OpenStreetMap y de los
  datos abiertos de Bogotá. `13_text_embeddings.py` descarga la primera vez el modelo
  `intfloat/multilingual-e5-small` de Hugging Face (lo usa el bosque 114).
- **Hiperparámetros.** Las búsquedas (`05`, `15`, `17`, `19`, `28`) no están en la ruta
  principal: tardan horas. Sus resultados quedan en `stores/processed/best_params.json`
  (versionado) y sus trazas en `views/tuning_*.csv`, `views/optuna_*.csv` y
  `views/genetic_*.csv`; los comandos están en `EXPERIMENTOS.md`.
- **Tablas de experimentos.** `views/cv_results.csv`, `views/kaggle_scores.csv` y
  `stores/processed/test_predictions.csv` (predicciones en el test de cada fila) guardan
  todas las filas probadas. La ruta principal reemplaza las de sus modelos y conserva las
  demás, que algunas figuras de la comparación usan (progresión, resultados negativos).
- **Presentaciones.** `36_build_presentations.py` arma los `.pptx` con python-pptx y los
  exporta a PDF con PowerPoint por COM; sin PowerPoint deja solo los `.pptx`.
- Los scripts se corren desde `scripts/`. La semilla es `utils.SEED`. Un envío ya escrito no
  se sobrescribe: si su contenido cambia, se guarda como `_v2`, `_v3`...

## Estructura

```
├── document/               best_equipo_06 y compare_equipo_06 (.pptx editables y .pdf)
├── scripts/
│   ├── utils.py, models.py rutas, validación, métricas, envíos; una receta por familia
│   ├── plot_style.py       un color fijo por familia, 300 dpi, millones de pesos
│   ├── deck.py             piezas comunes de las diapositivas
│   │   — ruta principal (run_all) —
│   ├── 01_download_osm.py, 11_download_bogota.py, 16_download_lots.py   fuentes externas
│   ├── 02_build_features.py   limpieza, predictores de texto, espaciales y de la manzana
│   ├── 03_naive_baselines.py  modelos ingenuos
│   ├── 13_text_embeddings.py  embeddings del texto (los usa el bosque 114)
│   ├── 04_train_models.py     todas las especificaciones; validación y envíos
│   ├── 20_twin_rules.py       regla de anuncios repetidos (125 → 127)
│   ├── 06_ensemble.py         mezclas con pesos fijos (152)
│   ├── 07_results_table.py    tabla de resultados en Markdown
│   ├── 31_data_overview.py    Etapa 1: precios, faltantes, mapa, Chapinero contra Bogotá
│   ├── 32_algorithm_comparison.py  Etapa 2: familias, progresión, receta, negativos
│   ├── 33_validation_schemes.py    Etapa 3: aleatoria, espacial, ponderado y Kaggle
│   ├── 34_best_model_errors.py     Etapa 4: anatomía del error, importancia, pérdida asimétrica
│   ├── 35_purchase_rule.py         Etapa 4: regla de compra por cuantil y descuento
│   ├── 36_build_presentations.py   las dos presentaciones
│   │   — experimentos (ver EXPERIMENTOS.md) —
│   └── 05, 08–10, 12, 14, 15, 17–19, 21–25, 27–30
├── stores/
│   ├── raw/                datos de Kaggle (train, test, plantilla de envío)
│   ├── external/           caché de descargas de Bogotá (no se versiona)
│   ├── processed/          bases con predictores, descargas externas, hiperparámetros
│   └── submissions/        un .csv por envío; los reemplazados, en obsoletos/
├── views/                  tablas y figuras
├── EXPERIMENTOS.md         rondas 1 a 10: comandos, tablas y lecciones
├── run_all.sh, run_all.ps1 ruta principal
└── requirements.txt
```

No se versionan y se regeneran con los scripts: `stores/processed/oof_predictions.csv`
(predicciones fuera de muestra de cada especificación y esquema), `text_embeddings.npz`,
`text_cache/` y los estudios de Optuna (`optuna_round*.db`).

## Datos y fuentes

- **Properati** (vía Kaggle): 38.644 anuncios de venta en Bogotá para entrenar y 10.286 en
  Chapinero para predecir, con título, descripción, tipo, alcobas, baños, área y ubicación.
  Los precios de train están entre 300 y 1.650 millones: se quitaron los anuncios fuera de
  ese rango, no se recortaron.
- **OpenStreetMap**: distancias y conteos de 11 categorías de amenidades, descargados con la
  API Overpass (`01_download_osm.py`). Datos © OpenStreetMap contributors, disponibles bajo
  la Open Database License (ODbL).
- **Datos abiertos de Bogotá**: estrato socioeconómico por manzana ("Estratificación Manzana
  Bogotá D.C.", Secretaría Distrital de Planeación) y valor de referencia del terreno por m²
  ("Valor de referencia por metro cuadrado de terreno", Unidad Administrativa Especial de
  Catastro Distrital), de 2019 a 2021 según el año del anuncio; y lote, pisos y área
  construida del catastro. Descargados de los servicios geográficos de Catastro Bogotá
  (`serviciosgis.catastrobogota.gov.co`), que publican esas capas de datos abiertos, el 19
  de noviembre de 2026.
- **Texto del anuncio**: 18 indicadores con reglas (parqueadero, terraza, ascensor, vista,
  estrato declarado, piso...) y un ridge sobre TF-IDF del título y la descripción, ajustado
  fuera de muestra dentro de cada partición.

Predictores externos y de texto, con su fuente, su intuición económica y si entran al mejor
modelo: `views/predictors_table.csv` (resumen en `views/predictors_summary.png`). La mezcla
152 usa 6 externos (estrato y valor del terreno de la manzana, distancias a 4 hitos) y 19 de
texto. Los 16 de OpenStreetMap se construyeron y se probaron, pero empeoraron la validación
espacial y quedaron fuera (`views/negative_results.csv`).

## Envíos a Kaggle

Puntaje público de cada envío subido (`views/kaggle_scores.csv`), en millones. Los envíos 10
y 11 se subieron sin OSM; sus archivos están en `stores/submissions/obsoletos/`.

| Envío | Familia | Kaggle público |
|---|---|---:|
| `00_naive_1_median` | ingenuo | 377,1 |
| `11_neural_net_full` | red neuronal | 232,4 |
| `31_ensemble_mean_3` | mezcla | 189,3 |
| `49_en_boosting_core_coords_residual` | híbrido lineal + boosting | 184,1 |
| `108_linear_ext_core_coords_splines_by_type_level_q70_ridge_oof` | regresión lineal | 180,5 |
| `10_boosting_full_coords` | boosting | 180,1 |
| `105_cart_ext_core_coords_by_type_level_q70_ridge_tuned` | CART | 177,5 |
| `54_en_boosting_core_coords_residual_q65` | híbrido lineal + boosting | 173,3 |
| `53_boosting_core_coords_by_type_level_q65` | boosting | 161,7 |
| `65_boosting_core_coords_by_type_level_q70` | boosting | 160,9 |
| `114_random_forest_ext_core_coords_by_type_level_q70_ridge_net_nocoords` | random forest | 158,9 |
| `148_quantile_net_ext_core_coords_by_type_q70_ridge_nocoords` | red neuronal | 157,1 |
| `98_boosting_ext_core_coords_by_type_level_q70_ridge_type_net_s10_tpe` | boosting | 153,4 |
| `75_boosting_core_coords_by_type_level_q70_ridge_v2` | boosting | 153,1 |
| `83_boosting_ext_core_coords_by_type_level_q70_ridge_s10_v2` | boosting | 152,1 |
| **`152_ensemble_w70_30_2`** | **mezcla** | **148,2** |

Son 16 modelos más una calibración (abajo): 17 envíos registrados. **El taller pide al menos
20 y al menos uno por familia: falta subir la elastic net** (por ejemplo
`104_elastic_net_ext_core_coords_splines_by_type_level_q70_ridge_tuned.csv`) y registrar
cualquier envío subido que no esté en `views/kaggle_scores.csv`.

### Sondeos y calibraciones con el tablero (no son modelos)

Multiplicar un envío por un factor fijo, o escalar solo su tramo alto, no es un modelo: es
calibrar el nivel con el leaderboard. Se reportan aparte y no cuentan para escoger el mejor
modelo.

| Envío | Qué hace | Kaggle público |
|---|---|---:|
| `32_boosting_full_coords_x115` | el envío 10 (sin OSM) por 1,15 | 171,7 |
| `57`, `58`, `59` (`..._q65_x110/x115/x120`) | el híbrido 54 por 1,10, 1,15 y 1,20 | sin subir |
| `60_boosting_full_coords_x120` | el envío 10 por 1,20 | sin subir |
| `189_probe_ensemble_w70_30_2_top5_x115` | el 5 % más alto de la 152 por 1,15 | sin subir |

Las cifras de validación de estas filas están en `views/level_calibration.csv` y
`views/round10_table.csv`.

## Experimentos

Todo lo que se probó en las rondas 1 a 10 (OSM, objetivos, pérdidas cuantílicas, texto,
datos de la manzana, gemelos, balanceo, modelo local, apilamiento, monotonía, kNN, mezclas),
con sus comandos y sus tablas, está en [`EXPERIMENTOS.md`](EXPERIMENTOS.md). La tabla
completa de resultados es `views/cv_results.csv` (`views/cv_results_table.md`).
