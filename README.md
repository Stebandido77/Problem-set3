# Problem Set 3 — Making Money with ML?

**MECA 4107 · Big Data and Machine Learning para Economía Aplicada · Universidad de los Andes · 2026-20**

Equipo 06: Neislen García y Esteban Labastidas

## Qué responde el proyecto

Una empresa quiere comprar vivienda en Chapinero (Bogotá) usando un modelo que prediga el
precio de cada anuncio. Entrenamos con anuncios de Properati del resto de la ciudad, así que
el modelo tiene que viajar a una zona que casi no vio, más cara y distinta. Kaggle evalúa con
el error absoluto medio (MAE), pero la pregunta de negocio es otra: **¿se puede confiar en el
modelo lo suficiente para comprar, sabiendo que sobrepredecir cuesta más que subpredecir?**

## Resultados principales

Cifras en millones de pesos. Salen de `views/family_table.csv`, `views/validation_schemes.csv`,
`views/beto_table.csv`, `views/kaggle_scores.csv` y `views/kaggle_final.csv`.

**Mejor modelo: `203_ensemble_w70_30_2`**, la mezcla del boosting 198 (70 %) y la red
neuronal 196 (30 %), los dos con BETO y el ridge como lectores del texto. **Quedó en el
primer puesto de los dos tableros de Kaggle:**

| Tablero | Parte del test | MAE | Puesto |
|---|---:|---:|---:|
| Público | 20 % | 145,7 (145.654.598) | 1 |
| Privado (definitivo) | 80 % | 148,6 (148.551.335) | 1; el segundo, 150,8 (150.790.433) |

Del público al privado el MAE sube 2,9 M (2,0 %). En la validación espacial propia la 203
da 124,8 M, optimista frente a los dos tableros. Frente al ingenuo (mediana constante,
377,1 M en el público) el error baja 61 %; frente a la mezcla anterior sin BETO (152,
148,2 M), 2,6 M.

- **Boosting 198:** la receta de la 127 (LightGBM con pérdida cuantílica en 0,70 sobre el
  precio en nivel, un modelo para apartamentos y otro para casas, estrato y valor del
  terreno de la manzana, el texto completo resumido en un ridge fuera de muestra, el área y
  los baños de los anuncios del mismo punto, sin latitud ni longitud crudas, 10 semillas y
  la regla que mezcla la predicción con el precio de anuncios de train con la misma
  descripción) más el precio que predice BETO con el texto (`txt_beto`).
- **Red 196:** la red cuantílica (0,70) por tipo de la 148, con los mismos predictores de la
  manzana, el ridge del texto y `txt_beto`, sin coordenadas. Mejora la mezcla en los dos
  deciles de precio más caros (desde 891 M): ahí el boosting subpredice más (sesgo de
  −215 M en el decil 10, frente a −163 M de la red), y la mezcla le gana al boosting solo por
  2,4 y 9,0 M; en los demás deciles pierde hasta 3,9 M. En la validación espacial la mezcla
  queda 0,4 M por encima del boosting 198 solo (ruido) y en el holdout nororiental 4,1 M por
  debajo; la correlación de errores entre los dos miembros es 0,90 y 0,96
  (`views/blend_deciles.csv`, `views/blend_members.csv`, script `42_deck_checks.py`). La
  razón no es pasar el tope de 1.650 M: los sondeos indican que el test no lo pasa, y las 20
  predicciones de la 203 por encima del tope (máximo 1.745 M, `views/test_over_cap.csv`)
  deberían recortarse, como hace la 206.

Una fila por familia (su mejor especificación con la receta común):

| Familia | Fila | Búsqueda de hiperparámetros | Aleatoria | Aleatoria con gemelos | Espacial | Ponderado | Kaggle |
|---|---:|---|---:|---:|---:|---:|---:|
| Regresión lineal | 108 | no aplica (mínimos cuadrados) | 138,8 | 144,1 | 164,1 | 166,0 | 180,5 |
| Elastic net | 104 | Optuna TPE, 15 evaluaciones | 139,0 | 144,5 | 158,5 | 165,8 | sin envío |
| CART | 105 | Optuna TPE, 25 evaluaciones | 133,2 | 141,3 | 159,8 | 168,7 | 177,5 |
| Random forest | 114 | Optuna TPE, 25; NSGA-II, 120 (no mejora) | 105,6 | 123,1 | 143,8 | 159,6 | 158,9 |
| Boosting | 127 | sin sintonizar (TPE y CMA-ES en la 98 no mejoraron en Kaggle) | 89,0 | 110,2 | 127,0 | 155,5 | sin envío |
| Boosting | 83 | igual que la 127 | 96,2 | 111,0 | 131,5 | 154,8 | 152,1 |
| Red neuronal | 148 | sin sintonizar (TPE en la 171: mejor espacial, peor ponderado) | 118,0 | 124,8 | 137,6 | 148,2 | 157,1 |
| Mezcla | 152 | pesos fijos 70/30 | 95,5 | 112,2 | 127,6 | 151,7 | 148,2 |
| **Mezcla con BETO** | **203** | pesos fijos 70/30 (se comparó con 60/40) | – | – | **124,8** | 150,5* | **145,7** |

\* Optimista: BETO se afinó con precios de la validación nororiental, incluidos los de más de
1.000 M que ese holdout deja fuera del ajuste. La 203 no tiene validación aleatoria.

**Qué aportó BETO** (`views/beto_check.csv`, `views/beto_deciles.csv`, `views/beto_table.csv`;
ronda 11 en `EXPERIMENTOS.md`). Como lector del texto solo, fuera de muestra con las mismas
particiones, BETO se equivoca 148,5 M y el ridge 169,1 M; el ridge acierta más en los
deciles de precio 4 a 8 y BETO en los extremos. Sumado al ridge, mejora los dos modelos de la
mezcla en la espacial: el boosting pasa de 127,0 a 124,4 (198) y la red de 137,6 a 133,7
(196). La mezcla 70/30 de ambos (203) da 124,8 en la espacial frente a 127,6 de la 152, y
en el tablero público 145,7 frente a 148,2.

Los hiperparámetros de cada familia y el detalle de cada búsqueda están en
`views/family_table.csv` (columnas `hiperparametros` y `busqueda`). Ojo: la elastic net,
el CART y el bosque se sintonizaron con la media del ponderado y la espacial como objetivo, y
los pesos de la mezcla se escogieron entre 70/30 y 50/50 con esos mismos criterios; para
esas filas, esas dos columnas son algo optimistas y Kaggle es la única medida fuera de
muestra.

- **Validación** (`views/validation_rank_corr.csv`, conjunto "comunes a los cuatro, sin el
  ingenuo"): sobre los mismos 8 envíos, el Spearman con el tablero público es 0,76 para la
  aleatoria simple, 0,74 con los anuncios repetidos en la misma partición, 0,93 para la
  espacial y 0,88 para el holdout nororiental ponderado. Con conjuntos distintos por esquema
  el orden cambiaba (la aleatoria simple quedaba en 0,44 con 13 envíos), así que la
  comparación válida es la del conjunto común: la espacial ordena mejor. Entre esos envíos,
  la aleatoria y la espacial escogen el mismo mejor modelo (la 152, también la mejor en el
  tablero); entre las 58 especificaciones con las dos validaciones, escogen la misma receta
  de boosting con distinta regla de anuncios repetidos (126 y 127;
  `views/scheme_best_model.csv`). Solo la 203 tiene puntaje privado registrado, así que no
  hay comparación contra el privado. Todos los esquemas son optimistas; la espacial, en unos
  18 M en el conjunto común. La espacial premia bajar el nivel de las predicciones y el
  tablero no.
- **OpenStreetMap:** empeoró la espacial (+7,9 a +8,6 M). En el test, el rango
  intercuartílico de cada distancia de OSM es entre 0,6 y 1,5 veces el de entrenamiento
  (`views/osm_dispersion.csv`), así que "en Chapinero todo queda cerca" no lo explica; la
  causa no está medida.
- **Asimetría y regla de compra:** ver la sección "Del modelo a la decisión de compra".

## Del modelo a la decisión de compra

Con las predicciones fuera de muestra de la validación espacial (`34_best_model_errors.py`
y `35_purchase_rule.py`; tablas `views/best_errors_*.csv`, `views/asymmetric_ranking.csv` y
`views/purchase_rule*.csv`). Cifras en millones.

**Dónde se equivoca la 203.** Sobrepredice el 66,6 % de los inmuebles, con un sesgo de
+39,4 M; la sobrepredicción media es de 123,3 M y la subpredicción media de 127,9 M. Por
precio pedido, sobrepredice el 92 % de los de menos de 400 M y solo el 18 % de los de más
de 1.200 M (sesgo de −205 M). Por precio predicho, que es lo que ve el comprador, la
sobrepredicción media sube de 37 M a 213 M. El error es mayor en casas (MAE 175,7 frente a
108,3 en apartamentos) y cuando falta el área (141,7 frente a 117,0). En el boosting sin
BETO (la receta de la 125), el ridge del texto concentra el 22 % de la ganancia en
apartamentos y el 13 % en casas (`views/best_errors_importance.csv`, un ajuste con una
semilla; no se recalculó con BETO).

**Pérdida asimétrica.** Si sobrepredecir cuesta 2 o 3 veces lo que subpredecir, el orden
entre familias no cambia, pero ganan los cuantiles bajos: con k = 3 el mejor es el boosting
en cuantil 0,20 (190, costo 174,0) y la 203 queda sexta (289,1). Es lo que predice la teoría:
con costo k veces mayor, el cuantil óptimo es 1/(1 + k). Los cuantiles bajos (190 a 192)
son el boosting sin BETO.

**Regla de compra.** La empresa ofrece la predicción con un descuento y compra si la oferta
alcanza el precio pedido; la diferencia es sobrepago. La receta del boosting 127 se entrenó
con cuantiles 0,20, 0,30 y 0,50 (filas 190 a 192, con la misma regla de anuncios repetidos).

| Regla | Descuento | Compras cerradas | Sobrepago medio | Sobrepago / capital |
|---|---:|---:|---:|---:|
| 203 (Kaggle, cuantil 0,70) | 0 % | 66,6 % | 123,3 | 18,0 % |
| 203 (Kaggle, cuantil 0,70) | 10 % | 45,4 % | 94,9 | 15,1 % |
| 203 (Kaggle, cuantil 0,70) | 20 % | 23,6 % | 79,3 | 13,5 % |
| Cuantil 0,50 | 0 % | 52,7 % | 98,1 | 15,5 % |
| Cuantil 0,30 | 0 % | 40,5 % | 77,0 | 13,2 % |
| Cuantil 0,20 | 0 % | 33,0 % | 66,9 | 12,1 % |
| Cuantil 0,20 | 20 % | 5,5 % | 50,5 | 10,0 % |

Hay selección adversa: la oferta solo alcanza el precio cuando el modelo sobreestima, así
que toda compra cerrada es un sobrepago frente al precio pedido. A igual tasa de compra,
bajar el cuantil sobrepaga menos que descontar la predicción de Kaggle, y ninguna regla baja
del 10 % del capital.

**Recomendación.** Sí se puede usar para comprar, con una oferta conservadora y un avalúo.
La regla que proponemos es ofertar el cuantil 0,20 sin descuento (un precio que el 80 % de
los inmuebles parecidos supera), solo en apartamentos con oferta por debajo de 900 M, y
cerrar después de un avalúo. En apartamentos se cierra el 32,4 % de las ofertas y la
empresa sobrepaga el 10,9 % del capital, frente al 16,6 % al ofertar la predicción de la
203 en apartamentos. En casas sobrepaga el 14,9 %, y en apartamentos con oferta desde
900 M, el 12,0 %, con 124 M por compra frente a 53 M por debajo de ese valor
(`views/purchase_rule_segments.csv`). Riesgos:
(i) los precios son de oferta y no de cierre, así que el sobrepago real puede ser mayor;
(ii) falta el área en uno de cada cuatro anuncios y ahí el error es mayor; (iii) los datos
son de 2019 a 2021; (iv) cambio de zona: la espacial (124,8) es optimista frente a Kaggle
(145,7 en el público y 148,6 en el privado), y en el holdout junto a Chapinero el sesgo
cambia de signo (−53,3), así que el
nivel correcto en Chapinero es incierto; (v) la regla (cuantil y segmento) se escogió con
las mismas predicciones espaciales con que se reporta, así que su sobrepago es algo
optimista. La validación espacial premia bajar el nivel (el
cuantil 0,50, fila 192, da 118,0 frente a 127,0 de la 127), pero en Kaggle el cuantil 0,70
quedó 0,7 M por delante del 0,65 (65 frente a 53), una diferencia dentro del ruido.

## Cómo reproducir

```bash
pip install -r requirements.txt
bash run_all.sh            # Linux / macOS / Git Bash
.\run_all.ps1              # Windows PowerShell (usa .venv si existe)
```

`run_all` es la ruta principal: desde los datos crudos de `stores/raw` reproduce los modelos
ingenuos, la mejor especificación de cada familia (108, 104, 105, 114, 83 y 125 → 127, 148),
la mezcla 152, dos comparaciones que usan las presentaciones (el kNN 185 y la mezcla
186, de la 177 → 181 con la red 148), el mejor modelo en Kaggle (194 → 198 y 196, mezclados
en la 203, con la predicción de BETO versionada), los tres cuantiles de la regla de compra
(190 a 192) y todas las figuras y tablas de `views/` que usan las presentaciones, que se
arman al final en `document/`. Una corrida completa desde cero en un clon limpio (sin los
predictores procesados, con las descargas externas versionadas) tardó 2 h 15 min en un
portátil con Windows 11 y CPU, y reprodujo los 17 envíos de la ruta principal de entonces
(3 ingenuos y 14 modelos, entre ellos la 152) con los mismos precios al peso:
`save_submission` no creó ninguna versión `_v2` nueva. Las cifras de validación de
`views/cv_results.csv` también coinciden; solo cambian los minutos de cada ajuste. El paso
de la 203 se agregó a `run_all` después de esa corrida y todavía no se ha corrido desde cero
dentro de la ruta; sus comandos son los de la ronda 11 de `EXPERIMENTOS.md`.

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
- **Presentaciones.** Son Beamer con la plantilla de los Andes (`document/*.tex`).
  `36_build_presentations.py` escribe desde `views/` las cifras, las tablas y las notas del
  orador, y compila con LuaLaTeX (`latexmk -lualatex`). Las figuras son los PDF vectoriales
  que dejan en `views/` los scripts 31 a 35 y 41. `42_deck_checks.py` y `33` aportan
  cifras que no son figuras: la mezcla por decil, el tope, OSM, el área faltante y el
  Spearman sobre los mismos envíos. Sin LaTeX instalado, genera todo menos los PDF.
- **Predictor de BETO (`txt_beto`).** Es el log del precio que predice
  [BETO](https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased)
  (`dccuchile/bert-base-spanish-wwm-uncased`) afinado con el título y la descripción. Se
  genera en tres pasos:
  1. `37_beto_input.py` escribe `stores/processed/beto_input.csv` con el texto, el precio y
     la partición de la validación espacial de cada anuncio; los anuncios con la misma
     descripción van a la partición del primero. El test lleva la partición -1.
  2. `beto_kaggle.py` corre en un notebook de Kaggle con GPU; por defecto, 3 épocas y 256
     tokens. Afina un modelo por partición y predice las filas que dejó fuera; en el test
     promedia los cinco.
  3. Su salida, `stores/processed/beto_predictions.csv`, se versiona para que el repo corra
     sin GPU. `utils.add_beto` la une por `property_id` y `38_beto_check.py` la verifica.
- Los scripts se corren desde `scripts/`. La semilla es `utils.SEED`. Un envío ya escrito no
  se sobrescribe: si su contenido cambia, se guarda como `_v2`, `_v3`...

## Estructura

```
├── document/               best_equipo_06 y compare_equipo_06 (.tex, .pdf y notas del orador)
├── scripts/
│   ├── utils.py, models.py rutas, validación, métricas, envíos; una receta por familia
│   ├── plot_style.py       paleta y fuente de la plantilla, tamaños de diapositiva, PDF
│   │   — ruta principal (run_all) —
│   ├── 01_download_osm.py, 11_download_bogota.py, 16_download_lots.py   fuentes externas
│   ├── 02_build_features.py   limpieza, predictores de texto, espaciales y de la manzana
│   ├── 03_naive_baselines.py  modelos ingenuos
│   ├── 13_text_embeddings.py  embeddings del texto (los usa el bosque 114)
│   ├── 04_train_models.py     todas las especificaciones; validación y envíos
│   ├── 20_twin_rules.py       regla de anuncios repetidos (125 → 127)
│   ├── 06_ensemble.py         mezclas con pesos fijos (152, 203)
│   ├── 38_beto_check.py       verifica la predicción de BETO y la compara con el ridge
│   ├── 07_results_table.py    tabla de resultados en Markdown
│   ├── 31_data_overview.py    Etapa 1: precios, faltantes, mapa, Chapinero contra Bogotá
│   ├── 32_algorithm_comparison.py  Etapa 2: familias, progresión, receta, negativos
│   ├── 33_validation_schemes.py    Etapa 3: aleatoria, espacial, ponderado y Kaggle
│   ├── 34_best_model_errors.py     Etapa 4: anatomía del error, importancia, pérdida asimétrica
│   ├── 35_purchase_rule.py         Etapa 4: regla de compra por cuantil y descuento
│   ├── 41_beto_figure.py           BETO contra el ridge por decil de precio
│   ├── 42_deck_checks.py           mezcla por decil, tope, dispersión de OSM, área faltante
│   ├── 36_build_presentations.py   las dos presentaciones
│   │   — experimentos (ver EXPERIMENTOS.md) —
│   ├── 37_beto_input.py, beto_kaggle.py   entrada y afinado de BETO (Kaggle, GPU)
│   ├── 39_beto_table.py, 40_clip_range.py   tabla de la ronda de BETO y recorte al rango
│   └── 05, 08–10, 12, 14, 15, 17–19, 21–25, 27–30
├── stores/
│   ├── raw/                datos de Kaggle (train, test, plantilla de envío)
│   ├── external/           caché de descargas de Bogotá (no se versiona)
│   ├── processed/          bases con predictores, descargas externas, hiperparámetros
│   └── submissions/        un .csv por envío; los reemplazados, en obsoletos/
├── views/                  tablas y figuras
├── EXPERIMENTOS.md         rondas 1 a 11: comandos, tablas y lecciones
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
  estrato declarado, piso...), un ridge sobre TF-IDF del título y la descripción, ajustado
  fuera de muestra dentro de cada partición, y la predicción de BETO afinado con el mismo
  texto (fuera de muestra con las particiones espaciales; ver "Cómo reproducir").

Predictores externos y de texto, con su fuente, su intuición económica y si entran al mejor
modelo: `views/predictors_table.csv` (resumen en `views/predictors_summary.png`). La mezcla
152 usa 6 externos (estrato y valor del terreno de la manzana, distancias a 4 hitos) y 19 de
texto; la 203 agrega a esos el precio que predice BETO. Los 16 de OpenStreetMap se construyeron y se probaron, pero empeoraron la validación
espacial y quedaron fuera (`views/negative_results.csv`).

## Envíos a Kaggle

Puntaje de cada envío subido (`views/kaggle_scores.csv`), en millones. Los envíos 10 y 11 se
subieron sin OSM; sus archivos están en `stores/submissions/obsoletos/`. La competencia ya
cerró: el puntaje privado (80 % del test) solo está registrado para el envío final, la 203;
la posición en los dos tableros está en `views/kaggle_final.csv`.

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
| `152_ensemble_w70_30_2` | mezcla | 148,2 |
| **`203_ensemble_w70_30_2`** | **mezcla con BETO** | **145,7** (privado: 148,6; 1.er puesto en los dos) |

Son 17 modelos más tres calibraciones (abajo): 20 envíos registrados. El taller pide al menos
uno por familia: la elastic net no tiene puntaje registrado en `views/kaggle_scores.csv`
(por ejemplo
`104_elastic_net_ext_core_coords_splines_by_type_level_q70_ridge_tuned.csv`). Si se subió
algún envío que no está en ese archivo, falta registrarlo.

### Sondeos y calibraciones con el tablero (no son modelos)

Multiplicar un envío por un factor fijo, o escalar solo su tramo alto, no es un modelo: es
calibrar el nivel con el leaderboard. Se reportan aparte y no cuentan para escoger el mejor
modelo.

| Envío | Qué hace | Kaggle público |
|---|---|---:|
| `32_boosting_full_coords_x115` | el envío 10 (sin OSM) por 1,15 | 171,7 |
| `57`, `58`, `59` (`..._q65_x110/x115/x120`) | el híbrido 54 por 1,10, 1,15 y 1,20 | sin subir |
| `60_boosting_full_coords_x120` | el envío 10 por 1,20 | sin subir |
| `189_probe_ensemble_w70_30_2_top5_x115` | el 5 % más alto de la 152 por 1,15 | 159,5 |
| `probe_152_top5_x090` | el 5 % más alto de la 152 por 0,90 | 148,9 |

La 189 sale de `30_probe_top.py --share 0.05 --factor 1.15`. `probe_152_top5_x090` se armó
fuera de los scripts y se conserva con el nombre con que se subió: es exactamente la regla
`scale_top` de ese script (`--share 0.05 --factor 0.90`) aplicada a
`152_ensemble_w70_30_2.csv`, sin fila en `cv_results.csv`. En los dos, el umbral es el
percentil 95 de las propias predicciones, sin usar precios. Subir el tramo caro un 15 %
empeora el puntaje en 11,3 M y bajarlo un 10 % lo empeora solo 0,7 M: ese tramo está bien
calibrado y nada indica que el test pase del tope de 1.650 M del train.

Las cifras de validación de estas filas están en `views/level_calibration.csv` y
`views/round10_table.csv`.

## Experimentos

Todo lo que se probó en las rondas 1 a 11 (OSM, objetivos, pérdidas cuantílicas, texto,
datos de la manzana, gemelos, balanceo, modelo local, apilamiento, monotonía, kNN, mezclas,
BETO),
con sus comandos y sus tablas, está en [`EXPERIMENTOS.md`](EXPERIMENTOS.md). La tabla
completa de resultados es `views/cv_results.csv` (`views/cv_results_table.md`).
