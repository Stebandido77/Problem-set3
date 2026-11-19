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
│   ├── 04_train_models.py     seis familias, CV aleatoria vs. espacial, envíos de Kaggle
│   └── 05_tune_models.py      búsqueda aleatoria de hiperparámetros con CV espacial
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
python 01_download_osm.py                 # una vez; requiere internet
python 02_build_features.py
python 03_naive_baselines.py
python 04_train_models.py                 # todas las especificaciones (~16 min en 2 núcleos)
python 04_train_models.py --families boosting random_forest   # solo algunas familias
python 05_tune_models.py --family boosting --features full --n-iter 30
python 04_train_models.py --families boosting                 # reentrena con lo sintonizado
```

`05_tune_models.py` guarda los mejores hiperparámetros en `stores/processed/best_params.json`;
`04_train_models.py` los toma automáticamente la siguiente vez que corre.

## Datos y predictores

| Bloque | Predictores | Fuente |
|---|---|---|
| Estructurales | tipo de inmueble, alcobas, baños, cuartos, área (y su logaritmo), indicador de área faltante, tendencia mensual | Properati |
| Texto (18) | parqueadero y número de parqueaderos, terraza, balcón, ascensor, depósito, chimenea, vigilancia, gimnasio, piscina, conjunto cerrado, remodelado, dúplex, penthouse, vista, estrato, piso, longitud de la descripción | título y descripción |
| Puntos de referencia (4) | distancia a Zona T, Parque de la 93, Centro Internacional y Plaza de Bolívar | coordenadas aproximadas |
| OpenStreetMap (16) | distancia al más cercano de 11 categorías (parque, estación de transporte, universidad, colegio, hospital, centro comercial, restaurante, bar, banco, supermercado, policía) y conteos a 500 m de 5 de ellas | Overpass API |

El área falta en cerca del 80 % de los anuncios; se recupera de la descripción cuando aparece
(queda faltante en 39 % de train y 36 % de test). Baños faltantes también se completan con el texto.

Conjuntos de predictores usados en los modelos: `base` (estructurales), `base_text`,
`full` (todo menos lat/lon crudos, para que el modelo aprenda de distancias y no de
coordenadas que en Chapinero casi no vio) y `full_coords` (agrega lat/lon).

## Validación

- **Aleatoria:** 5 particiones al azar.
- **Espacial:** k-means sobre lat/lon define 10 zonas; se dejan zonas completas por fuera
  (GroupKFold de 5). Imita el salto del resto de Bogotá a Chapinero.

Todos los modelos se entrenan sobre log(precio) y predicen en pesos.

## Resultados actuales

> **Estado:** estos resultados y los envíos en `stores/submissions/` se generaron **sin los
> predictores de OpenStreetMap**, porque aún no se ha corrido `01_download_osm.py`. Al
> correrlo y repetir los pasos 02–04, el conjunto `full` pasa de 30 a 46 predictores y la
> tabla se actualiza. Hiperparámetros iniciales, sin sintonizar.

| Modelo | Predictores | MAE CV aleatoria (M COP) | MAE CV espacial (M COP) | % sobrepredicho (espacial) |
|---|---:|---:|---:|---:|
| `naive_median` | 0 | 236.7 | 248.9 | – |
| `naive_mean` | 0 | 247.3 | 254.7 | – |
| `naive_median_by_type_bedrooms` | 0 | 217.7 | 242.6 | – |
| `01_linear_base` | 8 | 187.4 | 202.2 | 51% |
| `02_linear_base_text` | 26 | 173.3 | 186.0 | 52% |
| `03_linear_full` | 30 | 157.9 | 168.3 | 53% |
| `04_elastic_net_full` | 30 | 139.3 | 152.3 | 52% |
| `05_cart_full` | 30 | 131.3 | 163.6 | 49% |
| `06_random_forest_full` | 30 | 98.9 | 143.4 | 49% |
| `07_random_forest_full_coords` | 32 | 95.3 | 141.9 | 51% |
| `08_boosting_base_text` | 26 | 124.4 | 150.7 | 49% |
| `09_boosting_full` | 30 | 101.1 | 140.6 | 47% |
| `10_boosting_full_coords` | 32 | 97.3 | 137.0 | 50% |
| `11_neural_net_full` | 30 | 144.1 | 182.6 | 52% |

Lectura preliminar:

- La validación aleatoria es optimista: en bosques y boosting el MAE espacial es 40–45 % mayor
  que el aleatorio. Los modelos flexibles memorizan vecindarios y eso no viaja.
- El orden de los modelos cambia entre esquemas: CART le gana al elastic net en aleatoria y
  pierde en espacial; la red neuronal es la que peor viaja.
- El sesgo medio es negativo (subpredicción) en todos los modelos, propio de entrenar en
  logaritmos; aun así cerca de la mitad de los inmuebles queda sobrepredicha.

## Pendientes

- [ ] Correr `01_download_osm.py` y regenerar predictores, modelos y envíos.
- [ ] Sintonizar cada familia con `05_tune_models.py` y registrar el puntaje público de Kaggle por envío.
- [ ] Comparar CV aleatoria, CV espacial y leaderboard público (Etapa 3).
- [ ] Análisis de errores asimétricos y regla de compra (Etapa 4), a partir de `oof_predictions.csv`.
- [ ] Figuras y tablas para las dos presentaciones.
