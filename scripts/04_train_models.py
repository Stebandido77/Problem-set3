"""Entrena las seis familias de algoritmos (y el híbrido en_boosting), las evalúa con
validación cruzada aleatoria, validación cruzada espacial, el holdout tipo Chapinero y el
holdout nororiental, y escribe un envío de Kaggle por cada especificación.

Salidas:
  stores/submissions/<nn>_<familia>_<predictores>[_variante].csv   envíos para Kaggle
  stores/processed/oof_predictions.csv     predicciones fuera de muestra y del holdout
  stores/processed/test_predictions.csv    predicciones del test por especificación
  views/cv_results.csv                     tabla comparativa

Uso:
  python scripts/04_train_models.py                       # todas las especificaciones
  python scripts/04_train_models.py --families boosting   # solo una familia
  python scripts/04_train_models.py --specs 12 13         # solo algunos números
  python scripts/04_train_models.py --specs 54 --schemes northeast --no-submission
      # un solo esquema; los demás conservan su resultado anterior (ojo: si la
      # especificación cambió, hay que correrla con todos los esquemas)
"""

import argparse
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from models import (FAMILIES, MONOTONE_FEATURES, ByPropertyType, DropColumns, EmbeddingNet,
                    EmbeddingSVD, FocusFineTune, ImputedArea, LocalFit, NameLevel,
                    NeighborhoodLevel, PinballBoosting, PriceWeighted, QuantileKNN, QuantileNet,
                    ScaledColumns,
                    ResidualBoosting, RidgeText, RoundColumns, TailOversample, SameBuilding, SeedAverage, SyntheticTail,
                    TextFeatures, TextNet, TextPredictors, TrimPricePerM2, TruncationCorrection,
                    TwinFilter, build_model,
                    fit_model, load_best_params, northeast_building_groups)
from utils import (COORD_FEATURES, ID_COL, PROCESSED_DIR, TARGET, VIEWS_DIR, add_embeddings,
                   chapinero_holdout, error_summary, feature_sets, load_processed,
                   northeast_holdout, random_cv, random_twin_splits, save_submission,
                   similarity_weights, spatial_cv_splits, test_center, weighted_mae)


# Esquemas de validación: nombre -> función (train, test) -> lista de particiones.
SCHEMES = {
    "random": lambda train, test: list(random_cv().split(train)),
    # La misma validación aleatoria, pero con los gemelos siempre en la misma partición.
    "random_twins": lambda train, test: random_twin_splits(train),
    "spatial": lambda train, test: spatial_cv_splits(train),
    "chapinero": lambda train, test: [chapinero_holdout(train, test)],
    "northeast": lambda train, test: [northeast_holdout(train, test)],
}

# Familias que usan un solo núcleo: conviene ajustar sus particiones en paralelo.
# La red cuantílica usa 4 hilos de torch por ajuste: sus particiones también van en paralelo.
SINGLE_CORE_FAMILIES = {"linear", "elastic_net", "cart", "neural_net", "quantile_net", "knn"}


@dataclass(frozen=True)
class Spec:
    """Una especificación: familia, conjunto de predictores y variantes de entrenamiento."""
    family: str
    features: str
    variant: str = ""        # sufijo del nombre del envío
    target: str = "log"      # 'log', 'level' o 'per_m2' (ver models.build_model)
    weighted: bool = False   # pondera el entrenamiento por parecido al test
    by_type: bool = False    # modelo aparte para apartamentos y para casas
    tuned: bool = False      # usa los hiperparámetros de 05_tune_models.py
    params: tuple = ()       # hiperparámetros fijos, como pares (ruta, valor)
    neighbors: bool = False  # agrega el nivel de precio de los vecinos (NeighborhoodLevel)
    price_weight: float = 0  # peso máximo por decil de precio (PriceWeighted); 0 = sin pesos
    synthetic: bool = False  # sintéticos tipo SMOTER de la cola alta (SyntheticTail)
    impute_area: bool = False  # completa el área faltante con un modelo (ImputedArea)
    truncation: bool = False   # corrige por el truncamiento del precio (TruncationCorrection)
    upper_only: bool = False   # la corrección solo en el techo y sin soluciones en el borde
    same_building: float = 0     # fracción del ajuste con precio del mismo edificio
                                 # (SameBuilding); 0 = sin el predictor
    n_seeds: int = 1             # promedio de varias semillas (SeedAverage)
    fine_tune: bool = False      # sigue entrenando cerca de la Zona T (FocusFineTune)
    text: str = ""               # texto completo: 'svd20', 'svd50', 'svd100' o 'ridge'
    names: bool = False          # nivel del barrio o edificio del título (NameLevel)
    text_sources: tuple = ()     # fuentes de TEXT_SOURCES que alimentan al modelo
                                 # (TextPredictors, ronda 4)
    tune_key: str = ""           # clave de best_params.json si no es '<familia>_<conjunto>'
    recipe: str = ""             # 'q70': versión cuantílica de la ronda 5 (QUANTILE_RECIPES);
                                 # otras: variantes de la ronda 6 (ROUND6_RECIPES)
    drop_coords: bool = False    # el modelo no ve lat ni lon (sí el texto, para los gemelos)
    round_coords: int = 0        # decimales a los que se redondean lat y lon (RoundColumns)
    oversample: int = 0          # copias del 20 % más caro del nororiente (TailOversample)
    local: tuple = ()            # ('radius', metros) o ('northeast_high', 0) (LocalFit)
    twins: str = ""              # 'drop' o 'weight' para los anuncios repetidos (TwinFilter)
    trim: bool = False           # sin precios por m2 fuera de p1-p99 del tipo (TrimPricePerM2)
    surface: str = ""            # columna de área si no es la del conjunto ('surface_pt')
    pinball: bool = False        # LightGBM con pérdida pinball propia (PinballBoosting)
    monotone: tuple = ()         # columnas con restricción monótona creciente (con pinball)
    scale: bool = False          # estandariza los predictores (ScaledColumns)

    def name(self, number):
        suffix = f"_{self.variant}" if self.variant else ""
        return f"{number:02d}_{self.family}_{self.features}{suffix}"


# Hiperparámetros fijos de algunas variantes (rutas dentro de build_model).
L2_LOSS = (("regressor__lgbmregressor__objective", "l2"),)


# En el híbrido, el boosting de residuos es un pipeline simple (sin 'regressor__').
RESIDUAL_Q60 = (("lgbmregressor__objective", "quantile"), ("lgbmregressor__alpha", 0.60))
RESIDUAL_Q65 = (("lgbmregressor__objective", "quantile"), ("lgbmregressor__alpha", 0.65))


def quantile_loss(alpha):
    return (("regressor__lgbmregressor__objective", "quantile"),
            ("regressor__lgbmregressor__alpha", alpha))


# El orden cuenta la historia: de simple a complejo. Un número nunca se reutiliza.
SPECS = [
    Spec("linear", "base"),
    Spec("linear", "base_text"),
    Spec("linear", "full"),
    Spec("elastic_net", "full"),
    Spec("cart", "full"),
    Spec("random_forest", "full"),
    Spec("random_forest", "full_coords"),
    Spec("boosting", "base_text"),
    Spec("boosting", "full"),
    Spec("boosting", "full_coords"),
    Spec("neural_net", "full"),
    # Idea 1: ponderar el entrenamiento por parecido al test.
    Spec("boosting", "full_coords", "weighted", weighted=True),
    Spec("random_forest", "full", "weighted", weighted=True),
    # Idea 2: modelo aparte para apartamentos (97 % del test) y para casas.
    Spec("boosting", "full_coords", "by_type", by_type=True),
    Spec("random_forest", "full", "by_type", by_type=True),
    # Idea 3: objetivo alternativo. Precio en nivel con pérdida L1 y precio por m2.
    Spec("boosting", "full_coords", "by_type_level", target="level", by_type=True),
    Spec("boosting", "full_coords", "by_type_per_m2", target="per_m2", by_type=True),
    Spec("random_forest", "full", "by_type_per_m2", target="per_m2", by_type=True),
    # Idea 4: predictores de gama alta del texto, ejes de ciudad y conteos de OSM a 300 m y 1 km.
    Spec("boosting", "rich_coords", "by_type_per_m2", target="per_m2", by_type=True),
    Spec("random_forest", "rich", "by_type_per_m2", target="per_m2", by_type=True),
    # Idea 5: hiperparámetros sintonizados con validación espacial (05_tune_models.py).
    Spec("cart", "full", "tuned", tuned=True),
    Spec("neural_net", "full", "tuned", tuned=True),
    # Ojo: 05 busca sobre log(precio); estos parámetros se copian a los dos submodelos del
    # precio por m2, así que no están sintonizados para ese objetivo.
    Spec("boosting", "full_coords", "by_type_per_m2_tuned", target="per_m2", by_type=True,
         tuned=True),
    Spec("random_forest", "full", "by_type_per_m2_tuned", target="per_m2", by_type=True,
         tuned=True),
]
# Los números 30 a 32 son ensambles de 06_ensemble.py y la versión x1,15 del envío 10.

# Ronda 2, desde el 33. Criterio: holdout nororiental (train hasta 1.000 M, ver utils).
ROUND_2_START = 33
ROUND_2 = [
    # 2b. Elastic net en área, baños y ubicación, con boosting sobre sus residuos.
    Spec("en_boosting", "full_coords", "residual"),
    # 2c. Precio en nivel con pérdida cuadrática (frente a log en 14 y L1 en 16).
    Spec("boosting", "full_coords", "by_type_level_l2", target="level", by_type=True,
         params=L2_LOSS),
    # 3. Área ampliada y señales de lujo del texto; luego, área faltante imputada con modelo.
    Spec("boosting", "lux_coords", "by_type_per_m2", target="per_m2", by_type=True),
    Spec("boosting", "lux_coords", "by_type_per_m2_imputed", target="per_m2", by_type=True,
         impute_area=True),
    # 4. Nivel de precio de los 20 vecinos más cercanos, calculado dentro de cada ajuste.
    Spec("boosting", "full_coords", "by_type_per_m2_nbr", target="per_m2", by_type=True,
         neighbors=True),
    Spec("random_forest", "full", "by_type_per_m2_nbr", target="per_m2", by_type=True,
         neighbors=True),
    # 5. Rebalanceo hacia la cola alta, sobre la mejor variante (precio en nivel, 16).
    Spec("boosting", "full_coords", "by_type_level_pw3", target="level", by_type=True,
         price_weight=3),
    Spec("random_forest", "full", "by_type_per_m2_pw3", target="per_m2", by_type=True,
         price_weight=3),
    Spec("boosting", "full_coords", "by_type_level_q55", target="level", by_type=True,
         params=quantile_loss(0.55)),
    Spec("boosting", "full_coords", "by_type_level_q60", target="level", by_type=True,
         params=quantile_loss(0.60)),
    Spec("boosting", "full_coords", "by_type_level_q65", target="level", by_type=True,
         params=quantile_loss(0.65)),
    Spec("boosting", "full_coords", "by_type_level_smoter", target="level", by_type=True,
         synthetic=True),
    # Sin OSM: en el holdout nororiental el boosting 10 sin OSM (el subido) le gana al con OSM.
    Spec("boosting", "core_coords", "by_type_level", target="level", by_type=True),
    # 5d. Corrección por truncamiento: de la mediana truncada a la mediana sin truncar.
    Spec("boosting", "full_coords", "by_type_level_trunc", target="level", by_type=True,
         truncation=True),
    Spec("random_forest", "full", "by_type_per_m2_trunc", target="per_m2", by_type=True,
         truncation=True),
    Spec("boosting", "core_coords", "by_type_level_trunc", target="level", by_type=True,
         truncation=True),
    # 6. Combinaciones de lo que funcionó: híbrido lineal + boosting, sin OSM, cuantil 0,6
    # y corrección por truncamiento.
    Spec("en_boosting", "core_coords", "residual"),
    Spec("en_boosting", "full_coords", "residual_q60", params=RESIDUAL_Q60),
    Spec("en_boosting", "core_coords", "residual_q60", params=RESIDUAL_Q60),
    Spec("en_boosting", "core_coords", "residual_trunc", truncation=True),
    Spec("boosting", "core_coords", "by_type_level_q65", target="level", by_type=True,
         params=quantile_loss(0.65)),
    Spec("en_boosting", "core_coords", "residual_q65", params=RESIDUAL_Q65),
]
# Los números 55 a 60 son ensambles (06_ensemble.py) y versiones escaladas (09).
ROUND_2B_START = 61
ROUND_2B = [
    # 7. Señales de lujo y área ampliada dentro de la parte lineal del híbrido.
    Spec("en_boosting", "lux_core_coords", "residual_q65", params=RESIDUAL_Q65),
    # 8. Corrección por truncamiento arreglada (la revisión encontró soluciones en el borde).
    Spec("boosting", "core_coords", "by_type_level_trunc_up", target="level", by_type=True,
         truncation=True, upper_only=True),
]
# El 64 no se usa. En Kaggle el 53 (cuantil 0,65) le gana al híbrido 54: se sube el cuantil.
ROUND_2C_START = 65
ROUND_2C = [
    Spec("boosting", "core_coords", "by_type_level_q70", target="level", by_type=True,
         params=quantile_loss(0.70)),
    Spec("boosting", "core_coords", "by_type_level_q75", target="level", by_type=True,
         params=quantile_loss(0.75)),
]
# Ronda 3: mejorar el orden, no el nivel. Todo sobre la receta del 65 (cuantil 0,70).
ROUND_3_START = 67
ROUND_3 = [
    # 1. Mediana del precio y del precio por m2 de los anuncios a menos de 30 m.
    Spec("boosting", "core_coords", "by_type_level_q70_bld", target="level", by_type=True,
         params=quantile_loss(0.70), same_building=1.0),
    # 2. Promedio de 10 semillas.
    Spec("boosting", "core_coords", "by_type_level_q70_s10", target="level", by_type=True,
         params=quantile_loss(0.70), n_seeds=10),
]
# El 69 no se usa: el predictor del edificio no ayudó, así que no se promedian sus semillas.
ROUND_3B_START = 70
ROUND_3B = [
    # 3. Sigue entrenando con los inmuebles a menos de 4 km de la Zona T, tasa 0,01.
    Spec("boosting", "core_coords", "by_type_level_q70_ft", target="level", by_type=True,
         params=quantile_loss(0.70), fine_tune=True),
    # 1b. El 67 con el predictor del edificio solo en el 30 % del ajuste, como en el test.
    Spec("boosting", "core_coords", "by_type_level_q70_bld30", target="level", by_type=True,
         params=quantile_loss(0.70), same_building=0.3),
]
# Ronda 3b: información nueva sobre la receta del 65. Criterio: holdout ponderado como el
# test (mae_weighted).
ROUND_3C_START = 72
Q70 = dict(target="level", by_type=True, params=quantile_loss(0.70))
# Claves de best_params.json que escriben 05 --spec 65 y 05 --spec 4.
Q70_KEY = "boosting_core_coords_by_type_level_q70_holdout"
ELASTIC_NET_KEY = "elastic_net_full_holdout"
ROUND_3C = [
    # A. Texto completo: TF-IDF de palabras y bigramas, ajustado dentro de cada partición.
    Spec("boosting", "core_coords", "by_type_level_q70_svd20", text="svd20", **Q70),
    Spec("boosting", "core_coords", "by_type_level_q70_svd50", text="svd50", **Q70),
    Spec("boosting", "core_coords", "by_type_level_q70_svd100", text="svd100", **Q70),
    Spec("boosting", "core_coords", "by_type_level_q70_ridge", text="ridge", **Q70),
    # A'. Nombre del barrio o edificio en el título.
    Spec("boosting", "core_coords", "by_type_level_q70_names", names=True, **Q70),
    # C. Estrato de la manzana y valor de referencia catastral del terreno por m2.
    Spec("boosting", "ext_core_coords", "by_type_level_q70", **Q70),
    # D. Red por tipo con pérdida cuantílica en 0,70 y 5 semillas; luego con sintéticos SMOTER
    # de los apartamentos caros del nororiente (SyntheticTail, solo con el ajuste).
    Spec("quantile_net", "core_coords", "by_type_q70", target="level", by_type=True),
    Spec("quantile_net", "core_coords", "by_type_q70_smoter", target="level", by_type=True,
         synthetic=True),
    # B. La receta del 65 y el elastic net (fila 4) sintonizados en el holdout ponderado:
    # python 05_tune_models.py --spec 65 --n-iter 40 / --spec 4 --n-iter 15. Su MAE en el
    # holdout nororiental es optimista (se escogió ahí); valen la espacial y la de Chapinero.
    Spec("boosting", "core_coords", "by_type_level_q70_tuned", tuned=True,
         tune_key=Q70_KEY, **Q70),
    Spec("elastic_net", "full", "tuned_weighted", tuned=True, tune_key=ELASTIC_NET_KEY),
    # Combinación: lo que ayudó (A, texto con ridge; C, estrato y valor del terreno, que
    # mejora el holdout tipo Chapinero y la espacial, aunque no el nororiental). La
    # sintonización (B, fila 80) empeora el holdout tipo Chapinero y no entra.
    Spec("boosting", "ext_core_coords", "by_type_level_q70_ridge", text="ridge", **Q70),
    Spec("boosting", "ext_core_coords", "by_type_level_q70_ridge_s10", text="ridge",
         n_seeds=10, **Q70),
]
# Ronda 4: mejor lectura del texto, sobre la receta de la 75 (solo texto con ridge, una
# semilla) para medir cada paso contra ella. Criterio: holdout ponderado y validación espacial.
ROUND_4_START = 86
# Cuantil 0,70 en las otras dos librerías de boosting (pieza C).
CATBOOST_Q70 = dict(target="level", by_type=True, params=(
    ("regressor__catboostregressor__loss_function", "Quantile:alpha=0.7"),))
XGBOOST_Q70 = dict(target="level", by_type=True, params=(
    ("regressor__xgbregressor__objective", "reg:quantileerror"),
    ("regressor__xgbregressor__quantile_alpha", 0.70)))
ROUND_4 = [
    # Paso 1. Red cuantílica por tipo sobre embeddings de frases (13_text_embeddings.py):
    # en lugar del ridge, además del ridge, y la SVD de los embeddings además del ridge.
    Spec("boosting", "core_coords", "by_type_level_q70_net", text_sources=("net",), **Q70),
    Spec("boosting", "core_coords", "by_type_level_q70_ridge_net",
         text_sources=("ridge", "net"), **Q70),
    Spec("boosting", "core_coords", "by_type_level_q70_ridge_esvd20",
         text_sources=("ridge", "emb_svd20"), **Q70),
    # Paso 2. a) ridge con alfa, min_df y n-gramas escogidos en 14_tune_text_ridge.py;
    # b) un ridge por tipo; c) título y descripción con su propio ridge; d) el ridge de
    # palabras más uno de n-gramas de caracteres (3 a 5).
    Spec("boosting", "core_coords", "by_type_level_q70_ridge_tuned",
         text_sources=("ridge_tuned",), **Q70),
    Spec("boosting", "core_coords", "by_type_level_q70_ridge_type",
         text_sources=("ridge_type",), **Q70),
    Spec("boosting", "core_coords", "by_type_level_q70_ridge_title_desc",
         text_sources=("ridge_title", "ridge_desc"), **Q70),
    Spec("boosting", "core_coords", "by_type_level_q70_ridge_char",
         text_sources=("ridge", "ridge_char"), **Q70),
    # Paso 3. Lo que ayudó en algún criterio sin empeorar el otro: la red (87, −2,5 en la
    # espacial) y el ridge por tipo (90, −0,6 en el ponderado), junto al ridge.
    Spec("boosting", "core_coords", "by_type_level_q70_ridge_type_net",
         text_sources=("ridge", "ridge_type", "net"), **Q70),
    # La 93 con la receta de la 83 (estrato y valor del terreno, 10 semillas): candidata a
    # envío frente a la 83 y punto de partida de la pieza B.
    Spec("boosting", "ext_core_coords", "by_type_level_q70_ridge_type_net_s10",
         text_sources=("ridge", "ridge_type", "net"), n_seeds=10, **Q70),
    # Pieza C. CatBoost y XGBoost con la receta y los predictores de la 94, una semilla.
    Spec("catboost", "ext_core_coords", "by_type_level_q70_ridge_type_net",
         text_sources=("ridge", "ridge_type", "net"), **CATBOOST_Q70),
    Spec("xgboost", "ext_core_coords", "by_type_level_q70_ridge_type_net",
         text_sources=("ridge", "ridge_type", "net"), **XGBOOST_Q70),
    # Pieza D. La 94 con el área del lote, los pisos y el área construida del predio.
    Spec("boosting", "lot_core_coords", "by_type_level_q70_ridge_type_net_s10",
         text_sources=("ridge", "ridge_type", "net"), n_seeds=10, **Q70),
    # Pieza B. La 94 con los hiperparámetros que encontró TPE (15_optuna_search.py).
    Spec("boosting", "ext_core_coords", "by_type_level_q70_ridge_type_net_s10_tpe",
         text_sources=("ridge", "ridge_type", "net"), n_seeds=10, tuned=True,
         tune_key="optuna_tpe", **Q70),
]
# Ronda 5: la receta de la 83 en las demás familias, para compararlas en igualdad de
# condiciones. Un modelo por tipo, estrato y valor del terreno, el ridge del texto fuera de
# muestra (el mismo txt_ridge de la 83) y el precio en niveles llevado al cuantil 0,70 con
# lo que cada familia permite (models.QUANTILE_RECIPES).
ROUND_5_START = 99
R5 = dict(target="level", by_type=True, text_sources=("ridge",), recipe="q70")
ROUND_5 = [
    # Valores por defecto de cada receta: el punto de partida de 17_tune_families.py.
    Spec("random_forest", "ext_core_coords", "by_type_level_q70_ridge", **R5),
    Spec("elastic_net", "ext_core_coords", "splines_by_type_level_q70_ridge", **R5),
    Spec("linear", "ext_core_coords", "splines_by_type_level_q70_ridge", **R5),
    Spec("cart", "ext_core_coords", "by_type_level_q70_ridge", **R5),
    # Sintonizadas con 17_tune_families.py (claves '<familia>_q70').
    Spec("random_forest", "ext_core_coords", "by_type_level_q70_ridge_tuned", tuned=True,
         tune_key="random_forest_q70", **R5),
    Spec("elastic_net", "ext_core_coords", "splines_by_type_level_q70_ridge_tuned",
         tuned=True, tune_key="elastic_net_q70", **R5),
    Spec("cart", "ext_core_coords", "by_type_level_q70_ridge_tuned", tuned=True,
         tune_key="cart_q70", **R5),
]


def moved_params(key, moves):
    """Hiperparámetros de best_params.json[key] con cada ruta 'vieja' cambiada por la
    'nueva' de 'moves': para usar los valores sintonizados en la ronda 5 cuando la receta
    envuelve al pipeline."""
    params = []
    for path, value in load_best_params().get(key, {}).items():
        for old, new in moves:
            path = path.replace(old, new)
        params.append((path, value))
    return tuple(params)


# Ronda 6: acercar el bosque cuantílico (103) al boosting (83). Criterio: holdout ponderado
# y validación espacial. Todas con los hiperparámetros de la 103 salvo que se diga otra cosa.
ROUND_6_START = 106
# Hiperparámetros del mejor bosque tras el paso 2: los de la 103, porque con 5 semillas el
# genético no los supera (153,26 frente a 152,92 M, views/genetic_confirm.csv).
BEST_FOREST_KEY = "random_forest_q70"
FOREST_STEP = "regressor__randomforestquantileregressor__"
EXTRA_STEP = "regressor__extratreesquantileregressor__"
RESIDUAL_STEP = "regressor__forest__randomforestquantileregressor__"
# ExtraTrees no remuestrea filas por defecto: si hay fracción de filas, hace falta bootstrap.
EXTRA_BOOTSTRAP = (((EXTRA_STEP + "bootstrap", True),)
                   if FOREST_STEP + "max_samples" in load_best_params().get(BEST_FOREST_KEY, {})
                   else ())
RF_TUNED = dict(tuned=True, tune_key="random_forest_q70")
R6 = dict(target="level", by_type=True, recipe="q70")
ROUND_6 = [
    # Paso 1. La red sobre embeddings de la ronda 4 (fuera de muestra) en lugar del ridge
    # y junto a él.
    Spec("random_forest", "ext_core_coords", "by_type_level_q70_net_tuned",
         text_sources=("net",), **RF_TUNED, **R6),
    Spec("random_forest", "ext_core_coords", "by_type_level_q70_ridge_net_tuned",
         text_sources=("ridge", "net"), **RF_TUNED, **R6),
    # De paso: la corrección de nivel de las lineales y del CART con residuos fuera de
    # muestra (las 101, 104 y 105 con otra corrección).
    Spec("linear", "ext_core_coords", "splines_by_type_level_q70_ridge_oof",
         target="level", by_type=True, text_sources=("ridge",), recipe="q70_oof_linear"),
    Spec("elastic_net", "ext_core_coords", "splines_by_type_level_q70_ridge_tuned_oof",
         target="level", by_type=True, text_sources=("ridge",), recipe="q70_oof_elastic_net",
         params=moved_params("elastic_net_q70", [
             ("regressor__quantileshift__regressor__", "regressor__regressor__elasticnet__"),
             ("regressor__splinedesign__", "regressor__regressor__splinedesign__")])),
    Spec("cart", "ext_core_coords", "by_type_level_q70_ridge_tuned_oof",
         target="level", by_type=True, text_sources=("ridge",), recipe="q70_oof_cart",
         params=moved_params("cart_q70", [("regressor__leafquantiletree__",
                                            "regressor__regressor__leafquantiletree__")])),
    # Paso 2. La 107 (ridge y red) con los hiperparámetros del genético (19_genetic_search.py).
    Spec("random_forest", "ext_core_coords", "by_type_level_q70_ridge_net_ga",
         text_sources=("ridge", "net"), tuned=True, tune_key="random_forest_ga", **R6),
    # Paso 3, uno a la vez contra el mejor bosque hasta aquí (BEST_FOREST_KEY): a) bosque
    # sobre los residuos de un lineal, b) ExtraTrees cuantílico, c) sin lat ni lon.
    Spec("random_forest", "ext_core_coords", "by_type_level_q70_ridge_net_resid",
         text_sources=("ridge", "net"), target="level", by_type=True, recipe="q70_resid",
         params=moved_params(BEST_FOREST_KEY, [(FOREST_STEP, RESIDUAL_STEP)])),
    Spec("random_forest", "ext_core_coords", "by_type_level_q70_ridge_net_extra",
         text_sources=("ridge", "net"), target="level", by_type=True, recipe="q70_et",
         params=moved_params(BEST_FOREST_KEY, [(FOREST_STEP, EXTRA_STEP)]) + EXTRA_BOOTSTRAP),
    Spec("random_forest", "ext_core_coords", "by_type_level_q70_ridge_net_nocoords",
         text_sources=("ridge", "net"), drop_coords=True, tuned=True,
         tune_key=BEST_FOREST_KEY, **R6),
    # Combinación de lo que ayudó en algún criterio sin dañar el otro más allá del ruido:
    # ExtraTrees (113, −2,5 en el ponderado) y sin lat ni lon (114, −1,9 en la espacial).
    Spec("random_forest", "ext_core_coords", "by_type_level_q70_ridge_net_extra_nocoords",
         text_sources=("ridge", "net"), target="level", by_type=True, recipe="q70_et",
         drop_coords=True,
         params=moved_params(BEST_FOREST_KEY, [(FOREST_STEP, EXTRA_STEP)]) + EXTRA_BOOTSTRAP),
]
# Ronda 7: cuatro ideas sobre la receta de la 83, una a la vez con una semilla, contra la 82
# (la 83 con una semilla; su txt_ridge es el mismo de la fuente 'ridge'). Criterio: holdout
# ponderado y validación espacial.
ROUND_7_START = 120
R7 = dict(text_sources=("ridge",), **Q70)
ROUND_7 = [
    # A1. Área y baños de los anuncios del mismo punto, alcobas y tipo (sin precios).
    Spec("boosting", "pt_core_coords", "by_type_level_q70_ridge", **R7),
    # A4. Sin latitud ni longitud crudas, y con ellas redondeadas a 3 decimales (~100 m).
    Spec("boosting", "ext_core_coords", "by_type_level_q70_ridge_nocoords", drop_coords=True,
         **R7),
    Spec("boosting", "ext_core_coords", "by_type_level_q70_ridge_coords3", round_coords=3,
         **R7),
    # Juntas: A1 y A4 sin coordenadas (las dos bajan la espacial, 120 y 121); luego las
    # dos mejores con 10 semillas, como la 83. Las 126 a 129 son las reglas de anuncios
    # repetidos (A2 y A3) sobre la mejor, de 20_twin_rules.py.
    Spec("boosting", "pt_core_coords", "by_type_level_q70_ridge_nocoords", drop_coords=True,
         **R7),
    Spec("boosting", "pt_core_coords", "by_type_level_q70_ridge_s10", n_seeds=10, **R7),
    Spec("boosting", "pt_core_coords", "by_type_level_q70_ridge_nocoords_s10",
         drop_coords=True, n_seeds=10, **R7),
]
# Pieza B: lineal y elastic net sin ningún término de lat ni lon (la 108 y la 109 con
# SplineDesign(use_coords=False)); el elastic net con los hiperparámetros de la 104.
ROUND_7B_START = 130
ROUND_7B = [
    Spec("linear", "ext_core_coords", "by_type_level_q70_ridge_oof_nogeo",
         target="level", by_type=True, text_sources=("ridge",), recipe="q70_oof_linear_nogeo"),
    Spec("elastic_net", "ext_core_coords", "by_type_level_q70_ridge_tuned_oof_nogeo",
         target="level", by_type=True, text_sources=("ridge",),
         recipe="q70_oof_elastic_net_nogeo",
         params=moved_params("elastic_net_q70", [
             ("regressor__quantileshift__regressor__", "regressor__regressor__elasticnet__"),
             ("regressor__splinedesign__", "regressor__regressor__splinedesign__")])),
]
# Pieza C: balanceo hacia los inmuebles caros sobre la receta de la 83 (una semilla, como la
# 82). Pesos crecientes con el decil de precio (el más caro pesa 2 y 4 veces el más barato)
# y el 20 % más caro del nororiente repetido 2 y 3 veces, solo en cada ajuste. Cada
# variante con cuantil 0,55, 0,60, 0,65 y 0,70: se escoge el que deje la media en test
# más cerca de 880 M.
ROUND_7C_START = 132
BALANCE = [("pw2", dict(price_weight=2)), ("pw4", dict(price_weight=4)),
           ("os2", dict(oversample=2)), ("os3", dict(oversample=3))]
ROUND_7C = [Spec("boosting", "ext_core_coords", f"by_type_level_q{q}_ridge_{label}",
                 text_sources=("ridge",), target="level", by_type=True,
                 params=quantile_loss(q / 100), **extra)
            for label, extra in BALANCE for q in (55, 60, 65, 70)]
# Pieza B2: la red cuantílica por tipo (78) con la receta de la 83 (estrato, valor del
# terreno y el ridge del texto) y sin lat ni lon, que viajan mal a Chapinero.
ROUND_7D_START = 148
ROUND_7D = [
    Spec("quantile_net", "ext_core_coords", "by_type_q70_ridge_nocoords", target="level",
         by_type=True, text_sources=("ridge",), drop_coords=True),
]
# Pieza D: la red sobre el TF-IDF del ridge, sintonizada con NSGA-II (22_text_nets.py,
# clave 'text_net_ga'), junto al ridge dentro de la receta de la 82.
ROUND_7E_START = 149
ROUND_7E = [
    Spec("boosting", "ext_core_coords", "by_type_level_q70_ridge_tfidfnet",
         text_sources=("ridge", "tfidf_ga"), **Q70),
]
# Ronda 8: revisión de supuestos sobre la receta de la 82 (la 83 con una semilla), una idea
# a la vez. Criterio: holdout ponderado y validación espacial; menos de 2 M es ruido. Los
# números 150 a 153 son las mezclas de la ronda 7 (06_ensemble.py).
ROUND_8_START = 154
ROUND_8 = [
    # a. Otros cuantiles: insumo de la corrección de nivel por segmento (24_assumptions.py).
    *[Spec("boosting", "ext_core_coords", f"by_type_level_q{q}_ridge", text_sources=("ridge",),
           target="level", by_type=True, params=quantile_loss(q / 100)) for q in (60, 65, 75, 80)],
    # b. Modelo local: apartamentos de estrato alto del nororiente y radios alrededor del
    # centro del test (el texto se sigue ajustando con todo).
    Spec("boosting", "ext_core_coords", "by_type_level_q70_ridge_local_ne",
         local=("northeast_high", 0), **R7),
    *[Spec("boosting", "ext_core_coords", f"by_type_level_q70_ridge_local_{km}km",
           local=("radius", km * 1000), **R7) for km in (3, 5, 8)],
    # c. Sin gemelos (uno por grupo) y con los gemelos a peso 1 / tamaño del grupo.
    Spec("boosting", "ext_core_coords", "by_type_level_q70_ridge_notwins", twins="drop", **R7),
    Spec("boosting", "ext_core_coords", "by_type_level_q70_ridge_twinweight", twins="weight",
         **R7),
    # d. Sin precios por m2 fuera de los percentiles 1 y 99 de su tipo.
    Spec("boosting", "ext_core_coords", "by_type_level_q70_ridge_trim", trim=True, **R7),
    # e. Precio por m2 como objetivo con el área recuperada del mismo punto (A1, ronda 7).
    Spec("boosting", "pt_core_coords", "by_type_per_m2_q70_ridge", text_sources=("ridge",),
         target="per_m2", by_type=True, surface="surface_pt", params=quantile_loss(0.70)),
]
# Ronda 9: mejorar la red 148 (la que mejor mezcla con el boosting) y una segunda red
# distinta a propósito. Criterio: validación espacial; se reportan el ponderado y la media
# en test. Los números 166 y 167 son los apilamientos de 24_assumptions.py.
ROUND_9_START = 168
NET_148 = dict(target="level", by_type=True, drop_coords=True)
ROUND_9 = [
    # 1a. La 148 con 10 semillas (por defecto promedia 5).
    Spec("quantile_net", "ext_core_coords", "by_type_q70_ridge_nocoords_s10",
         text_sources=("ridge",), params=(("n_seeds", 10),), **NET_148),
    # 1b. La 148 con el área y los baños del mismo punto (A1, ronda 7).
    Spec("quantile_net", "pt_core_coords", "by_type_q70_ridge_nocoords",
         text_sources=("ridge",), **NET_148),
    # 2. Segunda red: una sola capa ancha y la red de texto sobre embeddings en lugar del
    # ridge, para que se equivoque en inmuebles distintos.
    Spec("quantile_net", "ext_core_coords", "by_type_q70_wide_embnet_nocoords",
         text_sources=("net",), params=(("hidden", (512,)), ("dropout", 0.2)), **NET_148),
]
# 1c. La 169 con los hiperparámetros de 17_tune_families.py --spec 169 --objective spatial
# (clave 'quantile_net_q70', buscados con una semilla). Su espacial es optimista: se escogió
# el mínimo de esas mismas particiones espaciales. Vuelve a las 5 semillas de la 148:
# las 10 de la 168 no la mejoran. Se agrega cuando existe la clave.
if "quantile_net_q70" in load_best_params():
    ROUND_9.append(Spec("quantile_net", "pt_core_coords", "by_type_q70_ridge_nocoords_tuned",
                        text_sources=("ridge",), tuned=True, tune_key="quantile_net_q70",
                        params=(("n_seeds", 5),), **NET_148))
# Ronda 10: la receta del 127 (la 125 más la regla 'mix' de gemelos de 20_twin_rules.py,
# que da la 181 a la 184) con restricciones monótonas y con hojas lineales. Criterio:
# validación espacial y después el ponderado; menos de 2 M es ruido.
ROUND_10_START = 177
R127 = dict(text_sources=("ridge",), target="level", by_type=True, drop_coords=True,
            n_seeds=10)
ROUND_10 = [
    # 1. LightGBM no admite restricciones monótonas con su pérdida cuantílica: se usa una
    # pinball propia. Control con esa pérdida y sin restricciones; luego con restricciones
    # en área, baños, parqueaderos, estrato y valor del terreno, sin y con el ridge del texto.
    Spec("boosting", "pt_core_coords", "by_type_pinball_ridge_nocoords_s10", pinball=True,
         **R127),
    Spec("boosting", "pt_core_coords", "by_type_pinball_mono_ridge_nocoords_s10",
         pinball=True, monotone=MONOTONE_FEATURES, **R127),
    Spec("boosting", "pt_core_coords", "by_type_pinball_mono_txt_ridge_nocoords_s10",
         pinball=True, monotone=MONOTONE_FEATURES + ("txt_ridge",), **R127),
    # 2. Hojas lineales con la parte lineal regularizada (linear_lambda) y los predictores
    # estandarizados. Con la pérdida cuantílica de LightGBM la penalización no actúa (209 M
    # en el nororiente con lambda de 1 a 1e7 y precios negativos en el test): se usa la
    # pinball propia, con la que lambda 100 ganó a 1 en una prueba de una semilla. Con 5
    # semillas en lugar de 10, para que la ronda cupiera en el tiempo.
    Spec("boosting", "pt_core_coords", "by_type_pinball_linear_ridge_nocoords_s5",
         pinball=True, scale=True, params=(("linear_tree", True), ("linear_lambda", 100.0)),
         **{**R127, "n_seeds": 5}),
]
# 3. kNN por tipo sobre texto, ubicación y características, con el percentil 70 de los
# vecinos. k y los pesos de los bloques salen de 28_tune_knn.py (clave 'knn_q70', con la
# espacial como objetivo, así que su espacial es optimista). Se agrega cuando existe la clave.
ROUND_10B_START = 185
ROUND_10B = []
if "knn_q70" in load_best_params():
    ROUND_10B.append(Spec("knn", "pt_core_coords", "by_type_q70_tuned", by_type=True,
                          tuned=True, tune_key="knn_q70"))
# Número de envío -> especificación.
NUMBERED_SPECS = {**dict(enumerate(SPECS, start=1)),
                  **dict(enumerate(ROUND_2, start=ROUND_2_START)),
                  **dict(enumerate(ROUND_2B, start=ROUND_2B_START)),
                  **dict(enumerate(ROUND_2C, start=ROUND_2C_START)),
                  **dict(enumerate(ROUND_3, start=ROUND_3_START)),
                  **dict(enumerate(ROUND_3B, start=ROUND_3B_START)),
                  **dict(enumerate(ROUND_3C, start=ROUND_3C_START)),
                  **dict(enumerate(ROUND_4, start=ROUND_4_START)),
                  **dict(enumerate(ROUND_5, start=ROUND_5_START)),
                  **dict(enumerate(ROUND_6, start=ROUND_6_START)),
                  **dict(enumerate(ROUND_7, start=ROUND_7_START)),
                  **dict(enumerate(ROUND_7B, start=ROUND_7B_START)),
                  **dict(enumerate(ROUND_7C, start=ROUND_7C_START)),
                  **dict(enumerate(ROUND_7D, start=ROUND_7D_START)),
                  **dict(enumerate(ROUND_7E, start=ROUND_7E_START)),
                  **dict(enumerate(ROUND_8, start=ROUND_8_START)),
                  **dict(enumerate(ROUND_9, start=ROUND_9_START)),
                  **dict(enumerate(ROUND_10, start=ROUND_10_START)),
                  **dict(enumerate(ROUND_10B, start=ROUND_10B_START))}


# Ronda 4, pieza A: fuentes de predictores de texto (models.TextPredictors). Cada una se
# calcula con particiones internas del ajuste.
TUNED_RIDGE = {k: tuple(v) if isinstance(v, list) else v
               for k, v in load_best_params().get("text_ridge", {}).items()}
TEXT_SOURCES = {
    # El txt_ridge de la fila 75: palabras y bigramas, min_df 5, alfa 3.
    "ridge": RidgeText(),
    # Paso 1: red cuantílica por tipo sobre los embeddings, y SVD de los embeddings.
    "net": EmbeddingNet(),
    "emb_svd20": EmbeddingSVD(20),
    # Paso 2: ridge sintonizado en 14_tune_text_ridge.py (clave 'text_ridge'), ridge por
    # tipo, título y descripción por separado, n-gramas de caracteres.
    "ridge_tuned": RidgeText(**TUNED_RIDGE),
    "ridge_type": RidgeText(name="txt_ridge_type", by_type=True),
    "ridge_title": RidgeText(name="txt_ridge_title", column="title_text", min_df=3),
    "ridge_desc": RidgeText(name="txt_ridge_desc", column="desc_text"),
    "ridge_char": RidgeText(name="txt_ridge_char", analyzer="char_wb", ngram_range=(3, 5),
                            max_features=300_000, alpha=10.0),
}
# Ronda 7, pieza D: red sobre el TF-IDF con los hiperparámetros del genético
# (22_text_nets.py --tune). Sin esa clave la fuente no existe y la fila 149 no corre.
if "text_net_ga" in load_best_params():
    TEXT_SOURCES["tfidf_ga"] = TextNet(name="txt_tfidf_net", **{
        k: tuple(v) if isinstance(v, list) else v
        for k, v in load_best_params()["text_net_ga"].items()})


def build_spec_model(spec, params):
    """Modelo de una especificación con sus envoltorios (vecinos, pesos, sintéticos)."""
    params = {**params, **dict(spec.params)}
    # Los conjuntos 'lux' usan el área ampliada con más patrones de texto.
    surface_col = spec.surface or ("surface_ext" if spec.features.startswith("lux")
                                   else "surface")
    if spec.family == "en_boosting":
        model = ResidualBoosting(params, luxury=spec.features.startswith("lux"))
    elif spec.family == "quantile_net":
        model = QuantileNet(**params)
        model = ByPropertyType(model) if spec.by_type else model
    elif spec.family == "knn":
        model = QuantileKNN(**params)
        model = ByPropertyType(model) if spec.by_type else model
    elif spec.pinball:
        model = PinballBoosting(increasing=spec.monotone, lgbm_params=spec.params)
        model = ByPropertyType(model) if spec.by_type else model
    elif spec.fine_tune:
        model = FocusFineTune(build_model(spec.family, params, spec.target, False, surface_col))
        model = ByPropertyType(model) if spec.by_type else model
    else:
        model = build_model(spec.family, params, spec.target, spec.by_type, surface_col,
                            spec.recipe)
    if spec.n_seeds > 1:
        model = SeedAverage(model, spec.n_seeds)
    if spec.scale:
        model = ScaledColumns(model)
    if spec.impute_area:
        model = ImputedArea(model, surface_col)
    if spec.synthetic:
        model = SyntheticTail(model)
    if spec.price_weight:
        model = PriceWeighted(model, spec.price_weight)
    if spec.oversample:
        model = TailOversample(model, copies=spec.oversample)
    if spec.truncation:
        model = TruncationCorrection(model, upper_only=spec.upper_only)
    if spec.local:
        rule, radius = spec.local
        model = LocalFit(model, rule=rule, radius_m=radius, center=test_center())
    if spec.neighbors:
        model = NeighborhoodLevel(model, drop_coords="coords" not in spec.features,
                                  surface_col=surface_col)
    if spec.same_building:
        model = SameBuilding(model, surface_col=surface_col, keep_share=spec.same_building)
    if spec.text:
        mode = "ridge" if spec.text == "ridge" else "svd"
        n_components = int(spec.text.removeprefix("svd")) if mode == "svd" else 0
        model = TextFeatures(model, mode=mode, n_components=n_components)
    if spec.names:
        model = NameLevel(model)
    if spec.drop_coords:
        model = DropColumns(model, tuple(COORD_FEATURES))
    if spec.round_coords:
        model = RoundColumns(model, tuple(COORD_FEATURES), spec.round_coords)
    if spec.text_sources:
        model = TextPredictors(model, [TEXT_SOURCES[key] for key in spec.text_sources])
    if spec.trim:
        model = TrimPricePerM2(model, surface_col=surface_col)
    if spec.twins:
        model = TwinFilter(model, spec.twins)
    return model


def spec_features(spec, sets):
    """Predictores de una especificación; los vecinos necesitan lat/lon para buscarse y los
    envoltorios de texto, el texto crudo o el nombre del título."""
    features = sets[spec.features]
    if spec.neighbors or spec.drop_coords:
        features = features + [c for c in COORD_FEATURES if c not in features]
    if spec.text or spec.family == "knn":
        features = features + ["text"]
    if spec.names:
        features = features + ["title_name"]
    if spec.text_sources:
        inputs = ["text"] + [c for key in spec.text_sources
                             for c in TEXT_SOURCES[key].inputs()]
        features = features + list(dict.fromkeys(inputs))
    return features


def fit_predict(spec, params, X_fit, y_fit, X_pred, weights=None):
    """Ajusta una copia del modelo y predice."""
    model = build_spec_model(spec, params)
    fit_model(model, X_fit, y_fit, weights)
    return model.predict(X_pred)


def evaluate(spec, params, X, y, splits, weights=None):
    """Predicciones para las filas de validación de cada partición (NaN en el resto).
    Las familias de un solo núcleo ajustan sus particiones en paralelo."""
    n_jobs = len(splits) if spec.family in SINGLE_CORE_FAMILIES else 1
    fold_preds = Parallel(n_jobs=n_jobs)(
        delayed(fit_predict)(spec, params, X.iloc[fit_idx], y.iloc[fit_idx], X.iloc[val_idx],
                             None if weights is None else weights[fit_idx])
        for fit_idx, val_idx in splits)
    pred = np.full(len(X), np.nan)
    for (_, val_idx), fold_pred in zip(splits, fold_preds):
        pred[val_idx] = fold_pred
    return pred


def update_table(path, new, key):
    """Agrega columnas o filas nuevas a una tabla guardada, reemplazando las repetidas."""
    if not path.exists():
        return new
    previous = pd.read_csv(path)
    if key == "columns":
        keep = [c for c in previous.columns if c not in new.columns]
        return pd.concat([new, previous[keep]], axis=1)
    # Una fila repetida conserva los valores anteriores de los esquemas que no se corrieron.
    repeated = previous["model"].isin(new["model"])
    merged = new.set_index("model").combine_first(previous[repeated].set_index("model"))
    return pd.concat([previous[~repeated], merged.reset_index()])


def main():
    parser = argparse.ArgumentParser()
    # El híbrido de 33 combina dos familias; la red cuantílica es la red de la ronda 3;
    # CatBoost y XGBoost son la pieza C de la ronda 4.
    families = FAMILIES + ["en_boosting", "quantile_net", "catboost", "xgboost", "knn"]
    parser.add_argument("--families", nargs="+", default=families, choices=families)
    parser.add_argument("--specs", nargs="+", type=int, help="números de SPECS a correr")
    parser.add_argument("--schemes", nargs="+", default=list(SCHEMES), choices=list(SCHEMES),
                        help="esquemas de validación; los demás conservan su resultado")
    parser.add_argument("--no-submission", action="store_true",
                        help="no reentrena con todo el train ni escribe el envío")
    args = parser.parse_args()

    train, test = load_processed()
    selected = [spec for number, spec in NUMBERED_SPECS.items()
                if spec.family in args.families and (not args.specs or number in args.specs)]
    if any(c.startswith("emb_") for spec in selected
           for key in spec.text_sources for c in TEXT_SOURCES[key].inputs()):
        train, test = add_embeddings(train, test)
    y = train[TARGET]
    sets = feature_sets(train.columns)
    best_params = load_best_params()
    weights = similarity_weights(train["test_similarity"])
    schemes = {scheme: make_splits(train, test) for scheme, make_splits in SCHEMES.items()
               if scheme in args.schemes}
    if "northeast" in schemes:
        val_idx, groups = northeast_building_groups(train, test, y)

    oof_table = train[[ID_COL, TARGET]].copy()
    test_table = test[[ID_COL]].copy()
    for number, spec in NUMBERED_SPECS.items():
        if spec.family not in args.families or (args.specs and number not in args.specs):
            continue
        start = time.time()
        name = spec.name(number)
        features = spec_features(spec, sets)
        key = spec.tune_key or f"{spec.family}_{spec.features}"
        params = best_params.get(key, {}) if spec.tuned else {}
        w = weights if spec.weighted else None

        row = {"model": name, "family": spec.family, "features": spec.features,
               "n_features": len(features), "tuned": bool(params), "target": spec.target,
               "weighted": spec.weighted, "by_type": spec.by_type}
        for scheme, splits in schemes.items():
            pred = evaluate(spec, params, train[features], y, splits, w)
            oof_table[f"{name}__{scheme}"] = pred
            rows_eval = ~np.isnan(pred)
            summary = error_summary(y[rows_eval], pred[rows_eval])
            row[f"mae_{scheme}"] = summary["mae"]
            row[f"bias_{scheme}"] = summary["bias"]
            row[f"share_over_{scheme}"] = summary["share_overpredicted"]
            if scheme == "northeast":
                # Ponderado como el test; en los modelos con SameBuilding no es comparable
                # (ver 10_same_building.py).
                row["mae_weighted"] = weighted_mae(y.iloc[val_idx], pred[val_idx], groups)

        # Ajuste final con todo el entrenamiento y envío a Kaggle.
        test_columns = [ID_COL]
        if not args.no_submission:
            test_pred = fit_predict(spec, params, train[features], y, test[features], w)
            test_table[name] = test_pred
            test_columns.append(name)
            row["submission"] = save_submission(test[ID_COL], test_pred, name).name
            row["test_mean"], row["test_p95"], row["test_max"] = (
                test_pred.mean(), np.quantile(test_pred, 0.95), test_pred.max())
        row["minutes"] = (time.time() - start) / 60
        print(f"{name:<40} " + " | ".join(
            f"{scheme} {row[f'mae_{scheme}']/1e6:6.1f} M (sesgo {row[f'bias_{scheme}']/1e6:+.1f})"
            for scheme in schemes) + f" | {row['minutes']:.1f} min", flush=True)
        # Se guarda tras cada especificación: si una corrida se corta, no se pierde lo hecho.
        spec_columns = [c for c in oof_table if c.startswith(f"{name}__")]
        save_results(row, oof_table[[ID_COL, TARGET] + spec_columns], test_table[test_columns])


@contextmanager
def results_lock():
    """Candado de archivo: dos corridas simultáneas no escriben las tablas a la vez."""
    lock = PROCESSED_DIR / "results.lock"
    while True:
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL)
            break
        except FileExistsError:
            # Un candado de más de 5 minutos es de una corrida que murió: se libera.
            if lock.exists() and time.time() - lock.stat().st_mtime > 300:
                lock.unlink(missing_ok=True)
            time.sleep(1)
    try:
        yield
    finally:
        os.close(handle)
        os.remove(lock)


def save_results(row, oof_part, test_part):
    """Actualiza la tabla de resultados y las predicciones guardadas de una especificación;
    las demás especificaciones conservan su resultado anterior."""
    VIEWS_DIR.mkdir(exist_ok=True)
    results_path = VIEWS_DIR / "cv_results.csv"
    with results_lock():
        update_table(results_path, pd.DataFrame([row]), "rows").sort_values(
            "model").to_csv(results_path, index=False)
        for table, file in [(oof_part, "oof_predictions.csv"),
                            (test_part, "test_predictions.csv")]:
            path = PROCESSED_DIR / file
            update_table(path, table, "columns").to_csv(path, index=False)


if __name__ == "__main__":
    main()
