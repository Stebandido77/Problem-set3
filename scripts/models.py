"""Registro de modelos: una receta por familia de algoritmos exigida en el taller.

Cada entrada define (i) el estimador con hiperparámetros iniciales razonables y
(ii) la distribución de búsqueda que usa 05_tune_models.py.
Todos los modelos se entrenan sobre log(precio) y predicen en pesos: con MAE como
métrica conviene apuntar a la mediana, y el logaritmo reduce el peso de la cola derecha.
"""

import json

import numpy as np
from lightgbm import LGBMRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from utils import BEST_PARAMS_PATH, SEED


def impute():
    """Mediana + indicador de faltante. Va dentro del pipeline para no filtrar información
    de validación al entrenamiento."""
    return SimpleImputer(strategy="median", add_indicator=True)


def log_target(regressor):
    """Entrena sobre log(precio) y devuelve predicciones en pesos."""
    return TransformedTargetRegressor(regressor=regressor, func=np.log, inverse_func=np.exp)


def build_model(family, params=None):
    """Devuelve el estimador de una familia; 'params' sobrescribe los valores iniciales."""
    params = params or {}
    if family == "linear":
        steps = make_pipeline(impute(), StandardScaler(), LinearRegression())
    elif family == "elastic_net":
        # Interacciones de segundo orden: el elastic net decide cuáles sobreviven.
        steps = make_pipeline(
            impute(), StandardScaler(),
            PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
            StandardScaler(),
            ElasticNet(alpha=0.003, l1_ratio=0.5, max_iter=5000, random_state=SEED),
        )
    elif family == "cart":
        steps = make_pipeline(
            impute(),
            DecisionTreeRegressor(max_depth=12, min_samples_leaf=20, random_state=SEED),
        )
    elif family == "random_forest":
        steps = make_pipeline(
            impute(),
            RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.4,
                                  n_jobs=-1, random_state=SEED),
        )
    elif family == "boosting":
        # LightGBM maneja los faltantes de forma nativa; pérdida L1 = la métrica de Kaggle.
        steps = make_pipeline(
            LGBMRegressor(objective="l1", n_estimators=1500, learning_rate=0.03,
                          num_leaves=63, min_child_samples=20, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
                          n_jobs=-1, random_state=SEED, verbose=-1),
        )
    elif family == "neural_net":
        steps = make_pipeline(
            impute(), StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=1e-3,
                         batch_size=256, early_stopping=True, n_iter_no_change=15,
                         max_iter=300, random_state=SEED),
        )
    else:
        raise ValueError(f"Familia desconocida: {family}")
    model = log_target(steps)
    model.set_params(**params)
    return model


# Espacios de búsqueda aleatoria. El prefijo es la ruta del parámetro dentro del pipeline.
SEARCH_SPACES = {
    "elastic_net": {
        "regressor__elasticnet__alpha": loguniform(1e-4, 1e-1),
        "regressor__elasticnet__l1_ratio": uniform(0.05, 0.95),
    },
    "cart": {
        "regressor__decisiontreeregressor__max_depth": randint(4, 25),
        "regressor__decisiontreeregressor__min_samples_leaf": randint(5, 100),
        "regressor__decisiontreeregressor__ccp_alpha": loguniform(1e-6, 1e-3),
    },
    "random_forest": {
        "regressor__randomforestregressor__max_features": uniform(0.2, 0.6),
        "regressor__randomforestregressor__min_samples_leaf": randint(1, 15),
        "regressor__randomforestregressor__max_depth": [None, 15, 25],
    },
    "boosting": {
        "regressor__lgbmregressor__num_leaves": randint(15, 128),
        "regressor__lgbmregressor__learning_rate": loguniform(0.01, 0.1),
        "regressor__lgbmregressor__n_estimators": randint(400, 2500),
        "regressor__lgbmregressor__min_child_samples": randint(5, 80),
        "regressor__lgbmregressor__colsample_bytree": uniform(0.4, 0.5),
        "regressor__lgbmregressor__reg_lambda": loguniform(0.1, 20),
    },
    "neural_net": {
        "regressor__mlpregressor__hidden_layer_sizes": [(64,), (128, 64), (256, 128, 64),
                                                         (64, 64, 64)],
        "regressor__mlpregressor__alpha": loguniform(1e-5, 1e-1),
        "regressor__mlpregressor__learning_rate_init": loguniform(3e-4, 1e-2),
    },
}

FAMILIES = ["linear", "elastic_net", "cart", "random_forest", "boosting", "neural_net"]


def load_best_params():
    """Hiperparámetros encontrados por 05_tune_models.py (vacío si aún no se ha corrido)."""
    if BEST_PARAMS_PATH.exists():
        return json.loads(BEST_PARAMS_PATH.read_text())
    return {}
