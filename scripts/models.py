"""Registro de modelos: una receta por familia de algoritmos exigida en el taller.

Cada entrada define (i) el estimador con hiperparámetros iniciales razonables y
(ii) la distribución de búsqueda que usa 05_tune_models.py.
Por defecto los modelos se entrenan sobre log(precio) y predicen en pesos: con MAE como
métrica conviene apuntar a la mediana, y el logaritmo reduce el peso de la cola derecha.
Como alternativas hay precio en nivel y log(precio por m2) cuando se conoce el área.
"""

import json

import numpy as np
from lightgbm import LGBMRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, RegressorMixin, clone
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


def level_target(regressor):
    """Entrena directamente sobre el precio en pesos (sin transformar)."""
    return TransformedTargetRegressor(regressor=regressor)


def fit_model(model, X, y, weights=None):
    """Ajusta un modelo con pesos de muestra opcionales, sin importar cómo esté envuelto."""
    if weights is None:
        return model.fit(X, y)
    if isinstance(model, TransformedTargetRegressor):
        last_step = model.regressor.steps[-1][0]
        return model.fit(X, y, **{f"{last_step}__sample_weight": weights})
    return model.fit(X, y, sample_weight=weights)


class PricePerSquareMeter(RegressorMixin, BaseEstimator):
    """Predice log(precio por m2) cuando hay área y lo multiplica por el área.
    Los inmuebles sin área usan un modelo de respaldo sobre log(precio)."""

    def __init__(self, per_m2, fallback, surface_col="surface"):
        self.per_m2 = per_m2
        self.fallback = fallback
        self.surface_col = surface_col

    def fit(self, X, y, sample_weight=None):
        has_area = X[self.surface_col].notna().to_numpy()
        y = np.asarray(y, dtype=float)
        w = None if sample_weight is None else np.asarray(sample_weight)[has_area]
        self.per_m2_ = fit_model(clone(self.per_m2), X[has_area],
                                 y[has_area] / X.loc[has_area, self.surface_col], w)
        self.fallback_ = fit_model(clone(self.fallback), X, y, sample_weight)
        return self

    def predict(self, X):
        pred = self.fallback_.predict(X)
        has_area = X[self.surface_col].notna().to_numpy()
        if has_area.any():
            pred[has_area] = (self.per_m2_.predict(X[has_area])
                              * X.loc[has_area, self.surface_col].to_numpy())
        return pred


class ByPropertyType(RegressorMixin, BaseEstimator):
    """Un modelo para apartamentos y otro para casas (el test es 97 % apartamentos)."""

    def __init__(self, model, type_col="is_house"):
        self.model = model
        self.type_col = type_col

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y, dtype=float)
        self.models_ = {}
        for kind in (0, 1):
            rows = (X[self.type_col] == kind).to_numpy()
            w = None if sample_weight is None else np.asarray(sample_weight)[rows]
            self.models_[kind] = fit_model(clone(self.model), X[rows], y[rows], w)
        return self

    def predict(self, X):
        pred = np.zeros(len(X))
        for kind, model in self.models_.items():
            rows = (X[self.type_col] == kind).to_numpy()
            if rows.any():
                pred[rows] = model.predict(X[rows])
        return pred


def build_model(family, params=None, target="log", by_type=False):
    """Devuelve el estimador de una familia.

    params: sobrescribe los hiperparámetros iniciales (rutas 'regressor__<paso>__<param>').
    target: 'log' (log del precio), 'level' (precio en pesos) o 'per_m2' (log del precio
            por m2 cuando hay área, con respaldo en log del precio).
    by_type: si es True, ajusta un modelo aparte para apartamentos y otro para casas."""
    params = params or {}
    if target == "per_m2":
        model = PricePerSquareMeter(build_model(family, params), build_model(family, params))
    else:
        steps = family_pipeline(family)
        model = log_target(steps) if target == "log" else level_target(steps)
        model.set_params(**params)
    return ByPropertyType(model) if by_type else model


def family_pipeline(family):
    """Pipeline de preprocesamiento + estimador con los hiperparámetros iniciales."""
    if family == "linear":
        steps = make_pipeline(impute(), StandardScaler(), LinearRegression())
    elif family == "elastic_net":
        # Interacciones de segundo orden: el elastic net decide cuáles sobreviven.
        # tol=1e-3 corta el tiempo de ajuste a la mitad sin mover el MAE.
        steps = make_pipeline(
            impute(), StandardScaler(),
            PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
            StandardScaler(),
            ElasticNet(alpha=0.003, l1_ratio=0.5, max_iter=5000, tol=1e-3,
                       random_state=SEED),
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
    return steps


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
