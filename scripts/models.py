"""Registro de modelos: una receta por familia de algoritmos exigida en el taller.

Cada entrada define (i) el estimador con hiperparámetros iniciales razonables y
(ii) la distribución de búsqueda que usa 05_tune_models.py.
Por defecto los modelos se entrenan sobre log(precio) y predicen en pesos: con MAE como
métrica conviene apuntar a la mediana, y el logaritmo reduce el peso de la cola derecha.
Como alternativas hay precio en nivel y log(precio por m2) cuando se conoce el área.
"""

import json

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.neighbors import BallTree, NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from utils import BEST_PARAMS_PATH, EXTENDED_AREA, LUXURY_FEATURES, SEED


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


class PriceWeighted(RegressorMixin, BaseEstimator):
    """Pondera el entrenamiento con pesos crecientes en el decil de precio (de 1 a
    'max_weight'), para darle más voz a la cola alta. Los deciles salen del precio de las
    filas de ajuste, así que no usa nada de la validación."""

    def __init__(self, model, max_weight=3.0):
        self.model = model
        self.max_weight = max_weight

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y, dtype=float)
        decile = np.searchsorted(np.quantile(y, np.linspace(0.1, 0.9, 9)), y)
        weights = 1 + (self.max_weight - 1) * decile / 9
        self.model_ = fit_model(clone(self.model), X, y, weights)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class NeighborhoodLevel(RegressorMixin, BaseEstimator):
    """Agrega como predictores la mediana del log(precio por m2) y del log(precio) de los
    k inmuebles de ajuste más cercanos. Se calcula dentro de cada ajuste: cada fila de
    entrenamiento usa sus vecinos sin contarse a sí misma y las de predicción usan solo
    vecinos del entrenamiento, así que el precio de la validación nunca entra.
    X debe traer lat y lon; si 'drop_coords', se quitan antes del modelo interno."""

    def __init__(self, model, k=20, drop_coords=True, surface_col="surface"):
        self.model = model
        self.k = k
        self.drop_coords = drop_coords
        self.surface_col = surface_col

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y, dtype=float)
        has_area = X[self.surface_col].notna().to_numpy()
        coords = np.radians(X[["lat", "lon"]].to_numpy())
        area = X.loc[has_area, self.surface_col].to_numpy()
        self.references_ = {
            "nbr_log_price": (BallTree(coords, metric="haversine"), np.log(y)),
            "nbr_log_price_m2": (BallTree(coords[has_area], metric="haversine"),
                                 np.log(y[has_area] / area)),
        }
        # Posición de cada fila de ajuste dentro de cada árbol (-1 si no está).
        self_index = {"nbr_log_price": np.arange(len(X)),
                      "nbr_log_price_m2": np.where(has_area, np.cumsum(has_area) - 1, -1)}
        X_fit = self._with_neighbors(X, self_index)
        self.model_ = fit_model(clone(self.model), X_fit, y, sample_weight)
        return self

    def _with_neighbors(self, X, self_index=None):
        coords = np.radians(X[["lat", "lon"]].to_numpy())
        features = {}
        for name, (tree, values) in self.references_.items():
            _, idx = tree.query(coords, k=min(self.k + 1, len(values)))
            if self_index is None:
                idx = idx[:, :self.k]
            else:
                # Se descarta la propia fila; si no apareció, se descarta el vecino k+1.
                own = idx == self_index[name][:, None]
                own[~own.any(axis=1), -1] = True
                idx = idx[~own].reshape(len(X), -1)
            features[name] = np.median(values[idx], axis=1)
        X = X.assign(**features)
        return X.drop(columns=["lat", "lon"]) if self.drop_coords else X

    def predict(self, X):
        return self.model_.predict(self._with_neighbors(X))


class SyntheticTail(RegressorMixin, BaseEstimator):
    """Sobremuestreo tipo SMOTER de los apartamentos caros del nororiente.

    Toma las filas de ajuste con precio en el cuantil 'quantile' o más y a menos de
    'max_dist_m' de la Zona T, y crea 'ratio' sintéticos por cada una interpolando con uno
    de sus k vecinos (en predictores estandarizados): x = x_i + u (x_j - x_i) y el log del
    precio igual. Solo usa las filas de ajuste de cada partición; los sintéticos nunca
    llegan a la validación. Por construcción no generan precios por encima del máximo."""

    def __init__(self, model, quantile=0.9, max_dist_m=4000, ratio=1.0, k=5):
        self.model = model
        self.quantile = quantile
        self.max_dist_m = max_dist_m
        self.ratio = ratio
        self.k = k

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y, dtype=float)
        rare = ((y >= np.quantile(y, self.quantile))
                & (X["dist_zona_t_m"] <= self.max_dist_m).to_numpy()
                & (X["is_house"] == 0).to_numpy())
        Xr = X[rare].astype(float)
        filled = Xr.fillna(Xr.median()).fillna(0)
        scaled = (filled - filled.mean()) / filled.std().replace(0, 1)
        k = min(self.k + 1, len(Xr))
        _, idx = NearestNeighbors(n_neighbors=k).fit(scaled).kneighbors(scaled)
        rng = np.random.default_rng(SEED)
        n_new = int(self.ratio * len(Xr))
        base = rng.integers(len(Xr), size=n_new)
        mate = idx[base, rng.integers(1, k, size=n_new)]
        u = rng.random(n_new)[:, None]
        xi, xj = Xr.to_numpy()[base], Xr.to_numpy()[mate]
        synthetic = np.where(np.isnan(xj), xi, xi + u * (xj - xi))
        binary = [c for c in X.columns if set(np.unique(Xr[c].dropna())) <= {0, 1}]
        cols = [X.columns.get_loc(c) for c in binary]
        synthetic[:, cols] = np.where(u < 0.5, xi[:, cols], xj[:, cols])
        log_y = np.log(y[rare])
        y_new = np.exp(log_y[base] + u[:, 0] * (log_y[mate] - log_y[base]))
        X_all = pd.concat([X, pd.DataFrame(synthetic, columns=X.columns)], ignore_index=True)
        self.n_synthetic_ = n_new
        self.model_ = fit_model(clone(self.model), X_all, np.concatenate([y, y_new]))
        return self

    def predict(self, X):
        return self.model_.predict(X)


class ImputedArea(RegressorMixin, BaseEstimator):
    """Completa el área faltante con un boosting de log(área) sobre alcobas, baños, cuartos,
    tipo y texto, ajustado solo con las filas de ajuste que sí tienen área. Deja el
    indicador de faltante intacto para que el modelo sepa qué área es imputada."""

    PREDICTORS = ["is_house", "bedrooms", "bathrooms", "rooms", "txt_n_parking",
                  "txt_stratum", "txt_floor", "txt_desc_length", "txt_penthouse",
                  "txt_duplex", "txt_terrace", "txt_service_room", "txt_study",
                  "dist_zona_t_m", "dist_parque_93_m"]

    def __init__(self, model, surface_col="surface"):
        self.model = model
        self.surface_col = surface_col

    def _predictors(self, X):
        return X[[c for c in self.PREDICTORS if c in X]]

    def _fill(self, X):
        missing = X[self.surface_col].isna().to_numpy()
        X = X.copy()
        if missing.any():
            area = np.exp(self.area_model_.predict(self._predictors(X[missing])))
            X.loc[missing, self.surface_col] = area
            X.loc[missing, f"log_{self.surface_col}"] = np.log(area)
        return X

    def fit(self, X, y, sample_weight=None):
        has_area = X[self.surface_col].notna().to_numpy()
        self.area_model_ = LGBMRegressor(objective="l1", n_estimators=400, learning_rate=0.05,
                                         num_leaves=31, random_state=SEED, verbose=-1)
        self.area_model_.fit(self._predictors(X[has_area]),
                             np.log(X.loc[has_area, self.surface_col]))
        self.model_ = fit_model(clone(self.model), self._fill(X), y, sample_weight)
        return self

    def predict(self, X):
        return self.model_.predict(self._fill(X))


class TruncationCorrection(RegressorMixin, BaseEstimator):
    """Corrige la predicción por el truncamiento del precio de entrenamiento.

    El train solo trae precios entre un mínimo y un máximo (300 M y 1.650 M), así que un
    modelo que apunta a la mediana predice la mediana *truncada*: nunca pasa del máximo y se
    queda corto justo en los inmuebles caros. Si log(precio) | x es normal con escala sigma,
    la mediana truncada m y la mediana sin truncar mu cumplen
        m = mu + sigma * Phi^-1((Phi((a - mu)/sigma) + Phi((b - mu)/sigma)) / 2),
    con [a, b] el rango del log-precio en el ajuste. Se despeja mu para cada fila. Lejos del
    tope mu es casi igual a m; cerca del tope sube. Los límites salen del precio de las filas
    de ajuste y sigma de residuos fuera de muestra dentro del mismo ajuste (validación de
    'cv' particiones), así que no usa nada de la validación externa."""

    def __init__(self, model, cv=3, max_lift=1.0):
        self.model = model
        self.cv = cv
        self.max_lift = max_lift   # tope de la corrección en log (1.0 = multiplicar por 2,7)

    def fit(self, X, y, sample_weight=None):
        from sklearn.model_selection import KFold
        y = np.asarray(y, dtype=float)
        self.bounds_ = (np.log(y.min()), np.log(y.max()))
        residuals = np.empty(len(y))
        for fit_idx, val_idx in KFold(self.cv, shuffle=True, random_state=SEED).split(X):
            inner = fit_model(clone(self.model), X.iloc[fit_idx], y[fit_idx])
            residuals[val_idx] = np.log(y[val_idx]) - np.log(inner.predict(X.iloc[val_idx]))
        self.sigma_ = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
        self.model_ = fit_model(clone(self.model), X, y, sample_weight)
        return self

    def _truncated_median(self, mu):
        from scipy.special import ndtr, ndtri
        a, b = self.bounds_
        s = self.sigma_
        return mu + s * ndtri((ndtr((a - mu) / s) + ndtr((b - mu) / s)) / 2)

    def predict(self, X):
        m = np.log(np.clip(self.model_.predict(X), np.exp(self.bounds_[0]),
                           np.exp(self.bounds_[1]) * 0.999))
        lo, hi = m - 1.0, m + self.max_lift
        for _ in range(60):   # bisección: la mediana truncada crece con mu
            mid = (lo + hi) / 2
            below = self._truncated_median(mid) < m
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        return np.exp((lo + hi) / 2)


class ResidualBoosting(RegressorMixin, BaseEstimator):
    """Elastic net sobre log(precio) con pocos predictores que pueden extrapolar (área,
    baños, parqueaderos, gradientes de ubicación) y boosting sobre sus residuos con todos
    los predictores. La parte lineal no tiene techo: puede pasar de 1.650 M si el área lo
    justifica; el boosting corrige alrededor de ella."""

    LINEAR_FEATURES = ["log_surface", "surface_missing", "bathrooms", "bedrooms",
                       "txt_n_parking", "txt_stratum", "is_house", "txt_penthouse",
                       "dist_zona_t_m", "dist_parque_93_m", "dist_centro_internacional_m",
                       "dist_plaza_bolivar_m"]

    def __init__(self, boosting_params=None, luxury=False):
        self.boosting_params = boosting_params
        self.luxury = luxury   # parte lineal con el área ampliada y las señales de lujo

    def fit(self, X, y, sample_weight=None):
        log_y = np.log(np.asarray(y, dtype=float))
        self.linear_ = make_pipeline(impute(), StandardScaler(),
                                     ElasticNet(alpha=1e-3, l1_ratio=0.5, random_state=SEED))
        self.linear_.fit(self._linear_X(X), log_y)
        residual = log_y - self.linear_.predict(self._linear_X(X))
        self.boosting_ = family_pipeline("boosting").set_params(
            **(self.boosting_params or {}))
        self.boosting_.fit(X, residual)
        return self

    def _linear_X(self, X):
        features = self.LINEAR_FEATURES
        if self.luxury:
            features = [EXTENDED_AREA.get(c, c) for c in features] + LUXURY_FEATURES
        X = X[features].copy()
        for col in [c for c in X if c.startswith("dist_")]:
            X[col] = np.log1p(X[col])
        return X

    def predict(self, X):
        return np.exp(self.linear_.predict(self._linear_X(X)) + self.boosting_.predict(X))


def build_model(family, params=None, target="log", by_type=False, surface_col="surface"):
    """Devuelve el estimador de una familia.

    params: sobrescribe los hiperparámetros iniciales (rutas 'regressor__<paso>__<param>').
    target: 'log' (log del precio), 'level' (precio en pesos) o 'per_m2' (log del precio
            por m2 cuando hay área, con respaldo en log del precio).
    by_type: si es True, ajusta un modelo aparte para apartamentos y otro para casas."""
    params = params or {}
    if target == "per_m2":
        model = PricePerSquareMeter(build_model(family, params), build_model(family, params),
                                    surface_col)
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
