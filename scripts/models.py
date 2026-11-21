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
from joblib import Memory
from lightgbm import LGBMRegressor
from scipy.special import ndtr, ndtri
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import BallTree, NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from utils import (BEST_PARAMS_PATH, EARTH_RADIUS_M, EXTENDED_AREA, LUXURY_FEATURES, SEED,
                   TEXT_CACHE_DIR, TWIN_COLUMNS, embedding_columns, northeast_holdout,
                   transductive_reference, twin_groups)

# Caché en disco de los predictores de texto ajustados (no se versiona; se puede borrar).
TEXT_CACHE = Memory(TEXT_CACHE_DIR, verbose=0)


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


def same_values(a, b):
    """Igualdad elemento a elemento que cuenta dos faltantes como iguales."""
    return (a == b) | (np.isnan(a) & np.isnan(b))


class SameBuilding(RegressorMixin, BaseEstimator):
    """Agrega la mediana del precio y del precio por m2 de los anuncios de referencia a menos
    de 'radius_m' (el mismo edificio) y cuántos son; las medianas quedan vacías si no hay.

    La referencia es el ajuste: el precio de la validación nunca entra. Se excluyen la propia
    fila y sus gemelos, los anuncios a menos de 'twin_m' con la misma área, alcobas y baños
    (el mismo inmueble publicado otra vez); en el ajuste, también los que tienen el mismo
    precio. X debe traer lat y lon.
    Con 'keep_share' < 1, en el ajuste solo esa fracción de filas (al azar) conserva los
    predictores y el resto queda como sin edificio, para que la cobertura del ajuste se
    parezca a la del test (78 % frente a 24 %)."""

    TWIN_COLUMNS = ["surface", "bedrooms", "bathrooms"]

    def __init__(self, model, radius_m=30, twin_m=1, surface_col="surface", keep_share=1.0):
        self.model = model
        self.radius_m = radius_m
        self.twin_m = twin_m
        self.surface_col = surface_col
        self.keep_share = keep_share

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y, dtype=float)
        self.reference_ = self.reference(X, y)
        X_fit = self.with_building(X, self.reference_, own_price=y, own_pos=np.arange(len(X)))
        hidden = np.random.default_rng(SEED).random(len(X)) >= self.keep_share
        X_fit.loc[hidden, ["bld_price", "bld_price_m2"]] = np.nan
        X_fit.loc[hidden, "bld_count"] = 0
        self.model_ = fit_model(clone(self.model), X_fit, y, sample_weight)
        return self

    def reference(self, X, y):
        """Anuncios de referencia: árbol de coordenadas, llaves de gemelo, precio y área."""
        return {"tree": BallTree(np.radians(X[["lat", "lon"]].to_numpy()), metric="haversine"),
                "twin": X[self.TWIN_COLUMNS].to_numpy(float),
                "price": np.asarray(y, dtype=float),
                "area": X[self.surface_col].to_numpy(float)}

    def with_building(self, X, reference, own_price=None, own_pos=None):
        """X con los tres predictores del edificio. 'own_pos' es la posición de cada fila
        dentro de la referencia (para no contarse a sí misma) y 'own_price' su precio."""
        coords = np.radians(X[["lat", "lon"]].to_numpy())
        idx, dist = reference["tree"].query_radius(
            coords, r=self.radius_m / EARTH_RADIUS_M, return_distance=True)
        rows = np.repeat(np.arange(len(X)), [len(i) for i in idx])
        cols = np.concatenate(idx).astype(int)
        close = np.concatenate(dist) * EARTH_RADIUS_M < self.twin_m
        own_keys = X[self.TWIN_COLUMNS].to_numpy(float)[rows]
        drop = close & same_values(own_keys, reference["twin"][cols]).all(axis=1)
        if own_price is not None:
            drop |= close & (reference["price"][cols] == own_price[rows])
        if own_pos is not None:
            drop |= cols == own_pos[rows]
        rows, cols = rows[~drop], cols[~drop]
        price = reference["price"][cols]
        stats = pd.DataFrame({"row": rows, "price": price,
                              "per_m2": price / reference["area"][cols]}).groupby("row")
        building = pd.DataFrame(index=np.arange(len(X)))
        building["bld_price"] = stats["price"].median()
        building["bld_price_m2"] = stats["per_m2"].median()
        building["bld_count"] = stats.size()
        building["bld_count"] = building["bld_count"].fillna(0)
        return X.assign(**{c: building[c].to_numpy() for c in building})

    def predict(self, X):
        return self.model_.predict(self.with_building(X, self.reference_))


def building_count(X_query, X_ref, y_ref, own_price=None, own_pos=None):
    """Número de anuncios del mismo edificio de cada fila, con las reglas de SameBuilding."""
    finder = SameBuilding(model=None)
    reference = finder.reference(X_ref, y_ref)
    return finder.with_building(X_query, reference, own_price, own_pos)["bld_count"].to_numpy()


def northeast_building_groups(train, test, y):
    """Grupos del holdout ponderado como el test (utils.weighted_mae): las filas de
    validación con anuncios del mismo edificio en la referencia transductiva, las que no los
    tienen en el ajuste, y la fracción del test que los tiene en train. Devuelve también los
    índices de validación."""
    columns = ["lat", "lon"] + SameBuilding.TWIN_COLUMNS
    fit_idx, val_idx = northeast_holdout(train, test)
    ref_idx, own_pos = transductive_reference(y, fit_idx, val_idx)
    X = train[columns]
    with_building = building_count(X.iloc[val_idx], X.iloc[ref_idx], y.iloc[ref_idx],
                                   own_price=y.iloc[val_idx].to_numpy(float),
                                   own_pos=own_pos) > 0
    without_building = building_count(X.iloc[val_idx], X.iloc[fit_idx], y.iloc[fit_idx]) == 0
    share = (building_count(test[columns], X, y) > 0).mean()
    return val_idx, (with_building, without_building, share)


class TextFeatures(RegressorMixin, BaseEstimator):
    """Predictores del texto completo del anuncio (columna 'text'), ajustados solo con las
    filas de ajuste. TF-IDF de palabras y bigramas y, según 'mode':
    - 'svd': los primeros 'n_components' componentes de una SVD truncada (txt_svd_k);
    - 'ridge': un ridge del log del precio sobre el TF-IDF (txt_ridge). En el ajuste, cada
      fila recibe la predicción fuera de muestra de un ridge que no vio su grupo de gemelos,
      así que su propio precio nunca entra en su predictor."""

    def __init__(self, model, mode="svd", n_components=50, text_col="text", alpha=3.0):
        self.model = model
        self.mode = mode
        self.n_components = n_components
        self.text_col = text_col
        self.alpha = alpha

    def _vectorizer(self):
        return TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=100_000,
                               sublinear_tf=True, strip_accents="unicode", dtype=np.float32)

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y, dtype=float)
        self.tfidf_ = self._vectorizer().fit(X[self.text_col])
        tfidf = self.tfidf_.transform(X[self.text_col])
        if self.mode == "svd":
            self.svd_ = TruncatedSVD(self.n_components, random_state=SEED).fit(tfidf)
            features = self._svd_features(tfidf)
        else:
            self.ridge_ = Ridge(alpha=self.alpha).fit(tfidf, np.log(y))
            features = {"txt_ridge": self._out_of_fold_ridge(X, tfidf, np.log(y))}
        X_fit = X.drop(columns=self.text_col).assign(**features)
        self.model_ = fit_model(clone(self.model), X_fit, y, sample_weight)
        return self

    def _svd_features(self, tfidf):
        components = self.svd_.transform(tfidf)
        return {f"txt_svd_{k}": components[:, k] for k in range(self.n_components)}

    def _out_of_fold_ridge(self, X, tfidf, log_y):
        out = np.zeros(len(log_y))
        folds = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        for fit_idx, pred_idx in folds.split(tfidf, groups=twin_groups(X)):
            ridge = Ridge(alpha=self.alpha).fit(tfidf[fit_idx], log_y[fit_idx])
            out[pred_idx] = ridge.predict(tfidf[pred_idx])
        return out

    def transform_text(self, X):
        tfidf = self.tfidf_.transform(X[self.text_col])
        features = (self._svd_features(tfidf) if self.mode == "svd"
                    else {"txt_ridge": self.ridge_.predict(tfidf)})
        return X.drop(columns=self.text_col).assign(**features)

    def predict(self, X):
        return self.model_.predict(self.transform_text(X))


RAW_TEXT_COLUMNS = ["text", "title_text", "desc_text"]


class RidgeText(BaseEstimator):
    """Fuente de texto de TextPredictors: ridge del log del precio sobre el TF-IDF de una
    columna. Con 'by_type', un ridge para apartamentos y otro para casas (el vocabulario es
    común). Con los valores por defecto es el txt_ridge de TextFeatures (fila 75)."""

    def __init__(self, name="txt_ridge", column="text", analyzer="word", ngram_range=(1, 2),
                 min_df=5, max_features=100_000, alpha=3.0, by_type=False):
        self.name = name
        self.column = column
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.alpha = alpha
        self.by_type = by_type

    def inputs(self):
        return [self.column]

    def _kinds(self, X):
        """Etiqueta de cada fila: el tipo de inmueble o una sola clase."""
        return X["is_house"].to_numpy() if self.by_type else np.zeros(len(X))

    def _fit_ridges(self, tfidf, log_y, kinds):
        return {kind: Ridge(alpha=self.alpha).fit(tfidf[kinds == kind], log_y[kinds == kind])
                for kind in np.unique(kinds)}

    @staticmethod
    def _predict(ridges, tfidf, kinds):
        out = np.zeros(tfidf.shape[0])
        for kind, ridge in ridges.items():
            if (kinds == kind).any():
                out[kinds == kind] = ridge.predict(tfidf[kinds == kind])
        return out

    def fit_features(self, X, log_y, folds):
        """Ajusta con todo el ajuste y devuelve el predictor fuera de muestra de sus filas."""
        self.tfidf_ = TfidfVectorizer(analyzer=self.analyzer, ngram_range=self.ngram_range,
                                      min_df=self.min_df, max_features=self.max_features,
                                      sublinear_tf=True, strip_accents="unicode",
                                      dtype=np.float32).fit(X[self.column])
        tfidf = self.tfidf_.transform(X[self.column])
        kinds = self._kinds(X)
        self.ridges_ = self._fit_ridges(tfidf, log_y, kinds)
        out = np.zeros(len(log_y))
        for fit_idx, pred_idx in folds:
            ridges = self._fit_ridges(tfidf[fit_idx], log_y[fit_idx], kinds[fit_idx])
            out[pred_idx] = self._predict(ridges, tfidf[pred_idx], kinds[pred_idx])
        return {self.name: out}

    def transform(self, X):
        tfidf = self.tfidf_.transform(X[self.column])
        return {self.name: self._predict(self.ridges_, tfidf, self._kinds(X))}


def embedding_inputs():
    """Columnas de embeddings de título y descripción (utils.add_embeddings)."""
    return embedding_columns("title") + embedding_columns("desc")


class EmbeddingNet(BaseEstimator):
    """Fuente de texto de TextPredictors: red con pérdida cuantílica ('alpha') sobre los
    embeddings de título y descripción, una por tipo de inmueble. Cada fila del ajuste
    recibe la predicción de las redes de las particiones que no la vieron; las filas nuevas,
    el promedio de esas redes. Sin coordenadas, el 10 % de parada temprana de QuantileNet se
    escoge sin juntar gemelos: la parada puede ser algo optimista dentro de cada ajuste
    interno, sin efecto sobre la validación externa."""

    def __init__(self, name="txt_net", alpha=0.70, hidden=(256, 64), dropout=0.2,
                 n_seeds=1):
        self.name = name
        self.alpha = alpha
        self.hidden = hidden
        self.dropout = dropout
        self.n_seeds = n_seeds

    def inputs(self):
        return embedding_inputs()

    def _net(self):
        return ByPropertyType(QuantileNet(alpha=self.alpha, hidden=self.hidden,
                                          dropout=self.dropout, n_seeds=self.n_seeds))

    def fit_features(self, X, log_y, folds):
        Z = X[self.inputs() + ["is_house"]]
        y = np.exp(log_y)
        out = np.zeros(len(y))
        self.nets_ = []
        for fit_idx, pred_idx in folds:
            net = self._net().fit(Z.iloc[fit_idx], y[fit_idx])
            out[pred_idx] = net.predict(Z.iloc[pred_idx])
            self.nets_.append(net)
        return {self.name: np.log(np.clip(out, 1e7, None))}

    def transform(self, X):
        Z = X[self.inputs() + ["is_house"]]
        pred = np.mean([net.predict(Z) for net in self.nets_], axis=0)
        return {self.name: np.log(np.clip(pred, 1e7, None))}


class EmbeddingSVD(BaseEstimator):
    """Fuente de texto de TextPredictors: los primeros 'n_components' componentes de una SVD
    de los embeddings de título y descripción, ajustada con las filas de ajuste. No usa el
    precio, así que no necesita particiones."""

    def __init__(self, n_components=20):
        self.n_components = n_components

    def inputs(self):
        return embedding_inputs()

    def fit_features(self, X, log_y, folds):
        self.svd_ = TruncatedSVD(self.n_components, random_state=SEED).fit(X[self.inputs()])
        return self.transform(X)

    def transform(self, X):
        components = self.svd_.transform(X[self.inputs()])
        return {f"emb_svd_{k}": components[:, k] for k in range(self.n_components)}


def text_code_version():
    """Huella del código que define los predictores de texto y de la semilla: entra en la
    clave de la caché para que un cambio en esas clases no devuelva resultados viejos."""
    import hashlib
    import inspect
    import utils
    code = "".join(inspect.getsource(obj) for obj in (
        RidgeText, EmbeddingNet, EmbeddingSVD, QuantileNet, ByPropertyType, utils.twin_groups))
    return hashlib.sha256(f"{code}{SEED}{utils.TWIN_DISTANCE_M}".encode()).hexdigest()


@TEXT_CACHE.cache
def fit_text_sources(sources, X, y, code_version):
    """Ajusta las fuentes de texto con 5 particiones internas (gemelos juntos) y devuelve
    las fuentes ajustadas y los predictores fuera de muestra de las filas de X. Se guarda en
    caché: el mismo ajuste (mismas filas, precios, fuentes y versión del código) no se
    recalcula entre corridas ni entre modelos que solo cambian el boosting."""
    log_y = np.log(y)
    folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(
        X, groups=twin_groups(X)))
    fitted = [clone(source) for source in sources]
    features = {}
    for source in fitted:
        features.update(source.fit_features(X, log_y, folds))
    return fitted, features


class TextPredictors(RegressorMixin, BaseEstimator):
    """Varios predictores de texto ('sources': RidgeText, EmbeddingNet, EmbeddingSVD) que
    alimentan al modelo. Todas las fuentes comparten las mismas 5 particiones internas del
    ajuste, con los gemelos (mismo inmueble o mismo texto) siempre juntos, así que el
    predictor de una fila de ajuste nunca vio su precio. Las columnas crudas de texto y de
    embeddings no llegan al modelo."""

    def __init__(self, model, sources=()):
        self.model = model
        self.sources = sources

    def _drop_raw(self, X):
        raw = [c for c in X.columns if c in RAW_TEXT_COLUMNS or c.startswith("emb_")]
        return X.drop(columns=raw)

    def fit_text(self, X, y):
        """Ajusta las fuentes y devuelve el ajuste con sus predictores fuera de muestra."""
        # Solo las columnas que leen las fuentes y los gemelos: así la caché sirve aunque
        # cambien los demás predictores.
        inputs = ["is_house", "lat", "lon", "text"] + TWIN_COLUMNS + [
            c for source in self.sources for c in source.inputs()]
        self.sources_, features = fit_text_sources(
            tuple(self.sources), X[list(dict.fromkeys(inputs))], np.asarray(y, dtype=float),
            text_code_version())
        return self._drop_raw(X).assign(**features)

    def fit(self, X, y, sample_weight=None):
        self.model_ = fit_model(clone(self.model), self.fit_text(X, y), y, sample_weight)
        return self

    def transform_text(self, X):
        features = {}
        for source in self.sources_:
            features.update(source.transform(X))
        return self._drop_raw(X).assign(**features)

    def predict(self, X):
        return self.model_.predict(self.transform_text(X))


class NameLevel(RegressorMixin, BaseEstimator):
    """Nivel de precio del nombre de barrio o edificio del título ('title_name', de
    02_build_features.py): media del log del precio de los anuncios de ajuste con el mismo
    nombre y cuántos son. En el ajuste, cada fila excluye a su grupo de gemelos (y a sí
    misma); con menos de 'min_count' anuncios el nivel queda vacío."""

    def __init__(self, model, name_col="title_name", min_count=5):
        self.model = model
        self.name_col = name_col
        self.min_count = min_count

    def fit(self, X, y, sample_weight=None):
        log_y = np.log(np.asarray(y, dtype=float))
        frame = pd.DataFrame({"name": X[self.name_col].to_numpy(), "twin": twin_groups(X),
                              "log_y": log_y})
        self.levels_ = frame.groupby("name")["log_y"].agg(["sum", "count"])
        own = frame.groupby(["name", "twin"])["log_y"].transform("sum")
        own_count = frame.groupby(["name", "twin"])["log_y"].transform("count")
        total = frame["name"].map(self.levels_["sum"])
        count = frame["name"].map(self.levels_["count"]) - own_count
        X_fit = self._assign(X, (total - own) / count, count)
        self.model_ = fit_model(clone(self.model), X_fit, y, sample_weight)
        return self

    def _assign(self, X, level, count):
        level = np.where(count >= self.min_count, level, np.nan)
        return X.drop(columns=self.name_col).assign(
            name_log_price=level, name_count=np.nan_to_num(np.asarray(count, dtype=float)))

    def predict(self, X):
        names = X[self.name_col]
        level = names.map(self.levels_["sum"] / self.levels_["count"]).to_numpy()
        return self.model_.predict(self._assign(X, level, names.map(self.levels_["count"])))


class QuantileNet(RegressorMixin, BaseEstimator):
    """Red neuronal (PyTorch) con pérdida cuantílica (pinball) en 'alpha' sobre el precio en
    nivel, promediada en 'n_seeds' semillas. Faltantes con la mediana más indicador y
    entradas estandarizadas, todo ajustado solo con las filas de ajuste. Parada temprana con
    un 10 % del ajuste que deja juntos a los gemelos."""

    def __init__(self, alpha=0.70, hidden=(128, 64), dropout=0.1, lr=3e-3, batch_size=2048,
                 max_epochs=150, patience=10, n_seeds=5, weight_decay=1e-4):
        self.alpha = alpha
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.n_seeds = n_seeds
        self.weight_decay = weight_decay

    SCALE = 1e9  # el precio se entrena en miles de millones

    def fit(self, X, y, sample_weight=None):
        import torch
        torch.set_num_threads(4)
        self.prep_ = make_pipeline(impute(), StandardScaler()).fit(X)
        Z = torch.tensor(self.prep_.transform(X), dtype=torch.float32)
        target = torch.tensor(np.asarray(y, dtype=float) / self.SCALE, dtype=torch.float32)
        groups = twin_groups(X) if {"lat", "lon"} <= set(X.columns) else np.arange(len(X))
        early = np.isin(groups, np.random.default_rng(SEED).choice(
            np.unique(groups), len(np.unique(groups)) // 10, replace=False))
        self.nets_ = [self._train(Z, target, torch.tensor(early), SEED + k)
                      for k in range(self.n_seeds)]
        return self

    def _network(self, n_inputs):
        import torch.nn as nn
        layers, width = [], n_inputs
        for size in self.hidden:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(self.dropout)]
            width = size
        return nn.Sequential(*layers, nn.Linear(width, 1))

    def _pinball(self, pred, target):
        diff = target - pred
        return (self.alpha * diff.clamp(min=0) + (1 - self.alpha) * (-diff).clamp(min=0)).mean()

    def _train(self, Z, target, early, seed):
        import copy
        import torch
        torch.manual_seed(seed)
        net = self._network(Z.shape[1])
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        Z_fit, y_fit, Z_stop, y_stop = Z[~early], target[~early], Z[early], target[early]
        best, best_state, waited = np.inf, None, 0
        generator = torch.Generator().manual_seed(seed)
        for _ in range(self.max_epochs):
            net.train()
            for batch in torch.randperm(len(Z_fit), generator=generator).split(self.batch_size):
                optimizer.zero_grad()
                loss = self._pinball(net(Z_fit[batch]).squeeze(1), y_fit[batch])
                loss.backward()
                optimizer.step()
            net.eval()
            with torch.no_grad():
                stop_loss = self._pinball(net(Z_stop).squeeze(1), y_stop).item()
            if stop_loss < best - 1e-5:
                best, best_state, waited = stop_loss, copy.deepcopy(net.state_dict()), 0
            else:
                waited += 1
                if waited >= self.patience:
                    break
        net.load_state_dict(best_state)
        return net.eval()

    def predict(self, X):
        import torch
        Z = torch.tensor(self.prep_.transform(X), dtype=torch.float32)
        with torch.no_grad():
            preds = [net(Z).squeeze(1).numpy() for net in self.nets_]
        return np.mean(preds, axis=0) * self.SCALE


class SeedAverage(RegressorMixin, BaseEstimator):
    """Promedio de 'n_seeds' copias del modelo con semillas SEED, SEED + 1, ..."""

    def __init__(self, model, n_seeds=10):
        self.model = model
        self.n_seeds = n_seeds

    def fit(self, X, y, sample_weight=None):
        self.models_ = []
        for k in range(self.n_seeds):
            model = clone(self.model)
            model.set_params(**{key: SEED + k for key in model.get_params()
                                if key.endswith("random_state")})
            self.models_.append(fit_model(model, X, y, sample_weight))
        return self

    def predict(self, X):
        return np.mean([model.predict(X) for model in self.models_], axis=0)


class FocusFineTune(RegressorMixin, BaseEstimator):
    """Boosting entrenado con todo el ajuste que después sigue entrenando (init_model) solo
    con los inmuebles a menos de 'focus_m' de la Zona T, con una tasa de aprendizaje menor.
    'model' es un boosting envuelto en TransformedTargetRegressor con objetivo en nivel.
    Si hay menos de 'min_rows' inmuebles en la zona, se queda con el modelo general."""

    def __init__(self, model, focus_col="dist_zona_t_m", focus_m=4000, n_estimators=300,
                 learning_rate=0.01, min_rows=500):
        self.model = model
        self.focus_col = focus_col
        self.focus_m = focus_m
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_rows = min_rows

    def fit(self, X, y, sample_weight=None):
        self.model_ = fit_model(clone(self.model), X, y, sample_weight)
        focus = (X[self.focus_col] < self.focus_m).to_numpy()
        self.tuned_ = None
        if focus.sum() >= self.min_rows:
            general = self.model_.regressor_.steps[-1][1]
            self.tuned_ = clone(general).set_params(n_estimators=self.n_estimators,
                                                    learning_rate=self.learning_rate)
            self.tuned_.fit(X[focus], np.asarray(y, dtype=float)[focus],
                            init_model=general.booster_)
        return self

    def predict(self, X):
        return self.model_.predict(X) if self.tuned_ is None else self.tuned_.predict(X)


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
    'cv' particiones), así que no usa nada de la validación externa.

    Con 'upper_only' solo se corrige el techo (a = -inf) y las filas cuya solución queda
    pegada al borde de la búsqueda conservan la predicción original: sin eso, la fórmula
    manda las predicciones cercanas al piso o al techo a los bordes (110 M y 4.481 M)."""

    def __init__(self, model, cv=3, max_lift=1.0, upper_only=False):
        self.model = model
        self.cv = cv
        self.max_lift = max_lift   # tope de la corrección en log (1.0 = multiplicar por 2,7)
        self.upper_only = upper_only

    def fit(self, X, y, sample_weight=None):
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
        a, b = self.bounds_
        if self.upper_only:
            a = -np.inf
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
        mu = (lo + hi) / 2
        if self.upper_only:
            at_edge = (mu <= m - 1.0 + 1e-6) | (mu >= m + self.max_lift - 1e-6)
            mu = np.where(at_edge, m, mu)
        return np.exp(mu)


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
    elif family == "catboost":
        # Ronda 4, pieza C: la misma receta que el LightGBM con otro algoritmo de boosting
        # (árboles simétricos). Faltantes nativos; pérdida L1 por defecto.
        from catboost import CatBoostRegressor
        steps = make_pipeline(
            CatBoostRegressor(loss_function="MAE", iterations=2000, learning_rate=0.06,
                              depth=8, l2_leaf_reg=3.0, thread_count=-1,
                              random_state=SEED, verbose=0, allow_writing_files=False),
        )
    elif family == "xgboost":
        from xgboost import XGBRegressor
        steps = make_pipeline(
            XGBRegressor(objective="reg:absoluteerror", n_estimators=1500,
                         learning_rate=0.03, max_depth=8, min_child_weight=5,
                         subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
                         tree_method="hist", n_jobs=-1, random_state=SEED),
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
