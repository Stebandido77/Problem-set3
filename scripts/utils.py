"""Utilidades compartidas del Problem Set 3: rutas, carga de datos, métricas,
esquemas de validación cruzada (aleatoria y espacial) y escritura de envíos a Kaggle."""

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.neighbors import BallTree

SEED = 4107
EARTH_RADIUS_M = 6_371_000.0
# Holdout tipo Chapinero: radio alrededor del test y franja excluida del entrenamiento.
# Calibrados para que el mejor envío sin OSM (boosting con coordenadas) dé un MAE cercano
# a su puntaje público en Kaggle (180 M): con 250 m y 400 m da 182 M.
HOLDOUT_RADIUS_M = 250
HOLDOUT_BUFFER_M = 400

# --- Rutas (plantilla del curso: document / scripts / stores / views) ---
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "stores" / "raw"
PROCESSED_DIR = ROOT / "stores" / "processed"
SUBMISSIONS_DIR = ROOT / "stores" / "submissions"
VIEWS_DIR = ROOT / "views"
OSM_POIS_PATH = PROCESSED_DIR / "osm_pois.csv"
BEST_PARAMS_PATH = PROCESSED_DIR / "best_params.json"

ID_COL = "property_id"
TARGET = "price"

# --- Conjuntos de predictores ---
STRUCTURAL_FEATURES = [
    "is_house", "bedrooms", "bathrooms", "rooms", "surface", "log_surface", "surface_missing",
    "time_index",
]
TEXT_FEATURES = [
    "txt_parking", "txt_n_parking", "txt_terrace", "txt_balcony", "txt_elevator",
    "txt_storage", "txt_fireplace", "txt_security", "txt_gym", "txt_pool",
    "txt_gated", "txt_remodeled", "txt_duplex", "txt_penthouse", "txt_view",
    "txt_stratum", "txt_floor", "txt_desc_length",
]
# Segunda tanda de predictores de texto: gama alta y antigüedad.
EXTRA_TEXT_FEATURES = [
    "txt_service_room", "txt_drivers_room", "txt_walk_in", "txt_study", "txt_luxury",
    "txt_spa", "txt_exterior", "txt_age",
]
COORD_FEATURES = ["lat", "lon"]
SIMILARITY_FEATURES = STRUCTURAL_FEATURES + TEXT_FEATURES
LANDMARK_FEATURES = [
    "dist_zona_t_m", "dist_parque_93_m", "dist_centro_internacional_m",
    "dist_plaza_bolivar_m",
]
CORRIDOR_FEATURES = ["dist_carrera_7_m", "dist_cerros_m"]


def haversine_m(lat1, lon1, lat2, lon2):
    """Distancia de círculo máximo en metros (acepta arreglos)."""
    radius = 6_371_000.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * radius * np.arcsin(np.sqrt(a))


def load_raw():
    """Lee train y test tal como vienen de Kaggle."""
    train = pd.read_csv(RAW_DIR / "train.csv")
    test = pd.read_csv(RAW_DIR / "test.csv")
    return train, test


def load_processed():
    """Lee las bases con predictores construidas por 02_build_features.py."""
    train = pd.read_csv(PROCESSED_DIR / "train_features.csv")
    test = pd.read_csv(PROCESSED_DIR / "test_features.csv")
    return train, test


def osm_feature_names(columns, extra=False):
    """Predictores de OpenStreetMap presentes en la base (vacío si no se descargó OSM).
    Por defecto: distancias y conteos a 500 m; con extra=True, los conteos a 300 m y 1 km."""
    extra_suffixes = ("_300m", "_1000m")
    return sorted(c for c in columns if c.startswith("osm_")
                  and c.endswith(extra_suffixes) == extra)


def feature_sets(columns):
    """Diccionario nombre -> lista de predictores, de menos a más información."""
    osm = osm_feature_names(columns)
    full = STRUCTURAL_FEATURES + TEXT_FEATURES + LANDMARK_FEATURES + osm
    rich = (full + EXTRA_TEXT_FEATURES + CORRIDOR_FEATURES
            + osm_feature_names(columns, extra=True))
    sets = {
        "base": STRUCTURAL_FEATURES,
        "base_text": STRUCTURAL_FEATURES + TEXT_FEATURES,
        # 'full' no usa lat/lon crudos: solo distancias, que viajan mejor a Chapinero.
        "full": full,
        "full_coords": full + COORD_FEATURES,
        # 'rich' suma señales de gama alta del texto, ejes de ciudad y conteos de OSM.
        "rich": rich,
        "rich_coords": rich + COORD_FEATURES,
    }
    return sets


def random_cv(n_splits=5):
    """Validación cruzada estándar (particiones aleatorias)."""
    return KFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def spatial_groups(df, n_clusters=10):
    """Bloques espaciales: k-means sobre lat/lon. Cada bloque es una 'zona' de la ciudad."""
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=SEED)
    return km.fit_predict(df[["lat", "lon"]].to_numpy())


def spatial_cv_splits(df, n_splits=5, n_clusters=10):
    """Validación cruzada espacial: zonas completas quedan fuera del entrenamiento,
    imitando el salto del resto de Bogotá a Chapinero."""
    groups = spatial_groups(df, n_clusters)
    return list(GroupKFold(n_splits=n_splits).split(df, groups=groups))


def distance_to_nearest_m(points, reference):
    """Distancia en metros de cada fila de 'points' al punto más cercano de 'reference'."""
    tree = BallTree(np.radians(reference[["lat", "lon"]].to_numpy()), metric="haversine")
    distance, _ = tree.query(np.radians(points[["lat", "lon"]].to_numpy()), k=1)
    return distance[:, 0] * EARTH_RADIUS_M


def test_similarity(train, test):
    """Probabilidad fuera de muestra de que cada fila de train 'parezca' del test, según un
    clasificador train-contra-test. No usa coordenadas (separan casi perfecto: AUC ~1),
    así que mide parecido en tipo, tamaño y texto del anuncio. No usa el precio."""
    columns = SIMILARITY_FEATURES
    X = pd.concat([train[columns], test[columns]], ignore_index=True)
    is_test = np.r_[np.zeros(len(train)), np.ones(len(test))]
    proba = np.zeros(len(X))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fit_idx, pred_idx in folds.split(X, is_test):
        clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                             random_state=SEED, verbose=-1)
        clf.fit(X.iloc[fit_idx], is_test[fit_idx])
        proba[pred_idx] = clf.predict_proba(X.iloc[pred_idx])[:, 1]
    return proba[:len(train)]


def similarity_weights(similarity, cap_quantile=0.99):
    """Pesos de importancia p / (1 - p) a partir del parecido al test, recortados en el
    percentil 99 para que pocas filas no dominen, y normalizados a media 1."""
    p = np.clip(np.asarray(similarity, dtype=float), 1e-3, 1 - 1e-3)
    odds = p / (1 - p)
    odds = np.minimum(odds, np.quantile(odds, cap_quantile))
    return odds / odds.mean()


def chapinero_holdout(train, test, radius_m=HOLDOUT_RADIUS_M, buffer_m=HOLDOUT_BUFFER_M):
    """Partición única que imita el salto a Chapinero.

    Validación: inmuebles de train a menos de 'radius_m' de algún inmueble del test.
    Entrenamiento: el resto de train, menos una franja de 'buffer_m' alrededor de la
    validación, para que el modelo no aprenda de los vecinos inmediatos (como pasa con el
    interior de Chapinero). Devuelve (índices de entrenamiento, índices de validación)."""
    holdout = distance_to_nearest_m(train, test) < radius_m
    gap = distance_to_nearest_m(train, train[holdout]) <= buffer_m
    return np.flatnonzero(~holdout & ~gap), np.flatnonzero(holdout)


def error_summary(y_true, y_pred):
    """MAE y asimetría del error. Sobrepredecir es el error caro (caso Zillow)."""
    err = np.asarray(y_pred) - np.asarray(y_true)
    return {
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "share_overpredicted": float(np.mean(err > 0)),
        "mean_overprediction": float(err[err > 0].mean()) if (err > 0).any() else 0.0,
    }


def submission_path(out, name):
    """Ruta libre para un envío. Un envío ya escrito no se sobrescribe: si el contenido
    cambió, se guarda como <name>_v2, <name>_v3...; si es idéntico, se reutiliza."""
    candidates = [SUBMISSIONS_DIR / f"{name}.csv"]
    candidates += [SUBMISSIONS_DIR / f"{name}_v{k}.csv" for k in range(2, 100)]
    for path in candidates:
        if not path.exists():
            return path
        existing = pd.read_csv(path)
        if (list(existing[ID_COL]) == list(out[ID_COL])
                and np.array_equal(existing[TARGET].to_numpy(float), out[TARGET].to_numpy(float))):
            return path
    raise RuntimeError(f"Demasiadas versiones de {name}")


def save_submission(ids, predictions, name):
    """Escribe un envío en el formato de Kaggle (property_id, price) y devuelve su ruta."""
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({ID_COL: ids, TARGET: np.round(predictions, 0)})
    template = pd.read_csv(RAW_DIR / "submission_template.csv")
    assert list(out[ID_COL]) == list(template[ID_COL]), "El orden de ids no coincide"
    assert out[TARGET].notna().all() and (out[TARGET] > 0).all()
    path = submission_path(out, name)
    if not path.exists():
        out.to_csv(path, index=False)
    return path
