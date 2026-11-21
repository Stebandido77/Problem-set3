"""Utilidades compartidas del Problem Set 3: rutas, carga de datos, métricas,
esquemas de validación cruzada (aleatoria y espacial) y escritura de envíos a Kaggle."""

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.neighbors import BallTree

SEED = 4107
EARTH_RADIUS_M = 6_371_000.0
# Holdout tipo Chapinero: radio alrededor del test y franja excluida del entrenamiento.
# Calibrados para que el mejor envío sin OSM (boosting con coordenadas) dé un MAE cercano
# a su puntaje público en Kaggle (180 M): con 250 m y 400 m da 182 M.
HOLDOUT_RADIUS_M = 250
HOLDOUT_BUFFER_M = 400
# Holdout nororiental (el criterio actual): se valida en el train a menos de 250 m del test,
# sin franja, y se entrena solo con inmuebles de hasta 1.000 M, imitando que el train real
# está truncado en 1.650 M y el test quizá no. Es el esquema que reproduce el orden de los
# cinco puntajes públicos y las diferencias entre ellos (08_calibrate_holdout.py).
NORTHEAST_RADIUS_M = 250
NORTHEAST_PRICE_CAP = 1.0e9

# --- Rutas (plantilla del curso: document / scripts / stores / views) ---
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "stores" / "raw"
PROCESSED_DIR = ROOT / "stores" / "processed"
SUBMISSIONS_DIR = ROOT / "stores" / "submissions"
# Envíos reemplazados por una versión corregida (_v2) o con un error conocido (46-48, 52).
OBSOLETE_DIR = SUBMISSIONS_DIR / "obsoletos"
VIEWS_DIR = ROOT / "views"
OSM_POIS_PATH = PROCESSED_DIR / "osm_pois.csv"
BEST_PARAMS_PATH = PROCESSED_DIR / "best_params.json"
BOGOTA_BLOCKS_PATH = PROCESSED_DIR / "bogota_blocks.csv"
BOGOTA_LOTS_PATH = PROCESSED_DIR / "bogota_lots.csv"
EMBEDDINGS_PATH = PROCESSED_DIR / "text_embeddings.npz"
TEXT_CACHE_DIR = PROCESSED_DIR / "text_cache"
EMBEDDING_DIM = 384   # dimensión de multilingual-e5-small (13_text_embeddings.py)

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
# Tercera tanda: señales de lujo para reconocer inmuebles por encima del tope del train.
LUXURY_FEATURES = [
    "txt_private_terrace", "txt_private_elevator", "txt_club_house", "txt_double_height",
    "txt_view_hills", "txt_penthouse_strict", "txt_stratum_6", "txt_parking_3plus",
    "txt_lux_score",
]
# Área recuperada con más patrones de texto; reemplaza a 'surface' en los conjuntos 'lux'.
EXTENDED_AREA = {"surface": "surface_ext", "surface_missing": "surface_ext_missing",
                 "log_surface": "log_surface_ext"}
COORD_FEATURES = ["lat", "lon"]
SIMILARITY_FEATURES = STRUCTURAL_FEATURES + TEXT_FEATURES
LANDMARK_FEATURES = [
    "dist_zona_t_m", "dist_parque_93_m", "dist_centro_internacional_m",
    "dist_plaza_bolivar_m",
]
CORRIDOR_FEATURES = ["dist_carrera_7_m", "dist_cerros_m"]
# Datos abiertos de Bogotá por manzana (11_download_bogota.py): estrato y valor del terreno.
EXTERNAL_FEATURES = ["ext_stratum", "ext_land_value_m2"]
# Catastro por lote (16_download_lots.py): área del lote, pisos y área construida del predio.
LOT_FEATURES = ["ext_lot_area_m2", "ext_floors", "ext_built_area_m2"]


def haversine_m(lat1, lon1, lat2, lon2):
    """Distancia de círculo máximo en metros (acepta arreglos)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def load_raw():
    """Lee train y test tal como vienen de Kaggle."""
    train = pd.read_csv(RAW_DIR / "train.csv")
    test = pd.read_csv(RAW_DIR / "test.csv")
    return train, test


def load_processed():
    """Lee las bases con predictores construidas por 02_build_features.py, más el texto
    crudo del anuncio (título y descripción) en la columna 'text' para los modelos que lo
    convierten en predictores dentro de cada partición; 'title_text' y 'desc_text' traen
    cada parte por separado."""
    train = pd.read_csv(PROCESSED_DIR / "train_features.csv")
    test = pd.read_csv(PROCESSED_DIR / "test_features.csv")
    raw_train, raw_test = load_raw()
    for df, raw in [(train, raw_train), (test, raw_test)]:
        text = raw["title"].fillna("") + " " + raw["description"].fillna("")
        df["text"] = df[ID_COL].map(dict(zip(raw[ID_COL], text)))
        for column, source in [("title_text", "title"), ("desc_text", "description")]:
            df[column] = df[ID_COL].map(dict(zip(raw[ID_COL], raw[source].fillna(""))))
    return train, test


def embedding_columns(prefix):
    """Nombres de las columnas de embeddings ('emb_title_k' o 'emb_desc_k')."""
    return [f"emb_{prefix}_{k}" for k in range(EMBEDDING_DIM)]


def add_embeddings(*frames):
    """Agrega a cada base los embeddings de título y descripción de 13_text_embeddings.py,
    alineados por property_id. No usan el precio."""
    stored = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    position = pd.Series(np.arange(len(stored["ids"])), index=stored["ids"])
    out = []
    for df in frames:
        rows = position.loc[df[ID_COL]].to_numpy()
        blocks = [pd.DataFrame(stored[part][rows], columns=embedding_columns(part),
                               index=df.index) for part in ("title", "desc")]
        out.append(pd.concat([df] + blocks, axis=1))
    return out


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
    lux = [EXTENDED_AREA.get(c, c) for c in full] + LUXURY_FEATURES
    sets = {
        "base": STRUCTURAL_FEATURES,
        "base_text": STRUCTURAL_FEATURES + TEXT_FEATURES,
        # 'full' no usa lat/lon crudos: solo distancias, que viajan mejor a Chapinero.
        "full": full,
        "full_coords": full + COORD_FEATURES,
        # 'rich' suma señales de gama alta del texto, ejes de ciudad y conteos de OSM.
        "rich": rich,
        "rich_coords": rich + COORD_FEATURES,
        # 'lux' es 'full' con el área ampliada y las señales de lujo.
        "lux": lux,
        "lux_coords": lux + COORD_FEATURES,
        # 'core' es 'full' sin OSM: los envíos 01 a 11 se subieron así a Kaggle.
        "core": [c for c in full if c not in osm],
        "core_coords": [c for c in full if c not in osm] + COORD_FEATURES,
        # 'ext_core' es 'core' con el estrato y el valor del terreno de la manzana.
        "ext_core_coords": ([c for c in full if c not in osm] + EXTERNAL_FEATURES
                            + COORD_FEATURES),
        # 'lot_core_coords' es 'ext_core_coords' con el lote y la construcción del predio.
        "lot_core_coords": ([c for c in full if c not in osm] + EXTERNAL_FEATURES
                            + [c for c in LOT_FEATURES if c in columns] + COORD_FEATURES),
        # 'lux_core' es 'core' con el área ampliada y las señales de lujo.
        "lux_core_coords": ([EXTENDED_AREA.get(c, c) for c in full if c not in osm]
                            + LUXURY_FEATURES + COORD_FEATURES),
    }
    return sets


def random_cv(n_splits=5):
    """Validación cruzada estándar (particiones aleatorias)."""
    return KFold(n_splits=n_splits, shuffle=True, random_state=SEED)


TWIN_DISTANCE_M = 1
TWIN_COLUMNS = ["surface", "bedrooms", "bathrooms"]


def twin_groups(df):
    """Grupo de gemelos de cada fila: anuncios a menos de 1 m con la misma área, alcobas y
    baños (dos faltantes cuentan como iguales) y, si df trae la columna 'text', anuncios con
    el mismo texto normalizado (reanuncios con otra ubicación o estructura). Se unen de
    forma transitiva; una fila sin gemelos forma su propio grupo. No usa el precio."""
    coords = np.radians(df[["lat", "lon"]].to_numpy())
    pairs = BallTree(coords, metric="haversine").query_radius(
        coords, r=TWIN_DISTANCE_M / EARTH_RADIUS_M)
    rows = np.repeat(np.arange(len(df)), [len(p) for p in pairs])
    cols = np.concatenate(pairs)
    keys = df[TWIN_COLUMNS].to_numpy(float)
    same = ((keys[rows] == keys[cols]) | (np.isnan(keys[rows]) & np.isnan(keys[cols])))
    rows, cols = rows[same.all(axis=1)], cols[same.all(axis=1)]
    if "text" in df.columns:
        # Cada texto repetido se une con su primera aparición.
        normalized = df["text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        first = pd.Series(np.arange(len(df))).groupby(normalized.to_numpy()).transform("first")
        rows = np.r_[rows, np.arange(len(df))]
        cols = np.r_[cols, first.to_numpy()]
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(df), len(df)))
    return connected_components(graph, directed=False)[1]


def random_twin_splits(df, n_splits=5):
    """Validación aleatoria con los gemelos siempre en la misma partición."""
    folds = GroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(folds.split(df, groups=twin_groups(df)))


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


def northeast_holdout(train, test, radius_m=NORTHEAST_RADIUS_M, buffer_m=0,
                      price_cap=NORTHEAST_PRICE_CAP, cell_m=None, apartments_only=False):
    """Holdout que imita el test: el train real está truncado en 1.650 M y el test no
    necesariamente; además, el test tiene inmuebles de train muy cerca (mediana de 96 m).

    Validación: inmuebles de train a menos de 'radius_m' del test (solo apartamentos si
    'apartments_only'); con 'cell_m', solo la mitad de las celdas de ese tamaño, escogidas
    al azar, para que la validación tenga vecinos en el entrenamiento como el test.
    Entrenamiento: el resto, menos una franja de 'buffer_m' y, si hay 'price_cap', solo los
    inmuebles con precio hasta ese tope (la validación conserva todos los precios).
    Devuelve (índices de entrenamiento, índices de validación)."""
    validation = distance_to_nearest_m(train, test) < radius_m
    if apartments_only:
        validation &= (train["is_house"] == 0).to_numpy()
    if cell_m:
        cells = (np.floor(train["lat"] * 111_320 / cell_m).astype(int).astype(str) + "_"
                 + np.floor(train["lon"] * 111_000 / cell_m).astype(int).astype(str))
        unique_cells = np.unique(cells[validation])
        chosen = np.random.default_rng(SEED).choice(unique_cells, len(unique_cells) // 2,
                                                    replace=False)
        validation &= cells.isin(chosen).to_numpy()
    fit = ~validation
    if buffer_m:
        fit &= distance_to_nearest_m(train, train[validation]) > buffer_m
    if price_cap:
        fit &= (train[TARGET] <= price_cap).to_numpy()
    return np.flatnonzero(fit), np.flatnonzero(validation)


def transductive_reference(y, fit_idx, val_idx, price_cap=NORTHEAST_PRICE_CAP):
    """Referencia del holdout transductivo: el ajuste más la validación con precio hasta el
    tope, y la posición de cada fila de validación dentro de ella (-1 si no está)."""
    in_reference = val_idx[np.asarray(y)[val_idx] <= price_cap]
    ref_idx = np.r_[fit_idx, in_reference]
    position = pd.Series(np.arange(len(ref_idx)), index=ref_idx)
    return ref_idx, position.reindex(val_idx).fillna(-1).astype(int).to_numpy()


def weighted_mae(y_true, y_pred, groups):
    """MAE del holdout nororiental ponderado como el test (la columna de la ronda 3a): el
    error de los inmuebles con anuncios del mismo edificio en la referencia transductiva
    pesa lo que pesan en el test (24 %), y el de los que no tienen anuncios del mismo
    edificio en el ajuste (casi toda la validación), lo demás. 'groups' viene de
    models.northeast_building_groups."""
    with_building, without_building, share = groups
    err = np.abs(np.asarray(y_pred) - np.asarray(y_true))
    return share * err[with_building].mean() + (1 - share) * err[without_building].mean()


def error_summary(y_true, y_pred):
    """MAE y asimetría del error. Sobrepredecir es el error caro (caso Zillow)."""
    err = np.asarray(y_pred) - np.asarray(y_true)
    return {
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "share_overpredicted": float(np.mean(err > 0)),
        "mean_overprediction": float(err[err > 0].mean()) if (err > 0).any() else 0.0,
    }


def find_submission(file_name):
    """Ruta de un envío guardado, en la carpeta de envíos o en la de obsoletos."""
    path = SUBMISSIONS_DIR / file_name
    return path if path.exists() else OBSOLETE_DIR / file_name


def submission_path(out, name):
    """Ruta libre para un envío. Un envío ya escrito (también si está en obsoletos) no se
    sobrescribe: si el contenido cambió, se guarda como <name>_v2, <name>_v3...; si es
    idéntico, se reutiliza."""
    candidates = [f"{name}.csv"] + [f"{name}_v{k}.csv" for k in range(2, 100)]
    for file_name in candidates:
        path = find_submission(file_name)
        if not path.exists():
            return SUBMISSIONS_DIR / file_name
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
