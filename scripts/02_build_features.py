"""Construye la base de modelación: limpieza, predictores de texto y predictores espaciales.

Entradas: stores/raw/{train,test}.csv y, si existe, stores/processed/osm_pois.csv.
Salidas:  stores/processed/{train,test}_features.csv

Uso: python scripts/02_build_features.py
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from utils import (EARTH_RADIUS_M, ID_COL, OSM_POIS_PATH, PROCESSED_DIR, TARGET,
                   distance_to_nearest_m, haversine_m, load_raw, test_similarity)

COUNT_RADIUS_M = 500
# Categorías densas: además de la distancia, cuántas hay a 500 m (densidad de servicios).
COUNT_CATEGORIES = ["restaurant", "bar", "bank", "park", "school"]
# Conteos adicionales a 300 m (cuadra) y 1 km (barrio), también para transporte y mercados.
EXTRA_COUNT_RADII_M = [300, 1000]
EXTRA_COUNT_CATEGORIES = COUNT_CATEGORIES + ["transit_station", "supermarket"]

# Puntos de referencia (coordenadas aproximadas). Capturan el gradiente de precios
# respecto a los centros de empleo y ocio del nororiente y el centro de la ciudad.
LANDMARKS = {
    "zona_t": (4.6672, -74.0535),
    "parque_93": (4.6765, -74.0482),
    "centro_internacional": (4.6135, -74.0695),
    "plaza_bolivar": (4.5981, -74.0760),
}

# Ejes trazados a mano con coordenadas aproximadas (sur a norte): la Carrera Séptima y el
# borde de los cerros orientales. El precio cae a medida que uno se aleja de ambos.
CORRIDORS = {
    "carrera_7": [(4.598, -74.0755), (4.6125, -74.0695), (4.632, -74.0655),
                  (4.645, -74.0625), (4.656, -74.057), (4.669, -74.050), (4.684, -74.041),
                  (4.706, -74.031), (4.727, -74.026), (4.751, -74.024), (4.770, -74.023)],
    "cerros": [(4.580, -74.065), (4.600, -74.060), (4.620, -74.058), (4.640, -74.055),
               (4.660, -74.045), (4.680, -74.033), (4.700, -74.026), (4.730, -74.020),
               (4.760, -74.018), (4.780, -74.017)],
}

NUMBER_WORDS = {"un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4,
                "cinco": 5, "seis": 6}
NUM = r"(\d+|un|uno|una|dos|tres|cuatro|cinco|seis)"

# Predictores binarios derivados del texto: nombre -> expresión regular.
TEXT_FLAGS = {
    "txt_parking": r"parqueadero|garaje|garage",
    "txt_terrace": r"terraza",
    "txt_balcony": r"balcon",
    "txt_elevator": r"ascensor",
    "txt_storage": r"deposito|cuarto util",
    "txt_fireplace": r"chimenea",
    "txt_security": r"vigilancia|porteria|seguridad 24|celaduria",
    "txt_gym": r"gimnasio|\bgym\b",
    "txt_pool": r"piscina",
    "txt_gated": r"conjunto cerrado|conjunto residencial",
    "txt_remodeled": r"remodelad|renovad|para estrenar|estrenar",
    "txt_duplex": r"duplex",
    "txt_penthouse": r"penthouse|pent house|\bph\b",
    "txt_view": r"vista panoramica|excelente vista|hermosa vista|vista a la ciudad|vista a los cerros",
    # Señales de gama alta que no dependen de la ubicación.
    "txt_service_room": r"(?:cuarto|alcoba|habitacion) (?:y bano )?de servicio|servicio con bano",
    "txt_drivers_room": r"conductores|choferes",
    "txt_walk_in": r"vestier|walking closet|walk in closet",
    "txt_study": r"estudio|family room|biblioteca",
    "txt_luxury": r"lujo|exclusiv",
    "txt_spa": r"sauna|turco|jacuzzi|squash",
    "txt_exterior": r"\bexterior\b",
}


def to_number(token):
    """'dos' -> 2, '3' -> 3."""
    return NUMBER_WORDS.get(token, None) if not str(token).isdigit() else int(token)


def first_number(text, pattern, low, high):
    """Primer número que casa con el patrón y cae en un rango plausible; si no, NaN."""
    found = text.str.extract(pattern, expand=False)
    values = found.map(to_number, na_action="ignore").astype(float)
    return values.where(values.between(low, high))


def area_from_text(text, is_house):
    """Área en m2 mencionada en el anuncio.

    El texto viene sin signos de puntuación, así que '146.3 m2' aparece como '1463 mts'.
    Si el número es implausiblemente grande se divide entre 10 hasta que sea razonable.
    """
    pattern = r"(\d{2,5})\s?(?:m2|mt2|mts2|mts|mt|m\b|metros|mtrs)"
    area = text.str.extract(pattern, expand=False).astype(float)
    ceiling = np.where(is_house == 1, 1200, 700)
    for _ in range(2):
        area = np.where(area > ceiling, area / 10, area)
    area = pd.Series(area, index=text.index)
    return area.where(area.between(18, 1200))


def add_structural_and_text(df):
    """Limpia las variables estructurales y agrega los predictores de texto."""
    out = df.copy()
    text = (out["title"].fillna("") + " " + out["description"].fillna("")).str.lower()

    out["is_house"] = (out["property_type"] == "Casa").astype(int)
    # Meses transcurridos desde enero de 2019: tendencia de precios nominales.
    out["time_index"] = (out["year"] - 2019) * 12 + out["month"]

    # --- Predictores de texto ---
    for name, pattern in TEXT_FLAGS.items():
        out[name] = text.str.contains(pattern, regex=True).astype(int)
    out["txt_n_parking"] = first_number(
        text, NUM + r"\s(?:parqueaderos?|garajes?|garages?)", 1, 6)
    out.loc[out["txt_parking"] == 0, "txt_n_parking"] = 0
    out["txt_n_parking"] = out["txt_n_parking"].fillna(1)  # lo menciona sin número
    out["txt_stratum"] = first_number(text, r"estrato\s(\d)\b", 1, 6)
    out["txt_floor"] = first_number(text, r"\bpiso\s(\d{1,2})\b", 1, 40)
    out["txt_desc_length"] = out["description"].fillna("").str.len()
    out["txt_age"] = first_number(text, r"(\d{1,2}) anos (?:de construid|de antig)", 0, 80)
    txt_bathrooms = first_number(text, NUM + r"\sbanos?\b", 1, 10)
    txt_area = area_from_text(text, out["is_house"])

    # --- Área: cubierta > total > texto. Valores absurdos se tratan como faltantes. ---
    covered = out["surface_covered"].where(out["surface_covered"].between(18, 1200))
    total = out["surface_total"].where(out["surface_total"].between(18, 1200))
    out["surface"] = covered.fillna(total).fillna(txt_area)
    out["surface_missing"] = out["surface"].isna().astype(int)
    out["log_surface"] = np.log(out["surface"])  # ayuda a los modelos lineales

    # --- Baños y cuartos: se completan con el texto cuando faltan ---
    out["bathrooms"] = out["bathrooms"].fillna(txt_bathrooms)
    out["rooms"] = out["rooms"].fillna(out["bedrooms"])
    return out


def add_landmark_distances(df):
    """Distancia en metros a cada punto de referencia."""
    out = df.copy()
    for name, (lat, lon) in LANDMARKS.items():
        out[f"dist_{name}_m"] = haversine_m(out["lat"], out["lon"], lat, lon)
    return out


def densify(points, step_m=50):
    """Interpola puntos cada ~step_m metros a lo largo de una polilínea (lat, lon)."""
    rows = []
    for (lat1, lon1), (lat2, lon2) in zip(points[:-1], points[1:]):
        n = max(int(haversine_m(lat1, lon1, lat2, lon2) // step_m), 1)
        t = np.linspace(0, 1, n, endpoint=False)
        rows.append(np.c_[lat1 + t * (lat2 - lat1), lon1 + t * (lon2 - lon1)])
    rows.append(np.array([points[-1]]))
    return pd.DataFrame(np.vstack(rows), columns=["lat", "lon"])


def add_corridor_distances(df):
    """Distancia en metros a la Carrera Séptima y al borde de los cerros orientales."""
    out = df.copy()
    for name, points in CORRIDORS.items():
        out[f"dist_{name}_m"] = distance_to_nearest_m(out, densify(points))
    return out


def add_osm_features(df, pois):
    """Distancia al punto más cercano de cada categoría OSM y conteos a 500 m."""
    out = df.copy()
    coords = np.radians(out[["lat", "lon"]].to_numpy())
    for category, group in pois.groupby("category"):
        tree = BallTree(np.radians(group[["lat", "lon"]].to_numpy()), metric="haversine")
        distance, _ = tree.query(coords, k=1)
        out[f"osm_dist_{category}_m"] = distance[:, 0] * EARTH_RADIUS_M
        radii = [COUNT_RADIUS_M] if category in COUNT_CATEGORIES else []
        if category in EXTRA_COUNT_CATEGORIES:
            radii += EXTRA_COUNT_RADII_M
        for radius in radii:
            out[f"osm_n_{category}_{radius}m"] = tree.query_radius(
                coords, r=radius / EARTH_RADIUS_M, count_only=True)
    return out


def main():
    train, test = load_raw()
    pois = pd.read_csv(OSM_POIS_PATH) if OSM_POIS_PATH.exists() else None
    if pois is None:
        print("AVISO: no existe osm_pois.csv; la base queda SIN predictores de OSM. "
              "Corra primero scripts/01_download_osm.py.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    built = {}
    for name, df in [("train", train), ("test", test)]:
        feats = add_corridor_distances(add_landmark_distances(add_structural_and_text(df)))
        if pois is not None:
            feats = add_osm_features(feats, pois)
        drop = ["title", "description", "city", "operation_type", "property_type",
                "surface_total", "surface_covered", "month", "year"]
        built[name] = feats.drop(columns=drop)
    # Parecido al test (no es predictor): sirve para ponderar el entrenamiento.
    built["train"]["test_similarity"] = test_similarity(built["train"], built["test"])
    for name, feats in built.items():
        feats.to_csv(PROCESSED_DIR / f"{name}_features.csv", index=False)
        print(f"{name}: {feats.shape[0]} filas, {feats.shape[1]} columnas")
    summary = built["test"].drop(columns=[ID_COL, TARGET]).isna().mean()
    print("Faltantes en test:\n", summary[summary > 0].round(3).to_string())


if __name__ == "__main__":
    main()
