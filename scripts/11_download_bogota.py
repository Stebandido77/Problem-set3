"""Asigna a cada inmueble el estrato y el valor de referencia del terreno de su manzana.

Fuentes (datos abiertos de Bogotá, consultados el 19 de noviembre de 2026):

1. Estrato: "Estratificación Manzana Bogotá D.C.", Secretaría Distrital de Planeación (SDP),
   publicado en datosabiertos.bogota.gov.co y servido por IDECA / Catastro en
   https://serviciosgis.catastrobogota.gov.co/arcgis/rest/services/
   ordenamientoterritorial/estratificacion/MapServer/1
   (capa "Manzanas de estrato", vigente a la fecha de descarga). El estrato 0 corresponde a
   manzanas sin uso residencial y se conserva tal cual.
2. Valor de referencia: "Mediana del valor comercial por metro cuadrado de terreno por
   manzana – Valor de referencia. Bogotá D.C.", Unidad Administrativa Especial de Catastro
   Distrital (UAECD), https://serviciosgis.catastrobogota.gov.co/arcgis/rest/services/
   catastro/valorreferencia/MapServer/0
   (campo V_REF en pesos por m2, una vigencia por año en el campo ANO). Cada inmueble recibe el
   valor de la vigencia del año de su anuncio (2019, 2020 o 2021).

Las geometrías se piden en WGS84 (outSR=4326). Cada inmueble toma el valor de la manzana que lo
contiene (punto en polígono con matplotlib); si no cae en ninguna, toma el de la manzana con
centroide más cercano. En ambos casos se guarda la distancia en metros al centroide de la manzana
asignada. Solo se usan las coordenadas de train y test, nunca el precio.

Salida: stores/processed/bogota_blocks.csv con property_id, ext_stratum, ext_stratum_dist_m,
ext_land_value_m2, ext_land_value_dist_m. Las descargas quedan en caché en
stores/external/bogota/ (no se versiona; el CSV de salida sí).

Uso (desde scripts/):
  python 11_download_bogota.py
"""

import gzip
import json

import numpy as np
import pandas as pd
import requests
from matplotlib.path import Path
from sklearn.neighbors import BallTree

from utils import PROCESSED_DIR, RAW_DIR, haversine_m

# Caché de la descarga (fuera de stores/raw, que es solo de Kaggle; no se versiona).
CACHE_DIR = RAW_DIR.parent / "external" / "bogota"
OUTPUT_PATH = PROCESSED_DIR / "bogota_blocks.csv"

STRATUM_URL = (
    "https://serviciosgis.catastrobogota.gov.co/arcgis/rest/services/"
    "ordenamientoterritorial/estratificacion/MapServer/1/query"
)
LAND_VALUE_URL = (
    "https://serviciosgis.catastrobogota.gov.co/arcgis/rest/services/"
    "catastro/valorreferencia/MapServer/0/query"
)
LAND_VALUE_YEARS = (2019, 2020, 2021)
PAGE_SIZE = 2000


def fetch_page(url, where, value_field, offset):
    """Pide una página de manzanas con su valor y su geometría en WGS84."""
    params = {
        "where": where,
        "outFields": value_field,
        "outSR": 4326,
        "geometryPrecision": 6,
        "orderByFields": "OBJECTID",
        "resultOffset": offset,
        "resultRecordCount": PAGE_SIZE,
        "returnGeometry": "true",
        "f": "json",
    }
    response = requests.get(url, params=params, timeout=180)
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"Error del servicio: {payload['error']}")
    return payload["features"]


def to_block(feature, value_field):
    """Reduce un feature a su valor y su anillo exterior; None si no tiene geometría."""
    rings = (feature.get("geometry") or {}).get("rings")
    value = feature["attributes"].get(value_field)
    if not rings or value is None:
        return None
    return {"value": value, "ring": rings[0]}


def download_blocks(name, url, where, value_field):
    """Descarga todas las manzanas de una capa paginando, con caché en disco."""
    cache_path = CACHE_DIR / f"{name}.json.gz"
    if cache_path.exists():
        with gzip.open(cache_path, "rt", encoding="utf-8") as file:
            return json.load(file)
    blocks, offset = [], 0
    while True:
        features = fetch_page(url, where, value_field, offset)
        blocks += [b for b in (to_block(f, value_field) for f in features) if b]
        print(f"  {name}: {offset + len(features)} manzanas")
        if len(features) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_path, "wt", encoding="utf-8") as file:
        json.dump(blocks, file)
    return blocks


def ring_centroid(ring):
    """Centroide aproximado (lat, lon) como promedio de los vértices del anillo."""
    lon, lat = np.asarray(ring).mean(axis=0)
    return lat, lon


def locate_in_blocks(points, blocks, tree):
    """Índice de la manzana que contiene cada punto (-1 si ninguna)."""
    owner = np.full(len(points), -1)
    for index, block in enumerate(blocks):
        ring = np.asarray(block["ring"])
        lat, lon = ring_centroid(ring)
        reach = np.radians(np.abs(ring - [lon, lat]).max() * 1.5)
        candidates = tree.query_radius(np.radians([[lat, lon]]), r=reach)[0]
        candidates = candidates[owner[candidates] == -1]
        if len(candidates):
            inside = Path(ring).contains_points(points[candidates][:, ::-1])
            owner[candidates[inside]] = index
    return owner


def assign_blocks(points, blocks):
    """Valor de la manzana de cada punto y distancia en metros al centroide de esa manzana."""
    centroids = np.array([ring_centroid(b["ring"]) for b in blocks])
    values = np.array([b["value"] for b in blocks], dtype=float)
    point_tree = BallTree(np.radians(points), metric="haversine")
    owner = locate_in_blocks(points, blocks, point_tree)
    centroid_tree = BallTree(np.radians(centroids), metric="haversine")
    nearest = centroid_tree.query(np.radians(points), k=1, return_distance=False)[:, 0]
    print(f"  {np.mean(owner >= 0) * 100:.1f}% de los inmuebles cae dentro de una manzana")
    owner = np.where(owner >= 0, owner, nearest)
    distance = haversine_m(points[:, 0], points[:, 1], centroids[owner, 0], centroids[owner, 1])
    return values[owner], distance


def load_properties():
    """Identificador, año y coordenadas de los inmuebles de train y test."""
    columns = ["property_id", "year", "lat", "lon"]
    train = pd.read_csv(RAW_DIR / "train.csv", usecols=columns)
    test = pd.read_csv(RAW_DIR / "test.csv", usecols=columns)
    return pd.concat([train, test], ignore_index=True)


def add_stratum(properties):
    """Agrega el estrato de la manzana y la distancia a su centroide."""
    blocks = download_blocks("stratum", STRATUM_URL, "1=1", "ESTRATO")
    points = properties[["lat", "lon"]].to_numpy()
    values, distance = assign_blocks(points, blocks)
    properties["ext_stratum"] = values
    properties["ext_stratum_dist_m"] = distance.round(1)


def add_land_value(properties):
    """Agrega el valor de referencia del terreno por m2 de la vigencia del año del anuncio."""
    properties["ext_land_value_m2"] = np.nan
    properties["ext_land_value_dist_m"] = np.nan
    for year in LAND_VALUE_YEARS:
        where = f"ANO = date '{year}-01-01'"
        blocks = download_blocks(f"land_value_{year}", LAND_VALUE_URL, where, "V_REF")
        mask = properties["year"] == year
        points = properties.loc[mask, ["lat", "lon"]].to_numpy()
        values, distance = assign_blocks(points, blocks)
        properties.loc[mask, "ext_land_value_m2"] = values
        properties.loc[mask, "ext_land_value_dist_m"] = distance.round(1)


def main():
    properties = load_properties()
    print("Estrato por manzana")
    add_stratum(properties)
    print("Valor de referencia del terreno")
    add_land_value(properties)
    columns = ["property_id", "ext_stratum", "ext_stratum_dist_m", "ext_land_value_m2",
               "ext_land_value_dist_m"]
    properties[columns].to_csv(OUTPUT_PATH, index=False)
    print(f"Escrito {OUTPUT_PATH} ({len(properties)} inmuebles)")


if __name__ == "__main__":
    main()
