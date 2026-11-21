"""Predio de cada inmueble en el catastro de Bogotá: área del lote, número de pisos y área
construida (ronda 4, pieza D).

Fuente: los mismos servicios ArcGIS de Catastro Bogotá (UAECD) que usa
11_download_bogota.py, consultados el 21 de noviembre de 2026:

1. "Lote": https://serviciosgis.catastrobogota.gov.co/arcgis/rest/services/
   catastro/lote/MapServer/0 (LOTCODIGO y el polígono del lote).
2. "Construcción": https://serviciosgis.catastrobogota.gov.co/arcgis/rest/services/
   catastro/construccion/MapServer/0 (LOTECODIGO, CONNPISOS y la huella de cada
   construcción).

Ninguna capa pública de estos servicios trae la antigüedad de la construcción, así que ese
predictor no se pudo construir.

Cómo se asigna: los lotes y construcciones a menos de 'RADIUS_M' de cada inmueble se piden
con una consulta espacial (solo identificadores) y luego por identificador, con su geometría
en WGS84. Cada inmueble toma el lote que lo contiene o, si no cae en ninguno, el de centroide
más cercano, y se guarda la distancia a ese centroide (0 si está adentro). Del lote salen:
  ext_lot_area_m2      área del polígono del lote
  ext_floors           máximo número de pisos de sus construcciones
  ext_built_area_m2    suma de huella por pisos de sus construcciones
Solo usa coordenadas, nunca el precio. Las áreas se calculan con una proyección local.

Salida: stores/processed/bogota_lots.csv. Caché en stores/external/bogota/ (no se versiona);
la caché depende solo de la capa: si cambian train o test, hay que borrarla.
Uso: python 16_download_lots.py
"""

import gzip
import json
import time

import numpy as np
import pandas as pd
import requests
from matplotlib.path import Path
from sklearn.neighbors import BallTree

from utils import EARTH_RADIUS_M, PROCESSED_DIR, RAW_DIR

CACHE_DIR = RAW_DIR.parent / "external" / "bogota"
OUTPUT_PATH = PROCESSED_DIR / "bogota_lots.csv"
SERVICE = "https://serviciosgis.catastrobogota.gov.co/arcgis/rest/services/catastro"
LAYERS = {"lot": ("lote", "LOTCODIGO"), "building": ("construccion", "LOTECODIGO,CONNPISOS")}
RADIUS_M = 15
POINTS_PER_QUERY = 100
IDS_PER_QUERY = 200


def post(layer, data):
    """Consulta a una capa; reintenta dos veces si el servicio falla."""
    url = f"{SERVICE}/{layer}/MapServer/0/query"
    for attempt in range(3):
        try:
            response = requests.post(url, data={**data, "f": "json"}, timeout=180)
            payload = response.json()
            if "error" not in payload:
                return payload
        except (requests.RequestException, ValueError):
            pass
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"La capa {layer} no respondió: {data.get('objectIds', '')[:50]}")


def nearby_ids(layer, points):
    """Identificadores de los polígonos a menos de RADIUS_M de algún punto (lat, lon)."""
    ids = set()
    for start in range(0, len(points), POINTS_PER_QUERY):
        batch = points[start:start + POINTS_PER_QUERY][:, ::-1].tolist()
        geometry = json.dumps({"points": batch, "spatialReference": {"wkid": 4326}})
        payload = post(layer, {"geometry": geometry, "geometryType": "esriGeometryMultipoint",
                               "inSR": 4326, "spatialRel": "esriSpatialRelIntersects",
                               "distance": RADIUS_M, "units": "esriSRUnit_Meter",
                               "returnIdsOnly": "true"})
        ids.update(payload.get("objectIds") or [])
        if start % 5000 == 0:
            print(f"  {layer}: {start + len(batch)} puntos, {len(ids)} polígonos", flush=True)
    return sorted(ids)


def fetch_features(name, points):
    """Atributos y anillo exterior de los polígonos cercanos, con caché en disco."""
    cache_path = CACHE_DIR / f"{name}.json.gz"
    if cache_path.exists():
        with gzip.open(cache_path, "rt", encoding="utf-8") as file:
            return json.load(file)
    layer, fields = LAYERS[name]
    ids = nearby_ids(layer, points)
    features = []
    for start in range(0, len(ids), IDS_PER_QUERY):
        chunk = ",".join(map(str, ids[start:start + IDS_PER_QUERY]))
        payload = post(layer, {"objectIds": chunk, "outFields": fields, "outSR": 4326,
                               "geometryPrecision": 7, "returnGeometry": "true"})
        for feature in payload["features"]:
            rings = (feature.get("geometry") or {}).get("rings")
            if rings:
                features.append({**feature["attributes"], "ring": rings[0]})
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_path, "wt", encoding="utf-8") as file:
        json.dump(features, file)
    return features


def ring_area_m2(ring):
    """Área de un anillo (lon, lat) con una proyección equirrectangular local."""
    ring = np.radians(np.asarray(ring))
    x = ring[:, 0] * np.cos(ring[:, 1].mean()) * EARTH_RADIUS_M
    y = ring[:, 1] * EARTH_RADIUS_M
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def assign_lots(points, lots):
    """Lote de cada punto (el que lo contiene o el de centroide más cercano) y distancia en
    metros al centroide (0 si está adentro)."""
    centroids = np.array([np.asarray(lot["ring"]).mean(axis=0)[::-1] for lot in lots])
    centroid_tree = BallTree(np.radians(centroids), metric="haversine")
    distance, nearest = centroid_tree.query(np.radians(points), k=1)
    owner, distance = nearest[:, 0], distance[:, 0] * EARTH_RADIUS_M
    # Se revisan los 10 lotes más cercanos por si alguno contiene el punto.
    _, candidates = centroid_tree.query(np.radians(points), k=min(10, len(lots)))
    for row, options in enumerate(candidates):
        for lot in options:
            if Path(lots[lot]["ring"]).contains_point(points[row][::-1]):
                owner[row], distance[row] = lot, 0.0
                break
    return owner, distance


def lot_table(lots, buildings):
    """Área del lote, pisos máximos y área construida por código de lote."""
    table = pd.DataFrame({"lot": [lot["LOTCODIGO"] for lot in lots],
                          "ext_lot_area_m2": [ring_area_m2(lot["ring"]) for lot in lots]})
    built = pd.DataFrame({"lot": [b["LOTECODIGO"] for b in buildings],
                          "floors": [b["CONNPISOS"] or np.nan for b in buildings],
                          "footprint": [ring_area_m2(b["ring"]) for b in buildings]})
    built["built"] = built["footprint"] * built["floors"].fillna(1)
    per_lot = built.groupby("lot").agg(ext_floors=("floors", "max"),
                                       ext_built_area_m2=("built", "sum"))
    return table.join(per_lot, on="lot")


def main():
    columns = ["property_id", "lat", "lon"]
    properties = pd.concat([pd.read_csv(RAW_DIR / "train.csv", usecols=columns),
                            pd.read_csv(RAW_DIR / "test.csv", usecols=columns)],
                           ignore_index=True)
    points = np.unique(properties[["lat", "lon"]].round(6).to_numpy(), axis=0)
    print(f"{len(points)} ubicaciones distintas")
    lots = fetch_features("lot", points)
    buildings = fetch_features("building", points)
    print(f"{len(lots)} lotes y {len(buildings)} construcciones")
    owner, distance = assign_lots(properties[["lat", "lon"]].to_numpy(), lots)
    table = lot_table(lots, buildings).iloc[owner].reset_index(drop=True)
    out = pd.concat([properties[["property_id"]], table.drop(columns="lot")], axis=1)
    out["ext_lot_dist_m"] = distance.round(1)
    print(f"{np.mean(distance == 0) * 100:.1f}% cae dentro de un lote; "
          f"{out['ext_floors'].notna().mean() * 100:.1f}% con pisos")
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Escrito {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
