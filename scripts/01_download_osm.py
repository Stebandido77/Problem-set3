"""Descarga puntos de interés de OpenStreetMap (API Overpass) para Bogotá.

Salida: stores/processed/osm_pois.csv con columnas category, lat, lon.
Requiere internet. Se corre una sola vez; 02_build_features.py usa el archivo si existe.

Uso: python scripts/01_download_osm.py
"""

import time

import pandas as pd
import requests

from utils import OSM_POIS_PATH

# Caja que cubre el área urbana de Bogotá: sur, oeste, norte, este.
BBOX = (4.45, -74.25, 4.85, -73.98)

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# Categoría -> filtros de etiquetas OSM. La intuición hedónica de cada una:
# amenidades (parques, restaurantes), transporte, servicios públicos y seguridad.
CATEGORIES = {
    "park": ['["leisure"="park"]'],
    "transit_station": ['["public_transport"="station"]', '["amenity"="bus_station"]'],
    "university": ['["amenity"="university"]'],
    "school": ['["amenity"="school"]'],
    "hospital": ['["amenity"~"^(hospital|clinic)$"]'],
    "mall": ['["shop"="mall"]'],
    "restaurant": ['["amenity"~"^(restaurant|cafe)$"]'],
    "bar": ['["amenity"~"^(bar|pub|nightclub)$"]'],
    "bank": ['["amenity"="bank"]'],
    "supermarket": ['["shop"="supermarket"]'],
    "police": ['["amenity"="police"]'],
}


def build_query(filters):
    """Consulta Overpass QL: nodos, vías y relaciones; 'out center' da un punto por objeto."""
    south, west, north, east = BBOX
    body = "".join(f"nwr{f}({south},{west},{north},{east});" for f in filters)
    return f"[out:json][timeout:180];({body});out center;"


def parse_elements(elements, category):
    """Convierte la respuesta de Overpass en filas (category, lat, lon)."""
    rows = []
    for el in elements:
        point = el if "lat" in el else el.get("center")
        if point:
            rows.append({"category": category, "lat": point["lat"], "lon": point["lon"]})
    return rows


def fetch(query, retries=3):
    """Consulta Overpass alternando servidores y esperando entre reintentos."""
    last_error = None
    for attempt in range(retries):
        for url in OVERPASS_URLS:
            try:
                response = requests.post(url, data={"data": query}, timeout=240)
                response.raise_for_status()
                return response.json()["elements"]
            except Exception as error:  # red caída, límite de uso, etc.
                last_error = error
        time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"Overpass no respondió: {last_error}")


def main():
    rows = []
    for category, filters in CATEGORIES.items():
        elements = fetch(build_query(filters))
        parsed = parse_elements(elements, category)
        print(f"{category:<16} {len(parsed):>6} puntos")
        rows.extend(parsed)
        time.sleep(2)  # cortesía con el servidor público
    OSM_POIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop_duplicates().to_csv(OSM_POIS_PATH, index=False)
    print(f"Guardado en {OSM_POIS_PATH}")


if __name__ == "__main__":
    main()
