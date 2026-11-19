"""Descarga puntos de interés de OpenStreetMap (API Overpass) para Bogotá.

Salida: stores/processed/osm_pois.csv con columnas category, lat, lon.
Requiere internet. Si se interrumpe, al volver a correr retoma donde quedó; 02_build_features.py usa el archivo si existe.

Uso:
  python scripts/01_download_osm.py               # descarga lo que falte y arma el archivo
  python scripts/01_download_osm.py --cache-only  # arma el archivo con lo ya descargado
"""

import argparse
import time

import pandas as pd
import requests

from utils import OSM_POIS_PATH

# Caja que cubre el área urbana de Bogotá: sur, oeste, norte, este.
BBOX = (4.45, -74.25, 4.85, -73.98)

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
# Los servidores públicos rechazan clientes sin identificación.
HEADERS = {"User-Agent": "problem-set-3-meca4107/1.0 (proyecto academico Uniandes)"}
# Descargas parciales: una por categoría, para reanudar sin repetir lo ya bajado.
CACHE_DIR = OSM_POIS_PATH.parent / "osm_cache"

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


def build_query(filters, bbox=BBOX):
    """Consulta Overpass QL: nodos, vías y relaciones; 'out center' da un punto por objeto."""
    south, west, north, east = bbox
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


def split_bbox(bbox):
    """Parte la caja en mitad sur y mitad norte."""
    south, west, north, east = bbox
    middle = (south + north) / 2
    return [(south, west, middle, east), (middle, west, north, east)]


def fetch_category(filters, retries=2):
    """Descarga una categoría; si la caja completa falla, la pide en dos mitades
    (consultas más pequeñas que los servidores saturados sí alcanzan a responder)."""
    try:
        return fetch(build_query(filters), retries)
    except RuntimeError as error:
        print(f"  caja completa falló; se parte en dos mitades\n  {error}", flush=True)
    halves = split_bbox(BBOX)
    return [el for half in halves for el in fetch(build_query(filters, half), retries)]


def fetch(query, retries=4):
    """Consulta Overpass alternando servidores y esperando cada vez más entre rondas."""
    errors = []
    for attempt in range(retries):
        for url in OVERPASS_URLS:
            try:
                response = requests.post(url, data={"data": query}, headers=HEADERS,
                                         timeout=240)
                response.raise_for_status()
                return response.json()["elements"]
            except Exception as error:  # red caída, límite de uso, respuesta no JSON
                errors.append(f"{url.split('/')[2]}: {error}")
        wait = 30 * (attempt + 1)
        print(f"  ningún servidor respondió; reintento en {wait} s", flush=True)
        time.sleep(wait)
    raise RuntimeError("Overpass no respondió:\n  " + "\n  ".join(errors[-len(OVERPASS_URLS):]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-only", action="store_true",
                        help="no descarga; usa solo las categorías ya guardadas")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for category, filters in CATEGORIES.items():
        cache = CACHE_DIR / f"{category}.csv"
        if cache.exists():
            frame = pd.read_csv(cache)
            print(f"{category:<16} {len(frame):>6} puntos (ya descargado)", flush=True)
        elif args.cache_only:
            print(f"{category:<16} sin descargar: queda por fuera", flush=True)
            continue
        else:
            rows = parse_elements(fetch_category(filters), category)
            frame = pd.DataFrame(rows, columns=["category", "lat", "lon"])
            frame.to_csv(cache, index=False)
            print(f"{category:<16} {len(frame):>6} puntos", flush=True)
            time.sleep(5)  # cortesía con el servidor público
        frames.append(frame)
    pd.concat(frames).drop_duplicates().to_csv(OSM_POIS_PATH, index=False)
    print(f"Guardado en {OSM_POIS_PATH}")


if __name__ == "__main__":
    main()
