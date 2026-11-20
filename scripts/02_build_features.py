"""Construye la base de modelación: limpieza, predictores de texto y predictores espaciales.

Entradas: stores/raw/{train,test}.csv y, si existen, stores/processed/osm_pois.csv y
stores/processed/bogota_blocks.csv (estrato y valor del terreno, de 11_download_bogota.py).
Salidas:  stores/processed/{train,test}_features.csv

Uso: python scripts/02_build_features.py
"""

import re

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from utils import (BOGOTA_BLOCKS_PATH, EARTH_RADIUS_M, ID_COL, OSM_POIS_PATH, PROCESSED_DIR,
                   TARGET, distance_to_nearest_m, haversine_m, load_raw, test_similarity)

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

# Señales de lujo que no estaban en TEXT_FLAGS. Todas suben el precio y no dependen del barrio.
LUXURY_FLAGS = {
    # Terraza de uso exclusivo (no la comunal con BBQ): más área útil y escasa en la ciudad.
    "txt_private_terrace": r"terrazas? (?:\w+ ){0,3}?(?:privadas?|de uso (?:privado|exclusivo)|exclusivas?|propias?)",
    # Ascensor que abre dentro del apartamento: típico de edificios de un apartamento por piso.
    "txt_private_elevator": r"ascensor(?:es)? (?:\w+ ){0,2}?(?:privados?|directos?|exclusivos?)",
    # Club house o casa club: conjuntos de gama alta con amenidades propias.
    "txt_club_house": r"club ?house|casa club",
    # Doble altura: diseño de gama alta, común en penthouses y dúplex.
    "txt_double_height": r"doble altura",
    # Vista a los cerros o la montaña (la vista panorámica ya está en txt_view).
    "txt_view_hills": r"vista (?:\w+ ){0,3}?(?:cerros|montana|monserrate)",
    # Penthouse sin '\bph\b', que en los anuncios suele significar propiedad horizontal.
    "txt_penthouse_strict": r"penthouse|pent house",
}

# Números escritos con palabras, para áreas como 'ciento catorce metros cuadrados'.
# 'noventa' llega como 'no' porque los datos eliminan la palabra 'venta'.
SPELLED_VALUES = {
    "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
    "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13,
    "catorce": 14, "quince": 15, "dieciseis": 16, "diecisiete": 17, "dieciocho": 18,
    "diecinueve": 19, "veinte": 20, "veintiun": 21, "veintiuno": 21, "veintidos": 22,
    "veintitres": 23, "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26,
    "veintisiete": 27, "veintiocho": 28, "veintinueve": 29, "treinta": 30, "cuarenta": 40,
    "cincuenta": 50, "sesenta": 60, "setenta": 70, "ochenta": 80, "no": 90, "cien": 100,
    "ciento": 100, "doscientos": 200, "trescientos": 300, "cuatrocientos": 400,
    "quinientos": 500, "seiscientos": 600, "setecientos": 700, "ochocientos": 800,
    "novecientos": 900,
}
_WORD = "|".join(sorted((w for w in SPELLED_VALUES if w != "no"), key=len, reverse=True))
_WORD = rf"(?:{_WORD}|no(?= y ))"
AREA_UNIT = (r"(?:m ?sup2|mtrs? ?2|mts ?2|mt2|m2|m 2\b|metros? cuadrados|mts cuadrados"
             r"|mtrs|mts|mt\b|metros\b|m\b)")
SPELLED_AREA = rf"\b({_WORD}(?: (?:y |mil |punto )?{_WORD}| mil\b)*)(?= {AREA_UNIT})"
AREA_CANDIDATE = (rf"\b(\d{{1,4}}(?:\.\d{{1,2}})?) ?{AREA_UNIT}"
                  r"|area (?:construida|privada|total|util)(?: aproximada)?(?: de| es de)? (\d{2,4})\b")
# Un número seguido de metros no es el área si describe una parte, una distancia o una altura.
NOT_AREA_BEFORE = (r"(?:\b(?:terrazas?|balcon(?:es)?|patio|lote|terreno|jardin|antejardin|frente"
                   r"|fondo|garajes?|parqueaderos?|deposito|altillo)(?: \w+){0,2}|\d ?x) $")
# Distancias ('a 200 metros'): solo se descartan si la unidad no es explícitamente cuadrada.
DISTANCE_BEFORE = r"\b(?:a|unos|pocos|escasos|solo) $"
SQUARE_UNIT = r"sup2|m[trs]{0,3} ?2\b|cuadrad"
NOT_AREA_AFTER = (r"^ ?(?:cuadrados )?(?:de (?:la|el|del|los|las|distancia|altura|alto|frente|fondo"
                  r"|terraza|balcon|patio|jardin|lote|transmilenio|parque|estacion|centro|avenida"
                  r"|calle|carrera)\b|al\b|a la\b|hacia\b|x ?\d|(?:de )?(?:frente|fondo)\b)")


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


def words_to_value(phrase):
    """'dos mil ochocientos ochenta' -> 2880."""
    total, current = 0, 0
    for token in phrase.split():
        if token == "mil":
            total, current = total + max(current, 1) * 1000, 0
        elif token != "y":
            current += SPELLED_VALUES[token]
    return total + current


def spelled_to_number(phrase):
    """'ciento setenta y cuatro punto ochenta y siete' -> '174.87'."""
    integer, _, decimals = phrase.partition(" punto ")
    value = str(words_to_value(integer))
    return f"{value}.{words_to_value(decimals)}" if decimals else value


def normalize_numbers(text):
    """Escribe con dígitos los números que preceden a una unidad de área y recupera el
    punto decimal dicho con palabras ('146 punto 3 mts' -> '146.3 mts'). También repara
    'aacute rea', resto de la entidad HTML de 'área'."""
    text = text.str.replace("aacute rea", "area", regex=False)
    text = text.str.replace(SPELLED_AREA, lambda m: spelled_to_number(m.group(1)), regex=True)
    return text.str.replace(r"(\d+) punto (\d+)", r"\1.\2", regex=True)


def is_area_context(text, start, end):
    """Falso si el número describe una terraza, un lote, una distancia o una altura."""
    before, after = text[max(0, start - 40):start], text[end:end + 30]
    is_distance = (re.search(DISTANCE_BEFORE, before)
                   and not re.search(SQUARE_UNIT, text[start:end + 10]))
    return not (is_distance or re.search(NOT_AREA_BEFORE, before)
                or re.match(NOT_AREA_AFTER, after))


def first_plausible_area(text, ceiling, rescale_above):
    """Primera mención de área que pasa el filtro de contexto y cae en [20, ceiling] m2.
    Sin punto decimal, '1463 mts' se lee como 146.3 si 1463 supera 'rescale_above'."""
    for match in re.finditer(AREA_CANDIDATE, text):
        token = match.group(1) or match.group(2)
        if not is_area_context(text, match.start(), match.end()):
            continue
        value = float(token)
        for _ in range(2 if "." not in token else 0):
            value = value / 10 if value > rescale_above else value
        if 20 <= value <= ceiling:
            return value
    return np.nan


def add_extended_area(out, text):
    """surface_ext: el área de 'surface' y, si falta, la que recuperan los patrones ampliados.
    Techo de plausibilidad: 2.000 m2 para casas y 1.000 m2 para apartamentos. En un
    apartamento, un entero mayor que 600 casi siempre es un decimal sin punto ('6871')."""
    is_house = out["is_house"] == 1
    ceilings = np.where(is_house, 2000, 1000)
    rescale_above = np.where(is_house, 2000, 600)
    normalized = normalize_numbers(text)
    recovered = [first_plausible_area(t, c, r)
                 for t, c, r in zip(normalized, ceilings, rescale_above)]
    out["surface_ext"] = out["surface"].fillna(pd.Series(recovered, index=out.index))
    out["surface_ext_missing"] = out["surface_ext"].isna().astype(int)
    out["log_surface_ext"] = np.log(out["surface_ext"])
    return out


def add_luxury_signals(out, text):
    """Señales de lujo del texto y su conteo (txt_lux_score)."""
    for name, pattern in LUXURY_FLAGS.items():
        out[name] = text.str.contains(pattern, regex=True).astype(int)
    out["txt_stratum_6"] = text.str.contains(r"estrato (?:6|seis)\b", regex=True).astype(int)
    # Tres o más parqueaderos: casas y apartamentos grandes de estrato alto.
    many_cars = text.str.contains(
        r"(?:parqueaderos?|garajes?|espacio) para (?:[3-6]|tres|cuatro|cinco|seis) "
        r"(?:carros|vehiculos)", regex=True)
    out["txt_parking_3plus"] = ((out["txt_n_parking"] >= 3) | many_cars).astype(int)
    out["txt_lux_score"] = out[list(LUXURY_FLAGS) + ["txt_stratum_6", "txt_parking_3plus"]].sum(axis=1)
    return out


def add_area_and_luxury(df):
    """Columnas nuevas que no tocan las existentes: área ampliada y señales de lujo."""
    out = df.copy()
    text = (out["title"].fillna("") + " " + out["description"].fillna("")).str.lower()
    return add_luxury_signals(add_extended_area(out, text), text)


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


# Nombres de barrio, edificio o conjunto en el título: n-gramas que aparecen en train y en
# test y que se concentran en un lugar (así se descartan palabras genéricas como "venta").
NAME_MIN_TRAIN, NAME_MIN_TEST, NAME_MAX_SPREAD_M = 10, 5, 1500
TITLE_STOPWORDS = {"en", "de", "del", "la", "el", "los", "las", "y", "o", "a", "al", "con",
                   "para", "por", "bogota", "x", "venta", "vendo", "vende", "se", "cod",
                   # tipo de inmueble: si no, "apartamento chico" y "chico" serían dos nombres
                   "apartamento", "apto", "apartaestudio", "aparta", "estudio", "casa",
                   "penthouse", "duplex", "pent", "house",
                   # códigos de inmobiliaria que no siempre vienen después de 'cod'
                   "rah", "co", "vrei", "vedu", "vpad", "vbks", "vixo", "fr", "ap",
                   # adjetivos frecuentes en títulos de la zona
                   "moderno", "remodelado", "exterior", "hermoso", "espectacular"}


def title_tokens(title):
    """Palabras del título sin acentos, sin códigos (todo lo que sigue a 'cod'), sin
    números y sin palabras vacías."""
    text = (pd.Series([title]).fillna("").str.lower().str.normalize("NFKD")
            .str.encode("ascii", "ignore").str.decode("ascii").iat[0])
    text = re.split(r"\bcod\b", text)[0]
    words = re.findall(r"[a-z]+", text)
    return [w for w in words if w not in TITLE_STOPWORDS and len(w) > 1]


def ngrams(tokens, max_n=3):
    return {" ".join(tokens[i:i + n]) for n in range(1, max_n + 1)
            for i in range(len(tokens) - n + 1)}


def name_vocabulary(train_grams, test_grams, train):
    """N-gramas frecuentes en train y en test cuyos anuncios de train están concentrados:
    la mediana de su distancia al punto mediano es menor que NAME_MAX_SPREAD_M."""
    count_train = pd.Series([g for grams in train_grams for g in grams]).value_counts()
    count_test = pd.Series([g for grams in test_grams for g in grams]).value_counts()
    common = count_train[count_train >= NAME_MIN_TRAIN].index.intersection(
        count_test[count_test >= NAME_MIN_TEST].index)
    spread = {}
    for gram in common:
        rows = train[[gram in grams for grams in train_grams]]
        center_lat, center_lon = rows["lat"].median(), rows["lon"].median()
        spread[gram] = np.median(haversine_m(rows["lat"], rows["lon"], center_lat, center_lon))
    return {g: d for g, d in spread.items() if d < NAME_MAX_SPREAD_M}


def best_name(grams, vocabulary):
    """El nombre más largo del vocabulario en el título; a igual largo, el más concentrado."""
    found = [g for g in grams if g in vocabulary]
    if not found:
        return np.nan
    return min(found, key=lambda g: (-len(g.split()), vocabulary[g]))


def add_title_names(built, raw):
    """Columna 'title_name' (texto) en train y test; vacía si el título no trae un nombre del
    vocabulario. Solo usa títulos y coordenadas, nunca el precio."""
    grams = {name: [ngrams(title_tokens(t)) for t in raw[name]["title"]] for name in raw}
    vocabulary = name_vocabulary(grams["train"], grams["test"], raw["train"])
    for name in built:
        built[name]["title_name"] = [best_name(g, vocabulary) for g in grams[name]]
    print(f"Nombres en el título: {len(vocabulary)}; cubren "
          f"{built['train']['title_name'].notna().mean():.0%} de train y "
          f"{built['test']['title_name'].notna().mean():.0%} de test")


def add_bogota_blocks(built):
    """Estrato socioeconómico de la manzana y valor de referencia catastral del terreno por
    m2 (11_download_bogota.py). El estrato 0 (manzana sin uso residencial) y el valor 0 se
    dejan como faltantes."""
    blocks = pd.read_csv(BOGOTA_BLOCKS_PATH).set_index(ID_COL)
    for feats in built.values():
        stratum = feats[ID_COL].map(blocks["ext_stratum"])
        land_value = feats[ID_COL].map(blocks["ext_land_value_m2"])
        feats["ext_stratum"] = stratum.where(stratum > 0)
        feats["ext_land_value_m2"] = land_value.where(land_value > 0)


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
        # Al final, para que las columnas anteriores conserven su orden y sus valores.
        feats = add_area_and_luxury(feats)
        drop = ["title", "description", "city", "operation_type", "property_type",
                "surface_total", "surface_covered", "month", "year"]
        built[name] = feats.drop(columns=drop)
    add_title_names(built, {"train": train, "test": test})
    if BOGOTA_BLOCKS_PATH.exists():
        add_bogota_blocks(built)
    # Parecido al test (no es predictor): sirve para ponderar el entrenamiento.
    built["train"]["test_similarity"] = test_similarity(built["train"], built["test"])
    for name, feats in built.items():
        feats.to_csv(PROCESSED_DIR / f"{name}_features.csv", index=False)
        print(f"{name}: {feats.shape[0]} filas, {feats.shape[1]} columnas")
    summary = built["test"].drop(columns=[ID_COL, TARGET]).isna().mean()
    print("Faltantes en test:\n", summary[summary > 0].round(3).to_string())


if __name__ == "__main__":
    main()
