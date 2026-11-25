"""Etapa 1, entender los datos: figuras y tablas para las presentaciones. No entrena nada:
lee las bases crudas y procesadas, las predicciones del test y los conjuntos de utils.

Salidas en views/:
  price_distribution.png         precio de train por tipo, con el piso y el tope del recorte
  test_prediction_distribution.png  precio observado en train frente a lo predicho por la 152
  missing_values.csv / .png      faltantes de área y baños antes y después de recuperarlos
  map_train_test.png             ubicación de train y test (Chapinero)
  chapinero_vs_bogota.csv        test (Chapinero) frente a train (resto de Bogotá)
  chapinero_vs_bogota_signals.png  señales del texto y tipo, en % de anuncios
  chapinero_vs_bogota_table.png  tamaño, estrato y valor del terreno, en tabla
  predictors_table.csv           predictores externos y de texto, con fuente, intuición y si
                                 entran al mejor modelo (152 = boosting 127 + red 148)
  predictors_summary.csv / .png  el mismo inventario resumido por bloque

Uso (desde scripts/):
  python 31_data_overview.py
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

import plot_style as ps
from utils import (COORD_FEATURES, EXTERNAL_FEATURES, ID_COL, LANDMARK_FEATURES,
                   NORTHEAST_RADIUS_M, PROCESSED_DIR, RAW_DIR, SAME_POINT_FEATURES, TARGET,
                   TEXT_FEATURES, VIEWS_DIR, distance_to_nearest_m, feature_sets,
                   osm_feature_names)

BEST = "152_ensemble_w70_30_2"
TEST_COLOR = "#B2182B"
TRAIN_COLOR = "#7F7F7F"
TYPE_COLORS = {"Apartamento": "#4D4D4D", "Casa": "#BDBDBD"}
# Componentes de la 152: conjunto de predictores de cada uno; los dos sin lat ni lon
# (drop_coords en 04_train_models.py) y con el ridge del texto fuera de muestra.
BEST_COMPONENTS = {"boosting 127": "pt_core_coords", "red 148": "ext_core_coords"}
TEXT_RIDGE = "txt_ridge"


def load_bases():
    """Bases crudas y procesadas de train y test."""
    raw = {s: pd.read_csv(RAW_DIR / f"{s}.csv") for s in ("train", "test")}
    processed = {s: pd.read_csv(PROCESSED_DIR / f"{s}_features.csv") for s in ("train", "test")}
    return raw, processed


def pct(value):
    return f"{ps.es_number(100 * value, 0)} %"


# --- Precios ---

def plot_price_distribution(train):
    """Histograma del precio de train por tipo con el piso y el tope del recorte."""
    low, high = train[TARGET].min(), train[TARGET].max()
    bins = np.linspace(low, high, 34)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    types = ["Apartamento", "Casa"]
    ax.hist([train.loc[train["property_type"] == t, TARGET] for t in types], bins=bins,
            stacked=True, color=[TYPE_COLORS[t] for t in types],
            label=[f"{t} ({pct((train['property_type'] == t).mean())})" for t in types])
    for value, text in [(low, "piso"), (high, "tope")]:
        ax.axvline(value, color=TEST_COLOR, lw=1.5, ls="--")
        ax.text(value, ax.get_ylim()[1] * 0.97, f" {text}: {ps.es_number(value / 1e6, 0)} M ",
                color=TEST_COLOR, ha="left" if text == "piso" else "right", va="top")
    ps.millions(ax, "x")
    ax.set_xlabel("Precio de venta (millones de pesos)")
    ax.set_ylabel("Anuncios de train")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: ps.es_number(v, 0)))
    ax.set_title(f"Los precios de train están recortados entre {ps.es_number(low / 1e6, 0)}"
                 f" y {ps.es_number(high / 1e6, 0)} millones de pesos", loc="left")
    ax.legend(loc="center right")
    ps.save(fig, "price_distribution.png")


def plot_test_predictions(train):
    """Precio observado en train frente a la predicción de la 152 en el test."""
    predicted = pd.read_csv(PROCESSED_DIR / "test_predictions.csv", usecols=[ID_COL, BEST])[BEST]
    high = train[TARGET].max()
    bins = np.linspace(train[TARGET].min() * 0.9, max(predicted.max(), high) * 1.02, 45)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.hist(train[TARGET], bins=bins, density=True, color=TRAIN_COLOR, alpha=0.55,
            label=f"Resto de Bogotá: precio observado (mediana {ps.es_number(train[TARGET].median() / 1e6, 0)} M)")
    ax.hist(predicted, bins=bins, density=True, histtype="step", lw=2.2, color=TEST_COLOR,
            label=f"Chapinero: PREDICCIÓN de la 152 (mediana {ps.es_number(predicted.median() / 1e6, 0)} M)")
    ax.axvline(high, color="black", lw=1, ls="--")
    ax.text(high, ax.get_ylim()[1] * 0.6, f" tope del train\n {ps.es_number(high / 1e6, 0)} M",
            va="top", fontsize=10)
    ps.millions(ax, "x")
    ax.set_yticks([])
    ax.set_ylabel("Densidad")
    ax.set_xlabel("Precio (millones de pesos)")
    ratio = predicted.median() / train[TARGET].median() - 1
    ax.set_title(f"Sin precios en Chapinero, la mejor predicción tiene una mediana {pct(ratio)}"
                 " más alta que la del resto de Bogotá", loc="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_title(ax.get_title(loc="left"), loc="left", pad=62)
    ps.save(fig, "test_prediction_distribution.png")


# --- Faltantes ---

def missing_table(raw, processed):
    """Faltantes de área y baños en cada etapa de recuperación, train y test."""
    rows = []
    for split in ("train", "test"):
        r, p = raw[split], processed[split]
        stages = [
            ("Área", "Properati", (r["surface_total"].isna() & r["surface_covered"].isna())),
            ("Área", "+ texto del anuncio", p["surface"].isna()),
            ("Área", "+ anuncios del mismo punto", p["surface_pt"].isna()),
            ("Baños", "Properati", r["bathrooms"].isna()),
            ("Baños", "+ texto del anuncio", p["bathrooms"].isna()),
            ("Baños", "+ anuncios del mismo punto", p["bathrooms_pt"].isna()),
            ("Cuartos", "Properati", r["rooms"].isna()),
            ("Cuartos", "+ alcobas", p["rooms"].isna()),
            ("Estrato", "Manzana (datos abiertos)", p["ext_stratum"].isna()),
            ("Estrato", "Texto del anuncio", p["txt_stratum"].isna()),
            ("Valor del terreno", "Manzana (datos abiertos)", p["ext_land_value_m2"].isna()),
        ]
        rows += [{"variable": v, "etapa": s, "base": split, "faltante_pct": 100 * m.mean(),
                  "faltantes": int(m.sum()), "filas": len(m)} for v, s, m in stages]
    table = pd.DataFrame(rows)
    table.to_csv(VIEWS_DIR / "missing_values.csv", index=False)
    return table


def plot_missing(table):
    """Barras de faltantes de área y baños por etapa, train y test."""
    data = table[table["variable"].isin(["Área", "Baños"])]
    wide = data.pivot_table(index=["variable", "etapa"], columns="base",
                            values="faltante_pct", sort=False)
    labels = [f"{v}: {s}" for v, s in wide.index]
    y = np.arange(len(wide))[::-1]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for offset, split, color, name in [(0.2, "train", TRAIN_COLOR, "Train (resto de Bogotá)"),
                                       (-0.2, "test", TEST_COLOR, "Test (Chapinero)")]:
        bars = ax.barh(y + offset, wide[split], height=0.38, color=color, label=name)
        ax.bar_label(bars, labels=[pct(v / 100) for v in wide[split]], padding=3, fontsize=10)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Anuncios sin el dato (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    area = wide.loc["Área"]
    ax.set_title(f"El texto y los anuncios del mismo punto bajan el área faltante del test de "
                 f"{pct(area['test'].iloc[0] / 100)} a {pct(area['test'].iloc[-1] / 100)}",
                 loc="left")
    ax.legend(loc="lower right")
    ps.save(fig, "missing_values.png")


# --- Mapa ---

def plot_map(processed):
    """Dispersión de train y test en lat/lon, con el test resaltado."""
    train, test = processed["train"], processed["test"]
    near = distance_to_nearest_m(train, test) < NORTHEAST_RADIUS_M
    print(f"Mapa: {pct(near.mean())} del train a menos de {NORTHEAST_RADIUS_M} m del test")
    fig, ax = plt.subplots(figsize=ps.HALF)
    ax.scatter(train["lon"], train["lat"], s=0.4, color=TRAIN_COLOR, alpha=0.3, lw=0,
               rasterized=True)
    ax.scatter(test["lon"], test["lat"], s=0.6, color=TEST_COLOR, alpha=0.6, lw=0,
               rasterized=True)
    ax.set_aspect(1 / np.cos(np.radians(train["lat"].mean())))
    ax.xaxis.set_major_locator(MaxNLocator(2))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: ps.es_number(v, 2)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: ps.es_number(v, 2)))
    ax.set_xlabel("Longitud (grados)")
    ax.set_ylabel("Latitud (grados)")
    ax.grid(False)
    label_point(ax, test, f"Test: Chapinero\n{ps.es_number(len(test), 0)} anuncios",
                0.35, (0.9, 0.4))
    label_point(ax, train, f"Entrenamiento:\nresto de Bogotá\n{ps.es_number(len(train), 0)} anuncios",
                0.85, (0.85, 0.92))
    ps.save(fig, "map_train_test.png")


def label_point(ax, points, text, y_frac, quantiles):
    """Etiqueta directa a la derecha del mapa con una guía gris hasta un punto de la nube
    (cuantiles de longitud y latitud elegidos para que las guías no se crucen)."""
    center = (points["lon"].quantile(quantiles[0]), points["lat"].quantile(quantiles[1]))
    ax.annotate(text, center, xytext=(1.06, y_frac), textcoords="axes fraction",
                ha="left", va="center", color=ps.ANDES["black"],
                arrowprops=dict(arrowstyle="-", color=ps.ANDES["mid"], lw=0.6,
                                shrinkA=2, shrinkB=0))


# --- Chapinero frente al resto de Bogotá ---

SIGNALS = {"is_house": "Casas", "txt_parking": "Parqueadero", "txt_terrace": "Terraza",
           "txt_elevator": "Ascensor", "txt_gym": "Gimnasio", "txt_view": "Vista",
           "txt_security": "Vigilancia", "txt_gated": "Conjunto cerrado",
           "txt_balcony": "Balcón", "txt_fireplace": "Chimenea", "txt_pool": "Piscina",
           "txt_penthouse": "Penthouse", "surface_missing": "Área faltante (tras el texto)"}
NUMERIC = [
    ("Área (m², mediana)", "surface", "median", 0),
    ("Área con mismo punto (m², mediana)", "surface_pt", "median", 0),
    ("Alcobas (media)", "bedrooms", "mean", 1),
    ("Baños (media)", "bathrooms", "mean", 1),
    ("Estrato de la manzana (media)", "ext_stratum", "mean", 1),
    ("Estrato del texto (media)", "txt_stratum", "mean", 1),
    ("Estrato del texto (% con dato)", "txt_stratum", "coverage", 0),
    ("Valor del terreno (miles de $/m², mediana)", "ext_land_value_m2", "median_k", 0),
    ("Longitud de la descripción (caracteres, mediana)", "txt_desc_length", "median", 0),
]


def summarize(series, how):
    if how == "coverage":
        return 100 * series.notna().mean()
    if how == "median_k":
        return series.median() / 1e3
    return getattr(series, how)()


def comparison_table(processed):
    """Test (Chapinero) frente a train (resto de Bogotá): señales en % y variables numéricas."""
    train, test = processed["train"], processed["test"]
    rows = [{"variable": label, "tipo": "porcentaje",
             "resto_bogota": 100 * train[col].mean(), "chapinero": 100 * test[col].mean()}
            for col, label in SIGNALS.items()]
    rows += [{"variable": label, "tipo": how,
              "resto_bogota": summarize(train[col], how), "chapinero": summarize(test[col], how)}
             for label, col, how, _ in NUMERIC]
    table = pd.DataFrame(rows)
    table["diferencia"] = table["chapinero"] - table["resto_bogota"]
    table.to_csv(VIEWS_DIR / "chapinero_vs_bogota.csv", index=False)
    return table


def plot_signals(table):
    """Barras pareadas de las señales del texto y del tipo de inmueble."""
    data = table[table["tipo"] == "porcentaje"].sort_values("chapinero")
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.barh(y - 0.2, data["resto_bogota"], height=0.38, color=TRAIN_COLOR,
            label="Train: resto de Bogotá")
    ax.barh(y + 0.2, data["chapinero"], height=0.38, color=TEST_COLOR, label="Test: Chapinero")
    for yi, value in zip(y, data["chapinero"]):
        ax.text(value + 1, yi + 0.2, pct(value / 100), va="center", fontsize=9, color=TEST_COLOR)
    ax.set_yticks(y, data["variable"])
    ax.set_xlabel("Anuncios con la característica (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    houses = table.set_index("variable").loc["Casas"]
    text_signals = data[data["variable"].isin([SIGNALS[c] for c in SIGNALS
                                               if c.startswith("txt_")])]
    top = text_signals.nlargest(2, "diferencia")
    gains = " y ".join(f"{v.lower()} (+{ps.es_number(d, 0)} pp)"
                       for v, d in zip(top["variable"], top["diferencia"]))
    ax.set_title(f"Chapinero es casi solo apartamentos ({pct(1 - houses['chapinero'] / 100)});"
                 f" en el texto apenas destacan {gains}", loc="left")
    ax.legend(loc="lower right")
    ps.save(fig, "chapinero_vs_bogota_signals.png")


def plot_numeric_table(table):
    """Tabla limpia de las variables numéricas, lista para diapositiva."""
    decimals = {label: d for label, _, _, d in NUMERIC}
    data = table[table["tipo"] != "porcentaje"]
    cells = [[row.variable, ps.es_number(row.resto_bogota, decimals[row.variable]),
              ps.es_number(row.chapinero, decimals[row.variable])] for row in data.itertuples()]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.axis("off")
    grid = ax.table(cellText=cells, colLabels=["", "Resto de Bogotá\n(train)", "Chapinero\n(test)"],
                    colWidths=[0.60, 0.20, 0.20], loc="center", cellLoc="right")
    grid.auto_set_font_size(False)
    grid.set_fontsize(12)
    grid.scale(1, 2.0)
    for c in range(3):
        grid[0, c].set_height(grid[0, c].get_height() * 1.6)
    for (r, c), cell in grid.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        cell.visible_edges = "B"
        if c == 0:
            cell.set_text_props(ha="left")
            cell._loc = "left"
        if r == 0:
            cell.set_text_props(weight="bold")
        if c == 2 and r > 0:
            cell.set_text_props(color=TEST_COLOR, weight="bold")
    land = data.set_index("variable").loc["Valor del terreno (miles de $/m², mediana)"]
    ax.set_title(f"Chapinero: área parecida, menos alcobas, estrato más alto y terreno "
                 f"{ps.es_number(land['chapinero'] / land['resto_bogota'], 1)} veces más caro",
                 loc="left", fontsize=13)
    ps.save(fig, "chapinero_vs_bogota_table.png")


# --- Predictores externos y de texto ---

TEXT_INTUITION = {
    "txt_parking": "el parqueadero se paga aparte en Bogotá",
    "txt_n_parking": "más parqueaderos, inmueble más grande y caro",
    "txt_terrace": "área exterior privada, amenidad de gama alta",
    "txt_balcony": "amenidad exterior",
    "txt_elevator": "edificio más nuevo o de más pisos",
    "txt_storage": "espacio adicional",
    "txt_fireplace": "acabado típico del nororiente",
    "txt_security": "costo de administración y seguridad",
    "txt_gym": "amenidad de conjunto de gama alta",
    "txt_pool": "amenidad de conjunto de gama alta",
    "txt_gated": "conjunto cerrado, amenidades compartidas",
    "txt_remodeled": "estado del inmueble",
    "txt_duplex": "tipología de mayor área",
    "txt_penthouse": "último piso, prima de lujo",
    "txt_view": "vista panorámica, prima de ubicación",
    "txt_stratum": "estrato declarado por el vendedor",
    "txt_floor": "piso alto, mejor vista y menos ruido",
    "txt_desc_length": "anuncios cuidados acompañan inmuebles caros",
    TEXT_RIDGE: "todo el vocabulario del anuncio resumido en un precio fuera de muestra",
}
EXTERNAL_INTUITION = {
    "ext_stratum": "estrato de la manzana: nivel socioeconómico del entorno",
    "ext_land_value_m2": "valor catastral del suelo: capitaliza la ubicación",
    "dist_zona_t_m": "cercanía al centro de comercio y ocio del norte",
    "dist_parque_93_m": "cercanía a un hito de gama alta",
    "dist_centro_internacional_m": "cercanía al centro de negocios",
    "dist_plaza_bolivar_m": "distancia al centro histórico",
}
OSM_INTUITION = "acceso a servicios y amenidades del entorno"


def source_and_block(col):
    """Bloque, fuente y tipo de un predictor externo o de texto."""
    if col in EXTERNAL_FEATURES:
        return "externo", "Datos abiertos de Bogotá, por manzana"
    if col in LANDMARK_FEATURES:
        return "externo", "Distancia a hitos (coordenadas aproximadas)"
    if col.startswith("osm_"):
        return "externo", "OpenStreetMap (Overpass API)"
    if col == TEXT_RIDGE:
        return "texto", "Ridge TF-IDF del título y la descripción"
    return "texto", "Reglas sobre el título y la descripción"


def best_model_inputs(columns):
    """Predictores de cada componente de la 152 según utils.feature_sets."""
    sets = feature_sets(columns)
    return {name: set(sets[key]) - set(COORD_FEATURES) | {TEXT_RIDGE}
            for name, key in BEST_COMPONENTS.items()}


def predictors_table(processed):
    """Inventario de predictores externos y de texto con su uso en el mejor modelo."""
    columns = processed["train"].columns
    osm = osm_feature_names(columns)
    candidates = EXTERNAL_FEATURES + LANDMARK_FEATURES + osm + TEXT_FEATURES + [TEXT_RIDGE]
    inputs = best_model_inputs(columns)
    rows = []
    for col in candidates:
        block, source = source_and_block(col)
        intuition = (TEXT_INTUITION.get(col) or EXTERNAL_INTUITION.get(col) or OSM_INTUITION)
        row = {"predictor": col, "tipo": block, "fuente": source, "intuicion": intuition}
        row.update({f"en_{name.replace(' ', '_')}": col in used for name, used in inputs.items()})
        row["en_152"] = any(col in used for used in inputs.values())
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(VIEWS_DIR / "predictors_table.csv", index=False)
    osm_used = table.loc[table["fuente"].str.startswith("OpenStreetMap"), "en_152"].any()
    assert not osm_used, "OSM no debería entrar al mejor modelo"
    return table, inputs


def predictors_summary(table, inputs):
    """Conteo por fuente y si entra al mejor modelo; incluye los del mismo punto."""
    summary = (table.groupby(["tipo", "fuente"], sort=False)
               .agg(n=("predictor", "size"), en_152=("en_152", "sum")).reset_index())
    same_point = [c for c in SAME_POINT_FEATURES if c in inputs["boosting 127"]]
    summary.loc[len(summary)] = ["estructural", "Área y baños del mismo punto (solo boosting 127)", len(SAME_POINT_FEATURES),
                                 len(same_point)]
    summary.to_csv(VIEWS_DIR / "predictors_summary.csv", index=False)
    return summary


def plot_predictors_summary(summary):
    """Tabla del inventario de predictores lista para diapositiva."""
    labels = {"externo": "Externo", "texto": "Texto", "estructural": "Estructural"}
    cells = [[labels[r.tipo], r.fuente, str(r.n), str(r.en_152)] for r in summary.itertuples()]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.axis("off")
    grid = ax.table(cellText=cells, colLabels=["Tipo", "Fuente", "Predictores", "En la 152"],
                    colWidths=[0.14, 0.58, 0.14, 0.14], loc="center", cellLoc="left")
    grid.auto_set_font_size(False)
    grid.set_fontsize(11)
    grid.scale(1, 2.3)
    for (r, c), cell in grid.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        cell.visible_edges = "B"
        if r == 0:
            cell.set_text_props(weight="bold")
        elif c == 3 and cells[r - 1][3] == "0":
            cell.set_text_props(color=TEST_COLOR, weight="bold")
    osm = summary[summary["fuente"].str.startswith("OpenStreetMap")].iloc[0]
    external_used = summary.loc[summary["tipo"] == "externo", "en_152"].sum()
    text_used = summary.loc[summary["tipo"] == "texto", "en_152"].sum()
    ax.set_title(f"El mejor modelo usa {external_used} predictores externos y {text_used} de "
                 f"texto; las {osm['n']} variables de OpenStreetMap quedaron fuera",
                 loc="left", fontsize=13)
    ps.save(fig, "predictors_summary.png")


def main():
    ps.setup()
    raw, processed = load_bases()
    plot_price_distribution(raw["train"])
    plot_test_predictions(raw["train"])
    missing = missing_table(raw, processed)
    plot_missing(missing)
    plot_map(processed)
    comparison = comparison_table(processed)
    plot_signals(comparison)
    plot_numeric_table(comparison)
    table, inputs = predictors_table(processed)
    summary = predictors_summary(table, inputs)
    plot_predictors_summary(summary)
    pd.set_option("display.width", 200)
    print(missing.round(1).to_string(index=False))
    print(comparison.round(2).to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
