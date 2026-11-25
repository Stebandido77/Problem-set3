"""Las dos presentaciones del taller en document/, con la plantilla Beamer de los Andes:
best_equipo_06 (el mejor modelo y la decisión de compra) y compare_equipo_06 (comparación
de familias y de esquemas de validación).

Los .tex (document/best_equipo_06.tex y compare_equipo_06.tex) se escriben a mano. Este
script genera lo que depende de los resultados, para que ninguna cifra se escriba a mano:
  document/generated/cifras.tex        una macro por cifra citada en las diapositivas
  document/generated/tabla_*.tex       tablas booktabs (Chapinero, predictores, pasos,
                                       resultados negativos, familias, BETO)
  document/notas_best.md y notas_compare.md   notas del orador, con las mismas cifras
y luego compila los dos PDF con LuaLaTeX (latexmk). Las figuras son los PDF de views/ que
escriben los scripts 31 a 35 y 41; las verificaciones de 33 y 42 alimentan las cifras.

Uso: python 36_build_presentations.py [--no-compile]
"""

import argparse
import re
import shutil
import subprocess

import pandas as pd

from plot_style import es_number
from utils import ROOT, VIEWS_DIR

DOCUMENT_DIR = ROOT / "document"
GENERATED_DIR = DOCUMENT_DIR / "generated"
TEAM = "06"
AUTHORS = "Neislen García y Esteban Labastidas"
DECKS = ("best", "compare")
BEST = "203_ensemble_w70_30_2"
KAGGLE_RULE = "203 (Kaggle, q0,70)"
LOW_RULE = "q0,20"
COMMON = "comunes a los cuatro, sin el ingenuo"
# Camino al mejor modelo: envío de cada paso, qué se probó y por qué. Son los mismos pasos
# de la figura de 32_algorithm_comparison.py; los puntajes salen de views/kaggle_scores.csv.
STEPS = [
    (0, "Mediana de Bogotá para todo", "Punto de partida, sin modelo"),
    (10, "Boosting: inmueble, texto, hitos y lat/lon", "Primer modelo que aprende"),
    (65, "Por tipo, en pesos, cuantil 0,70", "Chapinero es más caro"),
    (83, "Texto completo, estrato y Catastro", "El texto describe la calidad"),
    (152, "Mezcla 70/30 con una red neuronal", "Corrige los inmuebles más caros"),
    (203, "BETO lee el texto junto al ridge", "Lee mejor los precios extremos"),
]
STEP_KEYS = "ABCDEF"
# Ronda de BETO: (modelo, fila sin BETO, fila con BETO).
BETO_PAIRS = [("Boosting", 127, 198), ("Red neuronal", 148, 196), ("Mezcla 70/30", 152, 203)]
# Híbridos de elastic net con boosting sobre sus residuos (los únicos envíos con elastic net).
HYBRIDS = [(49, "Elastic net + boosting"), (54, "Elastic net + boosting q0,65")]
# Lote, pisos y área construida (97) frente a la misma receta sin ellos (94).
LOT_ROWS = ("97_boosting_lot_core_coords_by_type_level_q70_ridge_type_net_s10",
            "94_boosting_ext_core_coords_by_type_level_q70_ridge_type_net_s10")


def m(value, decimals=1):
    """Millones con coma decimal (los CSV de views/ guardan pesos o millones)."""
    value = value / 1e6 if abs(value) > 1e5 else value
    return es_number(round(value, decimals) + 0.0, decimals)


def signed(value, decimals=1):
    """Millones con signo explícito (+ empeora, − mejora)."""
    text = m(value, decimals)
    if text.strip("−0,") == "":
        return text.lstrip("−")
    return text if text.startswith("−") else "+" + text


def pct(share, decimals=0):
    """Porcentaje para LaTeX (espacio fino antes del signo)."""
    return es_number(100 * share, decimals) + r"\,\%"


def load():
    """Tablas de views/ que alimentan las cifras, las tablas y las notas."""
    read = lambda name: pd.read_csv(VIEWS_DIR / name)  # noqa: E731
    return {
        "kaggle": read("kaggle_scores.csv").set_index("model")["public_mae"],
        "final": read("kaggle_final.csv").set_index("board"),
        "schemes": read("validation_schemes.csv").set_index("numero"),
        "corr": read("validation_rank_corr.csv"),
        "choices": read("scheme_best_model.csv"),
        "family": read("family_table.csv"),
        "recipe": read("common_recipe.csv"),
        "blend": read("family_blend_with_127.csv").set_index("numero"),
        "blend_deciles": read("blend_deciles.csv"),
        "blend_members": read("blend_members.csv"),
        "over_cap": read("test_over_cap.csv"),
        "range": read("family_test_range.csv").set_index("numero"),
        "negative": read("negative_results.csv"),
        "osm": read("osm_dispersion.csv"),
        "area": read("area_missing.csv"),
        "chapinero": read("chapinero_vs_bogota.csv").set_index("variable"),
        "importance": read("best_errors_importance.csv"),
        "asym": read("asymmetric_ranking.csv"),
        "segments": read("best_errors_segments.csv"),
        "rule": read("purchase_rule.csv"),
        "rule_segments": read("purchase_rule_segments.csv"),
        "beto_check": read("beto_check.csv"),
        "beto_deciles": read("beto_deciles.csv").set_index("decile"),
        "beto_table": read("beto_table.csv"),
        "beto_corr": read("beto_blend_corr.csv").set_index("blend"),
        "cv": read("cv_results.csv").set_index("model"),
    }


def kaggle_of(t, number):
    names = [name for name in t["kaggle"].index if name.split("_")[0] == f"{number:02d}"]
    return t["kaggle"][names[0]]


def beto_row(t, number):
    """Fila de views/beto_table.csv (espacial y ponderado, en millones) por número."""
    table = t["beto_table"]
    return table[table["model"].str.split("_").str[0] == str(number)].iloc[0]


def rule_row(t, rule, discount=0.0, segment="todas"):
    """Una fila de la simulación de compra (35_purchase_rule.py)."""
    if segment == "todas" and rule in set(t["rule"]["rule"]):
        table = t["rule"]
        rows = table[(table["rule"] == rule) & (table["discount"].round(2) == discount)]
    else:
        table = t["rule_segments"]
        rows = table[(table["rule"] == rule) & (table["discount"].round(2) == discount)
                     & (table["segment"] == segment)]
    return rows.iloc[0]


def segment(t, variable, name, scheme="spatial"):
    table = t["segments"]
    return table[(table["scheme"] == scheme) & (table["variable"] == variable)
                 & (table["segment"] == name)].iloc[0]


def negative_range(t, category, column):
    """Rango del cambio de MAE (millones) de una categoría de views/negative_results.csv."""
    values = t["negative"].loc[t["negative"]["categoria"] == category, column].dropna()
    low, high = signed(values.min()), signed(values.max())
    return low if low == high else f"{low} a {high}"


def area_share(t, base, stage):
    """Porcentaje y número de anuncios sin área (views/area_missing.csv)."""
    row = t["area"][(t["area"]["base"] == base) & (t["area"]["area_desde"] == stage)].iloc[0]
    return row["porcentaje"], row["n_sin_area"]


# --------------------------------------------------------------------------------------
# Cifras
# --------------------------------------------------------------------------------------

def kaggle_numbers(t):
    """Resultado final y puntaje público de cada paso del camino."""
    public, private = t["final"].loc["publico"], t["final"].loc["privado"]
    steps = [kaggle_of(t, n) for n, *_ in STEPS]
    out = {
        "KagBest": m(public["mae_equipo"]), "KagPrivate": m(private["mae_equipo"]),
        "SharePublic": pct(public["share_test"]), "SharePrivate": pct(private["share_test"]),
        "KagSecond": m(private["mae_segundo"]),
        "GapSecond": m(private["mae_segundo"] - private["mae_equipo"]),
        "PrivateRise": m(private["mae_equipo"] - public["mae_equipo"]),
        "PrivateRisePct": pct(private["mae_equipo"] / public["mae_equipo"] - 1, 1),
        "GainNaive": pct(1 - steps[-1] / steps[0]),
        "GainFirst": m(steps[0] - steps[2]), "GainTotal": m(steps[0] - steps[-1]),
        "KagBoost": m(kaggle_of(t, 83)), "KagNet": m(kaggle_of(t, 148)),
        "KagLinear": m(kaggle_of(t, 108)), "KagScaled": m(kaggle_of(t, 32)),
        "KagQsixtyfive": m(kaggle_of(t, 53)),
        "GainBetoKaggle": m(kaggle_of(t, 152) - kaggle_of(t, 203)),
    }
    for key, value in zip(STEP_KEYS, steps):
        out[f"KagStep{key}"] = m(value)
    for key, before, after in zip(STEP_KEYS[1:], steps, steps[1:]):
        out[f"GainStep{key}"] = m(before - after)
    return out


def beto_numbers(t):
    """BETO como lector de texto solo y dentro de cada modelo (validación espacial)."""
    check, deciles = t["beto_check"], t["beto_deciles"]
    weighted = lambda col: (check[col] * check["n"]).sum() / check["n"].sum()  # noqa: E731
    out = {"MaeBeto": m(weighted("mae_beto")), "MaeRidge": m(weighted("mae_ridge")),
           "BetoTopDecile": m(deciles.loc[10, "mae_beto"], 0),
           "RidgeTopDecile": m(deciles.loc[10, "mae_ridge"], 0),
           "BetoMidDecile": m(deciles.loc[6, "mae_beto"], 0),
           "RidgeMidDecile": m(deciles.loc[6, "mae_ridge"], 0),
           "BetoMembersCorr": es_number(t["beto_corr"].loc[BEST, "corr_test"], 2)}
    for label, before, after in BETO_PAIRS:
        key = label.split()[0]
        out[f"Spatial{key}Before"] = m(beto_row(t, before)["spatial"])
        out[f"Spatial{key}After"] = m(beto_row(t, after)["spatial"])
    return out


def blend_numbers(t):
    """Por qué aporta la red: deciles donde la mezcla 203 le gana al boosting 198 solo,
    correlación de errores entre los miembros y predicciones sobre el tope."""
    dec = t["blend_deciles"][t["blend_deciles"]["mezcla"] == 203].set_index("decil")
    wins = dec[dec["ganancia_mezcla"] >= 1]  # menos de 1 M por decil se toma como empate
    members = t["blend_members"].set_index(["mezcla", "esquema"])
    spatial, northeast = members.loc[(203, "spatial")], members.loc[(203, "northeast")]
    cap = t["over_cap"].set_index("modelo")
    rng, blend = t["range"], t["blend"]
    return {
        "MixWinDeciles": " y ".join(str(d) for d in wins.index),
        "MixWinLow": es_number(wins["precio_min"].min(), 0),
        "MixWinGains": " y ".join(m(g) for g in wins["ganancia_mezcla"]),
        "MixLossMax": m(-dec["ganancia_mezcla"].min()),
        "BiasBoostTop": m(dec.loc[10, "sesgo_boosting"], 0),
        "BiasNetTop": m(dec.loc[10, "sesgo_red"], 0),
        "MixSpatialDelta": signed(spatial["mae_mezcla"] - spatial["mae_boosting"]),
        "MixNortheastGain": m(northeast["mae_boosting"] - northeast["mae_mezcla"]),
        "CorrMembersSpatial": es_number(spatial["corr_error"], 2),
        "CorrMembersNortheast": es_number(northeast["corr_error"], 2),
        "NOverBest": str(int(cap.loc[BEST, "n_sobre_tope"])),
        "NTest": es_number(cap.loc[BEST, "n_test"], 0),
        "MaxBest": m(cap.loc[BEST, "max_millones"], 0),
        "MaxNet": m(rng.loc[148, "max_test"], 0), "NOverNet": str(rng.loc[148, "n_sobre_1650"]),
        "MaxBoost": m(rng.loc[127, "max_test"], 0),
        "NOverBoost": str(rng.loc[127, "n_sobre_1650"]),
        "MaxForest": m(rng.loc[114, "max_test"], 0), "MaxCart": m(rng.loc[105, "max_test"], 0),
        "CorrForest": es_number(blend.loc[114, "corr_error_northeast"], 2),
        "CorrNet": es_number(blend.loc[148, "corr_error_northeast"], 2),
        "BlendNetGain": signed(blend.loc[148, "ganancia_weighted"]),
        "BlendNetSpatial": signed(blend.loc[148, "ganancia_spatial"]),
        "BlendForestGain": signed(blend.loc[114, "ganancia_weighted"]),
        "BlendForestSpatial": signed(blend.loc[114, "ganancia_spatial"]),
    }


def validation_numbers(t):
    """Spearman sobre los mismos envíos, modelo que escoge cada esquema y anuncios repetidos."""
    common = t["corr"][t["corr"]["conjunto"] == COMMON].set_index("columna")
    choices = t["choices"]
    spec = choices[choices["conjunto"] == "especificaciones con aleatoria y espacial"]
    schemes = t["schemes"]
    # La subida se mide en todos los modelos de la figura (incluida la 127, sin envío); la
    # distancia al tablero, solo en los que tienen puntaje público.
    both = schemes.dropna(subset=["mae_random", "mae_random_twins"])
    rise = (both["mae_random_twins"] - both["mae_random"])[both.index > 0]
    twins = both.dropna(subset=["kaggle"])
    twins = twins[twins.index > 0]
    level = schemes.loc[[53, 65]]
    return {
        "NCommon": str(int(common.loc["mae_spatial", "n"])),
        "SpRandom": es_number(common.loc["mae_random", "spearman"], 2),
        "SpTwins": es_number(common.loc["mae_random_twins", "spearman"], 2),
        "SpSpatial": es_number(common.loc["mae_spatial", "spearman"], 2),
        "SpWeighted": es_number(common.loc["mae_weighted", "spearman"], 2),
        "NSpecs": str(int(spec["n"].iloc[0])),
        "GapSpatial": m(-common.loc["mae_spatial", "brecha_media"]),
        "RandomBefore": m(schemes.loc[152, "mae_random"]),
        "TwinsBefore": m(schemes.loc[152, "mae_random_twins"]),
        "TwinRiseMin": m(rise.min()), "TwinRiseMax": m(rise.max()),
        "TwinGapMin": m((twins["kaggle"] - twins["mae_random_twins"]).min(), 0),
        "LevelSpatialGain": m(level.loc[65, "mae_spatial"] - level.loc[53, "mae_spatial"]),
        "LevelKaggleLoss": m(level.loc[53, "kaggle"] - level.loc[65, "kaggle"]),
    }


def data_numbers(t):
    """Etapa 1: Chapinero frente al resto de Bogotá, área faltante y predictores."""
    chap = t["chapinero"]
    land = chap.loc["Valor del terreno (miles de $/m², mediana)"]
    test_raw, n_test_raw = area_share(t, "test", "anuncio")
    test_pt, n_test_pt = area_share(t, "test", "anuncio, texto y mismo punto")
    lot, base = (t["cv"].loc[name] for name in LOT_ROWS)
    osm = t["osm"]["razon_iqr_test_entrenamiento"]
    imp = t["importance"][t["importance"]["tipo"] == "apartamento"].set_index("feature")
    params = t["family"].set_index("numero").loc[127, "hiperparametros"]
    value = lambda key: re.search(rf"{key} = ([\d,]+)", params).group(1)  # noqa: E731
    return {
        "LandRatio": es_number(land["chapinero"] / land["resto_bogota"], 1),
        "HousesTrain": es_number(chap.loc["Casas", "resto_bogota"], 0) + r"\,\%",
        "HousesChap": es_number(chap.loc["Casas", "chapinero"], 0) + r"\,\%",
        "AreaMissTest": pct(test_raw), "NAreaMissTest": es_number(n_test_raw, 0),
        "AreaMissTestPt": pct(test_pt), "NAreaMissTestPt": es_number(n_test_pt, 0),
        "LotSpatial": signed(lot["mae_spatial"] - base["mae_spatial"]),
        "OsmIqrMin": es_number(osm.min(), 1), "OsmIqrMax": es_number(osm.max(), 1),
        "ImpText": pct(imp.loc["txt_ridge", "gain_share"]),
        "ImpArea": pct(imp.loc["surface_pt", "gain_share"]),
        "ImpDesc": pct(imp.loc["txt_desc_length", "gain_share"]),
        "HypLeaves": value("num_leaves"), "HypDepth": value("max_depth"),
        "HypRate": value("learning_rate"),
        "HypTrees": es_number(float(value("n_estimators")), 0),
    }


def purchase_numbers(t):
    """Etapa 4: sobre y subpredicción, pérdida asimétrica y regla de compra."""
    asym = t["asym"].set_index("model")
    winner = t["asym"].sort_values("rank_k3").iloc[0]
    over, low = rule_row(t, KAGGLE_RULE), rule_row(t, LOW_RULE)
    apt = rule_row(t, LOW_RULE, segment="apartamento")
    house = rule_row(t, LOW_RULE, segment="casa")
    apt_kaggle = rule_row(t, KAGGLE_RULE, segment="apartamento")
    apt_high = rule_row(t, LOW_RULE, segment="apartamento, oferta >= 900 M")
    apt_low = rule_row(t, LOW_RULE, segment="apartamento, oferta < 900 M")
    total = segment(t, "total", "todas")
    bins = ["<400", "400-500", "500-700", "700-900", "900-1.200", "1.200+"]
    pred_bias = [segment(t, "precio_predicho", b)["bias"] for b in bins]
    q50 = asym.loc["192_boosting_pt_core_coords_by_type_level_q50_ridge_nocoords_s10_twin_mix"]
    return {
        "ShareOverTotal": pct(total["share_over"]), "BiasTotal": m(total["bias"]),
        "ShareOverCheap": pct(segment(t, "precio_observado", "<400")["share_over"]),
        "ShareOverRich": pct(segment(t, "precio_observado", "1.200+")["share_over"]),
        "BiasPredMin": m(min(pred_bias), 0), "BiasPredMax": m(max(pred_bias), 0),
        "MaeApt": m(segment(t, "tipo", "apartamento")["mae"]),
        "MaeHouse": m(segment(t, "tipo", "casa")["mae"]),
        "BiasNortheast": m(segment(t, "total", "todas", scheme="northeast")["bias"]),
        "AsymWinnerCost": m(winner["cost_k3"], 0),
        "BestCostKthree": m(asym.loc[BEST, "cost_k3"], 0),
        "BestRankKthree": str(int(asym.loc[BEST, "rank_k3"])),
        "BestRankKone": str(int(asym.loc[BEST, "rank_k1"])),
        "NRanked": str(len(asym)),
        "SpatialQfifty": m(q50["mae"]),
        "OverClosed": pct(over["share_closed"]), "OverCapital": pct(over["overpay_share_capital"]),
        "MinCapital": pct(t["rule"]["overpay_share_capital"].min()),
        "AptClosed": pct(apt["share_closed"]),
        "AptCapital": pct(apt["overpay_share_capital"], 1),
        "AptKaggleCapital": pct(apt_kaggle["overpay_share_capital"], 1),
        "HouseCapital": pct(house["overpay_share_capital"], 1),
        "AptHighCapital": pct(apt_high["overpay_share_capital"], 1),
        "AptHighMean": m(apt_high["mean_overpay"], 0),
        "AptLowMean": m(apt_low["mean_overpay"], 0),
    }


def figures(t):
    """Cada cifra citada en las diapositivas o en las notas, ya formateada."""
    schemes = t["schemes"]
    recipe = t["recipe"].pivot(index="familia", columns="etapa", values="mae_weighted")
    gain = recipe["antes"] - recipe["mejor"]
    return {
        **kaggle_numbers(t), **beto_numbers(t), **blend_numbers(t), **validation_numbers(t),
        **data_numbers(t), **purchase_numbers(t),
        "SpatialBest": m(schemes.loc[203, "mae_spatial"]),
        "RecipeGainMin": m(gain.min()), "RecipeGainMax": m(gain.max()),
        "NNegative": str(len(t["negative"])),
        "OsmSpatial": negative_range(t, "OpenStreetMap", "delta_spatial"),
        "LocalSpatial": negative_range(t, "Modelo local", "delta_spatial"),
        "BalanceWeighted": negative_range(t, "Balanceo", "delta_weighted"),
        "StackWeighted": negative_range(t, "Apilamiento", "delta_weighted"),
    }


def write_figures(numbers):
    lines = ["% Generado por scripts/36_build_presentations.py a partir de views/. No editar."]
    lines += [rf"\newcommand{{\{name}}}{{{value}}}" for name, value in numbers.items()]
    lines.append(rf"\newcommand{{\Authors}}{{{AUTHORS}}}")
    (GENERATED_DIR / "cifras.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------
# Tablas booktabs
# --------------------------------------------------------------------------------------

def tabular(spec, header, rows, highlight=()):
    """Tabla booktabs; las filas en 'highlight' van en negrita."""
    out = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(header) + r" \\",
           r"\midrule"]
    for k, row in enumerate(rows):
        cells = [rf"\textbf{{{c}}}" if k in highlight else c for c in row]
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


CHAPINERO_ROWS = [
    ("Casas", "Casas (\\%)", 0),
    ("Área (m², mediana)", "Área (m², mediana)", 0),
    ("Alcobas (media)", "Alcobas (media)", 1),
    ("Estrato de la manzana (media)", "Estrato (media)", 1),
    ("Valor del terreno (miles de $/m², mediana)", "Terreno (mil \\$/m²)", 0),
]


def chapinero_table(t):
    chap = t["chapinero"]
    rows = [[label, es_number(chap.loc[key, "resto_bogota"], d),
             es_number(chap.loc[key, "chapinero"], d)] for key, label, d in CHAPINERO_ROWS]
    train, _ = area_share(t, "entrenamiento", "anuncio")
    test, _ = area_share(t, "test", "anuncio")
    rows.append(["Sin área en el anuncio (\\%)", es_number(100 * train, 0),
                 es_number(100 * test, 0)])
    return tabular("lrr", ["", "Resto de Bogotá", "Chapinero"], rows)


def predictors_table(n):
    """Predictores externos y del texto: cuáles entran a la 203 y cuáles se descartaron."""
    rows = [
        ["Estrato de la manzana", "Planeación (datos abiertos)", "Sí"],
        ["Valor del terreno por m²", "Catastro", "Sí"],
        ["Distancias a 4 hitos", "Coordenadas de cada hito", "Sí"],
        ["Lote, pisos y área construida", "Catastro",
         f"No: espacial {n['LotSpatial']} M (ruido)"],
        ["11 distancias y 5 conteos", "OpenStreetMap", f"No: espacial {n['OsmSpatial']} M"],
        ["18 señales (parqueadero, terraza...)", "Reglas sobre el anuncio", "Sí"],
        ["Ridge sobre TF-IDF", "Título y descripción", "Sí"],
        ["BETO", "Título y descripción", "Sí"],
    ]
    return tabular("lll", ["Predictor", "Fuente", "En el mejor modelo"], rows)


def steps_table(t):
    """Los seis pasos: qué se probó, por qué, puntaje público y ganancia frente al anterior."""
    scores = [kaggle_of(t, n) for n, *_ in STEPS]
    rows = []
    for k, (number, what, why) in enumerate(STEPS):
        gain = "--" if k == 0 else m(scores[k - 1] - scores[k])
        rows.append([f"{k + 1}. {what} ({number:02d})", why, m(scores[k]), gain])
    header = ["Qué probamos (envío)", "Por qué", "Público", "Ganancia"]
    return tabular("llrr", header, rows)


def negative_table(n):
    """Lo que no funcionó, con el cambio de MAE frente a su referencia (+ empeora)."""
    rows = [
        ["OpenStreetMap", f"Espacial {n['OsmSpatial']}",
         "Hipótesis: su relación con el precio cambia por zona"],
        ["Modelo local", f"Espacial {n['LocalSpatial']}", "Entrena con muchos menos anuncios"],
        ["Balanceo hacia los caros", f"Ponderado {n['BalanceWeighted']}",
         "Cambia el nivel y deja igual el orden"],
        ["Apilamiento", f"Ponderado {n['StackWeighted']}", "Aprende el tope del holdout"],
        ["Mezcla con el bosque", f"Espacial {n['BlendForestSpatial']}",
         f"Errores casi iguales (correlación {n['CorrForest']})"],
    ]
    return tabular("lll", ["Idea", "Cambio de MAE (M)", "Por qué no sirvió"], rows)


def search_label(row):
    if row["familia"] == "Regresión lineal":
        return "No aplica (MCO)"
    if row["familia"] == "Mezcla":
        return "Pesos fijos 70/30"
    if "Probado" in row["busqueda"]:
        return "Probadas, sin mejora"
    if row["sintonizado"] == "no":
        return "Sin sintonizar"
    trials = re.search(r"Optuna TPE, (\d+) evaluaciones", row["busqueda"]).group(1)
    return f"Optuna TPE ({trials})"


def family_table(t):
    """Una fila por familia, los dos híbridos de elastic net con boosting y las mezclas."""
    rows, highlight = [], []
    columns = ["mae_random_twins", "mae_spatial", "mae_weighted", "kaggle"]
    cell = lambda v: m(v) if pd.notna(v) else "--"  # noqa: E731
    for _, r in t["family"].iterrows():
        cells = [cell(r[c]) for c in columns]
        if r["numero"] == 104:
            cells[3] = "sin envío"
        if r["numero"] == 203:
            cells[2] = "--"  # su ponderado es optimista (BETO vio precios de ese holdout)
            highlight.append(len(rows))
        label = r["familia"] + (" con BETO" if r["numero"] == 203 else "")
        rows.append([label, str(int(r["numero"])), search_label(r), *cells])
        if r["numero"] == 104:
            for number, name in HYBRIDS:
                s = t["schemes"].loc[number]
                rows.append([name, str(number), "--",
                             *[cell(s[c]) for c in columns]])
    header = ["Familia", "Envío", "Búsqueda", "Repetidos", "Espacial", "Ponderado",
              "Público"]
    return tabular("lrlrrrr", header, rows, highlight)


def beto_table(t):
    """Validación espacial de cada modelo sin y con BETO (views/beto_table.csv)."""
    rows = []
    for label, before, after in BETO_PAIRS:
        old, new = beto_row(t, before)["spatial"], beto_row(t, after)["spatial"]
        rows.append([label, f"{before} → {after}", m(old), m(new), signed(new - old)])
    header = ["Modelo", "Filas", "Sin BETO", "Con BETO", "Cambio"]
    return tabular("lcrrr", header, rows, highlight=(2,))


def write_tables(t, numbers):
    tables = [("chapinero", chapinero_table(t)), ("predictores", predictors_table(numbers)),
              ("pasos", steps_table(t)), ("negativos", negative_table(numbers)),
              ("familias", family_table(t)), ("beto", beto_table(t))]
    for name, body in tables:
        (GENERATED_DIR / f"tabla_{name}.tex").write_text(
            "% Generado por scripts/36_build_presentations.py. No editar.\n" + body,
            encoding="utf-8")


# --------------------------------------------------------------------------------------
# Notas del orador
# --------------------------------------------------------------------------------------

def plain(numbers):
    """Las cifras sin marcas de LaTeX, para las notas en Markdown."""
    return {k: v.replace(r"\,\%", " %") for k, v in numbers.items()}


def notes_best(n):
    return [
        ("Resultado", f"""Empezamos por el resultado. La mezcla 203 quedó primera en los dos
tableros de Kaggle. El MAE es el error promedio, en pesos, entre el precio predicho y el
observado. En el tablero público, que usa el {n['SharePublic']} del test, fue de
{n['KagBest']} millones; en el privado, el {n['SharePrivate']} restante y el que define la
competencia, fue de {n['KagPrivate']} M, {n['GapSecond']} M menos que el segundo equipo
({n['KagSecond']} M). Del público al privado el error sube {n['PrivateRise']} M
({n['PrivateRisePct']}) y el puesto se mantiene. La validación espacial prueba el modelo en
zonas de Bogotá que no vio al entrenar; da {n['SpatialBest']} M, por debajo de los dos
tableros. Los precios son de oferta: lo que pide el vendedor. La pregunta de la
presentación es si este modelo sirve para comprar."""),
        ("Chapinero frente al resto de Bogotá", f"""Entrenamos con el resto de Bogotá y
predecimos Chapinero, que el modelo casi no ve. Chapinero es distinto: las casas son el
{n['HousesChap']} de los anuncios frente al {n['HousesTrain']} en el resto de la ciudad, el
estrato medio es más alto y el valor del terreno por metro cuadrado es {n['LandRatio']} veces
el del resto de Bogotá. El test no trae precios, así que el valor del terreno es nuestra
mejor referencia de su nivel. Al {n['AreaMissTest']} de los anuncios del test les falta el
área ({n['NAreaMissTest']} anuncios); al leer el texto y tomar el área de otros anuncios del
mismo punto, la falta baja al {n['AreaMissTestPt']} ({n['NAreaMissTestPt']})."""),
        ("Predictores", f"""Los predictores externos que entran al mejor modelo son el
estrato de la manzana (Planeación), el valor del terreno por metro cuadrado (Catastro) y la
distancia a cuatro hitos de la ciudad. El lote, los pisos y el área construida del Catastro
se probaron: cambiaron la validación espacial en {n['LotSpatial']} M, que es ruido, y
quedaron fuera. Los 16 predictores de OpenStreetMap empeoraron la espacial
({n['OsmSpatial']} M) y también quedaron fuera. Del texto entran 18 señales armadas con
reglas (parqueadero, terraza, ascensor, vista, estrato declarado, piso...) y dos lectores
del texto completo. El ridge sobre TF-IDF es una regresión que le pone un peso a cada
palabra del anuncio; BETO es un modelo de lenguaje en español, afinado aquí para estimar el
precio desde el texto. Los dos se ajustan fuera de muestra: la estimación de un anuncio
nunca usa su propio precio."""),
        ("Qué es el modelo", """El modelo tiene dos partes que ven lo mismo: el inmueble, la
ubicación y las dos lecturas del texto. El boosting son muchos árboles de decisión
pequeños, cada uno corrige el error del anterior; la red neuronal es un modelo que puede
predecir por encima de los precios que vio. Los dos apuntan al cuantil 0,70: un precio que
el 70 % de los inmuebles parecidos no supera. Cada uno se entrena aparte para apartamentos y
para casas; el boosting promedia 10 semillas y la red 5. La predicción final es una mezcla,
un promedio ponderado: 70 % del boosting y 30 % de la red."""),
        ("El camino en seis pasos", f"""La tabla resume el camino en el tablero público. Del
ingenuo ({n['KagStepA']} M) al modelo final ({n['KagStepF']} M) el error baja
{n['GainNaive']}; los pasos 2 y 3 explican {n['GainFirst']} de los {n['GainTotal']} M. El
primer boosting usa el inmueble (tipo, alcobas, baños, área), las 18 señales del texto, las
distancias a los hitos y la latitud y longitud. Antes del paso 3 multiplicamos a mano ese
boosting por 1,15 y bajó a {n['KagScaled']} M: es una calibración con el tablero, sin
modelo, que mostró que el nivel estaba bajo. El cuantil 0,70 lo escogimos con la
retroalimentación del tablero: el 0,65 dio {n['KagQsixtyfive']} M, una diferencia de ruido.
El paso 4 agrega el texto completo, el estrato y el valor del terreno; el 5, la red; el 6,
BETO. Las diferencias de menos de 2 M en el tablero son ruido; el paso 6 gana
{n['GainStepF']} M."""),
        ("Qué aportó BETO", f"""Comparamos los dos lectores del texto solos, fuera de
muestra, con las mismas particiones espaciales. BETO se equivoca en promedio
{n['MaeBeto']} M y el ridge {n['MaeRidge']} M. Por decil de precio, el ridge acierta más en
los precios del medio (decil 6: {n['RidgeMidDecile']} contra {n['BetoMidDecile']} M) y BETO
en los extremos (decil 10: {n['BetoTopDecile']} contra {n['RidgeTopDecile']} M): el ridge
encoge más sus predicciones hacia la media. Con los dos lectores, en la validación espacial
el boosting pasa de {n['SpatialBoostingBefore']} a {n['SpatialBoostingAfter']} M y la red de
{n['SpatialRedBefore']} a {n['SpatialRedAfter']} M."""),
        ("Predictores más importantes", f"""La gráfica es la importancia por ganancia del
boosting sin BETO, ajustado una vez con todos los datos de entrenamiento. No tenemos esa
medición para el boosting con BETO, porque obtenerla exige reajustarlo. El texto resumido
por el ridge concentra el {n['ImpText']} de la ganancia en apartamentos: describe calidad,
acabados y amenidades que las columnas no recogen. Le sigue el área ({n['ImpArea']}), que
se asocia directamente con el precio. El tercero es el largo de la descripción
({n['ImpDesc']}): los anuncios más largos se asocian con inmuebles más caros. El boosting
usa {n['HypLeaves']} hojas por árbol, profundidad máxima {n['HypDepth']}, tasa de aprendizaje
{n['HypRate']} y {n['HypTrees']} árboles; la búsqueda de hiperparámetros no mejoró el
tablero, así que no se sintonizó."""),
        ("Lo que no funcionó", f"""OpenStreetMap empeoró la validación espacial
({n['OsmSpatial']} M). Pensábamos que en Chapinero todo queda cerca, pero los datos no lo
respaldan: en el test las distancias varían entre {n['OsmIqrMin']} y {n['OsmIqrMax']} veces lo
que varían en entrenamiento (rango intercuartílico). Nuestra hipótesis es que la relación
entre amenidades y precio cambia de una zona a otra; no la medimos. El modelo local entrena
con muchos menos anuncios y empeora la espacial ({n['LocalSpatial']} M). El holdout
ponderado son anuncios de entrenamiento vecinos a Chapinero, ponderados para parecerse al
test. Darles más peso a los caros cambia el nivel y deja igual el orden: el holdout
ponderado empeora ({n['BalanceWeighted']} M). El apilamiento aprende el tope de precios del
holdout, que no existe en el test ({n['StackWeighted']} M). Mezclar con el bosque deja el
holdout ponderado igual: sus errores son casi los del boosting (correlación
{n['CorrForest']})."""),
        ("Sobrepredicción", f"""Con las predicciones de la validación espacial, la mezcla
sobrepredice el {n['ShareOverTotal']} de los inmuebles, con un sesgo de +{n['BiasTotal']} M.
Es lo esperable al apuntar al cuantil 0,70. El panel por precio pedido muestra que
sobrepredice el {n['ShareOverCheap']} de los de menos de 400 M y el {n['ShareOverRich']} de
los de más de 1.200 M; ese patrón es en parte mecánico, porque cualquier modelo acerca sus
predicciones a la media. Para la calibración sirve el panel por precio predicho, que es lo
que ve la empresa: en cada rango el sesgo es positivo, de +{n['BiasPredMin']} a
+{n['BiasPredMax']} M, y la sobrepredicción media crece con la predicción. El error es mayor
en las casas (MAE {n['MaeHouse']} M frente a {n['MaeApt']} M en apartamentos)."""),
        ("Pérdida asimétrica", f"""Si sobrepredecir cuesta k veces lo que subpredecir, el
MAE deja de ser la medida correcta. Con k = 1, el MAE, la mezcla queda en el puesto
{n['BestRankKone']} de {n['NRanked']}: el boosting en cuantil 0,50 tiene menos error espacial
({n['SpatialQfifty']} M) porque la validación espacial premia bajar el nivel. Kaggle se
comportó distinto: el cuantil 0,70 dio {n['KagStepC']} M y el 0,65 {n['KagQsixtyfive']} M;
Chapinero es más caro que las zonas con que se valida. Con k = 3 gana el boosting en cuantil
0,20 ({n['AsymWinnerCost']} M) y la mezcla cae al puesto {n['BestRankKthree']}
({n['BestCostKthree']} M), como predice la teoría: con costo k veces mayor, el cuantil óptimo
es 1/(1+k). Los cuantiles bajos son el boosting sin BETO."""),
        ("Regla de compra", f"""Simulamos una regla: la empresa ofrece la predicción con
un descuento y compra si la oferta alcanza el precio pedido; la diferencia es sobrepago. La
oferta solo alcanza el precio cuando el modelo sobreestima, así que toda compra es un
sobrepago frente al precio pedido. Con la predicción de la mezcla, sin descuento, se cierra
el {n['OverClosed']} de las ofertas y la empresa sobrepaga el {n['OverCapital']} del
capital. A igual número de compras, bajar el cuantil sobrepaga menos que descontar. Ninguna
regla baja del {n['MinCapital']} del capital."""),
        ("Recomendación", f"""Sí recomendamos usar el modelo para comprar, con una oferta
conservadora y un avalúo. La oferta es el cuantil 0,20: un precio que el 80 % de los
inmuebles parecidos supera. En apartamentos se cierra el {n['AptClosed']} de las ofertas y
la empresa sobrepaga el {n['AptCapital']} del capital; al ofertar la predicción de Kaggle
(cuantil 0,70) en apartamentos sobrepaga el {n['AptKaggleCapital']}. En casas sobrepaga el
{n['HouseCapital']}, y en apartamentos con oferta desde 900 M, el {n['AptHighCapital']}, con
{n['AptHighMean']} M por compra frente a {n['AptLowMean']} M por debajo de ese valor; por eso
los dejamos fuera. La regla se simuló en zonas de Bogotá que el modelo no vio, con el
boosting sin BETO, y se escogió con esas mismas predicciones, así que es algo optimista.
Antes de cerrar, un avalúo fija el precio. Riesgos: los precios son de oferta, los datos son
de 2019 a 2021 y en el holdout vecino a Chapinero el sesgo cambia de signo
({n['BiasNortheast']} M)."""),
    ]


def notes_compare(n):
    return [
        ("Vista general", f"""Predecimos el precio de oferta, lo que pide el vendedor, de
vivienda en Chapinero, entrenando con el resto de Bogotá. El mejor modelo, la mezcla 203,
quedó primera en los dos tableros de Kaggle: {n['KagBest']} M en el público y
{n['KagPrivate']} M en el privado, {n['GapSecond']} M menos que el segundo. El MAE es el error
promedio en pesos. El boosting son muchos árboles pequeños, cada uno corrige el error del
anterior; la red neuronal puede predecir por encima de los precios que vio; la mezcla
pondera 70 % el boosting y 30 % la red. La validación aleatoria prueba en anuncios al azar
de las mismas zonas que el modelo ya vio; la espacial, en zonas que no vio, y da
{n['SpatialBest']} M."""),
        ("Tabla por familia", f"""Cada fila es la mejor especificación de su familia con la
receta común. La columna de repetidos es la validación aleatoria con las copias de un mismo
inmueble en la misma partición. El holdout ponderado son anuncios de entrenamiento vecinos a
Chapinero, ponderados para parecerse al test. El boosting gana la espacial y el tablero; la
red gana el holdout ponderado y en el tablero queda en {n['KagNet']} M. El elastic net puro
no tiene envío; los envíos 49 y 54 son híbridos: un elastic net con un boosting sobre sus
residuos. Para la 203 no reportamos el ponderado, porque BETO se afinó con precios de ese
holdout."""),
        ("Predicción máxima", f"""Los precios de entrenamiento van de 300 a 1.650 M. La
etiqueta de cada barra cuenta cuántas de las {n['NTest']} predicciones del test pasan de
1.650 M. El CART ({n['MaxCart']} M) y el bosque ({n['MaxForest']} M) promedian precios vistos
y se quedan bajo el tope. El boosting suma las salidas de muchos árboles, y esa suma puede
pasar el tope: la 127 llega a {n['MaxBoost']} M con {n['NOverBoost']} predicciones encima. La
red llega a {n['MaxNet']} M, con {n['NOverNet']} encima. La mezcla 203 tiene
{n['NOverBest']} predicciones sobre el tope, con máximo de {n['MaxBest']} M. Los sondeos del
tablero indican que el test tampoco pasa de 1.650 M, así que esas predicciones deben
recortarse al tope."""),
        ("Por qué aporta la red", f"""La figura mezcla, sin reentrenar, el boosting 127 con
cada familia. Con la red, la validación espacial cambia {n['BlendNetSpatial']} M, que es
ruido, y el holdout ponderado {n['BlendNetGain']} M. En el tablero, del mejor boosting solo
({n['KagBoost']} M) a la primera mezcla ({n['KagStepE']} M). Al separar por decil de precio,
la mezcla 203 le gana al boosting solo en los deciles {n['MixWinDeciles']} (desde
{n['MixWinLow']} M), por {n['MixWinGains']} M, y pierde hasta {n['MixLossMax']} M en los demás.
En el decil más caro, el boosting subpredice {n['BiasBoostTop']} M y la red
{n['BiasNetTop']} M. La correlación de errores entre los dos es {n['CorrMembersSpatial']} en
la espacial y {n['CorrMembersNortheast']} en el holdout; la del bosque con el boosting es
{n['CorrForest']}, más alta que la de la red ({n['CorrNet']})."""),
        ("Receta común", f"""Cada flecha va de la versión inicial de una familia (número a la
derecha) a su mejor versión (número a la izquierda); los números son filas de la tabla de
resultados. La receta: un modelo por tipo, el precio en pesos con el cuantil 0,70 como
objetivo (un precio que el 70 % de los parecidos no supera), el estrato y el valor del
terreno, y el ridge del texto. En el holdout ponderado todas ganan entre {n['RecipeGainMin']}
y {n['RecipeGainMax']} M; en la espacial ninguna pasa al mejor boosting."""),
        ("BETO en cada modelo", f"""El ridge sobre TF-IDF es una regresión que le pone un
peso a cada palabra del anuncio. BETO es un modelo de lenguaje en español, afinado aquí para
estimar el precio desde el texto. Lo sumamos al ridge en los dos miembros de la mezcla. En la
validación espacial el boosting pasa de {n['SpatialBoostingBefore']} a
{n['SpatialBoostingAfter']} M, la red de {n['SpatialRedBefore']} a {n['SpatialRedAfter']} M y la
mezcla de {n['SpatialMezclaBefore']} a {n['SpatialMezclaAfter']} M. En el tablero público la
mezcla baja {n['GainBetoKaggle']} M. Las predicciones de los dos miembros con BETO tienen una
correlación de {n['BetoMembersCorr']} en el test."""),
        ("Validación contra Kaggle", f"""Comparamos los cuatro esquemas sobre los mismos
{n['NCommon']} envíos. El Spearman mide si un esquema ordena los envíos como el tablero
público (1 es el mismo orden): aleatoria {n['SpRandom']}, repetidos juntos {n['SpTwins']},
espacial {n['SpSpatial']} y holdout ponderado {n['SpWeighted']}. Con el mismo conjunto, la
espacial es la que mejor ordena y las dos aleatorias quedan atrás, muy cerca entre sí. Todos
los esquemas son optimistas; la espacial, en {n['GapSpatial']} M en promedio. ¿Escogen la
aleatoria y la espacial el mismo mejor modelo? Entre estos envíos, sí: las dos escogen la
mezcla 152, que también es la mejor del tablero entre ellos. Entre las {n['NSpecs']}
especificaciones con las dos validaciones, escogen la misma receta de boosting con distinta
regla de anuncios repetidos (126 y 127). El puntaje privado solo está registrado para el
modelo final, así que no hay comparación contra el privado."""),
        ("Anuncios repetidos", f"""Muchos anuncios del entrenamiento son el mismo inmueble
publicado varias veces. Si una copia queda en el ajuste y otra en la validación, el modelo
acierta por memoria. Al poner las copias en la misma partición, el MAE aleatorio sube entre
{n['TwinRiseMin']} y {n['TwinRiseMax']} M; en la mezcla sin BETO, de {n['RandomBefore']} a
{n['TwinsBefore']} M. Aun así, todos los modelos quedan a más de {n['TwinGapMin']} M de su
puntaje en el tablero."""),
        ("Nivel en la espacial", f"""La validación espacial prefiere modelos de nivel más
bajo. En la escalera de cuantiles, menos sesgo da menos MAE espacial, mientras el holdout
ponderado casi no cambia. El cuantil 0,65 gana {n['LevelSpatialGain']} M en la espacial y en
el tablero queda {n['LevelKaggleLoss']} M detrás del 0,70, una diferencia de ruido. Por eso
usamos la espacial para ordenar modelos de nivel parecido, y el holdout y el tablero para el
nivel."""),
        ("Lo que no funcionó", f"""Ninguna de las {n['NNegative']} variantes mejoró a la vez la
espacial y el holdout ponderado: la esquina de abajo a la izquierda está vacía. OpenStreetMap
empeora la espacial ({n['OsmSpatial']} M). En el test sus distancias varían entre
{n['OsmIqrMin']} y {n['OsmIqrMax']} veces lo que varían en entrenamiento, así que la idea de
que en Chapinero todo queda cerca no se sostiene; por qué empeora no está medido."""),
        ("Intercambios", f"""El boosting tiene el menor error en zonas que no vio y
subpredice los inmuebles más caros. La red aporta en los dos deciles más caros y sola queda
en {n['KagNet']} M en el tablero. La regresión lineal es la más fácil de leer y queda en
{n['KagLinear']} M. BETO mejora a los dos miembros de la mezcla, requiere GPU y deja
optimista el holdout ponderado. Para comprar, el orden cambia: si sobrepagar cuesta el
triple, la mezcla cae al puesto {n['BestRankKthree']} y gana un boosting en cuantil 0,20."""),
    ]


def write_notes(numbers):
    n = plain(numbers)
    titles = {"best": "El mejor modelo y la decisión de compra",
              "compare": "Comparación de modelos y de validaciones"}
    for deck, build in [("best", notes_best), ("compare", notes_compare)]:
        lines = [f"# Notas del orador: {deck}_equipo_{TEAM}", "",
                 f"{titles[deck]}. {AUTHORS}, Equipo {TEAM}. Una sección por diapositiva de "
                 "contenido, en orden. Generado por `scripts/36_build_presentations.py`; las "
                 "cifras salen de `views/`.", ""]
        for k, (title, text) in enumerate(build(n), start=1):
            lines += [f"## {k}. {title}", "", " ".join(text.split()), ""]
        (DOCUMENT_DIR / f"notas_{deck}.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------------------
# Compilación
# --------------------------------------------------------------------------------------

def compile_decks():
    """Compila con LuaLaTeX: la plantilla usa Source Sans 3, que pdfLaTeX de MiKTeX no
    encuentra si sus mapas de fuentes no están actualizados."""
    if shutil.which("latexmk") is None:
        print("Sin latexmk: se generaron cifras, tablas y notas, pero no los PDF.")
        return
    for deck in DECKS:
        source = f"{deck}_equipo_{TEAM}.tex"
        subprocess.run(["latexmk", "-lualatex", "-interaction=nonstopmode", "-halt-on-error",
                        source], cwd=DOCUMENT_DIR, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["latexmk", "-c", source], cwd=DOCUMENT_DIR, check=True,
                       stdout=subprocess.DEVNULL)
        print(f"document/{deck}_equipo_{TEAM}.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    tables = load()
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    numbers = figures(tables)
    write_figures(numbers)
    write_tables(tables, numbers)
    write_notes(numbers)
    if not args.no_compile:
        compile_decks()


if __name__ == "__main__":
    main()
