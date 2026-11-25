"""Las dos presentaciones del taller en document/, con la plantilla Beamer de los Andes:
best_equipo_06 (el mejor modelo y la decisión de compra) y compare_equipo_06 (comparación
de familias y de esquemas de validación).

Los .tex (document/best_equipo_06.tex y compare_equipo_06.tex) se escriben a mano. Este
script genera lo que depende de los resultados, para que ninguna cifra se escriba a mano:
  document/generated/cifras.tex        una macro por cifra citada en las diapositivas
  document/generated/tabla_*.tex       tablas booktabs (familias, pasos, resultados
                                       negativos, BETO, regla de compra)
  document/notas_best.md y notas_compare.md   notas del orador, con las mismas cifras
y luego compila los dos PDF con LuaLaTeX (latexmk). Las figuras son los PDF de views/ que
escriben los scripts 31 a 35 y 41.

Uso: python 36_build_presentations.py [--no-compile]
"""

import argparse
import shutil
import subprocess

import pandas as pd

from plot_style import es_number
from utils import ROOT, VIEWS_DIR

DOCUMENT_DIR = ROOT / "document"
GENERATED_DIR = DOCUMENT_DIR / "generated"
TEAM = "06"
DECKS = ("best", "compare")
BEST = "203_ensemble_w70_30_2"
KAGGLE_RULE = "203 (Kaggle, q0,70)"
LOW_RULE = "q0,20"
# Camino al mejor modelo: envío de cada paso, qué se probó y por qué. Son los mismos pasos
# de la figura de 32_algorithm_comparison.py; los puntajes salen de views/kaggle_scores.csv.
STEPS = [
    (0, "Mediana de Bogotá para todo", "Punto de partida, sin modelo"),
    (10, "Boosting con el anuncio y coordenadas", "Aprender el precio de cada inmueble"),
    (65, "Por tipo, en pesos, cuantil 0,70", "Chapinero es más caro que el train"),
    (83, "Texto completo, estrato y Catastro", "Describe lo que no está en columnas"),
    (152, "Mezcla 70/30 con una red neuronal", "La red se equivoca en otros inmuebles"),
    (203, "BETO lee el texto junto al ridge", "Lee mejor los precios extremos"),
]
STEP_KEYS = "ABCDEF"
# Ronda de BETO: (modelo, fila sin BETO, fila con BETO).
BETO_PAIRS = [("Boosting", 127, 198), ("Red neuronal", 148, 196), ("Mezcla 70/30", 152, 203)]


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
        "family": read("family_table.csv"),
        "recipe": read("common_recipe.csv"),
        "blend": read("family_blend_with_127.csv").set_index("numero"),
        "negative": read("negative_results.csv"),
        "asym": read("asymmetric_ranking.csv"),
        "segments": read("best_errors_segments.csv"),
        "rule": read("purchase_rule.csv"),
        "rule_segments": read("purchase_rule_segments.csv"),
        "beto_check": read("beto_check.csv"),
        "beto_deciles": read("beto_deciles.csv").set_index("decile"),
        "beto_table": read("beto_table.csv"),
        "range": read("family_test_range.csv").set_index("numero"),
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


# --------------------------------------------------------------------------------------
# Cifras
# --------------------------------------------------------------------------------------

def kaggle_numbers(t):
    """Resultado final y puntaje público de cada paso del camino."""
    public, private = t["final"].loc["publico"], t["final"].loc["privado"]
    steps = [kaggle_of(t, n) for n, *_ in STEPS]
    out = {
        "KagBest": m(public["mae_equipo"]), "KagPrivate": m(private["mae_equipo"]),
        "RankPublic": str(int(public["puesto"])), "RankPrivate": str(int(private["puesto"])),
        "SharePublic": pct(public["share_test"]), "SharePrivate": pct(private["share_test"]),
        "KagSecond": m(private["mae_segundo"]),
        "GapSecond": m(private["mae_segundo"] - private["mae_equipo"]),
        "PrivateRise": m(private["mae_equipo"] - public["mae_equipo"]),
        "PrivateRisePct": pct(private["mae_equipo"] / public["mae_equipo"] - 1, 1),
        "GainNaive": pct(1 - steps[-1] / steps[0]),
        "KagBoost": m(kaggle_of(t, 83)), "KagNet": m(kaggle_of(t, 148)),
        "KagLinear": m(kaggle_of(t, 108)), "KagScaled": m(kaggle_of(t, 32)),
        "KagQsixtyfive": m(kaggle_of(t, 53)), "KagText": m(kaggle_of(t, 75)),
        "GainText": m(kaggle_of(t, 65) - kaggle_of(t, 75)),
        "GainBetoKaggle": m(kaggle_of(t, 152) - kaggle_of(t, 203)),
        "GainFirst": m(steps[0] - steps[2]), "GainLast": m(steps[2] - steps[-1]),
        "GainTotal": m(steps[0] - steps[-1]),
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
           "BetoCorr": es_number(weighted("corr_log"), 2),
           "BetoTopDecile": m(deciles.loc[10, "mae_beto"], 0),
           "RidgeTopDecile": m(deciles.loc[10, "mae_ridge"], 0),
           "BetoMidDecile": m(deciles.loc[6, "mae_beto"], 0),
           "RidgeMidDecile": m(deciles.loc[6, "mae_ridge"], 0),
           "BetoTopBias": m(deciles.loc[10, "bias_beto"], 0),
           "RidgeTopBias": m(deciles.loc[10, "bias_ridge"], 0)}
    for label, before, after in BETO_PAIRS:
        key = label.split()[0]
        out[f"Spatial{key}Before"] = m(beto_row(t, before)["spatial"])
        out[f"Spatial{key}After"] = m(beto_row(t, after)["spatial"])
    return out


def figures(t):
    """Cada cifra citada en las diapositivas o en las notas, ya formateada."""
    schemes, blend = t["schemes"], t["blend"]
    recipe = t["recipe"].pivot(index="familia", columns="etapa", values="mae_weighted")
    gain = recipe["antes"] - recipe["mejor"]
    lin = t["cv"].loc["108_linear_ext_core_coords_splines_by_type_level_q70_ridge_oof"]
    nogeo = t["cv"].loc["130_linear_ext_core_coords_by_type_level_q70_ridge_oof_nogeo"]
    no_naive = t["corr"][t["corr"]["conjunto"] == "sin el ingenuo"].set_index("esquema")
    asym = t["asym"].set_index("model")
    winner = t["asym"].sort_values("rank_k3").iloc[0]
    over, low = rule_row(t, KAGGLE_RULE), rule_row(t, LOW_RULE)
    apt, house = rule_row(t, LOW_RULE, segment="apartamento"), rule_row(t, LOW_RULE,
                                                                          segment="casa")
    same_share = t["rule"][t["rule"]["rule"] == KAGGLE_RULE]
    same_share = same_share.loc[(same_share["share_closed"] - low["share_closed"]).abs()
                                .idxmin()]
    total = segment(t, "total", "todas")
    cheap = segment(t, "precio_observado", "<400")
    rich = segment(t, "precio_observado", "1.200+")
    high_pred = [segment(t, "precio_predicho", s) for s in ("900-1.200", "1.200+")]
    area_ok, area_missing = segment(t, "area", "anuncio o texto"), segment(t, "area",
                                                                           "faltante")
    forest = blend.loc[114]
    return {
        **kaggle_numbers(t), **beto_numbers(t),
        "SpatialBest": m(schemes.loc[203, "mae_spatial"]),
        "RandomBefore": m(schemes.loc[152, "mae_random"]),
        "TwinsBefore": m(schemes.loc[152, "mae_random_twins"]),
        "RecipeGainMin": m(gain.min()), "RecipeGainMax": m(gain.max()),
        "LinSpatial": m(lin["mae_spatial"]), "LinNoGeoSpatial": m(nogeo["mae_spatial"]),
        "CorrForest": es_number(forest["corr_error_northeast"], 2),
        "CorrNet": es_number(blend.loc[148, "corr_error_northeast"], 2),
        "BlendNetGain": signed(blend.loc[148, "ganancia_weighted"]),
        "BlendForestGain": signed(forest["ganancia_weighted"]),
        "BlendForestSpatial": signed(forest["ganancia_spatial"]),
        "NNegative": str(len(t["negative"])),
        "MaxNet": m(t["range"].loc[148, "max_test"], 0),
        "MaxForest": m(t["range"].loc[114, "max_test"], 0),
        "MaxCart": m(t["range"].loc[105, "max_test"], 0),
        "OsmSpatial": negative_range(t, "OpenStreetMap", "delta_spatial"),
        "LocalSpatial": negative_range(t, "Modelo local", "delta_spatial"),
        "BalanceSpatial": negative_range(t, "Balanceo", "delta_spatial"),
        "BalanceWeighted": negative_range(t, "Balanceo", "delta_weighted"),
        "StackWeighted": negative_range(t, "Apilamiento", "delta_weighted"),
        "SpRandom": es_number(no_naive.loc["Aleatoria simple", "spearman"], 2),
        "SpTwins": es_number(no_naive.loc["Aleatoria con gemelos", "spearman"], 2),
        "SpSpatial": es_number(no_naive.loc["Espacial (10 zonas)", "spearman"], 2),
        "SpWeighted": es_number(no_naive.loc["Nororiental ponderado", "spearman"], 2),
        "GapSpatial": m(-no_naive.loc["Espacial (10 zonas)", "brecha_media"]),
        "AsymWinner": winner["label"].split(" ", 1)[0],
        "AsymWinnerQ": winner["label"].rsplit("q", 1)[-1],
        "AsymWinnerCost": m(winner["cost_k3"], 0),
        "BestCostKthree": m(asym.loc[BEST, "cost_k3"], 0),
        "BestRankKthree": str(int(asym.loc[BEST, "rank_k3"])),
        "BestRankKone": str(int(asym.loc[BEST, "rank_k1"])),
        "NRanked": str(len(asym)),
        "OverClosed": pct(over["share_closed"]),
        "OverCapital": pct(over["overpay_share_capital"]),
        "OverMean": m(over["mean_overpay"], 0),
        "LowClosed": pct(low["share_closed"]),
        "LowCapital": pct(low["overpay_share_capital"]),
        "LowMean": m(low["mean_overpay"], 0),
        "SameDiscount": pct(same_share["discount"]),
        "SameClosed": pct(same_share["share_closed"]),
        "SameCapital": pct(same_share["overpay_share_capital"]),
        "MinCapital": pct(t["rule"]["overpay_share_capital"].min()),
        "AptClosed": pct(apt["share_closed"]),
        "AptCapital": pct(apt["overpay_share_capital"], 1),
        "HouseCapital": pct(house["overpay_share_capital"], 1),
        "ShareOverCheap": pct(cheap["share_over"]),
        "ShareOverRich": pct(rich["share_over"]),
        "BiasRich": m(rich["bias"], 0),
        "ShareOverTotal": pct(total["share_over"]),
        "BiasTotal": m(total["bias"]),
        "MeanOver": m(total["mean_over"]), "MeanUnder": m(total["mean_under"]),
        "OverHighPred": m(min(s["mean_over"] for s in high_pred), 0),
        "MaeApt": m(segment(t, "tipo", "apartamento")["mae"]),
        "MaeHouse": m(segment(t, "tipo", "casa")["mae"]),
        "MaeAreaOk": m(area_ok["mae"]), "MaeAreaMissing": m(area_missing["mae"]),
        "ShareAreaMissing": pct(area_missing["n"] / total["n"]),
        "BiasNortheast": m(segment(t, "total", "todas", scheme="northeast")["bias"]),
    }


def write_figures(numbers):
    lines = ["% Generado por scripts/36_build_presentations.py a partir de views/. No editar."]
    lines += [rf"\newcommand{{\{name}}}{{{value}}}" for name, value in numbers.items()]
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


def search_label(row):
    if row["familia"] == "Regresión lineal":
        return "No aplica (MCO)"
    if row["familia"] == "Mezcla":
        return "Pesos fijos 70/30"
    if "Probado" in row["busqueda"]:
        return "Búsquedas probadas, sin mejora"
    if row["sintonizado"] == "no":
        return "Receta común, sin sintonizar"
    return row["busqueda"].split(";")[0].split(" (")[0]


def family_table(t):
    rows, highlight = [], []
    for k, (_, r) in enumerate(t["family"].iterrows()):
        cells = [m(r[c]) if pd.notna(r[c]) else "--"
                 for c in ["mae_random_twins", "mae_spatial", "mae_weighted", "kaggle"]]
        if r["numero"] == 203:
            cells[2] = "--"  # su ponderado es optimista (BETO vio precios de ese holdout)
            highlight.append(k)
        label = r["familia"] + (" con BETO" if r["numero"] == 203 else "")
        rows.append([label, str(int(r["numero"])), search_label(r), *cells])
    header = ["Familia", "Envío", "Búsqueda", "Gemelos", "Espacial", "Ponderado", "Público"]
    return tabular("lrlrrrr", header, rows, highlight)


def steps_table(t, first, last):
    """Pasos del camino: qué se probó, por qué, puntaje público y ganancia frente al paso
    anterior."""
    scores = [kaggle_of(t, n) for n, *_ in STEPS]
    rows = []
    for k in range(first, last + 1):
        number, what, why = STEPS[k]
        gain = "--" if k == 0 else m(scores[k - 1] - scores[k])
        rows.append([f"{k + 1}. {what}", why, f"{number:02d}", m(scores[k]), gain])
    header = ["Qué probamos", "Por qué", "Envío", "Público", "Ganancia"]
    return tabular("llrrr", header, rows)


def negative_table(n):
    """Lo que no funcionó, con el cambio de MAE frente a su referencia (+ empeora)."""
    rows = [
        ["OpenStreetMap", f"Espacial {n['OsmSpatial']}", "En Chapinero casi todo queda cerca"],
        ["Modelo local", f"Espacial {n['LocalSpatial']}", "Entrena con muchos menos anuncios"],
        ["Balanceo hacia los caros", f"Ponderado {n['BalanceWeighted']}",
         "Cambia el nivel, no el orden"],
        ["Apilamiento", f"Ponderado {n['StackWeighted']}", "Aprende el tope del holdout"],
        ["Mezcla con el bosque", f"Espacial {n['BlendForestSpatial']}",
         f"Mismos errores (correlación {n['CorrForest']})"],
    ]
    return tabular("lll", ["Idea", "Cambio de MAE (M)", "Por qué no sirvió"], rows)


def beto_table(t):
    """Validación espacial de cada modelo sin y con BETO (views/beto_table.csv)."""
    rows = []
    for label, before, after in BETO_PAIRS:
        old, new = beto_row(t, before)["spatial"], beto_row(t, after)["spatial"]
        rows.append([label, f"{before} → {after}", m(old), m(new), signed(new - old)])
    header = ["Modelo", "Filas", "Sin BETO", "Con BETO", "Cambio"]
    return tabular("lcrrr", header, rows, highlight=(2,))


RULE_ROWS = [
    (KAGGLE_RULE, "todas", "Mezcla 203 (q0,70)", "Todos"),
    (LOW_RULE, "todas", "Cuantil 0,20", "Todos"),
    (LOW_RULE, "apartamento, con área", "Cuantil 0,20", "Apartamentos con área"),
    (LOW_RULE, "apartamento, sin área", "Cuantil 0,20", "Apartamentos sin área"),
    (LOW_RULE, "casa, con área", "Cuantil 0,20", "Casas con área"),
    (LOW_RULE, "casa, sin área", "Cuantil 0,20", "Casas sin área"),
]


def rule_table(t):
    rows = []
    for rule, seg, label, seg_label in RULE_ROWS:
        r = rule_row(t, rule, 0.0, seg)
        rows.append([label, seg_label, pct(r["share_closed"]), m(r["mean_overpay"], 0),
                     pct(r["overpay_share_capital"], 1)])
    header = ["Regla", "Segmento", "Compras", "Sobrepago medio", "Sobrepago / capital"]
    return tabular("llrrr", header, rows, highlight=(2, 3))


def write_tables(t, numbers):
    tables = [("familias", family_table(t)), ("pasos_a", steps_table(t, 0, 2)),
              ("pasos_b", steps_table(t, 3, 5)), ("negativos", negative_table(numbers)),
              ("beto", beto_table(t)), ("regla", rule_table(t))]
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
        ("Resultado", f"""Empezamos por el resultado. La mezcla 203 quedó en el primer puesto
de los dos tableros de Kaggle. El MAE es el error promedio, en pesos, entre el precio
predicho y el observado. En el tablero público, que se calcula con el {n['SharePublic']} del
test, fue de {n['KagBest']} millones. En el privado, con el {n['SharePrivate']} restante y el
que define la competencia, fue de {n['KagPrivate']} M, {n['GapSecond']} M menos que el segundo
equipo ({n['KagSecond']} M). Del público al privado el error subió {n['PrivateRise']} M
({n['PrivateRisePct']}): el modelo se sostuvo y conservó el puesto. Nuestra validación
espacial, que prueba el modelo en zonas de Bogotá que no vio al entrenar, da
{n['SpatialBest']} M: es optimista frente a los dos tableros; Chapinero es más caro y
distinto que cualquier zona del train. Los precios son de oferta, lo que pide el vendedor. La
pregunta de la presentación es si este modelo sirve para comprar."""),
        ("Qué es el modelo", """El modelo tiene dos partes que leen lo mismo. Ve el inmueble
(tipo, alcobas, baños y área, también la recuperada de anuncios del mismo punto), la
ubicación (estrato y valor del terreno de la manzana, distancias a cuatro hitos) y el texto
del anuncio. El texto entra por dos lectores. El ridge sobre TF-IDF es una regresión que le
pone un peso a cada palabra; BETO es un modelo de lenguaje en español que afinamos para
estimar el precio desde el texto. Los dos lectores se ajustan fuera de muestra, así que su
estimación de un anuncio nunca usa el precio de ese anuncio. Con todo eso se entrenan un
boosting, muchos árboles pequeños en el que cada uno corrige el error del anterior, y una
red neuronal, que puede predecir por encima de los precios que vio. Los dos apuntan al
cuantil 0,70, un precio que el 70 % de los inmuebles parecidos no supera, y se entrenan
por separado para apartamentos y casas. La predicción final es una mezcla: 70 % del
boosting y 30 % de la red."""),
        ("Progresión", f"""Este es el camino en el tablero público, en seis pasos. Del
ingenuo, que predice la mediana de Bogotá para todo ({n['KagStepA']} M), al modelo final
({n['KagStepF']} M) el error baja {n['GainNaive']}. Casi toda la ganancia está en los tres
primeros pasos: el primer boosting y corregir el nivel. Los tres últimos suman menos, pero
cada uno supera los 2 M que tomamos como ruido del tablero."""),
        ("Pasos 1 a 3", f"""El primer boosting con las variables del anuncio y las
coordenadas baja el error de {n['KagStepA']} a {n['KagStepB']} M. Luego vimos que el modelo
subpredecía: Chapinero es más caro que el resto de Bogotá. Escalar a mano el envío 10 por
1,15 lo bajó a {n['KagScaled']} M; esa versión es una calibración con el tablero, no un
modelo, y solo nos sirvió para confirmar el problema de nivel. La solución dentro del modelo
fue entrenar uno por tipo de inmueble, con el precio en pesos y no en logaritmo, y apuntar
al cuantil 0,70. Ese cuantil lo escogimos con la retroalimentación del tablero: el 0,65
(envío 53) dio {n['KagQsixtyfive']} M y el 0,70 (envío 65) {n['KagStepC']} M, una diferencia
que está dentro del ruido."""),
        ("Pasos 4 a 6", f"""El cuarto paso lee el texto completo del anuncio con un ridge
sobre TF-IDF y agrega el estrato (Planeación) y el valor del terreno (Catastro), con 10
semillas: el error baja {n['GainStepD']} M, y de eso {n['GainText']} M aparecen solo con el texto (envío 75,
{n['KagText']} M). El quinto mezcla el boosting con una red neuronal, que se equivoca en
inmuebles distintos y puede predecir por encima del tope de 1.650 M del train: baja
{n['GainStepE']} M. El sexto agrega BETO como segundo lector del texto, en el boosting y en
la red: baja {n['GainStepF']} M, hasta {n['KagStepF']} M. Las dos últimas ganancias son
pequeñas, pero las dos superan los 2 M de ruido."""),
        ("Qué aportó BETO", f"""Comparamos los dos lectores del texto solos, fuera de
muestra, con las mismas particiones espaciales. BETO se equivoca en promedio
{n['MaeBeto']} M y el ridge {n['MaeRidge']} M. Por decil de precio, el ridge acierta más en
los precios del medio (decil 6: {n['RidgeMidDecile']} contra {n['BetoMidDecile']} M) y BETO
en los extremos (decil 10: {n['BetoTopDecile']} contra {n['RidgeTopDecile']} M). El ridge
encoge mucho las predicciones hacia la media; BETO, menos. Por eso conviene tener los dos:
en la validación espacial el boosting pasa de {n['SpatialBoostingBefore']} a
{n['SpatialBoostingAfter']} M y la red de {n['SpatialRedBefore']} a {n['SpatialRedAfter']} M.
En el tablero público, la mezcla pasa de {n['KagStepE']} a {n['KagStepF']} M."""),
        ("Lo que no funcionó", f"""También probamos ideas que no sirvieron, y explican el
camino. OpenStreetMap: las distancias y conteos de amenidades empeoran la validación
espacial ({n['OsmSpatial']} M); en Chapinero casi todo queda cerca y la variable distingue
poco. El modelo local, entrenado solo con anuncios cercanos o parecidos a Chapinero, se
queda con muchos menos datos y empeora la espacial ({n['LocalSpatial']} M). Darles más peso
a los inmuebles caros, con cuantiles de 0,55 a 0,65, cambia la espacial sobre todo por el
nivel, y en el holdout ponderado empeora ({n['BalanceWeighted']} M). El apilamiento aprende a
corregir el tope del holdout, que no existe en el test ({n['StackWeighted']} M en el
ponderado). Mezclar con el bosque no ayuda: se equivoca en los mismos inmuebles que el
boosting (correlación de errores {n['CorrForest']})."""),
        ("Dónde se equivoca", f"""Usamos las predicciones de la validación espacial: zonas
que el modelo no vio al entrenar. La mezcla 203 sobrepredice el {n['ShareOverTotal']} de los
inmuebles, con un sesgo de +{n['BiasTotal']} M. Sobrepredice el {n['ShareOverCheap']} de los
de menos de 400 M y solo el {n['ShareOverRich']} de los de más de 1.200 M, donde el sesgo es
de {n['BiasRich']} M. El error es mayor en las casas (MAE {n['MaeHouse']} M frente a
{n['MaeApt']} M en apartamentos) y cuando falta el área ({n['MaeAreaMissing']} frente a
{n['MaeAreaOk']} M). Para comprar importa el panel del precio predicho, que es lo único que
ve la empresa: cuanto más alta la predicción, mayor la sobrepredicción media."""),
        ("Sobre contra subpredicción", f"""Si sobrepredecir cuesta más que subpredecir,
el MAE no es la medida correcta. Recalculamos el costo de cada modelo cuando la
sobrepredicción cuesta k veces la subpredicción. Con k = 1 es el MAE y la 203 queda en el
puesto {n['BestRankKone']} de {n['NRanked']}. Con k = 3 gana el boosting {n['AsymWinner']},
que apunta al cuantil {n['AsymWinnerQ']} ({n['AsymWinnerCost']} M), y la 203 cae al puesto
{n['BestRankKthree']} ({n['BestCostKthree']} M). Es lo que dice la teoría: con costo k veces
mayor, el cuantil óptimo es 1/(1+k), es decir 0,25. Los cuantiles bajos son el boosting sin
BETO; no reentrenamos la mezcla 203 en cuantiles bajos."""),
        ("Regla de compra", f"""Simulamos una regla: la empresa ofrece la predicción con un
descuento y compra si la oferta alcanza el precio pedido; la diferencia es sobrepago. Hay
selección adversa: la oferta solo alcanza el precio cuando el modelo sobreestima. Si ofrece
lo que predice la 203, compra el {n['OverClosed']} de los anuncios y sobrepaga el
{n['OverCapital']} del capital. Con el cuantil 0,20 compra el {n['LowClosed']} y sobrepaga el
{n['LowCapital']}. A igual número de compras, bajar el cuantil sobrepaga menos que
descontar: la 203 con {n['SameDiscount']} de descuento compra el {n['SameClosed']} con
{n['SameCapital']}. Ninguna regla baja del {n['MinCapital']} del capital."""),
        ("Regla por segmento", f"""Con el cuantil 0,20, en apartamentos la empresa compra el
{n['AptClosed']} y sobrepaga el {n['AptCapital']} del capital; en casas, el
{n['HouseCapital']}. Que falte el área cambia poco el sobrepago. La regla y el segmento se
escogieron con las mismas predicciones espaciales con que se reportan, así que estas cifras
son algo optimistas."""),
        ("Recomendación", f"""¿Se puede confiar en el modelo para comprar? Para fijar el
precio, no: ofrecer su predicción compra justo los inmuebles que sobreestima. Para escoger
candidatos, sí, con una regla conservadora: el cuantil 0,20, solo en apartamentos y con una
tasación física antes de cerrar. No lo usaríamos en casas ni con predicciones de más de
900 M, donde la sobrepredicción media pasa de {n['OverHighPred']} M. Riesgos: los precios son
de oferta, no de cierre, así que el sobrepago real puede ser mayor; falta el área en el
{n['ShareAreaMissing']} de los anuncios; los datos son de 2019 a 2021; y el nivel en
Chapinero es incierto: en el holdout junto a Chapinero el sesgo cambia de signo
({n['BiasNortheast']} M)."""),
    ]


def notes_compare(n):
    return [
        ("Vista general", f"""Predecimos el precio de oferta, lo que pide el vendedor, de
vivienda en Chapinero, entrenando con el resto de Bogotá. El mejor modelo, la mezcla 203,
combina un boosting (70 %) y una red neuronal (30 %), los dos con dos lectores del texto: un
ridge sobre TF-IDF y BETO. Quedó primera en los dos tableros de Kaggle: {n['KagBest']} M en
el público y {n['KagPrivate']} M en el privado, {n['GapSecond']} M menos que el segundo. El
MAE es el error promedio en pesos. En nuestra validación espacial, que prueba en zonas que
el modelo no vio, da {n['SpatialBest']} M. Esta presentación explica por qué ganan el
boosting y la red, y cómo escogimos entre esquemas de validación."""),
        ("Tabla por familia", f"""Cada fila es la mejor especificación de su familia con la
receta común: un modelo por tipo, precio en pesos, cuantil 0,70, estrato y valor del terreno,
y el ridge del texto. Las columnas son tres formas de validar y el puntaje público. La
aleatoria con gemelos prueba en anuncios al azar, de zonas que el modelo ya vio, pero
mantiene juntas las copias del mismo inmueble. La espacial deja zonas completas fuera. El
ponderado es un holdout al lado de Chapinero, ponderado como el test. El boosting gana en la
espacial y la red en el ponderado, pero en el tablero público la red sola ({n['KagNet']} M)
quedó detrás del boosting ({n['KagBoost']} M). La elastic net no se subió a Kaggle. Para la
203 no reportamos el ponderado: BETO se afinó con precios de ese holdout y la cifra sería
optimista."""),
        ("Rango de predicción", f"""Los precios de entrenamiento están entre 300 y 1.650
millones. Un árbol predice promedios o cuantiles de lo que vio, así que el CART y el bosque
nunca pasan ese tope y el boosting apenas lo roza. La red sí puede pasarlo. La regresión
lineal no extrapola en precio porque su receta la recorta al rango del train, pero sus
splines de latitud y longitud se estiman casi sin datos de Chapinero: sin coordenadas, su
error espacial baja de {n['LinSpatial']} a {n['LinNoGeoSpatial']} M, y en el tablero público
fue de las peores ({n['KagLinear']} M)."""),
        ("Por qué la red y no el bosque", f"""Calculamos, sin reentrenar, una mezcla 70/30
del boosting 127 con cada familia, sobre las predicciones fuera de muestra. La correlación de
errores con el boosting en el holdout nororiental es {n['CorrForest']} para el bosque y
{n['CorrNet']} para la red. Con la red el ponderado cambia {n['BlendNetGain']} M; con el
bosque, {n['BlendForestGain']} M (negativo es mejora). La red es la única que es buena sola y
a la vez se equivoca en otros inmuebles: llega a la cola alta, donde el boosting subpredice.
Las lineales y el kNN son más distintos, pero tan malos solos que empeoran la mezcla."""),
        ("Receta común", f"""Antes, cada familia usaba el logaritmo del precio y los mismos
predictores. Trasladamos la receta del boosting a todas: un modelo por tipo, el precio en
pesos con el cuantil 0,70 como objetivo, el estrato y el valor del terreno de la manzana, y
el texto completo resumido por el ridge. Todas mejoran en el holdout ponderado, entre
{n['RecipeGainMin']} y {n['RecipeGainMax']} M. En la espacial la mejora es menor y el
boosting sigue adelante: la ventaja es del algoritmo, no solo de la receta."""),
        ("BETO en cada modelo", f"""El último paso fue BETO, un modelo de lenguaje en
español que afinamos para estimar el precio desde el texto, como segundo lector junto al
ridge. Solo lo probamos en los dos miembros de la mezcla. En la validación espacial, el
boosting pasa de {n['SpatialBoostingBefore']} a {n['SpatialBoostingAfter']} M, la red de
{n['SpatialRedBefore']} a {n['SpatialRedAfter']} M y la mezcla de {n['SpatialMezclaBefore']} a
{n['SpatialMezclaAfter']} M. En el tablero público, la mezcla bajó {n['GainBetoKaggle']} M.
BETO mejora a los dos miembros, pero no los vuelve más distintos: su mezcla gana porque cada
uno es mejor, no por diversidad."""),
        ("Validación contra Kaggle", f"""Cada punto es un envío con puntaje público; la línea
punteada es la igualdad. Todos los esquemas son optimistas; Chapinero es más caro y
distinto que el resto de Bogotá; la espacial, en promedio {n['GapSpatial']} M. El Spearman
mide si un esquema ordena los envíos como el tablero (1 es el mismo orden): aleatoria simple
{n['SpRandom']}, con gemelos {n['SpTwins']}, espacial {n['SpSpatial']} y nororiental
ponderado {n['SpWeighted']}. Cada esquema tiene un número distinto de envíos, así que la
comparación es aproximada. La aleatoria simple es la que peor ordena."""),
        ("Gemelos", f"""Muchos anuncios del train son el mismo inmueble publicado varias veces.
Si una copia queda en el ajuste y otra en la validación, el modelo acierta por memoria. Al
mantener los gemelos juntos, el error de la mezcla 152 sube de {n['RandomBefore']} a
{n['TwinsBefore']} M. En Chapinero casi no hay copias de inmuebles del train, así que la
versión con gemelos es la relevante. La 203 no tiene validación aleatoria calculada."""),
        ("Nivel en la espacial", """La espacial ordena razonablemente, pero prefiere modelos
con un nivel más bajo que el que premia el tablero: el cuantil 0,60 le gana al 0,70 en la
espacial y no en el ponderado, y en el tablero el 0,65 (envío 53) quedó detrás del 0,70
(65), aunque por menos de 1 M. Por eso usamos la espacial para ordenar modelos de nivel
parecido, y el ponderado y el tablero para el nivel."""),
        ("Lo que no funcionó", f"""Ninguna de estas {n['NNegative']} ideas mejoró a la vez la
espacial y el ponderado; la esquina de abajo a la izquierda está vacía. OpenStreetMap empeora
la espacial ({n['OsmSpatial']} M): en Chapinero casi todo queda cerca. El modelo local pierde
demasiados datos. El apilamiento aprende el tope del holdout, que no existe en el test.
Balancear hacia los caros cambia la espacial por nivel, no por orden. La monotonía y el kNN no
ganan en los dos criterios."""),
        ("Intercambios", f"""Cada familia gana algo y paga algo. El boosting ordena mejor en
zonas que no vio, pero no pasa el tope del train. La red sí lo pasa, pero sola es peor en el
tablero ({n['KagNet']} M). La regresión lineal es fácil de leer y viaja mal a Chapinero por
sus coordenadas. BETO lee mejor el texto, pero hay que afinarlo en GPU y se entrenó con
particiones propias, así que su ponderado no es limpio. Y el modelo para el tablero no es el
de comprar: con costo de sobrepago triple, la 203 cae al puesto {n['BestRankKthree']}.
Escogimos con varios esquemas y usamos el tablero como juez final."""),
    ]


def write_notes(numbers):
    n = plain(numbers)
    titles = {"best": "El mejor modelo y la decisión de compra",
              "compare": "Comparación de modelos y de validaciones"}
    for deck, build in [("best", notes_best), ("compare", notes_compare)]:
        lines = [f"# Notas del orador: {deck}_equipo_{TEAM}", "",
                 f"{titles[deck]}. Una sección por diapositiva de contenido, en orden. "
                 "Generado por `scripts/36_build_presentations.py`; las cifras salen de "
                 "`views/`.", ""]
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
