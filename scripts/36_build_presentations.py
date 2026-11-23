"""Las dos presentaciones del taller en document/, con la plantilla Beamer de los Andes:
best_equipo_06 (el mejor modelo y la decisión de compra) y compare_equipo_06 (comparación
de familias y de esquemas de validación).

Los .tex (document/best_equipo_06.tex y compare_equipo_06.tex) se escriben a mano. Este
script genera lo que depende de los resultados, para que ninguna cifra se escriba a mano:
  document/generated/cifras.tex        una macro por cifra citada en las diapositivas
  document/generated/tabla_*.tex       tablas booktabs (familias, Chapinero, regla)
  document/notas_best.md y notas_compare.md   notas del orador, con las mismas cifras
y luego compila los dos PDF con LuaLaTeX (latexmk). Las figuras son los PDF de views/ que
escriben los scripts 31 a 35.

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
BEST = "152_ensemble_w70_30_2"
KAGGLE_RULE = "152 (Kaggle, q0,70)"
LOW_RULE = "q0,20"


def m(value, decimals=1):
    """Millones con coma decimal (los CSV de views/ guardan pesos o millones)."""
    value = value / 1e6 if abs(value) > 1e5 else value
    return es_number(round(value, decimals) + 0.0, decimals)


def pct(share, decimals=0):
    """Porcentaje para LaTeX (espacio fino antes del signo)."""
    return es_number(100 * share, decimals) + r"\,\%"


def load():
    """Tablas de views/ que alimentan las cifras, las tablas y las notas."""
    read = lambda name: pd.read_csv(VIEWS_DIR / name)  # noqa: E731
    return {
        "kaggle": read("kaggle_scores.csv").set_index("model")["public_mae"],
        "schemes": read("validation_schemes.csv").set_index("numero"),
        "corr": read("validation_rank_corr.csv"),
        "family": read("family_table.csv"),
        "recipe": read("common_recipe.csv"),
        "blend": read("family_blend_with_127.csv").set_index("numero"),
        "negative": read("negative_results.csv"),
        "chapinero": read("chapinero_vs_bogota.csv").set_index("variable"),
        "asym": read("asymmetric_ranking.csv"),
        "segments": read("best_errors_segments.csv"),
        "rule": read("purchase_rule.csv"),
        "rule_segments": read("purchase_rule_segments.csv"),
        "importance": read("best_errors_importance.csv"),
        "cv": read("cv_results.csv").set_index("model"),
    }


def kaggle_of(t, number):
    names = [name for name in t["kaggle"].index if name.split("_")[0] == f"{number:02d}"]
    return t["kaggle"][names[0]]


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


def segment(t, variable, name):
    table = t["segments"]
    return table[(table["scheme"] == "spatial") & (table["variable"] == variable)
                 & (table["segment"] == name)].iloc[0]


# --------------------------------------------------------------------------------------
# Cifras
# --------------------------------------------------------------------------------------

def figures(t):
    """Cada cifra citada en las diapositivas o en las notas, ya formateada."""
    best, naive, boost = kaggle_of(t, 152), kaggle_of(t, 0), kaggle_of(t, 83)
    schemes, blend, chap = t["schemes"], t["blend"], t["chapinero"]
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
    importance = t["importance"]
    imp_apt = importance[importance["tipo"] == "apartamento"].iloc[0]
    houses = chap.loc["Casas"]
    land = chap.loc["Valor del terreno (miles de $/m², mediana)"]
    return {
        "KagBest": m(best), "KagNaive": m(naive), "KagBoost": m(boost),
        "KagLinear": m(kaggle_of(t, 108)), "KagNet": m(kaggle_of(t, 148)),
        "GainNaive": pct(1 - best / naive), "GainBoost": m(boost - best),
        "SpatialBest": m(schemes.loc[152, "mae_spatial"]),
        "RandomBest": m(schemes.loc[152, "mae_random"]),
        "TwinsBest": m(schemes.loc[152, "mae_random_twins"]),
        "WeightedBest": m(schemes.loc[152, "mae_weighted"]),
        "RecipeGainMin": m(gain.min()), "RecipeGainMax": m(gain.max()),
        "LinSpatial": m(lin["mae_spatial"]), "LinNoGeoSpatial": m(nogeo["mae_spatial"]),
        "CorrForest": es_number(blend.loc[114, "corr_error_northeast"], 2),
        "CorrNet": es_number(blend.loc[148, "corr_error_northeast"], 2),
        "BlendNetGain": m(blend.loc[148, "ganancia_weighted"]),
        "BlendForestGain": m(blend.loc[114, "ganancia_weighted"]),
        "NNegative": str(len(t["negative"])),
        "SpRandom": es_number(no_naive.loc["Aleatoria simple", "spearman"], 2),
        "SpTwins": es_number(no_naive.loc["Aleatoria con gemelos", "spearman"], 2),
        "SpSpatial": es_number(no_naive.loc["Espacial (10 zonas)", "spearman"], 2),
        "SpWeighted": es_number(no_naive.loc["Nororiental ponderado", "spearman"], 2),
        "SpNortheast": es_number(no_naive.loc["Nororiental sin ponderar", "spearman"], 2),
        "AsymWinner": winner["label"].split(" ", 1)[0],
        "AsymWinnerQ": winner["label"].rsplit("q", 1)[-1],
        "BestRankKthree": str(int(asym.loc[BEST, "rank_k3"])),
        "BestRankKone": str(int(asym.loc[BEST, "rank_k1"])),
        "OverClosed": pct(over["share_closed"]),
        "OverCapital": pct(over["overpay_share_capital"]),
        "OverCapitalD": pct(over["overpay_share_capital"], 1),
        "LowClosed": pct(low["share_closed"]),
        "LowCapital": pct(low["overpay_share_capital"]),
        "SameDiscount": pct(same_share["discount"]),
        "SameClosed": pct(same_share["share_closed"]),
        "SameCapital": pct(same_share["overpay_share_capital"]),
        "MinCapital": pct(t["rule"]["overpay_share_capital"].min(), 1),
        "AptClosed": pct(apt["share_closed"]),
        "AptCapital": pct(apt["overpay_share_capital"], 1),
        "HouseCapital": pct(house["overpay_share_capital"], 1),
        "HousesTrain": es_number(houses["resto_bogota"], 0) + r"\,\%",
        "HousesChap": es_number(houses["chapinero"], 0) + r"\,\%",
        "LandRatio": es_number(land["chapinero"] / land["resto_bogota"], 1),
        "ImpApt": pct(imp_apt["gain_share"]),
        "ShareOverCheap": pct(cheap["share_over"]),
        "ShareOverRich": pct(rich["share_over"]),
        "ShareOverTotal": pct(total["share_over"]),
        "BiasTotal": m(total["bias"]),
        "OverHighPred": m(min(s["mean_over"] for s in high_pred), 0),
        "MaeAreaOk": m(area_ok["mae"]), "MaeAreaMissing": m(area_missing["mae"]),
        "ShareAreaMissing": pct(area_missing["n"] / total["n"]),
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
        rows.append([r["familia"], str(int(r["numero"])), search_label(r), *cells])
        if r["numero"] == 152:
            highlight.append(k)
    header = ["Familia", "Envío", "Búsqueda", "Gemelos", "Espacial", "Ponderado", "Kaggle"]
    return tabular("lrlrrrr", header, rows, highlight)


CHAPINERO_ROWS = [
    ("Casas", "Casas (\\%)", 0),
    ("Área (m², mediana)", "Área (m², mediana)", 0),
    ("Alcobas (media)", "Alcobas (media)", 1),
    ("Estrato de la manzana (media)", "Estrato (media)", 1),
    ("Valor del terreno (miles de $/m², mediana)", "Terreno (mil \\$/m²)", 0),
    ("Área faltante (tras el texto)", "Área faltante (\\%)", 0),
]


def chapinero_table(t):
    chap = t["chapinero"]
    rows = [[label, es_number(chap.loc[key, "resto_bogota"], d),
             es_number(chap.loc[key, "chapinero"], d)] for key, label, d in CHAPINERO_ROWS]
    return tabular("lrr", ["", "Train", "Chapinero"], rows)


RULE_ROWS = [
    (KAGGLE_RULE, "todas", "Kaggle (q0,70)", "Todos"),
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
    header = ["Regla", "Segmento", "Compras", "Sobrepago medio (M)", "Sobrepago / capital"]
    return tabular("llrrr", header, rows, highlight=(2, 3))


def write_tables(t):
    for name, body in [("familias", family_table(t)), ("chapinero", chapinero_table(t)),
                       ("regla", rule_table(t))]:
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
        ("Resultado", f"""Empezamos por el resultado. La pregunta guía es si podemos confiar
en el modelo lo suficiente para comprar vivienda en Chapinero. El mejor modelo (la mezcla
152) tiene un error medio de {n['KagBest']} millones en el tablero público, {n['GainNaive']}
menos que predecir la mediana ({n['KagNaive']} M). En validación espacial, que deja zonas
completas de Bogotá por fuera, el error es {n['SpatialBest']} M. Pero si la empresa ofrece lo
que predice el modelo, solo compra cuando el modelo sobreestima, y paga de más: cierra el
{n['OverClosed']} de los anuncios con un sobrepago del {n['OverCapital']} del capital. Con el
cuantil 0,20 cierra el {n['LowClosed']} con {n['LowCapital']}. La respuesta corta: sirve para
filtrar candidatos con una regla conservadora, no para fijar el precio."""),
        ("El problema", f"""Los puntos grises son los anuncios de entrenamiento; los rojos,
los de Chapinero que hay que predecir. El modelo casi no ve Chapinero al entrenar. Además,
Chapinero es distinto: casi todo son apartamentos (casas: {n['HousesTrain']} en el train y
{n['HousesChap']} en Chapinero), el estrato es más alto y el terreno cuesta
{n['LandRatio']} veces más. Por eso validamos dejando zonas enteras por fuera."""),
        ("Qué es el modelo", f"""El boosting aprende bien el orden de los precios, pero un
árbol no predice por encima de lo que vio, y el train está recortado en 1.650 M. La red puede
extrapolar y corrige en parte la subpredicción de los inmuebles caros. Los dos apuntan al
cuantil 0,70 del precio y no a la mediana: Chapinero es más caro que el resto de Bogotá, y
apuntar alto compensa ese salto de nivel. Cada uno se entrena por separado para apartamentos y
para casas; el boosting promedia 10 semillas y aplica la regla de anuncios repetidos.
Qué ve el modelo, en detalle: tipo, alcobas, baños y área; área y baños recuperados de
anuncios del mismo punto; estrato y valor del terreno de la manzana (Catastro); distancias a
cuatro hitos; 18 señales del texto y un ridge TF-IDF fuera de muestra sobre el texto completo.
Quitamos latitud y longitud crudas porque no viajan a una zona que no se vio: la ubicación
entra por el estrato, el valor del terreno y las distancias."""),
        ("Progresión", f"""De dónde salió cada mejora, en puntaje público. Del ingenuo al
primer boosting. Subir el nivel (primero escalando, luego con la pérdida cuantílica) y
separar apartamentos y casas. El texto completo del anuncio fue el último salto grande de un
modelo solo. La mezcla con la red dio los últimos {n['GainBoost']} millones frente al mejor
boosting ({n['KagBoost']} M). El envío 32 es una calibración con el tablero, no un modelo, y
lo mostramos solo para explicar el efecto del nivel."""),
        ("Importancia", f"""Importancia por ganancia del boosting, ajustado una vez con todo el
train. El ridge sobre el texto completo resume en un número lo que dicen el título y la
descripción: {n['ImpApt']} de la ganancia en apartamentos. Le siguen el área (también la
recuperada de anuncios del mismo punto), el largo de la descripción y la ubicación a través
de distancias y valor del terreno. Es una importancia de un ajuste con una semilla: sirve
para ordenar, no para medir efectos causales."""),
        ("Dónde se equivoca", f"""Usamos las predicciones fuera de muestra de la validación
espacial. En total sobrepredice el {n['ShareOverTotal']} de los inmuebles, con un sesgo de
{n['BiasTotal']} millones. Sobrepredice el {n['ShareOverCheap']} de los de menos de 400 M y
solo el {n['ShareOverRich']} de los de más de 1.200 M. El error crece con el precio y es mayor
en las casas y cuando falta el área. Para comprar importa el precio predicho, que es lo único
que ve la empresa: cuanto más alta la predicción, más grande la sobrepredicción media."""),
        ("Pérdida asimétrica", f"""Recalculamos el costo medio de cada modelo cuando
sobrepredecir cuesta dos o tres veces más que subpredecir. Entre familias el orden casi no
cambia; lo que cambia es el nivel: ganan los cuantiles bajos, como predice la teoría (con
costo k veces mayor, el cuantil óptimo es 1/(1+k): 0,33 y 0,25). Con costo triple gana el
{n['AsymWinner']} (cuantil {n['AsymWinnerQ']}); la 152 de Kaggle queda en el puesto
{n['BestRankKthree']}. El modelo que gana en Kaggle no es el que conviene para comprar."""),
        ("Regla de compra", f"""Simulamos: la empresa ofrece la predicción con un descuento;
si la oferta alcanza el precio pedido, compra y la diferencia es sobrepago. Hay selección
adversa: solo se compra cuando el modelo sobreestima. Cada curva es un cuantil y cada punto
un descuento de 0 a 20 %. Ofrecer lo que predice el modelo de Kaggle cierra el
{n['OverClosed']} con {n['OverCapital']} de sobrepago. A igual número de compras, bajar el
cuantil sobrepaga menos que descontar: el cuantil 0,20 cierra el {n['LowClosed']} con
{n['LowCapital']}, y la 152 con {n['SameDiscount']} de descuento cierra el {n['SameClosed']}
con {n['SameCapital']}. Ninguna regla baja del {n['MinCapital']} del capital."""),
        ("Regla por segmento", f"""Con el cuantil 0,20 la empresa compra uno de cada tres
anuncios y el sobrepago cae frente al modelo de Kaggle. En apartamentos el sobrepago es
{n['AptCapital']} del capital; en casas, {n['HouseCapital']}: por eso no recomendamos usar el
modelo en casas. Que falte el área cambia poco la tasa de sobrepago, pero sí el error."""),
        ("Recomendación", f"""El modelo ordena bien los inmuebles y su error está muy por
debajo del ingenuo, pero ofrecer su predicción compra justo los inmuebles que sobreestima. Lo
usaríamos como filtro: el cuantil 0,20 marca candidatos en apartamentos ({n['AptClosed']} de
ellos, con {n['AptCapital']} de sobrepago frente a {n['OverCapitalD']} ofreciendo la
predicción de Kaggle), y una tasación física decide el precio. No lo usaríamos en casas ni
con predicciones de más de 900 millones, donde la sobrepredicción media pasa de
{n['OverHighPred']} millones. Antes de comprometer capital, revisaríamos la regla con precios
de cierre."""),
        ("Riesgos", f"""Primero, medimos el sobrepago contra el precio pedido, no contra el de
cierre: si los inmuebles se venden por menos, el sobrepago real es mayor. Segundo, falta el
área en el {n['ShareAreaMissing']} de los anuncios, y ahí el error sube de {n['MaeAreaOk']} a
{n['MaeAreaMissing']} M. Tercero, los anuncios son de 2019 a 2021. Cuarto, la validación
espacial es optimista en nivel frente a Kaggle ({n['SpatialBest']} frente a {n['KagBest']} M),
y en el holdout junto a Chapinero el sesgo cambia de signo: el nivel correcto en Chapinero es
incierto."""),
    ]


def notes_compare(n):
    return [
        ("Vista general", f"""Predecimos el precio de oferta de vivienda en Chapinero
entrenando con el resto de Bogotá. El mejor modelo combina un boosting cuantílico (70 %) con
una red neuronal cuantílica (30 %), un modelo para apartamentos y otro para casas. Las tres
cifras: el error público de Kaggle ({n['KagBest']} M), la mejora frente a predecir la mediana
({n['GainNaive']}) y el error en validación espacial ({n['SpatialBest']} M). El mejor boosting
solo obtiene {n['KagBoost']} M; la mezcla gana {n['GainBoost']} M. Las seis familias se
entrenan con la misma receta: por tipo, precio en nivel, cuantil 0,70, estrato y valor del
terreno, y el texto completo del anuncio."""),
        ("Tabla por familia", """Cada fila es la mejor especificación de su familia con la
receta común. Las columnas son tres formas de validar y el puntaje público. La aleatoria con
gemelos mantiene anuncios repetidos del mismo inmueble en la misma partición; la espacial
deja zonas completas fuera; el ponderado es un holdout al lado de Chapinero, ponderado como el
test. La elastic net no se subió a Kaggle. La red queda primera en el ponderado, pero en
Kaggle fue peor que el boosting: ese holdout la favorece."""),
        ("Receta común", f"""Antes, cada familia usaba logaritmo del precio y los mismos
predictores. La receta del boosting consiste en: un modelo por tipo de inmueble, el precio en
nivel con el cuantil 0,70 como objetivo, el estrato y el valor del terreno de la manzana, y
el texto completo resumido por un ridge fuera de muestra. Todas las familias mejoran en el
holdout ponderado, entre {n['RecipeGainMin']} y {n['RecipeGainMax']} M. En la espacial la
mejora es menor y el boosting sigue adelante: la ventaja es del algoritmo, no solo de la
receta."""),
        ("Rango de predicción", f"""Los precios de entrenamiento están entre 300 y 1.650
millones: un árbol predice promedios o cuantiles de lo que vio, así que el CART y el bosque
nunca pasan ese tope, y el boosting apenas lo roza. La red sí extrapola. La regresión lineal
no extrapola en precio, porque su receta la recorta al rango del train, pero sus splines de
latitud y longitud se estiman casi sin datos de Chapinero: sin coordenadas su error espacial
baja de {n['LinSpatial']} a {n['LinNoGeoSpatial']} M (fila 130), y en Kaggle fue de las peores
({n['KagLinear']} M)."""),
        ("Mezcla", f"""Calculamos, sin reentrenar, una mezcla 70/30 del boosting 127 con cada
familia sobre las predicciones fuera de muestra. La correlación de errores con el boosting en
el holdout nororiental es {n['CorrForest']} para el bosque y {n['CorrNet']} para la red. Con
la red el ponderado cambia {n['BlendNetGain']} M; con el bosque, {n['BlendForestGain']} M
(negativo = mejora). La diferencia decisiva es que la red sola es buena y sí pasa el tope de
1.650 M, donde el boosting subpredice los inmuebles caros. Las lineales y el kNN son más
distintos, pero tan malos solos que empeoran la mezcla."""),
        ("Resultados negativos", f"""Ninguna de estas {n['NNegative']} ideas mejoró a la vez
la espacial y el ponderado. OpenStreetMap empeora la validación espacial: las distancias a
amenidades no viajan bien a Chapinero, donde casi todo está cerca. Un modelo local solo con la
zona parecida pierde demasiados datos. El apilamiento aprende a corregir el tope artificial
del holdout, algo que no se puede llevar al test. Balancear hacia los caros baja la espacial
por nivel, no por orden. La monotonía y el kNN no ganan en los dos criterios."""),
        ("Validación contra Kaggle", f"""Cada punto es un envío con puntaje público; la línea
es la igualdad. Todos los esquemas son optimistas, porque Chapinero es más caro y distinto del
resto de Bogotá. Spearman con Kaggle sin el ingenuo: aleatoria simple {n['SpRandom']}, con
gemelos {n['SpTwins']}, espacial {n['SpSpatial']}, nororiental ponderado {n['SpWeighted']}
(sin ponderar llega a {n['SpNortheast']}). El holdout nororiental ordena mejor y, ponderado
como el test, acierta el nivel en boosting y bosque; pero es optimista con la red, la lineal,
el CART y los híbridos. La aleatoria simple es la que peor ordena."""),
        ("Gemelos", f"""Muchos anuncios del train son el mismo inmueble publicado varias veces.
Si una copia queda en el ajuste y otra en la validación, el modelo acierta por memoria. Al
mantener los gemelos juntos el error de la mezcla 152 sube de {n['RandomBest']} a
{n['TwinsBest']} M. En Chapinero casi no hay copias de inmuebles del train, así que la versión
con gemelos es la relevante."""),
        ("Nivel en la espacial", """La espacial ordena razonablemente, pero prefiere modelos con
un nivel más bajo que el que premia Kaggle: el cuantil 0,60 le gana al 0,70 en la espacial y
no en el ponderado, y en el tablero el 0,65 (envío 53) quedó detrás del 0,70 (65). Por eso no
escogimos la 186, que mejora la espacial casi solo por bajar el nivel. Usamos la espacial para
ordenar modelos de nivel parecido y el ponderado y Kaggle para el nivel."""),
        ("Pérdida asimétrica", f"""Con el error absoluto, la mezcla 152 queda en el puesto
{n['BestRankKone']} de la validación espacial. Si sobrepredecir cuesta dos o tres veces más
que subpredecir, el ranking premia a los modelos que apuntan más bajo: con costo triple gana
el {n['AsymWinner']} y la 152 cae al puesto {n['BestRankKthree']}. El modelo para Kaggle y el
modelo para comprar no tienen por qué ser el mismo: esa es la pregunta de la otra
presentación."""),
        ("Conclusión", """El boosting cuantílico por tipo es la base porque ordena bien los
precios en zonas que no vio. La receta común ayuda a todas las familias, pero la brecha con el
boosting se mantiene, así que la ventaja es del algoritmo. La red entra a la mezcla porque
llega a la cola alta. Para escoger modelos no confiamos en un solo esquema: la aleatoria
simple se engaña con los anuncios repetidos, la espacial premia bajar el nivel y el holdout
ponderado favorece a las redes. Por eso escogimos con los tres y usamos Kaggle como juez
final."""),
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
    write_tables(tables)
    write_notes(numbers)
    if not args.no_compile:
        compile_decks()


if __name__ == "__main__":
    main()
