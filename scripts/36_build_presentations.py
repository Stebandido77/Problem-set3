"""Las dos presentaciones del taller en document/: best_equipo_06 (el mejor modelo y la
decisión de compra) y compare_equipo_06 (comparación de familias y de esquemas de
validación). Se arman con python-pptx a partir de las figuras y tablas de views/ (scripts
31 a 35), con notas del orador en cada diapositiva, y se exportan a PDF con PowerPoint si
está instalado. Las cifras de los títulos salen de los CSV; ninguna se escribe a mano.

Uso: python 36_build_presentations.py [--preview]   (--preview guarda cada diapositiva
     como PNG en .claude/logs/preview/ para revisarla; esa carpeta no se versiona)
"""

import argparse

import pandas as pd
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

import deck as dk
from plot_style import FAMILY_COLORS, es_number
from utils import ROOT, VIEWS_DIR

DOCUMENT_DIR = ROOT / "document"
PREVIEW_DIR = ROOT / ".claude" / "logs" / "preview"
TEAM = "06"
BEST = "152_ensemble_w70_30_2"
NBSP = " "   # espacio que no se corta entre la cifra y su unidad


def m(value, decimals=1):
    """Millones con coma decimal (los CSV de views/ guardan pesos o millones)."""
    value = value / 1e6 if abs(value) > 1e5 else value
    return es_number(value, decimals)


def pct(share, decimals=0):
    return es_number(100 * share, decimals) + NBSP + "%"


def load():
    """Tablas de views/ que alimentan los títulos y las tablas editables."""
    read = lambda name: pd.read_csv(VIEWS_DIR / name)  # noqa: E731
    return {
        "kaggle": read("kaggle_scores.csv").set_index("model")["public_mae"],
        "schemes": read("validation_schemes.csv").set_index("numero"),
        "corr": read("validation_rank_corr.csv"),
        "family": read("family_table.csv"),
        "recipe": read("common_recipe.csv"),
        "blend": read("family_blend_with_127.csv").set_index("numero"),
        "range": read("family_test_range.csv").set_index("numero"),
        "negative": read("negative_results.csv"),
        "chapinero": read("chapinero_vs_bogota.csv").set_index("variable"),
        "missing": read("missing_values.csv"),
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


def overview(page, t, top):
    """Tres cifras grandes del mejor modelo: Kaggle, mejora sobre el ingenuo y espacial."""
    best, naive = kaggle_of(t, 152), kaggle_of(t, 0)
    spatial = t["schemes"].loc[152, "mae_spatial"]
    width = Inches(3.8)
    for k, (value, label) in enumerate([
            (m(best) + NBSP + "M", "MAE público en Kaggle\n(millones de pesos)"),
            ("−" + pct(1 - best / naive), f"frente al ingenuo\n(mediana constante, {m(naive)} M)"),
            (m(spatial) + NBSP + "M", "MAE en validación espacial\n(zonas de Bogotá fuera del ajuste)")]):
        dk.big_number(page, value, label, dk.MARGIN + k * (width + Inches(0.27)), top, width)


# --------------------------------------------------------------------------------------
# compare_equipo_06
# --------------------------------------------------------------------------------------

def family_rows(t):
    """Filas de la tabla por familia (cifras en millones)."""
    rows = [["Familia", "Fila", "Búsqueda de hiperparámetros", "Aleatoria con gemelos",
             "Espacial", "Ponderado", "Kaggle"]]
    short = {"no": "Sin sintonizar (valores de la receta)"}
    for _, r in t["family"].iterrows():
        search = r["busqueda"].split(";")[0].split(" (")[0].split(". Probado")[0]
        search = short.get(r["sintonizado"], search) if r["sintonizado"] == "no" else search
        if r["familia"] == "Regresión lineal":
            search = "No aplica (mínimos cuadrados)"
        if r["familia"] == "Mezcla":
            search = "Pesos fijos 70/30"
        rows.append([r["familia"], int(r["numero"]), search,
                     *(m(r[c]) if pd.notna(r[c]) else "–"
                       for c in ["mae_random_twins", "mae_spatial", "mae_weighted", "kaggle"])])
    return rows


def build_compare(t):
    deck = dk.new_deck()
    schemes, corr = t["schemes"], t["corr"]
    best, boost = kaggle_of(t, 152), kaggle_of(t, 83)

    page = dk.slide(
        deck, f"La mezcla de boosting y red neuronal es el mejor modelo: {m(best)} M en Kaggle",
        "Vista general. Predecimos el precio de oferta de vivienda en Chapinero entrenando con "
        "el resto de Bogotá. El mejor modelo combina un boosting cuantílico (70 %) con una red "
        "neuronal cuantílica (30 %). Estas tres cifras resumen su desempeño: el error público "
        "de Kaggle, la mejora frente a predecir la mediana y el error en validación espacial, "
        "que deja zonas completas de la ciudad por fuera. En el resto de la presentación "
        "explicamos por qué gana esta combinación y qué tan confiables son las validaciones.",
        subtitle="Boosting cuantílico (70 %) + red neuronal cuantílica (30 %), un modelo para "
                 "apartamentos y otro para casas")
    overview(page, t, Inches(2.4))
    dk.bullets(page, [
        f"El mejor boosting solo obtiene {m(boost)} M; la mezcla gana {m(boost - best)} M.",
        "Las seis familias se entrenan con la misma receta: por tipo, precio en nivel, "
        "cuantil 0,70, estrato y valor del terreno, y el texto completo del anuncio."],
        dk.MARGIN, Inches(4.9), dk.WIDTH - 2 * dk.MARGIN, Inches(2), size=18)

    page = dk.slide(
        deck, "Con la misma receta, el boosting gana en la espacial y en Kaggle; la red, solo "
              "en el ponderado",
        "Tabla por familia. Cada fila es la mejor especificación de su familia con la receta "
        "común. Las columnas son tres formas de validar y el puntaje público. La aleatoria con "
        "gemelos mantiene anuncios repetidos del mismo inmueble en la misma partición; la "
        "espacial deja zonas completas fuera; el ponderado es un holdout al lado de Chapinero, "
        "ponderado como el test. La elastic net no se subió a Kaggle. La red queda primera en "
        "el ponderado, pero en Kaggle fue peor que el boosting: ese holdout la favorece.",
        subtitle="MAE en millones de pesos; menos es mejor")
    dk.table(page, family_rows(t), dk.MARGIN, dk.top(page), dk.WIDTH - 2 * dk.MARGIN,
             col_widths=[1.6, 0.6, 3.6, 1.2, 1.0, 1.0, 1.0], size=14,
             highlight_rows=(len(t["family"]),), text_cols=(0, 2))

    recipe = t["recipe"].pivot(index="familia", columns="etapa", values="mae_weighted")
    gain = (recipe["antes"] - recipe["mejor"])
    page = dk.slide(
        deck, f"La receta común mejora todas las familias entre {m(gain.min())} y "
              f"{m(gain.max())} M en el ponderado",
        "Qué aportó la receta común. Antes, cada familia usaba logaritmo del precio y los "
        "mismos predictores. La receta del boosting consiste en: un modelo por tipo de "
        "inmueble, el precio en nivel con el cuantil 0,70 como objetivo, el estrato y el valor "
        "del terreno de la manzana, y el texto completo resumido por un ridge fuera de muestra. "
        "Todas las familias mejoran en el holdout ponderado. En la espacial la mejora es menor "
        "y el boosting sigue adelante: la ventaja es del algoritmo, no solo de la receta.")
    dk.figure(page, "common_recipe.png", top=dk.top(page))

    lin = t["cv"].loc["108_linear_ext_core_coords_splines_by_type_level_q70_ridge_oof"]
    nogeo = t["cv"].loc["130_linear_ext_core_coords_by_type_level_q70_ridge_oof_nogeo"]
    page = dk.slide(
        deck, "Los árboles no pasan el tope de precios y la lineal viaja mal por sus "
              "coordenadas",
        "Por qué cada familia queda donde queda. Los precios de entrenamiento están entre 300 "
        "y 1.650 millones: un árbol predice promedios o cuantiles de lo que vio, así que el "
        "CART y el bosque nunca pasan ese tope, y el boosting apenas lo roza. La red sí "
        "extrapola. La regresión lineal no extrapola en precio, porque su receta la recorta "
        "al rango del train, pero sus splines de latitud y longitud se estiman casi sin datos "
        f"de Chapinero: sin coordenadas su error espacial baja de {m(lin['mae_spatial'])} a "
        f"{m(nogeo['mae_spatial'])} M, y en Kaggle fue de las peores ({m(kaggle_of(t, 108))} M).",
        subtitle=f"Lineal 108: espacial {m(lin['mae_spatial'])} M con splines de lat/lon y "
                 f"{m(nogeo['mae_spatial'])} M sin ellos (fila 130)")
    dk.figure(page, "family_test_range.png", top=dk.top(page))

    blend = t["blend"]
    page = dk.slide(
        deck, "La red mejora la mezcla porque llega a la cola alta; el bosque no aporta nada "
              "nuevo",
        "Por qué la mezcla funciona con la red y no con el bosque. Calculamos, sin reentrenar, "
        "una mezcla 70/30 del boosting 127 con cada familia sobre las predicciones fuera de "
        f"muestra. La correlación de errores con el boosting es {es_number(blend.loc[114, 'corr_error_northeast'], 3)} "
        f"para el bosque y {es_number(blend.loc[148, 'corr_error_northeast'], 3)} para la red en "
        "el holdout nororiental. Pero la diferencia decisiva es que la red sola es buena y sí "
        "pasa el tope de 1.650 M, donde el boosting subpredice los inmuebles caros. Las lineales "
        "y el kNN son más distintos, pero tan malos solos que empeoran la mezcla.",
        subtitle=f"Mezcla 70/30 con la 127: con la red el ponderado cambia "
                 f"{m(blend.loc[148, 'ganancia_weighted'])} M; con el bosque, "
                 f"{m(blend.loc[114, 'ganancia_weighted'])} M (negativo = mejora)")
    dk.figure(page, "family_blend_with_127.png", top=dk.top(page))

    negative = t["negative"]
    page = dk.slide(
        deck, f"Ninguna de estas {len(negative)} ideas mejoró a la vez la espacial y el "
              "ponderado",
        "Resultados negativos que vale la pena mostrar. OpenStreetMap empeora la validación "
        "espacial: las distancias a amenidades no viajan bien a Chapinero, donde casi todo "
        "está cerca. Un modelo local solo con la zona parecida pierde demasiados datos. El "
        "apilamiento aprende a corregir el tope artificial del holdout, algo que no se puede "
        "llevar al test. Balancear hacia los caros baja la espacial por nivel, no por orden. La "
        "monotonía y el kNN no ganan en los dos criterios.")
    dk.figure(page, "negative_results.png", top=dk.top(page))

    no_naive = corr[corr["conjunto"] == "sin el ingenuo"].set_index("esquema")
    page = dk.slide(
        deck, "El holdout junto a Chapinero ordena mejor los envíos; la aleatoria simple casi "
              "no sirve",
        "Validación aleatoria, espacial, ponderada y Kaggle, modelo por modelo. Cada punto es "
        "un envío con puntaje público; la línea punteada es la igualdad. Todos los esquemas "
        "son optimistas, porque Chapinero es más caro y distinto del resto de Bogotá. El "
        "holdout nororiental (los inmuebles del train junto al test) ordena mejor y, "
        "ponderado como el test, acierta el nivel en boosting y bosque; pero es optimista con "
        "la red, la lineal, el CART y los híbridos. La aleatoria simple apenas ordena.",
        subtitle="Spearman con Kaggle, sin el ingenuo, en cada panel; el holdout nororiental "
                 "sin ponderar llega a " + es_number(
                     no_naive.loc["Nororiental sin ponderar", "spearman"], 2)
                 + f" (n = {no_naive.loc['Nororiental sin ponderar', 'n']})")
    dk.figure(page, "validation_vs_kaggle_wide.png", top=dk.top(page))

    twins = schemes.loc[[127, 152]]
    page = dk.slide(
        deck, "Los anuncios repetidos hacen que la validación aleatoria parezca mejor de lo "
              "que es",
        "Qué falla en la aleatoria. Muchos anuncios del train son el mismo inmueble publicado "
        "varias veces. Si una copia queda en el ajuste y otra en la validación, el modelo "
        "acierta por memoria. Al mantener los gemelos juntos el error sube. En Chapinero casi "
        "no hay copias de inmuebles del train, así que la versión con gemelos es la relevante.",
        subtitle=f"Mezcla 152: {m(twins.loc[152, 'mae_random'])} M en la aleatoria simple y "
                 f"{m(twins.loc[152, 'mae_random_twins'])} M con los gemelos juntos")
    dk.figure(page, "validation_twins.png", top=dk.top(page))

    page = dk.slide(
        deck, "La validación espacial premia bajar el nivel de las predicciones; el tablero no",
        "Qué falla en la espacial. Ordena razonablemente, pero prefiere modelos con un nivel "
        "más bajo que el que premia Kaggle: el cuantil 0,60 le gana al 0,70 en la espacial y "
        "pierde en el ponderado y en el tablero. Por eso no escogimos la 186, que mejora la "
        "espacial casi solo por bajar el nivel. Usamos la espacial para ordenar modelos de "
        "nivel parecido y el ponderado y Kaggle para el nivel.")
    dk.figure(page, "validation_level.png", top=dk.top(page))

    asym = t["asym"].set_index("model")
    page = dk.slide(
        deck, "Si sobrepredecir cuesta más, ganan los modelos de nivel más bajo",
        "Intercambios entre modelos. Con el error absoluto, la mezcla 152 y el boosting 127 "
        "quedan arriba. Si sobrepredecir cuesta dos o tres veces más que subpredecir, el "
        "ranking premia a los modelos que apuntan más bajo. El modelo para Kaggle y el modelo "
        "para comprar no tienen por qué ser el mismo: esa es la pregunta de la otra "
        "presentación.",
        subtitle="Costo medio en la validación espacial con sobrepredicción ponderada 1, 2 y "
                 "3 veces")
    dk.figure(page, "asymmetric_ranking.png", top=dk.top(page))

    page = dk.slide(
        deck, "Conclusión: el boosting con la receta común es la base y la red aporta en la "
              "cola alta",
        "Cierre. Resumimos qué aprendimos al comparar familias y esquemas de validación.")
    dk.bullets(page, [
        "El boosting cuantílico por tipo es la mejor familia en la validación espacial, en "
        "la aleatoria con gemelos y en Kaggle; la red solo le gana en el holdout ponderado.",
        "La receta común (tipo, nivel, cuantil 0,70, manzana y texto) mejora a todas las "
        "familias, pero no cierra la brecha con el boosting.",
        "La red sirve en la mezcla porque puede predecir por encima del tope de precios.",
        "Ningún esquema de validación basta: la aleatoria simple sobreajusta a los repetidos, "
        "la espacial premia bajar el nivel y el ponderado favorece a las redes.",
        "Escogimos con los tres esquemas y confirmamos en Kaggle."],
        dk.MARGIN, dk.top(page), dk.WIDTH - 2 * dk.MARGIN, Inches(5.5), size=22)
    return deck


# --------------------------------------------------------------------------------------
# best_equipo_06
# --------------------------------------------------------------------------------------

def rule_row(t, rule, discount, segment="todas"):
    """Una fila de la simulación de compra (35_purchase_rule.py)."""
    if segment == "todas":
        table = t["rule"]
        row = table[(table["rule"] == rule) & (table["discount"].round(2) == discount)]
    else:
        table = t["rule_segments"]
        row = table[(table["rule"] == rule) & (table["discount"].round(2) == discount)
                    & (table["segment"] == segment)]
    return row.iloc[0]


def segment(t, variable, name):
    table = t["segments"]
    return table[(table["scheme"] == "spatial") & (table["variable"] == variable)
                 & (table["segment"] == name)].iloc[0]


def model_diagram(page, top):
    """Esquema editable del modelo: entradas, dos modelos y la mezcla."""
    def box(text, left, y, width, height, color, font_color=dk.DARK, size=16):
        shape = page.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, y, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor.from_string(color.lstrip("#"))
        shape.line.fill.background()
        frame = shape.text_frame
        frame.word_wrap = True
        for k, line in enumerate(text.split("\n")):
            paragraph = frame.paragraphs[0] if k == 0 else frame.add_paragraph()
            run = paragraph.add_run()
            run.text = line
            run.font.size, run.font.name = Pt(size), dk.FONT
            run.font.bold = k == 0
            run.font.color.rgb = font_color
        return shape

    def arrow(left, y):
        shape = page.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, left, y, Inches(0.35),
                                      Inches(0.35))
        shape.fill.solid()
        shape.fill.fore_color.rgb = dk.MUTED
        shape.line.fill.background()

    white = RGBColor(0xFF, 0xFF, 0xFF)
    col, gap = Inches(3.9), Inches(0.45)
    left = [dk.MARGIN, dk.MARGIN + col + gap, dk.MARGIN + 2 * (col + gap)]
    box("Qué ve el modelo\nestructura: tipo, alcobas, baños, área\nárea y baños de anuncios "
        "del mismo punto\nestrato y valor del terreno de la manzana\ndistancias a cuatro hitos\n"
        "18 señales del texto + ridge TF-IDF\nsin latitud ni longitud crudas",
        left[0], top, col, Inches(3.6), "#EEEEEE", size=14)
    box("Boosting cuantílico (70 %)\nLightGBM, cuantil 0,70 del precio\nun modelo por tipo\n"
        "10 semillas\nregla de anuncios repetidos",
        left[1], top, col, Inches(1.7), FAMILY_COLORS["boosting"], white, size=15)
    box("Red neuronal cuantílica (30 %)\ncuantil 0,70, un modelo por tipo\n"
        "puede predecir por encima\ndel tope de 1.650 M",
        left[1], top + Inches(1.9), col, Inches(1.7), FAMILY_COLORS["neural_net"], white,
        size=15)
    box("Predicción\n0,7 × boosting + 0,3 × red\nprecio de oferta en millones",
        left[2], top + Inches(0.95), col, Inches(1.7), FAMILY_COLORS["ensemble"], white,
        size=15)
    arrow(left[0] + col + Inches(0.05), top + Inches(0.67))
    arrow(left[0] + col + Inches(0.05), top + Inches(2.57))
    arrow(left[1] + col + Inches(0.05), top + Inches(1.62))


def build_best(t):
    deck = dk.new_deck()
    best = kaggle_of(t, 152)
    over = rule_row(t, "152 (Kaggle, q0,70)", 0.0)
    q20 = rule_row(t, "q0,20", 0.0)
    total = segment(t, "total", "todas")

    page = dk.slide(
        deck, f"El modelo predice Chapinero con un error de {m(best)} M, pero no basta para "
              "fijar el precio de compra",
        "Empezamos por el resultado. La pregunta guía es si podemos confiar en el modelo lo "
        "suficiente para comprar vivienda en Chapinero. Nuestro mejor modelo tiene un error "
        f"medio de {m(best)} millones en el tablero público, "
        f"{pct(1 - best / kaggle_of(t, 0))} menos que predecir la mediana. Pero "
        "si la empresa ofrece lo que predice el modelo, solo compra cuando el modelo "
        "sobreestima, y paga de más. La respuesta corta: sirve para filtrar candidatos con una "
        "regla conservadora, no para fijar el precio.",
        subtitle="¿Se puede confiar en el modelo lo suficiente para comprar?")
    overview(page, t, Inches(2.4))
    dk.bullets(page, [
        f"Ofrecer la predicción cierra el {pct(over['share_closed'])} de los anuncios con un "
        f"sobrepago del {pct(over['overpay_share_capital'])} del capital.",
        f"Con el cuantil 0,20 se cierra el {pct(q20['share_closed'])} con un sobrepago del "
        f"{pct(q20['overpay_share_capital'])}."],
        dk.MARGIN, Inches(4.9), dk.WIDTH - 2 * dk.MARGIN, Inches(2), size=20)

    chap = t["chapinero"]
    houses = chap.loc["Casas"]
    land = chap.loc["Valor del terreno (miles de $/m², mediana)"]
    page = dk.slide(
        deck, "Entrenamos con el resto de Bogotá y predecimos Chapinero, una zona más cara y "
              "casi sin casas",
        "El problema. Los puntos grises son los anuncios de entrenamiento; los rojos, los de "
        "Chapinero que hay que predecir. El modelo casi no ve Chapinero al entrenar. Además, "
        "Chapinero es distinto: casi todo son apartamentos, el estrato es más alto y el "
        "terreno cuesta el doble. Por eso validamos dejando zonas enteras por fuera.",
        subtitle=f"Casas: {es_number(houses['resto_bogota'], 0)} % en el train y "
                 f"{es_number(houses['chapinero'], 0)} % en Chapinero; valor del terreno "
                 f"{es_number(land['chapinero'] / land['resto_bogota'], 1)} veces mayor")
    dk.figure(page, "map_train_test.png", left=dk.MARGIN, top=dk.top(page),
              max_width=Inches(5.6))
    dk.figure(page, "chapinero_vs_bogota_table.png", left=Inches(6.4), top=dk.top(page),
              max_width=Inches(6.4))

    page = dk.slide(
        deck, "Qué es el modelo: un boosting cuantílico y una red neuronal que se corrigen "
              "en la cola alta",
        "Qué es y por qué funciona. El boosting aprende bien el orden de los precios, pero un "
        "árbol no predice por encima de lo que vio, y el train está recortado en 1.650 M. La "
        "red puede extrapolar y corrige en parte la subpredicción de los inmuebles caros. Los "
        "dos apuntan al cuantil 0,70 del precio y no a la mediana: Chapinero es más caro que el "
        "resto de Bogotá, y apuntar alto compensa ese salto de nivel. Quitamos latitud y "
        "longitud porque no viajan a una zona que no se vio; la ubicación entra por el "
        "estrato, el valor del terreno y las distancias.")
    model_diagram(page, dk.top(page))
    dk.bullets(page, [
        "El boosting ordena bien los precios, pero un árbol no pasa del tope de 1.650 M.",
        "La red sí extrapola y corrige en parte a los inmuebles caros.",
        "Los dos apuntan al cuantil 0,70: Chapinero es más caro que el resto de Bogotá."],
        dk.MARGIN, dk.top(page) + Inches(3.85), dk.WIDTH - 2 * dk.MARGIN, Inches(1.6),
        size=18)

    page = dk.slide(
        deck, "Cada mejora tiene una causa: nivel, un modelo por tipo, el texto y la mezcla",
        "De dónde salió cada mejora, en puntaje público. Del ingenuo al primer boosting. Subir "
        "el nivel (primero escalando, luego con la pérdida cuantílica) y separar apartamentos "
        "y casas. El texto completo del anuncio fue el último salto grande de un modelo solo. "
        f"La mezcla con la red dio los últimos {m(kaggle_of(t, 83) - best)} millones. El envío 32 es una calibración con "
        "el tablero, no un modelo, y lo mostramos solo para explicar el efecto del nivel.")
    dk.figure(page, "kaggle_progression.png", top=dk.top(page))

    importance = t["importance"]
    apt = importance[importance["tipo"] == "apartamento"].iloc[0]
    page = dk.slide(
        deck, f"El texto del anuncio es lo que más pesa: {pct(apt['gain_share'])} de la "
              "ganancia en apartamentos",
        "Qué usa el modelo. Importancia por ganancia del boosting, ajustado una vez con todo el "
        "train. El ridge sobre el texto completo resume en un número lo que dicen el título y "
        "la descripción. Le siguen el área (también la recuperada de anuncios del mismo punto), "
        "el largo de la descripción y la ubicación a través de distancias y valor del terreno. "
        "Es una importancia de un ajuste con una semilla: sirve para ordenar, no para medir "
        "efectos causales.")
    dk.figure(page, "best_errors_importance.png", top=dk.top(page))

    cheap, rich = segment(t, "precio_observado", "<400"), segment(t, "precio_observado", "1.200+")
    page = dk.slide(
        deck, f"Sobrepredice el {pct(cheap['share_over'])} de los baratos y solo el "
              f"{pct(rich['share_over'])} de los de más de 1.200 M",
        "Dónde se equivoca. Usamos las predicciones fuera de muestra de la validación "
        f"espacial. En total sobrepredice el {pct(total['share_over'])} de los inmuebles, con "
        f"un sesgo de {m(total['bias'])} millones. El error crece con el precio y es mayor en "
        "las casas y cuando falta el área. Para comprar importa la columna del precio predicho, "
        "que es lo único que ve la empresa: cuanto más alta la predicción, más grande la "
        "sobrepredicción media.",
        subtitle="Validación espacial: barras rojas = sobrepredicción media, azules = "
                 "subpredicción media, puntos = sesgo")
    dk.figure(page, "best_errors_segments.png", top=dk.top(page))

    asym = t["asym"]
    winner = asym.sort_values("rank_k3").iloc[0]
    page = dk.slide(
        deck, "Si sobrepredecir cuesta el triple, gana el modelo que apunta al cuantil 0,20",
        "Sobre contra subpredicción. Recalculamos el costo medio de cada modelo cuando "
        "sobrepredecir cuesta dos o tres veces más que subpredecir. Entre familias el orden no "
        "cambia; lo que cambia es el nivel: ganan los cuantiles bajos, como predice la teoría "
        "(con costo k veces mayor, el cuantil óptimo es 1/(1+k): 0,33 y 0,25). El modelo que "
        "gana en Kaggle no es el que conviene para comprar.",
        subtitle=f"Con costo triple gana {winner['label']}; la 152 de Kaggle queda en el puesto "
                 f"{int(asym.set_index('model').loc[BEST, 'rank_k3'])}")
    dk.figure(page, "asymmetric_ranking.png", top=dk.top(page))

    page = dk.slide(
        deck, f"Ofrecer lo que predice el modelo cierra el {pct(over['share_closed'])} de los "
              f"anuncios y sobrepaga el {pct(over['overpay_share_capital'])} del capital",
        "La regla de compra. Simulamos: la empresa ofrece la predicción con un descuento; si la "
        "oferta alcanza el precio pedido, compra y la diferencia es sobrepago. Hay selección "
        "adversa: solo se compra cuando el modelo sobreestima. Cada curva es un cuantil y cada "
        "punto un descuento de 0 a 20 %. A igual número de compras, bajar el cuantil sobrepaga "
        "menos que aplicar un descuento al modelo de Kaggle. Ninguna regla baja el sobrepago "
        "del 10 % del capital.",
        subtitle="Validación espacial; oferta = predicción × (1 − descuento)")
    dk.figure(page, "purchase_rule_frontier.png", top=dk.top(page))

    rows = [["Regla", "Segmento", "Compras cerradas", "Sobrepago medio (M)",
             "Sobrepago / capital"]]
    for rule, discount, seg, label in [
            ("152 (Kaggle, q0,70)", 0.0, "todas", "Modelo de Kaggle, sin descuento"),
            ("q0,20", 0.0, "todas", "Cuantil 0,20, sin descuento"),
            ("q0,20", 0.0, "apartamento, con área", "Cuantil 0,20, sin descuento"),
            ("q0,20", 0.0, "apartamento, sin área", "Cuantil 0,20, sin descuento"),
            ("q0,20", 0.0, "casa, con área", "Cuantil 0,20, sin descuento"),
            ("q0,20", 0.0, "casa, sin área", "Cuantil 0,20, sin descuento")]:
        r = rule_row(t, rule, discount, seg)
        rows.append([label, seg, pct(r["share_closed"]), m(r["mean_overpay"]),
                     pct(r["overpay_share_capital"], 1)])
    apt = rule_row(t, "q0,20", 0.0, "apartamento, con área")
    house = rule_row(t, "q0,20", 0.0, "casa, con área")
    page = dk.slide(
        deck, "La regla que proponemos: cuantil 0,20 sin descuento, solo en apartamentos",
        "La regla por segmento. Con el cuantil 0,20 la empresa compra uno de cada tres "
        "anuncios y el sobrepago cae frente al modelo de Kaggle. En casas el sobrepago es "
        "mayor y por eso no recomendamos usar el modelo ahí. Que falte el área cambia poco la "
        "tasa de sobrepago, pero sí el error.",
        subtitle=f"Apartamentos con área: {pct(apt['overpay_share_capital'], 1)} de sobrepago "
                 f"sobre el capital; casas con área: {pct(house['overpay_share_capital'], 1)}")
    dk.table(page, rows, dk.MARGIN, dk.top(page), dk.WIDTH - 2 * dk.MARGIN,
             col_widths=[2.6, 1.9, 1.3, 1.5, 1.4], size=16, text_cols=(0, 1),
             highlight_rows=(3,))

    page = dk.slide(
        deck, "¿Se puede confiar? Para escoger candidatos sí; para fijar el precio, no",
        "Recomendación. El modelo ordena bien los inmuebles y su error está muy por debajo del "
        "ingenuo, pero ofrecer su predicción compra justo los inmuebles que sobreestima. Lo "
        "usaríamos como filtro: el cuantil 0,20 marca candidatos en apartamentos, y una "
        "tasación física decide el precio. No lo usaríamos en casas ni en predicciones por "
        "encima de 900 millones, donde la sobrepredicción media pasa de 200 millones.")
    dk.bullets(page, [
        "Usar el cuantil 0,20 sin descuento para escoger candidatos, no el modelo de Kaggle con "
        "descuento.",
        "Solo apartamentos; cerrar después de una tasación física.",
        "No usarlo en casas ni con predicciones de más de 900 M.",
        "Revisar la regla con precios de cierre antes de comprometer capital."],
        dk.MARGIN, dk.top(page), dk.WIDTH - 2 * dk.MARGIN, Inches(2.8), size=22)
    apt = rule_row(t, "q0,20", 0.0, "apartamento")
    width = Inches(5.6)
    dk.big_number(page, pct(apt["share_closed"]),
                  "de los apartamentos se compran\ncon el cuantil 0,20", dk.MARGIN,
                  Inches(4.6), width)
    dk.big_number(page, pct(apt["overpay_share_capital"], 1),
                  "de sobrepago sobre el capital, frente a "
                  f"{pct(over['overpay_share_capital'], 1)}\nofreciendo la predicción de Kaggle",
                  dk.WIDTH - dk.MARGIN - width, Inches(4.6), width)

    page = dk.slide(
        deck, "Los riesgos: precios de oferta, área faltante, datos viejos y cambio de zona",
        "Riesgos. Primero, medimos el sobrepago contra el precio pedido, no contra el de "
        "cierre: si los inmuebles se venden por menos, el sobrepago real es mayor. Segundo, "
        "falta el área en una de cada cuatro observaciones, y ahí el error es mayor. Tercero, "
        "los anuncios son de 2019 a 2021. Cuarto, la validación espacial es optimista en nivel "
        "frente a Kaggle, y en el holdout junto a Chapinero el sesgo cambia de signo: el nivel "
        "correcto en Chapinero es incierto.")
    spatial = t["schemes"].loc[152, "mae_spatial"]
    dk.bullets(page, [
        "Precios de oferta, no de cierre: el sobrepago real puede ser mayor.",
        f"Área faltante: el MAE sube de {m(segment(t, 'area', 'anuncio o texto')['mae'])} a "
        f"{m(segment(t, 'area', 'faltante')['mae'])} M.",
        "Datos de 2019 a 2021: el mercado pudo cambiar.",
        f"Cambio de zona: la espacial da {m(spatial)} M y Kaggle {m(best)} M; junto a "
        "Chapinero el modelo subpredice.",
        "La validación espacial premia bajar el nivel: no basta para escoger el cuantil."],
        dk.MARGIN, dk.top(page), dk.WIDTH - 2 * dk.MARGIN, Inches(5), size=22)
    return deck


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    tables = load()
    DOCUMENT_DIR.mkdir(exist_ok=True)
    for name, build in [("best", build_best), ("compare", build_compare)]:
        path = DOCUMENT_DIR / f"{name}_equipo_{TEAM}.pptx"
        build(tables).save(path)
        pdf = dk.export_pdf(path, PREVIEW_DIR / name if args.preview else None)
        print(f"{path.name}: {'PDF en ' + pdf.name if pdf else 'sin PowerPoint, solo .pptx'}")


if __name__ == "__main__":
    main()
