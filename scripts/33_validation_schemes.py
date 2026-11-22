"""Compara los cuatro esquemas de validación con el tablero público de Kaggle (etapa 3).

Esquemas: aleatoria simple (mae_random), aleatoria con los gemelos en la misma partición
(mae_random_twins), espacial por zonas de k-means (mae_spatial) y holdout nororiental
ponderado como el test (mae_weighted). Como referencia se agrega el holdout nororiental sin
ponderar (mae_northeast), el único que tienen todos los envíos viejos.

No entrena modelos. Cruza views/kaggle_scores.csv con views/cv_results.csv por el nombre del
archivo enviado y completa lo que falta así:
- 127 y 152, aleatoria con gemelos: la regla 'mix' de 20_twin_rules.py sobre las
  predicciones fuera de muestra del 125 en esas particiones; la 152 es 0,7 × 127 + 0,3 × 148.
- 00 (mediana constante): se recalcula en cada partición de cada esquema.
- 10 y 11: Kaggle tiene las versiones sin OSM; sus filas de cv_results son las _v2 con OSM.
  Se usan views/cv_results_pre_osm.csv y el esquema r250_b0_cap1000 de
  views/holdout_calibration.csv (el holdout nororiental actual, sin ponderar).
- 32 (10 sin OSM × 1,15): solo existe en views/holdout_calibration.csv.

Salidas en views/:
  validation_schemes.csv        un envío por fila con todos los esquemas
  validation_rank_corr.csv      Spearman, pares bien ordenados y brecha por esquema
  validation_vs_kaggle.png      cada esquema contra Kaggle (2 × 2)
  validation_vs_kaggle_wide.png la misma figura en una fila, para diapositiva 16:9
  validation_rank_corr.png      Spearman por esquema
  validation_twins.png          aleatoria simple contra aleatoria con gemelos
  validation_level.png          la espacial premia bajar el nivel; el tablero no

Uso (desde scripts/):
  python 33_validation_schemes.py
"""

import importlib
from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

import plot_style as ps
from models import northeast_building_groups
from utils import (ID_COL, PROCESSED_DIR, TARGET, VIEWS_DIR, find_submission, load_processed,
                   northeast_holdout, random_cv, random_twin_splits, spatial_cv_splits,
                   weighted_mae)

MILLION = 1e6
SCHEMES = {
    "mae_random": "Aleatoria simple",
    "mae_random_twins": "Aleatoria con gemelos",
    "mae_spatial": "Espacial (10 zonas)",
    "mae_weighted": "Nororiental ponderado",
}
REFERENCE = {"mae_northeast": "Nororiental sin ponderar"}
BASE_125 = "125_boosting_pt_core_coords_by_type_level_q70_ridge_nocoords_s10"
NAME_127 = "127_boosting_pt_core_coords_by_type_level_q70_ridge_nocoords_s10_twin_mix"
NAME_148 = "148_quantile_net_ext_core_coords_by_type_q70_ridge_nocoords"
NAME_152 = "152_ensemble_w70_30_2"
WEIGHTS_152 = (0.7, 0.3)
CALIBRATION_SCHEME = "r250_b0_cap1000"
PRE_OSM = {10: "10_boosting_full_coords", 11: "11_neural_net_full"}
# Escalera de cuantiles de la ronda 8 (la 83 es el cuantil 0,70) y modelos que bajan el nivel.
QUANTILE_LADDER = {154: 0.60, 155: 0.65, 83: 0.70, 156: 0.75, 157: 0.80}
LOWER_LEVEL = [127, 152, 181, 186]


def number_of(name):
    """Número de envío al inicio del nombre."""
    return int(name.split("_", 1)[0])


# ---------------------------------------------------------------- cálculos sin entrenar

def oof_aligned(train):
    """Predicciones fuera de muestra en el orden de train."""
    oof = pd.read_csv(PROCESSED_DIR / "oof_predictions.csv").set_index(ID_COL)
    return oof.reindex(train[ID_COL])


def mix_rule_twins(train, oof):
    """MAE y sesgo de la 127 y la 152 en la aleatoria con gemelos, aplicando la regla 'mix'
    al 125 dentro de cada partición. Antes comprueba que el mismo cálculo reproduce la
    aleatoria simple de cv_results.csv."""
    twin_rules = importlib.import_module("20_twin_rules")
    y = train[TARGET].to_numpy(float)
    description = twin_rules.normalized_description(train)

    def rule_127(scheme, splits):
        base = oof[f"{BASE_125}__{scheme}"].to_numpy(float)
        return twin_rules.evaluate_scheme("mix", base, description, y, splits)[0]

    def blend_152(pred_127, scheme):
        return WEIGHTS_152[0] * pred_127 + WEIGHTS_152[1] * oof[f"{NAME_148}__{scheme}"]

    check = rule_127("random", list(random_cv().split(train)))
    check_152 = blend_152(check, "random")
    assert np.allclose(check, oof[f"{NAME_127}__random"], equal_nan=True)
    assert np.allclose(check_152, oof[f"{NAME_152}__random"], equal_nan=True)

    pred_127 = rule_127("random_twins", random_twin_splits(train))
    pred_152 = blend_152(pred_127, "random_twins").to_numpy(float)
    return {127: error_stats(y, pred_127), 152: error_stats(y, pred_152)}


def error_stats(y, pred):
    """MAE y sesgo medio (predicción − precio) en las filas con predicción."""
    rows = ~np.isnan(pred)
    err = pred[rows] - y[rows]
    return {"mae": np.abs(err).mean(), "bias": err.mean()}


def naive_median_scores(train, test):
    """La mediana constante (envío 00) en cada esquema: la mediana sale del ajuste."""
    y = train[TARGET].to_numpy(float)
    splits = {"random": list(random_cv().split(train)),
              "random_twins": random_twin_splits(train),
              "spatial": spatial_cv_splits(train)}
    out = {}
    for scheme, folds in splits.items():
        pred = np.full(len(y), np.nan)
        for fit_idx, val_idx in folds:
            pred[val_idx] = np.median(y[fit_idx])
        stats = error_stats(y, pred)
        out[f"mae_{scheme}"], out[f"bias_{scheme}"] = stats["mae"], stats["bias"]
    fit_idx, val_idx = northeast_holdout(train, test)
    pred = np.full(len(val_idx), np.median(y[fit_idx]))
    _, groups = northeast_building_groups(train, test, train[TARGET])
    out["mae_northeast"] = np.abs(pred - y[val_idx]).mean()
    out["bias_northeast"] = (pred - y[val_idx]).mean()
    out["mae_weighted"] = weighted_mae(y[val_idx], pred, groups)
    return out


# ---------------------------------------------------------------- tabla por envío

def submission_mean(file_name):
    """Media de las predicciones de un envío guardado."""
    path = find_submission(file_name)
    return pd.read_csv(path)[TARGET].mean() if path.exists() else np.nan


def calibration_rows():
    """Holdout nororiental de los cinco envíos viejos, reentrenados tal como se subieron."""
    calib = pd.read_csv(VIEWS_DIR / "holdout_calibration.csv")
    calib = calib[calib["scheme"] == CALIBRATION_SCHEME].set_index("model")
    return calib[["mae", "bias"]]


def row_from_cv(result):
    """Columnas de esquemas de una fila de cv_results.csv."""
    keys = ["mae_random", "mae_random_twins", "mae_spatial", "mae_weighted", "mae_northeast",
            "bias_spatial", "bias_northeast"]
    return {key: result.get(key, np.nan) for key in keys}


def old_submission_row(kaggle_name, naive, calib, pre_osm):
    """Esquemas de los envíos 00, 10, 11 y 32, que no tienen fila comparable en cv_results."""
    number = number_of(kaggle_name)
    row = {key: np.nan for key in ["mae_random", "mae_random_twins", "mae_spatial",
                                   "mae_weighted", "bias_spatial"]}
    row["mae_northeast"] = calib.loc[kaggle_name, "mae"]
    row["bias_northeast"] = calib.loc[kaggle_name, "bias"]
    if number == 0:
        row.update(naive)
        return row, "naive", ("mediana constante recalculada aquí en cada partición; "
                              "nororiental igual a holdout_calibration")
    if number in PRE_OSM:
        source = pre_osm.loc[PRE_OSM[number]]
        row.update(mae_random=source["mae_random_cv"], mae_spatial=source["mae_spatial_cv"],
                   bias_spatial=source["bias_spatial_cv"])
        return row, source["family"], ("Kaggle es la versión sin OSM: aleatoria y espacial de "
                                       "cv_results_pre_osm, nororiental de holdout_calibration "
                                       "(r250_b0_cap1000); la fila de cv_results es la _v2")
    return row, "boosting", ("10 sin OSM × 1,15: solo holdout_calibration (r250_b0_cap1000); "
                             "sin predicciones fuera de muestra para otros esquemas")


def schemes_table(train, test, oof):
    """Un envío por fila: esquemas, Kaggle, sesgos y media en test (en millones)."""
    scores = pd.read_csv(VIEWS_DIR / "kaggle_scores.csv")
    results = pd.read_csv(VIEWS_DIR / "cv_results.csv")
    results["file"] = results["submission"].str.removesuffix(".csv")
    by_file = results.dropna(subset=["file"]).set_index("file")
    by_model = results.set_index("model")
    pre_osm = pd.read_csv(VIEWS_DIR / "cv_results_pre_osm.csv").set_index("model")
    calib = calibration_rows()
    naive = naive_median_scores(train, test)
    twins = mix_rule_twins(train, oof)

    rows = []
    for kaggle_name, public in zip(scores["model"], scores["public_mae"]):
        number = number_of(kaggle_name)
        if kaggle_name in by_file.index and number not in PRE_OSM:
            result = by_file.loc[kaggle_name]
            values, family, note = row_from_cv(result), result["family"], ""
            model = result["model"]
        else:
            values, family, note = old_submission_row(kaggle_name, naive, calib, pre_osm)
            model = kaggle_name
        rows.append({"numero": number, "modelo": model, "envio": kaggle_name,
                     "familia": ps.family_of(family), **values, "kaggle": public,
                     "test_mean": submission_mean(f"{kaggle_name}.csv"), "nota": note})
    result_127 = by_model.loc[NAME_127]
    rows.append({"numero": 127, "modelo": NAME_127, "envio": result_127["file"],
                 "familia": "boosting", **row_from_cv(result_127), "kaggle": np.nan,
                 "test_mean": submission_mean(result_127["submission"]), "nota": ""})

    table = pd.DataFrame(rows)
    for number, stats in twins.items():
        at = table["numero"] == number
        table.loc[at, "mae_random_twins"] = stats["mae"]
        table.loc[at, "nota"] = ("aleatoria con gemelos calculada aquí: regla mix sobre el 125"
                                 + (" y 0,7 × 127 + 0,3 × 148" if number == 152 else "")
                                 + ("; sin puntaje en Kaggle" if number == 127 else ""))
    for column in ["mae_spatial", "mae_weighted", "mae_northeast"]:
        table[column.replace("mae", "gap")] = table[column] - table["kaggle"]
    money = [c for c in table.columns if c.startswith(("mae_", "bias_", "gap_"))]
    table[money + ["kaggle", "test_mean"]] /= MILLION
    order = ["numero", "modelo", "familia", *SCHEMES, *REFERENCE, "kaggle", "bias_spatial",
             "bias_northeast", "gap_spatial", "gap_weighted", "gap_northeast", "test_mean",
             "envio", "nota"]
    return table[order].sort_values("kaggle", na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------- calidad del orden

def ranking_quality(table, column, subset):
    """Spearman con Kaggle, pares bien ordenados y brecha media de un esquema."""
    data = table.dropna(subset=[column, "kaggle"]).reset_index(drop=True)
    pairs = list(combinations(range(len(data)), 2))
    agree = [np.sign(data.loc[i, column] - data.loc[j, column])
             == np.sign(data.loc[i, "kaggle"] - data.loc[j, "kaggle"]) for i, j in pairs]
    gap = data[column] - data["kaggle"]
    best = data.loc[data[column].idxmin(), "numero"]
    return {"conjunto": subset, "esquema": (SCHEMES | REFERENCE)[column], "columna": column,
            "n": len(data), "envios": " ".join(map(str, sorted(data["numero"]))),
            "spearman": data[column].corr(data["kaggle"], method="spearman"),
            "pares_bien_ordenados": np.mean(agree), "brecha_media": gap.mean(),
            "brecha_absoluta_media": gap.abs().mean(), "mejor_segun_esquema": best,
            "mejor_en_kaggle": data.loc[data["kaggle"].idxmin(), "numero"]}


def rank_table(table):
    """Calidad por esquema: con todos sus envíos, sin el ingenuo 00 (que pesa mucho en la
    brecha media) y en el subconjunto común a los cuatro esquemas."""
    models = table[table["numero"] > 0]
    common = table.dropna(subset=list(SCHEMES) + ["kaggle"])
    subsets = {"todos los del esquema": table, "sin el ingenuo": models,
               "comunes a los cuatro esquemas": common}
    rows = [ranking_quality(data, column, subset) for subset, data in subsets.items()
            for column in [*SCHEMES, *REFERENCE]]
    return pd.DataFrame(rows)


def level_gap_link(table):
    """Correlación entre el sesgo espacial y la brecha espacial − Kaggle (sin el ingenuo):
    si es positiva, la espacial es más optimista con los modelos de nivel bajo."""
    data = table[table["numero"] > 0].dropna(subset=["bias_spatial", "gap_spatial"])
    return len(data), data["bias_spatial"].corr(data["gap_spatial"])


# ---------------------------------------------------------------- figuras

def family_legend(fig, families, **kwargs):
    """Leyenda por familia con los colores fijos del proyecto."""
    handles = [plt.Line2D([], [], marker="o", ls="", color=ps.FAMILY_COLORS[f],
                          label=ps.FAMILY_LABELS[f]) for f in families]
    fig.legend(handles=handles, **kwargs)


LABEL_OFFSETS = [(4, 2), (4, -10), (-5, 2), (-5, -10), (-2, 6), (-2, -15), (9, -4),
                 (-10, -4), (4, 10), (-5, 10), (4, -18), (-5, -18)]


def place_labels(ax, xs, ys, texts, fontsize=8):
    """Etiquetas cerca de cada punto sin chocar entre sí ni con los puntos: para cada una
    escoge, entre varias posiciones vecinas, la que menos se superpone con lo ya ubicado y
    penaliza las que quedan más cerca de otro punto que del propio (etiqueta ambigua)."""
    ax.figure.canvas.draw()
    points = ax.transData.transform(np.column_stack([xs, ys]))
    scale = ax.figure.dpi / 72
    placed = [(x - 6, y - 6, x + 6, y + 6) for x, y in points]
    for k, ((x, y), (xd, yd), text) in enumerate(zip(points, zip(xs, ys), texts)):
        width, height = 0.62 * fontsize * len(text) * scale, fontsize * scale
        best, best_cost = None, np.inf
        for dx, dy in LABEL_OFFSETS:
            x0 = x + dx * scale - (width if dx < 0 else 0)
            y0 = y + dy * scale
            box = (x0, y0, x0 + width, y0 + height)
            cost = sum(max(0, min(box[2], b[2]) - max(box[0], b[0]))
                       * max(0, min(box[3], b[3]) - max(box[1], b[1])) for b in placed)
            center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            distance = np.hypot(*(points - center).T)
            if np.delete(distance, k).min() < distance[k]:
                cost += 400
            if cost < best_cost:
                best, best_cost = (dx, dy, box), cost
        dx, dy, box = best
        placed.append(box)
        ax.annotate(text, (xd, yd), xytext=(dx, dy), textcoords="offset points",
                    fontsize=fontsize, ha="right" if dx < 0 else "left", va="bottom")


def kaggle_points(table):
    """Envíos con puntaje público, sin el ingenuo (377 M en Kaggle saca de escala los ejes)."""
    data = table.dropna(subset=["kaggle"])
    return data[data["numero"] > 0]


def draw_scheme_panels(axes, data, quality, label_size=8, title_size=10,
                       separator=", "):
    """Dibuja en cada eje un esquema contra Kaggle, con la línea de 45° y el número del envío."""
    stats = quality[quality["conjunto"] == "sin el ingenuo"].set_index("columna")
    for ax, (column, label) in zip(axes, SCHEMES.items()):
        points = data.dropna(subset=[column])
        low = 5 * np.floor(points[column].min() / 5) - 5
        high = max(5 * np.ceil(points[column].max() / 5) + 10, 165)
        ax.plot([80, 240], [80, 240], color="0.6", lw=1, ls="--")
        ax.scatter(points[column], points["kaggle"], s=40, zorder=3,
                   color=[ps.FAMILY_COLORS[f] for f in points["familia"]])
        ax.set_xlim(low, high)
        ax.set_ylim(140, 240)
        place_labels(ax, points[column].to_numpy(), points["kaggle"].to_numpy(),
                     points["numero"].astype(str).tolist(), fontsize=label_size)
        s = stats.loc[column]
        ax.set_title(f"{label}\nSpearman {ps.es_number(s['spearman'], 2)} (n = {s['n']})"
                     f"{separator}brecha media {ps.es_number(s['brecha_media'])} M",
                     fontsize=title_size)
        ax.set_xlabel("MAE del esquema (millones)")
    axes[0].set_ylabel("MAE público en Kaggle (millones)")


NOTE_VS_KAGGLE = ("Línea punteada: esquema = Kaggle; encima de ella, el esquema fue optimista "
                  "con ese envío. Sin el ingenuo 00 (377 M en Kaggle). Cada panel tiene su "
                  "propio eje horizontal.")


def plot_vs_kaggle(table, quality):
    """Un panel por esquema (2 × 2): MAE del esquema contra el puntaje público."""
    data = kaggle_points(table)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10.5), sharey=True)
    draw_scheme_panels(axes.flat, data, quality)
    axes[1, 0].set_ylabel("MAE público en Kaggle (millones)")
    fig.suptitle("Todos los esquemas son optimistas; el ponderado acierta el nivel de los "
                 "boosting,\npero no el de la red 148, el lineal, el CART ni los híbridos",
                 fontsize=13)
    family_legend(fig, sorted(data["familia"].unique()), loc="lower center", ncol=4,
                  bbox_to_anchor=(0.5, -0.05))
    fig.text(0.5, -0.08, NOTE_VS_KAGGLE.replace(" Sin", "\nSin"), ha="center", fontsize=9,
             color="0.4")
    fig.tight_layout()
    return ps.save(fig, "validation_vs_kaggle.png")


def plot_vs_kaggle_wide(table, quality):
    """La misma figura en una fila, para una diapositiva 16:9 (sin título general)."""
    data = kaggle_points(table)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.6), sharey=True)
    draw_scheme_panels(axes, data, quality, label_size=8, title_size=9.5, separator="\n")
    families = sorted(data["familia"].unique())
    family_legend(fig, families, loc="lower center", ncol=len(families),
                  bbox_to_anchor=(0.5, -0.07), fontsize=9.5, handletextpad=0.2,
                  columnspacing=1.2)
    fig.text(0.5, -0.12, NOTE_VS_KAGGLE, ha="center", fontsize=8.5, color="0.4")
    fig.tight_layout(w_pad=1.5)
    return ps.save(fig, "validation_vs_kaggle_wide.png")


def plot_rank_corr(quality):
    """Spearman por esquema, con todos sus envíos y en el subconjunto común."""
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list((SCHEMES | REFERENCE).values())
    x = np.arange(len(labels))
    subsets = [("todos los del esquema", "0.2"), ("sin el ingenuo", "0.5"),
               ("comunes a los cuatro esquemas", "0.8")]
    for offset, (subset, color) in zip([-0.28, 0, 0.28], subsets):
        part = quality[quality["conjunto"] == subset].set_index("esquema").loc[labels]
        bars = ax.bar(x + offset, part["spearman"], width=0.26, color=color, label=subset)
        for bar, (_, row) in zip(bars, part.iterrows()):
            ax.annotate(f"{ps.es_number(row['spearman'], 2)}\nn = {row['n']}",
                        (bar.get_x() + bar.get_width() / 2, max(row["spearman"], 0)),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 2),
                        textcoords="offset points")
    ax.set_xticks(x, [label.replace(" ", "\n", 1) for label in labels])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Spearman con Kaggle")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: ps.es_number(value)))
    ax.set_title("Con todos sus envíos, el holdout nororiental ordena mejor y la aleatoria "
                 "simple peor;\nen los 5 envíos comunes la espacial queda arriba, pero n = 5 "
                 "no alcanza para separarlas")
    return ps.save(fig, "validation_rank_corr.png")


def plot_twins(table, results):
    """Aleatoria simple contra aleatoria con gemelos en los envíos que tienen ambas, más el
    125 (el 127 sin la regla de gemelos) como referencia."""
    base = results[results["model"] == BASE_125]
    base = pd.DataFrame({"numero": [125], "familia": ["boosting"], "kaggle": [np.nan],
                         "mae_random": base["mae_random"].to_numpy() / MILLION,
                         "mae_random_twins": base["mae_random_twins"].to_numpy() / MILLION})
    data = pd.concat([table.dropna(subset=["mae_random", "mae_random_twins"]), base])
    data = data[data["numero"] > 0].sort_values("mae_random_twins", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(data))
    for i, (_, row) in enumerate(data.iterrows()):
        color = ps.FAMILY_COLORS[row["familia"]]
        ax.plot([row["mae_random"], row["mae_random_twins"]], [i, i], color=color, lw=2)
        ax.scatter(row["mae_random"], i, color=color, marker="o", s=50, zorder=3)
        ax.scatter(row["mae_random_twins"], i, color=color, marker="s", s=50, zorder=3)
        ax.annotate(f"+{ps.es_number(row['mae_random_twins'] - row['mae_random'])}",
                    (row["mae_random_twins"], i), xytext=(6, -3), textcoords="offset points",
                    fontsize=9)
        if not np.isnan(row["kaggle"]):
            ax.scatter(row["kaggle"], i, color="0.2", marker="*", s=90, zorder=3)
    names = {125: "125 boosting sin regla", 127: "127 = 125 + regla de gemelos"}
    ax.set_yticks(y, [names.get(row.numero,
                                f"{row.numero} {ps.FAMILY_LABELS[row.familia].lower()}")
                      for row in data.itertuples()])
    ax.set_xlabel("MAE (millones)")
    ax.set_xlim(80, 165)
    handles = [plt.Line2D([], [], marker=m, ls="", color="0.3", label=l, markersize=s)
               for m, l, s in [("o", "aleatoria simple", 7), ("s", "aleatoria con gemelos", 7),
                               ("*", "Kaggle público", 11)]]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
              fontsize=9)
    rise = data["mae_random_twins"] - data["mae_random"]
    by_number = data.set_index("numero")[["mae_random", "mae_random_twins"]]
    gain = by_number.loc[125] - by_number.loc[127]
    ax.set_title(f"Separar los gemelos sube la aleatoria entre {rise.min():.0f} y "
                 f"{rise.max():.0f} M; la regla de gemelos (127) gana\n"
                 f"{ps.es_number(gain['mae_random'])} M al 125 en la simple y solo "
                 f"{ps.es_number(gain['mae_random_twins'])} M con gemelos")
    return ps.save(fig, "validation_twins.png")


def plot_level(results, table):
    """Escalera de cuantiles: la espacial mejora al bajar el nivel; el ponderado y Kaggle no."""
    results = results.assign(numero=results["model"].map(number_of)).set_index("numero")
    ladder = results.loc[list(QUANTILE_LADDER)].assign(q=list(QUANTILE_LADDER.values()))
    others = results.loc[LOWER_LEVEL]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.2),
                                      gridspec_kw={"width_ratios": [1.4, 1]})
    for column, label, color, marker in [("mae_spatial", "espacial", "0.1", "o"),
                                         ("mae_weighted", "nororiental ponderado", "0.55", "s")]:
        x, y = ladder["bias_spatial"] / MILLION, ladder[column] / MILLION
        left.plot(x, y, "-", marker=marker, color=color,
                  label=f"{label}: cuantiles 0,60 a 0,80 (la 83 es 0,70)")
        for (_, row), xi, yi in zip(ladder.iterrows(), x, y):
            left.annotate(f"q{ps.es_number(row['q'], 2)}", (xi, yi), fontsize=8, color=color,
                          xytext=(5, -12 if column == "mae_spatial" else 6),
                          textcoords="offset points")
        x, y = others["bias_spatial"] / MILLION, others[column] / MILLION
        left.scatter(x, y, marker="D", facecolor="white", edgecolor=color, s=45, zorder=3)
        for number, xi, yi in zip(others.index, x, y):
            left.annotate(str(number), (xi, yi), xytext=(0, -14), textcoords="offset points",
                          fontsize=8, color=color, ha="center")
    left.set_xlim(10, 85)
    left.set_ylim(118, 163)
    left.set_xlabel("Sesgo medio en la espacial (millones; positivo = sobrepredice)")
    left.set_ylabel("MAE (millones)")
    left.legend(fontsize=8, loc="lower right")
    q60, q70 = ladder.loc[154], ladder.loc[83]
    left.set_title(
        f"Del cuantil 0,70 al 0,60 la espacial baja "
        f"{ps.es_number((q70['mae_spatial'] - q60['mae_spatial']) / MILLION)} M y el ponderado"
        f" sube {ps.es_number((q60['mae_weighted'] - q70['mae_weighted']) / MILLION)} M;\n"
        f"rombos: 127, 152 y los de nivel más bajo 181 y 186", fontsize=10)

    pair = table.set_index("numero").loc[[53, 65]]
    metrics = [("mae_spatial", "Espacial"), ("mae_weighted", "Ponderado"), ("kaggle", "Kaggle")]
    x = np.arange(len(metrics))
    for offset, (number, row), color in zip([-0.2, 0.2], pair.iterrows(), ["0.7", "0.3"]):
        values = [row[c] for c, _ in metrics]
        bars = right.bar(x + offset, values, width=0.38, color=color,
                         label=f"{number} (cuantil {'0,65' if number == 53 else '0,70'})")
        for bar, value in zip(bars, values):
            right.annotate(ps.es_number(value), (bar.get_x() + bar.get_width() / 2, value),
                           ha="center", va="bottom", fontsize=8, xytext=(0, 2),
                           textcoords="offset points")
    columns = [c for c, _ in metrics]
    gap = pair.loc[65, columns] - pair.loc[53, columns]
    right.set_xticks(x, [label for _, label in metrics])
    right.set_ylim(120, 170)
    right.set_ylabel("MAE (millones)")
    right.legend(fontsize=8, loc="upper left")
    right.set_title(f"Mismo boosting, dos cuantiles: la espacial prefiere el 0,65\n"
                    f"por {ps.es_number(gap['mae_spatial'])} M; Kaggle, el 0,70 por "
                    f"{ps.es_number(-gap['kaggle'])} M", fontsize=10)
    fig.suptitle("La espacial premia bajar el nivel de las predicciones; el tablero no",
                 fontsize=13)
    fig.tight_layout()
    return ps.save(fig, "validation_level.png")


def main():
    ps.setup()
    train, test = load_processed()
    oof = oof_aligned(train)
    table = schemes_table(train, test, oof)
    quality = rank_table(table)
    table.to_csv(VIEWS_DIR / "validation_schemes.csv", index=False)
    quality.to_csv(VIEWS_DIR / "validation_rank_corr.csv", index=False)
    results = pd.read_csv(VIEWS_DIR / "cv_results.csv")
    figures = [plot_vs_kaggle(table, quality), plot_vs_kaggle_wide(table, quality),
               plot_rank_corr(quality),
               plot_twins(table, results), plot_level(results, table)]
    for path in figures:
        print("figura:", path.name)
    n, corr = level_gap_link(table)
    print(f"sesgo espacial frente a brecha espacial menos Kaggle: r = {corr:.2f} (n = {n})")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.drop(columns=["modelo", "envio", "nota"]).round(1).to_string(index=False))
        print(quality.drop(columns=["envios"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
