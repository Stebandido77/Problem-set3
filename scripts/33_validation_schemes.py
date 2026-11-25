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
import numpy as np
import pandas as pd

import plot_style as ps
from models import northeast_building_groups
from utils import (ID_COL, PROCESSED_DIR, TARGET, VIEWS_DIR, find_submission,
                   load_kaggle_scores, load_processed, northeast_holdout, random_cv,
                   random_twin_splits, spatial_cv_splits, weighted_mae)

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
    scores = load_kaggle_scores()
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

LABELED = {203: "203", 152: "152", 83: "83", 148: "148"}
LABEL_SIZE = 7


def family_handles(families):
    """Marcadores de la leyenda por familia, con el color y la forma fijos del proyecto."""
    return [plt.Line2D([], [], marker=ps.FAMILY_MARKERS[f], ls="", color=ps.FAMILY_COLORS[f],
                       markeredgewidth=0, label=ps.FAMILY_LABELS[f]) for f in families]


def present_families(data):
    """Familias presentes, en el orden fijo de la paleta."""
    return [f for f in ps.FAMILY_COLORS if f in set(data["familia"])]


def scatter_families(ax, data, x, y, size=16):
    """Un punto por envío, con el color y el marcador de su familia y sin borde."""
    for family, part in data.groupby("familia"):
        marker = ps.FAMILY_MARKERS[family]
        ax.scatter(part[x], part[y], s=size * (2 if marker == "*" else 1), marker=marker,
                   color=ps.FAMILY_COLORS[family], linewidths=0, zorder=3)


LABEL_OFFSETS = [(4, 2), (4, -9), (-4, 2), (-4, -9), (0, 5), (0, -12), (7, -4), (-7, -4),
                 (10, 8), (10, -16), (-10, 8), (-10, -16)]
# Posición fija para la 203, que queda junto a la 152 en la espacial.
PREFERRED_OFFSET = {203: (-4, -9)}


def place_labels(ax, data, x, y, labels):
    """Etiqueta solo los envíos de 'labels'. Para cada uno escoge, entre posiciones vecinas,
    la que menos se superpone con los demás puntos y con las etiquetas ya puestas."""
    ax.figure.canvas.draw()
    points = ax.transData.transform(data[[x, y]].to_numpy(float))
    scale = ax.figure.dpi / 72
    frame = ax.get_window_extent()
    placed = [(px - 4 * scale, py - 4 * scale, px + 4 * scale, py + 4 * scale)
              for px, py in points]
    for k, number in enumerate(data["numero"]):
        if number not in labels:
            continue
        text = labels[number]
        (px, py), width = points[k], 0.55 * LABEL_SIZE * len(text) * scale
        height = LABEL_SIZE * scale
        options = []
        offsets = [PREFERRED_OFFSET[number]] if number in PREFERRED_OFFSET else LABEL_OFFSETS
        for dx, dy in offsets:
            x0 = px + dx * scale - (width if dx < 0 else width / 2 if dx == 0 else 0)
            box = (x0, py + dy * scale, x0 + width, py + dy * scale + height)
            cost = sum(max(0, min(box[2], b[2]) - max(box[0], b[0]))
                       * max(0, min(box[3], b[3]) - max(box[1], b[1])) for b in placed)
            if box[0] < frame.x0 or box[2] > frame.x1 or box[1] < frame.y0 or box[3] > frame.y1:
                cost += 1e6
            options.append((cost, dx, dy, box))
        _, dx, dy, box = min(options, key=lambda option: option[0])
        placed.append(box)
        ax.annotate(text, (data[x].iloc[k], data[y].iloc[k]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=LABEL_SIZE, color=ps.ANDES["black"],
                    ha="right" if dx < 0 else "center" if dx == 0 else "left", va="bottom")


def kaggle_points(table):
    """Envíos con puntaje público, sin el ingenuo (377 M en Kaggle saca de escala los ejes)."""
    data = table.dropna(subset=["kaggle"])
    return data[data["numero"] > 0]


def draw_scheme_panels(axes, data, quality):
    """Cada eje: un esquema contra Kaggle, con la línea de 45° y el Spearman en el título."""
    stats = quality[quality["conjunto"] == "sin el ingenuo"].set_index("columna")
    for ax, (column, label) in zip(axes, SCHEMES.items()):
        points = data.dropna(subset=[column]).reset_index(drop=True)
        low = 10 * np.floor(points[column].min() / 10) - 5
        high = max(10 * np.ceil(points[column].max() / 10) + 5, 165)
        ax.plot([80, 240], [80, 240], color=ps.ANDES["mid"], lw=0.8, ls="--", zorder=1)
        scatter_families(ax, points, column, "kaggle")
        ax.set_xlim(low, high)
        ax.set_ylim(130, 240)
        ax.set_yticks(range(140, 241, 20))
        place_labels(ax, points, column, "kaggle", LABELED)
        s = stats.loc[column]
        ax.set_title(f"{label}\nSpearman {ps.es_number(s['spearman'], 2)} (n = {s['n']})",
                     fontweight="normal")
        ax.set_xlabel("MAE del esquema (M)")
        ax.xaxis.set_major_formatter(ps.es_formatter())
        ax.yaxis.set_major_formatter(ps.es_formatter())
    axes[0].set_ylabel("MAE público en Kaggle (M)")
    edge = axes[0].get_xlim()[1]
    axes[0].annotate("esquema = Kaggle", (edge, edge), xytext=(-2, 2),
                     textcoords="offset points", ha="right", va="bottom", fontsize=LABEL_SIZE,
                     color=ps.ANDES["gray"])


def plot_vs_kaggle(table, quality):
    """Un panel por esquema (2 × 2): MAE del esquema contra el puntaje público."""
    data = kaggle_points(table)
    fig, axes = plt.subplots(2, 2, figsize=(ps.FULL[0], 2 * ps.FULL[1]), sharey=True)
    draw_scheme_panels(axes.flat, data, quality)
    axes[1, 0].set_ylabel("MAE público en Kaggle (M)")
    fig.legend(handles=family_handles(present_families(data)), loc="outside lower center",
               ncol=4)
    return ps.save(fig, "validation_vs_kaggle.png")


def plot_vs_kaggle_wide(table, quality):
    """La misma figura en una fila, para una diapositiva 16:9."""
    data = kaggle_points(table)
    fig, axes = plt.subplots(1, 4, figsize=ps.FULL_TALL, sharey=True)
    draw_scheme_panels(axes, data, quality)
    families = present_families(data)
    fig.legend(handles=family_handles(families), loc="outside lower center",
               ncol=4, handletextpad=0.1, columnspacing=1.2)
    return ps.save(fig, "validation_vs_kaggle_wide.png")


def plot_rank_corr(quality):
    """Spearman por esquema, con todos sus envíos y en el subconjunto común."""
    fig, ax = plt.subplots(figsize=ps.FULL)
    labels = list((SCHEMES | REFERENCE).values())
    x = np.arange(len(labels))
    subsets = [("todos los del esquema", ps.ANDES["black"]), ("sin el ingenuo", "#7A7A7A"),
               ("comunes a los cuatro esquemas", ps.ANDES["mid"])]
    for offset, (subset, color) in zip([-0.27, 0, 0.27], subsets):
        part = quality[quality["conjunto"] == subset].set_index("esquema").loc[labels]
        ax.bar(x + offset, part["spearman"], width=0.25, color=color, label=subset)
    ax.set_xticks(x, [label.replace(" ", "\n", 1) for label in labels])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Spearman con Kaggle")
    ax.grid(axis="x", visible=False)
    fig.legend(loc="outside upper center", ncol=3)
    ax.yaxis.set_major_formatter(ps.es_formatter(1))
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
    fig, ax = plt.subplots(figsize=ps.FULL)
    right_edge = 172
    for i, row in enumerate(data.itertuples()):
        color = ps.FAMILY_COLORS[row.familia]
        ax.plot([row.mae_random, row.mae_random_twins], [i, i], color=color, lw=1.2,
                solid_capstyle="butt", zorder=2)
        ax.scatter(row.mae_random, i, color="white", edgecolors=color, marker="o", s=16,
                   linewidths=1.1, zorder=3)
        ax.scatter(row.mae_random_twins, i, color=color, marker="o", s=16, linewidths=0,
                   zorder=3)
        if not np.isnan(row.kaggle):
            ax.scatter(row.kaggle, i, color=ps.ANDES["black"], marker="|", s=40,
                       linewidths=1.2, zorder=3)
        ax.text(right_edge, i, f"+{ps.es_number(row.mae_random_twins - row.mae_random)}",
                ha="right", va="center", fontsize=LABEL_SIZE, color=ps.ANDES["gray"])
    ax.text(right_edge, len(data) - 0.4, "subida (M)", ha="right", va="bottom",
            fontsize=LABEL_SIZE, color=ps.ANDES["gray"])
    names = {125: "125 boosting sin regla de gemelos", 127: "127 = 125 + regla de gemelos"}
    ax.set_yticks(range(len(data)),
                  [names.get(row.numero, f"{row.numero} {ps.FAMILY_LABELS[row.familia]}")
                   for row in data.itertuples()])
    ax.set_ylim(-0.6, len(data) - 0.1)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("MAE (millones de pesos)")
    ax.set_xlim(80, right_edge + 1)
    ax.xaxis.set_major_formatter(ps.es_formatter())
    gray = ps.ANDES["gray"]
    handles = [plt.Line2D([], [], marker="o", ls="", markerfacecolor="white",
                          markeredgecolor=gray, markeredgewidth=1.1, label="aleatoria simple"),
               plt.Line2D([], [], marker="o", ls="", color=gray, markeredgewidth=0,
                          label="aleatoria con gemelos"),
               plt.Line2D([], [], marker="|", ls="", color=ps.ANDES["black"], markersize=7,
                          markeredgewidth=1.2, label="Kaggle público")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3)
    return ps.save(fig, "validation_twins.png")


LEVEL_SERIES = [("mae_spatial", "Espacial", ps.ANDES["black"], "o"),
                ("mae_weighted", "Nororiental ponderado", "#8C8C8C", "s")]
QUANTILE_SHADES = {53: "#E3C25A", 65: ps.ANDES["accent"]}


def plot_level_ladder(ax, results):
    """Escalera de cuantiles del boosting: sesgo espacial contra MAE en dos esquemas."""
    ladder = results.loc[list(QUANTILE_LADDER)].assign(q=list(QUANTILE_LADDER.values()))
    others = results.loc[LOWER_LEVEL]
    for column, label, color, marker in LEVEL_SERIES:
        x, y = ladder["bias_spatial"] / MILLION, ladder[column] / MILLION
        ax.plot(x, y, "-", marker=marker, color=color, lw=1.2, ms=3.5, markeredgewidth=0)
        ax.annotate(label, (x.iloc[-1], y.iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=LABEL_SIZE,
                    color=ps.ANDES["black"])
        ax.scatter(others["bias_spatial"] / MILLION, others[column] / MILLION, marker="D",
                   facecolor="white", edgecolor=color, s=12, linewidths=0.9, zorder=3)
    spatial = ladder["mae_spatial"] / MILLION
    for row, xi, yi in zip(ladder.itertuples(), ladder["bias_spatial"] / MILLION, spatial):
        if row.q in (0.60, 0.70, 0.80):
            ax.annotate(f"q{ps.es_number(row.q, 2)}", (xi, yi), xytext=(-2, 3),
                        textcoords="offset points", ha="right", fontsize=LABEL_SIZE,
                        color=ps.ANDES["gray"])
    for number, xi, yi in zip(others.index, others["bias_spatial"] / MILLION,
                              others["mae_spatial"] / MILLION):
        ax.annotate(str(number), (xi, yi), xytext=(0, -10), textcoords="offset points",
                    fontsize=LABEL_SIZE, color=ps.ANDES["gray"], ha="center")
    ax.set_xlim(10, 95)
    ax.set_ylim(115, 165)
    ax.set_xlabel("Sesgo medio en la espacial (M; + = sobrepredice)")
    ax.set_ylabel("MAE (millones de pesos)")
    ax.set_title("Cuantiles 0,60 a 0,80; rombos: 127, 152, 181, 186",
                 fontweight="normal")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ps.es_formatter())


def plot_level_pair(ax, table):
    """El mismo boosting con cuantil 0,65 (53) y 0,70 (65) en tres esquemas."""
    pair = table.set_index("numero").loc[[53, 65]]
    metrics = [("mae_spatial", "Espacial"), ("mae_weighted", "Ponderado"), ("kaggle", "Kaggle")]
    x = np.arange(len(metrics))
    for offset, (number, row) in zip([-0.15, 0.15], pair.iterrows()):
        values = [row[c] for c, _ in metrics]
        quantile = "0,65" if number == 53 else "0,70"
        ax.scatter(x + offset, values, s=22, color=QUANTILE_SHADES[number], linewidths=0,
                   zorder=3, label=f"{number} (cuantil {quantile})")
        for xi, value in zip(x + offset, values):
            # 0,65 a la izquierda del punto y 0,70 encima, para que no choquen.
            left = number == 53
            ax.annotate(ps.es_number(value), (xi, value), xytext=(-3, 0) if left else (0, 3),
                        textcoords="offset points", va="center" if left else "bottom",
                        ha="right" if left else "center", fontsize=LABEL_SIZE,
                        color=ps.ANDES["black"])
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_xlim(-0.6, len(metrics) - 0.4)
    ax.set_ylim(135, 167)
    ax.set_ylabel("MAE (millones de pesos)")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="lower right", handletextpad=0.1, borderaxespad=0.2)
    ax.set_title("Mismo boosting, dos cuantiles", fontweight="normal")
    ax.yaxis.set_major_formatter(ps.es_formatter())


def plot_level(results, table):
    """Escalera de cuantiles: la espacial mejora al bajar el nivel; el ponderado y Kaggle no."""
    results = results.assign(numero=results["model"].map(number_of)).set_index("numero")
    fig, (left, right) = plt.subplots(1, 2, figsize=ps.FULL,
                                      gridspec_kw={"width_ratios": [1.3, 1]})
    plot_level_ladder(left, results)
    plot_level_pair(right, table)
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
