"""Etapa 2, comparación de algoritmos: tablas y figuras para las presentaciones. No entrena
nada ni escribe envíos: lee views/cv_results.csv, views/kaggle_scores.csv, las trazas de
las búsquedas, stores/processed/best_params.json y las predicciones guardadas (fuera de
muestra y del test). Las mezclas 70/30 con el boosting 127 son un cálculo sobre esas
predicciones, no un modelo nuevo.

Salidas en views/:
  family_table.csv               mejor especificación por familia, hiperparámetros, búsqueda
                                 y MAE (aleatoria con gemelos, espacial, ponderado, Kaggle)
  kaggle_progression.png         récord del puntaje público y el cambio que produjo cada salto
  common_recipe.csv / .png       cada familia antes y después de la receta común
  family_blend_with_127.csv      correlación con la 127 y ganancia de una mezcla 70/30
  family_blend_with_127.png      correlación del error espacial frente a la ganancia
  family_test_range.csv / .png   máximo predicho en el test y % por encima de 1.650 M
  negative_results.csv / .png    ideas que no funcionaron frente a su referencia

Uso (desde scripts/):
  python 32_algorithm_comparison.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style as ps
from models import QuantileNet, northeast_building_groups
from utils import BEST_PARAMS_PATH, PROCESSED_DIR, TARGET, VIEWS_DIR, weighted_mae

M = 1e6
CAP = 1.65e9
NOISE_M = 2.0     # diferencias menores a 2 M se tratan como ruido (README)
BOOSTING = 127
BLEND_WEIGHT = 0.7

# --- Lectura ---


def load_results():
    """cv_results con el número de fila como índice."""
    results = pd.read_csv(VIEWS_DIR / "cv_results.csv")
    results["number"] = results["model"].str.split("_").str[0].astype(int)
    return results.set_index("number")


def load_kaggle():
    kaggle = pd.read_csv(VIEWS_DIR / "kaggle_scores.csv")
    kaggle["number"] = kaggle["model"].str.split("_").str[0].astype(int)
    return kaggle


def load_schemes():
    """Aleatoria con gemelos de views/validation_schemes.csv (si ya existe)."""
    path = VIEWS_DIR / "validation_schemes.csv"
    if not path.exists():
        print("Aviso: falta views/validation_schemes.csv; la aleatoria con gemelos de la "
              "127 y la 152 queda vacía")
        return pd.Series(dtype=float)
    return pd.read_csv(path).set_index("numero")["mae_random_twins"] * M


def family_from_name(name):
    """Familia a partir del nombre de un modelo ('32_boosting_full...' -> 'boosting')."""
    rest = name.split("_", 1)[1]
    keys = sorted(set(ps.FAMILY_COLORS) | set(ps.FAMILY_ALIASES), key=len, reverse=True)
    return ps.family_of(next(k for k in keys if rest.startswith(k)))


def column(frame, number, suffix=""):
    """Columna de un modelo por su número en un archivo de predicciones."""
    names = [c for c in frame.columns
             if c.split("_")[0] == f"{number:02d}" and c.endswith(suffix)
             and (suffix or "__" not in c)]
    if len(names) != 1:
        raise KeyError(f"{number}{suffix}: {names}")
    return names[0]


# --- Tabla por familia ---

FAMILY_ROWS = [
    # número, familia, especificación (texto), clave de best_params o None
    (108, "linear", "Lineal por tipo con splines de área y de lat/lon, estrato, terreno y "
     "ridge del texto; nivel al p70 de los residuos fuera de muestra", None),
    (104, "elastic_net", "Elastic net con la misma base de splines; nivel al p70 de los "
     "residuos", "elastic_net_q70"),
    (105, "cart", "Árbol por tipo sobre el precio en nivel que predice el p70 de cada hoja",
     "cart_q70"),
    (114, "random_forest", "Bosque cuantílico (p70) por tipo con ridge y red de texto, sin "
     "lat/lon", "random_forest_q70"),
    (127, "boosting", "LightGBM cuantil 0,70 por tipo, área del mismo punto, sin lat/lon, 10 "
     "semillas y regla de gemelos", None),
    (83, "boosting", "LightGBM cuantil 0,70 por tipo con estrato, terreno y ridge del texto, "
     "10 semillas (mejor boosting solo en Kaggle)", None),
    (148, "neural_net", "Red cuantílica (0,70) por tipo con estrato, terreno y ridge, sin "
     "lat/lon, 5 semillas", None),
    (152, "ensemble", "Mezcla 70 % boosting 127 y 30 % red 148", None),
]


def format_params(params):
    """'ruta__param': valor -> 'param = valor' legible."""
    items = []
    for path, value in params.items():
        name = path.split("__")[-1]
        if isinstance(value, float):
            value = f"{value:.3g}".replace(".", ",")
        items.append(f"{name} = {value}")
    return "; ".join(items)


def trace_rows(name):
    return len(pd.read_csv(VIEWS_DIR / name))


def boosting_defaults():
    """Hiperparámetros de la receta de la 83 (primer candidato de 15_optuna_search.py)."""
    first = pd.read_csv(VIEWS_DIR / "optuna_tpe.csv").iloc[0]
    keys = ["num_leaves", "max_depth", "learning_rate", "n_estimators", "min_child_samples",
            "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]
    return {k: (int(first[k]) if float(first[k]).is_integer() else float(first[k]))
            for k in keys}


def net_defaults():
    """Valores por defecto de models.QuantileNet (los de la red 148)."""
    net = QuantileNet()
    return {k: getattr(net, k) for k in ("hidden", "dropout", "lr", "weight_decay", "n_seeds")}


def search_text(number, results, public):
    """Qué búsqueda se hizo para la familia y qué resultó, con cifras de los archivos."""
    r = lambda n, col: ps.es_number(results.loc[n, col] / M)  # noqa: E731
    if number == 108:
        return "No aplica: mínimos cuadrados sin hiperparámetros"
    if number == 104:
        return (f"Optuna TPE, {trace_rows('tuning_elastic_net_q70.csv')} evaluaciones; "
                "objetivo = media de ponderado y espacial (17_tune_families.py)")
    if number == 105:
        return (f"Optuna TPE, {trace_rows('tuning_cart_q70.csv')} evaluaciones; objetivo = "
                "media de ponderado y espacial (17_tune_families.py)")
    if number == 114:
        confirm = pd.read_csv(VIEWS_DIR / "genetic_confirm.csv").groupby("params")["objective"]
        mean = confirm.mean() / M
        return (f"Optuna TPE, {trace_rows('tuning_random_forest_q70.csv')} evaluaciones (17); "
                f"luego NSGA-II, {trace_rows('genetic_nsga2.csv')} evaluaciones (19): no mejora "
                f"(objetivo con 5 semillas {ps.es_number(mean['random_forest_ga'], 2)} frente "
                f"a {ps.es_number(mean['random_forest_q70'], 2)} M)")
    if number in (127, 83):
        return (f"Sin sintonizar. Probado: búsqueda aleatoria en el holdout ponderado "
                f"({trace_rows('tuning_boosting_core_coords_by_type_level_q70_holdout.csv')} "
                f"evaluaciones, 05; fila 80: ponderado {r(80, 'mae_weighted')} frente a "
                f"{r(65, 'mae_weighted')} de la 65) y Optuna TPE contra CMA-ES "
                f"({trace_rows('optuna_tpe.csv')} y {trace_rows('optuna_cmaes.csv')}, 15; "
                f"fila 98: Kaggle {ps.es_number(public[98] / M)} frente a "
                f"{ps.es_number(public[83] / M)} de la 83)")
    if number == 148:
        return (f"Sin sintonizar. Probado: Optuna TPE, {trace_rows('tuning_quantile_net_q70.csv')}"
                f" evaluaciones con objetivo espacial (17) -> red 171: espacial "
                f"{r(171, 'mae_spatial')} (optimista) y ponderado {r(171, 'mae_weighted')}; su "
                f"mezcla 172 gana {ps.es_number((results.loc[152, 'mae_spatial'] - results.loc[172, 'mae_spatial']) / M)} M "
                "de espacial a la 152")
    return "Pesos fijos; se compararon 70/30 y 50/50 (152 y 153)"


def params_text(number, key, best):
    if key:
        return format_params(best[key])
    if number in (127, 83):
        return "Receta de la 83: " + format_params(boosting_defaults())
    if number == 148:
        return "Por defecto de la red: " + format_params(net_defaults())
    if number == 152:
        return "Pesos 0,7 y 0,3"
    return "Ninguno"


def family_table(results, kaggle, twins):
    """Una fila por familia con su mejor especificación y la mezcla."""
    best = json.loads(BEST_PARAMS_PATH.read_text())
    public = kaggle.set_index("number")["public_mae"]
    rows = []
    for number, family, spec, key in FAMILY_ROWS:
        row = results.loc[number]
        rows.append({
            "familia": ps.FAMILY_LABELS[family], "numero": number, "modelo": row["model"],
            "especificacion": spec, "sintonizado": "sí" if key else "no",
            "hiperparametros": params_text(number, key, best),
            "busqueda": search_text(number, results, public),
            "mae_random": row["mae_random"] / M,
            "mae_random_twins": (twins.get(number, row["mae_random_twins"])) / M,
            "mae_spatial": row["mae_spatial"] / M, "mae_weighted": row["mae_weighted"] / M,
            "kaggle": public.get(number, np.nan) / M,
        })
    table = pd.DataFrame(rows)
    table.to_csv(VIEWS_DIR / "family_table.csv", index=False)
    return table


# --- Progresión de Kaggle ---

# Cambio que produjo cada récord (verificado en el README, "Puntajes públicos de Kaggle").
RECORD_CHANGES = {
    0: "Mediana constante",
    10: "Boosting con coordenadas",
    32: "Nivel × 1,15 sobre la 10 (calibración)",
    53: "Un modelo por tipo, precio en nivel, q0,65, sin OSM",
    65: "Cuantil 0,70",
    75: "Texto completo (ridge sobre TF-IDF)",
    83: "Estrato y valor del terreno, 10 semillas",
    152: "Mezcla con la red (70 % boosting 127)",
}


def records(kaggle):
    """Envíos que bajaron el mejor puntaje, en orden de número de envío."""
    ordered = kaggle.sort_values("number")
    best = ordered["public_mae"].cummin()
    is_record = ordered["public_mae"].eq(best) & best.ne(best.shift())
    out = ordered[is_record].copy()
    missing = set(out["number"]) ^ set(RECORD_CHANGES)
    assert not missing, f"Los récords no coinciden con los rótulos: {missing}"
    return out


def plot_kaggle_progression(kaggle):
    data = records(kaggle).reset_index(drop=True)
    y = np.arange(len(data))[::-1]
    fig, ax = plt.subplots(figsize=ps.FULL)
    for yi, row in zip(y, data.itertuples()):
        family = family_from_name(row.model)
        ax.barh(yi, row.public_mae, color=ps.FAMILY_COLORS[family], height=0.62, lw=0)
        ax.text(row.public_mae + 4 * M, yi, ps.es_number(row.public_mae / M), va="center",
                color=ps.ANDES["black"])
    labels = [f"{n:02d}  {RECORD_CHANGES[n]}" for n in data["number"]]
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0)
    ps.millions(ax, "x")
    ax.set_xlim(0, data["public_mae"].max() * 1.1)
    ax.set_ylim(-0.6, len(data) - 0.4)
    ax.set_xlabel("MAE público en Kaggle (millones de $)")
    ax.grid(axis="y", visible=False)
    families = list(dict.fromkeys(family_from_name(m) for m in data["model"]))
    handles = [plt.Rectangle((0, 0), 1, 1, color=ps.FAMILY_COLORS[f], lw=0) for f in families]
    ax.legend(handles, [ps.FAMILY_LABELS[f] for f in families], loc="lower right",
              handlelength=1, handleheight=0.8)
    ps.save(fig, "kaggle_progression.png")


# --- Receta común ---

RECIPE = [  # familia, antes, ronda 5, mejor versión
    ("linear", 3, 101, 108), ("elastic_net", 4, 104, 104), ("cart", 21, 105, 105),
    ("random_forest", 24, 103, 114), ("neural_net", 11, 78, 148),
]
REFERENCE = 83


def common_recipe(results):
    rows = []
    for family, before, round5, after in RECIPE:
        for stage, number in [("antes", before), ("ronda 5", round5), ("mejor", after)]:
            rows.append({"familia": ps.FAMILY_LABELS[family], "etapa": stage,
                         "modelo": results.loc[number, "model"],
                         "mae_weighted": results.loc[number, "mae_weighted"] / M,
                         "mae_spatial": results.loc[number, "mae_spatial"] / M})
    table = pd.DataFrame(rows)
    table.to_csv(VIEWS_DIR / "common_recipe.csv", index=False)
    return table


def plot_common_recipe(results):
    """Flecha de cada familia desde su versión inicial hasta la mejor, con el boosting 83 de
    referencia. El número del modelo va junto a cada extremo, por fuera de la flecha."""
    fig, axes = plt.subplots(1, 2, figsize=ps.FULL, sharey=True)
    y = np.arange(len(RECIPE))[::-1]
    gray = ps.ANDES["gray"]
    for ax, metric, label in [(axes[0], "mae_weighted", "Holdout ponderado"),
                              (axes[1], "mae_spatial", "Validación espacial")]:
        values = [results.loc[n, metric] for _, b, _, a in RECIPE for n in (b, a)]
        pad = 0.04 * (max(values) - min(values))
        for yi, (family, before, _, after) in zip(y, RECIPE):
            color = ps.FAMILY_COLORS[family]
            start, end = results.loc[before, metric], results.loc[after, metric]
            ax.annotate("", xy=(end, yi), xytext=(start, yi),
                        arrowprops=dict(arrowstyle="-|>,head_length=0.4,head_width=0.2",
                                        color=color, lw=1.4, shrinkA=0, shrinkB=0))
            ax.plot(start, yi, "o", color=color, ms=3.5, mec="none")
            ax.text(start + pad, yi, f"{before:02d}", color=gray, va="center", fontsize=7)
            ax.text(end - pad, yi, str(after), color=gray, va="center", ha="right", fontsize=7,
                    bbox=dict(fc="white", ec="none", pad=0.3), zorder=4)
        ref = results.loc[REFERENCE, metric]
        ax.axvline(ref, color=ps.ANDES["gray"], ls="--", lw=0.8)
        ax.text(ref, 1.0, f"boosting 83: {ps.es_number(ref / M)}",
                transform=ax.get_xaxis_transform(), color=gray, ha="center", va="bottom",
                fontsize=7)
        ax.set_xlim(min(values + [ref]) - 7 * pad, max(values) + 5 * pad)
        ps.millions(ax, "x")
        ax.set_xlabel(f"{label}: MAE (millones de $)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [ps.FAMILY_LABELS[f] for f, *_ in RECIPE])
    axes[0].tick_params(axis="y", length=0)
    axes[1].tick_params(axis="y", length=0)
    axes[0].set_ylim(-0.5, len(RECIPE) - 0.5)
    ps.save(fig, "common_recipe.png")


# --- Por qué cada familia queda donde queda ---

PARTNERS = [108, 104, 105, 114, 148, 185, 83]


def northeast_groups():
    """Validación y grupos del holdout ponderado (como 07_results_table.py)."""
    train = pd.read_csv(PROCESSED_DIR / "train_features.csv")
    test = pd.read_csv(PROCESSED_DIR / "test_features.csv")
    return northeast_building_groups(train, test, train[TARGET])


def error_corr(pred, base, y, rows):
    """Correlación de los errores de dos modelos en las filas dadas."""
    return np.corrcoef(pred[rows] - y[rows], base[rows] - y[rows])[0, 1]


def blend_table(results):
    """Correlación con la 127 y MAE de la mezcla 70/30 con cada familia, en la validación
    espacial (todas las filas) y en el holdout ponderado (validación nororiental)."""
    oof = pd.read_csv(PROCESSED_DIR / "oof_predictions.csv")
    test = pd.read_csv(PROCESSED_DIR / "test_predictions.csv")
    val_idx, groups = northeast_groups()
    if len(oof) <= val_idx.max():
        raise SystemExit("oof_predictions.csv está incompleto (¿lo está escribiendo 04?); "
                         "vuelve a correr cuando termine de escribirse")
    y = oof[TARGET].to_numpy()
    base = {s: oof[column(oof, BOOSTING, f"__{s}")].to_numpy() for s in ("spatial", "northeast")}
    base_test = test[column(test, BOOSTING)]
    rows = []
    for number in PARTNERS:
        own = {s: oof[column(oof, number, f"__{s}")].to_numpy() for s in base}
        mix = {s: BLEND_WEIGHT * base[s] + (1 - BLEND_WEIGHT) * own[s] for s in base}
        ok = ~np.isnan(own["spatial"]) & ~np.isnan(base["spatial"])
        mae_sp = lambda p: np.mean(np.abs(p[ok] - y[ok])) / M  # noqa: E731
        mae_w = lambda p: weighted_mae(y[val_idx], p[val_idx], groups) / M  # noqa: E731
        family = ps.family_of(results.loc[number, "family"])
        rows.append({
            "numero": number, "familia": ps.FAMILY_LABELS[family], "family_key": family,
            "corr_test": test[column(test, number)].corr(base_test),
            "corr_error_spatial": error_corr(own["spatial"], base["spatial"], y, ok),
            "corr_error_northeast": error_corr(own["northeast"], base["northeast"], y, val_idx),
            "mae_spatial_solo": mae_sp(own["spatial"]), "mae_spatial_127": mae_sp(base["spatial"]),
            "mae_spatial_mezcla": mae_sp(mix["spatial"]),
            "mae_weighted_solo": mae_w(own["northeast"]),
            "mae_weighted_127": mae_w(base["northeast"]),
            "mae_weighted_mezcla": mae_w(mix["northeast"]),
        })
    table = pd.DataFrame(rows)
    table["ganancia_spatial"] = table["mae_spatial_mezcla"] - table["mae_spatial_127"]
    table["ganancia_weighted"] = table["mae_weighted_mezcla"] - table["mae_weighted_127"]
    table.drop(columns="family_key").to_csv(VIEWS_DIR / "family_blend_with_127.csv", index=False)
    return table


def plot_blend(table):
    """Correlación del error con la 127 frente a la ganancia de mezclar 70/30, por validación.
    La red 148 lleva etiqueta directa; las demás familias se leen en la leyenda común."""
    fig, axes = plt.subplots(1, 2, figsize=ps.FULL, sharey=True)
    panels = [(axes[0], "corr_error_spatial", "ganancia_spatial", "Validación espacial"),
              (axes[1], "corr_error_northeast", "ganancia_weighted", "Holdout ponderado")]
    for ax, x, y, label in panels:
        ax.axhspan(-NOISE_M, NOISE_M, color=ps.ANDES["light"], lw=0, zorder=0)
        ax.axhline(0, color=ps.ANDES["gray"], lw=0.6, zorder=1)
        for row in table.itertuples():
            ax.plot(getattr(row, x), getattr(row, y), ls="", ms=5, mec="none", zorder=3,
                    marker=ps.FAMILY_MARKERS[row.family_key],
                    color=ps.FAMILY_COLORS[row.family_key])
        net = table.set_index("numero").loc[148]
        ax.annotate(f"red 148: {ps.es_number(net[y])} M", (net[x], net[y]), xytext=(-6, -2),
                    textcoords="offset points", ha="right", va="center", fontsize=7,
                    color=ps.ANDES["black"])
        ax.xaxis.set_major_formatter(ps.es_formatter(2))
        ax.set_xlabel("Correlación del error con el boosting 127")
        ax.set_title(label, loc="left")
    axes[0].set_ylabel("Cambio de MAE al mezclar\n(millones de $; < 0 mejora)")
    axes[0].yaxis.set_major_locator(plt.MultipleLocator(2))
    axes[0].yaxis.set_major_formatter(ps.es_formatter(0))
    axes[0].text(0.02, -NOISE_M, "ruido (± 2 M)", transform=axes[0].get_yaxis_transform(),
                 fontsize=7, color=ps.ANDES["gray"], va="bottom")
    families = list(dict.fromkeys(table["family_key"]))
    handles = [plt.Line2D([], [], ls="", ms=5, mec="none", marker=ps.FAMILY_MARKERS[f],
                          color=ps.FAMILY_COLORS[f]) for f in families]
    labels = [f"{ps.FAMILY_LABELS[f]} "
              f"{', '.join(map(str, table.loc[table.family_key == f, 'numero']))}"
              for f in families]
    fig.legend(handles, labels, loc="outside lower center", ncol=len(families),
               handletextpad=0.2, columnspacing=1.0)
    ps.save(fig, "family_blend_with_127.png")


RANGE_MODELS = [3, 11, 108, 104, 105, 114, 127, 83, 148, 152, 185, 49]


def test_range(results):
    """Máximo predicho en el test y cuántas predicciones pasan el tope del train."""
    test = pd.read_csv(PROCESSED_DIR / "test_predictions.csv")
    rows = []
    for number in RANGE_MODELS:
        pred = test[column(test, number)]
        family = ps.family_of(results.loc[number, "family"])
        rows.append({"numero": number, "modelo": results.loc[number, "model"],
                     "familia": ps.FAMILY_LABELS[family], "family_key": family,
                     "max_test": pred.max() / M, "min_test": pred.min() / M,
                     "pct_sobre_1650": 100 * (pred > CAP).mean(),
                     "n_sobre_1650": int((pred > CAP).sum()),
                     "n_en_1650": int(np.isclose(pred, CAP, rtol=0, atol=0.5e6).sum())})
    table = pd.DataFrame(rows)
    table.drop(columns="family_key").to_csv(VIEWS_DIR / "family_test_range.csv", index=False)
    return table


def plot_test_range(table):
    """Predicción máxima en el test por modelo frente al tope del train. Cada barra lleva su
    máximo y, si llega al tope o lo pasa, cuántas predicciones lo hacen."""
    data = table.sort_values("max_test")
    y = np.arange(len(data))
    cap = CAP / M
    longest = data["max_test"].max()
    fig, ax = plt.subplots(figsize=ps.FULL)
    ax.barh(y, data["max_test"], color=[ps.FAMILY_COLORS[f] for f in data["family_key"]],
            height=0.62, lw=0)
    for yi, row in zip(y, data.itertuples()):
        text = f"{ps.es_number(row.max_test, 0)} M"
        if np.isclose(row.max_test, cap):
            text += f": {row.n_en_1650} en el tope"
        elif row.max_test > cap:
            text += f": {ps.es_number(row.pct_sobre_1650, 1)} % sobre el tope"
        inside = row.max_test == longest
        ax.text(row.max_test - 40 if inside else max(row.max_test, cap) + 40, yi, text,
                va="center",
                ha="right" if inside else "left", fontsize=7,
                color="white" if inside else ps.ANDES["black"])
    ax.axvline(cap, color=ps.ANDES["black"], ls="--", lw=0.8)
    ax.text(cap + 40, len(data) - 0.55, f"tope del train: {ps.es_number(cap, 0)} M", fontsize=7,
            color=ps.ANDES["gray"], va="bottom")
    ax.set_yticks(y, [f"{r.familia} {r.numero:02d}" for r in data.itertuples()], fontsize=7)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, longest * 1.02)
    ax.set_ylim(-0.6, len(data) + 0.2)
    ax.xaxis.set_major_formatter(ps.es_formatter(0))
    ax.set_xlabel("Predicción máxima en el test (millones de $)")
    ax.grid(axis="y", visible=False)
    ps.save(fig, "family_test_range.png")


# --- Resultados negativos ---

NEGATIVE = [  # categoría, fila, referencia, descripción
    ("OpenStreetMap", 16, 45, "Boosting en nivel con OSM (16) frente a sin OSM (45)"),
    ("OpenStreetMap", 33, 49, "Híbrido con OSM (33) frente a sin OSM (49)"),
    ("OpenStreetMap", 50, 51, "Híbrido q0,60 con OSM (50) frente a sin OSM (51)"),
    ("Modelo local", 158, 83, "Solo apartamentos de estrato 5 y 6 del nororiente"),
    ("Modelo local", 159, 83, "Radio de 3 km alrededor del test"),
    ("Modelo local", 160, 83, "Radio de 5 km"),
    ("Modelo local", 161, 83, "Radio de 8 km"),
    ("Apilamiento", 166, 83, "Apilamiento lineal ajustado en la espacial"),
    ("Apilamiento", 167, 83, "Apilamiento LightGBM ajustado en la espacial"),
    ("Balanceo", 134, 82, "Pesos 1 a 2 por decil, q0,65"),
    ("Balanceo", 138, 82, "Pesos 1 a 4 por decil, q0,65"),
    ("Balanceo", 141, 82, "Caros duplicados, q0,60"),
    ("Balanceo", 144, 82, "Caros triplicados, q0,55"),
    ("Monotonía", 182, 181, "Monótona en área, baños, parqueaderos, estrato y terreno"),
    ("Monotonía", 183, 181, "La 182 también monótona en el ridge del texto"),
    ("kNN", 185, 127, "kNN cuantílico por tipo (40 vecinos)"),
]

# Las ideas no son familias: todas van en negro y se distinguen por el marcador.
NEGATIVE_MARKERS = {"OpenStreetMap": "s", "Modelo local": "o", "Apilamiento": "^",
                    "Balanceo": "D", "Monotonía": "v", "kNN": "X"}


def negative_results(results):
    """Efecto de cada idea en espacial y ponderado frente a su referencia."""
    rows = []
    for category, number, ref, text in NEGATIVE:
        row, base = results.loc[number], results.loc[ref]
        rows.append({"categoria": category, "fila": number, "referencia": ref,
                     "descripcion": text, "modelo": row["model"],
                     "mae_spatial": row["mae_spatial"] / M, "mae_weighted": row["mae_weighted"] / M,
                     "ref_spatial": base["mae_spatial"] / M, "ref_weighted": base["mae_weighted"] / M})
    table = pd.DataFrame(rows)
    table["delta_spatial"] = table["mae_spatial"] - table["ref_spatial"]
    table["delta_weighted"] = table["mae_weighted"] - table["ref_weighted"]
    table["mejora_ambos"] = (table["delta_spatial"] < -NOISE_M) & (table["delta_weighted"] < -NOISE_M)
    table.to_csv(VIEWS_DIR / "negative_results.csv", index=False)
    return table


def plot_negative(table):
    assert not table["mejora_ambos"].any(), "Hay una idea que mejora los dos criterios"
    """Cada idea es un punto: cambio de MAE espacial (x) frente al ponderado (y). La esquina
    que mejora los dos por más del ruido queda vacía. Las ideas sin validación espacial van
    en una franja aparte a la izquierda, con el mismo eje y."""
    fig, (side, ax) = plt.subplots(1, 2, figsize=ps.FULL, sharey=True,
                                   gridspec_kw={"width_ratios": [1, 9]})
    has_spatial = table["delta_spatial"].notna()
    categories = list(dict.fromkeys(table["categoria"]))
    for category in categories:
        rows = table[table["categoria"] == category]
        style = dict(ls="", ms=5, mec="none", marker=NEGATIVE_MARKERS[category],
                     color=ps.ANDES["black"])
        ax.plot(rows["delta_spatial"], rows["delta_weighted"], label=category, **style)
        missing = rows[rows["delta_spatial"].isna()]
        side.plot(np.zeros(len(missing)), missing["delta_weighted"], **style)
    for panel in (side, ax):
        panel.axhspan(-NOISE_M, NOISE_M, color=ps.ANDES["light"], lw=0, zorder=0)
        panel.axhline(0, color=ps.ANDES["gray"], lw=0.6, zorder=1)
    ax.axvspan(-NOISE_M, NOISE_M, color=ps.ANDES["light"], lw=0, zorder=0)
    ax.axvline(0, color=ps.ANDES["gray"], lw=0.6, zorder=1)
    ax.set_xscale("symlog", linthresh=10, linscale=1.5)
    ax.set_xticks([-10, -5, 0, 5, 10, 20, 50])
    ax.xaxis.set_major_formatter(ps.es_formatter(0))
    ax.yaxis.set_major_formatter(ps.es_formatter(0))
    low = min(table["delta_spatial"].min(), table["delta_weighted"].min()) * 1.3
    ax.set_xlim(low, table["delta_spatial"].max() * 1.3)
    ax.set_ylim(low * 0.6, table["delta_weighted"].max() * 1.12)
    ax.fill_between([low, -NOISE_M], low, -NOISE_M, color=ps.UNDER_COLOR, alpha=0.12, lw=0)
    ax.text(low * 0.95, -NOISE_M - 0.6, "mejora ambas: vacío", color=ps.ANDES["gray"],
            fontsize=7, va="top")
    ax.set_xlabel("Cambio de MAE espacial (millones de $; > 0 empeora)")
    side.set_ylabel("Cambio de MAE ponderado\n(millones de $; > 0 empeora)")
    side.set_xlim(-1, 1)
    side.set_xticks([0], ["sin\nespacial"])
    side.tick_params(axis="x", length=0)
    side.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2, handletextpad=0.2, columnspacing=1.0)
    print(f"Negativos: {(~has_spatial).sum()} ideas sin validación espacial")
    ps.save(fig, "negative_results.png")


def main():
    ps.setup()
    results, kaggle = load_results(), load_kaggle()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    family = family_table(results, kaggle, load_schemes())
    print(family.drop(columns=["especificacion", "hiperparametros", "busqueda"]).round(1)
          .to_string(index=False))
    plot_kaggle_progression(kaggle)
    print(common_recipe(results).round(1).to_string(index=False))
    plot_common_recipe(results)
    blend = blend_table(results)
    print(blend.drop(columns="family_key").round(3).to_string(index=False))
    plot_blend(blend)
    ranges = test_range(results)
    print(ranges.drop(columns="family_key").round(1).to_string(index=False))
    plot_test_range(ranges)
    negative = negative_results(results)
    print(negative.drop(columns=["modelo", "descripcion"]).round(1).to_string(index=False))
    plot_negative(negative)


if __name__ == "__main__":
    main()
