"""Errores del mejor modelo y pérdida asimétrica (etapa 4, piezas 1 a 3). Lee las
predicciones fuera de muestra de la validación espacial (la que imita el salto del resto de
Bogotá a Chapinero) y no las modifica.

1. Anatomía del error de la mezcla 203 (el mejor puntaje en Kaggle): sesgo, porcentaje
   sobrepredicho y tamaño medio de la sobre y de la subpredicción, en total y por rango de
   precio, área disponible, estrato, tipo, zona y año. Como referencia del cambio de zona,
   las mismas cifras en el holdout nororiental (sin ponderar).
2. Importancia por ganancia del boosting: la receta de la 125 (base de la 127) ajustada una
   vez con todo el train y una semilla, para apartamentos y para casas.
3. Pérdida asimétrica: costo medio cuando sobrepredecir cuesta k veces lo que subpredecir
   (k = 1 es el MAE) y ranking de modelos para k = 1, 2 y 3. Las filas 190 a 192 (la receta
   de la 125 con cuantiles 0,20, 0,30 y 0,50) reciben la regla 'mix' de gemelos de
   20_twin_rules.py, igual que la 127. Las columnas que falten se omiten con aviso.

Salidas:
  views/best_errors_segments.csv     anatomía del error por segmento (espacial y nororiente)
  views/best_errors_segments.png     sobre y subpredicción media por segmento
  views/best_errors_importance.csv   ganancia de cada predictor, por tipo de inmueble
  views/best_errors_importance.png   los predictores con más ganancia
  views/asymmetric_ranking.csv       costo asimétrico y ranking por k
  views/asymmetric_ranking.png       ranking de modelos por costo

Uso:
  python scripts/34_best_model_errors.py              (todo)
  python scripts/34_best_model_errors.py --skip-importance
"""

import argparse
import importlib

import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

import plot_style
from plot_style import FAMILY_COLORS, es_number, family_of, plt
from utils import (PROCESSED_DIR, RAW_DIR, TARGET, VIEWS_DIR, load_processed,
                   northeast_holdout, spatial_cv_splits, spatial_groups)

MILLION = 1e6
BEST = "203_ensemble_w70_30_2"
BEFORE_BETO = "152_ensemble_w70_30_2"
BASE_125 = "125_boosting_pt_core_coords_by_type_level_q70_ridge_nocoords_s10"
LOW_QUANTILES = {190: 20, 191: 30, 192: 50}
# Modelos del ranking: nombre en cv_results -> (etiqueta corta, en la ruta principal).
RANKED = {
    "108_linear_ext_core_coords_splines_by_type_level_q70_ridge_oof": ("108 lineal", True),
    "104_elastic_net_ext_core_coords_splines_by_type_level_q70_ridge_tuned":
        ("104 elastic net", True),
    "105_cart_ext_core_coords_by_type_level_q70_ridge_tuned": ("105 CART", True),
    "114_random_forest_ext_core_coords_by_type_level_q70_ridge_net_nocoords":
        ("114 random forest", True),
    "83_boosting_ext_core_coords_by_type_level_q70_ridge_s10": ("83 boosting", True),
    "127_boosting_pt_core_coords_by_type_level_q70_ridge_nocoords_s10_twin_mix":
        ("127 boosting q0,70", True),
    "148_quantile_net_ext_core_coords_by_type_q70_ridge_nocoords": ("148 red", True),
    BEFORE_BETO: ("152 mezcla sin BETO", True),
    BEST: ("203 mezcla con BETO", False),
    "186_ensemble_w70_30_2": ("186 mezcla", True),
}
COST_RATIOS = (1, 2, 3)
PRICE_BINS = [0, 400, 500, 700, 900, 1200, np.inf]
PRICE_LABELS = ["<400", "400-500", "500-700", "700-900", "900-1.200", "1.200+"]
SEGMENT_TITLES = {"precio_observado": "Precio pedido (M, desde)",
                  "precio_predicho": "Precio predicho (M, desde)", "area": "Área tomada de",
                  "estrato": "Estrato", "tipo": "Tipo", "zona": "Zona espacial",
                  "año": "Año del anuncio"}
# Etiquetas del eje en una línea: los rangos de precio se rotulan por su límite inferior (el
# título del panel lo dice); los valores del CSV no cambian.
SEGMENT_TICKS = {"400-500": "400", "500-700": "500", "700-900": "700", "900-1.200": "900",
                 "anuncio o texto": "anuncio", "faltante": "sin área",
                 "sin dato": "s. d."}
OVER_COLOR, UNDER_COLOR = plot_style.OVER_COLOR, plot_style.UNDER_COLOR
TOP_FEATURES = 12
FEATURE_LABELS = {
    "txt_ridge": "Texto (ridge TF-IDF)", "surface_pt": "Área (con mismo punto)",
    "bathrooms": "Baños", "bedrooms": "Alcobas", "rooms": "Cuartos",
    "ext_stratum": "Estrato", "ext_land_value_m2": "Valor del terreno por m²",
    "pt_surface_median": "Área mediana del mismo punto", "pt_count": "Anuncios en el mismo punto",
    "pt_bathrooms_median": "Baños medianos del mismo punto",
    "surface_pt_imputed": "Área recuperada (indicador)", "surface_missing": "Área faltante",
    "parking": "Parqueaderos", "dist_cerros_m": "Distancia a los cerros",
    "dist_carrera_7_m": "Distancia a la carrera 7", "year": "Año", "month": "Mes",
    "surface": "Área del anuncio o del texto", "log_surface_pt": "Log del área (mismo punto)",
    "txt_desc_length": "Largo de la descripción", "time_index": "Fecha del anuncio",
    "dist_centro_internacional_m": "Distancia al Centro Internacional",
    "dist_zona_t_m": "Distancia a la Zona T", "dist_plaza_bolivar_m": "Distancia a la Plaza de Bolívar",
    "dist_parque_93_m": "Distancia al Parque de la 93",
}


def twin_mix_spatial(name, train, oof):
    """Predicción espacial con la regla 'mix' de gemelos (20_twin_rules.py), la misma que
    convierte la 125 en la 127. Devuelve None si la columna base no existe."""
    column = f"{name}__spatial"
    if column not in oof.columns:
        return None
    twins = importlib.import_module("20_twin_rules")
    pred, _ = twins.evaluate_scheme("mix", oof[column].to_numpy(float),
                                    twins.normalized_description(train),
                                    train[TARGET].to_numpy(float), spatial_cv_splits(train))
    return pred


def low_quantile_name(number):
    """Nombre en oof_predictions.csv de la receta de la 125 con un cuantil bajo."""
    q = LOW_QUANTILES[number]
    return f"{number}_boosting_pt_core_coords_by_type_level_q{q}_ridge_nocoords_s10"


def load_oof():
    return pd.read_csv(PROCESSED_DIR / "oof_predictions.csv")


def error_stats(y, pred):
    """Sesgo, MAE, porcentaje sobrepredicho y tamaño medio de cada tipo de error."""
    err = pred - y
    over, under = err[err > 0], err[err < 0]
    return {"n": len(err), "mae": np.abs(err).mean(), "bias": err.mean(),
            "share_over": (err > 0).mean(),
            "mean_over": over.mean() if len(over) else 0.0,
            "mean_under": -under.mean() if len(under) else 0.0,
            "over_pct_price": (over / y[err > 0]).mean() if len(over) else 0.0}


def area_status(train):
    """Área efectiva del modelo: del anuncio o del texto, recuperada del mismo punto o
    faltante."""
    status = np.where(train["surface_pt"].isna(), "faltante", "anuncio o texto")
    return np.where(train["surface_pt_imputed"] == 1, "mismo punto", status)


def price_range(values):
    return pd.cut(values / MILLION, PRICE_BINS, labels=PRICE_LABELS, right=False).astype(str)


def segments(train, pred):
    """Segmentos de cada fila; el precio predicho es lo único que ve el comprador."""
    raw = pd.read_csv(RAW_DIR / "train.csv", usecols=["property_id", "year"])
    year = train["property_id"].map(dict(zip(raw["property_id"], raw["year"])))
    return pd.DataFrame({
        "total": "todas",
        "precio_observado": price_range(train[TARGET].to_numpy(float)),
        "precio_predicho": price_range(pred),
        "area": area_status(train),
        "estrato": (train["ext_stratum"].fillna(-1).astype(int).astype(str)
                    .replace("-1", "sin dato")),
        "tipo": np.where(train["is_house"] == 1, "casa", "apartamento"),
        "zona": spatial_groups(train).astype(str),
        "año": year.astype(int).astype(str),
    })


def anatomy(train, oof):
    """Anatomía del error de la 203 en la espacial y en el holdout nororiental."""
    y = train[TARGET].to_numpy(float)
    _, test = load_processed()
    _, ne_idx = northeast_holdout(train, test)
    rows = []
    for scheme in ("spatial", "northeast"):
        pred = oof[f"{BEST}__{scheme}"].to_numpy(float)
        keep = np.arange(len(y)) if scheme == "spatial" else ne_idx
        seg = segments(train, np.nan_to_num(pred)).iloc[keep]
        for variable in seg.columns:
            for value, idx in seg.groupby(variable).indices.items():
                rows_idx = keep[idx]
                rows.append({"scheme": scheme, "variable": variable, "segment": value,
                             **error_stats(y[rows_idx], pred[rows_idx])})
    table = pd.DataFrame(rows)
    table["share_rows"] = table["n"] / table.groupby(["scheme", "variable"])["n"].transform("sum")
    return table


def segment_order(variable, values):
    if variable.startswith("precio"):
        return [v for v in PRICE_LABELS if v in values]
    if variable == "zona":
        return sorted(values, key=int)
    return sorted(values)


def plot_anatomy(table):
    """Sobre y subpredicción media por segmento, con el sesgo y el % sobrepredicho."""
    spatial = table[table["scheme"] == "spatial"]
    # Los precios van arriba, a medio ancho cada uno, para que sus seis rangos quepan en una
    # línea; abajo, área, tipo y estrato. La zona espacial queda en el CSV.
    fig, axd = plt.subplot_mosaic([["precio_observado"] * 6 + ["precio_predicho"] * 6,
                                   ["area"] * 5 + ["tipo"] * 3 + ["estrato"] * 4],
                                  figsize=plot_style.FULL_TALL, sharey=True)
    for variable, ax in axd.items():
        part = spatial[spatial["variable"] == variable].set_index("segment")
        part = part.loc[segment_order(variable, part.index)]
        x = np.arange(len(part))
        ax.bar(x, part["mean_over"] / MILLION, color=OVER_COLOR, width=0.7,
               label="Sobrepredicción media")
        ax.bar(x, -part["mean_under"] / MILLION, color=UNDER_COLOR, width=0.7,
               label="Subpredicción media")
        ax.plot(x, part["bias"] / MILLION, "o", color=plot_style.ANDES["black"], ms=2.5,
                markeredgewidth=0, label="Sesgo medio")
        ax.axhline(0, color=plot_style.ANDES["gray"], lw=0.6)
        labels = [SEGMENT_TICKS.get(s, s) for s in part.index]
        ax.set_xticks(x, labels, fontsize=plot_style.LABEL_SIZE)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", visible=False)
        ax.set_title(SEGMENT_TITLES[variable])
        ax.set_xlim(-0.6, len(x) - 0.4)
        ax.yaxis.set_major_locator(MultipleLocator(200))
        ax.yaxis.set_major_formatter(plot_style.es_formatter())
    for variable in ("precio_observado", "area"):
        axd[variable].set_ylabel("Error medio (M)")
    handles, labels = axd["precio_observado"].get_legend_handles_labels()
    order = [1, 2, 0]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="outside lower center", ncol=3)
    return plot_style.save(fig, "best_errors_segments.png")


def lgbm_importance(model, kind):
    """Ganancia de cada predictor del LightGBM de un tipo, con las capas de la receta."""
    pipeline = model.model_.model_.models_[kind].regressor_
    booster = pipeline.steps[-1][1].booster_
    return pd.DataFrame({"feature": booster.feature_name(),
                         "gain": booster.feature_importance("gain")})


def importance(train):
    """Receta de la 125 con todo el train y una semilla (la 125 promedia 10)."""
    trainer = importlib.import_module("04_train_models")
    from utils import feature_sets
    spec = trainer.Spec("boosting", "pt_core_coords", "by_type_level_q70_ridge_nocoords_s1",
                        drop_coords=True, n_seeds=1, text_sources=("ridge",), target="level",
                        by_type=True, params=trainer.quantile_loss(0.70))
    features = trainer.spec_features(spec, feature_sets(train.columns))
    model = trainer.build_spec_model(spec, {})
    model.fit(train[features], train[TARGET])
    frames = []
    for kind, label in [(0, "apartamento"), (1, "casa")]:
        part = lgbm_importance(model, kind).assign(tipo=label)
        part["gain_share"] = part["gain"] / part["gain"].sum()
        frames.append(part.sort_values("gain_share", ascending=False))
    return pd.concat(frames)


def plot_importance(table):
    """Los predictores con más ganancia del boosting, por tipo de inmueble."""
    fig, axes = plt.subplots(1, 2, figsize=plot_style.FULL, sharex=True)
    for ax, kind in zip(axes, ["apartamento", "casa"]):
        part = table[table["tipo"] == kind].head(TOP_FEATURES).iloc[::-1]
        labels = [FEATURE_LABELS.get(f, f) for f in part["feature"]]
        share = part["gain_share"].to_numpy() * 100
        ax.barh(labels, share, color=FAMILY_COLORS["boosting"], height=0.7)
        ax.text(share[-1] + 0.4, len(share) - 1, f"{es_number(share[-1], 0)} %",
                va="center", fontsize=plot_style.LABEL_SIZE, color=plot_style.ANDES["black"])
        ax.set_title("Apartamentos" if kind == "apartamento" else "Casas")
        ax.set_xlabel("Ganancia (% del total)")
        ax.tick_params(axis="y", length=0, labelsize=plot_style.LABEL_SIZE)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(share) - 0.4)
        ax.xaxis.set_major_formatter(plot_style.es_formatter())
    return plot_style.save(fig, "best_errors_importance.png")


def asymmetric_cost(y, pred, k):
    """Costo medio cuando sobrepredecir cuesta k veces lo que subpredecir."""
    err = pred - y
    return np.mean(np.where(err > 0, k * err, -err))


def ranked_predictions(train, oof):
    """Predicciones espaciales de los modelos del ranking; avisa las que faltan."""
    families = pd.read_csv(VIEWS_DIR / "cv_results.csv").set_index("model")["family"]
    out = {}
    for name, (label, main_path) in RANKED.items():
        column = f"{name}__spatial"
        if column not in oof.columns:
            print(f"Aviso: falta {column}; se omite del ranking")
            continue
        out[name] = (label, family_of(families.get(name, "boosting")), main_path,
                     oof[column].to_numpy(float))
    for number, q in LOW_QUANTILES.items():
        pred = twin_mix_spatial(low_quantile_name(number), train, oof)
        if pred is None:
            print(f"Aviso: falta {low_quantile_name(number)}__spatial; se omite del ranking")
            continue
        out[f"{low_quantile_name(number)}_twin_mix"] = (
            f"{number} boosting q0,{q:02d}", "boosting", True, pred)
    return out


def ranking(train, oof):
    y = train[TARGET].to_numpy(float)
    rows = []
    for name, (label, family, main_path, pred) in ranked_predictions(train, oof).items():
        row = {"model": name, "label": label, "family": family, "main_path": main_path,
               **{k: v for k, v in error_stats(y, pred).items() if k != "n"}}
        for k in COST_RATIOS:
            row[f"cost_k{k}"] = asymmetric_cost(y, pred, k)
        rows.append(row)
    table = pd.DataFrame(rows)
    for k in COST_RATIOS:
        table[f"rank_k{k}"] = table[f"cost_k{k}"].rank(method="min").astype(int)
    return table.sort_values("cost_k1")


def plot_ranking(table):
    """Costo asimétrico por modelo para k = 1, 2 y 3, en el orden del MAE (k = 1): si el
    orden cambia, la barra más corta deja de ser la de arriba."""
    part = table.sort_values("cost_k1", ascending=False).reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=plot_style.FULL, sharey=True)
    y = np.arange(len(part))
    for ax, k in zip(axes, COST_RATIOS):
        cost = part[f"cost_k{k}"] / MILLION
        ax.barh(y, cost, color=[FAMILY_COLORS[f] for f in part["family"]], height=0.7)
        for i in (cost.idxmin(), part.index[part["model"] == BEST][0]):
            ax.text(cost[i] + cost.max() * 0.02, i, es_number(cost[i], 0), va="center",
                    fontsize=plot_style.LABEL_SIZE, color=plot_style.ANDES["black"])
        ax.set_title("k = 1 (MAE)" if k == 1 else f"k = {k}")
        ax.set_xlim(0, cost.max() * 1.15)
        ax.xaxis.set_major_formatter(plot_style.es_formatter())
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0, labelsize=plot_style.LABEL_SIZE)
    axes[1].set_xlabel("Costo medio por inmueble (millones de pesos)")
    axes[0].set_yticks(y, part["label"])
    axes[0].set_ylim(-0.6, len(part) - 0.4)
    return plot_style.save(fig, "asymmetric_ranking.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-importance", action="store_true",
                        help="no ajusta la receta de la 125 (usa el CSV guardado si existe)")
    args = parser.parse_args()
    plot_style.setup()
    train, _ = load_processed()
    oof = load_oof()

    table = anatomy(train, oof)
    table.to_csv(VIEWS_DIR / "best_errors_segments.csv", index=False)
    print(plot_anatomy(table))
    show = table[table["variable"] != "zona"].copy()
    for c in ["mae", "bias", "mean_over", "mean_under"]:
        show[c] = (show[c] / MILLION).round(1)
    print(show.round(3).to_string(index=False))

    path = VIEWS_DIR / "best_errors_importance.csv"
    if not args.skip_importance:
        importance(train).to_csv(path, index=False)
    if path.exists():
        imp = pd.read_csv(path)
        print(plot_importance(imp))
        print(imp.groupby("tipo").head(TOP_FEATURES).round(4).to_string(index=False))

    rank = ranking(train, oof)
    rank.to_csv(VIEWS_DIR / "asymmetric_ranking.csv", index=False)
    print(plot_ranking(rank))
    show = rank.drop(columns=["model", "family"]).copy()
    for c in [c for c in show if c.startswith(("cost", "mae", "bias", "mean_"))]:
        show[c] = (show[c] / MILLION).round(1)
    print(show.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
