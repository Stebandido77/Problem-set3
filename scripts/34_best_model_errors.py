"""Errores del mejor modelo y pérdida asimétrica (etapa 4, piezas 1 a 3). Lee las
predicciones fuera de muestra de la validación espacial (la que imita el salto del resto de
Bogotá a Chapinero) y no las modifica.

1. Anatomía del error de la mezcla 152 (el mejor puntaje en Kaggle): sesgo, porcentaje
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

import plot_style
from plot_style import FAMILY_COLORS, FAMILY_LABELS, es_number, family_of, millions, plt
from utils import (PROCESSED_DIR, RAW_DIR, TARGET, VIEWS_DIR, load_processed,
                   northeast_holdout, spatial_cv_splits, spatial_groups)

MILLION = 1e6
BEST = "152_ensemble_w70_30_2"
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
    BEST: ("152 mezcla (Kaggle)", True),
    "186_ensemble_w70_30_2": ("186 mezcla*", False),
}
COST_RATIOS = (1, 2, 3)
PRICE_BINS = [0, 400, 500, 700, 900, 1200, np.inf]
PRICE_LABELS = ["<400", "400-500", "500-700", "700-900", "900-1.200", "1.200+"]
SEGMENT_TITLES = {"precio_observado": "Precio pedido (M)", "precio_predicho": "Precio predicho (M)",
                  "area": "Área", "estrato": "Estrato", "tipo": "Tipo", "zona": "Zona espacial",
                  "año": "Año del anuncio"}
OVER_COLOR, UNDER_COLOR = "#B2182B", "#2166AC"
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
    """Anatomía del error de la 152 en la espacial y en el holdout nororiental."""
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
    total = spatial[spatial["variable"] == "total"].iloc[0]
    panels = ["precio_observado", "precio_predicho", "area", "tipo", "estrato", "zona"]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharey=True)
    for ax, variable in zip(axes.flat, panels):
        part = spatial[spatial["variable"] == variable].set_index("segment")
        part = part.loc[segment_order(variable, part.index)]
        x = np.arange(len(part))
        ax.bar(x, part["mean_over"], color=OVER_COLOR, width=0.7, label="Sobrepredicción media")
        ax.bar(x, -part["mean_under"], color=UNDER_COLOR, width=0.7,
               label="Subpredicción media")
        ax.plot(x, part["bias"], "o", color="black", ms=5, label="Sesgo")
        for xi, share in zip(x, part["share_over"]):
            ax.text(xi, 0.97, f"{share * 100:.0f} %", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8.5, color=OVER_COLOR)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(x, part.index, rotation=30 if len(x) > 5 else 0, ha="right"
                      if len(x) > 5 else "center", fontsize=9)
        ax.set_title(SEGMENT_TITLES[variable])
        millions(ax)
    top = spatial[spatial["variable"] != "total"]["mean_over"].max()
    axes[0, 0].set_ylim(top=top * 1.25)
    for ax in axes[:, 0]:
        ax.set_ylabel("Error medio (millones de pesos)")
    axes[0, 0].legend(loc="lower left", fontsize=8.5)
    predicted = spatial[spatial["variable"] == "precio_predicho"].set_index("segment")
    low, high = predicted.loc[PRICE_LABELS[0], "mean_over"], predicted.loc[PRICE_LABELS[-1],
                                                                          "mean_over"]
    fig.suptitle(
        f"La mezcla 152 sobrepredice el {total['share_over'] * 100:.0f} % de los inmuebles "
        f"(sesgo de +{es_number(total['bias'] / MILLION, 0)} M), y la sobrepredicción media "
        f"pasa de {es_number(low / MILLION, 0)} a {es_number(high / MILLION, 0)} M cuando el "
        "precio predicho sube", fontsize=12.5)
    fig.text(0.5, -0.01, "Validación espacial fuera de muestra (10 zonas de Bogotá, 5 "
             "particiones). Arriba de cada barra: % de inmuebles sobrepredichos.",
             ha="center", fontsize=9)
    fig.tight_layout()
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
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, kind in zip(axes, ["apartamento", "casa"]):
        part = table[table["tipo"] == kind].head(TOP_FEATURES).iloc[::-1]
        labels = [FEATURE_LABELS.get(f, f) for f in part["feature"]]
        ax.barh(labels, part["gain_share"] * 100, color=FAMILY_COLORS["boosting"])
        for yi, value in enumerate(part["gain_share"] * 100):
            ax.text(value + 0.4, yi, es_number(value, 1), va="center", fontsize=8.5)
        ax.set_title("Apartamentos" if kind == "apartamento" else "Casas")
        ax.set_xlabel("Ganancia (% del total)")
        ax.grid(axis="y", visible=False)
    top = table[table["tipo"] == "apartamento"].iloc[0]
    fig.suptitle(f"{FEATURE_LABELS.get(top['feature'], top['feature'])} concentra el "
                 f"{es_number(top['gain_share'] * 100, 0)} % de la ganancia del boosting en "
                 "apartamentos", fontsize=12.5)
    fig.text(0.5, -0.02, "Receta de la 125 (base de la 127) ajustada con todo el train y una "
             "semilla; ganancia por división de LightGBM.", ha="center", fontsize=9)
    fig.tight_layout()
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
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8))
    for ax, k in zip(axes, COST_RATIOS):
        part = table.sort_values(f"cost_k{k}", ascending=False)
        colors = [FAMILY_COLORS[f] for f in part["family"]]
        ax.barh(part["label"], part[f"cost_k{k}"], color=colors)
        for yi, value in enumerate(part[f"cost_k{k}"]):
            ax.text(value, yi, " " + es_number(value / MILLION, 0), va="center", fontsize=8.5)
        title = "k = 1 (MAE)" if k == 1 else f"k = {k}: sobrepredecir cuesta {k} veces"
        ax.set_title(title)
        ax.set_xlabel("Costo medio (millones de pesos)")
        ax.set_xlim(0, part[f"cost_k{k}"].max() * 1.18)
        millions(ax, "x")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=9)
    main = table[table["main_path"]]
    best = {k: main.loc[main[f"cost_k{k}"].idxmin(), "label"] for k in COST_RATIOS}
    families = table["family"].drop_duplicates()
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f]) for f in families]
    fig.legend(handles, [FAMILY_LABELS[f] for f in families], loc="lower center",
               ncol=len(handles), fontsize=9, bbox_to_anchor=(0.5, -0.06))
    same = (main["rank_k1"] == main["rank_k3"]).all()
    title = (f"El orden no cambia al castigar la sobrepredicción: {best[1]} gana con k = 1, 2 "
             "y 3 (ruta principal)" if same else
             f"Con el MAE gana {best[1]}; si sobrepredecir cuesta 3 veces, gana {best[3]} "
             "(ruta principal)")
    fig.suptitle(title, fontsize=12.5)
    fig.text(0.5, -0.1, "Validación espacial fuera de muestra. Costo = k × sobrepredicción "
             "+ subpredicción, promedio por inmueble. * La 186 no está en la ruta principal.",
             ha="center", fontsize=9)
    fig.tight_layout()
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
