"""Anatomía del error de los mejores boosting (ronda 8, parte 1). No reentrena nada: lee las
predicciones fuera de muestra de 04_train_models.py en el holdout nororiental ponderado como
el test y en la validación espacial, y descompone el error absoluto total por segmentos.

En el holdout ponderado cada fila pesa lo que pesa en utils.weighted_mae: las filas con
anuncios del mismo edificio comparten la fracción del test que los tiene (24 %) y las que no
los tienen comparten el resto; las demás filas de validación no cuentan.

Salidas:
  views/error_anatomy_segments.csv   MAE, sesgo y aporte al error por segmento
  views/error_anatomy_top50.csv      los 50 inmuebles con mayor error de la 83 por esquema
  views/error_anatomy_share.png      aporte al error frente a peso en filas, por segmento
  views/error_anatomy_deciles.png    sesgo por decil de precio observado y predicho
  views/error_anatomy_area.png       MAE por estado del área y rango de área

Uso:
  python scripts/23_error_anatomy.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import northeast_building_groups
from utils import (PROCESSED_DIR, RAW_DIR, TARGET, VIEWS_DIR, load_processed,
                   northeast_holdout, spatial_cv_splits, twin_groups)

MODELS = {"83": "83_boosting_ext_core_coords_by_type_level_q70_ridge_s10",
          "127": "127_boosting_pt_core_coords_by_type_level_q70_ridge_nocoords_s10_twin_mix"}
SCHEMES = {"weighted": "northeast", "spatial": "spatial"}
TOP_N = 50


def scheme_weights(train, test, y):
    """Peso de cada fila de train en cada esquema (cero si no se valida en él)."""
    val_idx, (with_building, without_building, share) = northeast_building_groups(train, test, y)
    weighted = np.zeros(len(train))
    weighted[val_idx[with_building]] = share / with_building.sum()
    weighted[val_idx[without_building]] += (1 - share) / without_building.sum()
    return {"weighted": weighted, "spatial": np.ones(len(train))}


def has_fit_twin(train, splits):
    """Si la fila tiene un gemelo (utils.twin_groups) entre las filas de ajuste de la
    partición que la valida."""
    groups = twin_groups(train)
    flag = np.zeros(len(train), dtype=bool)
    for fit_idx, val_idx in splits:
        flag[val_idx] = np.isin(groups[val_idx], groups[fit_idx])
    return flag


def area_status(train, raw):
    """Origen del área: la del anuncio, extraída del texto, recuperada del mismo punto o
    faltante."""
    raw_area = raw.set_index("property_id")[["surface_total", "surface_covered"]]
    reported = raw_area.notna().any(axis=1).reindex(train["property_id"]).to_numpy()
    status = np.where(train["surface"].notna(), "texto", "faltante")
    status = np.where(reported, "anuncio", status)
    status = np.where(train["surface_pt_imputed"] == 1, "mismo punto", status)
    return status


def segment_columns(train, raw, twin):
    """Segmentos de cada fila, salvo los deciles de precio."""
    year = raw.set_index("property_id")["year"].reindex(train["property_id"]).to_numpy()
    surface = train["surface_pt"].fillna(train["surface"])
    return pd.DataFrame({
        # Los deciles se llenan en main con las filas validadas de cada esquema.
        "decil_observado": np.nan, "decil_predicho": np.nan,
        "area": area_status(train, raw),
        "tipo": np.where(train["is_house"] == 1, "casa", "apartamento"),
        "estrato": (train["ext_stratum"].fillna(-1).astype(int).astype(str)
                    .replace("-1", "sin dato")),
        "rango_area": pd.cut(surface, [0, 60, 90, 130, 200, np.inf],
                             labels=["<60", "60-90", "90-130", "130-200", "200+"]
                             ).astype(str).replace("nan", "sin área"),
        "alcobas": train["bedrooms"].clip(upper=5).fillna(-1).astype(int).astype(str)
                   .replace({"5": "5+", "-1": "sin dato"}),
        "gemelo": np.where(twin, "con gemelo en ajuste", "sin gemelo"),
        "largo_descripcion": pd.qcut(train["txt_desc_length"], 4,
                                     labels=["Q1 corta", "Q2", "Q3", "Q4 larga"]).astype(str),
        "año": year.astype(str),
    }, index=train.index)


def decompose(err, weight, segments):
    """Aporte de cada segmento al error total ponderado, con su MAE, sesgo y peso."""
    rows = []
    total = (np.abs(err) * weight).sum()
    for column in segments:
        for value, idx in segments.groupby(column, observed=True).groups.items():
            w = weight[idx]
            if w.sum() == 0:
                continue
            e = err[idx]
            rows.append({"segmento": column, "valor": str(value),
                         "filas": int((w > 0).sum()), "peso": w.sum() / weight.sum(),
                         "aporte": (np.abs(e) * w).sum() / total,
                         "mae": np.average(np.abs(e), weights=w) / 1e6,
                         "sesgo": np.average(e, weights=w) / 1e6,
                         "sobrepredicho": np.average(e > 0, weights=w)})
    out = pd.DataFrame(rows)
    # Error recuperable: lo que sobra frente a la MAE global si el segmento tuviera esa MAE.
    global_mae = np.average(np.abs(err), weights=weight) / 1e6
    out["exceso"] = (out["mae"] - global_mae).clip(lower=0) * out["peso"]
    out["mae_global"] = global_mae
    return out


def top_errors(train, raw, err, weight, segments, scheme):
    """Los TOP_N inmuebles validados con mayor error absoluto."""
    rows = np.flatnonzero(weight > 0)
    rows = rows[np.argsort(-np.abs(err[rows]))][:TOP_N]
    text = raw.set_index("property_id")["description"].reindex(train["property_id"])
    out = segments.iloc[rows].copy()
    out.insert(0, "esquema", scheme)
    out.insert(1, "property_id", train["property_id"].iloc[rows].to_numpy())
    out["precio"] = train[TARGET].iloc[rows].to_numpy() / 1e6
    out["prediccion"] = (train[TARGET].iloc[rows].to_numpy() + err[rows]) / 1e6
    out["error"] = err[rows] / 1e6
    out["superficie"] = train["surface_pt"].fillna(train["surface"]).iloc[rows].to_numpy()
    out["dist_zona_t_km"] = train["dist_zona_t_m"].iloc[rows].to_numpy() / 1000
    out["descripcion"] = text.iloc[rows].str.slice(0, 160).to_numpy()
    return out


def plot_share(table):
    """Aporte al error frente a peso en filas de los segmentos del 83, por esquema."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, scheme in zip(axes, SCHEMES):
        part = table[(table["modelo"] == "83") & (table["esquema"] == scheme)
                     & table["segmento"].isin(["area", "decil_observado", "estrato", "tipo"])]
        ax.scatter(part["peso"], part["aporte"], s=25, color="#3a6ea5")
        for _, r in part.iterrows():
            if r["aporte"] > 0.08 or abs(r["aporte"] - r["peso"]) > 0.04:
                ax.annotate(f"{r['segmento']}={r['valor']}", (r["peso"], r["aporte"]),
                            fontsize=7, xytext=(3, 3), textcoords="offset points")
        lim = max(part["peso"].max(), part["aporte"].max()) * 1.1
        ax.plot([0, lim], [0, lim], color="grey", lw=0.8, ls="--")
        ax.set_title({"weighted": "Holdout ponderado", "spatial": "Validación espacial"}[scheme])
        ax.set_xlabel("Peso del segmento en las filas validadas")
    axes[0].set_ylabel("Aporte al error absoluto total")
    fig.suptitle("Fila 83: segmentos por encima de la diagonal concentran más error que filas")
    fig.tight_layout()
    fig.savefig(VIEWS_DIR / "error_anatomy_share.png", dpi=150)


def plot_deciles(table):
    """Sesgo medio por decil de precio observado y predicho, 83 y 127."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, segment in zip(axes, ["decil_observado", "decil_predicho"]):
        for (model, scheme), part in table[table["segmento"] == segment].groupby(
                ["modelo", "esquema"]):
            part = part.assign(valor=part["valor"].astype(float).astype(int)).sort_values("valor")
            ax.plot(part["valor"], part["sesgo"], marker="o",
                    ls="-" if scheme == "weighted" else "--",
                    label=f"{model}, {'ponderado' if scheme == 'weighted' else 'espacial'}")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel(segment.replace("_", " "))
        ax.set_xticks(range(1, 11))
    axes[0].set_ylabel("Sesgo medio (predicho − observado), M")
    axes[1].legend(fontsize=8)
    fig.suptitle("Sesgo por decil: el error de nivel crece en las colas")
    fig.tight_layout()
    fig.savefig(VIEWS_DIR / "error_anatomy_deciles.png", dpi=150)


def plot_area(table):
    """MAE por origen del área y por rango de área de la 83."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, segment in zip(axes, ["area", "rango_area"]):
        part = table[(table["modelo"] == "83") & (table["segmento"] == segment)]
        pivot = part.pivot(index="valor", columns="esquema", values="mae")
        pivot.plot.bar(ax=ax, color=["#c0504d", "#3a6ea5"], rot=0)
        ax.set_xlabel(segment.replace("_", " "))
        ax.set_ylabel("MAE, M")
    fig.suptitle("Fila 83: MAE por origen y rango del área")
    fig.tight_layout()
    fig.savefig(VIEWS_DIR / "error_anatomy_area.png", dpi=150)


def main():
    train, test = load_processed()
    raw = pd.read_csv(RAW_DIR / "train.csv")
    y = train[TARGET]
    oof = pd.read_csv(PROCESSED_DIR / "oof_predictions.csv")
    assert (oof["property_id"] == train["property_id"]).all()
    weights = scheme_weights(train, test, y)
    twins = {"weighted": has_fit_twin(train, [northeast_holdout(train, test)]),
             "spatial": has_fit_twin(train, spatial_cv_splits(train))}

    tables, tops = [], []
    for model, name in MODELS.items():
        for scheme, column in SCHEMES.items():
            pred = oof[f"{name}__{column}"].to_numpy()
            weight = np.where(np.isnan(pred), 0, weights[scheme])
            pred_filled = np.where(np.isnan(pred), np.nanmedian(pred), pred)
            segments = segment_columns(train, raw, twins[scheme])
            validated = weight > 0
            for col, values in [("decil_observado", y), ("decil_predicho", pred_filled)]:
                deciles = pd.qcut(values[validated], 10, labels=False) + 1
                segments.loc[validated, col] = deciles.astype(int)
            err = pred_filled - y.to_numpy()
            table = decompose(err, weight, segments)
            table.insert(0, "modelo", model)
            table.insert(1, "esquema", scheme)
            tables.append(table)
            if model == "83":
                tops.append(top_errors(train, raw, err, weight, segments, scheme))
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(VIEWS_DIR / "error_anatomy_segments.csv", index=False)
    pd.concat(tops).to_csv(VIEWS_DIR / "error_anatomy_top50.csv", index=False)
    plot_share(table)
    plot_deciles(table)
    plot_area(table)
    best = table[table["modelo"] == "83"].sort_values("exceso", ascending=False)
    print(best.groupby("esquema").head(8).round(3).to_string())


if __name__ == "__main__":
    main()
