"""Verificaciones para las presentaciones finales. No entrena nada: lee las predicciones
guardadas (fuera de muestra y del test) y las bases con predictores.

1. Por qué aporta la red a la mezcla: MAE de la mezcla 203 frente al boosting 198 solo (y de
   la 152 frente al boosting 127) por decil del precio pedido, en la validación espacial, y
   la correlación de errores entre los dos miembros de cada mezcla.
2. Cuántas predicciones de la 203 en el test pasan del tope de precios de entrenamiento
   (1.650 M).
3. ¿En Chapinero "todo queda cerca"? Dispersión de las distancias de OpenStreetMap en los
   datos de entrenamiento y en el test.
4. Anuncios sin área: en el anuncio, después de leer el texto y después de recuperar el área
   de anuncios del mismo punto, en entrenamiento y en el test.

Salidas en views/: blend_deciles.csv, blend_members.csv, test_over_cap.csv,
osm_dispersion.csv y area_missing.csv.

Uso (desde scripts/):
  python 42_deck_checks.py
"""

import numpy as np
import pandas as pd

from utils import PROCESSED_DIR, TARGET, VIEWS_DIR

CAP = 1.65e9
MILLION = 1e6
# Mezcla -> (boosting, red), los nombres de las columnas en las predicciones guardadas.
BLENDS = {
    "203_ensemble_w70_30_2": (
        "198_boosting_pt_core_coords_by_type_level_q70_ridge_beto_nocoords_s10_twin_mix",
        "196_quantile_net_ext_core_coords_by_type_q70_ridge_beto_nocoords"),
    "152_ensemble_w70_30_2": (
        "127_boosting_pt_core_coords_by_type_level_q70_ridge_nocoords_s10_twin_mix",
        "148_quantile_net_ext_core_coords_by_type_q70_ridge_nocoords"),
}
AREA_COLUMNS = {"anuncio": "surface", "anuncio y texto": "surface_ext",
                "anuncio, texto y mismo punto": "surface_pt"}


def number(name):
    return name.split("_")[0]


def blend_deciles(train, oof):
    """MAE espacial por decil del precio pedido: mezcla, boosting solo y red sola."""
    y = train[TARGET].to_numpy(float)
    decile = pd.qcut(y, 10, labels=False) + 1
    rows = []
    for blend, (boost, net) in BLENDS.items():
        preds = {k: oof[f"{name}__spatial"].to_numpy(float)
                 for k, name in [("mezcla", blend), ("boosting", boost), ("red", net)]}
        for d in range(1, 11):
            rows_d = decile == d
            row = {"mezcla": number(blend), "decil": d,
                   "precio_min": y[rows_d].min() / MILLION,
                   "precio_max": y[rows_d].max() / MILLION, "n": int(rows_d.sum())}
            for k, pred in preds.items():
                err = pred[rows_d] - y[rows_d]
                row[f"mae_{k}"] = np.abs(err).mean() / MILLION
                row[f"sesgo_{k}"] = err.mean() / MILLION
            row["ganancia_mezcla"] = row["mae_boosting"] - row["mae_mezcla"]
            rows.append(row)
    return pd.DataFrame(rows)


def blend_members(train, oof):
    """Correlación de errores entre los miembros y MAE total, por esquema."""
    y = train[TARGET].to_numpy(float)
    rows = []
    for blend, (boost, net) in BLENDS.items():
        for scheme in ("spatial", "northeast"):
            cols = [f"{name}__{scheme}" for name in (blend, boost, net)]
            if not all(c in oof.columns for c in cols):
                continue
            keep = oof[cols].notna().all(axis=1).to_numpy()
            mix, b, n = (oof[c].to_numpy(float)[keep] for c in cols)
            yk = y[keep]
            rows.append({"mezcla": number(blend), "esquema": scheme, "n": int(keep.sum()),
                         "corr_error": np.corrcoef(b - yk, n - yk)[0, 1],
                         "mae_mezcla": np.abs(mix - yk).mean() / MILLION,
                         "mae_boosting": np.abs(b - yk).mean() / MILLION,
                         "mae_red": np.abs(n - yk).mean() / MILLION})
    return pd.DataFrame(rows)


def over_cap(test_pred):
    """Predicciones del test por encima del tope de entrenamiento, por modelo."""
    rows = []
    for blend, members in BLENDS.items():
        for name in (blend, *members):
            pred = test_pred[name].to_numpy(float)
            rows.append({"modelo": name, "n_test": len(pred), "n_sobre_tope": int((pred > CAP).sum()),
                         "max_millones": pred.max() / MILLION})
    return pd.DataFrame(rows)


def osm_dispersion(train, test):
    """Mediana, rango intercuartílico y desviación estándar de cada distancia de OSM."""
    rows = []
    for column in [c for c in train.columns if c.startswith("osm_dist_")]:
        row = {"variable": column}
        for label, frame in [("entrenamiento", train), ("test", test)]:
            values = frame[column].dropna()
            q1, q3 = values.quantile([0.25, 0.75])
            row |= {f"mediana_{label}": values.median(), f"iqr_{label}": q3 - q1,
                    f"sd_{label}": values.std()}
        row["razon_iqr_test_entrenamiento"] = row["iqr_test"] / row["iqr_entrenamiento"]
        rows.append(row)
    return pd.DataFrame(rows)


def area_missing(train, test):
    rows = []
    for label, frame in [("entrenamiento", train), ("test", test)]:
        for stage, column in AREA_COLUMNS.items():
            missing = frame[column].isna()
            rows.append({"base": label, "area_desde": stage, "n": len(frame),
                         "n_sin_area": int(missing.sum()), "porcentaje": missing.mean()})
    return pd.DataFrame(rows)


def main():
    train = pd.read_csv(PROCESSED_DIR / "train_features.csv")
    test = pd.read_csv(PROCESSED_DIR / "test_features.csv")
    names = [n for blend, members in BLENDS.items() for n in (blend, *members)]
    oof_cols = [f"{n}__{s}" for n in names for s in ("spatial", "northeast")]
    oof = pd.read_csv(PROCESSED_DIR / "oof_predictions.csv",
                      usecols=lambda c: c in oof_cols)
    test_pred = pd.read_csv(PROCESSED_DIR / "test_predictions.csv",
                            usecols=lambda c: c in names)
    outputs = {"blend_deciles": blend_deciles(train, oof),
               "blend_members": blend_members(train, oof),
               "test_over_cap": over_cap(test_pred),
               "osm_dispersion": osm_dispersion(train, test),
               "area_missing": area_missing(train, test)}
    pd.set_option("display.width", 220)
    for name, table in outputs.items():
        table.to_csv(VIEWS_DIR / f"{name}.csv", index=False)
        print(f"\n{name}\n{table.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
