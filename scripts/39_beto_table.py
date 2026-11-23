"""Tabla de la ronda de BETO (filas 193 en adelante) contra la 127, la 148 y la 152.

No entrena nada. Para cada modelo lee de views/cv_results.csv y de
stores/processed/{oof,test}_predictions.csv:
- MAE espacial y del holdout nororiental ponderado como el test;
- media, mediana, p95 y máximo de las predicciones en test, y cuántas pasan de 1.650 M;
- lo mismo con las predicciones recortadas al rango del train (300 a 1.650 M), con el número
  de predicciones del test que cambian. El recorte usa solo el rango conocido del train.
Además, la correlación en test entre los miembros de cada mezcla.

Salidas: views/beto_table.csv y views/beto_blend_corr.csv.

Uso:
  python scripts/39_beto_table.py
"""

import numpy as np
import pandas as pd

from models import northeast_building_groups
from utils import PROCESSED_DIR, TARGET, VIEWS_DIR, load_processed, weighted_mae

M = 1e6
PRICE_RANGE = (300e6, 1650e6)
REFERENCES = ["127_", "148_", "152_"]
FIRST_NEW = 193


def selected_models(results):
    """Referencias y filas nuevas de la ronda, en orden de número."""
    number = results["model"].str.split("_").str[0].astype(int)
    keep = results["model"].str.startswith(tuple(REFERENCES)) | (number >= FIRST_NEW)
    return results.loc[keep].assign(number=number[keep]).sort_values("number")


def criteria(name, oof, y, val_idx, groups, clip):
    """MAE espacial y ponderado de un modelo, con o sin recorte."""
    spatial = oof[f"{name}__spatial"].to_numpy(float)
    northeast = oof[f"{name}__northeast"].to_numpy(float)[val_idx]
    if clip:
        spatial, northeast = np.clip(spatial, *PRICE_RANGE), np.clip(northeast, *PRICE_RANGE)
    rows = ~np.isnan(spatial)
    return (np.abs(spatial[rows] - y[rows]).mean() / M,
            weighted_mae(y[val_idx], northeast, groups) / M)


def test_stats(pred):
    return {"test_mean": pred.mean() / M, "test_median": np.median(pred) / M,
            "test_p95": np.quantile(pred, 0.95) / M, "test_max": pred.max() / M,
            "test_over_cap": int((pred > PRICE_RANGE[1]).sum())}


def model_row(name, oof, test_pred, y, val_idx, groups):
    pred = test_pred[name].to_numpy(float)
    clipped = np.clip(pred, *PRICE_RANGE)
    spatial, weighted = criteria(name, oof, y, val_idx, groups, clip=False)
    spatial_clip, weighted_clip = criteria(name, oof, y, val_idx, groups, clip=True)
    return {"model": name, "spatial": spatial, "weighted": weighted, **test_stats(pred),
            "spatial_clip": spatial_clip, "weighted_clip": weighted_clip,
            "test_mean_clip": clipped.mean() / M, "test_max_clip": clipped.max() / M,
            "n_clipped_low": int((pred < PRICE_RANGE[0]).sum()),
            "n_clipped_high": int((pred > PRICE_RANGE[1]).sum())}


def blend_correlations(results, test_pred):
    """Correlación en test entre los miembros de cada mezcla de la ronda."""
    rows = []
    blends = results[(results["family"] == "ensemble") & (results["number"] >= FIRST_NEW)]
    for name, members in zip(blends["model"], blends["features"].str.split("+")):
        corr = test_pred[members].corr().to_numpy()
        rows += [{"blend": name, "member_a": members[i], "member_b": members[j],
                  "corr_test": corr[i, j]}
                 for i in range(len(members)) for j in range(i + 1, len(members))]
    return pd.DataFrame(rows)


def main():
    train, test = load_processed()
    y = train[TARGET].to_numpy(float)
    val_idx, groups = northeast_building_groups(train, test, train[TARGET])
    results = selected_models(pd.read_csv(VIEWS_DIR / "cv_results.csv"))
    oof = pd.read_csv(PROCESSED_DIR / "oof_predictions.csv")
    test_pred = pd.read_csv(PROCESSED_DIR / "test_predictions.csv")

    available = [m for m in results["model"]
                 if m in test_pred and f"{m}__spatial" in oof and f"{m}__northeast" in oof]
    table = pd.DataFrame([model_row(m, oof, test_pred, y, val_idx, groups) for m in available])
    reference = table.set_index("model").filter(regex="^152_", axis=0).iloc[0]
    table["d_spatial_152"] = table["spatial"] - reference["spatial"]
    table["d_weighted_152"] = table["weighted"] - reference["weighted"]
    table.to_csv(VIEWS_DIR / "beto_table.csv", index=False)
    correlations = blend_correlations(results, test_pred)
    correlations.to_csv(VIEWS_DIR / "beto_blend_corr.csv", index=False)

    pd.set_option("display.width", 250)
    print(table.round(1).to_string(index=False))
    print(correlations.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
