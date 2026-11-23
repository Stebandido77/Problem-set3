"""Verifica el predictor de BETO (stores/processed/beto_predictions.csv) antes de usarlo.

1. Filas, ids y faltantes contra train y test.
2. BETO solo frente al ridge de texto solo (models.RidgeText, el txt_ridge de la 127), en
   las mismas filas y con las mismas particiones de BETO (las espaciales con los anuncios de
   igual descripción juntos): MAE total y por partición, y correlación entre ambos.
3. Error de cada uno por decil de precio.

Salidas: views/beto_check.csv (MAE por partición y total) y views/beto_deciles.csv.

Uso:
  python scripts/38_beto_check.py
"""

import numpy as np
import pandas as pd

from models import RidgeText
from utils import BETO_PATH, ID_COL, TARGET, VIEWS_DIR, load_processed


def check_rows(beto, train, test):
    """Cuenta filas, ids repetidos o ajenos y faltantes; falla si algo no cuadra."""
    by_split = beto.groupby("split")[ID_COL]
    checks = {
        "rows_train": len(by_split.get_group("train")),
        "rows_test": len(by_split.get_group("test")),
        "duplicated_ids": int(beto[ID_COL].duplicated().sum()),
        "missing_train_ids": int((~train[ID_COL].isin(beto[ID_COL])).sum()),
        "missing_test_ids": int((~test[ID_COL].isin(beto[ID_COL])).sum()),
        "missing_values": int(beto[["beto_log_price", "beto_price"]].isna().sum().sum()),
        "test_fold_is_minus_one": bool((beto.loc[beto["split"] == "test", "fold"] == -1).all()),
    }
    print("Chequeo de filas:", checks)
    assert checks["rows_train"] == len(train) and checks["rows_test"] == len(test)
    assert checks["duplicated_ids"] == checks["missing_train_ids"] == 0
    assert checks["missing_test_ids"] == checks["missing_values"] == 0
    return checks


def ridge_oof(train, folds):
    """Ridge del texto fuera de muestra con las particiones de BETO (log del precio)."""
    splits = [(np.flatnonzero(folds != k), np.flatnonzero(folds == k))
              for k in np.unique(folds)]
    log_y = np.log(train[TARGET].to_numpy(float))
    return RidgeText().fit_features(train, log_y, splits)["txt_ridge"]


def mae_table(y, predictions, folds):
    """MAE en millones de cada predictor, por partición y total."""
    rows = []
    for fold in [*np.unique(folds), "total"]:
        mask = np.ones(len(y), bool) if fold == "total" else folds == fold
        rows.append({"fold": fold, "n": int(mask.sum()),
                     **{f"mae_{name}": np.abs(pred[mask] - y[mask]).mean() / 1e6
                        for name, pred in predictions.items()}})
    return pd.DataFrame(rows)


def decile_table(y, predictions):
    """MAE y sesgo en millones de cada predictor por decil de precio."""
    deciles = pd.qcut(y, 10, labels=False) + 1
    rows = []
    for decile in range(1, 11):
        mask = deciles == decile
        row = {"decile": decile, "price_min": y[mask].min() / 1e6, "price_max": y[mask].max() / 1e6}
        for name, pred in predictions.items():
            row[f"mae_{name}"] = np.abs(pred[mask] - y[mask]).mean() / 1e6
            row[f"bias_{name}"] = (pred[mask] - y[mask]).mean() / 1e6
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    train, test = load_processed()
    beto = pd.read_csv(BETO_PATH)
    check_rows(beto, train, test)
    beto_train = beto[beto["split"] == "train"].set_index(ID_COL).loc[train[ID_COL]]
    folds = beto_train["fold"].to_numpy(int)
    y = train[TARGET].to_numpy(float)

    log_ridge = ridge_oof(train, folds)
    log_beto = beto_train["beto_log_price"].to_numpy(float)
    predictions = {"beto": np.exp(log_beto), "ridge": np.exp(log_ridge)}

    table = mae_table(y, predictions, folds)
    table["corr_log"] = [np.corrcoef(log_beto[m], log_ridge[m])[0, 1] for m in
                         [folds == k for k in np.unique(folds)] + [np.ones(len(y), bool)]]
    table.to_csv(VIEWS_DIR / "beto_check.csv", index=False)
    deciles = decile_table(y, predictions)
    deciles.to_csv(VIEWS_DIR / "beto_deciles.csv", index=False)

    test_beto = beto[beto["split"] == "test"]["beto_price"]
    print(table.round(3).to_string(index=False))
    print(deciles.round(1).to_string(index=False))
    print(f"Correlación en niveles: {np.corrcoef(*predictions.values())[0, 1]:.3f}")
    print(f"BETO en test: media {test_beto.mean() / 1e6:.0f} M, mediana "
          f"{test_beto.median() / 1e6:.0f} M, máx. {test_beto.max() / 1e6:.0f} M")


if __name__ == "__main__":
    main()
