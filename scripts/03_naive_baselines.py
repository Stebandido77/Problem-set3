"""Modelos ingenuos: el piso contra el que se compara todo lo demás.

Con MAE como métrica, la mejor constante es la mediana (no la media).

Uso: python scripts/03_naive_baselines.py
"""

import numpy as np
import pandas as pd

from utils import (ID_COL, TARGET, VIEWS_DIR, chapinero_holdout, error_summary,
                   load_processed, random_cv, save_submission, spatial_cv_splits)


def predict_naive(train, target_df, kind):
    """Predicción ingenua de 'kind' para target_df usando solo 'train'."""
    if kind == "median":
        return np.full(len(target_df), train[TARGET].median())
    if kind == "mean":
        return np.full(len(target_df), train[TARGET].mean())
    if kind == "median_by_type_bedrooms":
        # Mediana por tipo de inmueble y número de alcobas (tope en 5).
        cell = lambda d: d["is_house"] * 10 + d["bedrooms"].clip(upper=5)
        medians = train.groupby(cell(train))[TARGET].median()
        pred = cell(target_df).map(medians).to_numpy(dtype=float)
        return np.where(np.isnan(pred), train[TARGET].median(), pred)
    raise ValueError(kind)


def cv_mae(train, splits, kind):
    """Predicción fuera de muestra con las particiones dadas (solo filas de validación)."""
    oof = np.full(len(train), np.nan)
    for fit_idx, val_idx in splits:
        oof[val_idx] = predict_naive(train.iloc[fit_idx], train.iloc[val_idx], kind)
    rows = ~np.isnan(oof)
    return error_summary(train[TARGET][rows], oof[rows])


def main():
    train, test = load_processed()
    schemes = {"random": list(random_cv().split(train)),
               "spatial": spatial_cv_splits(train),
               "chapinero": [chapinero_holdout(train, test)]}
    rows = []
    for i, kind in enumerate(["median", "mean", "median_by_type_bedrooms"], start=1):
        row = {"model": f"naive_{kind}"}
        for scheme, splits in schemes.items():
            row[f"mae_{scheme}"] = cv_mae(train, splits, kind)["mae"]
        path = save_submission(test[ID_COL], predict_naive(train, test, kind),
                               f"00_naive_{i}_{kind}")
        rows.append(row)
        print(f"{row['model']:<32} MAE aleatoria {row['mae_random']/1e6:7.1f} M | "
              f"espacial {row['mae_spatial']/1e6:7.1f} M | "
              f"chapinero {row['mae_chapinero']/1e6:7.1f} M -> {path.name}")
    VIEWS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(VIEWS_DIR / "cv_results_naive.csv", index=False)


if __name__ == "__main__":
    main()
