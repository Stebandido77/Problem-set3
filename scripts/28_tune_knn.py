"""Sintoniza el kNN cuantílico de la ronda 10 (models.QuantileKNN) con validación espacial.

Estrategia: rejilla completa sobre k y los pesos del texto y de la ubicación (el de las
características vale 1). En cada partición espacial y para cada tipo de inmueble, los
bloques (TF-IDF reducido, ubicación y características) se ajustan una sola vez con las
filas de ajuste; para cada par de pesos se buscan los vecinos una vez con el k más grande y
el percentil 70 se calcula para cada k con los primeros vecinos. Nada de la validación
entra a los bloques.

Salidas: views/tuning_knn_q70.csv (la rejilla con su MAE espacial) y la clave 'knn_q70' en
stores/processed/best_params.json, que usa la especificación 185 de 04_train_models.py.

Uso:
  python 28_tune_knn.py
"""

import itertools
import json

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from models import (RECIPE_QUANTILE, KnnBlocks, knn_matrix, load_best_params,
                    neighbor_quantile)
from utils import BEST_PARAMS_PATH, TARGET, VIEWS_DIR, load_processed, spatial_cv_splits

K_GRID = [5, 10, 20, 40]
TEXT_WEIGHTS = [0.0, 0.5, 1.0, 2.0]
LOCATION_WEIGHTS = [0.1, 0.25, 0.5, 1.0, 2.0]


def fold_predictions(X_fit, y_fit, X_val):
    """Predicción de las filas de validación para cada (pesos, k) de la rejilla."""
    blocks = KnnBlocks().fit(X_fit)
    fit_blocks, val_blocks = blocks.transform(X_fit), blocks.transform(X_val)
    out = {}
    for w_text, w_location in itertools.product(TEXT_WEIGHTS, LOCATION_WEIGHTS):
        index = NearestNeighbors(n_neighbors=max(K_GRID)).fit(
            knn_matrix(fit_blocks, w_text, w_location))
        _, idx = index.kneighbors(knn_matrix(val_blocks, w_text, w_location))
        prices = y_fit[idx]
        for k in K_GRID:
            out[(k, w_text, w_location)] = neighbor_quantile(prices[:, :k], RECIPE_QUANTILE)
    return out


def main():
    train, _ = load_processed()
    y = train[TARGET].to_numpy(float)
    preds = {}
    for fold, (fit_idx, val_idx) in enumerate(spatial_cv_splits(train)):
        for kind in (0, 1):
            fit_rows = fit_idx[train["is_house"].to_numpy()[fit_idx] == kind]
            val_rows = val_idx[train["is_house"].to_numpy()[val_idx] == kind]
            if len(val_rows) == 0:
                continue
            for key, pred in fold_predictions(train.iloc[fit_rows], y[fit_rows],
                                              train.iloc[val_rows]).items():
                preds.setdefault(key, np.full(len(y), np.nan))[val_rows] = pred
        print(f"partición {fold + 1} lista", flush=True)

    trace = pd.DataFrame([{"k": k, "w_text": w_text, "w_location": w_location,
                           "mae_spatial": np.nanmean(np.abs(pred - y))}
                          for (k, w_text, w_location), pred in preds.items()])
    trace = trace.sort_values("mae_spatial")
    trace.to_csv(VIEWS_DIR / "tuning_knn_q70.csv", index=False)
    best = trace.iloc[0]
    best_params = load_best_params()
    best_params["knn_q70"] = {"k": int(best["k"]), "w_text": float(best["w_text"]),
                              "w_location": float(best["w_location"])}
    BEST_PARAMS_PATH.write_text(json.dumps(best_params, indent=2))
    print(trace.head(10).assign(mae_spatial=lambda d: d["mae_spatial"] / 1e6).to_string())


if __name__ == "__main__":
    main()
