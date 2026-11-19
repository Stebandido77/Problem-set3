"""Calibra el holdout contra los puntajes públicos de Kaggle.

Para cada esquema candidato de utils.northeast_holdout reentrena los cinco envíos que ya
tienen puntaje público (views/kaggle_scores.csv) y compara su MAE en el holdout con el de
Kaggle: si el orden coincide y qué tan cerca quedan los valores. Los envíos 10 y 11 se
subieron sin OSM, así que aquí se entrenan sin esos predictores.

Salida: views/holdout_calibration.csv (una fila por esquema y modelo).

Uso:
  python 08_calibrate_holdout.py --schemes old r250_b0 r250_cells_cap1100_apt
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from models import build_model, fit_model
from utils import (TARGET, VIEWS_DIR, chapinero_holdout, error_summary, feature_sets,
                   load_processed, northeast_holdout)

# Esquemas candidatos: argumentos de northeast_holdout ('old' es el holdout anterior).
SCHEMES = {
    "old": None,
    "r250_b0": dict(price_cap=None),
    "r250_b150": dict(buffer_m=150, price_cap=None),
    "r250_b0_apt": dict(apartments_only=True, price_cap=None),
    "r250_cells": dict(cell_m=300, price_cap=None),
    "r250_cells_apt": dict(cell_m=300, apartments_only=True, price_cap=None),
    "r250_b0_cap1300": dict(price_cap=1.3e9),
    "r250_b0_cap1100": dict(price_cap=1.1e9),
    "r250_b0_cap1200": dict(price_cap=1.2e9),
    "r250_b0_cap1000": dict(price_cap=1.0e9),
    "r250_b0_cap900": dict(price_cap=0.9e9),
    "r500_b0_cap1000": dict(radius_m=500, price_cap=1.0e9),
    "r250_b0_cap1100_apt": dict(price_cap=1.1e9, apartments_only=True),
    "r250_cells_cap1100_apt": dict(cell_m=300, price_cap=1.1e9, apartments_only=True),
    "r250_cells_cap1300_apt": dict(cell_m=300, price_cap=1.3e9, apartments_only=True),
    "r250_cells_cap900_apt": dict(cell_m=300, price_cap=0.9e9, apartments_only=True),
    "r500_cells_cap1100_apt": dict(radius_m=500, cell_m=300, price_cap=1.1e9,
                                   apartments_only=True),
}

# Componentes del ensamble 31 (con OSM): familia, conjunto de predictores, objetivo, por tipo.
ENSEMBLE_31 = [("boosting", "full_coords", "per_m2", True),
               ("elastic_net", "full", "log", False),
               ("random_forest", "full", "per_m2", True)]


def without_osm(features):
    """Predictores de los envíos subidos antes de descargar OSM."""
    return [f for f in features if not f.startswith("osm_")]


def fit_and_predict(family, features, target, by_type, train, fit_idx, val_idx):
    model = build_model(family, target=target, by_type=by_type)
    fit_model(model, train.iloc[fit_idx][features], train[TARGET].iloc[fit_idx])
    return model.predict(train.iloc[val_idx][features])


def scheme_predictions(train, fit_idx, val_idx):
    """Predicciones en la validación de los cinco envíos con puntaje público."""
    sets = feature_sets(train.columns)
    boosting = fit_and_predict("boosting", without_osm(sets["full_coords"]), "log", False,
                               train, fit_idx, val_idx)
    neural_net = fit_and_predict("neural_net", without_osm(sets["full"]), "log", False,
                                 train, fit_idx, val_idx)
    ensemble = np.mean([fit_and_predict(family, sets[features], target, by_type, train,
                                        fit_idx, val_idx)
                        for family, features, target, by_type in ENSEMBLE_31], axis=0)
    naive = np.full(len(val_idx), train[TARGET].iloc[fit_idx].median())
    return {"00_naive_1_median": naive,
            "11_neural_net_full": neural_net,
            "10_boosting_full_coords": boosting,
            "31_ensemble_mean_3": ensemble,
            "32_boosting_full_coords_x115": boosting * 1.15}


def decile_errors(y, pred, n=10):
    """MAE por decil del precio observado en la validación."""
    deciles = pd.qcut(y, n, labels=False, duplicates="drop")
    return pd.Series(np.abs(pred - y)).groupby(deciles).mean().to_numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schemes", nargs="+", default=list(SCHEMES), choices=list(SCHEMES))
    args = parser.parse_args()

    train, test = load_processed()
    kaggle = pd.read_csv(VIEWS_DIR / "kaggle_scores.csv").set_index("model")["public_mae"]
    out_path = VIEWS_DIR / "holdout_calibration.csv"
    for scheme in args.schemes:
        start = time.time()
        params = SCHEMES[scheme]
        fit_idx, val_idx = (chapinero_holdout(train, test) if params is None
                            else northeast_holdout(train, test, **params))
        y = train[TARGET].iloc[val_idx].to_numpy()
        rows = []
        for model, pred in scheme_predictions(train, fit_idx, val_idx).items():
            summary = error_summary(y, pred)
            row = {"scheme": scheme, "model": model, "n_fit": len(fit_idx),
                   "n_val": len(val_idx), "mae": summary["mae"], "bias": summary["bias"],
                   "share_over": summary["share_overpredicted"],
                   "mean_pred": pred.mean(), "max_pred": pred.max(),
                   "public_mae": kaggle.get(model, np.nan)}
            for i, err in enumerate(decile_errors(y, pred), start=1):
                row[f"mae_decile_{i}"] = err
            rows.append(row)
        table = pd.DataFrame(rows)
        rho = spearmanr(table["mae"], table["public_mae"]).statistic
        table["spearman"] = rho
        table["mean_abs_gap"] = (table["mae"] - table["public_mae"]).abs().mean()
        print(f"{scheme}: Spearman {rho:.2f} | desvío medio "
              f"{table['mean_abs_gap'].iloc[0] / 1e6:.1f} M | "
              + " | ".join(f"{m[:2]} {v / 1e6:.1f}" for m, v in zip(table["model"],
                                                                   table["mae"]))
              + f" | {(time.time() - start) / 60:.1f} min", flush=True)
        if out_path.exists():
            previous = pd.read_csv(out_path)
            table = pd.concat([previous[previous["scheme"] != scheme], table])
        table.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
