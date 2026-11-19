"""Ensamble de especificaciones ya entrenadas: promedio o mediana de sus predicciones.

No reentrena nada: combina las predicciones fuera de muestra de cada esquema de validación
(stores/processed/oof_predictions.csv) y las del test (stores/processed/test_predictions.csv)
que dejó 04_train_models.py. Como cada componente se evaluó con las mismas particiones, el
error del ensamble en cada esquema también es fuera de muestra.

Uso:
  python scripts/06_ensemble.py --number 31 --how mean \
      --models 17_boosting_full_coords_by_type_per_m2 04_elastic_net_full \
               18_random_forest_full_by_type_per_m2
"""

import argparse

import numpy as np
import pandas as pd

from utils import (ID_COL, PROCESSED_DIR, TARGET, VIEWS_DIR, error_summary,
                   save_submission)

SCHEMES = ["random", "spatial", "chapinero", "northeast"]


def combine(frame, how):
    """Promedio o mediana fila a fila."""
    return frame.median(axis=1) if how == "median" else frame.mean(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--how", choices=["mean", "median"], default="median")
    parser.add_argument("--number", type=int, required=True, help="número del envío")
    args = parser.parse_args()

    oof = pd.read_csv(PROCESSED_DIR / "oof_predictions.csv")
    test_pred = pd.read_csv(PROCESSED_DIR / "test_predictions.csv")
    results_path = VIEWS_DIR / "cv_results.csv"
    results = pd.read_csv(results_path)

    name = f"{args.number:02d}_ensemble_{args.how}_{len(args.models)}"
    row = {"model": name, "family": "ensemble", "features": "+".join(args.models),
           "n_features": np.nan, "tuned": False}
    for scheme in SCHEMES:
        columns = [f"{m}__{scheme}" for m in args.models]
        if not set(columns) <= set(oof.columns):
            continue  # algún componente no se evaluó con este esquema
        pred = combine(oof[columns], args.how)
        rows = pred.notna()
        summary = error_summary(oof.loc[rows, TARGET], pred[rows])
        row[f"mae_{scheme}"] = summary["mae"]
        row[f"bias_{scheme}"] = summary["bias"]
        row[f"share_over_{scheme}"] = summary["share_overpredicted"]
    final = combine(test_pred[args.models], args.how)
    row["submission"] = save_submission(test_pred[ID_COL], final, name).name
    row["test_mean"], row["test_p95"], row["test_max"] = (
        final.mean(), final.quantile(0.95), final.max())

    results = pd.concat([results[results["model"] != name], pd.DataFrame([row])])
    results.sort_values("model").to_csv(results_path, index=False)
    print(f"{name}: " + " | ".join(f"{s} {row[f'mae_{s}'] / 1e6:.1f} M" for s in SCHEMES
                                   if f"mae_{s}" in row) + f" -> {row['submission']}")


if __name__ == "__main__":
    main()
