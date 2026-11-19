"""Entrena las seis familias de algoritmos, las evalúa con validación cruzada aleatoria
y espacial, y escribe un envío de Kaggle por cada especificación.

Salidas:
  stores/submissions/<nn>_<familia>_<predictores>.csv   envíos para Kaggle
  stores/processed/oof_predictions.csv                  predicciones fuera de muestra
  views/cv_results.csv                                  tabla comparativa

Uso:
  python scripts/04_train_models.py                       # todas las especificaciones
  python scripts/04_train_models.py --families boosting   # solo una familia
"""

import argparse
import time

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict

from models import FAMILIES, build_model, load_best_params
from utils import (ID_COL, PROCESSED_DIR, TARGET, VIEWS_DIR, error_summary, feature_sets,
                   load_processed, random_cv, save_submission, spatial_cv_splits)

# (familia, conjunto de predictores). El orden cuenta la historia: de simple a complejo.
SPECS = [
    ("linear", "base"),
    ("linear", "base_text"),
    ("linear", "full"),
    ("elastic_net", "full"),
    ("cart", "full"),
    ("random_forest", "full"),
    ("random_forest", "full_coords"),
    ("boosting", "base_text"),
    ("boosting", "full"),
    ("boosting", "full_coords"),
    ("neural_net", "full"),
]


def evaluate(model, X, y, splits):
    """Predicciones fuera de muestra y resumen de error para un esquema de validación."""
    oof = cross_val_predict(clone(model), X, y, cv=splits)
    return oof, error_summary(y, oof)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES)
    args = parser.parse_args()

    train, test = load_processed()
    y = train[TARGET]
    sets = feature_sets(train.columns)
    best_params = load_best_params()
    schemes = {"random": list(random_cv().split(train)),
               "spatial": spatial_cv_splits(train)}

    rows, oof_table = [], train[[ID_COL, TARGET]].copy()
    for number, (family, set_name) in enumerate(SPECS, start=1):
        if family not in args.families:
            continue
        start = time.time()
        name = f"{number:02d}_{family}_{set_name}"
        features = sets[set_name]
        params = best_params.get(f"{family}_{set_name}", {})
        model = build_model(family, params)

        row = {"model": name, "family": family, "features": set_name,
               "n_features": len(features), "tuned": bool(params)}
        for scheme, splits in schemes.items():
            oof, summary = evaluate(model, train[features], y, splits)
            oof_table[f"{name}__{scheme}"] = oof
            row[f"mae_{scheme}_cv"] = summary["mae"]
            row[f"bias_{scheme}_cv"] = summary["bias"]
            row[f"share_over_{scheme}_cv"] = summary["share_overpredicted"]

        # Ajuste final con todo el entrenamiento y envío a Kaggle.
        model.fit(train[features], y)
        save_submission(test[ID_COL], model.predict(test[features]), name)
        row["minutes"] = (time.time() - start) / 60
        rows.append(row)
        print(f"{name:<34} MAE aleatoria {row['mae_random_cv']/1e6:6.1f} M | "
              f"espacial {row['mae_spatial_cv']/1e6:6.1f} M | "
              f"sobrepredice {row['share_over_spatial_cv']:.0%} | {row['minutes']:.1f} min",
              flush=True)

    # Se actualizan solo las filas corridas, conservando resultados previos de otras familias.
    VIEWS_DIR.mkdir(exist_ok=True)
    results_path = VIEWS_DIR / "cv_results.csv"
    results = pd.DataFrame(rows)
    if results_path.exists():
        previous = pd.read_csv(results_path)
        results = pd.concat([previous[~previous["model"].isin(results["model"])], results])
    results.sort_values("model").to_csv(results_path, index=False)

    oof_path = PROCESSED_DIR / "oof_predictions.csv"
    if oof_path.exists():
        previous = pd.read_csv(oof_path)
        keep = [c for c in previous.columns if c not in oof_table.columns]
        oof_table = pd.concat([oof_table, previous[keep]], axis=1)
    oof_table.to_csv(oof_path, index=False)


if __name__ == "__main__":
    main()
