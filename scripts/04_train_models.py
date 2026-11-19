"""Entrena las seis familias de algoritmos, las evalúa con validación cruzada aleatoria,
validación cruzada espacial y el holdout tipo Chapinero, y escribe un envío de Kaggle por
cada especificación.

Salidas:
  stores/submissions/<nn>_<familia>_<predictores>[_variante].csv   envíos para Kaggle
  stores/processed/oof_predictions.csv     predicciones fuera de muestra y del holdout
  stores/processed/test_predictions.csv    predicciones del test por especificación
  views/cv_results.csv                     tabla comparativa

Uso:
  python scripts/04_train_models.py                       # todas las especificaciones
  python scripts/04_train_models.py --families boosting   # solo una familia
  python scripts/04_train_models.py --specs 12 13         # solo algunos números
"""

import argparse
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pandas as pd

from models import FAMILIES, build_model, fit_model, load_best_params
from utils import (ID_COL, PROCESSED_DIR, TARGET, VIEWS_DIR, chapinero_holdout,
                   error_summary, feature_sets, load_processed, random_cv, save_submission,
                   similarity_weights, spatial_cv_splits)


@dataclass(frozen=True)
class Spec:
    """Una especificación: familia, conjunto de predictores y variantes de entrenamiento."""
    family: str
    features: str
    variant: str = ""        # sufijo del nombre del envío
    target: str = "log"      # 'log', 'level' o 'per_m2' (ver models.build_model)
    weighted: bool = False   # pondera el entrenamiento por parecido al test
    by_type: bool = False    # modelo aparte para apartamentos y para casas
    tuned: bool = False      # usa los hiperparámetros de 05_tune_models.py

    def name(self, number):
        suffix = f"_{self.variant}" if self.variant else ""
        return f"{number:02d}_{self.family}_{self.features}{suffix}"


# El orden cuenta la historia: de simple a complejo. Un número nunca se reutiliza.
SPECS = [
    Spec("linear", "base"),
    Spec("linear", "base_text"),
    Spec("linear", "full"),
    Spec("elastic_net", "full"),
    Spec("cart", "full"),
    Spec("random_forest", "full"),
    Spec("random_forest", "full_coords"),
    Spec("boosting", "base_text"),
    Spec("boosting", "full"),
    Spec("boosting", "full_coords"),
    Spec("neural_net", "full"),
    # Idea 1: ponderar el entrenamiento por parecido al test.
    Spec("boosting", "full_coords", "weighted", weighted=True),
    Spec("random_forest", "full", "weighted", weighted=True),
    # Idea 2: modelo aparte para apartamentos (97 % del test) y para casas.
    Spec("boosting", "full_coords", "by_type", by_type=True),
    Spec("random_forest", "full", "by_type", by_type=True),
    # Idea 3: objetivo alternativo. Precio en nivel con pérdida L1 y precio por m2.
    Spec("boosting", "full_coords", "by_type_level", target="level", by_type=True),
    Spec("boosting", "full_coords", "by_type_per_m2", target="per_m2", by_type=True),
    Spec("random_forest", "full", "by_type_per_m2", target="per_m2", by_type=True),
]


def fit_predict(spec, params, X_fit, y_fit, X_pred, weights=None):
    """Ajusta una copia del modelo y predice."""
    model = build_model(spec.family, params, spec.target, spec.by_type)
    fit_model(model, X_fit, y_fit, weights)
    return model.predict(X_pred)


def evaluate(spec, params, X, y, splits, weights=None):
    """Predicciones para las filas de validación de cada partición (NaN en el resto)."""
    pred = np.full(len(X), np.nan)
    for fit_idx, val_idx in splits:
        w = None if weights is None else weights[fit_idx]
        pred[val_idx] = fit_predict(spec, params, X.iloc[fit_idx], y.iloc[fit_idx],
                                    X.iloc[val_idx], w)
    return pred


def update_table(path, new, key):
    """Agrega columnas o filas nuevas a una tabla guardada, reemplazando las repetidas."""
    if not path.exists():
        return new
    previous = pd.read_csv(path)
    if key == "columns":
        keep = [c for c in previous.columns if c not in new.columns]
        return pd.concat([new, previous[keep]], axis=1)
    return pd.concat([previous[~previous["model"].isin(new["model"])], new])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES)
    parser.add_argument("--specs", nargs="+", type=int, help="números de SPECS a correr")
    args = parser.parse_args()

    train, test = load_processed()
    y = train[TARGET]
    sets = feature_sets(train.columns)
    best_params = load_best_params()
    weights = similarity_weights(train["test_similarity"])
    schemes = {"random": list(random_cv().split(train)),
               "spatial": spatial_cv_splits(train),
               "chapinero": [chapinero_holdout(train, test)]}

    oof_table = train[[ID_COL, TARGET]].copy()
    test_table = test[[ID_COL]].copy()
    for number, spec in enumerate(SPECS, start=1):
        if spec.family not in args.families or (args.specs and number not in args.specs):
            continue
        start = time.time()
        name = spec.name(number)
        features = sets[spec.features]
        key = f"{spec.family}_{spec.features}"
        params = best_params.get(key, {}) if spec.tuned else {}
        w = weights if spec.weighted else None

        row = {"model": name, "family": spec.family, "features": spec.features,
               "n_features": len(features), "tuned": bool(params), "target": spec.target,
               "weighted": spec.weighted, "by_type": spec.by_type}
        for scheme, splits in schemes.items():
            pred = evaluate(spec, params, train[features], y, splits, w)
            oof_table[f"{name}__{scheme}"] = pred
            rows_eval = ~np.isnan(pred)
            summary = error_summary(y[rows_eval], pred[rows_eval])
            row[f"mae_{scheme}"] = summary["mae"]
            row[f"bias_{scheme}"] = summary["bias"]
            row[f"share_over_{scheme}"] = summary["share_overpredicted"]

        # Ajuste final con todo el entrenamiento y envío a Kaggle.
        test_pred = fit_predict(spec, params, train[features], y, test[features], w)
        test_table[name] = test_pred
        row["submission"] = save_submission(test[ID_COL], test_pred, name).name
        row["minutes"] = (time.time() - start) / 60
        print(f"{name:<40} MAE aleatoria {row['mae_random']/1e6:6.1f} M | "
              f"espacial {row['mae_spatial']/1e6:6.1f} M | "
              f"chapinero {row['mae_chapinero']/1e6:6.1f} M "
              f"(sesgo {row['bias_chapinero']/1e6:+6.1f}) | {row['minutes']:.1f} min",
              flush=True)
        # Se guarda tras cada especificación: si una corrida se corta, no se pierde lo hecho.
        spec_columns = [c for c in oof_table if c.startswith(f"{name}__")]
        save_results(row, oof_table[[ID_COL, TARGET] + spec_columns], test_table[[ID_COL, name]])


@contextmanager
def results_lock():
    """Candado de archivo: dos corridas simultáneas no escriben las tablas a la vez."""
    lock = PROCESSED_DIR / "results.lock"
    while True:
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL)
            break
        except FileExistsError:
            time.sleep(1)
    try:
        yield
    finally:
        os.close(handle)
        os.remove(lock)


def save_results(row, oof_part, test_part):
    """Actualiza la tabla de resultados y las predicciones guardadas de una especificación;
    las demás especificaciones conservan su resultado anterior."""
    VIEWS_DIR.mkdir(exist_ok=True)
    results_path = VIEWS_DIR / "cv_results.csv"
    with results_lock():
        update_table(results_path, pd.DataFrame([row]), "rows").sort_values(
            "model").to_csv(results_path, index=False)
        for table, file in [(oof_part, "oof_predictions.csv"),
                            (test_part, "test_predictions.csv")]:
            path = PROCESSED_DIR / file
            update_table(path, table, "columns").to_csv(path, index=False)


if __name__ == "__main__":
    main()
