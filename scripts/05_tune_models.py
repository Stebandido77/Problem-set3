"""Búsqueda aleatoria de hiperparámetros con validación cruzada ESPACIAL.

Se sintoniza con validación espacial porque el objetivo es predecir en una zona que el
modelo no vio (Chapinero). Los mejores valores se guardan en
stores/processed/best_params.json y 04_train_models.py los usa automáticamente.

Uso:
  python scripts/05_tune_models.py --family boosting --features full --n-iter 30
  python scripts/05_tune_models.py --family boosting --features full --weighted --by-type

Con --weighted y --by-type se sintoniza la variante ponderada por parecido al test y con
modelo aparte por tipo de inmueble; los parámetros se guardan con la misma clave y se
aplican a todas las especificaciones de esa familia y conjunto.

Con --spec N se sintoniza una especificación de 04_train_models.py tal cual (con sus
envoltorios: por tipo, objetivo en nivel, cuantil...) en el holdout nororiental ponderado
como el test (utils.weighted_mae), que es el criterio de la ronda 3. Cada candidato se
entrena con el ajuste del holdout y se evalúa en su validación. Los parámetros se guardan
con la clave '<nombre de la especificación sin número>_holdout' y la traza en
views/tuning_<clave>.csv.

  python scripts/05_tune_models.py --spec 65 --n-iter 40 --n-jobs 4
"""

import argparse
import importlib
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV

from models import SEARCH_SPACES, build_model, load_best_params, northeast_building_groups
from utils import (BEST_PARAMS_PATH, SEED, TARGET, VIEWS_DIR, feature_sets,
                   load_processed, northeast_holdout, similarity_weights,
                   spatial_cv_splits, weighted_mae)

# Núcleos por candidato de LightGBM cuando se evalúan varios a la vez.
THREADS_PARAM = {"boosting": "regressor__lgbmregressor__n_jobs"}


def spec_key(spec):
    """Clave de best_params.json de una especificación sintonizada en el holdout: su nombre
    sin el número y con el sufijo '_holdout', para no chocar con las claves de la búsqueda
    con validación espacial ('<familia>_<conjunto>')."""
    return f"{spec.name(0)[3:]}_holdout"


def evaluate_candidate(training, spec, params, X, y, fit_idx, val_idx, groups):
    """MAE ponderado como el test de un candidato entrenado con el ajuste del holdout."""
    start = time.time()
    model = training.build_spec_model(spec, params)
    model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
    pred = model.predict(X.iloc[val_idx])
    return {**{f"param_{k}": v for k, v in params.items()},
            "mae_weighted": weighted_mae(y.iloc[val_idx], pred, groups),
            "mae_northeast": float(np.mean(np.abs(pred - y.iloc[val_idx]))),
            "bias_northeast": float(np.mean(pred - y.iloc[val_idx])),
            "seconds": time.time() - start}


def tune_spec(number, n_iter, n_jobs):
    """Búsqueda aleatoria sobre una especificación de 04 con el holdout ponderado."""
    training = importlib.import_module("04_train_models")
    spec = training.NUMBERED_SPECS[number]
    train, test = load_processed()
    y = train[TARGET]
    X = train[training.spec_features(spec, feature_sets(train.columns))]
    fit_idx, val_idx = northeast_holdout(train, test)
    _, groups = northeast_building_groups(train, test, y)
    sampler = list(ParameterSampler(SEARCH_SPACES[spec.family], n_iter, random_state=SEED))
    threads = ({THREADS_PARAM[spec.family]: max(1, 16 // n_jobs)}
               if spec.family in THREADS_PARAM else {})
    # El punto de partida (los parámetros por defecto) entra como candidato 0.
    candidates = [{}] + [{k: (v.item() if hasattr(v, "item") else v) for k, v in c.items()}
                         for c in sampler]
    rows = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(evaluate_candidate)(training, spec, {**c, **threads}, X, y, fit_idx,
                                    val_idx, groups)
        for c in candidates)
    trace = pd.DataFrame(rows).assign(candidate=range(len(rows))).sort_values("mae_weighted")
    key = spec_key(spec)
    VIEWS_DIR.mkdir(exist_ok=True)
    trace.to_csv(VIEWS_DIR / f"tuning_{key}.csv", index=False)
    best = load_best_params()
    best[key] = candidates[trace["candidate"].iloc[0]]
    BEST_PARAMS_PATH.write_text(json.dumps(best, indent=2))
    print(f"{key}: ponderado {trace['mae_weighted'].iloc[0]/1e6:.1f} M "
          f"(por defecto {rows[0]['mae_weighted']/1e6:.1f} M): {best[key]}")


def search_space(family, by_type):
    """Espacio de búsqueda con la ruta correcta si el modelo va separado por tipo."""
    prefix = "model__" if by_type else ""
    return {prefix + key: value for key, value in SEARCH_SPACES[family].items()}


def weight_params(family, weights, by_type):
    """Pesos de muestra con la ruta que espera el estimador."""
    if weights is None:
        return {}
    if by_type:
        return {"sample_weight": weights}
    last_step = build_model(family).regressor.steps[-1][0]
    return {f"{last_step}__sample_weight": weights}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=int, help="número de 04 a sintonizar en el holdout")
    parser.add_argument("--family", choices=list(SEARCH_SPACES))
    parser.add_argument("--features", default="full")
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--by-type", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="candidatos en paralelo (útil en familias de un solo núcleo)")
    args = parser.parse_args()
    if args.spec:
        tune_spec(args.spec, args.n_iter, args.n_jobs)
        return

    train, _ = load_processed()
    features = feature_sets(train.columns)[args.features]
    weights = similarity_weights(train["test_similarity"]) if args.weighted else None
    search = RandomizedSearchCV(
        build_model(args.family, by_type=args.by_type),
        param_distributions=search_space(args.family, args.by_type),
        n_iter=args.n_iter,
        scoring="neg_mean_absolute_error",
        cv=spatial_cv_splits(train),
        random_state=SEED,
        n_jobs=args.n_jobs,
        verbose=1,
    )
    search.fit(train[features], train[TARGET],
               **weight_params(args.family, weights, args.by_type))

    key = f"{args.family}_{args.features}"
    best = load_best_params()
    # json no entiende tipos de numpy ni tuplas; la ruta se guarda sin el prefijo 'model__'.
    params = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
    best[key] = json.loads(json.dumps(params, default=lambda v: v.item()))
    BEST_PARAMS_PATH.write_text(json.dumps(best, indent=2))

    # Traza completa de la búsqueda, para las diapositivas de comparación.
    VIEWS_DIR.mkdir(exist_ok=True)
    trace = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    trace.to_csv(VIEWS_DIR / f"tuning_{key}.csv", index=False)
    print(f"Mejor MAE espacial: {-search.best_score_/1e6:.1f} M\n{best[key]}")


if __name__ == "__main__":
    main()
