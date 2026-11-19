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
"""

import argparse
import json

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from models import SEARCH_SPACES, build_model, load_best_params
from utils import (BEST_PARAMS_PATH, SEED, TARGET, VIEWS_DIR, feature_sets,
                   load_processed, similarity_weights, spatial_cv_splits)


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
    parser.add_argument("--family", required=True, choices=list(SEARCH_SPACES))
    parser.add_argument("--features", default="full")
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--by-type", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="candidatos en paralelo (útil en familias de un solo núcleo)")
    args = parser.parse_args()

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
