"""Búsqueda aleatoria de hiperparámetros con validación cruzada ESPACIAL.

Se sintoniza con validación espacial porque el objetivo es predecir en una zona que el
modelo no vio (Chapinero). Los mejores valores se guardan en
stores/processed/best_params.json y 04_train_models.py los usa automáticamente.

Uso:
  python scripts/05_tune_models.py --family boosting --features full --n-iter 30
"""

import argparse
import json

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from models import SEARCH_SPACES, build_model, load_best_params
from utils import (BEST_PARAMS_PATH, SEED, TARGET, VIEWS_DIR, feature_sets,
                   load_processed, spatial_cv_splits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=list(SEARCH_SPACES))
    parser.add_argument("--features", default="full")
    parser.add_argument("--n-iter", type=int, default=20)
    args = parser.parse_args()

    train, _ = load_processed()
    features = feature_sets(train.columns)[args.features]
    search = RandomizedSearchCV(
        build_model(args.family),
        param_distributions=SEARCH_SPACES[args.family],
        n_iter=args.n_iter,
        scoring="neg_mean_absolute_error",
        cv=spatial_cv_splits(train),
        random_state=SEED,
        verbose=1,
    )
    search.fit(train[features], train[TARGET])

    key = f"{args.family}_{args.features}"
    best = load_best_params()
    # json no entiende tipos de numpy ni tuplas.
    best[key] = json.loads(json.dumps(search.best_params_, default=lambda v: v.item()))
    BEST_PARAMS_PATH.write_text(json.dumps(best, indent=2))

    # Traza completa de la búsqueda, para las diapositivas de comparación.
    VIEWS_DIR.mkdir(exist_ok=True)
    trace = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    trace.to_csv(VIEWS_DIR / f"tuning_{key}.csv", index=False)
    print(f"Mejor MAE espacial: {-search.best_score_/1e6:.1f} M\n{best[key]}")


if __name__ == "__main__":
    main()
