"""Sintoniza el ridge de texto (ronda 4, pieza A, paso 2a): alfa, min_df y n-gramas de
palabras, en una rejilla completa.

Solo usa las filas de ajuste del holdout nororiental (la validación no se toca) y mide el
MAE del log del precio en 5 particiones que dejan juntos a los gemelos, que es como se
calcula txt_ridge dentro de cada ajuste. El ganador se guarda en best_params.json con la
clave 'text_ridge' y lo usa la fuente 'ridge_tuned' de 04_train_models.py.

Salida: views/tuning_text_ridge.csv
Uso: python 14_tune_text_ridge.py
"""

import itertools
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from models import RidgeText, load_best_params
from utils import (BEST_PARAMS_PATH, SEED, TARGET, VIEWS_DIR, load_processed,
                   northeast_holdout, twin_groups)

GRID = {"alpha": [0.3, 1.0, 3.0, 10.0], "min_df": [2, 5], "ngram_range": [(1, 1), (1, 2), (1, 3)]}
MAX_FEATURES = 200_000


def main():
    train, test = load_processed()
    fit_idx, _ = northeast_holdout(train, test)
    X = train.iloc[fit_idx].reset_index(drop=True)
    log_y = np.log(X[TARGET].to_numpy(float))
    folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(
        X, groups=twin_groups(X)))
    rows = []
    for min_df, ngram_range in itertools.product(GRID["min_df"], GRID["ngram_range"]):
        for alpha in GRID["alpha"]:
            start = time.time()
            source = RidgeText(alpha=alpha, min_df=min_df, ngram_range=ngram_range,
                               max_features=MAX_FEATURES)
            oof = source.fit_features(X, log_y, folds)["txt_ridge"]
            rows.append({"alpha": alpha, "min_df": min_df, "ngram_max": ngram_range[1],
                         "mae_log": float(np.mean(np.abs(oof - log_y))),
                         "seconds": time.time() - start})
            print(rows[-1], flush=True)
    # El ridge de la fila 75 (alfa 3, min_df 5, bigramas, 100.000 términos) como referencia.
    reference = RidgeText().fit_features(X, log_y, folds)["txt_ridge"]
    rows.append({"alpha": 3.0, "min_df": 5, "ngram_max": 2, "max_features": 100_000,
                 "mae_log": float(np.mean(np.abs(reference - log_y))), "reference": True})
    trace = pd.DataFrame(rows).sort_values("mae_log")
    VIEWS_DIR.mkdir(exist_ok=True)
    trace.to_csv(VIEWS_DIR / "tuning_text_ridge.csv", index=False)
    best = trace[trace["reference"].isna()].iloc[0]
    params = {"alpha": float(best["alpha"]), "min_df": int(best["min_df"]),
              "ngram_range": [1, int(best["ngram_max"])], "max_features": MAX_FEATURES}
    stored = load_best_params()
    stored["text_ridge"] = params
    BEST_PARAMS_PATH.write_text(json.dumps(stored, indent=2))
    print(trace.to_string())
    print(f"Ganador: {params}")


if __name__ == "__main__":
    main()
