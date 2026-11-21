"""Hiperparámetros del LightGBM con optimización guiada (ronda 4, pieza B).

Compara dos estrategias de Optuna con el mismo presupuesto: TPE (bayesiana) y CMA-ES
(evolutiva). La función objetivo es el promedio del MAE del holdout nororiental ponderado
como el test y del MAE de la validación espacial (5 particiones), con una semilla.

Los predictores de texto (fuera de muestra, models.TextPredictors) no dependen de los
hiperparámetros del LightGBM: se calculan una vez por partición y se reutilizan en todos
los candidatos. El punto de partida (los parámetros de la fila 83, que usa también la 94)
entra como primer candidato en las dos búsquedas.

Ojo: la búsqueda, la confirmación con 10 semillas y la comparación de cuantiles usan las
mismas particiones (holdout nororiental y validación espacial), así que sus cifras, y las de
la fila 98 de 04, son optimistas frente a la 94. La medida limpia es el tablero de Kaggle.

Salidas:
  views/optuna_<sampler>.csv          traza de cada búsqueda
  views/optuna_convergence.png        mejor valor acumulado por evaluación
  views/optuna_confirm_<clave>.csv    10 semillas por esquema del ganador (la 94 es el de partida)
  views/optuna_quantiles_<clave>.csv  ganador con cuantiles 0,65, 0,70 y 0,75 (3 semillas)
  best_params.json, clave 'optuna_<sampler>'

Uso:
  python 15_optuna_search.py --spec 94 --sampler tpe cmaes --n-trials 60
  python 15_optuna_search.py --plot
  python 15_optuna_search.py --spec 94 --confirm optuna_tpe
"""

import argparse
import dataclasses
import importlib
import json
import time

import numpy as np
import pandas as pd

from models import load_best_params, northeast_building_groups
from utils import (BEST_PARAMS_PATH, SEED, TARGET, VIEWS_DIR, add_embeddings, feature_sets,
                   load_processed, northeast_holdout, spatial_cv_splits, weighted_mae)

PREFIX = "regressor__lgbmregressor__"
# Los valores de la receta de la 83 (models.family_pipeline) son el punto de partida. En la
# búsqueda, una profundidad de 16 quiere decir sin límite (-1, el valor de la 83), y L1 parte
# de 0,001 en lugar de 0.
DEPTH_UNLIMITED = 16
START = {"num_leaves": 63, "max_depth": 16, "learning_rate": 0.03, "n_estimators": 1500,
         "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.6,
         "reg_alpha": 1e-3, "reg_lambda": 1.0}
N_SEEDS = 10
QUANTILES = (0.65, 0.70, 0.75)
QUANTILE_SEEDS = 3


def lightgbm_params(params):
    """Hiperparámetros de la búsqueda tal como los recibe LightGBM."""
    depth = params["max_depth"]
    return {**params, "max_depth": -1 if depth >= DEPTH_UNLIMITED else depth}


def suggest(trial):
    """Espacio de búsqueda: hojas, profundidad, tasa, árboles, mínimo por hoja, muestreo de
    filas y columnas, L1 y L2."""
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 16),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 300, 3000, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
    }


class Problem:
    """Particiones del holdout nororiental y de la validación espacial con los predictores
    de texto ya calculados, y el modelo de la especificación sin la capa de texto."""

    def __init__(self, number):
        training = importlib.import_module("04_train_models")
        spec = training.NUMBERED_SPECS[number]
        train, test = load_processed()
        if spec.text_sources and any(c.startswith("emb_") for key in spec.text_sources
                                     for c in training.TEXT_SOURCES[key].inputs()):
            train, test = add_embeddings(train, test)
        y = train[TARGET]
        X = train[training.spec_features(spec, feature_sets(train.columns))]
        self.training, self.y = training, y
        self.inner = dataclasses.replace(spec, text="", text_sources=(), n_seeds=1)
        self.val_northeast, self.groups = northeast_building_groups(train, test, y)
        splits = [northeast_holdout(train, test)] + spatial_cv_splits(train)
        self.parts = [self._text_part(spec, X, y, fit_idx, val_idx)
                      for fit_idx, val_idx in splits]

    def _text_part(self, spec, X, y, fit_idx, val_idx):
        """(X de ajuste, y de ajuste, X de validación, índices de validación) con texto."""
        wrapper = self.training.build_spec_model(spec, {})
        X_fit = wrapper.fit_text(X.iloc[fit_idx], y.iloc[fit_idx])
        X_val = wrapper.transform_text(X.iloc[val_idx])
        return X_fit, y.iloc[fit_idx], X_val, val_idx

    def predictions(self, params, seed=SEED, quantile=None):
        """Predicción de cada partición con unos hiperparámetros, una semilla y, si se da,
        otro cuantil en lugar del de la especificación."""
        full = {PREFIX + k: v for k, v in lightgbm_params(params).items()}
        full[PREFIX + "random_state"] = seed
        inner = self.inner
        if quantile is not None:
            inner = dataclasses.replace(inner, params=self.training.quantile_loss(quantile))
        preds = []
        for X_fit, y_fit, X_val, _ in self.parts:
            model = self.training.build_spec_model(inner, full)
            preds.append(model.fit(X_fit, y_fit).predict(X_val))
        return preds

    def score(self, preds):
        """MAE ponderado del holdout nororiental y MAE de la validación espacial."""
        weighted = weighted_mae(self.y.iloc[self.parts[0][3]], preds[0], self.groups)
        errors = np.concatenate([np.abs(p - self.y.iloc[part[3]].to_numpy())
                                 for p, part in zip(preds[1:], self.parts[1:])])
        return weighted, errors.mean()


def search(problem, sampler_name, n_trials):
    import optuna
    sampler = (optuna.samplers.TPESampler(seed=SEED) if sampler_name == "tpe"
               else optuna.samplers.CmaEsSampler(seed=SEED, x0=START, sigma0=None))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(START)
    rows = []

    def objective(trial):
        start = time.time()
        params = suggest(trial)
        weighted, spatial = problem.score(problem.predictions(params))
        rows.append({"trial": trial.number, **params, "mae_weighted": weighted,
                     "mae_spatial": spatial, "objective": (weighted + spatial) / 2,
                     "seconds": time.time() - start})
        print(f"{sampler_name} {trial.number:2d}: {rows[-1]['objective'] / 1e6:6.2f} M "
              f"(pond. {weighted / 1e6:.1f}, esp. {spatial / 1e6:.1f}) "
              f"{rows[-1]['seconds']:.0f} s", flush=True)
        pd.DataFrame(rows).to_csv(VIEWS_DIR / f"optuna_{sampler_name}.csv", index=False)
        return (weighted + spatial) / 2

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)
    best = load_best_params()
    best[f"optuna_{sampler_name}"] = {PREFIX + k: v
                                      for k, v in lightgbm_params(study.best_params).items()}
    BEST_PARAMS_PATH.write_text(json.dumps(best, indent=2))
    print(f"Mejor {sampler_name}: {study.best_value / 1e6:.2f} M {study.best_params}")


def plot():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, label in [("tpe", "TPE (bayesiana)"), ("cmaes", "CMA-ES (evolutiva)")]:
        path = VIEWS_DIR / f"optuna_{name}.csv"
        if not path.exists():
            continue
        trace = pd.read_csv(path).sort_values("trial")
        ax.plot(trace["trial"], trace["objective"].cummin() / 1e6, label=label)
        ax.scatter(trace["trial"], trace["objective"] / 1e6, s=8, alpha=0.4)
    ax.set_xlabel("Evaluación")
    ax.set_ylabel("(MAE ponderado + espacial) / 2, M")
    ax.set_title("Convergencia de la búsqueda de hiperparámetros del LightGBM")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VIEWS_DIR / "optuna_convergence.png", dpi=150)


def tuned_params(key):
    """Hiperparámetros guardados en best_params.json, en el formato de la búsqueda."""
    tuned = {k.removeprefix(PREFIX): v for k, v in load_best_params()[key].items()}
    tuned["max_depth"] = DEPTH_UNLIMITED if tuned["max_depth"] < 0 else tuned["max_depth"]
    return tuned


def confirm_seeds(problem, key):
    """10 semillas con los parámetros de 'key', en cada esquema: MAE de cada semilla (para ver
    la variación) y MAE del promedio de las 10. Los de partida con 10 semillas son la fila
    94 de 04_train_models.py, que ya tiene esos dos números."""
    runs = [problem.predictions(tuned_params(key), SEED + k) for k in range(N_SEEDS)]
    rows = [dict(zip(["mae_weighted", "mae_spatial"], problem.score(preds)), seed=k)
            for k, preds in enumerate(runs)]
    average = problem.score([np.mean(p, axis=0) for p in zip(*runs)])
    rows.append(dict(zip(["mae_weighted", "mae_spatial"], average), seed="average"))
    print(pd.DataFrame(rows).to_string(), flush=True)
    pd.DataFrame(rows).to_csv(VIEWS_DIR / f"optuna_confirm_{key}.csv", index=False)


def compare_quantiles(problem, key):
    """Los parámetros de 'key' con cada cuantil de QUANTILES, promedio de 3 semillas."""
    rows = []
    for quantile in QUANTILES:
        runs = [problem.predictions(tuned_params(key), SEED + k, quantile)
                for k in range(QUANTILE_SEEDS)]
        averaged = [np.mean(p, axis=0) for p in zip(*runs)]
        weighted, spatial = problem.score(averaged)
        bias = np.mean(averaged[0] - problem.y.iloc[problem.parts[0][3]].to_numpy())
        rows.append({"quantile": quantile, "mae_weighted": weighted, "mae_spatial": spatial,
                     "bias_northeast": float(bias)})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(VIEWS_DIR / f"optuna_quantiles_{key}.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=int, help="número de 04_train_models.py")
    parser.add_argument("--sampler", nargs="+", choices=["tpe", "cmaes"],
                        help="una o las dos; con las dos, el texto se calcula una sola vez")
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--confirm", help="clave de best_params.json a confirmar")
    args = parser.parse_args()
    VIEWS_DIR.mkdir(exist_ok=True)
    if args.plot:
        plot()
    elif args.confirm:
        if args.spec is None:
            parser.error("--confirm necesita --spec")
        problem = Problem(args.spec)
        confirm_seeds(problem, args.confirm)
        compare_quantiles(problem, args.confirm)
    else:
        if args.spec is None or args.sampler is None:
            parser.error("la búsqueda necesita --spec y --sampler")
        problem = Problem(args.spec)
        for sampler_name in args.sampler:
            search(problem, sampler_name, args.n_trials)
        plot()


if __name__ == "__main__":
    main()
