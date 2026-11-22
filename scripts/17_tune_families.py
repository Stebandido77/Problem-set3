"""Sintoniza las versiones cuantílicas de la ronda 5 (random forest, elastic net y CART) con
la misma búsqueda que el boosting de la ronda 4 (15_optuna_search.py): Optuna TPE, función
objetivo igual al promedio del MAE del holdout nororiental ponderado como el test y del MAE
de la validación espacial, y los valores por defecto de la receta como primer candidato.

El predictor de texto (ridge fuera de muestra) se calcula una vez por partición y se
reutiliza en todos los candidatos (15_optuna_search.Problem).

Ojo: como en la ronda 4, la búsqueda usa las mismas particiones con las que luego se
reporta, así que el MAE de la fila sintonizada es algo optimista. La medida limpia es Kaggle.

Salidas:
  views/tuning_<familia>_q70.csv   traza de la búsqueda
  best_params.json, clave '<familia>_q70'
  stores/processed/optuna_round5.db  estudio de Optuna (no se versiona): una búsqueda
                                     cortada sigue donde iba al correrla otra vez

Ronda 9: también la red cuantílica de la 148, con '--objective spatial' (solo la validación
espacial, que acertó con la red en Kaggle).

Uso:
  python 17_tune_families.py --spec 99 --n-trials 25   # 25 evaluaciones en total
  python 17_tune_families.py --spec 169 --n-trials 20 --objective spatial
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from models import load_best_params
from utils import BEST_PARAMS_PATH, PROCESSED_DIR, SEED, VIEWS_DIR

PROBLEM = __import__("importlib").import_module("15_optuna_search").Problem
ROOT = "regressor__"


def suggest_random_forest(trial):
    step = ROOT + "randomforestquantileregressor__"
    return {step + "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
            step + "max_features": trial.suggest_float("max_features", 0.15, 0.9),
            step + "max_depth": trial.suggest_categorical("max_depth", [None, 12, 20, 30])}


def suggest_elastic_net(trial):
    step = ROOT + "quantileshift__regressor__"
    return {step + "alpha": trial.suggest_float("alpha", 1e-3, 3.0, log=True),
            step + "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 1.0),
            ROOT + "splinedesign__n_knots_geo": trial.suggest_categorical("n_knots_geo",
                                                                          [4, 6, 8])}


def suggest_cart(trial):
    step = ROOT + "leafquantiletree__"
    return {step + "max_depth": trial.suggest_int("max_depth", 4, 25),
            step + "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 200, log=True),
            step + "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-2, 1e3, log=True)}


def suggest_quantile_net(trial):
    """Ronda 9: la red cuantílica de la 148 con una semilla (la fila final, la 171, promedia
    5)."""
    width = trial.suggest_categorical("width", [64, 128, 256, 512])
    n_layers = trial.suggest_int("n_layers", 1, 3)
    return {"hidden": tuple(max(width // 2 ** k, 32) for k in range(n_layers)),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "lr": trial.suggest_float("lr", 1e-3, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "n_seeds": 1}


SEARCHES = {"random_forest": suggest_random_forest, "elastic_net": suggest_elastic_net,
            "cart": suggest_cart, "quantile_net": suggest_quantile_net}
# Valores por defecto de models.QUANTILE_RECIPES: el primer candidato de cada búsqueda.
STARTS = {"random_forest": {"min_samples_leaf": 5, "max_features": 0.4, "max_depth": None},
          "elastic_net": {"alpha": 0.01, "l1_ratio": 0.5, "n_knots_geo": 6},
          "cart": {"max_depth": 12, "min_samples_leaf": 20, "ccp_alpha": 1e-2},
          # La 148: capas de 128 y 64, dropout 0,1, tasa 3e-3 y regularización 1e-4.
          "quantile_net": {"width": 128, "n_layers": 2, "dropout": 0.1, "lr": 3e-3,
                           "weight_decay": 1e-4}}
# Las familias de un núcleo ajustan las 6 particiones en paralelo; el bosque usa todos.
PARALLEL_FAMILIES = {"elastic_net", "cart", "quantile_net"}


class FamilyProblem(PROBLEM):
    """Las particiones de 15 con un modelo cualquiera: 'params' son rutas completas."""

    def _fit_one(self, params, part):
        X_fit, y_fit, X_val, _ = part
        model = self.training.build_spec_model(self.inner, params)
        return model.fit(X_fit, y_fit).predict(X_val)

    def predictions(self, params, seed=SEED, quantile=None):
        n_jobs = len(self.parts) if self.inner.family in PARALLEL_FAMILIES else 1
        return Parallel(n_jobs=n_jobs)(delayed(self._fit_one)(params, part)
                                       for part in self.parts)


def search(problem, family, n_trials, objective_name="mean"):
    import optuna
    key = f"{family}_q70"
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
        storage=f"sqlite:///{(PROCESSED_DIR / 'optuna_round5.db').as_posix()}",
        study_name=key, load_if_exists=True)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not study.trials:
        study.enqueue_trial(STARTS[family])
    trace_path = VIEWS_DIR / f"tuning_{key}.csv"
    rows = pd.read_csv(trace_path).to_dict("records") if done and trace_path.exists() else []

    def objective(trial):
        start = time.time()
        params = SEARCHES[family](trial)
        preds = problem.predictions(params)
        weighted, spatial = problem.score(preds)
        value = spatial if objective_name == "spatial" else (weighted + spatial) / 2
        rows.append({"trial": trial.number, **trial.params, "mae_weighted": weighted,
                     "mae_spatial": spatial, "objective": value,
                     "mean_northeast": float(np.mean(preds[0])),
                     "seconds": time.time() - start})
        print(f"{family} {trial.number:2d}: {rows[-1]['objective'] / 1e6:6.2f} M "
              f"(pond. {weighted / 1e6:.1f}, esp. {spatial / 1e6:.1f}) "
              f"{rows[-1]['seconds']:.0f} s {trial.params}", flush=True)
        pd.DataFrame(rows).to_csv(trace_path, index=False)
        trial.set_user_attr("params", params)
        return value

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=max(n_trials - len(done), 0))
    best = load_best_params()
    best[key] = study.best_trial.user_attrs["params"]
    BEST_PARAMS_PATH.write_text(json.dumps(best, indent=2))
    print(f"Mejor {family}: {study.best_value / 1e6:.2f} M {study.best_params}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=int, required=True,
                        help="número de 04_train_models.py (sin sintonizar)")
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--objective", choices=["mean", "spatial"], default="mean",
                        help="promedio de ponderado y espacial, o solo la espacial (ronda 9)")
    args = parser.parse_args()
    problem = FamilyProblem(args.spec)
    search(problem, problem.inner.family, args.n_trials, args.objective)


if __name__ == "__main__":
    main()
