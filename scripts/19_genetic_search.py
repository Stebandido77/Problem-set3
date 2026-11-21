"""Hiperparámetros del bosque cuantílico con un algoritmo genético (ronda 6, paso 2).

NSGA-II de Optuna con un solo objetivo: población de 20, 6 generaciones (120 evaluaciones),
cruce uniforme con probabilidad 0,9 y mutación con probabilidad 0,2 por hiperparámetro. La
primera generación es aleatoria; las siguientes salen de cruzar y mutar a los mejores.
Se compara con búsqueda aleatoria pura sobre el mismo espacio, evaluación por evaluación
(en la ronda 6 no alcanzó el tiempo para correrla: la primera generación del genético, que
es aleatoria, hace de referencia). La generación se cuenta como número de ensayo // 20; los
ensayos cortados al reanudar también consumen número, así que puede llegar a 6.

Función objetivo: la de 15 y 17, promedio del MAE del holdout nororiental ponderado como el
test y del MAE de la validación espacial, con una semilla. El texto se calcula una vez por
partición (15_optuna_search.Problem).

Ojo: como en las rondas 4 y 5, la búsqueda usa las particiones con las que luego se reporta.

Salidas:
  views/genetic_<sampler>.csv          traza por evaluación, con su generación
  views/genetic_convergence.png        mejor valor acumulado y media por generación
  views/genetic_confirm.csv            5 semillas del ganador y de los parámetros de la 103
  best_params.json, clave 'random_forest_ga'
  stores/processed/optuna_round6.db    estudios (no se versiona): una búsqueda cortada sigue
                                       donde iba al correrla otra vez

Uso:
  python 19_genetic_search.py --spec 107 --sampler nsga2 --n-trials 120
  python 19_genetic_search.py --spec 107 --sampler random --n-trials 60
  python 19_genetic_search.py --plot
  python 19_genetic_search.py --spec 107 --confirm
"""

import argparse
import importlib
import json
import time

import numpy as np
import pandas as pd

from models import load_best_params
from utils import BEST_PARAMS_PATH, PROCESSED_DIR, SEED, VIEWS_DIR

FamilyProblem = importlib.import_module("17_tune_families").FamilyProblem

STEP = "regressor__randomforestquantileregressor__"
POPULATION = 20
N_SEEDS = 5
KEY = "random_forest_ga"


def suggest(trial):
    """Árboles, profundidad máxima, mínimo por hoja, fracción de predictores por corte y
    fracción de filas por árbol."""
    return {STEP + "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            STEP + "max_depth": trial.suggest_int("max_depth", 8, 40),
            STEP + "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30, log=True),
            STEP + "max_features": trial.suggest_float("max_features", 0.15, 0.9),
            STEP + "max_samples": trial.suggest_float("max_samples", 0.3, 1.0)}


def sampler(name):
    import optuna
    if name == "nsga2":
        return optuna.samplers.NSGAIISampler(population_size=POPULATION, crossover_prob=0.9,
                                             mutation_prob=0.2, seed=SEED)
    return optuna.samplers.RandomSampler(seed=SEED)


def search(problem, name, n_trials):
    import optuna
    study = optuna.create_study(
        direction="minimize", sampler=sampler(name), study_name=f"rf_{name}",
        storage=f"sqlite:///{(PROCESSED_DIR / 'optuna_round6.db').as_posix()}",
        load_if_exists=True)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trace_path = VIEWS_DIR / f"genetic_{name}.csv"
    rows = pd.read_csv(trace_path).to_dict("records") if done and trace_path.exists() else []

    def objective(trial):
        start = time.time()
        params = suggest(trial)
        weighted, spatial = problem.score(problem.predictions(params))
        rows.append({"trial": trial.number, "generation": trial.number // POPULATION,
                     **trial.params, "mae_weighted": weighted, "mae_spatial": spatial,
                     "objective": (weighted + spatial) / 2, "seconds": time.time() - start})
        print(f"{name} {trial.number:3d} (gen. {trial.number // POPULATION}): "
              f"{rows[-1]['objective'] / 1e6:6.2f} M (pond. {weighted / 1e6:.1f}, "
              f"esp. {spatial / 1e6:.1f}) {rows[-1]['seconds']:.0f} s", flush=True)
        pd.DataFrame(rows).to_csv(trace_path, index=False)
        trial.set_user_attr("params", params)
        return (weighted + spatial) / 2

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=max(n_trials - len(done), 0))
    if name == "nsga2":
        best = load_best_params()
        best[KEY] = study.best_trial.user_attrs["params"]
        BEST_PARAMS_PATH.write_text(json.dumps(best, indent=2))
    print(f"Mejor {name}: {study.best_value / 1e6:.2f} M {study.best_params}")


def plot():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, label in [("nsga2", "Genético (NSGA-II)"), ("random", "Aleatoria")]:
        path = VIEWS_DIR / f"genetic_{name}.csv"
        if not path.exists():
            continue
        trace = pd.read_csv(path).sort_values("trial")
        line = ax.plot(trace["trial"], trace["objective"].cummin() / 1e6, label=label)[0]
        means = trace.groupby("generation")["objective"].mean() / 1e6
        ax.plot((means.index + 0.5) * POPULATION, means.to_numpy(), "o--",
                color=line.get_color(), alpha=0.5, label=f"{label}: media por bloque de 20")
    ax.set_xlabel("Evaluación")
    ax.set_ylabel("(MAE ponderado + espacial) / 2, M")
    ax.set_title("Bosque cuantílico: genético contra búsqueda aleatoria")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(VIEWS_DIR / "genetic_convergence.png", dpi=150)


def confirm(problem):
    """5 semillas con los parámetros del genético y con los de la 103 (random_forest_q70)."""
    rows = []
    for key in (KEY, "random_forest_q70"):
        params = load_best_params()[key]
        for k in range(N_SEEDS):
            seeded = {**params, STEP + "random_state": SEED + k}
            weighted, spatial = problem.score(problem.predictions(seeded))
            rows.append({"params": key, "seed": k, "mae_weighted": weighted,
                         "mae_spatial": spatial, "objective": (weighted + spatial) / 2})
            print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(VIEWS_DIR / "genetic_confirm.csv", index=False)
    print((table.groupby("params")[["mae_weighted", "mae_spatial", "objective"]]
           .agg(["mean", "std"]) / 1e6).round(2).to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=int, help="número de 04_train_models.py")
    parser.add_argument("--sampler", choices=["nsga2", "random"])
    parser.add_argument("--n-trials", type=int, default=120)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if args.plot:
        plot()
        return
    if args.spec is None:
        parser.error("hace falta --spec")
    problem = FamilyProblem(args.spec)
    if args.confirm:
        confirm(problem)
    else:
        search(problem, args.sampler, args.n_trials)
        plot()


if __name__ == "__main__":
    main()
