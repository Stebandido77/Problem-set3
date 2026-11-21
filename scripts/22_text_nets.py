"""Ronda 7, pieza D: ¿por qué la red no le gana al ridge en el texto?

Cada predictor de texto se evalúa solo, fuera de muestra: se ajusta con las filas de ajuste
del holdout nororiental y de cada partición de la validación espacial, y su predicción del
log del precio, llevada a pesos, se compara con el precio. Fuentes: el ridge de la 83, la red
cuantílica de la ronda 4 sobre embeddings de frases (EmbeddingNet) y las tres TextNet nuevas
(TF-IDF disperso, SVD de 300 componentes y embeddings tipo fastText aprendidos desde cero).
Ojo: EmbeddingNet apunta al cuantil 0,70, las demás a la media del log (como el ridge).

Con --tune se sintoniza una TextNet con NSGA-II de Optuna (algoritmo genético, un objetivo),
'--n-trials' evaluaciones. Objetivo: MAE de una partición interna (20 % de los grupos de
gemelos) de una submuestra de 12.000 filas del ajuste del holdout nororiental; ni la validación ni el test entran. Guarda la
traza en views/text_net_ga.csv y el ganador en best_params.json (clave 'text_net_ga').

Salidas: views/text_nets.csv (MAE ponderado, nororiental y espacial de cada fuente sola).

Uso:
  python scripts/22_text_nets.py                       # las cinco fuentes, sin sintonizar
  python scripts/22_text_nets.py --tune tfidf_net --n-trials 60   # se repite hasta 60
  python scripts/22_text_nets.py --sources text_net_ga # el ganador del genético
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from models import (EmbeddingNet, RidgeText, TextNet, load_best_params,
                    northeast_building_groups)
from utils import (BEST_PARAMS_PATH, PROCESSED_DIR, SEED, TARGET, VIEWS_DIR, add_embeddings, load_processed,
                   northeast_holdout, spatial_cv_splits, twin_groups, weighted_mae)

SOURCES = {
    "ridge": lambda: RidgeText(),
    "net": lambda: EmbeddingNet(),
    "tfidf_net": lambda: TextNet(name="txt_tfidf_net", representation="tfidf"),
    "svd_net": lambda: TextNet(name="txt_svd_net", representation="svd"),
    "hash_net": lambda: TextNet(name="txt_hash_net", representation="hash"),
}
TUNE_ROWS = 12_000
REPRESENTATIONS = {"tfidf_net": "tfidf", "svd_net": "svd", "hash_net": "hash"}


def tuned_source():
    params = load_best_params()["text_net_ga"]
    return TextNet(name="txt_net_ga", **{k: tuple(v) if isinstance(v, list) else v
                                        for k, v in params.items()})


SOURCES["text_net_ga"] = tuned_source


def fit_transform(source, X_fit, y_fit, X_pred):
    """Ajusta la fuente con todas las filas de ajuste (una sola 'partición' que las ve todas)
    y devuelve su predicción del log del precio para X_pred."""
    everything = [(np.arange(len(X_fit)), np.array([], dtype=int))]
    source.fit_features(X_fit, np.log(y_fit), everything)
    return next(iter(source.transform(X_pred).values()))


def evaluate_source(key, train, test, y, val_idx, groups, spatial):
    start = time.time()
    columns = list(dict.fromkeys(["is_house", "lat", "lon", "text", "surface", "bedrooms",
                                  "bathrooms"] + SOURCES[key]().inputs()))
    X = train[columns]
    fit_idx, _ = northeast_holdout(train, test)
    northeast = np.exp(fit_transform(SOURCES[key](), X.iloc[fit_idx], y[fit_idx],
                                     X.iloc[val_idx]))
    spatial_pred = np.full(len(y), np.nan)
    for f_idx, v_idx in spatial:
        spatial_pred[v_idx] = np.exp(fit_transform(SOURCES[key](), X.iloc[f_idx], y[f_idx],
                                                   X.iloc[v_idx]))
    row = {"source": key, "mae_weighted": weighted_mae(y[val_idx], northeast, groups) / 1e6,
           "mae_northeast": np.abs(northeast - y[val_idx]).mean() / 1e6,
           "mae_spatial": np.nanmean(np.abs(spatial_pred - y)) / 1e6,
           "minutes": (time.time() - start) / 60}
    print(row, flush=True)
    return row


def tune(representation, train, test, y, n_trials, timeout_s):
    """NSGA-II sobre una partición interna del ajuste del holdout nororiental."""
    import optuna
    fit_idx, _ = northeast_holdout(train, test)
    # Submuestra fija del ajuste para que 60 evaluaciones quepan en el tiempo de la ronda.
    fit_idx = np.random.default_rng(SEED).choice(fit_idx, TUNE_ROWS, replace=False)
    X = train.iloc[fit_idx][["is_house", "lat", "lon", "text", "surface", "bedrooms",
                             "bathrooms"]]
    y_fit = y[fit_idx]
    inner_fit, inner_val = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(
        X, groups=twin_groups(X)))

    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 0, 2)
        width = trial.suggest_categorical("width", [32, 64, 128, 256])
        params = dict(
            representation=representation,
            embed_dim=trial.suggest_categorical("embed_dim", [32, 64, 128, 256]),
            hidden=tuple([width] * n_layers),
            dropout=trial.suggest_float("dropout", 0.0, 0.6),
            lr=trial.suggest_float("lr", 3e-4, 1e-2, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
            max_epochs=trial.suggest_int("max_epochs", 4, 20))
        trial.set_user_attr("params", {k: list(v) if isinstance(v, tuple) else v
                                       for k, v in params.items()})
        pred = np.exp(fit_transform(TextNet(**params), X.iloc[inner_fit], y_fit[inner_fit],
                                    X.iloc[inner_val]))
        return np.abs(pred - y_fit[inner_val]).mean() / 1e6

    sampler = optuna.samplers.NSGAIISampler(population_size=12, crossover_prob=0.9,
                                            mutation_prob=0.2, seed=SEED)
    # Con la base en disco la búsqueda se reanuda en tramos (cada corrida, máximo 30 min).
    storage = f"sqlite:///{PROCESSED_DIR / 'optuna_round7.db'}"
    study = optuna.create_study(direction="minimize", sampler=sampler, storage=storage,
                                study_name=f"text_net_{representation}", load_if_exists=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    done = len([t for t in study.trials if t.state.name == "COMPLETE"])
    study.optimize(objective, n_trials=max(n_trials - done, 0), timeout=timeout_s,
                   callbacks=[lambda s, t: print(f"ensayo {t.number}: {t.value:.2f}",
                                                 flush=True)])
    trace = study.trials_dataframe()
    trace["generation"] = trace["number"] // 12
    trace.to_csv(VIEWS_DIR / "text_net_ga.csv", index=False)
    best = load_best_params()
    best["text_net_ga"] = study.best_trial.user_attrs["params"]
    BEST_PARAMS_PATH.write_text(json.dumps(best, indent=2))
    print("mejor", study.best_value, study.best_trial.user_attrs["params"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=list(SOURCES)[:5], choices=SOURCES)
    parser.add_argument("--tune", choices=list(REPRESENTATIONS))
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--timeout-min", type=float, default=25,
                        help="minutos de búsqueda en esta corrida; se reanuda después")
    args = parser.parse_args()

    train, test = load_processed()
    y = train[TARGET].to_numpy(float)
    if args.tune:
        tune(REPRESENTATIONS[args.tune], train, test, y, args.n_trials, args.timeout_min * 60)
        return
    if "net" in args.sources:
        train, test = add_embeddings(train, test)
    val_idx, groups = northeast_building_groups(train, test, train[TARGET])
    spatial = spatial_cv_splits(train)
    path = VIEWS_DIR / "text_nets.csv"
    for key in args.sources:
        # Se guarda tras cada fuente: si el tope de tiempo corta la corrida, no se pierde.
        table = pd.DataFrame([evaluate_source(key, train, test, y, val_idx, groups, spatial)])
        if path.exists():
            previous = pd.read_csv(path)
            table = pd.concat([previous[~previous["source"].isin(table["source"])], table])
        table.to_csv(path, index=False)
    print(table.round(1).to_string())


if __name__ == "__main__":
    main()
