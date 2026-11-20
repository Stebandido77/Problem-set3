"""Entrena las seis familias de algoritmos (y el híbrido en_boosting), las evalúa con
validación cruzada aleatoria, validación cruzada espacial, el holdout tipo Chapinero y el
holdout nororiental, y escribe un envío de Kaggle por cada especificación.

Salidas:
  stores/submissions/<nn>_<familia>_<predictores>[_variante].csv   envíos para Kaggle
  stores/processed/oof_predictions.csv     predicciones fuera de muestra y del holdout
  stores/processed/test_predictions.csv    predicciones del test por especificación
  views/cv_results.csv                     tabla comparativa

Uso:
  python scripts/04_train_models.py                       # todas las especificaciones
  python scripts/04_train_models.py --families boosting   # solo una familia
  python scripts/04_train_models.py --specs 12 13         # solo algunos números
  python scripts/04_train_models.py --specs 54 --schemes northeast --no-submission
      # un solo esquema; los demás conservan su resultado anterior (ojo: si la
      # especificación cambió, hay que correrla con todos los esquemas)
"""

import argparse
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from models import (FAMILIES, ByPropertyType, FocusFineTune, ImputedArea, NeighborhoodLevel,
                    PriceWeighted, ResidualBoosting, SameBuilding, SeedAverage,
                    SyntheticTail, TruncationCorrection, build_model, fit_model,
                    load_best_params)
from utils import (COORD_FEATURES, ID_COL, PROCESSED_DIR, TARGET, VIEWS_DIR,
                   chapinero_holdout, error_summary, feature_sets, load_processed,
                   northeast_holdout, random_cv, save_submission, similarity_weights,
                   spatial_cv_splits)


# Esquemas de validación: nombre -> función (train, test) -> lista de particiones.
SCHEMES = {
    "random": lambda train, test: list(random_cv().split(train)),
    "spatial": lambda train, test: spatial_cv_splits(train),
    "chapinero": lambda train, test: [chapinero_holdout(train, test)],
    "northeast": lambda train, test: [northeast_holdout(train, test)],
}

# Familias que usan un solo núcleo: conviene ajustar sus particiones en paralelo.
SINGLE_CORE_FAMILIES = {"linear", "elastic_net", "cart", "neural_net"}


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
    params: tuple = ()       # hiperparámetros fijos, como pares (ruta, valor)
    neighbors: bool = False  # agrega el nivel de precio de los vecinos (NeighborhoodLevel)
    price_weight: float = 0  # peso máximo por decil de precio (PriceWeighted); 0 = sin pesos
    synthetic: bool = False  # sintéticos tipo SMOTER de la cola alta (SyntheticTail)
    impute_area: bool = False  # completa el área faltante con un modelo (ImputedArea)
    truncation: bool = False   # corrige por el truncamiento del precio (TruncationCorrection)
    upper_only: bool = False   # la corrección solo en el techo y sin soluciones en el borde
    same_building: float = 0     # fracción del ajuste con precio del mismo edificio
                                 # (SameBuilding); 0 = sin el predictor
    n_seeds: int = 1             # promedio de varias semillas (SeedAverage)
    fine_tune: bool = False      # sigue entrenando cerca de la Zona T (FocusFineTune)

    def name(self, number):
        suffix = f"_{self.variant}" if self.variant else ""
        return f"{number:02d}_{self.family}_{self.features}{suffix}"


# Hiperparámetros fijos de algunas variantes (rutas dentro de build_model).
L2_LOSS = (("regressor__lgbmregressor__objective", "l2"),)


# En el híbrido, el boosting de residuos es un pipeline simple (sin 'regressor__').
RESIDUAL_Q60 = (("lgbmregressor__objective", "quantile"), ("lgbmregressor__alpha", 0.60))
RESIDUAL_Q65 = (("lgbmregressor__objective", "quantile"), ("lgbmregressor__alpha", 0.65))


def quantile_loss(alpha):
    return (("regressor__lgbmregressor__objective", "quantile"),
            ("regressor__lgbmregressor__alpha", alpha))


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
    # Idea 4: predictores de gama alta del texto, ejes de ciudad y conteos de OSM a 300 m y 1 km.
    Spec("boosting", "rich_coords", "by_type_per_m2", target="per_m2", by_type=True),
    Spec("random_forest", "rich", "by_type_per_m2", target="per_m2", by_type=True),
    # Idea 5: hiperparámetros sintonizados con validación espacial (05_tune_models.py).
    Spec("cart", "full", "tuned", tuned=True),
    Spec("neural_net", "full", "tuned", tuned=True),
    # Ojo: 05 busca sobre log(precio); estos parámetros se copian a los dos submodelos del
    # precio por m2, así que no están sintonizados para ese objetivo.
    Spec("boosting", "full_coords", "by_type_per_m2_tuned", target="per_m2", by_type=True,
         tuned=True),
    Spec("random_forest", "full", "by_type_per_m2_tuned", target="per_m2", by_type=True,
         tuned=True),
]
# Los números 30 a 32 son ensambles de 06_ensemble.py y la versión x1,15 del envío 10.

# Ronda 2, desde el 33. Criterio: holdout nororiental (train hasta 1.000 M, ver utils).
ROUND_2_START = 33
ROUND_2 = [
    # 2b. Elastic net en área, baños y ubicación, con boosting sobre sus residuos.
    Spec("en_boosting", "full_coords", "residual"),
    # 2c. Precio en nivel con pérdida cuadrática (frente a log en 14 y L1 en 16).
    Spec("boosting", "full_coords", "by_type_level_l2", target="level", by_type=True,
         params=L2_LOSS),
    # 3. Área ampliada y señales de lujo del texto; luego, área faltante imputada con modelo.
    Spec("boosting", "lux_coords", "by_type_per_m2", target="per_m2", by_type=True),
    Spec("boosting", "lux_coords", "by_type_per_m2_imputed", target="per_m2", by_type=True,
         impute_area=True),
    # 4. Nivel de precio de los 20 vecinos más cercanos, calculado dentro de cada ajuste.
    Spec("boosting", "full_coords", "by_type_per_m2_nbr", target="per_m2", by_type=True,
         neighbors=True),
    Spec("random_forest", "full", "by_type_per_m2_nbr", target="per_m2", by_type=True,
         neighbors=True),
    # 5. Rebalanceo hacia la cola alta, sobre la mejor variante (precio en nivel, 16).
    Spec("boosting", "full_coords", "by_type_level_pw3", target="level", by_type=True,
         price_weight=3),
    Spec("random_forest", "full", "by_type_per_m2_pw3", target="per_m2", by_type=True,
         price_weight=3),
    Spec("boosting", "full_coords", "by_type_level_q55", target="level", by_type=True,
         params=quantile_loss(0.55)),
    Spec("boosting", "full_coords", "by_type_level_q60", target="level", by_type=True,
         params=quantile_loss(0.60)),
    Spec("boosting", "full_coords", "by_type_level_q65", target="level", by_type=True,
         params=quantile_loss(0.65)),
    Spec("boosting", "full_coords", "by_type_level_smoter", target="level", by_type=True,
         synthetic=True),
    # Sin OSM: en el holdout nororiental el boosting 10 sin OSM (el subido) le gana al con OSM.
    Spec("boosting", "core_coords", "by_type_level", target="level", by_type=True),
    # 5d. Corrección por truncamiento: de la mediana truncada a la mediana sin truncar.
    Spec("boosting", "full_coords", "by_type_level_trunc", target="level", by_type=True,
         truncation=True),
    Spec("random_forest", "full", "by_type_per_m2_trunc", target="per_m2", by_type=True,
         truncation=True),
    Spec("boosting", "core_coords", "by_type_level_trunc", target="level", by_type=True,
         truncation=True),
    # 6. Combinaciones de lo que funcionó: híbrido lineal + boosting, sin OSM, cuantil 0,6
    # y corrección por truncamiento.
    Spec("en_boosting", "core_coords", "residual"),
    Spec("en_boosting", "full_coords", "residual_q60", params=RESIDUAL_Q60),
    Spec("en_boosting", "core_coords", "residual_q60", params=RESIDUAL_Q60),
    Spec("en_boosting", "core_coords", "residual_trunc", truncation=True),
    Spec("boosting", "core_coords", "by_type_level_q65", target="level", by_type=True,
         params=quantile_loss(0.65)),
    Spec("en_boosting", "core_coords", "residual_q65", params=RESIDUAL_Q65),
]
# Los números 55 a 60 son ensambles (06_ensemble.py) y versiones escaladas (09).
ROUND_2B_START = 61
ROUND_2B = [
    # 7. Señales de lujo y área ampliada dentro de la parte lineal del híbrido.
    Spec("en_boosting", "lux_core_coords", "residual_q65", params=RESIDUAL_Q65),
    # 8. Corrección por truncamiento arreglada (la revisión encontró soluciones en el borde).
    Spec("boosting", "core_coords", "by_type_level_trunc_up", target="level", by_type=True,
         truncation=True, upper_only=True),
]
# El 64 no se usa. En Kaggle el 53 (cuantil 0,65) le gana al híbrido 54: se sube el cuantil.
ROUND_2C_START = 65
ROUND_2C = [
    Spec("boosting", "core_coords", "by_type_level_q70", target="level", by_type=True,
         params=quantile_loss(0.70)),
    Spec("boosting", "core_coords", "by_type_level_q75", target="level", by_type=True,
         params=quantile_loss(0.75)),
]
# Ronda 3: mejorar el orden, no el nivel. Todo sobre la receta del 65 (cuantil 0,70).
ROUND_3_START = 67
ROUND_3 = [
    # 1. Mediana del precio y del precio por m2 de los anuncios a menos de 30 m.
    Spec("boosting", "core_coords", "by_type_level_q70_bld", target="level", by_type=True,
         params=quantile_loss(0.70), same_building=1.0),
    # 2. Promedio de 10 semillas.
    Spec("boosting", "core_coords", "by_type_level_q70_s10", target="level", by_type=True,
         params=quantile_loss(0.70), n_seeds=10),
]
# El 69 no se usa: el predictor del edificio no ayudó, así que no se promedian sus semillas.
ROUND_3B_START = 70
ROUND_3B = [
    # 3. Sigue entrenando con los inmuebles a menos de 4 km de la Zona T, tasa 0,01.
    Spec("boosting", "core_coords", "by_type_level_q70_ft", target="level", by_type=True,
         params=quantile_loss(0.70), fine_tune=True),
    # 1b. El 67 con el predictor del edificio solo en el 30 % del ajuste, como en el test.
    Spec("boosting", "core_coords", "by_type_level_q70_bld30", target="level", by_type=True,
         params=quantile_loss(0.70), same_building=0.3),
]
# Número de envío -> especificación.
NUMBERED_SPECS = {**dict(enumerate(SPECS, start=1)),
                  **dict(enumerate(ROUND_2, start=ROUND_2_START)),
                  **dict(enumerate(ROUND_2B, start=ROUND_2B_START)),
                  **dict(enumerate(ROUND_2C, start=ROUND_2C_START)),
                  **dict(enumerate(ROUND_3, start=ROUND_3_START)),
                  **dict(enumerate(ROUND_3B, start=ROUND_3B_START))}


def build_spec_model(spec, params):
    """Modelo de una especificación con sus envoltorios (vecinos, pesos, sintéticos)."""
    params = {**params, **dict(spec.params)}
    # Los conjuntos 'lux' usan el área ampliada con más patrones de texto.
    surface_col = "surface_ext" if spec.features.startswith("lux") else "surface"
    if spec.family == "en_boosting":
        model = ResidualBoosting(params, luxury=spec.features.startswith("lux"))
    elif spec.fine_tune:
        model = FocusFineTune(build_model(spec.family, params, spec.target, False, surface_col))
        model = ByPropertyType(model) if spec.by_type else model
    else:
        model = build_model(spec.family, params, spec.target, spec.by_type, surface_col)
    if spec.n_seeds > 1:
        model = SeedAverage(model, spec.n_seeds)
    if spec.impute_area:
        model = ImputedArea(model, surface_col)
    if spec.synthetic:
        model = SyntheticTail(model)
    if spec.price_weight:
        model = PriceWeighted(model, spec.price_weight)
    if spec.truncation:
        model = TruncationCorrection(model, upper_only=spec.upper_only)
    if spec.neighbors:
        model = NeighborhoodLevel(model, drop_coords="coords" not in spec.features,
                                  surface_col=surface_col)
    if spec.same_building:
        model = SameBuilding(model, surface_col=surface_col, keep_share=spec.same_building)
    return model


def spec_features(spec, sets):
    """Predictores de una especificación; los vecinos necesitan lat/lon para buscarse."""
    features = sets[spec.features]
    if spec.neighbors:
        features = features + [c for c in COORD_FEATURES if c not in features]
    return features


def fit_predict(spec, params, X_fit, y_fit, X_pred, weights=None):
    """Ajusta una copia del modelo y predice."""
    model = build_spec_model(spec, params)
    fit_model(model, X_fit, y_fit, weights)
    return model.predict(X_pred)


def evaluate(spec, params, X, y, splits, weights=None):
    """Predicciones para las filas de validación de cada partición (NaN en el resto).
    Las familias de un solo núcleo ajustan sus particiones en paralelo."""
    n_jobs = len(splits) if spec.family in SINGLE_CORE_FAMILIES else 1
    fold_preds = Parallel(n_jobs=n_jobs)(
        delayed(fit_predict)(spec, params, X.iloc[fit_idx], y.iloc[fit_idx], X.iloc[val_idx],
                             None if weights is None else weights[fit_idx])
        for fit_idx, val_idx in splits)
    pred = np.full(len(X), np.nan)
    for (_, val_idx), fold_pred in zip(splits, fold_preds):
        pred[val_idx] = fold_pred
    return pred


def update_table(path, new, key):
    """Agrega columnas o filas nuevas a una tabla guardada, reemplazando las repetidas."""
    if not path.exists():
        return new
    previous = pd.read_csv(path)
    if key == "columns":
        keep = [c for c in previous.columns if c not in new.columns]
        return pd.concat([new, previous[keep]], axis=1)
    # Una fila repetida conserva los valores anteriores de los esquemas que no se corrieron.
    repeated = previous["model"].isin(new["model"])
    merged = new.set_index("model").combine_first(previous[repeated].set_index("model"))
    return pd.concat([previous[~repeated], merged.reset_index()])


def main():
    parser = argparse.ArgumentParser()
    families = FAMILIES + ["en_boosting"]  # el híbrido de 33 combina dos familias
    parser.add_argument("--families", nargs="+", default=families, choices=families)
    parser.add_argument("--specs", nargs="+", type=int, help="números de SPECS a correr")
    parser.add_argument("--schemes", nargs="+", default=list(SCHEMES), choices=list(SCHEMES),
                        help="esquemas de validación; los demás conservan su resultado")
    parser.add_argument("--no-submission", action="store_true",
                        help="no reentrena con todo el train ni escribe el envío")
    args = parser.parse_args()

    train, test = load_processed()
    y = train[TARGET]
    sets = feature_sets(train.columns)
    best_params = load_best_params()
    weights = similarity_weights(train["test_similarity"])
    schemes = {scheme: make_splits(train, test) for scheme, make_splits in SCHEMES.items()
               if scheme in args.schemes}

    oof_table = train[[ID_COL, TARGET]].copy()
    test_table = test[[ID_COL]].copy()
    for number, spec in NUMBERED_SPECS.items():
        if spec.family not in args.families or (args.specs and number not in args.specs):
            continue
        start = time.time()
        name = spec.name(number)
        features = spec_features(spec, sets)
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
        test_columns = [ID_COL]
        if not args.no_submission:
            test_pred = fit_predict(spec, params, train[features], y, test[features], w)
            test_table[name] = test_pred
            test_columns.append(name)
            row["submission"] = save_submission(test[ID_COL], test_pred, name).name
            row["test_mean"], row["test_p95"], row["test_max"] = (
                test_pred.mean(), np.quantile(test_pred, 0.95), test_pred.max())
        row["minutes"] = (time.time() - start) / 60
        print(f"{name:<40} " + " | ".join(
            f"{scheme} {row[f'mae_{scheme}']/1e6:6.1f} M (sesgo {row[f'bias_{scheme}']/1e6:+.1f})"
            for scheme in schemes) + f" | {row['minutes']:.1f} min", flush=True)
        # Se guarda tras cada especificación: si una corrida se corta, no se pierde lo hecho.
        spec_columns = [c for c in oof_table if c.startswith(f"{name}__")]
        save_results(row, oof_table[[ID_COL, TARGET] + spec_columns], test_table[test_columns])


@contextmanager
def results_lock():
    """Candado de archivo: dos corridas simultáneas no escriben las tablas a la vez."""
    lock = PROCESSED_DIR / "results.lock"
    while True:
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL)
            break
        except FileExistsError:
            # Un candado de más de 5 minutos es de una corrida que murió: se libera.
            if lock.exists() and time.time() - lock.stat().st_mtime > 300:
                lock.unlink(missing_ok=True)
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
