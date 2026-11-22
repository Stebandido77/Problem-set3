"""Reglas sobre anuncios repetidos (ronda 7, piezas A2 y A3), aplicadas a las predicciones
de un modelo ya entrenado por 04_train_models.py.

- 'median' (A2): si un anuncio tiene exactamente la misma descripción (más de 30 caracteres)
  que uno o más anuncios de entrenamiento, su predicción se reemplaza por la mediana del
  precio de esos gemelos. Es una regla de vecino más cercano sobre anuncios repetidos.
- 'mix' (A2): promedio 50/50 de esa mediana y la predicción del modelo.
- 'avg' (A3): los anuncios a predecir que comparten descripción reciben el promedio de sus
  predicciones (consistencia entre gemelos).
- 'mix_avg': 'mix' y después 'avg'.

Validación: en cada partición de cada esquema, la mediana sale solo de las filas de ajuste
y el promedio de A3 solo de las filas de validación de esa partición. La validación
aleatoria simple ('random') no agrupa gemelos, así que simula el caso del test (anuncios con
gemelo en el train). En el test, la mediana sale del train completo; ningún paso usa precios
del test.

Salidas: views/twin_rules.csv (error en las filas afectadas y en total por esquema), filas
nuevas en views/cv_results.csv, columnas en oof/test_predictions.csv y un envío por regla.

Uso:
  python scripts/20_twin_rules.py --model 123_boosting_..._s10 --numbers 124 125 126 127
  python scripts/20_twin_rules.py --model 177_boosting_... --rules mix --numbers 181
"""

import argparse
import importlib

import numpy as np
import pandas as pd

from models import northeast_building_groups
from utils import (ID_COL, PROCESSED_DIR, TARGET, VIEWS_DIR, error_summary, load_processed,
                   save_submission, weighted_mae)

MIN_LENGTH = 30
MIX_WEIGHT = 0.5
RULES = ["median", "mix", "avg", "mix_avg"]


def normalized_description(df):
    """Descripción en minúsculas con los espacios colapsados; vacía si es corta."""
    text = df["desc_text"].fillna("").str.lower().str.replace(r"\s+", " ", regex=True)
    text = text.str.strip()
    return text.where(text.str.len() > MIN_LENGTH, "")


def twin_median(query, reference, reference_price):
    """Mediana del precio de los gemelos de referencia de cada fila (NaN si no tiene)."""
    medians = pd.Series(reference_price).groupby(reference.to_numpy()).median()
    medians = medians.drop("", errors="ignore")
    return query.map(medians).to_numpy(float)


def twin_average(description, pred):
    """Promedio de las predicciones de las filas con la misma descripción."""
    pred = pd.Series(np.asarray(pred, dtype=float))
    mean = pred.groupby(description.to_numpy()).transform("mean").to_numpy()
    return np.where(description.to_numpy() != "", mean, pred)


def apply_rule(rule, pred, query, reference, reference_price):
    """Predicción con la regla y máscara de filas que cambian."""
    pred = np.asarray(pred, dtype=float)
    out = pred.copy()
    if rule in ("median", "mix", "mix_avg"):
        median = twin_median(query, reference, reference_price)
        has = ~np.isnan(median)
        weight = 1.0 if rule == "median" else MIX_WEIGHT
        out[has] = weight * median[has] + (1 - weight) * pred[has]
    if rule in ("avg", "mix_avg"):
        out = twin_average(query, out)
    return out, ~np.isclose(out, pred)


def evaluate_scheme(rule, base_pred, description, y, splits):
    """Predicción fuera de muestra con la regla en todas las particiones del esquema."""
    out, changed = base_pred.copy(), np.zeros(len(y), dtype=bool)
    for fit_idx, val_idx in splits:
        new, mask = apply_rule(rule, base_pred[val_idx], description.iloc[val_idx],
                               description.iloc[fit_idx], y[fit_idx])
        out[val_idx], changed[val_idx] = new, mask
    return out, changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="nombre del modelo base en cv_results")
    parser.add_argument("--rules", nargs="+", default=RULES, choices=RULES)
    parser.add_argument("--numbers", nargs="+", type=int, required=True,
                        help="un número de envío por regla, en el orden de --rules")
    args = parser.parse_args()
    if len(args.numbers) != len(args.rules):
        parser.error("--numbers necesita un número por regla")

    trainer = importlib.import_module("04_train_models")
    train, test = load_processed()
    y = train[TARGET].to_numpy(float)
    description, test_description = normalized_description(train), normalized_description(test)
    oof = pd.read_csv(PROCESSED_DIR / "oof_predictions.csv")
    test_pred = pd.read_csv(PROCESSED_DIR / "test_predictions.csv")
    results_path = VIEWS_DIR / "cv_results.csv"
    results = pd.read_csv(results_path)
    val_idx, groups = northeast_building_groups(train, test, train[TARGET])
    schemes = [s for s in trainer.SCHEMES if f"{args.model}__{s}" in oof.columns]
    splits = {s: trainer.SCHEMES[s](train, test) for s in schemes}
    base_rest = args.model.split("_", 1)[1]

    report = []
    for rule, number in zip(args.rules, args.numbers):
        name = f"{number}_{base_rest}_twin_{rule}"
        row = {"model": name, "family": "boosting", "features": args.model,
               "tuned": False}
        for scheme in schemes:
            base = oof[f"{args.model}__{scheme}"].to_numpy(float)
            pred, changed = evaluate_scheme(rule, base, description, y, splits[scheme])
            oof[f"{name}__{scheme}"] = pred
            rows = ~np.isnan(pred)
            summary = error_summary(y[rows], pred[rows])
            row.update({f"mae_{scheme}": summary["mae"], f"bias_{scheme}": summary["bias"],
                        f"share_over_{scheme}": summary["share_overpredicted"]})
            entry = {"rule": rule, "scheme": scheme, "n_changed": int(changed.sum()),
                     "mae_changed_before": np.abs(base - y)[changed].mean(),
                     "mae_changed_after": np.abs(pred - y)[changed].mean(),
                     "mae_before": np.abs(base - y)[rows].mean(), "mae_after": summary["mae"]}
            if scheme == "northeast":
                row["mae_weighted"] = weighted_mae(y[val_idx], pred[val_idx], groups)
                entry["mae_weighted_before"] = weighted_mae(y[val_idx], base[val_idx], groups)
                entry["mae_weighted_after"] = row["mae_weighted"]
            report.append(entry)
        final, changed = apply_rule(rule, test_pred[args.model], test_description,
                                    description, y)
        test_pred[name] = final
        report.append({"rule": rule, "scheme": "test", "n_changed": int(changed.sum()),
                       "test_mean_before": test_pred[args.model].mean(),
                       "test_mean_after": final.mean()})
        row["submission"] = save_submission(test[ID_COL], final, name).name
        row["test_mean"], row["test_p95"], row["test_max"] = (
            final.mean(), np.quantile(final, 0.95), final.max())
        results = pd.concat([results[results["model"] != name], pd.DataFrame([row])])
        print(f"{name}: " + " | ".join(f"{s} {row[f'mae_{s}'] / 1e6:.1f}" for s in schemes)
              + f" | ponderado {row.get('mae_weighted', np.nan) / 1e6:.1f}"
              + f" | test cambia {int(changed.sum())}")

    oof.to_csv(PROCESSED_DIR / "oof_predictions.csv", index=False)
    test_pred.to_csv(PROCESSED_DIR / "test_predictions.csv", index=False)
    results.sort_values("model").to_csv(results_path, index=False)
    path = VIEWS_DIR / "twin_rules.csv"
    report_table = pd.DataFrame(report).assign(model=args.model)
    if path.exists():
        previous = pd.read_csv(path)
        report_table = pd.concat([previous[previous["model"] != args.model], report_table])
    report_table.to_csv(path, index=False)
    print(pd.DataFrame(report).round(0).to_string())


if __name__ == "__main__":
    main()
