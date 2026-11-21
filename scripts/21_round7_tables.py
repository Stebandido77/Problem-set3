"""Tablas de la ronda 7: cada modelo contra la referencia, en total y en las filas que toca
cada idea. Lee las predicciones guardadas por 04_train_models.py; no reentrena nada.

- Resumen (views/round7_summary.csv): media, mediana y p95 del envío en test, MAE ponderado
  y espacial, validación aleatoria con gemelos.
- Filas afectadas (views/round7_subsets.csv), en el holdout nororiental y en la validación
  espacial: inmuebles con área imputada del mismo punto (A1) e inmuebles con y sin un anuncio
  de ajuste a menos de 30 m (A4).
- Por decil de precio (views/round7_deciles.csv, pieza C), en el holdout nororiental: MAE,
  sesgo y porcentaje sobrepredicho.

Uso:
  python scripts/21_round7_tables.py --models 82 120 121 122
"""

import argparse
import importlib

import numpy as np
import pandas as pd

from utils import (PROCESSED_DIR, TARGET, VIEWS_DIR, distance_to_nearest_m, find_submission,
                   load_processed)

NEAR_M = 30


def full_names(numbers, results):
    """Nombre completo de cada modelo a partir de su número."""
    prefixes = results["model"].str.split("_").str[0]
    return [results.loc[prefixes == str(n), "model"].iloc[0] for n in numbers]


def summary_rows(names, results):
    rows = []
    for name in names:
        row = results.set_index("model").loc[name]
        test = pd.read_csv(find_submission(row["submission"]))[TARGET]
        rows.append({"model": name, "test_mean": test.mean() / 1e6,
                     "test_median": test.median() / 1e6,
                     "test_p95": test.quantile(0.95) / 1e6,
                     "test_at_floor": int((test <= 300e6 + 1).sum()),
                     "test_at_ceiling": int((test >= 1650e6 - 1).sum()),
                     "test_min": test.min() / 1e6, "test_max": test.max() / 1e6,
                     "mae_weighted": row["mae_weighted"] / 1e6,
                     "mae_spatial": row["mae_spatial"] / 1e6,
                     "mae_random_twins": row.get("mae_random_twins", np.nan) / 1e6})
    return pd.DataFrame(rows)


def subset_masks(train, fit_idx, val_idx):
    """Filas de validación que toca cada idea, dentro de una partición."""
    val = train.iloc[val_idx]
    near = distance_to_nearest_m(val, train.iloc[fit_idx]) < NEAR_M
    return {"area_imputed_same_point": val["surface_pt_imputed"].to_numpy() == 1,
            f"near_{NEAR_M}m": near, f"far_{NEAR_M}m": ~near}


def subset_rows(names, train, test, oof):
    trainer = importlib.import_module("04_train_models")
    y = train[TARGET].to_numpy(float)
    rows = []
    for scheme in ["northeast", "spatial"]:
        splits = trainer.SCHEMES[scheme](train, test)
        masks = {}
        for fit_idx, val_idx in splits:
            for key, mask in subset_masks(train, fit_idx, val_idx).items():
                masks.setdefault(key, []).append(val_idx[mask])
        for key, parts in masks.items():
            idx = np.concatenate(parts)
            for name in names:
                pred = oof[f"{name}__{scheme}"].to_numpy(float)[idx]
                rows.append({"scheme": scheme, "subset": key, "n": len(idx), "model": name,
                             "mae": np.abs(pred - y[idx]).mean() / 1e6,
                             "bias": (pred - y[idx]).mean() / 1e6})
    return pd.DataFrame(rows)


def decile_rows(names, train, oof):
    """Error por decil del precio real en el holdout nororiental."""
    y = train[TARGET].to_numpy(float)
    rows = []
    for name in names:
        pred = oof[f"{name}__northeast"].to_numpy(float)
        val = ~np.isnan(pred)
        decile = pd.qcut(y[val], 10, labels=False)
        err = pred[val] - y[val]
        for d in range(10):
            e = err[decile == d]
            rows.append({"model": name, "decile": d + 1, "mae": np.abs(e).mean() / 1e6,
                         "bias": e.mean() / 1e6, "share_over": (e > 0).mean()})
        top = decile >= 7
        rows.append({"model": name, "decile": "top30", "mae": np.abs(err[top]).mean() / 1e6,
                     "bias": err[top].mean() / 1e6, "share_over": (err[top] > 0).mean(),
                     "spearman_top30": pd.Series(pred[val][top]).corr(
                         pd.Series(y[val][top]), method="spearman")})
    return pd.DataFrame(rows)


def save(table, file):
    path = VIEWS_DIR / file
    if path.exists():
        previous = pd.read_csv(path)
        table = pd.concat([previous[~previous["model"].isin(table["model"])], table])
    table.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", type=int, required=True)
    parser.add_argument("--subsets", action="store_true", help="error en filas afectadas")
    parser.add_argument("--deciles", action="store_true", help="error por decil (pieza C)")
    args = parser.parse_args()

    results = pd.read_csv(VIEWS_DIR / "cv_results.csv")
    names = full_names(args.models, results)
    summary = summary_rows(names, results)
    save(summary, "round7_summary.csv")
    print(summary.round(1).to_string())
    if args.subsets or args.deciles:
        train, test = load_processed()
        oof = pd.read_csv(PROCESSED_DIR / "oof_predictions.csv")
    if args.subsets:
        subsets = subset_rows(names, train, test, oof)
        save(subsets, "round7_subsets.csv")
        print(subsets.pivot_table(index=["scheme", "subset", "n"], columns="model",
                                  values="mae").round(1).to_string())
    if args.deciles:
        deciles = decile_rows(names, train, oof)
        save(deciles, "round7_deciles.csv")
        print(deciles.round(3).to_string())


if __name__ == "__main__":
    main()
