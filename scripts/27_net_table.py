"""Tabla de la ronda 9: redes y mezclas contra la red 148 y la mezcla 152. No reentrena
nada: lee views/cv_results.csv y las predicciones del test de stores/processed.

Para cada fila: MAE espacial (el criterio), ponderado, media en test, correlación en test con
la red 148 y con el boosting 127 y, para las mezclas, la correlación entre sus componentes.

Salida:
  views/round9_nets.csv

Uso (las mezclas se arman antes con 06_ensemble.py):
  python 27_net_table.py --models 148 168 169 170 171 152
"""

import argparse

import numpy as np
import pandas as pd

from utils import PROCESSED_DIR, VIEWS_DIR

NET = "148_quantile_net_ext_core_coords_by_type_q70_ridge_nocoords"
BOOSTING = "127_boosting_pt_core_coords_by_type_level_q70_ridge_nocoords_s10_twin_mix"
REFERENCES = {"148": NET, "152": "152_ensemble_w70_30_2"}


def full_names(numbers, results):
    """Nombre completo de cada fila a partir de su número."""
    prefixes = results["model"].str.split("_").str[0]
    missing = [n for n in numbers if str(n) not in set(prefixes)]
    if missing:
        raise SystemExit(f"Sin fila en cv_results.csv: {missing}")
    return [results.loc[prefixes == str(n), "model"].iloc[0] for n in numbers]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", type=int, required=True)
    args = parser.parse_args()
    results = pd.read_csv(VIEWS_DIR / "cv_results.csv").set_index("model")
    test = pd.read_csv(PROCESSED_DIR / "test_predictions.csv")
    rows = []
    for name in full_names(args.models, results.reset_index()):
        row = results.loc[name]
        components = (row["features"].split("+") if row["family"] == "ensemble" else [])
        pairs = [test[a].corr(test[b]) for i, a in enumerate(components)
                 for b in components[i + 1:]]
        rows.append({"modelo": name, "espacial": row["mae_spatial"] / 1e6,
                     "ponderado": row["mae_weighted"] / 1e6,
                     "media_test": test[name].mean() / 1e6,
                     "corr_148": test[name].corr(test[NET]),
                     "corr_127": test[name].corr(test[BOOSTING]),
                     "corr_componentes": np.mean(pairs) if pairs else np.nan})
    table = pd.DataFrame(rows)
    for label, name in REFERENCES.items():
        ref = table.set_index("modelo").loc[name] if name in set(table["modelo"]) else None
        if ref is not None:
            table[f"dif_espacial_{label}"] = table["espacial"] - ref["espacial"]
            table[f"dif_ponderado_{label}"] = table["ponderado"] - ref["ponderado"]
    table.to_csv(VIEWS_DIR / "round9_nets.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
