"""Tabla de la ronda 10: monotonía, hojas lineales, kNN y mezclas contra el boosting 127 y
la mezcla 152. No reentrena nada: lee views/cv_results.csv (con el ponderado que recalcula
07_results_table.py) y las predicciones del test de stores/processed.

Para cada fila: MAE espacial (el criterio), ponderado, media, mínimo y máximo en test,
cuántas predicciones del test pasan de 1.650 M (el techo del train), la correlación en test
con el boosting 127, la red 148 y el kNN 185, y la diferencia con la 127 y con la 152.

Salida:
  views/round10_table.csv

Uso (las mezclas se arman antes con 06_ensemble.py):
  python 29_round10_table.py --models 127 152 177 178 179 180 181 182 183 184 185 186 187 188
"""

import argparse

import pandas as pd

from utils import PROCESSED_DIR, VIEWS_DIR

TRAIN_CEILING = 1.65e9
CORRELATED_WITH = {"127": "127_", "148": "148_", "185": "185_"}
REFERENCES = ["127", "152"]


def full_names(numbers, names):
    """Nombre completo de cada fila a partir de su número."""
    by_number = {name.split("_")[0]: name for name in names}
    missing = [n for n in numbers if str(n) not in by_number]
    if missing:
        raise SystemExit(f"Sin fila en cv_results.csv: {missing}")
    return [by_number[str(n)] for n in numbers]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", type=int, required=True)
    args = parser.parse_args()
    results = pd.read_csv(VIEWS_DIR / "cv_results.csv").set_index("model")
    test = pd.read_csv(PROCESSED_DIR / "test_predictions.csv")
    anchors = {label: next((c for c in test.columns if c.startswith(prefix)), None)
               for label, prefix in CORRELATED_WITH.items()}
    rows = []
    for name in full_names(args.models, results.index):
        row, pred = results.loc[name], test[name]
        rows.append({"numero": name.split("_")[0], "modelo": name,
                     "espacial": row["mae_spatial"] / 1e6,
                     "ponderado": row["mae_weighted"] / 1e6,
                     "media_test": pred.mean() / 1e6, "min_test": pred.min() / 1e6,
                     "max_test": pred.max() / 1e6,
                     "n_sobre_1650": int((pred > TRAIN_CEILING).sum()),
                     **{f"corr_{label}": pred.corr(test[anchor])
                        for label, anchor in anchors.items() if anchor}})
    table = pd.DataFrame(rows)
    for label in [r for r in REFERENCES if r in set(table["numero"])]:
        ref = table.set_index("numero").loc[label]
        table[f"dif_espacial_{label}"] = table["espacial"] - ref["espacial"]
        table[f"dif_ponderado_{label}"] = table["ponderado"] - ref["ponderado"]
    table.to_csv(VIEWS_DIR / "round10_table.csv", index=False)
    print(table.drop(columns="modelo").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
