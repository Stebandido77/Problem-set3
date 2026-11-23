"""Compara, envío por envío, el holdout nororiental ponderado y la validación espacial con el
puntaje público de Kaggle (ronda 8). No reentrena nada: cruza views/kaggle_scores.csv con
views/cv_results.csv por el nombre del archivo enviado.

Salidas:
  views/criteria_vs_kaggle.csv          un envío por fila, con los tres números
  views/criteria_vs_kaggle_rank.csv     Spearman, brecha media y pares bien ordenados por criterio

Uso:
  python scripts/25_criteria_vs_kaggle.py
"""

from itertools import combinations

import numpy as np
import pandas as pd

from utils import VIEWS_DIR, load_kaggle_scores

CRITERIA = {"mae_weighted": "ponderado", "mae_spatial": "espacial"}


def comparison_table():
    """Envíos con puntaje público y con los dos criterios calculados, en millones."""
    scores = load_kaggle_scores()
    results = pd.read_csv(VIEWS_DIR / "cv_results.csv")
    results["file"] = results["submission"].str.removesuffix(".csv")
    merged = scores.merge(results, left_on="model", right_on="file", how="left",
                          suffixes=("", "_cv"))
    table = pd.DataFrame({"envio": merged["model"], "kaggle": merged["public_mae"] / 1e6,
                          **{label: merged[col] / 1e6 for col, label in CRITERIA.items()}})
    return table.dropna().sort_values("kaggle").reset_index(drop=True)


def ranking_quality(table):
    """Qué tan bien ordena cada criterio a los envíos frente a Kaggle."""
    rows = []
    pairs = list(combinations(range(len(table)), 2))
    for label in CRITERIA.values():
        same_order = [np.sign(table.loc[i, label] - table.loc[j, label])
                      == np.sign(table.loc[i, "kaggle"] - table.loc[j, "kaggle"])
                      for i, j in pairs]
        rows.append({"criterio": label, "envios": len(table),
                     "spearman": table[label].corr(table["kaggle"], method="spearman"),
                     "pares_bien_ordenados": np.mean(same_order),
                     "brecha_media": (table[label] - table["kaggle"]).mean(),
                     "brecha_absoluta_media": (table[label] - table["kaggle"]).abs().mean()})
    return pd.DataFrame(rows)


def main():
    table = comparison_table()
    quality = ranking_quality(table)
    table.to_csv(VIEWS_DIR / "criteria_vs_kaggle.csv", index=False)
    quality.to_csv(VIEWS_DIR / "criteria_vs_kaggle_rank.csv", index=False)
    print(table.round(1).to_string(index=False), "\n")
    print(quality.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
