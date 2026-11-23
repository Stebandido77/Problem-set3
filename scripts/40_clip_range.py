"""Recorta las predicciones de un modelo ya entrenado al rango de precios del train
(300 a 1.650 M): el train se depuró a ese rango, así que se supone que el test también lo
está. El rango sale del enunciado, no de ningún precio del test ni del leaderboard.

Aplica el mismo recorte a las predicciones fuera de muestra de cada esquema y a las del
test. Salidas: un envío '<número>_<modelo>_clip', columnas en oof/test_predictions.csv y
una fila en views/cv_results.csv con la familia del modelo base.

Uso:
  python 40_clip_range.py --model 203_ensemble_w70_30_2 --number 206
"""

import argparse

import numpy as np
import pandas as pd

from utils import ID_COL, PROCESSED_DIR, TARGET, VIEWS_DIR, error_summary, save_submission

PRICE_RANGE = (300e6, 1650e6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--number", type=int, required=True)
    args = parser.parse_args()

    oof = pd.read_csv(PROCESSED_DIR / "oof_predictions.csv")
    test_pred = pd.read_csv(PROCESSED_DIR / "test_predictions.csv")
    results_path = VIEWS_DIR / "cv_results.csv"
    results = pd.read_csv(results_path)
    base = results.set_index("model").loc[args.model]
    name = f"{args.number}_{args.model.split('_', 1)[1]}_clip"
    row = {"model": name, "family": base["family"], "features": args.model,
           "tuned": bool(base["tuned"])}
    for column in [c for c in oof.columns if c.startswith(f"{args.model}__")]:
        scheme = column.split("__")[-1]
        oof[f"{name}__{scheme}"] = oof[column].clip(*PRICE_RANGE)
        rows = oof[column].notna()
        summary = error_summary(oof.loc[rows, TARGET], oof.loc[rows, f"{name}__{scheme}"])
        row.update({f"mae_{scheme}": summary["mae"], f"bias_{scheme}": summary["bias"],
                    f"share_over_{scheme}": summary["share_overpredicted"]})
    final = test_pred[args.model].clip(*PRICE_RANGE)
    test_pred[name] = final
    row["submission"] = save_submission(test_pred[ID_COL], final, name).name
    row["test_mean"], row["test_p95"], row["test_max"] = (
        final.mean(), final.quantile(0.95), final.max())
    oof.to_csv(PROCESSED_DIR / "oof_predictions.csv", index=False)
    test_pred.to_csv(PROCESSED_DIR / "test_predictions.csv", index=False)
    results = pd.concat([results[results["model"] != name], pd.DataFrame([row])])
    results.sort_values("model").to_csv(results_path, index=False)
    changed = int((test_pred[args.model] != final).sum())
    print(f"{name}: espacial {row.get('mae_spatial', np.nan) / 1e6:.1f} M, "
          f"{changed} predicciones del test recortadas -> {row['submission']}")


if __name__ == "__main__":
    main()
