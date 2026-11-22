"""Sondeo del leaderboard, no un modelo: multiplica por un factor la fracción más alta de las
predicciones de un envío ya hecho. Sirve para medir en Kaggle si el tramo caro de Chapinero
está subpredicho; no se debe presentar como un modelo.

El umbral es el cuantil (1 - share) de las predicciones del mismo conjunto: el del test
para el envío y el de las filas de validación de cada esquema para la validación. No usa
ningún precio.

Salidas: un envío con 'probe' en el nombre, columnas en oof/test_predictions.csv y una fila
en views/cv_results.csv (familia 'probe'); 07_results_table.py le calcula el ponderado.

Uso:
  python 30_probe_top.py --model 152_ensemble_w70_30_2 --number 189 --share 0.05 --factor 1.15
"""

import argparse

import numpy as np
import pandas as pd

from utils import ID_COL, PROCESSED_DIR, TARGET, VIEWS_DIR, error_summary, save_submission


def scale_top(pred, share, factor):
    """Multiplica por 'factor' las predicciones por encima del cuantil (1 - share)."""
    pred = pd.Series(pred, dtype=float)
    top = pred > pred.quantile(1 - share)
    return pred.where(~top, pred * factor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--number", type=int, required=True)
    parser.add_argument("--share", type=float, default=0.05)
    parser.add_argument("--factor", type=float, default=1.15)
    args = parser.parse_args()

    oof = pd.read_csv(PROCESSED_DIR / "oof_predictions.csv")
    test_pred = pd.read_csv(PROCESSED_DIR / "test_predictions.csv")
    results_path = VIEWS_DIR / "cv_results.csv"
    results = pd.read_csv(results_path)
    label = f"top{args.share * 100:.0f}_x{args.factor * 100:.0f}"
    name = f"{args.number}_probe_{args.model.split('_', 1)[1]}_{label}"
    row = {"model": name, "family": "probe", "features": args.model, "tuned": False}
    for column in [c for c in oof.columns if c.startswith(f"{args.model}__")]:
        scheme = column.split("__")[-1]
        rows = oof[column].notna()
        pred = scale_top(oof.loc[rows, column], args.share, args.factor)
        oof[f"{name}__{scheme}"] = np.nan
        oof.loc[rows, f"{name}__{scheme}"] = pred.to_numpy()
        summary = error_summary(oof.loc[rows, TARGET], pred.to_numpy())
        row.update({f"mae_{scheme}": summary["mae"], f"bias_{scheme}": summary["bias"],
                    f"share_over_{scheme}": summary["share_overpredicted"]})
    final = scale_top(test_pred[args.model], args.share, args.factor)
    test_pred[name] = final
    row["submission"] = save_submission(test_pred[ID_COL], final, name).name
    row["test_mean"], row["test_p95"], row["test_max"] = (
        final.mean(), final.quantile(0.95), final.max())
    oof.to_csv(PROCESSED_DIR / "oof_predictions.csv", index=False)
    test_pred.to_csv(PROCESSED_DIR / "test_predictions.csv", index=False)
    results = pd.concat([results[results["model"] != name], pd.DataFrame([row])])
    results.sort_values("model").to_csv(results_path, index=False)
    print(f"{name}: espacial {row.get('mae_spatial', np.nan) / 1e6:.1f} M -> "
          f"{row['submission']}")


if __name__ == "__main__":
    main()
