"""Versiones escaladas de un envío (x1,10, x1,15...): calibración de nivel con el leaderboard.

No es un modelo nuevo: multiplica un envío ya escrito por un factor fijo. Sirve para medir
cuánto del error en Kaggle es de nivel y cuánto de orden. Por eso estos envíos no van en la
tabla principal (views/cv_results.csv) sino en views/level_calibration.csv, junto con el MAE
que tendría el mismo factor en el holdout nororiental si la especificación está guardada.

Uso:
  python 09_level_calibration.py --number 57 --factors 1.10 1.15 1.20 \
      --submission 54_en_boosting_core_coords_residual_q65
  python 09_level_calibration.py --number 60 --factors 1.20 --model none \
      --submission 10_boosting_full_coords
"""

import argparse
import re

import numpy as np
import pandas as pd

from utils import (ID_COL, PROCESSED_DIR, TARGET, VIEWS_DIR, error_summary, find_submission,
                   save_submission)


def holdout_summary(model, factor):
    """MAE y sesgo del modelo escalado en el holdout nororiental (NaN si no está guardado)."""
    oof = pd.read_csv(PROCESSED_DIR / "oof_predictions.csv")
    column = f"{model}__northeast"
    if model == "none" or column not in oof:
        return {"mae": np.nan, "bias": np.nan}
    rows = oof[column].notna()
    return error_summary(oof.loc[rows, TARGET], factor * oof.loc[rows, column])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True, help="envío base, sin .csv")
    parser.add_argument("--model", help="especificación en oof_predictions (por defecto, la "
                                        "del envío sin sufijo _vN; 'none' si el envío base "
                                        "no corresponde a la versión guardada)")
    parser.add_argument("--factors", nargs="+", type=float, required=True)
    parser.add_argument("--number", type=int, required=True, help="primer número de envío")
    args = parser.parse_args()

    base = pd.read_csv(find_submission(f"{args.submission}.csv"))
    model = args.model or re.sub(r"_v\d+$", "", args.submission)
    stem = args.submission.split("_", 1)[1]
    rows = []
    for number, factor in enumerate(args.factors, start=args.number):
        name = f"{number:02d}_{stem}_x{round(factor * 100):d}"
        path = save_submission(base[ID_COL], factor * base[TARGET], name)
        summary = holdout_summary(model, factor)
        rows.append({"submission": path.name, "base": args.submission, "factor": factor,
                     "test_mean": factor * base[TARGET].mean(),
                     "mae_northeast": summary["mae"], "bias_northeast": summary["bias"]})
        print(f"{path.name}: media {rows[-1]['test_mean'] / 1e6:.0f} M | holdout "
              f"{summary['mae'] / 1e6:.1f} M")
    out = VIEWS_DIR / "level_calibration.csv"
    table = pd.DataFrame(rows)
    if out.exists():
        previous = pd.read_csv(out)
        table = pd.concat([previous[~previous["submission"].isin(table["submission"])], table])
    table.to_csv(out, index=False)


if __name__ == "__main__":
    main()
