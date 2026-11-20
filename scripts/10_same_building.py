"""Error en el holdout nororiental por separado para inmuebles con y sin anuncios del mismo
edificio (a menos de 30 m), y resumen de las predicciones del test.

En el test, el 24 % de los inmuebles tiene un anuncio de train del mismo edificio; en el
holdout estricto, solo el 2 %: los vecinos de cada inmueble de validación también están en la
validación. Por eso, además del holdout estricto (referencia = solo el ajuste), se reporta un
holdout transductivo: el modelo se entrena igual, pero al predecir cada inmueble de validación
su edificio se busca en el ajuste y en los demás inmuebles de validación con precio hasta el
tope del ajuste, excluyendo la propia fila, sus gemelos y los anuncios del mismo precio a
menos de 1 m. Así se imita el test, cuyos vecinos son anuncios de train truncados.
El precio de la fila evaluada nunca entra en sus predictores.

Salida: views/same_building.csv (holdout) y views/same_building_test.csv (test).

Uso:
  python 10_same_building.py --numbers 65 67 68
"""

import argparse
import importlib

import numpy as np
import pandas as pd

from models import building_count
from utils import (PROCESSED_DIR, TARGET, VIEWS_DIR, feature_sets, load_processed,
                   northeast_holdout, transductive_reference)

training = importlib.import_module("04_train_models")


def transductive_prediction(spec, X, y, fit_idx, val_idx):
    """Predicción de la validación con el edificio buscado también entre la validación."""
    model = training.build_spec_model(spec, {})
    model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
    ref_idx, own_pos = transductive_reference(y, fit_idx, val_idx)
    reference = model.reference(X.iloc[ref_idx], y.iloc[ref_idx])
    X_val = model.with_building(X.iloc[val_idx], reference,
                                own_price=y.iloc[val_idx].to_numpy(float), own_pos=own_pos)
    return model.model_.predict(X_val)


def summarize(name, view, y_true, pred, has_building):
    """MAE, mediana y sesgo del error en todo el holdout, con edificio y sin edificio."""
    rows = []
    for group, mask in [("all", np.ones_like(has_building)), ("with_building", has_building),
                        ("without_building", ~has_building)]:
        err = pred[mask] - y_true[mask]
        rows.append({"model": name, "view": view, "group": group, "n": int(mask.sum()),
                     "mae": np.abs(err).mean(), "median_ae": np.median(np.abs(err)),
                     "bias": err.mean()})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numbers", nargs="+", type=int, required=True)
    args = parser.parse_args()

    train, test = load_processed()
    y = train[TARGET]
    fit_idx, val_idx = northeast_holdout(train, test)
    oof = pd.read_csv(PROCESSED_DIR / "oof_predictions.csv")
    test_pred = pd.read_csv(PROCESSED_DIR / "test_predictions.csv")
    X_all = train[feature_sets(train.columns)["core_coords"]]
    y_val = y.iloc[val_idx].to_numpy(float)

    strict_has = building_count(X_all.iloc[val_idx], X_all.iloc[fit_idx], y.iloc[fit_idx]) > 0
    ref_idx, own_pos = transductive_reference(y, fit_idx, val_idx)
    trans_has = building_count(X_all.iloc[val_idx], X_all.iloc[ref_idx], y.iloc[ref_idx],
                               own_price=y_val, own_pos=own_pos) > 0
    rows, test_rows = [], []
    test_has = building_count(test[X_all.columns], X_all, y) > 0
    for number in args.numbers:
        spec = training.NUMBERED_SPECS[number]
        name = spec.name(number)
        strict = oof[f"{name}__northeast"].to_numpy()[val_idx]
        rows += summarize(name, "strict", y_val, strict, strict_has)
        if spec.same_building:
            trans = transductive_prediction(
                spec, train[training.spec_features(spec, feature_sets(train.columns))], y,
                fit_idx, val_idx)
        else:
            trans = strict  # sin el predictor, la referencia no cambia la predicción
        rows += summarize(name, "transductive", y_val, trans, trans_has)
        if name in test_pred:
            pred = test_pred[name].to_numpy()
            test_rows.append({"model": name, "mean": pred.mean(), "median": np.median(pred),
                              "p95": np.quantile(pred, 0.95),
                              "mean_with_building": pred[test_has].mean(),
                              "mean_without_building": pred[~test_has].mean(),
                              "share_with_building": test_has.mean()})
        print(name, "listo", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(VIEWS_DIR / "same_building.csv", index=False)
    pd.DataFrame(test_rows).to_csv(VIEWS_DIR / "same_building_test.csv", index=False)
    print(f"Holdout con edificio: estricto {strict_has.mean():.1%}, "
          f"transductivo {trans_has.mean():.1%}; test {test_has.mean():.1%}")
    pivot = table.pivot_table(index=["model", "view"], columns="group", values="mae") / 1e6
    print(pivot.round(1).to_string())


if __name__ == "__main__":
    main()
