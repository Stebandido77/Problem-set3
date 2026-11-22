"""Revisión de supuestos de la ronda 8 que no necesitan un modelo nuevo de 04: corrección de
nivel por segmento (a), apilamiento (f) y el tope de 1.650 M (g), más la tabla de todas las
pruebas contra la 83. Lee las predicciones fuera de muestra de 04_train_models.py.

Toda elección se hace con validación cruzada dentro de cada esquema, para que el error
reportado sea fuera de muestra:
- espacial: se escoge con cuatro de las cinco particiones espaciales y se aplica a la quinta;
- ponderado: la validación nororiental se parte en cinco zonas (k-means sobre lat y lon); se
  escoge con cuatro y se aplica a la quinta, con los pesos del holdout ponderado.

Salidas:
  views/round8_level_by_segment.csv   (a) cuantil escogido por segmento y su error
  views/round8_stacking.csv           (f) apilamiento frente a la mezcla 70/30
  views/round8_ceiling.csv            (g) predicciones cerca del tope y holdout sin tope
  views/round8_summary.csv            todas las pruebas contra la 83

Uso:
  python scripts/24_assumptions.py            # solo lee predicciones guardadas
  python scripts/24_assumptions.py --nocap    # además reentrena la 82 sin el tope de 1.000 M
"""

import argparse
import importlib

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.linear_model import QuantileRegressor

from models import northeast_building_groups
from utils import (ID_COL, PROCESSED_DIR, SEED, TARGET, VIEWS_DIR, feature_sets, find_submission,
                   load_processed, northeast_holdout, save_submission, spatial_cv_splits,
                   weighted_mae)

anatomy = importlib.import_module("23_error_anatomy")

REFERENCE = "83_boosting_ext_core_coords_by_type_level_q70_ridge_s10"
ONE_SEED = "82_boosting_ext_core_coords_by_type_level_q70_ridge"
QUANTILES = {60: "154_boosting_ext_core_coords_by_type_level_q60_ridge",
             65: "155_boosting_ext_core_coords_by_type_level_q65_ridge",
             70: ONE_SEED,
             75: "156_boosting_ext_core_coords_by_type_level_q75_ridge",
             80: "157_boosting_ext_core_coords_by_type_level_q80_ridge"}
STACK = {"boosting": REFERENCE,
         "net": "148_quantile_net_ext_core_coords_by_type_q70_ridge_nocoords",
         "forest": "114_random_forest_ext_core_coords_by_type_level_q70_ridge_net_nocoords"}
MIX_70_30 = "150_ensemble_w70_30_2"
ROUND_8 = range(154, 166)
CEILING = 1650e6
N_ZONES = 5


def scheme_folds(train, test):
    """Filas validadas, pesos y particiones de selección (lista de pares escoger/aplicar)
    de cada esquema."""
    weights = anatomy.scheme_weights(train, test, train[TARGET])
    _, val_idx = northeast_holdout(train, test)
    zones = KMeans(N_ZONES, n_init=10, random_state=SEED).fit_predict(
        train.iloc[val_idx][["lat", "lon"]].to_numpy())
    northeast = [(val_idx[zones != k], val_idx[zones == k]) for k in range(N_ZONES)]
    spatial = [(np.setdiff1d(np.arange(len(train)), val), val)
               for _, val in spatial_cv_splits(train)]
    return {"weighted": (weights["weighted"], northeast),
            "spatial": (weights["spatial"], spatial)}


def weighted_error(y, pred, weight, rows):
    """MAE de las filas 'rows' con los pesos del esquema."""
    return np.average(np.abs(pred[rows] - y[rows]), weights=weight[rows])


def segment_labels(train, base_pred, how):
    """Segmento de cada fila: tercil del precio predicho por la 82 o estrato agrupado."""
    if how == "tercil":
        cuts = np.nanquantile(base_pred, [1 / 3, 2 / 3])
        return np.array(["bajo", "medio", "alto"])[np.digitize(base_pred, cuts)]
    stratum = train["ext_stratum"]
    return np.select([stratum <= 4, stratum == 5, stratum == 6], ["1-4", "5", "6"], "sin dato")


def level_by_segment(train, y, oof, folds):
    """(a) Cuantil por segmento escogido en validación cruzada dentro de cada esquema."""
    rows, summary = [], []
    for scheme, (weight, splits) in folds.items():
        column = {"weighted": "northeast", "spatial": "spatial"}[scheme]
        preds = {q: oof[f"{name}__{column}"].to_numpy() for q, name in QUANTILES.items()}
        for how in ["global", "tercil", "estrato"]:
            labels = (np.full(len(y), "todo") if how == "global"
                      else segment_labels(train, preds[70], how))
            final = np.full(len(y), np.nan)
            for choose, apply in splits:
                for segment in np.unique(labels[apply]):
                    pick = choose[labels[choose] == segment]
                    target = apply[labels[apply] == segment]
                    errors = {q: weighted_error(y, p, weight, pick) for q, p in preds.items()}
                    best = min(errors, key=errors.get)
                    final[target] = preds[best][target]
                    rows.append({"esquema": scheme, "segmentacion": how, "segmento": segment,
                                 "cuantil": best / 100})
            validated = np.concatenate([apply for _, apply in splits])
            summary.append({"esquema": scheme, "segmentacion": how,
                            "mae": weighted_error(y, final, weight, validated) / 1e6})
    choices = pd.DataFrame(rows).groupby(["esquema", "segmentacion", "segmento"])["cuantil"] \
        .agg(lambda s: ", ".join(f"{q:.2f}" for q in sorted(s))).reset_index()
    return pd.DataFrame(summary), choices


def stack_features(train, oof, column):
    """Entradas del segundo nivel: las tres predicciones de un esquema, el tipo y el estrato."""
    frame = pd.DataFrame({key: oof[f"{name}__{column}"] for key, name in STACK.items()})
    return frame.assign(is_house=train["is_house"], stratum=train["ext_stratum"])


# Segundo nivel: regresión de la mediana sobre las tres predicciones (pesos libres con
# intercepto) y un LightGBM pequeño con pérdida L1 que además ve el tipo y el estrato.
STACKERS = {
    "lineal_mediana": (list(STACK), lambda: QuantileRegressor(quantile=0.5, alpha=0,
                                                              solver="highs")),
    "lgbm_l1": (list(STACK) + ["is_house", "stratum"],
                lambda: LGBMRegressor(objective="l1", n_estimators=300, learning_rate=0.03,
                                      num_leaves=8, min_child_samples=100, verbose=-1,
                                      random_state=SEED)),
}
STACK_NUMBERS = {"lineal_mediana": 166, "lgbm_l1": 167}


class Stacker:
    """Un modelo de STACKERS ajustado en millones: la regresión de la mediana es un programa
    lineal y en pesos el solucionador tiene problemas numéricos."""

    def __init__(self, name):
        self.inputs, self.make = STACKERS[name]
        self.scale = pd.Series([1e6 if c in STACK else 1 for c in self.inputs],
                               index=self.inputs)

    def fit(self, X, y, weight):
        self.model = self.make().fit(X[self.inputs] / self.scale, np.asarray(y) / 1e6,
                                     sample_weight=weight / weight.mean())
        return self

    def predict(self, X):
        return self.model.predict(X[self.inputs] / self.scale) * 1e6

    def describe(self):
        if not hasattr(self.model, "coef_"):
            return ""
        weights = ", ".join(f"{k} {c:.2f}" for k, c in zip(STACK, self.model.coef_))
        return weights + f", intercepto {self.model.intercept_:.0f} M"


def stacking(train, y, oof, folds):
    """(f) Apilamiento con validación cruzada dentro de cada esquema, frente a la 83 y a la
    mezcla 70/30 (150). En el ponderado, el segundo nivel aprende dentro del holdout, así
    que también aprende a corregir su tope de 1.000 M: ver stack_transfer."""
    rows = []
    for scheme, (weight, splits) in folds.items():
        column = {"weighted": "northeast", "spatial": "spatial"}[scheme]
        X = stack_features(train, oof, column)
        validated = np.concatenate([apply for _, apply in splits])
        for name, model in [("83 sola", REFERENCE), ("mezcla 70/30", MIX_70_30)]:
            pred = oof[f"{model}__{column}"].to_numpy()
            rows.append({"esquema": scheme, "modelo": name,
                         "mae": weighted_error(y, pred, weight, validated) / 1e6})
        for name in STACKERS:
            final = np.full(len(y), np.nan)
            described = []
            for choose, apply in splits:
                choose = choose[weight[choose] > 0]
                stacker = Stacker(name).fit(X.iloc[choose], y.iloc[choose], weight[choose])
                final[apply] = stacker.predict(X.iloc[apply])
                described.append(stacker.describe())
            rows.append({"esquema": scheme, "modelo": f"{name} (dentro del esquema)",
                         "mae": weighted_error(y, final, weight, validated) / 1e6,
                         "pesos": " | ".join(described)})
    return pd.DataFrame(rows)


def stack_transfer(train, test, y, oof, folds, stack):
    """(f) El apilamiento que se puede enviar: el segundo nivel se ajusta con las
    predicciones de la validación espacial y se aplica a las del holdout nororiental (sin
    sus filas de validación en el ajuste) y al test (con todas). Escribe los envíos 166 y
    167 y sus filas en cv_results.csv."""
    weight, splits = folds["weighted"]
    val_idx = np.concatenate([apply for _, apply in splits])
    spatial = stack_features(train, oof, "spatial")
    northeast = stack_features(train, oof, "northeast")
    test_pred = pd.read_csv(PROCESSED_DIR / "test_predictions.csv")
    test_X = pd.DataFrame({key: test_pred[name] for key, name in STACK.items()}).assign(
        is_house=test["is_house"].to_numpy(), stratum=test["ext_stratum"].to_numpy())
    results_path = VIEWS_DIR / "cv_results.csv"
    results = pd.read_csv(results_path)
    rows = []
    ones = np.ones(len(train))
    for name in STACKERS:
        fit_rows = np.setdiff1d(np.arange(len(train)), val_idx)
        stacker = Stacker(name).fit(spatial.iloc[fit_rows], y.iloc[fit_rows], ones[fit_rows])
        pred = np.full(len(train), np.nan)
        pred[val_idx] = stacker.predict(northeast.iloc[val_idx])
        mae_weighted = weighted_error(y, pred, weight, val_idx) / 1e6
        final = Stacker(name).fit(spatial, y, ones)
        test_values = final.predict(test_X)
        model = f"{STACK_NUMBERS[name]}_stack_{name}_3"
        within = stack.set_index(["esquema", "modelo"])["mae"]
        mae_spatial = within[("spatial", f"{name} (dentro del esquema)")]
        file = save_submission(test[ID_COL], test_values, model).name
        rows.append({"esquema": "espacial -> ponderado", "modelo": name, "mae": mae_weighted,
                     "mae_espacial": mae_spatial,
                     "pesos": stacker.describe(), "pesos_test": final.describe(),
                     "test_media": test_values.mean() / 1e6,
                     "test_p95": np.quantile(test_values, 0.95) / 1e6,
                     "test_max": test_values.max() / 1e6, "envio": file})
        results = pd.concat([results[results["model"] != model], pd.DataFrame([{
            "model": model, "family": "stacking", "features": "+".join(STACK.values()),
            # La espacial queda vacía: la del apilamiento dentro del esquema (en
            # round8_stacking.csv) no es la del modelo enviado.
            "mae_weighted": mae_weighted * 1e6, "submission": file,
            "test_mean": test_values.mean(), "test_p95": np.quantile(test_values, 0.95),
            "test_max": test_values.max()}])])
    results.sort_values("model").to_csv(results_path, index=False)
    return pd.DataFrame(rows)


def ceiling(train, test, y, oof, nocap):
    """(g) Predicciones del test cerca del tope y efecto del tope en el holdout."""
    results = pd.read_csv(VIEWS_DIR / "cv_results.csv").set_index("model")
    rows = []
    for name in [REFERENCE, anatomy.MODELS["127"], "152_ensemble_w70_30_2"]:
        pred = pd.read_csv(find_submission(results.loc[name, "submission"]))[TARGET]
        rows.append({"medida": f"test a menos de 5 % del tope ({name.split('_')[0]})",
                     "valor": int((pred >= 0.95 * CEILING).sum()),
                     "detalle": f"de {len(pred)}; máximo {pred.max() / 1e6:.0f} M"})
    near_test = anatomy.scheme_weights(train, test, y)["weighted"] > 0
    rows.append({"medida": "train junto al test con precio de 1.567 M o más",
                 "valor": float((y[near_test] >= 0.95 * CEILING).mean()),
                 "detalle": f"{int((y[near_test] >= 0.95 * CEILING).sum())} de "
                            f"{int(near_test.sum())} filas validadas del holdout"})
    val_idx, groups = northeast_building_groups(train, test, y)
    base = oof[f"{ONE_SEED}__northeast"].to_numpy()
    below = y.iloc[val_idx].to_numpy() <= 1e9
    rows.append({"medida": "82, holdout con tope de 1.000 M: ponderado",
                 "valor": weighted_mae(y.iloc[val_idx], base[val_idx], groups) / 1e6,
                 "detalle": f"{(~below).mean():.1%} de la validación pasa de 1.000 M"})
    if nocap:
        trainer = importlib.import_module("04_train_models")
        spec = trainer.NUMBERED_SPECS[82]
        features = trainer.spec_features(spec, feature_sets(train.columns))
        split = [northeast_holdout(train, test, price_cap=None)]
        pred = trainer.evaluate(spec, {}, train[features], y, split)
        rows.append({"medida": "82, holdout sin tope (ajuste hasta 1.650 M): ponderado",
                     "valor": weighted_mae(y.iloc[val_idx], pred[val_idx], groups) / 1e6,
                     "detalle": "mismas filas de validación y pesos"})
        for label, mask in [("hasta 1.000 M", below), ("más de 1.000 M", ~below)]:
            idx = val_idx[mask]
            rows.append({"medida": f"MAE simple, validación {label}: con tope / sin tope",
                         "valor": np.abs(base[idx] - y.iloc[idx]).mean() / 1e6,
                         "detalle": f"{np.abs(pred[idx] - y.iloc[idx]).mean() / 1e6:.1f} M "
                                    f"sin tope; {mask.sum()} filas"})
    return pd.DataFrame(rows)


def summary_table(level, stack):
    """Cada prueba contra la 83 (y contra la 82, su versión de una semilla)."""
    results = pd.read_csv(VIEWS_DIR / "cv_results.csv")
    number = results["model"].str.split("_").str[0].astype(int)
    keep = results[number.isin([82, 83, *ROUND_8])]
    rows = [{"prueba": r["model"], "ponderado": r["mae_weighted"] / 1e6,
             "espacial": r["mae_spatial"] / 1e6} for _, r in keep.iterrows()]
    for how, part in level.groupby("segmentacion"):
        values = part.set_index("esquema")["mae"]
        rows.append({"prueba": f"a. cuantil por {how}", "ponderado": values["weighted"],
                     "espacial": values["spatial"]})
    within = stack[stack["esquema"] != "espacial -> ponderado"]
    for name, part in within.groupby("modelo"):
        values = part.set_index("esquema")["mae"]
        rows.append({"prueba": f"f. {name}", "ponderado": values["weighted"],
                     "espacial": values["spatial"]})
    for _, r in stack[stack["esquema"] == "espacial -> ponderado"].iterrows():
        rows.append({"prueba": f"f. {r['modelo']} enviable (ajustado en la espacial)",
                     "ponderado": r["mae"], "espacial": r["mae_espacial"]})
    out = pd.DataFrame(rows)
    ref = out.set_index("prueba").loc[REFERENCE]
    out["dif_ponderado"] = out["ponderado"] - ref["ponderado"]
    out["dif_espacial"] = out["espacial"] - ref["espacial"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nocap", action="store_true",
                        help="reentrena la 82 en el holdout nororiental sin tope de precio")
    args = parser.parse_args()
    train, test = load_processed()
    y = train[TARGET]
    oof = pd.read_csv(PROCESSED_DIR / "oof_predictions.csv")
    assert (oof["property_id"] == train["property_id"]).all()
    folds = scheme_folds(train, test)

    level, choices = level_by_segment(train, y.to_numpy(), oof, folds)
    pd.concat([level, choices]).to_csv(VIEWS_DIR / "round8_level_by_segment.csv", index=False)
    stack = stacking(train, y, oof, folds)
    transfer = stack_transfer(train, test, y, oof, folds, stack)
    stack = pd.concat([stack, transfer], ignore_index=True)
    stack.to_csv(VIEWS_DIR / "round8_stacking.csv", index=False)
    cap = ceiling(train, test, y, oof, args.nocap)
    cap.to_csv(VIEWS_DIR / "round8_ceiling.csv", index=False)
    summary = summary_table(level, stack)
    summary.to_csv(VIEWS_DIR / "round8_summary.csv", index=False)
    for table in [level, choices, stack, cap, summary]:
        print(table.round(2).to_string(index=False), "\n")


if __name__ == "__main__":
    main()
