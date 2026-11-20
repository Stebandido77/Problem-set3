"""Qué pesa en los predictores del texto completo (filas 72 a 75 de 04_train_models.py).

Ajusta los envoltorios de texto con todo el train, igual que el envío final, y reporta:
  views/text_ridge_terms.csv     los 40 términos con coeficiente más alto y más bajo del ridge
                                 del log del precio sobre el TF-IDF (fila 75)
  views/text_svd_importance.csv  importancia (ganancia) de cada componente de la SVD en el
                                 boosting de apartamentos, con sus términos de mayor carga

Uso: python 12_text_report.py --svd 50
"""

import argparse
import importlib

import numpy as np
import pandas as pd

from utils import TARGET, VIEWS_DIR, feature_sets, load_processed

training = importlib.import_module("04_train_models")
SVD_SPECS = {20: 72, 50: 73, 100: 74}
RIDGE_SPEC = 75


def fitted(number, train):
    """Envoltorio de texto de una especificación ajustado con todo el train."""
    spec = training.NUMBERED_SPECS[number]
    features = training.spec_features(spec, feature_sets(train.columns))
    return training.build_spec_model(spec, {}).fit(train[features], train[TARGET])


def ridge_terms(model, n_terms=40):
    """Términos con mayor y menor coeficiente en el ridge del texto."""
    terms = model.tfidf_.get_feature_names_out()
    coef = model.ridge_.coef_
    order = np.argsort(coef)
    rows = [("sube", terms[i], coef[i]) for i in order[::-1][:n_terms]]
    rows += [("baja", terms[i], coef[i]) for i in order[:n_terms]]
    return pd.DataFrame(rows, columns=["direction", "term", "coefficient"])


def apartment_gain(model):
    """Participación de cada predictor en la ganancia del boosting de apartamentos."""
    apartments = model.model_.models_[0].regressor_.steps[-1][1]
    gain = pd.Series(apartments.booster_.feature_importance("gain"),
                     index=apartments.booster_.feature_name())
    return gain / gain.sum()


def svd_importance(model, n_terms=8):
    """Ganancia de cada componente en el boosting de apartamentos y sus términos principales."""
    gain = apartment_gain(model)
    terms = model.tfidf_.get_feature_names_out()
    rows = []
    for k, loadings in enumerate(model.svd_.components_):
        top = np.argsort(np.abs(loadings))[::-1][:n_terms]
        rows.append({"component": f"txt_svd_{k}", "gain_share": gain[f"txt_svd_{k}"],
                     "top_terms": ", ".join(f"{terms[i]} ({loadings[i]:+.2f})" for i in top)})
    return pd.DataFrame(rows).sort_values("gain_share", ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--svd", type=int, default=50, choices=list(SVD_SPECS))
    args = parser.parse_args()
    train, _ = load_processed()
    ridge = fitted(RIDGE_SPEC, train)
    terms = ridge_terms(ridge)
    print("Ganancia en el boosting de apartamentos (fila 75):")
    print(apartment_gain(ridge).sort_values(ascending=False).head(8).round(3).to_string())
    terms.to_csv(VIEWS_DIR / "text_ridge_terms.csv", index=False)
    importance = svd_importance(fitted(SVD_SPECS[args.svd], train))
    importance.to_csv(VIEWS_DIR / "text_svd_importance.csv", index=False)
    print(terms.groupby("direction").head(15).to_string(index=False))
    print(importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
