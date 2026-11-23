"""
Genera nuevos candidatos de submission para Kaggle a partir
de las predicciones históricas almacenadas en:

    stores/processed/test_predictions.csv

Las combinaciones fueron definidas previamente mediante:
    - desempeño OOF,
    - correlación de errores,
    - comportamiento espacial,
    - comportamiento en Chapinero/northeast,
    - diversidad entre familias.

Este script NO entrena modelos.
Este script NO optimiza pesos contra Kaggle.

Salida:
    stores/submissions/*.csv

Uso:
    python scripts/25_new_submissions.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

TEST_PREDICTIONS_PATH = (
    ROOT
    / "stores"
    / "processed"
    / "test_predictions.csv"
)

TEST_FEATURES_PATH = (
    ROOT
    / "stores"
    / "processed"
    / "test_features.csv"
)

SUBMISSIONS_DIR = (
    ROOT
    / "stores"
    / "submissions"
)

VIEWS_DIR = ROOT / "views"

MANIFEST_PATH = (
    VIEWS_DIR
    / "new_submissions_manifest.csv"
)


# ---------------------------------------------------------------------
# Columnas
# ---------------------------------------------------------------------

ID_COL = "property_id"
PREDICTION_COL = "price"


MODELS = {
    "03": "03_linear_full",

    "45": (
        "45_boosting_core_coords_by_type_level"
    ),

    "54": (
        "54_en_boosting_core_coords_residual_q65"
    ),

    "77": (
        "77_boosting_ext_core_coords_by_type_level_q70"
    ),

    "83": (
        "83_boosting_ext_core_coords_by_type_level_q70_ridge_s10"
    ),

    "114": (
        "114_random_forest_ext_core_coords_by_type_level_q70_"
        "ridge_net_nocoords"
    ),

    "127": (
        "127_boosting_pt_core_coords_by_type_level_q70_"
        "ridge_nocoords_s10_twin_mix"
    ),

    "148": (
        "148_quantile_net_ext_core_coords_by_type_q70_"
        "ridge_nocoords"
    ),
}


# ---------------------------------------------------------------------
# Candidatos
# ---------------------------------------------------------------------

CANDIDATES = {
    "154_ensemble_w80_20_83_54": {
        "type": "weighted",
        "models": {
            "83": 0.80,
            "54": 0.20,
        },
        "hypothesis": (
            "Ancla 83 con 20% del híbrido 54 para aportar "
            "extrapolación y reducir dependencia del boosting."
        ),
    },

    "155_ensemble_w60_20_20_83_54_45": {
        "type": "weighted",
        "models": {
            "83": 0.60,
            "54": 0.20,
            "45": 0.20,
        },
        "hypothesis": (
            "Combina el ancla 83 con el mejor modelo northeast "
            "entre las OOF reconstruidas y el modelo con mejor "
            "desempeño spatial."
        ),
    },

    "156_ensemble_w70_20_10_83_54_03": {
        "type": "weighted",
        "models": {
            "83": 0.70,
            "54": 0.20,
            "03": 0.10,
        },
        "hypothesis": (
            "Agrega 10% de regresión lineal, cuya correlación "
            "de error fue considerablemente menor y permite "
            "extrapolación fuera del soporte de los árboles."
        ),
    },

    "157_ensemble_w70_15_15_83_54_77": {
        "type": "weighted",
        "models": {
            "83": 0.70,
            "54": 0.15,
            "77": 0.15,
        },
        "hypothesis": (
            "Ancla 83 más señales complementarias: 54 fue fuerte "
            "en northeast y 77 fue fuerte en Chapinero."
        ),
    },

    "158_ensemble_w60_20_20_83_127_148": {
        "type": "weighted",
        "models": {
            "83": 0.60,
            "127": 0.20,
            "148": 0.20,
        },
        "hypothesis": (
            "Combina el ancla 83 con una variante de gemelos "
            "y la red cuantílica extrapoladora 148."
        ),
    },

    "159_ensemble_mean_83_127_148": {
        "type": "mean",
        "models": [
            "83",
            "127",
            "148",
        ],
        "hypothesis": (
            "Promedio simple de tres familias predictivas "
            "diferenciadas, sin ajuste de pesos."
        ),
    },

    "160_ensemble_median_83_127_148": {
        "type": "median",
        "models": [
            "83",
            "127",
            "148",
        ],
        "hypothesis": (
            "Mediana robusta entre 83, 127 y 148 para reducir "
            "la influencia de predicciones extremas."
        ),
    },

    "161_ensemble_w70_15_15_83_114_148": {
        "type": "weighted",
        "models": {
            "83": 0.70,
            "114": 0.15,
            "148": 0.15,
        },
        "hypothesis": (
            "Combina boosting, Random Forest y red cuantílica "
            "para maximizar diversidad estructural manteniendo "
            "al modelo 83 como ancla."
        ),
    },
}


# ---------------------------------------------------------------------
# Validaciones
# ---------------------------------------------------------------------

def validate_inputs(predictions, test_features):
    """
    Comprueba integridad y alineación entre archivos.
    """

    if ID_COL not in predictions.columns:
        raise ValueError(
            f"Falta {ID_COL} en test_predictions.csv."
        )

    if ID_COL not in test_features.columns:
        raise ValueError(
            f"Falta {ID_COL} en test_features.csv."
        )

    if len(predictions) != len(test_features):
        raise ValueError(
            "test_predictions y test_features "
            "tienen diferente número de filas."
        )

    if not predictions[ID_COL].equals(
        test_features[ID_COL]
    ):
        raise ValueError(
            "El orden de property_id no coincide entre "
            "test_predictions y test_features."
        )

    if predictions[ID_COL].duplicated().any():
        raise ValueError(
            "Existen property_id duplicados."
        )

    missing_models = [
        column
        for column in MODELS.values()
        if column not in predictions.columns
    ]

    if missing_models:
        raise ValueError(
            "Faltan las siguientes predicciones:\n"
            + "\n".join(missing_models)
        )


def validate_submission(submission, expected_ids):
    """
    Comprueba que un archivo tenga formato válido para Kaggle.
    """

    if list(submission.columns) != [
        ID_COL,
        PREDICTION_COL,
    ]:
        raise ValueError(
            "La submission debe contener únicamente "
            "property_id y price."
        )

    if len(submission) != len(expected_ids):
        raise ValueError(
            "Número incorrecto de filas."
        )

    if not submission[ID_COL].equals(
        expected_ids
    ):
        raise ValueError(
            "Los property_id no coinciden con el test."
        )

    if submission[ID_COL].duplicated().any():
        raise ValueError(
            "La submission contiene IDs duplicados."
        )

    values = submission[
        PREDICTION_COL
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "La submission contiene NaN o infinito."
        )

    if (values <= 0).any():
        raise ValueError(
            "La submission contiene precios no positivos."
        )


# ---------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------

def model_prediction(predictions, model_id):
    """
    Obtiene la predicción de un modelo a partir de su ID.
    """

    column = MODELS[model_id]

    return predictions[
        column
    ].to_numpy(dtype=float)


def build_candidate(predictions, config):
    """
    Construye el vector de predicción de un candidato.
    """

    candidate_type = config["type"]

    if candidate_type == "weighted":

        models = config["models"]

        weights = np.array(
            list(models.values()),
            dtype=float,
        )

        if not np.isclose(
            weights.sum(),
            1.0,
        ):
            raise ValueError(
                "Los pesos no suman 1."
            )

        matrix = np.column_stack([
            model_prediction(
                predictions,
                model_id,
            )
            for model_id in models
        ])

        return matrix @ weights

    if candidate_type in {
        "mean",
        "median",
    }:

        models = config["models"]

        matrix = np.column_stack([
            model_prediction(
                predictions,
                model_id,
            )
            for model_id in models
        ])

        if candidate_type == "mean":
            return matrix.mean(axis=1)

        return np.median(
            matrix,
            axis=1,
        )

    raise ValueError(
        f"Tipo de candidato desconocido: {candidate_type}"
    )


# ---------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------

def main():

    print("=" * 78)
    print("GENERACIÓN DE NUEVOS SUBMISSIONS")
    print("=" * 78)

    print(
        f"Predicciones:\n  {TEST_PREDICTIONS_PATH}"
    )

    print(
        f"Características test:\n  {TEST_FEATURES_PATH}"
    )

    predictions = pd.read_csv(
        TEST_PREDICTIONS_PATH
    )

    test_features = pd.read_csv(
        TEST_FEATURES_PATH
    )

    validate_inputs(
        predictions,
        test_features,
    )

    print()
    print(
        f"Filas test: {len(predictions):,}"
    )

    SUBMISSIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VIEWS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_rows = []

    for candidate_name, config in CANDIDATES.items():

        predicted_price = build_candidate(
            predictions,
            config,
        )

        submission = pd.DataFrame({
            ID_COL: predictions[ID_COL],
            PREDICTION_COL: predicted_price,
        })

        validate_submission(
            submission,
            predictions[ID_COL],
        )

        output_path = (
            SUBMISSIONS_DIR
            / f"{candidate_name}.csv"
        )

        submission.to_csv(
            output_path,
            index=False,
        )

        manifest_rows.append(
            {
                "candidate": candidate_name,
                "type": config["type"],
                "hypothesis": config["hypothesis"],
                "n": len(submission),
                "prediction_mean": float(
                    predicted_price.mean()
                ),
                "prediction_median": float(
                    np.median(predicted_price)
                ),
                "prediction_min": float(
                    predicted_price.min()
                ),
                "prediction_max": float(
                    predicted_price.max()
                ),
                "n_above_1650m": int(
                    np.sum(
                        predicted_price
                        > 1_650_000_000
                    )
                ),
                "file": str(
                    output_path.relative_to(ROOT)
                ),
            }
        )

        print(
            f"{candidate_name:<45} "
            f"media: {predicted_price.mean() / 1e6:8.2f} M | "
            f"máx: {predicted_price.max() / 1e6:8.2f} M | "
            f">1650M: "
            f"{np.sum(predicted_price > 1_650_000_000):4d}"
        )

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
    )

    print()
    print("=" * 78)
    print("GENERACIÓN FINALIZADA")
    print("=" * 78)

    print(
        f"Submissions generadas: {len(manifest)}"
    )

    print(
        f"Carpeta:\n  {SUBMISSIONS_DIR}"
    )

    print(
        f"Manifiesto:\n  {MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()