"""
Analiza las predicciones out-of-fold reconstruidas por
23_rebuild_oof.py.

Entradas:
    stores/processed/oof_predictions.csv

Salidas:
    views/oof_model_metrics.csv
    views/oof_error_correlations.csv

Este script NO entrena modelos y NO genera submissions.

Uso desde la raíz del repositorio:

    python scripts/24_analyze_oof.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

OOF_PATH = (
    ROOT
    / "stores"
    / "processed"
    / "oof_predictions.csv"
)

VIEWS_DIR = ROOT / "views"

METRICS_PATH = VIEWS_DIR / "oof_model_metrics.csv"

CORRELATIONS_PATH = (
    VIEWS_DIR
    / "oof_error_correlations.csv"
)

ENSEMBLES_PATH = (
    VIEWS_DIR
    / "oof_ensemble_candidates.csv"
)

# ---------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------

ID_COL = "property_id"
TARGET = "price"

SCHEMES = [
    "random_twins",
    "spatial",
    "chapinero",
    "northeast",
]

# ---------------------------------------------------------------------
# Ensembles candidatos
# ---------------------------------------------------------------------

# No hacemos búsqueda automática de pesos.
# Estas combinaciones se definen previamente a partir de:
#   - desempeño individual,
#   - sesgo,
#   - diversidad de errores,
#   - comportamiento espacial.

ENSEMBLE_CANDIDATES = {
    "E01_45_54_w70_30": {
        "type": "weighted",
        "models": {
            "45_boosting_core_coords_by_type_level": 0.70,
            "54_en_boosting_core_coords_residual_q65": 0.30,
        },
    },

    "E02_45_77_w70_30": {
        "type": "weighted",
        "models": {
            "45_boosting_core_coords_by_type_level": 0.70,
            "77_boosting_ext_core_coords_by_type_level_q70": 0.30,
        },
    },

    "E03_45_54_77_w60_20_20": {
        "type": "weighted",
        "models": {
            "45_boosting_core_coords_by_type_level": 0.60,
            "54_en_boosting_core_coords_residual_q65": 0.20,
            "77_boosting_ext_core_coords_by_type_level_q70": 0.20,
        },
    },

    "E04_45_54_03_w70_20_10": {
        "type": "weighted",
        "models": {
            "45_boosting_core_coords_by_type_level": 0.70,
            "54_en_boosting_core_coords_residual_q65": 0.20,
            "03_linear_full": 0.10,
        },
    },

    "E05_45_77_03_w70_20_10": {
        "type": "weighted",
        "models": {
            "45_boosting_core_coords_by_type_level": 0.70,
            "77_boosting_ext_core_coords_by_type_level_q70": 0.20,
            "03_linear_full": 0.10,
        },
    },

    "E06_45_54_77_03_w60_20_10_10": {
        "type": "weighted",
        "models": {
            "45_boosting_core_coords_by_type_level": 0.60,
            "54_en_boosting_core_coords_residual_q65": 0.20,
            "77_boosting_ext_core_coords_by_type_level_q70": 0.10,
            "03_linear_full": 0.10,
        },
    },

    "E07_median_45_54_77": {
        "type": "median",
        "models": [
            "45_boosting_core_coords_by_type_level",
            "54_en_boosting_core_coords_residual_q65",
            "77_boosting_ext_core_coords_by_type_level_q70",
        ],
    },
}

# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------

def split_prediction_name(column):
    """
    Separa:

        45_boosting_...__spatial

    en:

        model  = 45_boosting_...
        scheme = spatial
    """

    if "__" not in column:
        raise ValueError(
            f"La columna no sigue el formato modelo__esquema: {column}"
        )

    model, scheme = column.rsplit("__", 1)

    return model, scheme


def prediction_columns(oof):
    """
    Devuelve únicamente las columnas que corresponden
    a predicciones OOF.
    """

    excluded = {
        ID_COL,
        TARGET,
    }

    return [
        column
        for column in oof.columns
        if column not in excluded
    ]


def validate_oof(oof):
    """
    Validaciones básicas del archivo OOF.
    """

    if ID_COL not in oof.columns:
        raise ValueError(
            f"No existe la columna {ID_COL}."
        )

    if TARGET not in oof.columns:
        raise ValueError(
            f"No existe la columna {TARGET}."
        )

    if oof.empty:
        raise ValueError(
            "oof_predictions.csv está vacío."
        )

    if oof[ID_COL].duplicated().any():
        duplicated = int(
            oof[ID_COL].duplicated().sum()
        )

        raise ValueError(
            f"Existen {duplicated} property_id duplicados."
        )

    if oof[TARGET].isna().any():
        raise ValueError(
            "La variable price contiene valores faltantes."
        )

    columns = prediction_columns(oof)

    if not columns:
        raise ValueError(
            "No se encontraron columnas de predicción OOF."
        )

    for column in columns:

        model, scheme = split_prediction_name(
            column
        )

        if scheme not in SCHEMES:
            raise ValueError(
                f"Esquema desconocido en {column}: {scheme}"
            )

        values = oof[column].dropna()

        if len(values) == 0:
            raise ValueError(
                f"La columna {column} no contiene predicciones."
            )

        if not np.isfinite(
            values.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"{column} contiene valores no finitos."
            )


# ---------------------------------------------------------------------
# Métricas por modelo y esquema
# ---------------------------------------------------------------------

def build_metrics(oof):
    """
    Calcula métricas individuales para cada combinación
    modelo-esquema.
    """

    y = oof[TARGET].to_numpy(dtype=float)

    rows = []

    for column in prediction_columns(oof):

        model, scheme = split_prediction_name(
            column
        )

        prediction = oof[column]

        mask = prediction.notna().to_numpy()

        y_valid = y[mask]

        pred_valid = prediction.loc[
            mask
        ].to_numpy(dtype=float)

        errors = pred_valid - y_valid

        absolute_errors = np.abs(errors)

        rows.append(
            {
                "model": model,
                "scheme": scheme,
                "n": int(mask.sum()),
                "coverage": float(mask.mean()),
                "mae": float(
                    absolute_errors.mean()
                ),
                "bias": float(
                    errors.mean()
                ),
                "rmse": float(
                    np.sqrt(
                        np.mean(errors ** 2)
                    )
                ),
                "overprediction_rate": float(
                    np.mean(errors > 0)
                ),
                "underprediction_rate": float(
                    np.mean(errors < 0)
                ),
                "median_error": float(
                    np.median(errors)
                ),
                "median_absolute_error": float(
                    np.median(
                        absolute_errors
                    )
                ),
            }
        )

    metrics = pd.DataFrame(rows)

    scheme_order = {
        scheme: position
        for position, scheme
        in enumerate(SCHEMES)
    }

    metrics["_scheme_order"] = (
        metrics["scheme"]
        .map(scheme_order)
    )

    metrics = (
        metrics
        .sort_values(
            [
                "_scheme_order",
                "mae",
            ]
        )
        .drop(
            columns="_scheme_order"
        )
        .reset_index(drop=True)
    )

    return metrics


# ---------------------------------------------------------------------
# Correlaciones de errores
# ---------------------------------------------------------------------

def build_error_correlations(oof):
    """
    Calcula correlaciones de errores entre modelos dentro
    de cada esquema.

    Dos modelos con correlación cercana a 1 cometen errores
    muy parecidos.

    Una correlación menor indica mayor diversidad potencial
    para construir ensembles.
    """

    y = oof[TARGET]

    rows = []

    for scheme in SCHEMES:

        scheme_columns = [
            column
            for column in prediction_columns(oof)
            if column.endswith(
                f"__{scheme}"
            )
        ]

        if len(scheme_columns) < 2:
            continue

        errors = pd.DataFrame(
            index=oof.index
        )

        model_names = {}

        for column in scheme_columns:

            model, _ = split_prediction_name(
                column
            )

            errors[model] = (
                oof[column] - y
            )

            model_names[model] = column

        correlations = errors.corr()

        models = list(
            correlations.columns
        )

        for i, model_a in enumerate(models):

            for j in range(i + 1, len(models)):

                model_b = models[j]

                pair = errors[
                    [model_a, model_b]
                ].dropna()

                if pair.empty:
                    continue

                disagreement = np.abs(
                    oof.loc[
                        pair.index,
                        model_names[model_a],
                    ]
                    - oof.loc[
                        pair.index,
                        model_names[model_b],
                    ]
                )

                rows.append(
                    {
                        "scheme": scheme,
                        "model_a": model_a,
                        "model_b": model_b,
                        "n_common": len(pair),
                        "error_correlation": float(
                            pair.corr().iloc[0, 1]
                        ),
                        "mean_prediction_disagreement": float(
                            disagreement.mean()
                        ),
                        "median_prediction_disagreement": float(
                            disagreement.median()
                        ),
                    }
                )

    correlations = pd.DataFrame(rows)

    if not correlations.empty:

        correlations = (
            correlations
            .sort_values(
                [
                    "scheme",
                    "error_correlation",
                ]
            )
            .reset_index(drop=True)
        )

    return correlations

# ---------------------------------------------------------------------
# Evaluación de ensembles
# ---------------------------------------------------------------------

def build_ensemble_candidates(oof):
    """
    Evalúa ensembles predefinidos utilizando exclusivamente
    las predicciones OOF.

    No se optimizan automáticamente los pesos.
    """

    y = oof[TARGET].to_numpy(dtype=float)

    rows = []

    for ensemble_name, config in ENSEMBLE_CANDIDATES.items():

        ensemble_type = config["type"]

        for scheme in SCHEMES:

            if ensemble_type == "weighted":

                models = list(
                    config["models"].keys()
                )

                weights = np.array(
                    list(config["models"].values()),
                    dtype=float,
                )

                if not np.isclose(weights.sum(), 1.0):
                    raise ValueError(
                        f"Los pesos de {ensemble_name} "
                        f"suman {weights.sum():.4f}."
                    )

            elif ensemble_type == "median":

                models = list(
                    config["models"]
                )

                weights = None

            else:
                raise ValueError(
                    f"Tipo de ensemble desconocido: "
                    f"{ensemble_type}"
                )

            columns = [
                f"{model}__{scheme}"
                for model in models
            ]

            missing = [
                column
                for column in columns
                if column not in oof.columns
            ]

            if missing:
                raise ValueError(
                    f"Faltan columnas para {ensemble_name} "
                    f"en {scheme}:\n{missing}"
                )

            predictions = oof[
                columns
            ].copy()

            mask = predictions.notna().all(axis=1)

            if not mask.any():
                continue

            prediction_matrix = predictions.loc[
                mask
            ].to_numpy(dtype=float)

            if ensemble_type == "weighted":

                ensemble_prediction = (
                    prediction_matrix @ weights
                )

            else:

                ensemble_prediction = np.median(
                    prediction_matrix,
                    axis=1,
                )

            y_valid = y[mask.to_numpy()]

            errors = (
                ensemble_prediction - y_valid
            )

            absolute_errors = np.abs(
                errors
            )

            rows.append(
                {
                    "ensemble": ensemble_name,
                    "type": ensemble_type,
                    "scheme": scheme,
                    "models": " + ".join(models),
                    "n": int(mask.sum()),
                    "coverage": float(
                        mask.mean()
                    ),
                    "mae": float(
                        absolute_errors.mean()
                    ),
                    "bias": float(
                        errors.mean()
                    ),
                    "rmse": float(
                        np.sqrt(
                            np.mean(errors ** 2)
                        )
                    ),
                    "overprediction_rate": float(
                        np.mean(errors > 0)
                    ),
                    "underprediction_rate": float(
                        np.mean(errors < 0)
                    ),
                    "median_absolute_error": float(
                        np.median(
                            absolute_errors
                        )
                    ),
                }
            )

    ensembles = pd.DataFrame(
        rows
    )

    scheme_order = {
        scheme: position
        for position, scheme
        in enumerate(SCHEMES)
    }

    ensembles["_scheme_order"] = (
        ensembles["scheme"]
        .map(scheme_order)
    )

    ensembles = (
        ensembles
        .sort_values(
            [
                "_scheme_order",
                "mae",
            ]
        )
        .drop(
            columns="_scheme_order"
        )
        .reset_index(drop=True)
    )

    return ensembles

# ---------------------------------------------------------------------
# Resumen en consola
# ---------------------------------------------------------------------

def print_summary(metrics, correlations):
    """
    Muestra los resultados principales de forma legible.
    """

    print()
    print("=" * 78)
    print("RESUMEN OOF POR ESQUEMA")
    print("=" * 78)

    for scheme in SCHEMES:

        subset = metrics[
            metrics["scheme"] == scheme
        ].copy()

        if subset.empty:
            continue

        print()
        print(f"[{scheme}]")

        for _, row in subset.iterrows():

            print(
                f"  {row['model']:<55} "
                f"MAE: {row['mae'] / 1e6:8.2f} M | "
                f"sesgo: {row['bias'] / 1e6:+8.2f} M | "
                f"cobertura: {row['coverage']:6.2%}"
            )

    if correlations.empty:
        return

    print()
    print("=" * 78)
    print("PARES CON MAYOR DIVERSIDAD DE ERROR")
    print("=" * 78)

    for scheme in SCHEMES:

        subset = correlations[
            correlations["scheme"] == scheme
        ].head(5)

        if subset.empty:
            continue

        print()
        print(f"[{scheme}]")

        for _, row in subset.iterrows():

            print(
                f"  {row['model_a']} "
                f"<-> {row['model_b']} | "
                f"corr: {row['error_correlation']:.4f} | "
                f"desacuerdo medio: "
                f"{row['mean_prediction_disagreement'] / 1e6:.2f} M"
            )

def print_ensemble_summary(ensembles):
    """
    Muestra los ensembles ordenados por MAE dentro
    de cada esquema.
    """

    print()
    print("=" * 78)
    print("ENSEMBLES CANDIDATOS")
    print("=" * 78)

    for scheme in SCHEMES:

        subset = ensembles[
            ensembles["scheme"] == scheme
        ]

        if subset.empty:
            continue

        print()
        print(f"[{scheme}]")

        for _, row in subset.iterrows():

            print(
                f"  {row['ensemble']:<35} "
                f"MAE: {row['mae'] / 1e6:8.2f} M | "
                f"sesgo: {row['bias'] / 1e6:+8.2f} M | "
                f"cobertura: {row['coverage']:6.2%}"
            )

# ---------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------

def main():

    print("=" * 78)
    print("ANÁLISIS DE PREDICCIONES OOF")
    print("=" * 78)

    print(f"Entrada: {OOF_PATH}")

    if not OOF_PATH.exists():
        raise FileNotFoundError(
            "No se encontró oof_predictions.csv. "
            "Ejecuta primero 23_rebuild_oof.py."
        )

    oof = pd.read_csv(
        OOF_PATH
    )

    print(
        f"Filas: {len(oof):,}"
    )

    print(
        f"Columnas: {oof.shape[1]}"
    )

    validate_oof(oof)

    metrics = build_metrics(oof)

    correlations = build_error_correlations(
        oof
    )

    ensembles = build_ensemble_candidates(
        oof
    )

    VIEWS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.to_csv(
        METRICS_PATH,
        index=False,
    )

    correlations.to_csv(
        CORRELATIONS_PATH,
        index=False,
    )

    ensembles.to_csv(
        ENSEMBLES_PATH,
        index=False,
    )

    print_summary(
        metrics,
        correlations,
    )

    print_ensemble_summary(
        ensembles
    )

    print()
    print("=" * 78)
    print("ANÁLISIS FINALIZADO")
    print("=" * 78)

    print(
        f"Métricas:\n  {METRICS_PATH}"
    )

    print(
        f"Correlaciones:\n  {CORRELATIONS_PATH}"
    )

    print(
        f"Ensembles:\n  {ENSEMBLES_PATH}"
    )


if __name__ == "__main__":
    main()