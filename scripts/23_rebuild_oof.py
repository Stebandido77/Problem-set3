"""
Reconstruye selectivamente las predicciones out-of-fold (OOF) necesarias
para analizar diversidad de errores y construir nuevos ensambles.

Este script reutiliza directamente:

    scripts/04_train_models.py
    scripts/models.py
    scripts/utils.py

No vuelve a entrenar todas las especificaciones históricas y no genera
submissions. Su única salida es:

    stores/processed/oof_predictions.csv

Uso desde la raíz del repositorio:

    python scripts/23_rebuild_oof.py

Ejecución rápida para comprobar que todo funciona:

    python scripts/23_rebuild_oof.py --specs 3 --schemes northeast

Ejecución personalizada:

    python scripts/23_rebuild_oof.py \
        --specs 3 45 54 65 77 \
        --schemes random_twins spatial chapinero northeast
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Rutas del repositorio
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
TRAIN_SCRIPT = SCRIPTS_DIR / "04_train_models.py"
OOF_PATH = ROOT / "stores" / "processed" / "oof_predictions.csv"

# ---------------------------------------------------------------------
# Funciones originales de validación del proyecto
# ---------------------------------------------------------------------

# Agregamos scripts/ al path para reutilizar directamente utils.py.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from utils import (
    chapinero_holdout,
    northeast_holdout,
    random_cv,
    random_twin_splits,
    spatial_cv_splits,
)

# ---------------------------------------------------------------------
# Configuración predeterminada
# ---------------------------------------------------------------------

DEFAULT_SPECS = [3, 45, 54, 65, 77]

DEFAULT_SCHEMES = [
    "random_twins",
    "spatial",
    "chapinero",
    "northeast",
]


# ---------------------------------------------------------------------
# Carga de 04_train_models.py
# ---------------------------------------------------------------------

def load_training_module():
    """
    Carga 04_train_models.py como módulo de Python.

    No se puede importar mediante:

        import 04_train_models

    porque un módulo de Python no puede comenzar con un número.
    Por eso se utiliza importlib.
    """

    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(
            f"No se encontró el script de entrenamiento:\n{TRAIN_SCRIPT}"
        )

    module_name = "train_models_module"

    spec = importlib.util.spec_from_file_location(
        module_name,
        TRAIN_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"No fue posible cargar el módulo desde:\n{TRAIN_SCRIPT}"
        )

    module = importlib.util.module_from_spec(spec)

    # Es necesario registrarlo antes de ejecutarlo porque 04_train_models.py
    # contiene una dataclass llamada Spec.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


# ---------------------------------------------------------------------
# Validaciones
# ---------------------------------------------------------------------

def validate_requested_specs(train_module, numbers):
    """
    Verifica que todos los números solicitados existan en NUMBERED_SPECS.
    """

    missing = [
        number
        for number in numbers
        if number not in train_module.NUMBERED_SPECS
    ]

    if missing:
        available = sorted(train_module.NUMBERED_SPECS)

        raise ValueError(
            "Las siguientes especificaciones no existen: "
            f"{missing}\n"
            f"Rango disponible: {available[0]} a {available[-1]}"
        )


def validate_requested_schemes(train_module, schemes):
    """
    Verifica que los esquemas solicitados existan en SCHEMES.
    """

    missing = [
        scheme
        for scheme in schemes
        if scheme not in train_module.SCHEMES
    ]

    if missing:
        raise ValueError(
            "Los siguientes esquemas no existen: "
            f"{missing}\n"
            "Esquemas disponibles: "
            f"{list(train_module.SCHEMES)}"
        )


def validate_oof_table(oof_table, train_module, selected_columns):
    """
    Comprueba la integridad básica de la tabla OOF.
    """

    id_col = train_module.ID_COL
    target = train_module.TARGET

    if len(oof_table) == 0:
        raise ValueError("La tabla OOF quedó vacía.")

    if id_col not in oof_table.columns:
        raise ValueError(
            f"La tabla OOF no contiene la columna {id_col}."
        )

    if target not in oof_table.columns:
        raise ValueError(
            f"La tabla OOF no contiene la columna {target}."
        )

    if oof_table[id_col].duplicated().any():
        duplicated = int(oof_table[id_col].duplicated().sum())

        raise ValueError(
            "La tabla OOF contiene "
            f"{duplicated} property_id duplicados."
        )

    if oof_table[target].isna().any():
        raise ValueError(
            "La variable objetivo contiene valores faltantes."
        )

    for column in selected_columns:
        if column not in oof_table.columns:
            raise ValueError(
                f"No se generó la columna esperada: {column}"
            )

        values = oof_table[column]
        finite = np.isfinite(values.dropna().to_numpy(dtype=float))

        if not finite.all():
            raise ValueError(
                f"La columna {column} contiene valores infinitos."
            )

        if (values.dropna() <= 0).any():
            raise ValueError(
                f"La columna {column} contiene precios no positivos."
            )


# ---------------------------------------------------------------------
# Escritura incremental
# ---------------------------------------------------------------------

def save_oof(train_module, new_table):
    """
    Guarda las columnas nuevas y conserva las columnas anteriores.

    Si oof_predictions.csv ya existe, update_table reemplaza las columnas
    repetidas y mantiene los demás modelos previamente reconstruidos.
    """

    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)

    combined = train_module.update_table(
        OOF_PATH,
        new_table,
        "columns",
    )

    combined.to_csv(OOF_PATH, index=False)

    return combined


# ---------------------------------------------------------------------
# Diagnóstico de una predicción OOF
# ---------------------------------------------------------------------

def prediction_summary(y_true, prediction):
    """
    Calcula cobertura, MAE y sesgo sobre las filas que tienen predicción.
    """

    prediction = np.asarray(prediction, dtype=float)
    available = ~np.isnan(prediction)

    if not available.any():
        return {
            "n_predicted": 0,
            "coverage": 0.0,
            "mae": np.nan,
            "bias": np.nan,
        }

    errors = prediction[available] - np.asarray(
        y_true,
        dtype=float,
    )[available]

    return {
        "n_predicted": int(available.sum()),
        "coverage": float(available.mean()),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
    }

# ---------------------------------------------------------------------
# Evaluación compatible con módulos cargados dinámicamente
# ---------------------------------------------------------------------

def evaluate_sequential(
    train_module,
    specification,
    parameters,
    X,
    y,
    splits,
    weights=None,
):
    """
    Construye predicciones OOF evaluando los folds secuencialmente.

    Se evita joblib/loky porque, en Windows, los procesos separados no
    pueden volver a importar el módulo dinámico train_models_module.

    La lógica estadística es la misma que en 04_train_models.evaluate:
    cada modelo se ajusta únicamente con fit_idx y predice val_idx.
    """

    predictions = np.full(len(X), np.nan, dtype=float)

    for fold_number, (fit_idx, val_idx) in enumerate(
        splits,
        start=1,
    ):
        print(
            f"      fold {fold_number}/{len(splits)} | "
            f"train: {len(fit_idx):,} | "
            f"validación: {len(val_idx):,}",
            flush=True,
        )

        fold_weights = (
            None
            if weights is None
            else weights[fit_idx]
        )

        fold_predictions = train_module.fit_predict(
            specification,
            parameters,
            X.iloc[fit_idx],
            y.iloc[fit_idx],
            X.iloc[val_idx],
            fold_weights,
        )

        predictions[val_idx] = fold_predictions

    return predictions

# ---------------------------------------------------------------------
# Reconstrucción
# ---------------------------------------------------------------------

def rebuild_oof(spec_numbers, scheme_names):
    """
    Reconstruye las predicciones OOF para los modelos y esquemas indicados.
    """

    train_module = load_training_module()

    validate_requested_specs(train_module, spec_numbers)
    validate_requested_schemes(train_module, scheme_names)

    print("=" * 78)
    print("RECONSTRUCCIÓN SELECTIVA DE PREDICCIONES OOF")
    print("=" * 78)
    print(f"Repositorio: {ROOT}")
    print(f"Salida:      {OOF_PATH}")
    print(f"Modelos:     {spec_numbers}")
    print(f"Esquemas:    {scheme_names}")
    print()

    print("Cargando datos procesados...")
    train, test = train_module.load_processed()

    print(
        f"Train: {len(train):,} filas y "
        f"{train.shape[1]:,} columnas"
    )
    print(
        f"Test: {len(test):,} filas y "
        f"{test.shape[1]:,} columnas"
    )
    print()

    target = train[train_module.TARGET]
    feature_sets = train_module.feature_sets(train.columns)
    best_params = train_module.load_best_params()

    # Construcción de las particiones una sola vez.
    # Se reutilizan directamente las funciones originales de utils.py.
    print("Construyendo particiones de validación...")

    scheme_builders = {
        "random": lambda: list(random_cv().split(train)),
        "random_twins": lambda: random_twin_splits(train),
        "spatial": lambda: spatial_cv_splits(train),
        "chapinero": lambda: [chapinero_holdout(train, test)],
        "northeast": lambda: [northeast_holdout(train, test)],
    }

    schemes = {
        scheme_name: scheme_builders[scheme_name]()
        for scheme_name in scheme_names
    }

    for scheme_name, splits in schemes.items():

        validation_rows = np.unique(
            np.concatenate([
                validation_index
                for _, validation_index in splits
            ])
        )

        n_splits = len(splits)
        word = "partición" if n_splits == 1 else "particiones"

        print(
            f"  {scheme_name:<15} "
            f"{n_splits:>2} {word} | "
            f"{len(validation_rows):>6,} filas de validación"
        )

    print()

    all_generated_columns = []

    for number in spec_numbers:
        n_splits = len(splits)
        word = "partición" if n_splits == 1 else "particiones"

        print(
            f"  {scheme_name:<15} "
            f"{n_splits:>2} {word} | "
            f"{len(validation_rows):>6,} filas de validación"
        )

    print()

    all_generated_columns = []

    for number in spec_numbers:
        specification = train_module.NUMBERED_SPECS[number]
        model_name = specification.name(number)

        start = time.time()

        print("-" * 78)
        print(f"Modelo {number}: {model_name}")
        print(
            f"Familia: {specification.family} | "
            f"Features: {specification.features}"
        )

        features = train_module.spec_features(
            specification,
            feature_sets,
        )

        missing_features = [
            column
            for column in features
            if column not in train.columns
        ]

        if missing_features:
            raise ValueError(
                f"El modelo {model_name} requiere columnas ausentes:\n"
                f"{missing_features}"
            )

        parameter_key = (
            specification.tune_key
            or f"{specification.family}_{specification.features}"
        )

        parameters = (
            best_params.get(parameter_key, {})
            if specification.tuned
            else {}
        )

        model_weights = (
            train_module.similarity_weights(
                train["test_similarity"]
            )
            if specification.weighted
            else None
        )

        print(f"Número de predictores: {len(features)}")

        if parameters:
            print(
                "Parámetros recuperados de best_params.json: "
                f"{parameter_key}"
            )

        model_table = train[
            [
                train_module.ID_COL,
                train_module.TARGET,
            ]
        ].copy()

        model_columns = []

        for scheme_name, splits in schemes.items():
            scheme_start = time.time()

            print(
                f"  Ejecutando {scheme_name}...",
                flush=True,
            )

            predictions = evaluate_sequential(
                train_module=train_module,
                specification=specification,
                parameters=parameters,
                X=train[features],
                y=target,
                splits=splits,
                weights=model_weights,
            )
            
            column_name = f"{model_name}__{scheme_name}"

            model_table[column_name] = predictions
            model_columns.append(column_name)
            all_generated_columns.append(column_name)

            summary = prediction_summary(
                target,
                predictions,
            )

            elapsed = (time.time() - scheme_start) / 60

            print(
                f"    cobertura: {summary['coverage']:6.2%} | "
                f"MAE: {summary['mae'] / 1e6:8.2f} M | "
                f"sesgo: {summary['bias'] / 1e6:+8.2f} M | "
                f"tiempo: {elapsed:6.2f} min"
            )

        # Guardado después de cada modelo para no perder avances.
        saved_table = save_oof(
            train_module,
            model_table[
                [
                    train_module.ID_COL,
                    train_module.TARGET,
                ]
                + model_columns
            ],
        )

        validate_oof_table(
            saved_table,
            train_module,
            model_columns,
        )

        model_minutes = (time.time() - start) / 60

        print(
            f"Modelo guardado correctamente en "
            f"{model_minutes:.2f} minutos."
        )

    # Validación final.
    final_table = pd.read_csv(OOF_PATH)

    validate_oof_table(
        final_table,
        train_module,
        all_generated_columns,
    )

    print()
    print("=" * 78)
    print("RECONSTRUCCIÓN FINALIZADA")
    print("=" * 78)
    print(f"Archivo: {OOF_PATH}")
    print(f"Filas:   {len(final_table):,}")
    print(f"Columnas totales: {final_table.shape[1]}")
    print(
        "Columnas generadas en esta ejecución: "
        f"{len(all_generated_columns)}"
    )

    print("\nColumnas nuevas:")

    for column in all_generated_columns:
        coverage = final_table[column].notna().mean()

        print(
            f"  {column:<75} "
            f"{coverage:6.2%}"
        )

    return final_table


# ---------------------------------------------------------------------
# Interfaz de línea de comandos
# ---------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruye selectivamente predicciones OOF "
            "para modelos históricos del proyecto."
        )
    )

    parser.add_argument(
        "--specs",
        nargs="+",
        type=int,
        default=DEFAULT_SPECS,
        help=(
            "Números de especificación que se reconstruirán. "
            f"Por defecto: {DEFAULT_SPECS}"
        ),
    )

    parser.add_argument(
        "--schemes",
        nargs="+",
        default=DEFAULT_SCHEMES,
        help=(
            "Esquemas de validación que se reconstruirán. "
            f"Por defecto: {DEFAULT_SCHEMES}"
        ),
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    rebuild_oof(
        spec_numbers=arguments.specs,
        scheme_names=arguments.schemes,
    )


if __name__ == "__main__":
    main()