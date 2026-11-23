"""Arma stores/processed/beto_input.csv, la entrada del modelo de texto (BETO).

Cada anuncio de entrenamiento lleva la partición de la validación espacial
(utils.spatial_cv_splits) a la que pertenece. Los anuncios con la misma descripción
se mueven completos a la partición de su primer miembro, para que el modelo de texto
no memorice precios de anuncios repetidos. En test la partición es -1."""
import numpy as np
import pandas as pd

from utils import ID_COL, PROCESSED_DIR, TARGET, load_raw, spatial_cv_splits

OUTPUT_PATH = PROCESSED_DIR / "beto_input.csv"


def spatial_fold_ids(train):
    """Partición espacial (0 a 4) de cada fila de train."""
    folds = np.full(len(train), -1)
    for fold, (_, validation_idx) in enumerate(spatial_cv_splits(train)):
        folds[validation_idx] = fold
    return folds


def group_twins(train, folds):
    """Lleva cada grupo de descripciones idénticas a la partición de su primer miembro.
    Las descripciones vacías no forman grupo. Devuelve las particiones nuevas."""
    description = train["description"]
    first_fold = pd.Series(folds).groupby(description.to_numpy()).transform("first")
    has_text = description.notna() & (description.str.strip() != "")
    return np.where(has_text, first_fold, folds).astype(int)


def build_input(train, test):
    folds = spatial_fold_ids(train)
    grouped = group_twins(train, folds)
    print(f"Filas que cambiaron de partición: {int((grouped != folds).sum())}")
    train = train.assign(split="train", fold=grouped)
    test = test.assign(split="test", fold=-1, price=np.nan)
    columns = [ID_COL, "split", "fold", "title", "description", TARGET]
    return pd.concat([train[columns], test[columns]], ignore_index=True)


def main():
    train, test = load_raw()
    result = build_input(train, test)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Guardado {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
