"""Embeddings de frases del título y de la descripción de cada anuncio (ronda 4, pieza A).

Usa un modelo multilingüe preentrenado de sentence-transformers. Los embeddings solo
dependen del texto (no del precio), así que se calculan una vez para train y test; lo que
se aprende con el precio (la red sobre los embeddings, la SVD) se ajusta dentro de cada
partición en models.TextPredictors.

Salida: stores/processed/text_embeddings.npz (no se versiona: pesa unos 150 MB), con
  ids             property_id de train y luego de test
  title, desc     matrices float32 de n x 384, normalizadas a norma 1

Uso:
  python 13_text_embeddings.py                 # unos 20 minutos en CPU (16 núcleos)
  python 13_text_embeddings.py --limit 500     # prueba rápida con pocas filas
"""

import argparse
import time

import numpy as np
import pandas as pd

from utils import EMBEDDINGS_PATH, ID_COL, load_raw

# multilingual-e5-small: 384 dimensiones, hasta 512 tokens, entrenado en ~100 idiomas.
MODEL_NAME = "intfloat/multilingual-e5-small"
# Versión del modelo en Hugging Face con la que se calcularon los embeddings de la ronda 4.
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MAX_TOKENS = 256
BATCH_SIZE = 64


def encode(model, texts):
    """Embeddings normalizados; e5 espera el prefijo 'passage: ' en los documentos."""
    return model.encode([f"passage: {t}" for t in texts], batch_size=BATCH_SIZE,
                        normalize_embeddings=True, show_progress_bar=True,
                        convert_to_numpy=True).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="solo las primeras filas (prueba)")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    train, test = load_raw()
    rows = pd.concat([train, test], ignore_index=True)
    if args.limit:
        rows = rows.head(args.limit)
    model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device="cpu")
    model.max_seq_length = MAX_TOKENS
    start = time.time()
    title = encode(model, rows["title"].fillna("").tolist())
    desc = encode(model, rows["description"].fillna("").tolist())
    print(f"{len(rows)} anuncios en {(time.time() - start) / 60:.1f} min")
    if args.limit:
        return
    np.savez_compressed(EMBEDDINGS_PATH, ids=rows[ID_COL].to_numpy(), title=title, desc=desc)
    print(f"Guardado {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
