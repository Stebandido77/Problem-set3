"""Afinamiento de BETO para predecir el precio a partir del texto del anuncio.

Corre en un notebook de Kaggle con GPU e Internet activados. Entrada: beto_input.csv
(property_id, split, fold, title, description, price). Salida: beto_predictions.csv con la
predicción de precio según el texto, fuera de muestra para train y promedio de los modelos
de cada partición para test. Ese archivo entra al boosting como un predictor más.
"""

import glob
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

MODEL_NAME = os.environ.get("BETO_MODEL", "dccuchile/bert-base-spanish-wwm-uncased")
OUTPUT_DIR = os.environ.get("BETO_OUTPUT", "/kaggle/working")
MAX_LENGTH = int(os.environ.get("BETO_MAX_LENGTH", 256))
EPOCHS = int(os.environ.get("BETO_EPOCHS", 3))
BATCH_SIZE = int(os.environ.get("BETO_BATCH_SIZE", 32))
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_SHARE = 0.1
SEED = 4107  # igual a utils.SEED; el notebook de Kaggle no ve el repo

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"


def find_input():
    """Ubica beto_input.csv dentro de los datos adjuntos al notebook."""
    explicit = os.environ.get("BETO_INPUT")
    if explicit:
        return explicit
    matches = glob.glob("/kaggle/input/**/beto_input.csv", recursive=True)
    if not matches:
        raise FileNotFoundError("No encuentro beto_input.csv en /kaggle/input")
    return matches[0]


class ListingDataset(Dataset):
    """Título y descripción como par de frases; el objetivo es el log del precio estandarizado."""

    def __init__(self, frame, tokenizer, targets=None):
        self.titles = frame["title"].fillna("").astype(str).tolist()
        self.descriptions = frame["description"].fillna("").astype(str).tolist()
        self.targets = targets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, index):
        item = self.tokenizer(self.titles[index], self.descriptions[index],
                              truncation="only_second", max_length=MAX_LENGTH)
        if self.targets is not None:
            item["labels"] = float(self.targets[index])
        return item


def make_loader(dataset, tokenizer, shuffle):
    """Lotes con relleno dinámico hasta el texto más largo de cada lote."""
    def collate(batch):
        labels = [b.pop("labels") for b in batch] if "labels" in batch[0] else None
        padded = tokenizer.pad(batch, return_tensors="pt")
        if labels is not None:
            padded["labels"] = torch.tensor(labels, dtype=torch.float32)
        return padded

    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, collate_fn=collate,
                      num_workers=2, pin_memory=USE_AMP)


@torch.no_grad()
def predict(model, loader):
    """Predicciones en la escala estandarizada del objetivo."""
    model.eval()
    outputs = []
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP):
            logits = model(**batch).logits
        outputs.append(logits.float().squeeze(-1).cpu().numpy())
    return np.concatenate(outputs)


def train_fold(train_frame, tokenizer, fold):
    """Afina BETO con las filas de entrenamiento de una partición."""
    torch.manual_seed(SEED + fold)
    log_price = np.log(train_frame["price"].to_numpy())
    center, scale = log_price.mean(), log_price.std()
    loader = make_loader(ListingDataset(train_frame, tokenizer, (log_price - center) / scale),
                         tokenizer, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    total_steps = len(loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_SHARE * total_steps),
                                                total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        running, start = 0.0, time.time()
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP):
                logits = model(**batch).logits.float().squeeze(-1)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
        print(f"  partición {fold} época {epoch + 1}: pérdida {running / len(loader):.4f} "
              f"({(time.time() - start) / 60:.1f} min)", flush=True)
    return model, center, scale


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = pd.read_csv(find_input())
    train = data[data["split"] == "train"].reset_index(drop=True)
    test = data[data["split"] == "test"].reset_index(drop=True)
    folds = sorted(train["fold"].unique())
    print(f"Dispositivo: {DEVICE} | train {len(train)} | test {len(test)} | "
          f"particiones {folds}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_loader = make_loader(ListingDataset(test, tokenizer), tokenizer, shuffle=False)
    oof_log = np.full(len(train), np.nan)
    test_log = np.zeros(len(test))

    for fold in folds:
        is_valid = (train["fold"] == fold).to_numpy()
        model, center, scale = train_fold(train[~is_valid].reset_index(drop=True),
                                          tokenizer, fold)
        valid_loader = make_loader(ListingDataset(train[is_valid], tokenizer), tokenizer,
                                   shuffle=False)
        oof_log[is_valid] = predict(model, valid_loader) * scale + center
        test_log += (predict(model, test_loader) * scale + center) / len(folds)

        mae = np.mean(np.abs(np.exp(oof_log[is_valid]) - train.loc[is_valid, "price"]))
        print(f"Partición {fold}: MAE fuera de muestra {mae / 1e6:.1f} M", flush=True)
        # Copia parcial por si la sesión se corta antes de terminar.
        partial = train.loc[~np.isnan(oof_log), ["property_id", "fold"]].copy()
        partial["beto_log_price"] = oof_log[~np.isnan(oof_log)]
        partial.to_csv(os.path.join(OUTPUT_DIR, "beto_oof_partial.csv"), index=False)
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    overall = np.mean(np.abs(np.exp(oof_log) - train["price"]))
    print(f"MAE fuera de muestra total: {overall / 1e6:.1f} M", flush=True)

    out_train = train[["property_id", "split", "fold"]].copy()
    out_train["beto_log_price"] = oof_log
    out_test = test[["property_id", "split", "fold"]].copy()
    out_test["beto_log_price"] = test_log
    out = pd.concat([out_train, out_test], ignore_index=True)
    out["beto_price"] = np.exp(out["beto_log_price"])
    path = os.path.join(OUTPUT_DIR, "beto_predictions.csv")
    out.to_csv(path, index=False)
    print(f"Guardado en {path}: {len(out)} filas", flush=True)


if __name__ == "__main__":
    main()
