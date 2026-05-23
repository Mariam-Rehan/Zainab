"""Train LSTM, save model + scaler + history."""
from __future__ import annotations

import argparse
import json
import random

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

import config
from src.data_loader import load_stock
from src.model import build_lstm
from src.preprocess import prepare_training_data


def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main() -> None:
    p = argparse.ArgumentParser(description="Train LSTM on stock close prices.")
    p.add_argument("--ticker", default=config.TICKER)
    p.add_argument("--start", default=config.START_DATE)
    p.add_argument("--end", default=config.END_DATE)
    p.add_argument("--window", type=int, default=config.WINDOW)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    p.add_argument("--refresh", action="store_true", help="Force re-download.")
    args = p.parse_args()

    set_seed()

    print(f"[1/4] Loading {args.ticker} ({args.start} -> {args.end or 'today'})")
    df = load_stock(args.ticker, args.start, args.end, refresh=args.refresh)
    print(f"      rows={len(df)}")

    print("[2/4] Preprocessing")
    prep = prepare_training_data(df["Close"], window=args.window)
    print(f"      X_train={prep['X_train'].shape}  X_test={prep['X_test'].shape}")

    print("[3/4] Building + training model")
    model = build_lstm(window=args.window)
    model.summary()
    es = EarlyStopping(
        monitor="val_loss",
        patience=config.PATIENCE,
        restore_best_weights=True,
    )
    history = model.fit(
        prep["X_train"],
        prep["y_train"],
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=[es],
        verbose=2,
    )

    print("[4/4] Saving artifacts")
    model.save(config.MODEL_PATH)
    joblib.dump(prep["scaler"], config.SCALER_PATH)
    with open(config.HISTORY_PATH, "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            f,
            indent=2,
        )

    meta = {
        "ticker": args.ticker,
        "start": args.start,
        "end": args.end,
        "window": args.window,
        "epochs_requested": args.epochs,
        "epochs_run": len(history.history["loss"]),
        "split_idx": int(prep["split_idx"]),
        "rows": int(len(df)),
    }
    with open(config.ARTIFACTS / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"      model -> {config.MODEL_PATH}")
    print(f"      scaler -> {config.SCALER_PATH}")
    print(f"      history -> {config.HISTORY_PATH}")


if __name__ == "__main__":
    main()
