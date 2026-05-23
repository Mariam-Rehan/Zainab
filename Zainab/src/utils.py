"""Plotting + metric helpers."""
from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def plot_loss(history: dict, out_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["loss"], label="train")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="val")
    ax.set_title("Training Loss (MSE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
    return fig


def plot_predictions(dates, actual, predicted, ticker: str, out_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates, actual, label="Actual", linewidth=1.5)
    ax.plot(dates, predicted, label="Predicted", linewidth=1.5, alpha=0.85)
    ax.set_title(f"{ticker} — Actual vs Predicted Close Price (Test Set)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
    return fig


def plot_forecast(history_dates, history_prices, future_dates, future_prices, ticker, out_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(history_dates, history_prices, label="Historical", linewidth=1.2)
    ax.plot(future_dates, future_prices, label="Forecast", linewidth=2.0, color="crimson")
    ax.axvline(history_dates[-1], color="gray", linestyle="--", alpha=0.6)
    ax.set_title(f"{ticker} — {len(future_dates)}-Day Forward Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
    return fig


def forecast_future(model, last_window: np.ndarray, scaler, steps: int = 30) -> np.ndarray:
    """Autoregressive multi-step forecast in original price units.

    Uses direct model __call__ (not .predict) — avoids Keras-3 per-call
    data-adapter overhead which is catastrophic for single-sample loops.
    """
    window = last_window.flatten().astype(np.float32).copy()
    preds_scaled = []
    for _ in range(steps):
        x = window.reshape(1, -1, 1).astype(np.float32)  # numpy avoids Keras DataAdapter pool bug
        nxt = float(model(x, training=False).numpy()[0, 0])
        preds_scaled.append(nxt)
        window = np.append(window[1:], nxt)
    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    return scaler.inverse_transform(preds_scaled).flatten()
