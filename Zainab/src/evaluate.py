"""Evaluate saved LSTM: metrics, plots, forecast."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

import config
from src.data_loader import load_stock
from src.preprocess import prepare_training_data
from src.utils import (
    forecast_future,
    mae,
    mape,
    plot_forecast,
    plot_loss,
    plot_predictions,
    rmse,
)


def main() -> None:
    meta_path = config.ARTIFACTS / "meta.json"
    if not meta_path.exists():
        raise SystemExit("No meta.json — run `python -m src.train` first.")
    with open(meta_path) as f:
        meta = json.load(f)

    ticker = meta["ticker"]
    window = meta["window"]

    print(f"Loading data + model for {ticker}")
    df = load_stock(ticker, meta["start"], meta["end"])
    model = load_model(config.MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)

    import tensorflow as tf

    prep = prepare_training_data(df["Close"], window=window)
    print(f"  test set: {prep['X_test'].shape}", flush=True)
    print("  running test prediction...", flush=True)
    y_pred_scaled = (
        model(tf.convert_to_tensor(prep["X_test"]), training=False).numpy().flatten()
    )
    print("  done test prediction", flush=True)

    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(prep["y_test"].reshape(-1, 1)).flatten()

    metrics = {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE_%": mape(y_true, y_pred),
        "n_test": int(len(y_true)),
    }
    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    test_dates = df.index[prep["split_idx"] :]
    n = min(len(test_dates), len(y_true))
    test_dates = test_dates[-n:]
    y_true_plot = y_true[-n:]
    y_pred_plot = y_pred[-n:]

    plot_predictions(
        test_dates, y_true_plot, y_pred_plot, ticker,
        out_path=config.PLOTS_DIR / "actual_vs_predicted.png",
    )

    with open(config.HISTORY_PATH) as f:
        history = json.load(f)
    plot_loss(history, out_path=config.PLOTS_DIR / "loss_curve.png")

    print("  generating 30-day forecast...", flush=True)
    last_window_scaled = scaler.transform(
        df["Close"].values[-window:].reshape(-1, 1)
    )
    future_prices = forecast_future(model, last_window_scaled, scaler, steps=30)
    print("  done forecast", flush=True)
    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=30)
    plot_forecast(
        df.index[-180:],
        df["Close"].values[-180:],
        future_dates,
        future_prices,
        ticker,
        out_path=config.PLOTS_DIR / "forecast_30d.png",
    )

    forecast_df = pd.DataFrame({"date": future_dates, "predicted_close": future_prices})
    forecast_df.to_csv(config.ARTIFACTS / "forecast_30d.csv", index=False)
    print(f"\nPlots -> {config.PLOTS_DIR}")
    print(f"Forecast CSV -> {config.ARTIFACTS / 'forecast_30d.csv'}")


if __name__ == "__main__":
    main()
