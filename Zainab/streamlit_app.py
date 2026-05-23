"""Streamlit demo for stock LSTM."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

import config
from src.data_loader import load_stock
from src.preprocess import prepare_training_data
from src.utils import forecast_future, mae, mape, rmse

st.set_page_config(page_title="Stock Trend LSTM", page_icon="📈", layout="wide")
st.title("📈 Stock Price Trend Prediction — LSTM")
st.caption("University project · Educational use only · **Not financial advice**")

with st.sidebar:
    st.header("Settings")
    meta_path = config.ARTIFACTS / "meta.json"
    default_ticker = config.TICKER
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        default_ticker = meta.get("ticker", default_ticker)

    ticker = st.text_input("Ticker", value=default_ticker).upper().strip()
    start = st.date_input("Start date", value=date(2014, 1, 1))
    end = st.date_input("End date", value=date.today())
    forecast_days = st.slider("Forecast horizon (days)", 5, 60, 30)

    st.markdown("---")
    st.info(
        "Trained on AAPL by default. For other tickers, re-train via:\n\n"
        "`python -m src.train --ticker TSLA`"
    )

if not config.MODEL_PATH.exists() or not config.SCALER_PATH.exists():
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()

@st.cache_resource
def _load_artifacts():
    return load_model(config.MODEL_PATH), joblib.load(config.SCALER_PATH)


@st.cache_data(show_spinner=False)
def _load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    return load_stock(ticker, start, end)


model, scaler = _load_artifacts()

try:
    df = _load_data(ticker, str(start), str(end))
except Exception as e:
    st.error(f"Failed to load {ticker}: {e}")
    st.stop()

st.subheader(f"{ticker} — Historical Close Price")
st.line_chart(df[["Close"]], height=320)

col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{len(df):,}")
col2.metric("Latest Close", f"${df['Close'].iloc[-1]:,.2f}")
col3.metric(
    "% Change (1Y)",
    f"{(df['Close'].iloc[-1] / df['Close'].iloc[-252] - 1) * 100:+.2f}%"
    if len(df) > 252
    else "n/a",
)

st.subheader("Actual vs Predicted (Test Set)")
prep = prepare_training_data(df["Close"], window=config.WINDOW)
with st.spinner("Running test predictions..."):
    y_pred_scaled = model.predict(prep["X_test"], verbose=0).flatten()
y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_true = scaler.inverse_transform(prep["y_test"].reshape(-1, 1)).flatten()

test_dates = df.index[prep["split_idx"] :]
n = min(len(test_dates), len(y_true))
plot_df = pd.DataFrame(
    {"Actual": y_true[-n:], "Predicted": y_pred[-n:]},
    index=test_dates[-n:],
)
st.line_chart(plot_df, height=380)

m1, m2, m3 = st.columns(3)
m1.metric("RMSE", f"${rmse(y_true, y_pred):.2f}")
m2.metric("MAE", f"${mae(y_true, y_pred):.2f}")
m3.metric("MAPE", f"{mape(y_true, y_pred):.2f}%")

st.subheader(f"{forecast_days}-Day Forward Forecast (Autoregressive)")
last_window = scaler.transform(df["Close"].values[-config.WINDOW :].reshape(-1, 1))
with st.spinner(f"Generating {forecast_days}-day forecast..."):
    future = forecast_future(model, last_window, scaler, steps=forecast_days)
future_dates = pd.bdate_range(start=df.index[-1] + timedelta(days=1), periods=forecast_days)

hist_tail = df["Close"].iloc[-180:]
forecast_df = pd.DataFrame(
    {"Historical": list(hist_tail.values) + [np.nan] * forecast_days,
     "Forecast": [np.nan] * len(hist_tail) + list(future)},
    index=list(hist_tail.index) + list(future_dates),
)
st.line_chart(forecast_df, height=380)

with st.expander("Forecast table"):
    st.dataframe(
        pd.DataFrame({"date": future_dates, "predicted_close": future.round(2)}),
        hide_index=True,
        use_container_width=True,
    )

st.caption(
    "Model: 2-layer LSTM(50) + Dropout(0.2) + Dense(25→1). "
    "Window=60 days, MinMaxScaler, Adam+MSE. "
    "Long-horizon autoregressive forecasts compound errors — use as trend indicator only."
)
