"""Scale + window time series for LSTM training."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import config


def make_sequences(series: np.ndarray, window: int = config.WINDOW):
    """Convert 1-D series into (X, y) windowed pairs.

    X shape: (N, window, 1); y shape: (N,)
    """
    series = np.asarray(series, dtype=np.float32).reshape(-1)
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window : i])
        y.append(series[i])
    X = np.array(X, dtype=np.float32).reshape(-1, window, 1)
    y = np.array(y, dtype=np.float32)
    return X, y


def prepare_training_data(
    close: pd.Series,
    window: int = config.WINDOW,
    train_split: float = config.TRAIN_SPLIT,
):
    """Split chronologically, fit scaler on train only, return windowed train/test."""
    close = close.astype(np.float32).values.reshape(-1, 1)
    split_idx = int(len(close) * train_split)

    train_raw = close[:split_idx]
    test_raw = close[split_idx - window :]  # include lookback overlap for test windows

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled = scaler.transform(test_raw)

    X_train, y_train = make_sequences(train_scaled.flatten(), window)
    X_test, y_test = make_sequences(test_scaled.flatten(), window)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "split_idx": split_idx,
    }
