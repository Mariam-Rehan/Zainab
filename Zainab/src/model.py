"""LSTM model factory."""
from __future__ import annotations

from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential

import config


def build_lstm(
    window: int = config.WINDOW,
    units: int = config.LSTM_UNITS,
    dropout: float = config.DROPOUT,
    dense_units: int = config.DENSE_UNITS,
) -> Sequential:
    model = Sequential(
        [
            Input(shape=(window, 1)),
            LSTM(units, return_sequences=True),
            Dropout(dropout),
            LSTM(units, return_sequences=False),
            Dropout(dropout),
            Dense(dense_units, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model
