"""Fetch & cache stock OHLCV data via yfinance."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

import config


def load_stock(
    ticker: str = config.TICKER,
    start: str = config.START_DATE,
    end: str | None = config.END_DATE,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return OHLCV DataFrame for `ticker`. Caches to artifacts/data/{ticker}.csv."""
    end = end or date.today().isoformat()
    csv_path: Path = config.DATA_DIR / f"{ticker}.csv"

    if csv_path.exists() and not refresh:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if not df.empty:
            return df

    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker} ({start} -> {end})")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    df.to_csv(csv_path)
    return df


if __name__ == "__main__":
    df = load_stock(refresh=True)
    print(df.tail())
    print(f"Rows: {len(df)}  Range: {df.index.min().date()} -> {df.index.max().date()}")
