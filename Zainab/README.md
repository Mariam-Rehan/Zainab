# Stock Price Trend Prediction (LSTM)

University deep learning project — predicts stock closing price trend using an LSTM sequence model on historical OHLCV data fetched live from Yahoo Finance.

**Topic:** Time Series · **Concepts:** LSTM, sequence modeling, windowed forecasting

> Educational project only. **Not financial advice.**

## Stack

- TensorFlow / Keras (LSTM)
- yfinance (data)
- scikit-learn (MinMax scaling, metrics)
- Streamlit (interactive demo)
- Jupyter (EDA + narrative)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt   # training, plots, notebook
# Cloud / demo only: pip install -r requirements.txt
```

## Usage

Train (default AAPL):

```bash
python -m src.train --ticker AAPL --epochs 25
```

Evaluate (uses saved model/scaler):

```bash
python -m src.evaluate
```

Streamlit demo:

```bash
streamlit run streamlit_app.py
```

Notebook:

```bash
jupyter notebook notebooks/stock_lstm.ipynb
```

Render PDF report (requires `pandoc` + a PDF engine — easiest is WeasyPrint):

```bash
# macOS (one-time):
brew install pandoc pango
pip install weasyprint

# Render:
DYLD_LIBRARY_PATH=/opt/homebrew/lib \
  pandoc report/report.md -o report/report.pdf \
  --pdf-engine=weasyprint --resource-path=report
```

Or with LaTeX installed: `pandoc report/report.md -o report/report.pdf`.

## Layout

```
src/           core library (data, preprocess, model, train, eval)
notebooks/     stock_lstm.ipynb — EDA + results narrative
streamlit_app.py   interactive demo
artifacts/     generated: model.keras, scaler.pkl, plots/, data/
report/        report.md + report.pdf
config.py      central constants
```

## Model

```
LSTM(50, return_sequences=True) → Dropout(0.2)
LSTM(50)                        → Dropout(0.2)
Dense(25) → Dense(1)
Optimizer: Adam, Loss: MSE, Window: 60 days
```

## Metrics

After training, `python -m src.evaluate` prints RMSE / MAE / MAPE on the test set and writes plots to `artifacts/plots/`.

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (include `artifacts/model.keras` and `artifacts/scaler.pkl`).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and **New app**.
3. Set **Repository** to your fork, **Branch** `main`, **Main file path** `streamlit_app.py`.
4. Python version: **3.12** (or add `.python-version` — already in repo).
5. Deploy. The app loads the committed model; live prices come from yfinance.

**Repo root must contain:** `streamlit_app.py`, `requirements.txt`, `config.py`, `src/`, and `artifacts/model.keras` + `artifacts/scaler.pkl`.
