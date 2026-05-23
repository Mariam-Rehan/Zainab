"""Central configuration for stock LSTM project."""
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ARTIFACTS / "data"
PLOTS_DIR = ARTIFACTS / "plots"
MODEL_PATH = ARTIFACTS / "model.keras"
SCALER_PATH = ARTIFACTS / "scaler.pkl"
HISTORY_PATH = ARTIFACTS / "history.json"
METRICS_PATH = ARTIFACTS / "metrics.json"

TICKER = "AAPL"
START_DATE = "2014-01-01"
END_DATE = None  # None = today

WINDOW = 60
TRAIN_SPLIT = 0.80
EPOCHS = 25
BATCH_SIZE = 32
LSTM_UNITS = 50
DROPOUT = 0.2
DENSE_UNITS = 25
PATIENCE = 5
SEED = 42

for d in (ARTIFACTS, DATA_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
