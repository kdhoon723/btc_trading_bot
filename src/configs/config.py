import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")

DATA_RAW_PATH = BASE_DIR / "data" / "raw"
DATA_RAW_PATH.mkdir(parents=True, exist_ok=True)

PAIR_SYMBOL = os.getenv("PAIR_SYMBOL", "BTC/USDT")
TIMEFRAME = os.getenv("TIMEFRAME", "1m")

FETCH_LIMIT = int(os.getenv("FETCH_LIMIT", "1000"))
FETCH_INTERVAL_SEC = float(os.getenv("FETCH_INTERVAL_SEC", "0.6"))
YEARS_TO_FETCH = int(os.getenv("YEARS_TO_FETCH", "2"))
