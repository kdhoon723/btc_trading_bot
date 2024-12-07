# F:\git\btc_trading_bot\src\configs\config.py

import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ENV_PATH = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path=ENV_PATH)

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")

# 데이터 저장 경로
DATA_RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw')
os.makedirs(DATA_RAW_PATH, exist_ok=True)

# 거래 페어 및 타임프레임 설정
PAIR_SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"

# Fetch 설정
FETCH_LIMIT = 1000       # 한번에 가져올 캔들 수
FETCH_INTERVAL_SEC = 0.6 # API 호출 간 대기 시간 (rate limit 대응)
YEARS_TO_FETCH = 2       # 몇 년치 데이터를 가져올지

# 실제 Binance rate limit은 1200 requests per minute 수준이므로
# 넉넉히 interval을 두어 안정적 수집 보장
