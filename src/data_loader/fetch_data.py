# F:\git\btc_trading_bot\src\data_loader\fetch_data.py

import os
import time
import ccxt
import pandas as pd
from datetime import datetime, timedelta
import logging

from src.configs.config import (
    BINANCE_API_KEY,
    BINANCE_SECRET_KEY,
    DATA_RAW_PATH,
    PAIR_SYMBOL,
    TIMEFRAME,
    FETCH_LIMIT,
    FETCH_INTERVAL_SEC,
    YEARS_TO_FETCH
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def init_exchange():
    exchange = ccxt.binance({
        'apiKey': BINANCE_API_KEY,
        'secret': BINANCE_SECRET_KEY,
        'enableRateLimit': True,
        # 실전 거래시에는 testnet=False (기본값), 아래 testnet 예시는 
        # 실제거래 여부에 따라 수정 필요
        'options': {'defaultType': 'future'}, # 선물시장 등 필요시 옵션 설정
    })
    return exchange

def safe_fetch_ohlcv(exchange, symbol, timeframe, since=None, limit=1000):
    """주어진 파라미터로 OHLCV 데이터를 안전하게 가져오는 함수.
       예외 발생 시 재시도, rate limit 대기 등을 구현."""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            return data
        except ccxt.NetworkError as e:
            logging.warning(f"Network error {e}, retrying {attempt+1}/{max_retries}...")
            time.sleep(2)
        except ccxt.ExchangeError as e:
            logging.error(f"Exchange error: {e}")
            break
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            break
    return []

def fetch_full_data():
    """2년치(설정값 기준) 1분봉 데이터를 Binance에서 연속적으로 수집 후 CSV 저장.
       실전용: 안정성 강화, 로그 출력, 중간 세이브 가능.
    """
    exchange = init_exchange()
    now = datetime.utcnow()
    start_time = now - timedelta(days=365*YEARS_TO_FETCH)
    since_timestamp = int(start_time.timestamp() * 1000)

    all_data = []
    current_since = since_timestamp

    logging.info(f"Start fetching {YEARS_TO_FETCH} years of {PAIR_SYMBOL} data from {start_time} to now.")

    # 반복적으로 데이터 수집
    while True:
        ohlcv = safe_fetch_ohlcv(exchange, PAIR_SYMBOL, TIMEFRAME, since=current_since, limit=FETCH_LIMIT)
        if not ohlcv:
            logging.info("No more data returned. Stopping.")
            break

        all_data.extend(ohlcv)

        last_timestamp = ohlcv[-1][0]
        current_since = last_timestamp + (60 * 1000)  # 다음 분부터 가져오기

        # 종료 조건: 현재 시간 도달 시 종료
        if current_since >= int(now.timestamp() * 1000):
            logging.info("Reached current time. Fetching complete.")
            break

        # Rate limit 대기
        time.sleep(FETCH_INTERVAL_SEC)

    if all_data:
        df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.sort_values("timestamp", inplace=True)
        df.set_index("timestamp", inplace=True)

        csv_path = os.path.join(DATA_RAW_PATH, f"btc_{TIMEFRAME}_{YEARS_TO_FETCH}years.csv")
        df.to_csv(csv_path)
        logging.info(f"Data saved to {csv_path}, total rows: {len(df)}")
    else:
        logging.warning("No data fetched. Check API keys, network, or symbol correctness.")

    return True

if __name__ == "__main__":
    fetch_full_data()
