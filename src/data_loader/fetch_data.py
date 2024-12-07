# F:\git\btc_trading_bot\src\data_loader\fetch_data.py

import os
import time
import ccxt
import pandas as pd
from datetime import datetime, timedelta, UTC
import logging
from math import ceil
from tqdm import tqdm

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
        'options': {
            'defaultType': 'future',  
            'adjustForTimeDifference': True,
            'recvWindow': 60000
        },
    })
    exchange.load_markets()
    return exchange

def safe_fetch_ohlcv(exchange, symbol, timeframe, since=None, limit=1000):
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
    exchange = init_exchange()

    now = datetime.now(UTC)  
    start_time = now - timedelta(days=365*YEARS_TO_FETCH)
    since_timestamp = int(start_time.timestamp() * 1000)
    now_timestamp = int(now.timestamp() * 1000)

    all_data = []
    current_since = since_timestamp

    logging.info(f"Start fetching {YEARS_TO_FETCH} years of {PAIR_SYMBOL} data from {start_time} to now.")

    # 총 필요한 대략적 호출 수 추정 (총 분 수 / FETCH_LIMIT)
    # 총 분: YEARS_TO_FETCH * 365일 * 24시간 * 60분 = 약 1,051,200분 (2년 기준)
    # 정확한 계산:
    total_minutes = YEARS_TO_FETCH * 365 * 24 * 60
    total_calls_estimate = ceil(total_minutes / FETCH_LIMIT)

    # tqdm 프로그레스 바 초기화
    # desc는 시작 시점 설명, dynamic desc로 현재 구간 갱신 예정
    pbar = tqdm(total=total_calls_estimate, desc="Fetching data...", unit="req", dynamic_ncols=True)

    while True:
        ohlcv = safe_fetch_ohlcv(exchange, PAIR_SYMBOL, TIMEFRAME, since=current_since, limit=FETCH_LIMIT)
        if not ohlcv:
            logging.info("No more data returned. Stopping.")
            break

        all_data.extend(ohlcv)

        last_timestamp = ohlcv[-1][0]
        current_since = last_timestamp + (60 * 1000)

        # 현재 구간 표시: last_timestamp를 datetime으로 변환해 tqdm desc 업데이트
        last_dt = datetime.utcfromtimestamp(last_timestamp/1000.0).replace(tzinfo=UTC)
        # desc에 현재 어디까지 받았는지 표시
        pbar.set_description_str(f"Fetching up to {last_dt.isoformat()}")

        pbar.update(1)  # fetch 한 번 할 때마다 progress +1

        if current_since >= now_timestamp:
            logging.info("Reached current time. Fetching complete.")
            break

        time.sleep(FETCH_INTERVAL_SEC)

    pbar.close()

    if all_data:
        df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
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