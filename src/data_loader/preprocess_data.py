# F:\git\btc_trading_bot\src\data_loader\preprocess_data.py
# Colab용으로 수정: Windows 절대경로 대신 상대경로 사용

import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ta
import joblib

def load_and_preprocess_data(
    data_path="data/raw/btc_1m_2years.csv",  # 상대경로
    recent_days=None,
    models_dir="models"
):
    """
    Colab에서 동작 가능하도록 경로를 상대경로로 수정.
    data_path: default "data/raw/btc_1m_2years.csv"
    recent_days=None -> 전체 데이터 사용
    models_dir: default "models" (상대경로)
    """

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info(f"Loading data from: {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    if 'timestamp' not in df.columns:
        raise ValueError("Data must have a 'timestamp' column.")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    logging.info("Converted 'timestamp' to DatetimeIndex.")
    
    # 결측치 처리
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    # recent_days 필터 (없으면 전체 사용)
    if recent_days is not None:
        end_date = df.index.max()
        start_date = end_date - pd.Timedelta(days=recent_days)
        df = df.loc[start_date:end_date]
        logging.info(f"Filtered last {recent_days} days. Rows: {len(df)}")

    # close=0 제거
    before_len = len(df)
    df = df[df['close'] != 0]
    logging.info(f"Removed close=0. Before: {before_len}, After: {len(df)}")

    # 지표 계산 (MA, RSI, Bollinger Bands)
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_60'] = df['close'].rolling(window=60).mean()

    bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bollinger_mid'] = bb_indicator.bollinger_mavg()
    df['bollinger_upper'] = bb_indicator.bollinger_hband()
    df['bollinger_lower'] = bb_indicator.bollinger_lband()

    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    df.dropna(inplace=True)
    logging.info(f"Dropped NaN. Rows: {len(df)}")

    # 피처 목록
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'ma_5', 'ma_20', 'ma_60', 'rsi',
        'bollinger_mid', 'bollinger_upper', 'bollinger_lower'
    ]
    X_data = df[feature_columns]
    Y_data = df['close']

    # 스케일링
    scaler_X = MinMaxScaler()
    scaled_X = scaler_X.fit_transform(X_data)

    scaler_Y = MinMaxScaler()
    scaled_Y = scaler_Y.fit_transform(Y_data.values.reshape(-1, 1)).reshape(-1)

    X_data_array = scaled_X.astype(np.float32)
    Y_data_array = scaled_Y.astype(np.float32)

    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler_X, os.path.join(models_dir, "scaler_X.pkl"))
    joblib.dump(scaler_Y, os.path.join(models_dir, "scaler_Y.pkl"))

    with open(os.path.join(models_dir, "features.txt"), "w") as f:
        f.write("\n".join(feature_columns))

    logging.info("Preprocessing complete. Scalers and features saved.")
    return X_data_array, Y_data_array, feature_columns, df.index


if __name__ == "__main__":
    # Colab에서 모듈 실행 예시: 
    # !python -m src.data_loader.preprocess_data
    X_data, Y_data, features, idx = load_and_preprocess_data(
        data_path="data/raw/btc_1m_2years.csv", 
        recent_days=None,    # 전체 데이터 사용
        models_dir="models"  # 상대경로 예시
    )
    print("X_data shape:", X_data.shape)
    print("Y_data shape:", Y_data.shape)
    print("Feature columns:", features)
    print("Index range:", idx.min(), idx.max())
