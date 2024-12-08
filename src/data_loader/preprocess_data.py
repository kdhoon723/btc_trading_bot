# F:\git\btc_trading_bot\src\data_loader\preprocess_data.py

import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ta
from datetime import timedelta
import joblib

def load_and_preprocess_data(
    data_path="F:\\git\\btc_trading_bot\\data\\raw\\btc_1m_2years.csv",
    recent_days=730,  # 약 2년치
    models_dir="F:\\git\\btc_trading_bot\\models"
):
    """
    전처리 과정 (2년치 1분봉 데이터 기준):
    - data_path: 2년치 1분봉 데이터 CSV (예: btc_1m_2years.csv)
    - recent_days=730으로 설정하여 약 2년치 데이터 필터링 (원한다면 None으로 전체 사용 가능)
    - close=0 제거, 결측치 처리
    - 지표계산: MA_5, MA_20, MA_60, RSI(14), Bollinger Bands(mid, upper, lower)
      성능 향상 기대 근거로 Bollinger 상하단 사용
    - NaN 제거
    - MinMaxScaler로 X, Y 정규화 후 스케일러와 피처목록 저장
    - 윈도우 생성은 하지 않음 (학습 시 Dataset에서 처리)

    반환:
    X_data_array (N,F), Y_data_array (N,), features(피처 목록), df.index
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

    # 최근 N일(2년) 필터링
    if recent_days is not None:
        end_date = df.index.max()
        start_date = end_date - pd.Timedelta(days=recent_days)
        df = df.loc[start_date:end_date]
        logging.info(f"Filtered last {recent_days} days (~2 years). Rows: {len(df)}")

    # close=0 제거
    before_len = len(df)
    df = df[df['close'] != 0]
    logging.info(f"Removed close=0. Before: {before_len}, After: {len(df)}")

    # 지표 계산
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

    # 피처 목록 (기획서 합의)
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
    scaled_Y = scaler_Y.fit_transform(Y_data.values.reshape(-1,1)).reshape(-1)

    X_data_array = scaled_X.astype(np.float32)
    Y_data_array = scaled_Y.astype(np.float32)

    # 스케일러, 피처 목록 저장
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler_X, os.path.join(models_dir, "scaler_X.pkl"))
    joblib.dump(scaler_Y, os.path.join(models_dir, "scaler_Y.pkl"))

    with open(os.path.join(models_dir, "features.txt"), "w") as f:
        f.write("\n".join(feature_columns))

    logging.info("Preprocessing complete. Scalers and features saved.")
    return X_data_array, Y_data_array, feature_columns, df.index


# 아래 main 부분은 필요하다면 테스트용. 실제 운영 시 제거 가능.
if __name__ == "__main__":
    # 실제 2년치 1분봉 데이터 파일명 확인 후 경로 설정
    X_data, Y_data, features, idx = load_and_preprocess_data(
        data_path="F:\\git\\btc_trading_bot\\data\\raw\\btc_1m_2years.csv",
        recent_days=None  # 2년치
    )
    print("X shape:", X_data.shape, "Y shape:", Y_data.shape)
    print("Features:", features)
    print("Index range:", idx.min(), idx.max())
