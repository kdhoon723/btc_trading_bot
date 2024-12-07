# F:\git\btc_trading_bot\src\data_loader\preprocess_data.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import timedelta

from src.configs.config import DATA_RAW_PATH, YEARS_TO_FETCH
from src.features.indicators import add_moving_averages, add_rsi, add_bollinger_bands

def load_raw_data(file_name=None):
    if file_name is None:
        file_name = f"btc_1m_{YEARS_TO_FETCH}years.csv"
    csv_path = os.path.join(DATA_RAW_PATH, file_name)
    df = pd.read_csv(csv_path, parse_dates=True, index_col="timestamp")
    df.sort_index(inplace=True)
    return df

def compute_features(df):
    # OHLCV: df에 이미 있음
    # 지표 추가
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_bollinger_bands(df)

    # 보조지표 계산 후 초기 구간에 NaN 발생 → 제거
    df.dropna(inplace=True)

    return df

def add_target(df, prediction_horizon=10):
    # prediction_horizon 후의 close 가격 대비 현재 close의 변동률을 타겟으로 설정
    # (close(t+10min)/close(t)) - 1 = 변동률
    df["future_close"] = df["close"].shift(-prediction_horizon)
    df["target_return"] = (df["future_close"] / df["close"]) - 1.0
    df.dropna(inplace=True)  # 마지막 부분에서 future_close 없으니 NaN 발생
    return df

def scale_features(df):
    # 스케일링 대상 컬럼 선정
    # OHLCV + ma_5, ma_20, ma_60, rsi, bollinger_mid 등
    # target_return는 나중에 분리하므로 스케일링 안 함
    feature_cols = ["open", "high", "low", "close", "volume", 
                    "ma_5", "ma_20", "ma_60", "rsi", "bollinger_mid"]
    
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[feature_cols])
    for i, col in enumerate(feature_cols):
        df[col] = scaled_values[:, i]
    
    return df, scaler

def create_timeseries_data(df, input_window=4320, prediction_horizon=10):
    """
    input_window 분(72시간)이 입력, prediction_horizon=10분 후의 변동률 예측.
    df는 이미 target_return 컬럼 포함, 모든 스케일링 완료 상태 가정.
    """

    feature_cols = ["open", "high", "low", "close", "volume", 
                    "ma_5", "ma_20", "ma_60", "rsi", "bollinger_mid"]
    target_col = "target_return"

    X = []
    y = []
    timestamps = df.index.values

    # 마지막 예측이 가능하려면: i+input_window+prediction_horizon <= len(df)
    for i in range(len(df) - input_window - prediction_horizon):
        X_seq = df.iloc[i:(i+input_window)][feature_cols].values
        y_val = df.iloc[i+input_window][target_col]  # input_window 뒤 10분 후 return
        X.append(X_seq)
        y.append(y_val)

    X = np.array(X)
    y = np.array(y)
    return X, y

def preprocess_data():
    df = load_raw_data()
    df = compute_features(df)
    df = add_target(df, prediction_horizon=10)  # 예: 10분 후 변동률 예측
    df, scaler = scale_features(df)
    X, y = create_timeseries_data(df, input_window=4320, prediction_horizon=10)

    # 이후 학습/검증/테스트 분리
    # 예: 70% train, 20% valid, 10% test
    train_size = int(0.7*len(X))
    valid_size = int(0.2*len(X))
    test_size = len(X) - train_size - valid_size

    X_train, y_train = X[:train_size], y[:train_size]
    X_valid, y_valid = X[train_size:train_size+valid_size], y[train_size:train_size+valid_size]
    X_test, y_test = X[train_size+valid_size:], y[train_size+valid_size:]

    return X_train, y_train, X_valid, y_valid, X_test, y_test, scaler


if __name__ == "__main__":
    X_train, y_train, X_valid, y_valid, X_test, y_test, scaler = preprocess_data()
    print("Preprocessing Done!")
    print("Train set shape:", X_train.shape, y_train.shape)
    print("Valid set shape:", X_valid.shape, y_valid.shape)
    print("Test set shape:", X_test.shape, y_test.shape)
