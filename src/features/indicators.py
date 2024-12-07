# F:\git\btc_trading_bot\src\features\indicators.py
import pandas as pd
import numpy as np

def add_moving_averages(df, short_window=5, medium_window=20, long_window=60):
    # 단기, 중기, 장기 이동평균
    df[f"ma_{short_window}"] = df["close"].rolling(window=short_window, min_periods=short_window).mean()
    df[f"ma_{medium_window}"] = df["close"].rolling(window=medium_window, min_periods=medium_window).mean()
    df[f"ma_{long_window}"] = df["close"].rolling(window=long_window, min_periods=long_window).mean()
    return df

def add_rsi(df, period=14):
    # RSI 계산
    # RSI = 100 - 100/(1+RS), RS = avg_gain/avg_loss
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)  # division by zero 방지
    df["rsi"] = 100 - (100 / (1 + rs))
    return df

def add_bollinger_bands(df, period=20, num_std=2):
    # Bollinger Bands 중심선(이동평균), 상단선, 하단선
    rolling_mean = df["close"].rolling(window=period, min_periods=period).mean()
    rolling_std = df["close"].rolling(window=period, min_periods=period).std()
    df["bollinger_mid"] = rolling_mean
    # 필요하다면 상단, 하단선도 추가할 수 있으나, 기획서에서는 중심선만 사용
    # df["bollinger_upper"] = rolling_mean + num_std * rolling_std
    # df["bollinger_lower"] = rolling_mean - num_std * rolling_std
    return df
