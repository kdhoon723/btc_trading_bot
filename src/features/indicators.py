# F:\git\btc_trading_bot\src\features\indicators.py
import pandas as pd
import numpy as np

def add_moving_averages(df, short_window=5, medium_window=20, long_window=60):
    df[f"ma_{short_window}"] = df["close"].rolling(window=short_window, min_periods=short_window).mean()
    df[f"ma_{medium_window}"] = df["close"].rolling(window=medium_window, min_periods=medium_window).mean()
    df[f"ma_{long_window}"] = df["close"].rolling(window=long_window, min_periods=long_window).mean()
    return df

def add_rsi(df, period=14):
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df

def add_bollinger_bands(df, period=20, num_std=2):
    rolling_mean = df["close"].rolling(window=period, min_periods=period).mean()
    rolling_std = df["close"].rolling(window=period, min_periods=period).std()

    df["bollinger_mid"] = rolling_mean
    df["bollinger_upper"] = rolling_mean + num_std * rolling_std
    df["bollinger_lower"] = rolling_mean - num_std * rolling_std

    return df
