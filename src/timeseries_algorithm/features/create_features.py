# create_features.py

import pandas as pd

def create_lag_features(df: pd.DataFrame, target_col: str, lags: list) -> pd.DataFrame:
    """
    時系列データに対して、指定したラグ数だけ新しいカラムを追加する。
    例: lags=[1,7] なら target_col= 'sales' に対し 'sales_lag1', 'sales_lag7' を作成。
    """
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    return df

def create_rolling_features(df: pd.DataFrame, target_col: str, windows: list) -> pd.DataFrame:
    """
    時系列データに対して、移動平均などのローリング統計量をカラムとして追加する。
    例: windows=[7,30] で 7日移動平均, 30日移動平均を追加。
    """
    df = df.copy()
    for w in windows:
        df[f"{target_col}_rollmean{w}"] = df[target_col].rolling(w).mean()
        df[f"{target_col}_rollstd{w}"] = df[target_col].rolling(w).std()
    return df

def add_time_features(df: pd.DataFrame, date_index: bool = True) -> pd.DataFrame:
    """
    日付に基づく特徴量 (曜日、月など) を追加する。
    dfがDatetimeIndexを持っている場合はdate_index=Trueとし、indexから曜日などを取得。
    """
    df = df.copy()
    if date_index and isinstance(df.index, pd.DatetimeIndex):
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        # さらに祝日フラグなどを追加するのも可
    return df
