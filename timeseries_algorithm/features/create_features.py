from typing import List

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import jpholiday


def create_lag_features(df: pd.DataFrame, target_col: str, lags: list) -> pd.DataFrame:
    """
    時系列データに対して、指定したラグ数だけ新しいカラムを追加する。
    例: lags=[1,7] なら target_col= 'sales' に対し 'sales_lag1', 'sales_lag7' を作成。
    Args:
        df (pd.DataFrame): 時系列データ。
        target_col (str): ラグ特徴量を作成する対象の列名。
        lags (list): ラグのリスト。
    Returns:
        df (pd.DataFrame): ラグ特徴量が追加されたデータフレーム。
    """
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    return df


def create_rolling_features(
    df: pd.DataFrame, target_col: str, windows: list
) -> pd.DataFrame:
    """
    時系列データに対して、移動平均などのローリング統計量をカラムとして追加する。
    例: windows=[7,30] で 7日移動平均, 30日移動平均を追加。

    Args:
        df (pd.DataFrame): 時系列データ。
        target_col (str): ローリング特徴量を作成する対象の列名。
        windows (list): 移動平均のウィンドウ幅のリスト。
    Returns:
        df (pd.DataFrame): ローリング特徴量が追加されたデータフレーム。

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

    Args:
        df (pd.DataFrame): 時系列データ。
        date_index (bool): インデックスから特徴量を作成するかどうか。
    Returns:
        df (pd.DataFrame): 日付特徴量が追加されたデータフレーム。
    """
    df = df.copy()
    if date_index and isinstance(df.index, pd.DatetimeIndex):
        df["year"] = df.index.year
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        df["day"] = df.index.day
        df["is_Sunday"] = (df.index.dayofweek == 6).astype(int)
        df["is_Saturday"] = (df.index.dayofweek == 5).astype(int)
        df["is_month_start"] = df.index.is_month_start.astype(int)
        df["is_month_end"] = df.index.is_month_end.astype(int)
        df["is_japanese_holiday"] = df.index.isin(
            jpholiday.between(df.index[0], df.index[-1])
        ).astype(int)
    return df


def add_differencing_features(
    df: pd.DataFrame, column: str, order: int = 1
) -> pd.DataFrame:
    """
    指定した列に対して差分特徴量を追加する関数。

    Args:
        df (pd.DataFrame):
            特徴量を追加する元のデータフレーム。
        column (str):
            差分を計算する対象の列名。
        order (int, optional):
            差分の次数。デフォルトは 1。

    Returns:
        pd.DataFrame:
            差分特徴量が追加された新しいデータフレーム。
    """
    df_diff = df.copy()
    df_diff[f"diff_{order}"] = df_diff[column].diff(order)
    return df_diff


def add_exponential_moving_average(
    df: pd.DataFrame, column: str, span: int = 3
) -> pd.DataFrame:
    """
    指定した列に対して指数移動平均（EMA）を追加する関数。

    Args:
        df (pd.DataFrame):
            特徴量を追加する元のデータフレーム。
        column (str):
            EMAを計算する対象の列名。
        span (int, optional):
            EMAのスパン。デフォルトは 3。

    Returns:
        pd.DataFrame:
            EMAが追加された新しいデータフレーム。
    """
    df_ema = df.copy()
    df_ema[f"ema_{span}"] = df_ema[column].ewm(span=span, adjust=False).mean()
    return df_ema


def add_fft_features(df: pd.DataFrame, column: str, order: int = 10) -> pd.DataFrame:
    """
    指定した列に対して高速フーリエ変換（FFT）を行い、周波数成分を特徴量として追加する関数。

    Args:
        df (pd.DataFrame):
            特徴量を追加する元のデータフレーム。
        column (str):
            FFTを計算する対象の列名。
        order (int, optional):
            FFTの周波数成分の数。デフォルトは 10。

    Returns:
        pd.DataFrame:
            FFTの周波数成分が追加された新しいデータフレーム。
    """
    df_fft = df.copy()
    fft_result = np.fft.fft(df_fft[column])
    for i in range(1, order + 1):
        df_fft[f"fft_{i}"] = np.abs(fft_result[i])
    return df_fft


def add_seasonal_decomposition_features(
    df: pd.DataFrame, column: str, model: str = "additive", period: int = 7
) -> pd.DataFrame:
    """
    指定した列に対して季節分解（トレンド、季節性、残差）を追加する関数。

    Args:
        df (pd.DataFrame):
            特徴量を追加する元のデータフレーム。
        column (str):
            季節分解を行う対象の列名。
        model (str, optional):
            季節分解のモデルタイプ（'additive' または 'multiplicative'）。デフォルトは 'additive'。
        period (int, optional):
            季節性の周期。デフォルトは 7。

    Returns:
        pd.DataFrame:
            季節分解特徴量（trend, seasonal, residual）が追加された新しいデータフレーム。
    """
    df_seasonal = df.copy()
    decomposition = seasonal_decompose(
        df_seasonal[column], model=model, period=period, extrapolate_trend="freq"
    )
    df_seasonal["trend"] = decomposition.trend
    df_seasonal["seasonal"] = decomposition.seasonal
    df_seasonal["residual"] = decomposition.resid
    return df_seasonal


def add_autocorrelation_features(
    df: pd.DataFrame, column: str, lags: List[int]
) -> pd.DataFrame:
    """
    指定した列に対して自己相関特徴量を追加する関数。

    Args:
        df (pd.DataFrame):
            特徴量を追加する元のデータフレーム。
        column (str):
            自己相関を計算する対象の列名。
        lags (List[int]):
            計算するラグのリスト。例： [1, 2, 3]

    Returns:
        pd.DataFrame:
            自己相関特徴量が追加された新しいデータフレーム。
    """
    df_autocorr = df.copy()
    for lag in lags:
        autocorr = df_autocorr[column].autocorr(lag=lag)
        df_autocorr[f"autocorr_{lag}"] = autocorr
    return df_autocorr
