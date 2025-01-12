import numpy as np
from typing import Tuple
from statsmodels.tsa.stattools import ccf

def cross_correlation_values(
    ts1: np.ndarray,
    ts2: np.ndarray,
    max_lag: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    2つの時系列のクロス相関を計算し、一部を返す。

    Args:
        ts1 (np.ndarray): shape (T,)
        ts2 (np.ndarray): shape (T,)
        max_lag (int): ±max_lag の範囲で出力

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - lags: shape (2*max_lag+1,), ラグ配列
            - c_sub: shape (2*max_lag+1,), 各ラグの相関
    """
    c = ccf(ts1 - ts1.mean(), ts2 - ts2.mean())
    mid: int = len(c)//2
    lags = np.arange(-max_lag, max_lag+1)
    c_sub = c[mid - max_lag : mid + max_lag + 1]
    return lags, c_sub
