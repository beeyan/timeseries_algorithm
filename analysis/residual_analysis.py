import numpy as np
from typing import Tuple
from statsmodels.tsa.stattools import acf, pacf

def compute_residual_acf_pacf(
    residuals: np.ndarray,
    nlags: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    残差系列に対するACF, PACFを計算する関数。

    Args:
        residuals (np.ndarray): shape (N,) の1次元残差ベクトル。
        nlags (int): 計算するラグ数。

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - acf_vals: ACF値配列 (nlags+1 次元)
            - pacf_vals: PACF値配列 (nlags+1 次元)
    """
    acf_vals = acf(residuals, nlags=nlags)
    pacf_vals = pacf(residuals, nlags=nlags)
    return acf_vals, pacf_vals
