import numpy as np
from typing import Any, Tuple

def detect_outliers_by_residuals(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    モデルの残差を用いて外れ値を検知する関数。

    Args:
        model (Any): fit, predict可能モデル。
        X (np.ndarray): shape (N, feature_dim) など。
        y (np.ndarray): shape (N,) の真値。
        threshold (float): 残差絶対値がthresholdを超えると外れ値。

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - outlier_indices: shape (K,), 外れ値インデックス
            - outlier_residuals: shape (K,), 残差値
    """
    if threshold < 0:
        raise ValueError(f"Threshold must be non-negative, got {threshold}.")

    model.fit(X, y)  # fit
    preds = model.predict(X)
    if preds.shape[0] != y.shape[0]:
        raise ValueError("predict output length differs from y length.")

    residuals = y - preds
    outlier_indices = np.where(np.abs(residuals) > threshold)[0]
    return outlier_indices, residuals[outlier_indices]
