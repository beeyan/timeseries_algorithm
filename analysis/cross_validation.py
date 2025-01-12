import numpy as np
from typing import Any, Callable, List
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

def time_series_cv(
    model_factory: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5
) -> List[float]:
    """
    時系列クロスバリデーション (TimeSeriesSplit) を行うサンプル関数。

    Args:
        model_factory (Callable[[], Any]): モデルインスタンスを返すファクトリ関数。
        X (np.ndarray): 入力特徴量 (N, seq_len, ...など)。
        y (np.ndarray): 予測対象 (N,) or (N, steps)。
        n_splits (int): TimeSeriesSplitの分割数。

    Returns:
        List[float]: 各分割でのMAEリスト。
    """
    ts_cv = TimeSeriesSplit(n_splits=n_splits)
    scores: List[float] = []

    for train_idx, test_idx in ts_cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = model_factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        scores.append(mae)

    return scores
