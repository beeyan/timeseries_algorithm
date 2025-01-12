import numpy as np
from typing import Any, List
from sklearn.metrics import mean_squared_error

def rolling_backtest(
    model: Any,
    data: np.ndarray,
    window_size: int,
    forecast_horizon: int
) -> float:
    """
    ローリング方式のバックテストを行うサンプル関数。
    マルチステップ予測を想定し、test_sliceは 'forecast_horizon' 点。
    モデルも 'forecast_horizon' 個の予測を返すことを期待。

    Args:
        model (Any): fit, predict メソッドを持つ時系列モデル。
                     predict(X) -> shape (forecast_horizon,) の配列を返す想定。
        data (np.ndarray): shape (T,) の1次元時系列データ。
        window_size (int): 学習ウィンドウ。
        forecast_horizon (int): 予測ホライズン（マルチステップ）。

    Returns:
        float: ローリング検証における平均MSE。
    """
    T: int = len(data)
    if window_size + forecast_horizon > T:
        raise ValueError(
            f"window_size + forecast_horizon = {window_size + forecast_horizon} "
            f"exceeds data length {T}."
        )

    mses: List[float] = []
    start: int = 0

    while start + window_size + forecast_horizon <= T:
        train_slice = data[start : start + window_size]   # window_size points
        test_slice = data[start + window_size : start + window_size + forecast_horizon]
        # test_slice => length=forecast_horizon

        # fit
        # 例: X_train=(window_size,) => shape(1, window_size) or something
        #   multi-step => we keep it simple
        X_train = train_slice.reshape(1, -1)   # shape(1, window_size)
        y_train = None  # modelに合わせる (some models only need X= full y)
        model.fit(X_train, y_train)

        # predict
        X_test = np.array([[window_size, forecast_horizon]])  
        # ↑ 例: ここでは時間情報をダミーで与えるだけの実装
        #   実際にはモデルによって形状が異なる
        y_pred = model.predict(forecast_horizon)  # int指定など

        # test_slice => shape(forecast_horizon,)
        if y_pred.shape[0] != forecast_horizon:
            raise ValueError(
                f"Model returned {y_pred.shape[0]} steps, expected {forecast_horizon}."
            )

        # MSE
        mse = mean_squared_error(test_slice, y_pred)
        mses.append(mse)

        start += forecast_horizon

    return float(np.mean(mses))
