# metrics.py

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def mape(y_true, y_pred, eps=1e-9):
    """
    Mean Absolute Percentage Error
    y_trueが0に近い場合、誤差が大きくなる可能性があるため注意。
    epsを加えるなどの対策をとる。
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100

def calculate_metrics(y_true, y_pred):
    """
    MAE, RMSE, MAPE をまとめて計算して返すユーティリティ。
    戻り値は辞書形式。
    """
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred)
    }
