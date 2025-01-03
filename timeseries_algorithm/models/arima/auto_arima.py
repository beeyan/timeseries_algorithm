# auto_arima_model.py (pmdarima利用例)
from pmdarima import auto_arima
from ..base_model import BaseTimeSeriesModel

class AutoARIMAModel(BaseTimeSeriesModel):
    """
    pmdarimaのauto_arimaを使った実装例
    """

    def __init__(self, seasonal=True, m=12):
        self.seasonal = seasonal
        self.m = m
        self.model_ = None

    def fit(self, X, y):
        # pmdarimaのauto_arima関数
        # Xを exogenous として渡すことも可能
        self.model_ = auto_arima(y, exogenous=X, seasonal=self.seasonal, m=self.m)

    def predict(self, X, steps=1):
        return self.model_.predict(n_periods=steps, exogenous=X)
