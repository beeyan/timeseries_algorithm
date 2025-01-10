from statsmodels.tsa.forecasting.theta import ThetaModel
from .base_model import BaseTimeSeriesModel

class ThetaMethodModel(BaseTimeSeriesModel):
    """
    Theta法 (シンプルな指数平滑の組み合わせ)
    """
    def __init__(self):
        self.model_ = None

    def fit(self, X, y):
        self.model_ = ThetaModel(y).fit()

    def predict(self, X):
        if self.model_ is None:
            raise ValueError("Model is not fitted yet.")
        return self.model_.forecast(len(X))
