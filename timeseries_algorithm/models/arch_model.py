from arch import arch_model
from .interface import BaseTimeSeriesModel

class ARCHModel(BaseTimeSeriesModel):
    """
    ARCHモデル (分散の時間依存性をモデリング)
    """
    def __init__(self, p=1):
        self.p = p
        self.model_ = None
        self.res_ = None

    def fit(self, X, y):
        self.model_ = arch_model(y, vol='ARCH', p=self.p)
        self.res_ = self.model_.fit(disp='off')

    def predict(self, X):
        if self.res_ is None:
            raise ValueError("Model is not fitted yet.")
        # 分散予測を返すなど。ここでは簡易例として0を返す。
        steps = len(X)
        forecast = self.res_.forecast(horizon=steps)
        return forecast.variance.values[-1]
    
class GARCHModel(BaseTimeSeriesModel):
    """
    GARCH(1,1)モデルの例
    """
    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q
        self.model_ = None
        self.res_ = None


    def fit(self, X, y):
        self.model_ = arch_model(y, vol='GARCH', p=self.p, q=self.q)
        self.res_ = self.model_.fit(disp='off')

    def predict(self, X):
        if self.res_ is None:
            raise ValueError("Model is not fitted yet.")
        steps = len(X)
        forecast = self.res_.forecast(horizon=steps)
        # 分散予測を返す例
        return forecast.variance.values[-1]
    

class EGARCHModel(BaseTimeSeriesModel):
    """
    EGARCHモデル
    """
    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q
        self.model_ = None
        self.res_ = None


    def fit(self, X, y):
        self.model_ = arch_model(y, vol='EGARCH', p=self.p, q=self.q)
        self.res_ = self.model_.fit(disp='off')

    def predict(self, X):
        if self.res_ is None:
            raise ValueError("Model is not fitted yet.")
        steps = len(X)
        forecast = self.res_.forecast(horizon=steps)
        # 分散予測を返す例
        return forecast.variance.values[-1]