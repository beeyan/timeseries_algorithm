from statsmodels.tsa.statespace.sarimax import SARIMAX
from ..base_model import BaseTimeSeriesModel

class SARIMAXModel(BaseTimeSeriesModel):
    """
    SARIMAX: 季節性と外因性変数を同時に考慮
    """

    def __init__(self, order=(1,1,1), seasonal_order=(0,0,0,0)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_ = None
        self.results_ = None

    def fit(self, X, y):
        """
        X : exog (DataFrameやndarray)
        y : ターゲット時系列
        """
        self.model_ = SARIMAX(endog=y, exog=X, order=self.order, seasonal_order=self.seasonal_order)
        self.results_ = self.model_.fit()

    def predict(self, X, steps):
        if self.results_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.results_.forecast(steps=steps, exog=X)

    def save_model(self, filepath: str):
        """
        学習済みモデルを指定ファイルパスに保存する。
        """
        if self.results_ is None:
            raise ValueError("No fitted model to save.")

        import joblib
        joblib.dump({
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "results": self.results_
        }, filepath)

    def load_model(self, filepath: str):
        """
        学習済みモデルを指定ファイルパスから読み込む。
        """
        import joblib
        data = joblib.load(filepath)
        self.order = data["order"]
        self.results_ = data["results"]
        # self.model_ は再構成する場合もあるが、基本は results_ が推論に使える