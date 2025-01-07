import joblib
import pandas as pd
from statsmodels.tsa.api import VAR
from .base_model import BaseTimeSeriesModel

class VARModel(BaseTimeSeriesModel):
    """
    Vector Autoregression (VAR) モデルの実装。
    """

    def __init__(self, maxlags=15, ic='aic'):
        """
        Parameters
        ----------
        maxlags : int
            過去ラグ数の最大値。
        ic : str
            ラグ選択基準 ('aic', 'bic', 'hqic', etc.)。
        """
        self.maxlags = maxlags
        self.ic = ic
        self.model_ = None
        self.results_ = None

    def fit(self, X, y=None):
        """
        モデルを学習する。

        Parameters
        ----------
        X : pd.DataFrame
            複数時系列を含むデータフレーム。
        y : None
            VARは教師なしの多変量モデルなので、yは無視。
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a pandas DataFrame containing multiple time series.")

        self.model_ = VAR(X)
        self.results_ = self.model_.select_order(maxlags=self.maxlags, ic=self.ic)
        self.results_ = self.model_.fit(self.results_.selected_orders['aic'])

    def predict(self, X, steps=5):
        """
        予測を行う。

        Parameters
        ----------
        X : pd.DataFrame
            過去のデータ。モデルの最後の `maxlags` ラグが必要。
        steps : int
            予測ステップ数。

        Returns
        -------
        forecast : pd.DataFrame
            予測結果。
        """
        if self.results_ is None:
            raise ValueError("Model has not been fitted yet.")

        forecast = self.results_.forecast(y=X.values[-self.results_.k_ar:], steps=steps)
        forecast_df = pd.DataFrame(forecast, index=pd.date_range(start=X.index[-1], periods=steps+1, freq=X.index.freq)[1:], columns=X.columns)
        return forecast_df

    def save_model(self, filepath: str):
        if self.results_ is None:
            raise ValueError("No fitted model to save.")
        joblib.dump(self.results_, filepath)

    def load_model(self, filepath: str):
        self.results_ = joblib.load(filepath)