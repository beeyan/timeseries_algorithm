# arima_model.py
from statsmodels.tsa.arima.model import ARIMA
from ..base_model import BaseTimeSeriesModel

class ARIMAModel(BaseTimeSeriesModel):
    """
    BaseTimeSeriesModel を継承した ARIMA モデルのwrapper class
    """

    def __init__(self, order=(1,1,1)):
        """
        Parameters
        ----------
        order : tuple
            ARIMA の (p, d, q) パラメータ。デフォルトは (1,1,1)。
        """
        self.order = order
        self.model_ = None
        self.results_ = None  # statsmodelsで学習済みモデルを保持

    def fit(self, X, y):
        """
        モデルを学習するメソッド。
        ARIMA は通常、y のみが必須。X (外因性変数) を使わない場合は None でもよい。

        Parameters
        ----------
        X : array-like, optional
            特徴量。ARIMA 単体の場合は使わない想定。
        y : array-like
            時系列データ（1次元）
        """
        # statsmodelsのARIMAに y と order を渡して学習
        self.model_ = ARIMA(endog=y, order=self.order)
        self.results_ = self.model_.fit()

    def predict(self, X):
        """
        予測を行うメソッド。
        簡易的に、X の行数（len(X)）分だけ将来ステップを予測すると想定。

        Parameters
        ----------
        X : array-like
            予測に使う新規データ。ここでは将来のステップ数を決めるために利用。
        
        Returns
        -------
        y_pred : array-like
            予測値
        """
        if self.results_ is None:
            raise ValueError("Model has not been fitted yet.")

        forecast_steps = len(X) if X is not None else 1  # X が None の場合は1ステップだけ
        y_pred = self.results_.forecast(steps=forecast_steps)
        return y_pred

    def save_model(self, filepath: str):
        """
        学習済みモデルを指定ファイルパスに保存する。
        """
        if self.results_ is None:
            raise ValueError("No fitted model to save.")

        import joblib
        joblib.dump({
            "order": self.order,
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