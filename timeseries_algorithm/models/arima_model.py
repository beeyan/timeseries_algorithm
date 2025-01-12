# arima_model.py
import joblib
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter

from .interface import BaseTimeSeriesModel

class ARIMAXModel(BaseTimeSeriesModel):
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
        self.seasonal_order = (0,0,0,0)
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
        self.model_ = ARIMA(endog=y, exog=X, order=self.order)
        self.results_ = self.model_.fit()

    def predict(self, X, forecast_steps=7):
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
        y_pred = self.results_.forecast(steps=forecast_steps, exog=X)
        return y_pred

    def save_model(self, filepath: str):
        """
        学習済みモデルを指定ファイルパスに保存する。
        """
        if self.results_ is None:
            raise ValueError("No fitted model to save.")

        
        joblib.dump({
            "order": self.order,
            "results": self.results_
        }, filepath)

    def load_model(self, filepath: str):
        """
        学習済みモデルを指定ファイルパスから読み込む。
        """
        
        data = joblib.load(filepath)
        self.order = data["order"]
        self.results_ = data["results"]
        # self.model_ は再構成する場合もあるが、基本は results_ が推論に使える


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

        
        joblib.dump({
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "results": self.results_
        }, filepath)

    def load_model(self, filepath: str):
        """
        学習済みモデルを指定ファイルパスから読み込む。
        """
        
        data = joblib.load(filepath)
        self.order = data["order"]
        self.seasonal_order = data["seasonal_order"]
        self.results_ = data["results"]

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
    
    def save_model(self, filepath: str):
        """
        学習済みモデルを指定ファイルパスに保存する。
        """
        if self.results_ is None:
            raise ValueError("No fitted model to save.")

        
        joblib.dump({
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "results": self.results_
        }, filepath)

    def load_model(self, filepath: str):
        """
        学習済みモデルを指定ファイルパスから読み込む。
        """
        
        data = joblib.load(filepath)
        self.order = data["order"]
        self.seasonal_order = data["seasonal_order"]
        self.results_ = data["results"]

class TimeVaryingARModel(BaseTimeSeriesModel):
    """
    時変係数ARモデルの実装。
    """

    def __init__(self, lags=1, k_states=None):
        """
        Parameters
        ----------
        lags : int
            ARのラグ数。
        k_states : int or None
            状態空間の状態数。Noneの場合、ラグ数と同じ。
        """
        self.lags = lags
        self.k_states = lags if k_states is None else k_states
        self.kf = None
        self.params_ = None

    def fit(self, X, y):
        """
        モデルを学習する。時変係数を考慮したARモデル。

        Parameters
        ----------
        X : None or array-like
            VARとは異なり、単変量時系列を想定。
        y : pd.Series or np.ndarray
            ターゲット時系列データ。
        """
        if X is not None:
            raise ValueError("TimeVaryingARModel does not support exogenous variables.")

        if isinstance(y, pd.Series):
            y = y.values

        # 状態空間モデルの設定
        self.kf = KalmanFilter(k_endog=1, k_states=self.k_states, initialization='approximate_diffuse')

        # 観測方程式: y_t = Z_t * alpha_t + epsilon_t
        self.kf['design'] = np.ones((1, self.k_states))

        # 状態遷移方程式: alpha_t = T_t * alpha_{t-1} + R_t * eta_t
        self.kf['transition'] = np.eye(self.k_states)  # 単純なランダムウォーク
        self.kf['selection'] = np.eye(self.k_states)

        # 観測ノイズ
        self.kf['obs_cov'] = np.eye(1) * 1.0

        # 状態ノイズ
        self.kf['state_cov'] = np.eye(self.k_states) * 0.1

        # 初期状態
        self.kf.initialize_known(
            initial_state=np.zeros(self.k_states),
            initial_state_cov=np.eye(self.k_states) * 10
        )

        # カルマンフィルタの適用
        filtered_state, filtered_state_cov, _, _ = self.kf.filter(y)

        self.params_ = filtered_state

    def predict(self, X, steps=5):
        """
        予測を行う。未来の係数を予測し、ARモデルで予測を行う。

        Parameters
        ----------
        X : None or array-like
            予測時に使用する外因性変数。サポートしない。
        steps : int
            予測ステップ数。

        Returns
        -------
        y_pred : np.ndarray
            予測結果。
        """
        if X is not None:
            raise ValueError("TimeVaryingARModel does not support exogenous variables.")

        if self.kf is None or self.params_ is None:
            raise ValueError("Model has not been fitted yet.")

        # 未来の係数を予測（ここではランダムウォークを仮定）
        forecast_state = self.params_[-1]  # 最後の状態
        y_pred = []
        for _ in range(steps):
            # 現在の状態から次の状態を予測
            forecast_state = self.kf['transition'] @ forecast_state
            # 観測値を予測
            y_forecast = self.kf['design'] @ forecast_state
            y_pred.append(y_forecast[0])
        return np.array(y_pred)

    def save_model(self, filepath: str):
        """
        モデルを保存する。
        """
        if self.kf is None or self.params_ is None:
            raise ValueError("No fitted model to save.")
        joblib.dump({
            "lags": self.lags,
            "k_states": self.k_states,
            "filtered_state": self.params_,
            "kf_params": {
                "design": self.kf['design'],
                "transition": self.kf['transition'],
                "selection": self.kf['selection'],
                "obs_cov": self.kf['obs_cov'],
                "state_cov": self.kf['state_cov'],
            }
        }, filepath)

    def load_model(self, filepath: str):
        """
        モデルをロードする。
        """
        data = joblib.load(filepath)
        self.lags = data["lags"]
        self.k_states = data["k_states"]
        self.params_ = data["filtered_state"]

        # 再構築したカルマンフィルタ
        self.kf = KalmanFilter(k_endog=1, k_states=self.k_states, initialization='approximate_diffuse')
        self.kf['design'] = data["kf_params"]["design"]
        self.kf['transition'] = data["kf_params"]["transition"]
        self.kf['selection'] = data["kf_params"]["selection"]
        self.kf['obs_cov'] = data["kf_params"]["obs_cov"]
        self.kf['state_cov'] = data["kf_params"]["state_cov"]

        self.kf.initialize_known(
            initial_state=np.zeros(self.k_states),
            initial_state_cov=np.eye(self.k_states) * 10
        )


class ARIMAXGARCHHybridModel(BaseTimeSeriesModel):
    """
    ARIMAXで平均値過程を予測し、残差にGARCHを適用するハイブリッド例
    """
    def __init__(self, arima_order=(1,1,1), garch_order=(1,1)):
        self.arima_order = arima_order
        self.garch_order = garch_order
        self.arima_ = None
        self.garch_ = None
        self.resid_ = None

    def fit(self, X, y):
        # 1) ARIMAX
        self.arima_ = ARIMA(endog=y, exog=X, order=self.arima_order).fit()
        self.resid_ = self.arima_.resid
        # 2) GARCH
        self.garch_ = arch_model(self.resid_, p=self.garch_order[0], q=self.garch_order[1]).fit(disp='off')

    def predict(self, X):
        # 平均値予測
        mean_pred = self.arima_.forecast(steps=len(X), exog=X)
        # 分散予測なども可能だが簡略化
        return mean_pred
    

class ARIMAErrorsModel(BaseTimeSeriesModel):
    """
    回帰モデルの残差にARIMAをあてる例（簡略）
    """
    def __init__(self, arima_order=(1,1,1)):
        self.arima_order = arima_order
        self.reg_model_ = None  # ここでは省略
        self.arima_model_ = None
        self.resid_ = None

    def fit(self, X, y):
        # 1) 回帰モデルをフィット (省略: self.reg_model_)
        # 2) 残差に対してARIMA
        resid = y  # 本来は y - reg_model.predict(X)
        self.arima_model_ = ARIMA(resid, order=self.arima_order).fit()

    def predict(self, X):
        steps = len(X)
        # 回帰モデルによる予測(省略)
        reg_pred = np.zeros(steps)
        resid_pred = self.arima_model_.forecast(steps=steps)
        return reg_pred + resid_pred