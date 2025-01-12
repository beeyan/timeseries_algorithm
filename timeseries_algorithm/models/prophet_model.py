import joblib
import numpy as np
import pandas as pd
from typing import Any, Optional
from prophet import Prophet
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from timeseries_algorithm.models.interface import BaseTimeSeriesModel

class ProphetModel(BaseTimeSeriesModel):
    """
    Prophetを用いた時系列予測モデル。
    MLflowによるロギング機能を統合。
    """

    def __init__(self, 
                 growth='linear', 
                 changepoint_prior_scale=0.05, 
                 seasonality_prior_scale=10.0, 
                 holidays_prior_scale=10.0, 
                 seasonality_mode='additive', 
                 yearly_seasonality=True, 
                 weekly_seasonality=True, 
                 daily_seasonality=False):
        """
        Parameters
        ----------
        growth : str
            'linear' または 'logistic' のいずれか。トレンドの種類。
        changepoint_prior_scale : float
            変化点の柔軟性を制御するパラメータ。
        seasonality_prior_scale : float
            季節性の柔軟性を制御するパラメータ。
        holidays_prior_scale : float
            休日効果の柔軟性を制御するパラメータ。
        seasonality_mode : str
            'additive' または 'multiplicative' のいずれか。季節性のモデルタイプ。
        yearly_seasonality : bool or int
            年間季節性の有無。
        weekly_seasonality : bool or int
            週間季節性の有無。
        daily_seasonality : bool
            日間季節性の有無。
        """
        super().__init__()
        self.growth = growth
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        
        self.model = Prophet(
            growth=self.growth,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality
        )
        self.fitted = False

    def fit(self, X, y):
        """
        モデルを学習する。

        Parameters
        ----------
        X : pd.DataFrame
            日付を含むデータフレーム。Prophetでは通常、'ds' と 'y' のカラムが必要。
        y : None
            Prophetは 'y' カラムをデータフレーム内で扱うため、yは無視。
        """
        self.model.fit(X)
        self.fitted = True

    def predict(self, X, periods=5):
        """
        予測を行う。

        Parameters
        ----------
        X : pd.DataFrame
            未来のデータフレーム。'ds' カラムのみ必要。
        periods : int
            予測する期間数（日数など、データの頻度に依存）。
        
        Returns
        -------
        forecast : pd.DataFrame
            予測結果
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # 未来のデータフレームを生成
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        
        # 予測期間のみを抽出
        forecast = forecast.tail(periods)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    def save_model(self, filepath: str):
        """
        モデルを保存する。

        Parameters
        ----------
        filepath : str
            モデル保存先のファイルパス
        """
        joblib.dump(self.model, filepath)

    def load_model(self, filepath: str):
        """
        モデルをロードする。

        Parameters
        ----------
        filepath : str
            モデル読み込み元のファイルパス
        """
        self.model = joblib.load(filepath)
        self.fitted = True

    def _calculate_metrics(self, X, forecast):
        """
        トレーニングデータに対する評価指標を計算する。

        Parameters
        ----------
        X : pd.DataFrame
            トレーニングデータ
        forecast : pd.DataFrame
            予測結果

        Returns
        -------
        metrics : dict
            計算された評価指標
        """

        # 実測値と予測値を結合
        df = X.merge(forecast, on='ds', how='left')
        y_true = df['y']
        y_pred = df['yhat']
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


class NeuralProphetModel(BaseTimeSeriesModel):
    """
    NeuralProphetを用いた時系列予測モデル。
    """

    def __init__(
        self,
        n_forecasts: int = 1,
        n_lags: int = 0,
        epochs: int = 10,
        learning_rate: float = 1e-3,
        verbose: bool = False
    ) -> None:
        """
        NeuralProphetモデルの初期化。

        Args:
            n_forecasts (int, optional): 未来予測ステップ数。デフォルトは1。
            n_lags (int, optional): 過去ラグ数。デフォルトは0（ラグなし）。
            epochs (int, optional): 学習エポック数。デフォルトは10。
            learning_rate (float, optional): 学習率。デフォルトは1e-3。
            verbose (bool, optional): 学習時にログを出力するかどうか。デフォルトはFalse。
        """
        super().__init__()
        self.model = NeuralProphet(
            n_forecasts=n_forecasts,
            n_lags=n_lags,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose
        )
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> None:
        """
        モデルを学習する。

        Args:
            X (pd.DataFrame): トレーニングデータ。'ds' カラム（日付）と 'y' カラム（ターゲット）が必要。
            y (Optional[Any], optional): 無視される。NeuralProphetではX内の 'y' を用いる。
        """
        self.model.fit(X, freq="D")  # freqは日次データの場合
        self.fitted = True

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        予測を行う。

        Args:
            X (pd.DataFrame): 予測に使用するデータ。'ds' カラムを含む。

        Returns:
            pd.DataFrame: 予測結果。
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet.")
        return self.model.predict(X)

    def save_model(self, filepath: str) -> None:
        """
        学習済みモデルを指定ファイルパスに保存する。

        Args:
            filepath (str): 保存先のファイルパス。
        """
        joblib.dump(self.model, filepath)

    def load_model(self, filepath: str) -> None:
        """
        学習済みモデルを指定ファイルパスから読み込む。

        Args:
            filepath (str): 読み込み元のファイルパス。
        """
        self.model = joblib.load(filepath)
        self.fitted = True
