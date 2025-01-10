import joblib
import pandas as pd
from typing import Any, Optional
from orbit.models import ETS
from timeseries_algorithm import BaseTimeSeriesModel

class ETSModel(BaseTimeSeriesModel):
    """
    OrbitライブラリのError-Trend-Seasonality (ETS)モデルを用いた時系列予測。
    """

    def __init__(
        self,
        error: str = "add",
        trend: str = "add",
        seasonal: str = "add",
        seasonal_periods: Optional[int] = None,
        seed: int = 42,
        verbose: bool = False
    ) -> None:
        """
        ETSモデルの初期化。

        Args:
            error (str, optional): 誤差項のモデル。デフォルトは"add"。
            trend (str, optional): トレンドのモデル。デフォルトは"add"。
            seasonal (str, optional): 季節性のモデル。デフォルトは"add"。
            seasonal_periods (Optional[int], optional): 季節周期。デフォルトはNone。
            seed (int, optional): 乱数シード。デフォルトは42。
            verbose (bool, optional): 学習時にログを表示するかどうか。デフォルトはFalse。
        """
        super().__init__()
        self.error = error
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.seed = seed
        self.verbose = verbose
        self.model = ETS(
            error=error,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            seed=seed,
            verbose=verbose
        )
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> None:
        """
        モデルを学習する。

        Args:
            X (pd.DataFrame): 'ds' カラムと 'y' カラムを含むトレーニングデータ。
            y (Optional[Any], optional): 無視される。OrbitではX内の 'y' を用いる。
        """
        self.model.fit(X)
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
        モデルを保存する。
        """
        joblib.dump(self.model, filepath)

    def load_model(self, filepath: str) -> None:
        """
        モデルをロードする。
        """
        self.model = joblib.load(filepath)
        self.fitted = True
