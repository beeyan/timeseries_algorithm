# bats_model.py

import numpy as np
import joblib
from typing import Optional, List, Any
from .interface import BaseTimeSeriesModel

from tbats import BATS, TBATS


class BATSModel(BaseTimeSeriesModel):
    """
    tbatsライブラリの BATS クラスをラップし、BaseTimeSeriesModel インターフェイスを提供するモデル。

    共変量(covariates)を扱いたい場合は、fit時に `X` を外因性変数として与え、yを1次元時系列として分離する実装例。
    """

    def __init__(
        self,
        use_box_cox: bool = False,
        use_trend: bool = False,
        use_damped_trend: bool = False,
        seasonal_periods: Optional[List[float]] = None,
        use_arma_errors: bool = True,
        show_warnings: bool = True,
        n_jobs: int = 1,
        # ここからBaseTimeSeriesModelとは無関係なパラメータ
        ):
        """
        Args:
            use_box_cox (bool): Box-Cox変換を使うかどうか。
            use_trend (bool): トレンド項をモデル化するか。
            use_damped_trend (bool): トレンドをdamped(減衰)にするかどうか。
            seasonal_periods (List[float]): 季節周期（小数も可）をリストで指定。
            use_arma_errors (bool): ARMA誤差を使用するかどうか。
            show_warnings (bool): tbats内部の警告を表示するかどうか。
            n_jobs (int): 並列実行のジョブ数。
        """
        self.use_box_cox = use_box_cox
        self.use_trend = use_trend
        self.use_damped_trend = use_damped_trend
        self.seasonal_periods = seasonal_periods if seasonal_periods is not None else []
        self.use_arma_errors = use_arma_errors
        self.show_warnings = show_warnings
        self.n_jobs = n_jobs

        self.model_ = None
        self.fitted_ = False

    def fit(self, X: Any, y: Optional[np.ndarray] = None) -> None:
        """
        BATSモデルをフィットする。

        Args:
            X (Any): shape (T,) or shape (T, c+1) など。以下の2パターンを想定。
                     - 共変量なし: Xが1次元配列 => yが None => y_data = X
                     - 共変量あり: Xが (T, c+1) => X[:,0] がターゲットy, X[:,1:] が外因性変数
            y (np.ndarray): 非使用。BaseTimeSeriesModelとの互換のため。
        """
        # データ形状の解釈
        exog = None
        if isinstance(X, np.ndarray) and X.ndim == 1:
            # 共変量なし => そのままy_data
            y_data = X
        elif isinstance(X, np.ndarray) and X.ndim == 2:
            # 0列目 => y, 1列目以降 => exogenous
            y_data = X[:, 0]
            if X.shape[1] > 1:
                exog = X[:, 1:]
        else:
            raise ValueError("X must be 1D or 2D np.ndarray for BATSModel.")

        # BATSインスタンス生成
        estimator = BATS(
            use_box_cox=self.use_box_cox,
            use_trend=self.use_trend,
            use_damped_trend=self.use_damped_trend,
            seasonal_periods=self.seasonal_periods,
            use_arma_errors=self.use_arma_errors,
            show_warnings=self.show_warnings,
            n_jobs=self.n_jobs
        )
        self.model_ = estimator.fit(y_data, exogenous=exog)
        self.fitted_ = True

    def predict(self, X: Any) -> np.ndarray:
        """
        予測を行う。BATSのPython実装では forecast(steps=...) が通常。
        Xが int なら そのステップ先を予測。
        Xが shape (T, c+1) の場合、(先の期間ぶんexog指定) + steps=T ?

        Args:
            X (Any): 
                - int: 先のステップ数
                - 2D array: exogデータ + stepsを infer ?

        Returns:
            np.ndarray: 予測値1次元配列
        """
        if not self.fitted_:
            raise ValueError("BATSModel is not fitted yet.")

        if isinstance(X, int):
            # forecast(steps=X)
            steps = X
            forecast_vals = self.model_.forecast(steps=steps)
            return forecast_vals
        elif isinstance(X, np.ndarray) and X.ndim == 2:
            # exogがある場合 => shape (T, c)
            steps = X.shape[0]
            exog = X
            forecast_vals = self.model_.forecast(steps=steps, exogenous=exog)
            return forecast_vals
        else:
            raise ValueError("X must be int or 2D array for BATSModel predict")

    def save_model(self, filepath: str) -> None:
        """
        学習済みBATSモデルをファイルに保存。
        """
        if not self.fitted_ or self.model_ is None:
            raise ValueError("No fitted model to save.")

        joblib.dump(self.model_, filepath)

    def load_model(self, filepath: str) -> None:
        """
        モデルをファイルから読み込み。
        """
        self.model_ = joblib.load(filepath)
        self.fitted_ = True


class TBATSModel(BaseTimeSeriesModel):
    """
    tbatsライブラリの TBATS クラスをラップし、BaseTimeSeriesModel インターフェイスを提供するモデル。
    BATS と同様に exogenous を利用可能。
    """

    def __init__(
        self,
        use_box_cox: bool = False,
        use_trend: bool = False,
        use_damped_trend: bool = False,
        seasonal_periods: Optional[List[float]] = None,
        use_arma_errors: bool = True,
        show_warnings: bool = True,
        n_jobs: int = 1
    ) -> None:
        """
        Args:
            use_box_cox (bool): Box-Cox変換。
            use_trend (bool): トレンド項。
            use_damped_trend (bool): Dampedトレンド。
            seasonal_periods (List[float]): 季節周期。
            use_arma_errors (bool): ARMA誤差を使用するかどうか。
            show_warnings (bool): tbats内部の警告を表示するか。
            n_jobs (int): 並列実行ジョブ数。
        """
        self.use_box_cox = use_box_cox
        self.use_trend = use_trend
        self.use_damped_trend = use_damped_trend
        self.seasonal_periods = seasonal_periods if seasonal_periods is not None else []
        self.use_arma_errors = use_arma_errors
        self.show_warnings = show_warnings
        self.n_jobs = n_jobs

        self.model_ = None
        self.fitted_ = False

    def fit(self, X: Any, y: Optional[np.ndarray] = None) -> None:
        """
        TBATSモデルをフィット。

        Args:
            X (Any): shape (T,) or shape (T, c+1)
            y (np.ndarray): None
        """
        if isinstance(X, np.ndarray) and X.ndim == 1:
            y_data = X
            exog = None
        elif isinstance(X, np.ndarray) and X.ndim == 2:
            y_data = X[:, 0]
            exog = X[:, 1:] if X.shape[1] > 1 else None
        else:
            raise ValueError("X must be 1D or 2D np.ndarray for TBATSModel.")

        estimator = TBATS(
            use_box_cox=self.use_box_cox,
            use_trend=self.use_trend,
            use_damped_trend=self.use_damped_trend,
            seasonal_periods=self.seasonal_periods,
            use_arma_errors=self.use_arma_errors,
            show_warnings=self.show_warnings,
            n_jobs=self.n_jobs
        )
        self.model_ = estimator.fit(y_data, exogenous=exog)
        self.fitted_ = True

    def predict(self, X: Any) -> np.ndarray:
        """
        予測を行う。BATSと同様に:
          - int: 先のステップ数 -> forecast(steps=int)
          - 2D array -> exog指定 + steps = row数

        Args:
            X (Any): int or np.ndarray

        Returns:
            np.ndarray: 予測値 (1次元配列)
        """
        if not self.fitted_:
            raise ValueError("TBATSModel is not fitted yet.")

        if isinstance(X, int):
            steps = X
            fc_vals = self.model_.forecast(steps=steps)
            return fc_vals
        elif isinstance(X, np.ndarray) and X.ndim == 2:
            steps = X.shape[0]
            exog = X
            fc_vals = self.model_.forecast(steps=steps, exogenous=exog)
            return fc_vals
        else:
            raise ValueError("X must be int or 2D array for TBATSModel predict")

    def save_model(self, filepath: str) -> None:
        """
        学習済みTBATSモデルをファイル保存。
        """
        if not self.fitted_ or self.model_ is None:
            raise ValueError("No fitted model to save.")
        joblib.dump(self.model_, filepath)

    def load_model(self, filepath: str) -> None:
        """
        モデルをファイルから読み込み。
        """
        self.model_ = joblib.load(filepath)
        self.fitted_ = True
