# gdbt_model.py

from typing import Any, Optional
from .interface import BaseTimeSeriesModel

import lightgbm as lgb
import xgboost as xgb
import catboost as cb


class LightGBMTimeSeriesModel(BaseTimeSeriesModel):
    """
    LightGBMを用いた時系列予測モデルの例。
    単純化のため X, y は (batch, feature_dim) として扱い、seq_len等をflattenなど事前加工済みと仮定。
    """
    def __init__(
        self,
        params: Optional[dict] = None,
        num_boost_round: int = 100
    ) -> None:
        """
        Args:
            params (dict, optional): LightGBMのハイパーパラメータ辞書。
            num_boost_round (int): ブーストラウンド数。
        """
        self.params = params if params is not None else {
            "objective": "regression",
            "metric": "rmse"
        }
        self.num_boost_round = num_boost_round
        self.model = None
        self.fitted = False

    def fit(self, X: Any, y: Any) -> None:
        """
        LightGBMで学習を行う。

        Args:
            X (Any): shape (N, feature_dim) など、flattenされた特徴量。
            y (Any): shape (N,) のターゲット。
        """
        # LightGBM用のDataset作成
        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_boost_round
        )
        self.fitted = True

    def predict(self, X: Any) -> Any:
        """
        予測を行う。

        Args:
            X (Any): shape (N, feature_dim)

        Returns:
            Any: shape (N,) の予測値。
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet.")
        preds = self.model.predict(X)
        return preds

    def save_model(self, filepath: str) -> None:
        """
        学習済みモデルをファイルに保存。
        """
        if self.model is None:
            raise ValueError("No model to save.")
        self.model.save_model(filepath)

    def load_model(self, filepath: str) -> None:
        """
        学習済みモデルをファイルから読み込み。
        """
        self.model = lgb.Booster(model_file=filepath)
        self.fitted = True


class XGBoostTimeSeriesModel(BaseTimeSeriesModel):
    """
    XGBoostを用いた時系列予測モデルの例。
    """
    def __init__(
        self,
        params: Optional[dict] = None,
        num_boost_round: int = 100
    ) -> None:
        self.params = params if params is not None else {
            "objective": "reg:squarederror",
            "eval_metric": "rmse"
        }
        self.num_boost_round = num_boost_round
        self.model = None
        self.fitted = False

    def fit(self, X: Any, y: Any) -> None:
        dtrain = xgb.DMatrix(X, label=y)
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round
        )
        self.fitted = True

    def predict(self, X: Any) -> Any:
        if not self.fitted:
            raise ValueError("Model not fitted yet.")
        dtest = xgb.DMatrix(X)
        preds = self.model.predict(dtest)
        return preds

    def save_model(self, filepath: str) -> None:
        if self.model is None:
            raise ValueError("No model to save.")
        self.model.save_model(filepath)

    def load_model(self, filepath: str) -> None:
        self.model = xgb.Booster()
        self.model.load_model(filepath)
        self.fitted = True


class CatBoostTimeSeriesModel(BaseTimeSeriesModel):
    """
    CatBoostを用いた時系列予測モデルの例。
    """
    def __init__(
        self,
        params: Optional[dict] = None,
        iterations: int = 100
    ) -> None:
        self.params = params if params is not None else {
            "loss_function": "RMSE",
            "eval_metric": "RMSE"
        }
        self.iterations = iterations
        self.model = cb.CatBoostRegressor(**self.params, iterations=iterations)
        self.fitted = False

    def fit(self, X: Any, y: Any) -> None:
        """
        CatBoostで学習。

        Args:
            X (Any): shape (N, feature_dim)
            y (Any): shape (N,)
        """
        self.model.fit(X, y)
        self.fitted = True

    def predict(self, X: Any) -> Any:
        if not self.fitted:
            raise ValueError("Model not fitted yet.")
        return self.model.predict(X)

    def save_model(self, filepath: str) -> None:
        if not self.fitted:
            raise ValueError("No model to save.")
        self.model.save_model(filepath)

    def load_model(self, filepath: str) -> None:
        self.model.load_model(filepath)
        self.fitted = True
