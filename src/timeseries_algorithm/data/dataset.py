# dataset.py

import pandas as pd
from typing import Tuple

class TimeSeriesDataset:
    """
    時系列データセットを管理するクラスの例。
    - 内部で pandas.DataFrame を保持
    - 学習用/テスト用への分割などを行う
    """

    def __init__(self, data: pd.DataFrame, target_col: str, date_col: str = None):
        """
        Parameters
        ----------
        data : pd.DataFrame
            時系列データ全体
        target_col : str
            予測したい目的変数のカラム名
        date_col : str, optional
            日付情報を含むカラム名（あればdatetimeに変換し、indexにする）
        """
        self.data = data.copy()
        self.target_col = target_col

        if date_col and date_col in self.data.columns:
            self.data[date_col] = pd.to_datetime(self.data[date_col])
            self.data.set_index(date_col, inplace=True)
            self.data.sort_index(inplace=True)

    def split_train_test(self, split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        指定した日付を境にデータを分割して返す (例: '2022-01-01')
        """
        train_data = self.data.loc[:split_date].copy()
        test_data = self.data.loc[split_date:].copy()
        return train_data, test_data

    def get_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        入力: 任意のDataFrame
        出力: 特徴量X, ターゲットy
        """
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        return X, y