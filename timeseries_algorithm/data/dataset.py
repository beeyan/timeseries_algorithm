import pandas as pd
from typing import Tuple, Optional

class TimeSeriesDataset:
    """
    時系列データセットを管理するクラス。

    Attributes:
        data (pd.DataFrame): 時系列データ全体を保持するDataFrame。
        target_col (str): 予測対象の目的変数のカラム名。
    """

    def __init__(self, data: pd.DataFrame, target_col: str, date_col: Optional[str] = None):
        """
        TimeSeriesDatasetクラスのコンストラクタ。

        Args:
            data (pd.DataFrame): 時系列データ全体。
            target_col (str): 予測したい目的変数のカラム名。
            date_col (Optional[str]): 日付情報を含むカラム名。指定された場合、datetimeに変換しインデックスとする。

        Raises:
            KeyError: 指定されたdate_colがdataに存在しない場合。
            KeyError: 指定されたtarget_colがdataに存在しない場合。
        """
        if target_col not in data.columns:
            raise KeyError(f"target_col '{target_col}' はデータに存在しません。")
        
        self.data = data.copy()
        self.target_col = target_col

        if date_col:
            if date_col not in self.data.columns:
                raise KeyError(f"date_col '{date_col}' はデータに存在しません。")
            self.data[date_col] = pd.to_datetime(self.data[date_col])
            self.data.set_index(date_col, inplace=True)
            self.data.sort_index(inplace=True)

    def split_train_test(self, split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        指定した日付を境にデータをトレーニングデータとテストデータに分割します。

        Args:
            split_date (str): 分割の境界となる日付（例: '2022-01-01'）。

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: トレーニングデータとテストデータのタプル。

        Raises:
            ValueError: split_dateがデータの範囲外の場合。
        """
        split_timestamp = pd.to_datetime(split_date)
        if split_timestamp < self.data.index.min() or split_timestamp > self.data.index.max():
            raise ValueError("split_date がデータの範囲外です。")
        
        train_data = self.data.loc[:split_date].copy()
        test_data = self.data.loc[split_date:].copy()
        return train_data, test_data

    def get_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        指定されたDataFrameから特徴量とターゲットを抽出します。

        Args:
            df (pd.DataFrame): 特徴量とターゲットを抽出するDataFrame。

        Returns:
            Tuple[pd.DataFrame, pd.Series]: 特徴量DataFrameとターゲットSeriesのタプル。

        Raises:
            KeyError: target_colがDataFrameに存在しない場合。
        """
        if self.target_col not in df.columns:
            raise KeyError(f"target_col '{self.target_col}' は指定されたDataFrameに存在しません。")
        
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        return X, y