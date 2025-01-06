import pytest
import pandas as pd
import numpy as np
from timeseries_algorithm.data.dataset import TimeSeriesDataset

# ヘルパー関数: サンプルデータを生成
def create_sample_data() -> pd.DataFrame:
    dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
    np.random.seed(0)  # 再現性のためシードを固定
    values = np.random.rand(100) * 100
    data = pd.DataFrame({"date": dates, "sales": values})
    return data

# テスト: クラスの初期化（正常系） with date_col
def test_TimeSeriesDataset_init_with_date_col():
    """
    date_col を指定して TimeSeriesDataset を初期化した際に、
    インデックスが DatetimeIndex でソートされていることを確認するテスト。
    """
    data = create_sample_data()
    dataset = TimeSeriesDataset(data=data, target_col='sales', date_col='date')
    
    # インデックスが datetime であることを確認
    assert isinstance(dataset.data.index, pd.DatetimeIndex)
    
    # データがソートされていることを確認
    assert dataset.data.index.is_monotonic_increasing

# テスト: クラスの初期化（正常系） without date_col
def test_TimeSeriesDataset_init_without_date_col():
    """
    date_col を指定せずに TimeSeriesDataset を初期化した際に、
    インデックスが変更されていないことを確認するテスト。
    """
    data = create_sample_data()
    dataset = TimeSeriesDataset(data=data, target_col='sales')
    
    # インデックスがそのままであることを確認
    assert not isinstance(dataset.data.index, pd.DatetimeIndex)
    assert list(dataset.data.index) == list(range(len(data)))

# テスト: クラスの初期化（異常系） - 存在しない date_col を指定
def test_TimeSeriesDataset_init_invalid_date_col():
    """
    存在しない date_col を指定した際に、KeyError が発生することを確認するテスト。
    """
    data = create_sample_data()
    with pytest.raises(KeyError) as exc_info:
        TimeSeriesDataset(data=data, target_col='sales', date_col='invalid_date')
    assert "date_col 'invalid_date' はデータに存在しません。" in str(exc_info.value)

# テスト: クラスの初期化（異常系） - 存在しない target_col を指定
def test_TimeSeriesDataset_init_invalid_target_col():
    """
    存在しない target_col を指定した際に、KeyError が発生することを確認するテスト。
    """
    data = create_sample_data()
    with pytest.raises(KeyError) as exc_info:
        TimeSeriesDataset(data=data, target_col='invalid_sales')
    assert "target_col 'invalid_sales' はデータに存在しません。" in str(exc_info.value)

# テスト: split_train_test メソッド（正常系）
def test_split_train_test():
    """
    split_train_test メソッドが指定した split_date でデータを正しく分割することを確認するテスト。
    """
    data = create_sample_data()
    dataset = TimeSeriesDataset(data=data, target_col='sales', date_col='date')
    
    split_date = '2022-02-15'
    train_data, test_data = dataset.split_train_test(split_date)
    
    # トレーニングデータの最終日が split_date 以前であること
    assert train_data.index.max() <= pd.Timestamp(split_date)
    
    # テストデータの開始日が split_date 以降であること
    assert test_data.index.min() >= pd.Timestamp(split_date)
    
    # 行数の確認
    expected_train_rows = (pd.Timestamp(split_date) - pd.Timestamp('2022-01-01')).days + 1
    expected_test_rows = 100 - expected_train_rows + 1  # split_date を含むため +1
    assert len(train_data) == expected_train_rows
    assert len(test_data) == expected_test_rows

# テスト: split_train_test メソッド（異常系） - split_date がデータ範囲外
def test_split_train_test_split_date_out_of_range():
    """
    split_train_test メソッドにデータ範囲外の split_date を指定した際に、ValueError が発生することを確認するテスト。
    """
    data = create_sample_data()
    dataset = TimeSeriesDataset(data=data, target_col='sales', date_col='date')
    
    split_date_before = '2021-12-31'  # データより前
    with pytest.raises(ValueError) as exc_info_before:
        dataset.split_train_test(split_date_before)
    assert "split_date がデータの範囲外です。" in str(exc_info_before.value)
    
    split_date_after = '2022-04-11'  # データより後
    with pytest.raises(ValueError) as exc_info_after:
        dataset.split_train_test(split_date_after)
    assert "split_date がデータの範囲外です。" in str(exc_info_after.value)

# テスト: get_features_and_target メソッド（正常系）
def test_get_features_and_target():
    """
    get_features_and_target メソッドが特徴量とターゲットを正しく抽出することを確認するテスト。
    """
    data = create_sample_data()
    dataset = TimeSeriesDataset(data=data, target_col='sales', date_col='date')
    
    X, y = dataset.get_features_and_target(dataset.data)
    
    # X が sales 列を含まないこと
    assert 'sales' not in X.columns
    
    # y が sales 列のみであること
    assert y.name == 'sales'
    
    # 行数の確認
    assert len(X) == len(y)
    
    # 特徴量の列数を確認（date はインデックスとして設定されているため、特徴量はなし）
    assert len(X.columns) == 0

# テスト: get_features_and_target メソッド（異常系） - target_col が存在しない
def test_get_features_and_target_missing_target_col():
    """
    get_features_and_target メソッドに target_col が存在しない DataFrame を渡した際に、KeyError が発生することを確認するテスト。
    """
    data = create_sample_data()
    dataset = TimeSeriesDataset(data=data, target_col='sales', date_col='date')
    
    df_without_target = dataset.data.drop(columns=['sales'])
    with pytest.raises(KeyError) as exc_info:
        dataset.get_features_and_target(df_without_target)
    assert "target_col 'sales' は指定されたDataFrameに存在しません。" in str(exc_info.value)

# テスト: get_features_and_target メソッド（異常系） - 空の DataFrame を渡す
def test_get_features_and_target_empty_dataframe():
    """
    get_features_and_target メソッドに空の DataFrame を渡した際に、空の特徴量とターゲットが返されることを確認するテスト。
    """
    data = create_sample_data()
    dataset = TimeSeriesDataset(data=data, target_col='sales', date_col='date')
    
    empty_df = pd.DataFrame(columns=['date', 'sales'])
    X, y = dataset.get_features_and_target(empty_df)
    
    # X と y が空であることを確認
    assert X.empty
    assert y.empty

# テスト: split_train_test メソッド（正常系） - split_date が最小値または最大値
def test_split_train_test_split_date_on_boundary():
    """
    split_train_test メソッドに split_date をデータの最小日付または最大日付として指定した際に、
    正しくデータが分割されることを確認するテスト。
    """
    data = create_sample_data()
    dataset = TimeSeriesDataset(data=data, target_col='sales', date_col='date')
    
    # split_date を最小日付に設定
    min_date = data['date'].min().strftime('%Y-%m-%d')
    train_data, test_data = dataset.split_train_test(min_date)
    assert len(train_data) == 1
    assert len(test_data) == 100
    
    # split_date を最大日付に設定
    max_date = data['date'].max().strftime('%Y-%m-%d')
    train_data, test_data = dataset.split_train_test(max_date)
    assert len(train_data) == 100
    assert len(test_data) == 1