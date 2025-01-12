import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal
from timeseries_algorithm.features.create_features import (
    create_lag_features,
    create_rolling_features,
    add_time_features,
    add_differencing_features,
    add_exponential_moving_average,
    add_fft_features,
    add_seasonal_decomposition_features,
    add_autocorrelation_features
)
from typing import List

# フィクスチャ: サンプルデータの生成
@pytest.fixture
def sample_data() -> pd.DataFrame:
    """
    テスト用のサンプル時系列データを生成します。

    Returns:
        pd.DataFrame: 'date' 列と 'sales' 列を持つ100行のデータフレーム。
    """
    dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
    np.random.seed(0)  # 再現性のためシードを固定
    values = np.random.rand(100) * 100
    data = pd.DataFrame({"date": dates, "sales": values})
    return data

# テストクラス: create_lag_features 関数
class TestCreateLagFeatures:
    @pytest.mark.parametrize("lags, expected_columns", [
        ([1], ['date', 'sales', 'sales_lag1']),
        ([1, 7], ['date', 'sales', 'sales_lag1', 'sales_lag7']),
        ([], ['date', 'sales'])
    ])
    def test_create_lag_features_normal(self, sample_data, lags, expected_columns):
        """
        create_lag_features 関数が指定したラグ数だけ正しくカラムを追加することを確認するテスト。
        """
        df = create_lag_features(sample_data, target_col='sales', lags=lags)
        assert list(df.columns) == expected_columns
        
        # 各ラグの正確性を確認
        for lag in lags:
            expected_series = sample_data['sales'].shift(lag)
            assert_series_equal(df[f'sales_lag{lag}'].dropna(), expected_series.dropna(), check_names=False)
    
    @pytest.mark.parametrize("invalid_col", ['invalid_col1', 'invalid_col2'])
    def test_create_lag_features_invalid_target_col(self, sample_data, invalid_col):
        """
        create_lag_features 関数に存在しない target_col を指定した際に、KeyError が発生することを確認するテスト。
        """
        with pytest.raises(KeyError) as exc_info:
            create_lag_features(sample_data, target_col=invalid_col, lags=[1, 2])
        assert invalid_col in str(exc_info.value)
    
    def test_create_lag_features_empty_lags(self, sample_data):
        """
        create_lag_features 関数に空のラグリストを指定した際に、元のデータフレームがそのまま返されることを確認するテスト。
        """
        df = create_lag_features(sample_data, target_col='sales', lags=[])
        assert_frame_equal(df, sample_data)

# テストクラス: create_rolling_features 関数
class TestCreateRollingFeatures:
    @pytest.mark.parametrize("windows, expected_columns", [
        ([7], ['date', 'sales', 'sales_rollmean7', 'sales_rollstd7']),
        ([7, 30], ['date', 'sales', 'sales_rollmean7', 'sales_rollstd7', 'sales_rollmean30', 'sales_rollstd30']),
        ([], ['date', 'sales'])
    ])
    def test_create_rolling_features_normal(self, sample_data, windows, expected_columns):
        """
        create_rolling_features 関数が指定したウィンドウ幅で正しくローリング統計量を追加することを確認するテスト。
        """
        df = create_rolling_features(sample_data, target_col='sales', windows=windows)
        assert list(df.columns) == expected_columns
        
        # 各ウィンドウのローリング統計量の正確性を確認
        for w in windows:
            expected_rollmean = sample_data['sales'].rolling(window=w).mean()
            expected_rollstd = sample_data['sales'].rolling(window=w).std()
            assert_series_equal(df[f'sales_rollmean{w}'], expected_rollmean, check_names=False)
            assert_series_equal(df[f'sales_rollstd{w}'], expected_rollstd, check_names=False)
    
    @pytest.mark.parametrize("invalid_col", ['invalid_col1', 'invalid_col2'])
    def test_create_rolling_features_invalid_target_col(self, sample_data, invalid_col):
        """
        create_rolling_features 関数に存在しない target_col を指定した際に、KeyError が発生することを確認するテスト。
        """
        with pytest.raises(KeyError) as exc_info:
            create_rolling_features(sample_data, target_col=invalid_col, windows=[5])
        assert invalid_col in str(exc_info.value)
    
    def test_create_rolling_features_empty_windows(self, sample_data):
        """
        create_rolling_features 関数に空のウィンドウリストを指定した際に、元のデータフレームがそのまま返されることを確認するテスト。
        """
        df = create_rolling_features(sample_data, target_col='sales', windows=[])
        assert_frame_equal(df, sample_data)

# テストクラス: add_time_features 関数
class TestAddTimeFeatures:
    @pytest.mark.parametrize("date_index, has_datetime_index, expected_columns", [
        (True, True, [
            'sales', 'year', 'day_of_week', 'month', 'day',
            'is_Sunday', 'is_Saturday', 'is_month_start', 'is_month_end', 'is_japanese_holiday'
        ]),
        (False, False, ['date', 'sales']),
    ])
    def test_add_time_features(self, sample_data, date_index, has_datetime_index, expected_columns):
        """
        add_time_features 関数がdate_indexの有無に応じて正しく時間特徴量を追加することを確認するテスト。
        """
        if has_datetime_index:
            df_input = sample_data.set_index('date')
        else:
            df_input = sample_data.copy()
        
        df = add_time_features(df_input, date_index=date_index)
        assert list(df.columns) == expected_columns
        
        if date_index and has_datetime_index:
            # 'year' 列の値が正しいことを確認
            assert (df['year'] == 2022).all()
            
            # 'day_of_week' 列の値が0-6であることを確認
            assert df['day_of_week'].between(0, 6).all()
            
            # 'is_japanese_holiday' 列の値が0または1であることを確認
            assert df['is_japanese_holiday'].isin([0, 1]).all()
        else:
            # date_index=False または has_datetime_index=False の場合、時間特徴量が追加されていないことを確認
            pass
    
    def test_add_time_features_invalid_index(self, sample_data):
        """
        add_time_features 関数にDatetimeIndexを持たないデータフレームをdate_index=Trueで渡した際に、時間特徴量が追加されないことを確認するテスト。
        """
        df = add_time_features(sample_data, date_index=True)
        expected_columns = ['date', 'sales']
        assert list(df.columns) == expected_columns

# テストクラス: add_differencing_features 関数
class TestAddDifferencingFeatures:
    @pytest.mark.parametrize("order, expected_diff_name", [
        (1, 'diff_1'),
        (2, 'diff_2'),
    ])
    def test_add_differencing_features_normal(self, sample_data, order, expected_diff_name):
        """
        add_differencing_features 関数が指定した列に対して正しく差分特徴量を追加することを確認するテスト。
        """
        df = add_differencing_features(sample_data, column='sales', order=order)
        expected_columns = ['date', 'sales', expected_diff_name]
        assert list(df.columns) == expected_columns
        
        # 差分が正しく計算されていることを確認
        expected_diff = sample_data['sales'].diff(order)
        assert_series_equal(df[expected_diff_name], expected_diff, check_names=False)
    
    @pytest.mark.parametrize("invalid_col", ['invalid_col1', 'invalid_col2'])
    def test_add_differencing_features_invalid_column(self, sample_data, invalid_col):
        """
        add_differencing_features 関数に存在しない列名を指定した際に、KeyError が発生することを確認するテスト。
        """
        with pytest.raises(KeyError) as exc_info:
            add_differencing_features(sample_data, column=invalid_col, order=1)
        assert invalid_col in str(exc_info.value)

# テストクラス: add_exponential_moving_average 関数
class TestAddExponentialMovingAverage:
    @pytest.mark.parametrize("span, expected_span", [
        (3, 3),
        (5, 5),
    ])
    def test_add_exponential_moving_average_normal(self, sample_data, span, expected_span):
        """
        add_exponential_moving_average 関数が指定した列に対して正しくEMAを追加することを確認するテスト。
        """
        df = add_exponential_moving_average(sample_data, column='sales', span=span)
        expected_column = f'ema_{span}'
        expected_columns = ['date', 'sales', expected_column]
        assert list(df.columns) == expected_columns
        
        # EMAの計算を確認
        expected_ema = sample_data['sales'].ewm(span=span, adjust=False).mean()
        assert_series_equal(df[expected_column], expected_ema, check_names=False)
    
    @pytest.mark.parametrize("invalid_col", ['invalid_col1', 'invalid_col2'])
    def test_add_exponential_moving_average_invalid_column(self, sample_data, invalid_col):
        """
        add_exponential_moving_average 関数に存在しない列名を指定した際に、KeyError が発生することを確認するテスト。
        """
        with pytest.raises(KeyError) as exc_info:
            add_exponential_moving_average(sample_data, column=invalid_col, span=3)
        assert invalid_col in str(exc_info.value)

# テストクラス: add_fft_features 関数
class TestAddFFTFeatures:

        # テストクラス: add_fft_features のヘルパー関数
    def calculate_expected_fft(df: pd.DataFrame, column: str, order: int) -> List[float]:
        """
        FFTの期待される結果を計算するヘルパー関数。
        
        Args:
            df (pd.DataFrame): データフレーム。
            column (str): FFTを計算する列名。
            order (int): FFTの順序。
        
        Returns:
            List[float]: FFTの絶対値のリスト。
        """
        fft_result = np.fft.fft(df[column])
        return [np.abs(fft_result[i]) for i in range(1, order + 1)]
    
    @pytest.mark.parametrize("order, expected_columns_count", [
        (10, 12),  # 2 original + 10 FFT features
        (5, 7),
    ])
    def test_add_fft_features_normal(self, sample_data, order, expected_columns_count):
        """
        add_fft_features 関数が指定した列に対して正しくFFT特徴量を追加することを確認するテスト。
        """
        df = add_fft_features(sample_data, column='sales', order=order)
        expected_columns = ['date', 'sales'] + [f'fft_{i}' for i in range(1, order + 1)]
        assert list(df.columns) == expected_columns
        
        # FFTの結果が正しく追加されていることを確認
        fft_result = np.fft.fft(sample_data['sales'])
        for i in range(1, order + 1):
            expected_fft_value = np.abs(fft_result[i])
            assert df[f'fft_{i}'].iloc[0] == expected_fft_value
    
    @pytest.mark.parametrize("invalid_col", ['invalid_col1', 'invalid_col2'])
    def test_add_fft_features_invalid_column(self, sample_data, invalid_col):
        """
        add_fft_features 関数に存在しない列名を指定した際に、KeyError が発生することを確認するテスト。
        """
        with pytest.raises(KeyError) as exc_info:
            add_fft_features(sample_data, column=invalid_col, order=5)
        assert invalid_col in str(exc_info.value)


# テストクラス: add_seasonal_decomposition_features 関数
class TestAddSeasonalDecompositionFeatures:
    @pytest.mark.parametrize("model, period, expected_columns", [
        ('additive', 7, ['date', 'sales', 'trend', 'seasonal', 'residual']),
        ('multiplicative', 7, ['date', 'sales', 'trend', 'seasonal', 'residual']),
    ])
    def test_add_seasonal_decomposition_features_normal(self, sample_data, model, period, expected_columns):
        """
        add_seasonal_decomposition_features 関数が指定した列に対して正しく季節分解特徴量を追加することを確認するテスト。
        """
        df = add_seasonal_decomposition_features(sample_data, column='sales', model=model, period=period)
        assert list(df.columns) == expected_columns
        
        # 季節分解の結果が存在することを確認
        assert df['trend'].notna().all()
        assert df['seasonal'].notna().all()
        assert df['residual'].notna().all()
    
    @pytest.mark.parametrize("invalid_col", ['invalid_col1', 'invalid_col2'])
    def test_add_seasonal_decomposition_features_invalid_column(self, sample_data, invalid_col):
        """
        add_seasonal_decomposition_features 関数に存在しない列名を指定した際に、KeyError または ValueError が発生することを確認するテスト。
        """
        with pytest.raises(KeyError) as exc_info:
            add_seasonal_decomposition_features(sample_data, column=invalid_col, model='additive', period=7)
        assert invalid_col in str(exc_info.value)
    

# テストクラス: add_autocorrelation_features 関数
class TestAddAutocorrelationFeatures:
    @pytest.mark.parametrize("lags, expected_columns", [
        ([1, 2, 3], ['date', 'sales', 'autocorr_1', 'autocorr_2', 'autocorr_3']),
        ([1], ['date', 'sales', 'autocorr_1']),
        ([], ['date', 'sales']),
    ])
    def test_add_autocorrelation_features_normal(self, sample_data, lags, expected_columns):
        """
        add_autocorrelation_features 関数が指定した列に対して正しく自己相関特徴量を追加することを確認するテスト。
        """
        df = add_autocorrelation_features(sample_data, column='sales', lags=lags)
        assert list(df.columns) == expected_columns
        
        # 各ラグの自己相関値が正しく計算されていることを確認
        for lag in lags:
            expected_autocorr = sample_data['sales'].autocorr(lag=lag)
            assert df[f'autocorr_{lag}'].iloc[0] == expected_autocorr
    
    @pytest.mark.parametrize("invalid_col", ['invalid_col1', 'invalid_col2'])
    def test_add_autocorrelation_features_invalid_column(self, sample_data, invalid_col):
        """
        add_autocorrelation_features 関数に存在しない列名を指定した際に、KeyError が発生することを確認するテスト。
        """
        with pytest.raises(KeyError) as exc_info:
            add_autocorrelation_features(sample_data, column=invalid_col, lags=[1])
        assert invalid_col in str(exc_info.value)
    
    def test_add_autocorrelation_features_empty_lags(self, sample_data):
        """
        add_autocorrelation_features 関数に空のラグリストを指定した際に、元のデータフレームがそのまま返されることを確認するテスト。
        """
        df = add_autocorrelation_features(sample_data, column='sales', lags=[])
        assert_frame_equal(df, sample_data)
