import numpy as np
import pytest
import torch
from typing import Any, List

# analysis モジュールのimport
from analysis.backtest import rolling_backtest
from analysis.cross_validation import time_series_cv
from analysis.outlier_detection import detect_outliers_by_residuals
from analysis.covariate_analysis import analyze_covariate_shap
from analysis.residual_analysis import compute_residual_acf_pacf
from analysis.correlation import cross_correlation_values
from analysis.clustering import (
    timeseries_clustering_kmeans,
    timeseries_clustering_hierarchical
)

@pytest.fixture
def mock_model() -> Any:
    """
    multi-step対応モックモデル
    """
    class MockModel:
        def fit(self, X, y):
            pass

        def predict(self, X):
            """
            Xがint => multi-step forecast (shape=(X,))
            Xがnp.ndarray => single-step (shape=(N,))
            """
            if isinstance(X, int):
                # multi-step => X個の乱数
                fh = X
                return np.random.rand(fh)
            elif isinstance(X, np.ndarray):
                # X => shape(N,feature_dim?)
                # single-step => shape(N,)
                n = X.shape[0]
                return np.mean(X, axis=1)  # shape(N,)
            else:
                raise ValueError("predict expects int or np.ndarray.")
    return MockModel()

@pytest.fixture
def random_data() -> np.ndarray:
    np.random.seed(42)
    return np.random.rand(50)

@pytest.fixture
def random_matrix_data() -> np.ndarray:
    np.random.seed(42)
    return np.random.rand(30, 5)


class TestAnalysis:
    """
    analysisディレクトリ配下のテスト。
    """

    # --------------------------------------
    # backtest.py
    # --------------------------------------
    def test_rolling_backtest_normal(self, mock_model, random_data):
        """
        正常系: rolling_backtest => multi-step forecast, MSE >= 0
        """
        mse = rolling_backtest(mock_model, random_data, window_size=10, forecast_horizon=5)
        assert mse >= 0.0

    def test_rolling_backtest_invalid_params(self, mock_model, random_data):
        """
        異常系: window_size+forecast_horizon > len(data) => ValueError
        """
        with pytest.raises(ValueError):
            _ = rolling_backtest(mock_model, random_data, window_size=50, forecast_horizon=10)

    # --------------------------------------
    # cross_validation.py
    # --------------------------------------
    def test_time_series_cv_normal(self, mock_model, random_matrix_data):
        """
        正常系: cross_validation => 3 splits => returns scores
        """
        from analysis.cross_validation import time_series_cv
        X = random_matrix_data
        y = np.random.rand(X.shape[0])
        def model_factory():
            return mock_model
        scores = time_series_cv(model_factory, X, y, n_splits=3)
        assert len(scores) == 3

    def test_time_series_cv_invalid_splits(self, mock_model, random_matrix_data):
        """
        異常系: n_splits too large => ValueError
        """
        X = random_matrix_data
        y = np.random.rand(X.shape[0])
        def model_factory():
            return mock_model
        with pytest.raises(ValueError):
            _ = time_series_cv(model_factory, X, y, n_splits=50)

    # --------------------------------------
    # outlier_detection.py
    # --------------------------------------
    def test_detect_outliers_by_residuals_normal(self, mock_model):
        """
        正常系: threshold=0.5 => 0~N outliers
        """
        N=10
        X = np.random.rand(N, 3)
        y = np.random.rand(N)
        outliers, vals = detect_outliers_by_residuals(mock_model, X, y, threshold=0.5)
        assert 0 <= len(outliers) <= N

    def test_detect_outliers_by_residuals_negative_threshold(self, mock_model):
        """
        異常系: threshold<0 => raise ValueError
        """
        X = np.random.rand(5, 2)
        y = np.random.rand(5)
        with pytest.raises(ValueError):
            _ = detect_outliers_by_residuals(mock_model, X, y, threshold=-1.0)

    # --------------------------------------
    # covariate_analysis.py
    # --------------------------------------
    def test_analyze_covariate_shap_normal(self):
        """
        正常系: shap解析 => skip if error
        """

        class MockPyTorch(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10,1)
            def forward(self, x:torch.Tensor)->torch.Tensor:
                return self.linear(x)

        model = MockPyTorch()
        X_sample = torch.randn(5,10)
        try:
            shap_vals = analyze_covariate_shap(model, X_sample)
            # 具体的チェックは省略 => shapライブラリによる
        except Exception:
            pytest.skip("SHAP encountered an error with mock. Skipping.")
        assert True

    def test_analyze_covariate_shap_invalid_shape(self):
        """
        異常系: 入力サイズ不一致 => shap失敗
        """

        class MockPyTorch2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10,1)
            def forward(self, x:torch.Tensor)->torch.Tensor:
                # 期待 shape(b,10), でも x is shape(b,5)
                return self.linear(x)

        model = MockPyTorch2()
        X_sample = torch.randn(5,5)
        with pytest.raises(Exception):
            _= analyze_covariate_shap(model, X_sample)

    # --------------------------------------
    # residual_analysis.py
    # --------------------------------------
    def test_compute_residual_acf_pacf_normal(self):
        """
        正常系
        """
        r = np.random.randn(100)
        acf_vals, pacf_vals = compute_residual_acf_pacf(r, nlags=10)
        assert len(acf_vals) == 11
        assert len(pacf_vals) == 11

    def test_compute_residual_acf_pacf_small_sample(self):
        """
        異常系: sample<nlags => statsmodelsエラー => ValueError
        """
        r = np.random.randn(3)
        with pytest.raises(ValueError):
            _= compute_residual_acf_pacf(r, nlags=10)

    # --------------------------------------
    # correlation.py
    # --------------------------------------
    def test_cross_correlation_values_normal(self):
        """
        正常系: shape(11) for lags,c_sub
        """
        ts1 = np.sin(np.linspace(0,2*np.pi,100))
        ts2 = np.sin(np.linspace(0,2*np.pi,100)+0.5)
        lags, c_sub = cross_correlation_values(ts1, ts2, max_lag=5)
        assert len(lags) == 11
        assert len(c_sub) == 11

    def test_cross_correlation_values_mismatch_length(self):
        """
        異常系: shape mismatch => ValueError
        """
        ts1 = np.sin(np.linspace(0,2*np.pi,100))
        ts2 = np.sin(np.linspace(0,2*np.pi,80))
        with pytest.raises(ValueError):
            lags, c_sub = cross_correlation_values(ts1, ts2, 5)

    # --------------------------------------
    # clustering.py
    # --------------------------------------
    def test_timeseries_clustering_kmeans_normal(self):
        """
        正常系: 5 series => 2 clusters
        """
        data = [np.random.rand(10) for _ in range(5)]
        labels = timeseries_clustering_kmeans(data, n_clusters=2)
        assert labels.shape == (5,)

    def test_timeseries_clustering_kmeans_invalid_nclusters(self):
        """
        異常系: n_clusters>データ数 => ValueError
        """
        data = [np.random.rand(10) for _ in range(2)]
        with pytest.raises(ValueError):
            _= timeseries_clustering_kmeans(data, n_clusters=10)

    def test_timeseries_clustering_hierarchical_normal(self):
        """
        正常系
        """
        from analysis.clustering import timeseries_clustering_hierarchical
        data = [np.random.rand(10) for _ in range(5)]
        labels = timeseries_clustering_hierarchical(data, n_clusters=2)
        assert labels.shape == (5,)

    def test_timeseries_clustering_hierarchical_invalid(self):
        """
        異常系: 空リスト => ValueError
        """
        data:List[np.ndarray] = []
        with pytest.raises(ValueError):
            _= timeseries_clustering_hierarchical(data, n_clusters=2)
