import pytest
import numpy as np
import plotly.graph_objects as go
from typing import List

# visualizationモジュールのimport
from visualization.plot_backtest import plot_backtest_results
from visualization.plot_outliers import plot_outliers_on_series
from visualization.plot_covariate_influence import plot_covariate_influence
from visualization.plot_residuals import plot_residual_acf_pacf
from visualization.plot_correlation import plot_cross_correlation
from visualization.plot_clustering import plot_timeseries_clustering

class TestVisualization:
    """
    visualizationディレクトリのPlotly可視化関数をテスト。
    """

    def test_plot_backtest_results_normal(self):
        """
        正常系: バックテストMSEリストをplotlyでプロット。
        """
        mses = [0.1, 0.2, 0.15, 0.05]
        horizon = 5
        fig = plot_backtest_results(mses, horizon)
        assert isinstance(fig, go.Figure)

    def test_plot_backtest_results_empty(self):
        """
        異常系: 空のMSEリストを与えた場合。
        """
        mses: List[float] = []
        horizon = 5
        with pytest.raises(ValueError):
            # plot_backtest_results関数内でチェックを実装 => raise ValueError
            fig = plot_backtest_results(mses, horizon)

    def test_plot_outliers_on_series_normal(self):
        """
        正常系: 外れ値を指定した散布図を描画。
        """
        data = np.random.rand(50)
        outlier_idx = np.array([10, 20, 30])
        fig = plot_outliers_on_series(data, outlier_idx)
        assert isinstance(fig, go.Figure)

    def test_plot_outliers_on_series_invalid_index(self):
        """
        異常系: outlier_indicesが範囲外の場合。
        """
        data = np.random.rand(50)
        outlier_idx = np.array([100, 200])  # 範囲外
        with pytest.raises(IndexError):
            fig = plot_outliers_on_series(data, outlier_idx)

    def test_plot_covariate_influence_normal(self):
        """
        正常系: SHAP等の重要度を棒グラフで表示。
        """
        feature_names = ["temp", "holiday", "price"]
        shap_vals = np.array([[0.2, -0.3, 0.1], [0.1, 0.05, 0.2]])
        fig = plot_covariate_influence(feature_names, shap_vals)
        assert isinstance(fig, go.Figure)

    def test_plot_covariate_influence_mismatch_feature_len(self):
        """
        異常系: 特徴名の数と列数が不一致。
        """
        feature_names = ["temp", "holiday"]
        shap_vals = np.array([[0.2, -0.3, 0.1]])
        with pytest.raises(ValueError):
            # ここはplot_covariate_influenceの中でチェックする想定 => raise ValueError
            fig = plot_covariate_influence(feature_names, shap_vals)

    def test_plot_residual_acf_pacf_normal(self):
        """
        正常系: ACF,PACFを棒グラフで描画。
        """
        acf_vals = np.random.rand(11)
        pacf_vals = np.random.rand(11)
        fig = plot_residual_acf_pacf(acf_vals, pacf_vals)
        assert isinstance(fig, go.Figure)

    def test_plot_residual_acf_pacf_length_mismatch(self):
        """
        異常系: ACFとPACFの長さが異なるケース。
        """
        acf_vals = np.random.rand(10)
        pacf_vals = np.random.rand(11)
        with pytest.raises(ValueError):
            # plot_residual_acf_pacf で長さを合わせる必要 => raise ValueError
            fig = plot_residual_acf_pacf(acf_vals, pacf_vals)

    def test_plot_cross_correlation_normal(self):
        """
        正常系: クロス相関のstem図を描画。
        """
        lags = np.arange(-5, 6)
        c_vals = np.random.rand(11)
        fig = plot_cross_correlation(lags, c_vals)
        assert isinstance(fig, go.Figure)

    def test_plot_cross_correlation_mismatch(self):
        """
        異常系: lagsとc_valuesの長さが合わない。
        """
        lags = np.arange(-5, 6)
        c_vals = np.random.rand(10)  # 1つ足りない
        with pytest.raises(ValueError):
            fig = plot_cross_correlation(lags, c_vals)

    def test_plot_timeseries_clustering_normal(self):
        """
        正常系: クラスタリングされた複数時系列をPlotlyで表示。
        """
        data = [np.random.rand(20), np.random.rand(20), np.random.rand(20)]
        labels = np.array([0, 0, 1])
        fig = plot_timeseries_clustering(data, labels)
        assert isinstance(fig, go.Figure)

    def test_plot_timeseries_clustering_label_mismatch(self):
        """
        異常系: データ数とラベル数が不一致。
        """
        data = [np.random.rand(20), np.random.rand(20)]
        labels = np.array([0, 0, 1])  # 3つあるがデータは2つ
        with pytest.raises(ValueError):
            fig = plot_timeseries_clustering(data, labels)
