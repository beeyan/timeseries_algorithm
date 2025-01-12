import numpy as np
import plotly.graph_objects as go

def plot_outliers_on_series(
    data: np.ndarray,
    outlier_indices: np.ndarray
) -> go.Figure:
    """
    時系列データ上に外れ値を示すプロットを、Plotlyで作成する。

    Args:
        data (np.ndarray): shape (T,) の1次元時系列データ。
        outlier_indices (np.ndarray): shape (K,), 外れ値インデックス配列。

    Returns:
        go.Figure: Plotly図表オブジェクト。
    """
    fig = go.Figure()

    # 通常部分
    fig.add_trace(
        go.Scatter(
            x=list(range(len(data))),
            y=data,
            mode="lines",
            name="Data"
        )
    )

    # 外れ値
    fig.add_trace(
        go.Scatter(
            x=outlier_indices,
            y=data[outlier_indices],
            mode="markers",
            marker=dict(color='red', size=8, symbol='x'),
            name="Outliers"
        )
    )
    fig.update_layout(
        title="Time Series with Outliers",
        xaxis_title="Time index",
        yaxis_title="Value"
    )
    return fig
