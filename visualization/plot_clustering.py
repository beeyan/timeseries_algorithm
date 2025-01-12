import plotly.graph_objects as go
import numpy as np
from typing import List


def plot_timeseries_clustering(
    data: List[np.ndarray],
    labels: np.ndarray
) -> go.Figure:
    """
    時系列クラスタリング結果をPlotly上で可視化する。
    各時系列を重ねて描画し、クラスタごとに色を変える簡易例。

    Args:
        data (List[np.ndarray]): 各要素が shape (T,) の時系列データ
        labels (np.ndarray): shape (N,), 各時系列のクラスタラベル

    Returns:
        go.Figure: Plotly図表オブジェクト

    Raises:
        ValueError: dataの長さとlabelsの長さが異なる場合。
    """
    if len(data) != len(labels):
        raise ValueError(
            f"Mismatch: len(data)={len(data)} != len(labels)={len(labels)}."
        )

    fig = go.Figure()
    unique_labels = np.unique(labels)
    colors = [
        "blue", "red", "green", "orange", "purple", 
        "brown", "magenta", "cyan", "black"
    ]

    for i, series in enumerate(data):
        c_idx = labels[i] % len(colors)
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(series)),
                y=series,
                mode="lines",
                name=f"TS{i}, cluster={labels[i]}",
                line=dict(color=colors[c_idx])
            )
        )

    fig.update_layout(
        title="Time Series Clustering",
        xaxis_title="Time index",
        yaxis_title="Value"
    )
    return fig