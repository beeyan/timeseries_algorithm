import plotly.graph_objects as go
import numpy as np


def plot_cross_correlation(
    lags: np.ndarray,
    c_values: np.ndarray
) -> go.Figure:
    """
    クロス相関値をstem風にPlotlyで可視化する。

    Args:
        lags (np.ndarray): shape (L,), ラグ配列
        c_values (np.ndarray): shape (L,), 各ラグの相関値

    Returns:
        go.Figure: Plotly図表オブジェクト。

    Raises:
        ValueError: lagsとc_valuesの長さが異なる場合。
    """
    if len(lags) != len(c_values):
        raise ValueError(
            f"Mismatch: len(lags)={len(lags)} != len(c_values)={len(c_values)}."
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=lags,
            y=c_values,
            mode='markers+lines',
            name="CCF"
        )
    )
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="black")
    fig.update_layout(
        title="Cross-correlation",
        xaxis_title="Lag",
        yaxis_title="Correlation"
    )
    return fig