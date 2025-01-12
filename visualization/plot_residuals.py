import plotly.graph_objects as go
import numpy as np
import plotly.subplots as sp

def plot_residual_acf_pacf(
    acf_vals: np.ndarray,
    pacf_vals: np.ndarray
) -> go.Figure:
    """
    残差の ACF, PACF を Plotly で2つのサブプロットに表示する例。

    Args:
        acf_vals (np.ndarray): shape (l+1,) の ACF値配列。
        pacf_vals (np.ndarray): shape (l+1,) の PACF値配列。

    Returns:
        go.Figure: Plotly図表オブジェクト。

    Raises:
        ValueError: acf_valsとpacf_valsの長さが異なる場合。
    """
    if len(acf_vals) != len(pacf_vals):
        raise ValueError(
            f"Length mismatch: acf_vals={len(acf_vals)}, pacf_vals={len(pacf_vals)}."
        )

    fig = sp.make_subplots(rows=1, cols=2, subplot_titles=["ACF", "PACF"])

    lags_acf = np.arange(len(acf_vals))
    lags_pacf = np.arange(len(pacf_vals))

    fig.add_trace(
        go.Bar(x=lags_acf, y=acf_vals, name="ACF"),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=lags_pacf, y=pacf_vals, name="PACF"),
        row=1, col=2
    )

    fig.update_layout(
        title="Residual ACF and PACF"
    )
    fig.update_xaxes(title_text="Lag", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=1, col=2)
    fig.update_yaxes(title_text="Value", row=1, col=2)

    return fig