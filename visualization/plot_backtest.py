import plotly.graph_objects as go
from typing import List

def plot_backtest_results(
    mses: List[float],
    horizon: int
) -> go.Figure:
    """
    ローリング検証(バックテスト)で得られたMSE推移をPlotlyで可視化する。

    Args:
        mses (List[float]): 各ローリング区間でのMSEリスト。
        horizon (int): 予測ホライズン（表示用）。

    Returns:
        go.Figure: Plotly図表オブジェクト。

    Raises:
        ValueError: mses が空の場合。
    """
    if len(mses) == 0:
        raise ValueError("MSE list is empty. Cannot plot backtest results.")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(mses))),
            y=mses,
            mode="lines+markers",
            name="MSE"
        )
    )
    fig.update_layout(
        title=f"Rolling Backtest MSE (Horizon={horizon})",
        xaxis_title="Rolling iteration",
        yaxis_title="MSE"
    )
    return fig
