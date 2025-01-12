import plotly.graph_objects as go
import numpy as np
from typing import List

def plot_covariate_influence(
    feature_names: List[str],
    shap_values: np.ndarray
) -> go.Figure:
    """
    SHAP値など、外因性変数の影響度を棒グラフで可視化する。

    Args:
        feature_names (List[str]): 特徴量名のリスト。
        shap_values (np.ndarray): shape (N, d) または shape (d,).

    Returns:
        go.Figure: Plotly図表オブジェクト。

    Raises:
        ValueError: feature_namesの数とshap_valuesの次元が対応しない場合。
    """
    # shap_valuesが 1次元(d,) の場合 => feature_namesと次元合わせ
    # shap_valuesが 2次元(N, d) の場合 => 平均絶対値を計算 => shape(d,)

    if shap_values.ndim == 2:
        d = shap_values.shape[1]
        # d features
        if len(feature_names) != d:
            raise ValueError(
                f"Mismatch: len(feature_names)={len(feature_names)} but shap_values.shape[1]={d}."
            )
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    elif shap_values.ndim == 1:
        d = shap_values.shape[0]
        if len(feature_names) != d:
            raise ValueError(
                f"Mismatch: len(feature_names)={len(feature_names)} but shap_values.shape[0]={d}."
            )
        mean_abs_shap = np.abs(shap_values)
    else:
        raise ValueError(
            f"shap_values must be 1D or 2D, got shape={shap_values.shape}."
        )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=mean_abs_shap,
            y=feature_names,
            orientation='h',
            name="Importance"
        )
    )
    fig.update_layout(
        title="Covariate Influence (SHAP)",
        xaxis_title="Mean |SHAP value|",
        yaxis_title="Features",
        yaxis=dict(autorange="reversed")
    )
    return fig