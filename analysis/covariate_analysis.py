import torch
import shap
import numpy as np

def analyze_covariate_shap(
    model: torch.nn.Module,
    X_sample: torch.Tensor
) -> np.ndarray:
    """
    SHAPを用いて外因性特徴量などの影響度を解析するサンプル関数。

    Args:
        model (torch.nn.Module): PyTorchモデル (線形・MLP向き; RNN等は注意)。
        X_sample (torch.Tensor): shape (N, seq_len, input_dim) のサンプル入力データ。

    Returns:
        np.ndarray: SHAP値（形状はモデルやDeepExplainerに応じて変わる）。
    """
    # デモ向け：DeepExplainerを使用 (RNNやCNNに対しては互換性要チェック)
    # backgroundを少量サンプル
    background = X_sample[:10]
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(X_sample)
    return shap_values
