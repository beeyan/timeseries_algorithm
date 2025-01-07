import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from filterpy.monte_carlo import systematic_resample
from .base_model import BaseTimeSeriesModel

class KalmanFilterModel(BaseTimeSeriesModel):
    """
    カルマンフィルタを用いた状態空間モデルの実装。
    """

    def __init__(self, k_states=2):
        """
        Parameters
        ----------
        k_states : int
            状態空間の状態数。
        """
        self.k_states = k_states
        self.kf = None
        self.filtered_state = None

    def fit(self, X, y):
        """
        モデルを学習する。ここでは単純な線形トレンドモデルを仮定。

        Parameters
        ----------
        X : None or array-like
            外因性変数はサポートしない。
        y : pd.Series or np.ndarray
            ターゲット時系列データ。
        """
        if X is not None:
            raise ValueError("KalmanFilterModel does not support exogenous variables.")

        if isinstance(y, pd.Series):
            y = y.values

        # 状態空間モデルの定義（ここでは線形トレンドモデル）
        self.kf = KalmanFilter(k_endog=1, k_states=self.k_states, initialization='approximate_diffuse')

        # 観測方程式
        self.kf['design'] = np.eye(1, self.k_states)

        # 状態遷移方程式（単純なランダムウォーク）
        self.kf['transition'] = np.eye(self.k_states)

        # 観測ノイズ
        self.kf['obs_cov'] = np.eye(1) * 1.0

        # 状態ノイズ
        self.kf['state_cov'] = np.eye(self.k_states) * 0.1

        # 初期状態
        self.kf.initialize_known(
            initial_state=np.zeros(self.k_states),
            initial_state_cov=np.eye(self.k_states) * 10
        )

        # フィルタリングの実行
        self.filtered_state, _, _, _ = self.kf.filter(y)

    def predict(self, X, steps=5):
        """
        予測を行う。未来の状態を予測し、観測値を推定。

        Parameters
        ----------
        X : None or array-like
            外因性変数はサポートしない。
        steps : int
            予測ステップ数。

        Returns
        -------
        y_pred : np.ndarray
            予測結果。
        """
        if X is not None:
            raise ValueError("KalmanFilterModel does not support exogenous variables.")

        if self.kf is None or self.filtered_state is None:
            raise ValueError("Model has not been fitted yet.")

        y_pred = []
        state = self.filtered_state[-1]

        for _ in range(steps):
            # 状態遷移
            state = self.kf['transition'] @ state
            # 観測値の予測
            y_forecast = self.kf['design'] @ state
            y_pred.append(y_forecast[0])

        return np.array(y_pred)

    def save_model(self, filepath: str):
        """
        モデルを保存する。
        """
        if self.kf is None or self.filtered_state is None:
            raise ValueError("No fitted model to save.")
        joblib.dump({
            "k_states": self.k_states,
            "kf_params": {
                "design": self.kf['design'],
                "transition": self.kf['transition'],
                "obs_cov": self.kf['obs_cov'],
                "state_cov": self.kf['state_cov'],
            },
            "filtered_state": self.filtered_state
        }, filepath)

    def load_model(self, filepath: str):
        """
        モデルをロードする。
        """
        data = joblib.load(filepath)
        self.k_states = data["k_states"]
        self.filtered_state = data["filtered_state"]

        self.kf = KalmanFilter(k_endog=1, k_states=self.k_states, initialization='approximate_diffuse')
        self.kf['design'] = data["kf_params"]["design"]
        self.kf['transition'] = data["kf_params"]["transition"]
        self.kf['obs_cov'] = data["kf_params"]["obs_cov"]
        self.kf['state_cov'] = data["kf_params"]["state_cov"]

        self.kf.initialize_known(
            initial_state=np.zeros(self.k_states),
            initial_state_cov=np.eye(self.k_states) * 10
        )


class ParticleFilterModel(BaseTimeSeriesModel):
    """
    パーティクルフィルタを用いたカスタム状態空間モデル。
    """

    def __init__(self, num_particles=100, resample_threshold=0.5):
        """
        Parameters
        ----------
        num_particles : int
            パーティクル数。
        resample_threshold : float
            効率低下時のリサンプリング閾値（通常は粒子のエフィシエンシー）。
        """
        self.num_particles = num_particles
        self.resample_threshold = resample_threshold
        self.particles = None
        self.weights = None

    def initialize_particles(self, initial_state, initial_cov):
        """
        パーティクルを初期化する。

        Parameters
        ----------
        initial_state : array-like
            初期状態の平均。
        initial_cov : array-like
            初期状態の共分散。
        """
        self.particles = np.random.multivariate_normal(initial_state, initial_cov, self.num_particles)
        self.weights = np.ones(self.num_particles) / self.num_particles

    def transition_model(self, state):
        """
        状態遷移モデル。ここでは簡単なランダムウォークを仮定。

        Parameters
        ----------
        state : array-like
            現在の状態。

        Returns
        -------
        new_state : array-like
            次の状態。
        """
        # ランダムウォーク: x_t = x_{t-1} + noise
        noise = np.random.normal(0, 1, size=state.shape)
        return state + noise

    def observation_model(self, state, observation):
        """
        観測モデル。ここではガウス分布を仮定。

        Parameters
        ----------
        state : array-like
            現在の状態。
        observation : float
            観測値。

        Returns
        -------
        likelihood : float
            観測値の尤度。
        """
        # y = x + noise
        sigma = 1.0  # 観測ノイズの標準偏差
        return (1.0 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-0.5 * ((observation - state) ** 2) / sigma**2)

    def fit(self, X, y=None):
        """
        パーティクルフィルタを初期化する。教師なし学習として実装。

        Parameters
        ----------
        X : None or array-like
            外因性変数はサポートしない。
        y : pd.Series or np.ndarray
            観測データ。
        """
        if X is not None:
            raise ValueError("ParticleFilterModel does not support exogenous variables.")

        if isinstance(y, pd.Series):
            y = y.values

        # 初期状態の仮定
        initial_state = np.mean(y)
        initial_cov = np.var(y) * np.eye(1)

        self.initialize_particles(np.array([initial_state]), initial_cov)

        for obs in y:
            # パーティクルの移行
            self.particles = np.array([self.transition_model(p) for p in self.particles])

            # パーティクルの重み更新
            self.weights *= np.array([self.observation_model(p, obs) for p in self.particles])
            self.weights += 1.e-300  # 重みが0になるのを防ぐ
            self.weights /= np.sum(self.weights)

            # エフィシエンシーの計算
            neff = 1. / np.sum(self.weights ** 2)

            if neff < self.num_particles * self.resample_threshold:
                # リサンプリング
                indices = systematic_resample(self.weights)
                self.particles = self.particles[indices]
                self.weights.fill(1.0 / self.num_particles)

    def predict(self, X, steps=5):
        """
        予測を行う。パーティクルの移行と観測をシミュレート。

        Parameters
        ----------
        X : None or array-like
            外因性変数はサポートしない。
        steps : int
            予測ステップ数。

        Returns
        -------
        y_pred : np.ndarray
            予測値の平均。
        """
        if X is not None:
            raise ValueError("ParticleFilterModel does not support exogenous variables.")

        y_pred = []
        for _ in range(steps):
            # パーティクルの移行
            self.particles = np.array([self.transition_model(p) for p in self.particles])

            # 観測値の予測
            y_step = self.particles.mean(axis=0)
            y_pred.append(y_step[0])

            # パーティクルの重み更新（予測では観測がないため重みは一様）
            self.weights.fill(1.0 / self.num_particles)

        return np.array(y_pred)

    def save_model(self, filepath: str):
        """
        モデルの状態を保存する。
        """
        if self.particles is None or self.weights is None:
            raise ValueError("No fitted model to save.")
        joblib.dump({
            "num_particles": self.num_particles,
            "resample_threshold": self.resample_threshold,
            "particles": self.particles,
            "weights": self.weights
        }, filepath)

    def load_model(self, filepath: str):
        """
        モデルの状態をロードする。
        """
        data = joblib.load(filepath)
        self.num_particles = data["num_particles"]
        self.resample_threshold = data["resample_threshold"]
        self.particles = data["particles"]
        self.weights = data["weights"]