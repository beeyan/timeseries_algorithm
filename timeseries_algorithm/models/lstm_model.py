import pandas as pd
import joblib
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional, Any

from .interface import BaseTimeSeriesModel

class LSTMLightningModule(pl.LightningModule):
    """
    LSTMベースの時系列予測を行うLightningModule。
    単一ステップ予測を想定。
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_size: int = 32,
        lr: float = 1e-3
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向き計算: LSTM => 最終ステップ => 全結合 => (batch,1)
        """
        out, (_, _) = self.lstm(x)
        # 最終時刻だけ
        out = out[:, -1, :]
        out = self.fc(out)
        return out

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        学習ステップ: 損失を計算。
        """
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        """
        検証ステップ: 検証データでの損失を計算してログを取る。
        """
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        """
        オプティマイザの設定
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    

class LSTMModel(BaseTimeSeriesModel):
    """
    LSTMモデルを PyTorch Lightning でトレーニングし、
    BaseTimeSeriesModel インターフェイスを提供するクラス。
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_size: int = 32,
        lr: float = 1e-3,
        max_epochs: int = 10,
        batch_size: int = 16,
        early_stopping: bool = False
    ) -> None:
        """
        Args:
            input_dim (int, optional): 入力次元。デフォルトは1。
            hidden_size (int, optional): LSTMの隠れユニット数。デフォルトは32。
            lr (float, optional): 学習率。デフォルトは1e-3。
            max_epochs (int, optional): 学習の最大エポック数。デフォルトは10。
            batch_size (int, optional): ミニバッチサイズ。デフォルトは16。
            early_stopping (bool, optional): EarlyStoppingを使用するかどうか。デフォルトはFalse。
        """
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping

        # LightningModule インスタンス
        self.module = LSTMLightningModule(input_dim=input_dim, hidden_size=hidden_size, lr=lr)

        self.trainer: Optional[Trainer] = None
        self.fitted = False

    def fit(self, X: Any, y: Any) -> None:
        """
        PyTorch LightningのTrainerを用いて学習。

        Args:
            X (Any): shape (batch, seq_len, input_dim) のデータ
            y (Any): shape (batch, 1) のターゲット
        """
        # pandas DataFrame なら numpy に変換
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # numpy なら Tensor に変換
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
            X = X.unsqueeze(2)  # (batch, seq_len, 1)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        val_loader = None  # 簡易に検証セットなしの場合
        callbacks = []
        if self.early_stopping:
            callbacks.append(EarlyStopping(monitor="val_loss", patience=3, mode="min"))

        self.trainer = Trainer(
            max_epochs=self.max_epochs,
            callbacks=callbacks,
            enable_progress_bar=True
        )
        self.trainer.fit(self.module, loader, val_loader)
        self.fitted = True

    def predict(self, X: Any) -> Any:
        """
        学習済みLightningModuleを用いて推論を行う。

        Args:
            X (Any): shape (batch, seq_len, input_dim)

        Returns:
            Any: shape (batch,) の予測値
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet.")

        # pandas DataFrame なら numpy に変換
        if isinstance(X, pd.DataFrame):
            X = X.values

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
            X = X.unsqueeze(2)

        # LightningModuleを直接呼び出して推論
        self.module.eval()
        with torch.no_grad():
            preds = self.module(X)
        return preds.squeeze(-1).numpy()

    def save_model(self, filepath: str) -> None:
        """
        学習済みモデルの重みを保存する。
        """
        state_dict = self.module.state_dict()
        joblib.dump(state_dict, filepath)

    def load_model(self, filepath: str) -> None:
        """
        学習済みモデルの重みをロードする。
        """
        state_dict = joblib.load(filepath)
        self.module.load_state_dict(state_dict)
        self.fitted = True