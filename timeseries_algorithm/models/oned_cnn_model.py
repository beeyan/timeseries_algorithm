import joblib
import torch
import torch.nn as nn
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional, Any

from .interface import BaseTimeSeriesModel

class CNNLightningModule(pl.LightningModule):
    """
    CNNベースの時系列予測を行うLightningModule。
    単一ステップ予測を想定。
    """
    def __init__(self, input_dim=1, seq_len=30, num_filters=16, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=3)
        self.fc = nn.Linear((num_filters - 3 + 1)*num_filters, 1)
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1,2)
        x = torch.relu(self.conv(x))
        x = x.reshape(x.size(0), -1)
        out = self.fc(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    

class CNNModel(BaseTimeSeriesModel):
    """
    CNNモデルを PyTorch Lightning でトレーニングし、
    BaseTimeSeriesModel インターフェイスを提供するクラス。
    """

    def __init__(
        self,
        input_dim: int = 1,
        lr: float = 1e-3,
        max_epochs: int = 10,
        batch_size: int = 16,
        early_stopping: bool = False
    ) -> None:
        """
        Args:
            input_dim (int, optional): 入力次元。デフォルトは1。
            lr (float, optional): 学習率。デフォルトは1e-3。
            max_epochs (int, optional): 学習の最大エポック数。デフォルトは10。
            batch_size (int, optional): ミニバッチサイズ。デフォルトは16。
            early_stopping (bool, optional): EarlyStoppingを使用するかどうか。デフォルトはFalse。
        """
        self.input_dim = input_dim
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping

        # LightningModule インスタンス
        self.module = CNNLightningModule(input_dim=input_dim, lr=lr)

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
            X = X.unsqueeze(-1)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        print(f"input shape in train X.shape : {X.shape}, y.shape : {y.shape}")

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

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
            X = X.unsqueeze(-1)
        
        print(f"input shape in predict X.shape : {X.shape}")

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