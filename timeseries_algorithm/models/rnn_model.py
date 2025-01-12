import torch
import torch.nn as nn
import joblib
from typing import Any, Optional
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from .interface import BaseTimeSeriesModel

class RNNLightningModule(pl.LightningModule):
    """
    シンプルなRNN時系列予測。
    """
    def __init__(self, input_dim: int=1, hidden_size: int=32, lr: float=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.rnn = nn.RNN(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, h = self.rnn(x)
        out = out[:, -1, :]  # 最終時刻のみ
        out = self.fc(out)
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
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

class RNNModel(BaseTimeSeriesModel):
    def __init__(self, input_dim: int=1, hidden_size: int=32, lr: float=1e-3, max_epochs: int=10, batch_size: int=16):
        self.module = RNNLightningModule(input_dim, hidden_size, lr)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.fitted = False
        self.trainer: Optional[Trainer] = None

    def fit(self, X: Any, y: Any) -> None:
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.trainer = Trainer(max_epochs=self.max_epochs)
        self.trainer.fit(self.module, loader)
        self.fitted = True

    def predict(self, X: Any) -> Any:
        if not self.fitted:
            raise ValueError("Model not fitted yet.")
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        self.module.eval()
        with torch.no_grad():
            preds = self.module(X)
        return preds.squeeze(-1).numpy()

    def save_model(self, filepath: str) -> None:
        state_dict = self.module.state_dict()
        joblib.dump(state_dict, filepath)

    def load_model(self, filepath: str) -> None:
        state_dict = joblib.load(filepath)
        self.module.load_state_dict(state_dict)
        self.fitted = True
