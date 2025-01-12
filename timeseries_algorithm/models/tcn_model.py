# tcn_model.py

import torch
import torch.nn as nn
import joblib
from typing import Any, Optional
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from .interface import BaseTimeSeriesModel


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNLayer(nn.Module):
    def __init__(self, channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(len(channels)-1):
            dilation_size = 2 ** i
            in_ch = channels[i]
            out_ch = channels[i+1]
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1)*dilation_size, dropout=dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class TCNLightningModule(pl.LightningModule):
    def __init__(self, seq_len: int=30, input_dim: int=1, hidden_ch: int=16, kernel_size: int=3, dropout: float=0.2, lr: float=1e-3):
        super().__init__()
        self.save_hyperparameters()
        channels = [input_dim, hidden_ch, hidden_ch]
        self.tcn = TCNLayer(channels, kernel_size, dropout)
        # 最終: flatten => fc
        self.fc = nn.Linear(hidden_ch*seq_len, 1)
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1,2)  # => (batch, input_dim, seq_len)
        out = self.tcn(x)     # => (batch, hidden_ch, seq_len)
        out = out.view(out.size(0), -1)
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


class TCNModel(BaseTimeSeriesModel):
    def __init__(self, seq_len=30, input_dim=1, hidden_ch=16, kernel_size=3, dropout=0.2, lr=1e-3, max_epochs=10, batch_size=16):
        self.module = TCNLightningModule(seq_len, input_dim, hidden_ch, kernel_size, dropout, lr)
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
