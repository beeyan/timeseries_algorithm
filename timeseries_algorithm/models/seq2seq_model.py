# seq2seq_model.py

import torch
import torch.nn as nn
import joblib
from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import TensorDataset, DataLoader
from .interface import BaseTimeSeriesModel


class Seq2SeqEncoder(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return h, c

class Seq2SeqDecoder(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x, h, c):
        out, (h, c) = self.lstm(x, (h, c))
        out = self.fc(out)
        return out, h, c

class Seq2SeqLightningModule(pl.LightningModule):
    def __init__(self, enc_input_dim=1, dec_input_dim=1, hidden_size=32, out_seq_len=5, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = Seq2SeqEncoder(enc_input_dim, hidden_size)
        self.decoder = Seq2SeqDecoder(dec_input_dim, hidden_size)
        self.out_seq_len = out_seq_len
        self.criterion = nn.MSELoss()

    def forward(self, enc_x, dec_x):
        h, c = self.encoder(enc_x)
        out, _, _ = self.decoder(dec_x, h, c)
        return out

    def training_step(self, batch, batch_idx):
        enc_x, dec_in, dec_target = batch
        preds = self.forward(enc_x, dec_in)
        # preds: (batch, dec_seq_len, 1)
        loss = self.criterion(preds, dec_target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        enc_x, dec_in, dec_target = batch
        preds = self.forward(enc_x, dec_in)
        loss = self.criterion(preds, dec_target)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

class Seq2SeqModel(BaseTimeSeriesModel):
    def __init__(self, enc_input_dim=1, dec_input_dim=1, hidden_size=32, out_seq_len=5, lr=1e-3, max_epochs=10, batch_size=16):
        self.module = Seq2SeqLightningModule(enc_input_dim, dec_input_dim, hidden_size, out_seq_len, lr)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.fitted = False
        self.trainer: Optional[Trainer] = None
        self.out_seq_len = out_seq_len

    def fit(self, X: Any, y: Any) -> None:
        """
        X: (batch, enc_seq_len, enc_input_dim)
        y: (batch, dec_seq_len, 1)
        ここでは dec_in と dec_target を作り学習する必要があるが、サンプルでは準備済み前提
        """
        # 例: (enc_x, dec_in, dec_target) 三つ組
        # y を dec_target として使うなら dec_in = zeros or shift
        dataset = TensorDataset(*X, y)  # Xが(enc_x, dec_in)というタプルになっている想定
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.trainer = Trainer(max_epochs=self.max_epochs)
        self.trainer.fit(self.module, loader)
        self.fitted = True

    def predict(self, X: Any) -> Any:
        """
        X: tuple of (enc_x, dec_in) => shapes: (batch, enc_seq_len, enc_input_dim), (batch, dec_seq_len, dec_input_dim)
        Returns: (batch, dec_seq_len, 1)
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet.")
        enc_x, dec_in = X
        if not isinstance(enc_x, torch.Tensor):
            enc_x = torch.tensor(enc_x, dtype=torch.float32)
        if not isinstance(dec_in, torch.Tensor):
            dec_in = torch.tensor(dec_in, dtype=torch.float32)

        self.module.eval()
        with torch.no_grad():
            preds = self.module(enc_x, dec_in)
        return preds.numpy()

    def save_model(self, filepath: str) -> None:
        state_dict = self.module.state_dict()
        joblib.dump(state_dict, filepath)

    def load_model(self, filepath: str) -> None:
        state_dict = joblib.load(filepath)
        self.module.load_state_dict(state_dict)
        self.fitted = True
