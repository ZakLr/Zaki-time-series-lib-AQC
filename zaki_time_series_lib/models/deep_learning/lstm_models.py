import torch
import torch.nn as nn

from typing import Optional

from zaki_time_series_lib.models.deep_learning.base_torch import BaseTorchWrapper


class _LSTMNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 dropout: float, bidirectional: bool, use_gru: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.D = 2 if bidirectional else 1

        rnn_cls = nn.GRU if use_gru else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        self.fc = nn.Linear(hidden_dim * self.D, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * self.D, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers * self.D, x.size(0), self.hidden_dim, device=x.device)

        if isinstance(self.rnn, nn.LSTM):
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)

        out = out[:, -1, :]
        out = self.fc(out)
        return out


class LSTMModel(BaseTorchWrapper):
    def __init__(self, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__("LSTM")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.params.update({
            "hidden_dim": hidden_dim, "num_layers": num_layers, "dropout": dropout,
            "bidirectional": False
        })

    def _build_model(self):
        return _LSTMNet(self._input_dim, self.hidden_dim, self.num_layers,
                        self.dropout, bidirectional=False, use_gru=False)


class BiLSTMModel(BaseTorchWrapper):
    def __init__(self, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__("BiLSTM")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.params.update({
            "hidden_dim": hidden_dim, "num_layers": num_layers, "dropout": dropout,
            "bidirectional": True
        })

    def _build_model(self):
        return _LSTMNet(self._input_dim, self.hidden_dim, self.num_layers,
                        self.dropout, bidirectional=True, use_gru=False)


class GRUModel(BaseTorchWrapper):
    def __init__(self, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__("GRU")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.params.update({
            "hidden_dim": hidden_dim, "num_layers": num_layers, "dropout": dropout,
        })

    def _build_model(self):
        return _LSTMNet(self._input_dim, self.hidden_dim, self.num_layers,
                        self.dropout, bidirectional=False, use_gru=True)
