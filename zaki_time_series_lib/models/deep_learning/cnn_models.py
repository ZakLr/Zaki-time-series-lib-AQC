import torch
import torch.nn as nn
import torch.nn.functional as F

from zaki_time_series_lib.models.deep_learning.base_torch import BaseTorchWrapper


class _CNNNet(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, hidden_channels: int = 64,
                 kernel_size: int = 3, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i in range(num_layers):
            layers.append(nn.Conv1d(in_ch, hidden_channels, kernel_size, padding='same'))
            layers.append(nn.BatchNorm1d(hidden_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_ch = hidden_channels
        self.convs = nn.Sequential(*layers)
        self._to_linear = hidden_channels * seq_len
        self.fc = nn.Sequential(
            nn.Linear(self._to_linear, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CNNModel(BaseTorchWrapper):
    def __init__(self, hidden_channels: int = 64, kernel_size: int = 3,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__("CNN")
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.params.update({
            "hidden_channels": hidden_channels, "kernel_size": kernel_size,
            "num_layers": num_layers, "dropout": dropout
        })

    def _build_model(self):
        return _CNNNet(self._input_dim, self._seq_len, self.hidden_channels,
                       self.kernel_size, self.num_layers, self.dropout)


class _TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.resample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        residual = x if self.resample is None else self.resample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop1(out)
        out = self.bn2(self.conv2(out))
        out = self.drop2(out)
        return self.relu(out + residual[..., :out.size(-1)])


class _TCNNet(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, hidden_channels: int = 64,
                 num_layers: int = 3, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(_TCNBlock(in_ch, hidden_channels, dilation, kernel_size, dropout))
            in_ch = hidden_channels
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_channels * seq_len, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TCNModel(BaseTorchWrapper):
    def __init__(self, hidden_channels: int = 64, num_layers: int = 3,
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__("TCN")
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.params.update({
            "hidden_channels": hidden_channels, "num_layers": num_layers,
            "kernel_size": kernel_size, "dropout": dropout
        })

    def _build_model(self):
        return _TCNNet(self._input_dim, self._seq_len, self.hidden_channels,
                       self.num_layers, self.kernel_size, self.dropout)
