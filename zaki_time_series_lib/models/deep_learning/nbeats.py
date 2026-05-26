import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

from zaki_time_series_lib.models.deep_learning.base_torch import BaseTorchWrapper


class _NBeatsBlock(nn.Module):
    def __init__(self, input_size: int, theta_size: int, hidden_dim: int = 256,
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = input_size
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, theta_size))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class _TrendModel(nn.Module):
    def __init__(self, forecast_len: int, backcast_len: int, degree: int = 2):
        super().__init__()
        self.forecast_len = forecast_len
        self.backcast_len = backcast_len
        self.degree = degree
        t = torch.linspace(-1, 1, backcast_len + forecast_len)
        poly = torch.stack([t ** i for i in range(degree + 1)], dim=-1)
        self.register_buffer('poly_basis', poly)

    def forward(self, theta):
        poly = self.poly_basis
        backcast = theta[:, :self.degree + 1] @ poly.T[:, :self.backcast_len]
        forecast = theta[:, self.degree + 1:] @ poly.T[:, self.backcast_len:]
        return backcast, forecast


class _SeasonalityModel(nn.Module):
    def __init__(self, forecast_len: int, backcast_len: int, num_harmonics: int = 6):
        super().__init__()
        self.forecast_len = forecast_len
        self.backcast_len = backcast_len
        t = torch.linspace(-1, 1, backcast_len + forecast_len)
        freq = torch.arange(1, num_harmonics + 1, dtype=torch.float) * torch.pi
        cos_waves = torch.cos(2 * freq.unsqueeze(1) * t.unsqueeze(0))
        sin_waves = torch.sin(2 * freq.unsqueeze(1) * t.unsqueeze(0))
        basis = torch.cat([cos_waves, sin_waves], dim=0).T
        self.register_buffer('basis', basis)

    def forward(self, theta):
        basis = self.basis
        basis_b = basis[:self.backcast_len, :]
        basis_f = basis[self.backcast_len:, :]
        backcast = theta @ basis_b.T
        forecast = theta @ basis_f.T
        return backcast, forecast


class _NBeatsNet(nn.Module):
    def __init__(self, seq_len: int, forecast_len: int = 1, stack_types: List[str] = None,
                 num_blocks: int = 3, hidden_dim: int = 256, num_layers: int = 4,
                 dropout: float = 0.1, degree: int = 2, num_harmonics: int = 6):
        super().__init__()
        self.seq_len = seq_len
        self.forecast_len = forecast_len

        if stack_types is None:
            stack_types = ['trend', 'seasonality']

        self.stacks = nn.ModuleList()
        for stack_type in stack_types:
            blocks = nn.ModuleList()
            for _ in range(num_blocks):
                if stack_type == 'trend':
                    theta_size = 2 * (degree + 1)
                    block = _NBeatsBlock(seq_len, theta_size, hidden_dim, num_layers, dropout)
                    interpret = _TrendModel(forecast_len, seq_len, degree)
                else:
                    theta_size = 2 * num_harmonics
                    block = _NBeatsBlock(seq_len, theta_size, hidden_dim, num_layers, dropout)
                    interpret = _SeasonalityModel(forecast_len, seq_len, num_harmonics)
                blocks.append(nn.ModuleDict({'fc': block, 'interpret': interpret}))
            self.stacks.append(blocks)

    def forward(self, x):
        x = x.squeeze(-1)
        backcast_total = torch.zeros_like(x)
        forecast_total = torch.zeros(x.size(0), self.forecast_len, device=x.device)

        for stack in self.stacks:
            for block in stack:
                theta = block['fc'](x)
                backcast, forecast = block['interpret'](theta)
                x = x - backcast
                backcast_total = backcast_total + backcast
                forecast_total = forecast_total + forecast

        return forecast_total


class NBeatsModel(BaseTorchWrapper):
    def __init__(self, num_blocks: int = 3, hidden_dim: int = 256,
                 num_layers: int = 4, dropout: float = 0.1, degree: int = 2,
                 num_harmonics: int = 6, stack_types: List[str] = None):
        super().__init__("NBeats")
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.degree = degree
        self.num_harmonics = num_harmonics
        self.stack_types = stack_types or ['trend', 'seasonality']
        self.params.update({
            "num_blocks": num_blocks, "hidden_dim": hidden_dim,
            "num_layers": num_layers, "dropout": dropout, "degree": degree,
            "num_harmonics": num_harmonics, "stack_types": self.stack_types
        })

    def _build_model(self):
        return _NBeatsNet(
            self._seq_len, 1, self.stack_types, self.num_blocks,
            self.hidden_dim, self.num_layers, self.dropout,
            self.degree, self.num_harmonics
        )
