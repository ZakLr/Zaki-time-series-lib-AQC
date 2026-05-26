import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from zaki_time_series_lib.models.deep_learning.base_torch import BaseTorchWrapper


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class _TransformerEncoderNet(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 3, dim_feedforward: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = _PositionalEncoding(d_model, seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model * seq_len, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TransformerModel(BaseTorchWrapper):
    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 3,
                 dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__("Transformer")
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.params.update({
            "d_model": d_model, "nhead": nhead, "num_layers": num_layers,
            "dim_feedforward": dim_feedforward, "dropout": dropout
        })

    def _build_model(self):
        return _TransformerEncoderNet(
            self._input_dim, self._seq_len, self.d_model, self.nhead,
            self.num_layers, self.dim_feedforward, self.dropout
        )


class _ProbSparseAttention(nn.Module):
    def __init__(self, d_k: int, n_heads: int, top_k: int = 5):
        super().__init__()
        self.d_k = d_k
        self.n_heads = n_heads
        self.top_k = top_k

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        M = Q_K_sample.max(-1)[0] - Q_K_sample.div(1).sum(-1) / sample_k
        M_top = M.topk(n_top, sorted=False)[1]
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top

    def forward(self, Q, K, V):
        B, H, L_Q, d_k = Q.shape
        _, _, L_K, _ = K.shape
        sample_k = min(self.top_k, L_K)
        n_top = min(self.top_k, L_Q)
        Q_K, M_top = self._prob_QK(Q, K, sample_k, n_top)
        attn = F.softmax(Q_K / math.sqrt(self.d_k), dim=-1)
        attn_out = torch.matmul(attn, V)
        full_out = torch.zeros(B, H, L_Q, self.d_k, device=Q.device)
        full_out[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :] = attn_out
        return full_out


class _InformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float, top_k: int = 5):
        super().__init__()
        self.d_k = d_model // nhead
        self.nhead = nhead
        self.prob_attn = _ProbSparseAttention(self.d_k, nhead, top_k)
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        Q = self.W_Q(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        attn_out = self.prob_attn(Q, K, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        x = self.norm1(x + self.dropout(self.W_O(attn_out)))
        x = self.norm2(x + self.ff(x))
        return x


class _InformerNet(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2, dim_ff: int = 256,
                 dropout: float = 0.1, top_k: int = 5):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = _PositionalEncoding(d_model, seq_len)
        self.blocks = nn.ModuleList([
            _InformerBlock(d_model, nhead, dim_ff, dropout, top_k)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model * seq_len, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class InformerModel(BaseTorchWrapper):
    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 dim_ff: int = 256, dropout: float = 0.1, top_k: int = 5):
        super().__init__("Informer")
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.top_k = top_k
        self.params.update({
            "d_model": d_model, "nhead": nhead, "num_layers": num_layers,
            "dim_ff": dim_ff, "dropout": dropout, "top_k": top_k
        })

    def _build_model(self):
        return _InformerNet(
            self._input_dim, self._seq_len, self.d_model, self.nhead,
            self.num_layers, self.dim_ff, self.dropout, self.top_k
        )
