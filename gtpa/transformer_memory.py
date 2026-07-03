import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class CausalTransformer(nn.Module):
    """
    Causal transformer over a temporal window of agent latents.
    Input:  [B, K, N, D]  where K = temporal window, N = agents
    Output: [B, N, D]     (transformed latent for the last timestep)
    """
    def __init__(self, d_model: int, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 256, dropout: float = 0.1, max_len: int = 64):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        self.input_proj = nn.Linear(d_model, d_model) if d_model != d_model else nn.Identity()
        self.norm_in = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.out_proj = nn.Linear(d_model, d_model)
        self.norm_out = nn.LayerNorm(d_model)

    def _causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K, N, D = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, K, D)
        x = self.norm_in(self.input_proj(x))
        x = self.pos_encoder(x)
        mask = self._causal_mask(K, x.device)
        x = self.transformer(x, mask=mask, is_causal=False)
        x = x[:, -1]
        x = self.norm_out(self.out_proj(x))
        x = x.reshape(B, N, D)
        return x
