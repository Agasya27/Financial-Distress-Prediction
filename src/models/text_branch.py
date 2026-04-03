from __future__ import annotations

import torch
from torch import nn


class TextBranch(nn.Module):
    """
    Encodes FinBERT chunk embeddings with a lightweight BiLSTM.

    Input shape: (batch, chunks, 768)
    Output shape: (batch, out_dim)
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_size: int = 96,
        num_layers: int = 1,
        out_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 2, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 768)
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)  # (B, 2H)
        return self.proj(pooled)
