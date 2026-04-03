from __future__ import annotations

import torch
from torch import nn


class TabularBranch(nn.Module):
    """Small MLP tabular encoder for Mac-friendly multimodal training."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
