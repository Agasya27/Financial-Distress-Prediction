from __future__ import annotations

import torch
from torch import nn


class GraphBranch(nn.Module):
    """
    Lightweight graph-context encoder.

    Uses a scalar graph proxy (degree/centrality-like feature) and maps it to embedding space.
    """

    def __init__(self, in_dim: int = 1, hidden_dim: int = 32, out_dim: int = 32, dropout: float = 0.1) -> None:
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
