from __future__ import annotations

import torch
from torch import nn

from src.models.cvae import CVAE, cvae_loss
from src.models.graph_branch import GraphBranch
from src.models.tabular_branch import TabularBranch
from src.models.text_branch import TextBranch


class MultiModalFusionModel(nn.Module):
    """Mac-safe multimodal model: tabular + text + graph with attention-like fusion."""

    def __init__(
        self,
        tabular_in_dim: int,
        tabular_dim: int = 64,
        text_dim: int = 64,
        graph_dim: int = 32,
        fusion_hidden: int = 128,
        modality_dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.modality_dropout_p = float(modality_dropout_p)
        self.tabular_branch = TabularBranch(in_dim=tabular_in_dim, out_dim=tabular_dim)
        self.text_branch = TextBranch(out_dim=text_dim)
        self.graph_branch = GraphBranch(out_dim=graph_dim)

        total = tabular_dim + text_dim + graph_dim
        self.modality_gate = nn.Sequential(
            nn.Linear(total, 3),
            nn.Softmax(dim=-1),
        )
        self.cvae = CVAE(in_dim=total, latent_dim=32, hidden_dim=fusion_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(total, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, tab_x: torch.Tensor, text_x: torch.Tensor, graph_x: torch.Tensor) -> dict[str, torch.Tensor]:
        tab_h = self.tabular_branch(tab_x)
        txt_h = self.text_branch(text_x)
        gph_h = self.graph_branch(graph_x)

        if self.training and self.modality_dropout_p > 0.0:
            p = self.modality_dropout_p
            b = tab_h.size(0)
            dev = tab_h.device
            dt = tab_h.dtype
            tab0, txt0, gph0 = tab_h, txt_h, gph_h
            k_t = torch.bernoulli(torch.full((b, 1), 1.0 - p, device=dev, dtype=dt))
            k_x = torch.bernoulli(torch.full((b, 1), 1.0 - p, device=dev, dtype=dt))
            k_g = torch.bernoulli(torch.full((b, 1), 1.0 - p, device=dev, dtype=dt))
            tab_h = tab_h * k_t
            txt_h = txt_h * k_x
            gph_h = gph_h * k_g
            dead = (k_t * k_x * k_g).squeeze(1) < 0.5
            if dead.any():
                tab_h = tab_h.clone()
                txt_h = txt_h.clone()
                gph_h = gph_h.clone()
                tab_h[dead] = tab0[dead]
                txt_h[dead] = txt0[dead]
                gph_h[dead] = gph0[dead]

        fused = torch.cat([tab_h, txt_h, gph_h], dim=1)
        attn = self.modality_gate(fused)  # (B, 3)

        # Scale modality embeddings by learned gates.
        tab_s = tab_h * attn[:, 0:1]
        txt_s = txt_h * attn[:, 1:2]
        gph_s = gph_h * attn[:, 2:3]
        fused_scaled = torch.cat([tab_s, txt_s, gph_s], dim=1)

        x_hat, mu, logvar = self.cvae(fused_scaled)
        logits = self.classifier(fused_scaled).squeeze(1)

        return {
            "logits": logits,
            "attention": attn,
            "fused": fused_scaled,
            "x_hat": x_hat,
            "mu": mu,
            "logvar": logvar,
            "cvae_loss": cvae_loss(fused_scaled, x_hat, mu, logvar),
        }
