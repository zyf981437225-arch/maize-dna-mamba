"""Lightweight bidirectional-consistent memory writing."""

from __future__ import annotations

import torch
from torch import nn


class BidirectionalMemoryWriter(nn.Module):
    """Fuse aligned forward/backward summaries into one memory entry.

    Pooling happens before learned projections, so the expensive learned
    operations are independent of the RNA sequence length.
    """

    def __init__(
        self,
        d_model: int,
        d_sum: int = 64,
        d_mem: int = 64,
        pool: str = "mean",
        gate: str = "vector",
    ):
        super().__init__()
        if pool not in {"mean", "last", "cls"}:
            raise ValueError(f"Unknown pool: {pool}")
        if gate not in {"scalar", "vector"}:
            raise ValueError(f"Unknown gate: {gate}")

        self.pool = pool
        self.gate = gate
        self.shared_projection = nn.Linear(d_model, d_sum)
        self.shared_norm = nn.LayerNorm(d_sum)

        gate_dim = 1 if gate == "scalar" else d_sum
        self.gate_mlp = nn.Sequential(
            nn.Linear(4 * d_sum, d_sum),
            nn.GELU(),
            nn.Linear(d_sum, gate_dim),
        )
        self.compress = nn.Sequential(
            nn.Linear(d_sum, d_sum),
            nn.GELU(),
            nn.Linear(d_sum, d_mem),
        )
        self.norm = nn.LayerNorm(d_mem)

    def _pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.pool == "mean":
            if attention_mask is None:
                return hidden_states.mean(dim=1)
            weights = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
            numerator = (hidden_states * weights).sum(dim=1)
            denominator = weights.sum(dim=1).clamp_min(1.0)
            return numerator / denominator
        if self.pool == "last":
            return hidden_states[:, -1]
        return hidden_states[:, 0]

    def forward(
        self,
        h_fwd: torch.Tensor,
        h_bwd: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ):
        if h_fwd.shape != h_bwd.shape:
            raise ValueError("Forward and backward states must have identical shapes")
        if attn_mask is not None and attn_mask.shape != h_fwd.shape[:2]:
            raise ValueError("attn_mask must have shape [B, L]")

        pooled_fwd = self._pool(h_fwd, attn_mask)
        pooled_bwd = self._pool(h_bwd, attn_mask)
        projected = self.shared_norm(
            self.shared_projection(torch.cat((pooled_fwd, pooled_bwd), dim=0))
        )
        z_fwd, z_bwd = projected.chunk(2, dim=0)

        relation = torch.cat(
            (z_fwd, z_bwd, torch.abs(z_fwd - z_bwd), z_fwd * z_bwd),
            dim=-1,
        )
        direction_gate = torch.sigmoid(self.gate_mlp(relation))
        fused = direction_gate * z_fwd + (1.0 - direction_gate) * z_bwd
        entry = self.norm(self.compress(fused))

        return entry, {
            "s_fwd": z_fwd,
            "s_bwd": z_bwd,
            "gate": direction_gate,
            "s": fused,
        }
