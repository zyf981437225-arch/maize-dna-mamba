"""Low-overhead cross-layer memory reader."""

import torch
from torch import nn


class MemoryCrossAttention(nn.Module):
    """Inject a pooled summary of earlier-layer memory.

    The original prototype projected every sequence token and ran multi-head
    cross-attention. This reader instead summarizes the small memory bank once
    per sample, projects that summary, and broadcasts it through a conservative
    learned gate. Its learned matrix cost is independent of sequence length.
    """

    def __init__(self, d_model: int, d_mem: int, n_heads: int = 4):
        super().__init__()
        del n_heads
        self.memory_norm = nn.LayerNorm(d_mem)
        self.out_proj = nn.Linear(d_mem, d_model)
        self.gate = nn.Parameter(torch.tensor(-4.0))
        self.scale = 0.1

    def forward(self, hidden_states: torch.Tensor, memory: torch.Tensor):
        if hidden_states.ndim != 3 or memory.ndim != 3:
            raise ValueError("hidden_states and memory must have shape [B, L, D]")
        if hidden_states.shape[0] != memory.shape[0]:
            raise ValueError("hidden_states and memory must share the batch dimension")

        summary = self.memory_norm(memory).mean(dim=1)
        context = self.out_proj(summary).unsqueeze(1)
        return self.scale * torch.sigmoid(self.gate) * context
