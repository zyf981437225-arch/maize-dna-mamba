"""Distributed coverage accounting for deterministic Phase-I windows."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.distributed as dist


def distributed_sampler_stats(num_regions: int, world_size: int, drop_last: bool = False) -> dict[str, int]:
    """Return the exact DistributedSampler draw/duplicate arithmetic."""
    n, w = int(num_regions), int(world_size)
    if n <= 0 or w <= 0:
        raise ValueError("num_regions and world_size must be positive")
    if drop_last:
        per_rank = n // w
    else:
        per_rank = (n + w - 1) // w
    global_draws = per_rank * w
    return {
        "per_rank_samples": per_rank,
        "global_samples_drawn": global_draws,
        "unique_regions": n,
        "duplicate_samples": max(0, global_draws - n),
    }


class Phase1CoverageTracker:
    """Track unique fixed windows and valid genomic bp seen in an epoch."""

    def __init__(self, dataset) -> None:
        self.num_regions = int(len(dataset.rows))
        self.total_valid_bp = int(dataset.total_valid_bp)
        self.valid_bp = torch.tensor(
            [int(row["valid_bp"]) for row in dataset.rows], dtype=torch.int64
        )
        self.is_tail = torch.tensor(
            [bool(row["is_tail"]) for row in dataset.rows], dtype=torch.bool
        )
        self.reset()

    def reset(self) -> None:
        self.samples_seen = 0
        self._seen = torch.zeros(self.num_regions, dtype=torch.int32)
        self._tail_seen = 0

    @property
    def has_seen(self) -> bool:
        return self.samples_seen > 0

    def update(self, region_ids: Iterable[int] | torch.Tensor) -> None:
        ids = torch.as_tensor(region_ids, dtype=torch.long).detach().cpu().reshape(-1)
        if ids.numel() == 0:
            return
        if int(ids.min()) < 0 or int(ids.max()) >= self.num_regions:
            raise ValueError("Phase-I region_id outside manifest range")
        self.samples_seen += int(ids.numel())
        self._seen[ids.unique()] = 1

    def compute(self, device: torch.device | str | None = None) -> dict[str, torch.Tensor]:
        device = torch.device(device or "cpu")
        seen = self._seen.to(device=device)
        samples = torch.tensor(float(self.samples_seen), device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(samples, op=dist.ReduceOp.SUM)
            dist.all_reduce(seen, op=dist.ReduceOp.MAX)
        unique = seen.sum().to(dtype=torch.float32)
        total = torch.tensor(float(self.num_regions), device=device)
        total_bp = torch.tensor(float(self.total_valid_bp), device=device)
        covered_bp = self.valid_bp.to(device=device)[seen.bool()].sum().to(dtype=torch.float32)
        tails = (seen.bool() & self.is_tail.to(device=device)).sum().to(dtype=torch.float32)
        return {
            "train/phase1_samples_seen": samples,
            "train/phase1_unique_regions": unique,
            "train/phase1_unique_fraction": unique / total,
            "train/phase1_duplicate_fraction": (samples - unique) / torch.clamp(samples, min=1.0),
            "train/phase1_genomic_bp_coverage": covered_bp / total_bp,
            "train/phase1_tail_sequences_seen": tails,
        }
