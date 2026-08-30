#!/usr/bin/env python3
"""Short Phase-I 8K I/O + forward/backward benchmark (single GPU or torchrun)."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from functools import partial
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dna_runtime import autocast_context, build_model, build_tokenizer, make_grad_scaler
from src.dataloaders.datasets.onemaize_phase1_dataset import (
    OneMaizePhase1FullGenomeMLMDataset,
    collate_onemaize_phase1_mlm,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--fasta", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--io-steps", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    p.add_argument("--d-model", type=int, default=864)
    p.add_argument("--n-layer", type=int, default=24)
    p.add_argument("--output-json", type=Path)
    return p


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parser().parse_args()
    if args.steps <= 0 or args.io_steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps, io-steps, and batch-size must be positive")
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        torch.distributed.init_process_group(backend="nccl")
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank if distributed else 0)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    tokenizer = build_tokenizer(8192)
    dataset = OneMaizePhase1FullGenomeMLMDataset(
        args.manifest, tokenizer=tokenizer, context_length=8192, window_size=8192,
        stride=8192, fasta_path=args.fasta, deterministic=False,
    )
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False) if distributed else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        shuffle=(sampler is None), num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=partial(collate_onemaize_phase1_mlm, pad_token_id=int(tokenizer.pad_token_id)),
    )
    io_start = time.perf_counter()
    io_batches = 0
    io_tokens = 0
    first_batch = None
    for batch in loader:
        first_batch = batch if first_batch is None else first_batch
        io_batches += 1
        io_tokens += int(batch[0].numel())
        if io_batches >= args.io_steps:
            break
    io_seconds = time.perf_counter() - io_start
    if first_batch is None:
        raise RuntimeError("Phase-I DataLoader produced no batch")
    model = build_model(
        tokenizer, d_model=args.d_model, n_layer=args.n_layer, use_memory=True,
        memory_d_sum=64, memory_d_mem=64, memory_write_stride=6, memory_read_stride=2,
    ).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=0.01)
    scaler = make_grad_scaler(device, args.precision)
    model.train()
    iterator = iter(loader)
    step_times = []
    losses = []
    for step in range(args.warmup_steps + args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            if sampler is not None:
                sampler.set_epoch(step)
            iterator = iter(loader)
            batch = next(iterator)
        input_ids, labels, metadata = batch
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        attention_mask = metadata["attention_mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        sync(device)
        start = time.perf_counter()
        with autocast_context(device, args.precision):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite Phase-I benchmark loss: {loss}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        sync(device)
        elapsed = time.perf_counter() - start
        if step >= args.warmup_steps:
            step_times.append(elapsed)
            losses.append(float(loss.detach().cpu()))
    mean_step = statistics.mean(step_times)
    if distributed:
        elapsed_tensor = torch.tensor(mean_step, device=device)
        torch.distributed.all_reduce(elapsed_tensor, op=torch.distributed.ReduceOp.MAX)
        mean_step = float(elapsed_tensor.item())
    per_rank_samples = (len(dataset) + world - 1) // world
    result = {
        "phase": "phase1_full_genome",
        "world_size": world,
        "rank": rank,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "precision": args.precision,
        "context_length": 8192,
        "batch_size_per_gpu": args.batch_size,
        "manifest_regions": len(dataset),
        "per_rank_sampler_samples": per_rank_samples,
        "io": {"batches": io_batches, "tokens": io_tokens, "seconds": io_seconds, "tokens_per_second": io_tokens / max(io_seconds, 1e-9)},
        "mean_step_seconds_max_rank": mean_step,
        "global_samples_per_second": world * args.batch_size / mean_step,
        "global_tokens_per_second": world * args.batch_size * 8192 / mean_step,
        "estimated_epoch_seconds": ((per_rank_samples + args.batch_size - 1) // args.batch_size) * mean_step,
        "losses": losses,
    }
    if rank == 0:
        rendered = json.dumps(result, indent=2, sort_keys=True)
        print(rendered)
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(rendered + "\n", encoding="utf-8")
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
