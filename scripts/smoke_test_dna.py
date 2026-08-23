#!/usr/bin/env python
"""Run FASTA-prepared 10,240-bp DNA MLM through loss/backward/optimizer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dna_runtime import (
    autocast_context,
    build_model,
    load_batch,
    make_grad_scaler,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--window-size", type=int, default=10240)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--memory-d-sum", type=int, default=64)
    parser.add_argument("--memory-d-mem", type=int, default=64)
    parser.add_argument("--memory-write-stride", type=int, default=2)
    parser.add_argument("--memory-read-stride", type=int, default=1)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2222)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    tokenizer, input_ids, labels, metadata = load_batch(
        Path(args.data_dir),
        split=args.split,
        window_size=args.window_size,
        batch_size=args.batch_size,
        mlm_probability=args.mlm_probability,
        seed=args.seed,
    )
    if tuple(input_ids.shape) != (args.batch_size, args.window_size):
        raise AssertionError(f"Unexpected DNA input shape: {tuple(input_ids.shape)}")
    if "U" in tokenizer.get_vocab() or "T" not in tokenizer.get_vocab():
        raise AssertionError("DNA tokenizer alphabet is invalid")

    input_ids = input_ids.to(device)
    labels = labels.to(device)
    attention_mask = metadata["attention_mask"].to(device)
    model = build_model(
        tokenizer,
        d_model=args.d_model,
        n_layer=args.n_layer,
        use_memory=True,
        memory_d_sum=args.memory_d_sum,
        memory_d_mem=args.memory_d_mem,
        memory_write_stride=args.memory_write_stride,
        memory_read_stride=args.memory_read_stride,
    ).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=0.01)
    scaler = make_grad_scaler(device, args.precision)
    losses = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, args.precision):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError(f"DNA smoke loss is not finite: {loss}")
        if outputs.logits.shape[:2] != input_ids.shape:
            raise AssertionError(
                f"MLM logits lost sequence alignment: {tuple(outputs.logits.shape)}"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"Non-finite gradient in {name}")
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))

    summary = {
        "status": "PASS",
        "device": str(device),
        "input_shape": list(input_ids.shape),
        "logits_shape": list(outputs.logits.shape),
        "steps": args.steps,
        "losses": losses,
        "losses_finite": all(math.isfinite(value) for value in losses),
        "use_bidirectional": True,
        "use_memory": True,
        "rcps": False,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
