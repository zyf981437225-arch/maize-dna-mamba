#!/usr/bin/env python
"""Benchmark 10,240-bp bidirectional DNA-Mamba with and without memory."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

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
    parser.add_argument("--output-json")
    parser.add_argument("--mode", choices=("both", "backbone", "memory"), default="both")
    parser.add_argument("--window-size", type=int, default=10240)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--memory-d-sum", type=int, default=64)
    parser.add_argument("--memory-d-mem", type=int, default=64)
    parser.add_argument("--memory-write-stride", type=int, default=6)
    parser.add_argument("--memory-read-stride", type=int, default=2)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2222)
    return parser


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_case(args, tokenizer, batch, use_memory: bool) -> dict:
    input_ids, labels, attention_mask = batch
    device = input_ids.device
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = build_model(
        tokenizer,
        d_model=args.d_model,
        n_layer=args.n_layer,
        use_memory=use_memory,
        memory_d_sum=args.memory_d_sum,
        memory_d_mem=args.memory_d_mem,
        memory_write_stride=args.memory_write_stride,
        memory_read_stride=args.memory_read_stride,
    ).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=0.01)
    scaler = make_grad_scaler(device, args.precision)

    forward_times = []
    backward_times = []
    step_times = []
    losses = []
    total_steps = args.warmup_steps + args.steps
    if args.steps <= 0 or args.warmup_steps < 0:
        raise ValueError("steps must be positive and warmup_steps cannot be negative")

    for step in range(total_steps):
        optimizer.zero_grad(set_to_none=True)
        _synchronize(device)
        step_start = time.perf_counter()
        forward_start = step_start
        with autocast_context(device, args.precision):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
        _synchronize(device)
        forward_end = time.perf_counter()
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite benchmark loss: {loss}")
        scaler.scale(loss).backward()
        _synchronize(device)
        backward_end = time.perf_counter()
        scaler.step(optimizer)
        scaler.update()
        _synchronize(device)
        step_end = time.perf_counter()
        if step >= args.warmup_steps:
            forward_times.append(forward_end - forward_start)
            backward_times.append(backward_end - forward_end)
            step_times.append(step_end - step_start)
            losses.append(float(loss.detach().cpu()))

    mean_step = statistics.mean(step_times)
    return {
        "name": "backbone_plus_bcw_memory" if use_memory else "backbone_only",
        "use_memory": use_memory,
        "input_shape": list(input_ids.shape),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "mean_step_seconds": mean_step,
        "mean_forward_seconds": statistics.mean(forward_times),
        "mean_backward_seconds": statistics.mean(backward_times),
        "tokens_per_second": input_ids.numel() / mean_step,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "losses": losses,
    }


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    tokenizer, input_ids, labels, metadata = load_batch(
        Path(args.data_dir),
        split="train",
        window_size=args.window_size,
        batch_size=args.batch_size,
        mlm_probability=args.mlm_probability,
        seed=args.seed,
    )
    batch = (
        input_ids.to(device),
        labels.to(device),
        metadata["attention_mask"].to(device),
    )

    cases = []
    requested = {
        "both": (False, True),
        "backbone": (False,),
        "memory": (True,),
    }[args.mode]
    for use_memory in requested:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        cases.append(run_case(args, tokenizer, batch, use_memory))

    result = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": args.precision,
        "d_model": args.d_model,
        "n_layer": args.n_layer,
        "window_size": args.window_size,
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.steps,
        "cases": cases,
    }
    if len(cases) == 2:
        base, memory = cases
        overhead = {
            "step_time_percent": 100.0
            * (memory["mean_step_seconds"] / base["mean_step_seconds"] - 1.0),
        }
        if base["peak_gpu_memory_bytes"] and memory["peak_gpu_memory_bytes"]:
            overhead["peak_memory_percent"] = 100.0 * (
                memory["peak_gpu_memory_bytes"] / base["peak_gpu_memory_bytes"] - 1.0
            )
        else:
            overhead["peak_memory_percent"] = None
        result["memory_overhead"] = overhead
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
