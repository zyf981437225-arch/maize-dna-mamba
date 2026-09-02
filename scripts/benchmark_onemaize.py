#!/usr/bin/env python
"""Phase-0 benchmark for dynamic OneMaize data, BiMamba, BCW, and memory."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dna_runtime import (
    autocast_context,
    build_model,
    build_tokenizer,
    make_grad_scaler,
)
from src.dataloaders.datasets.onemaize_dataset import (
    OneMaizeRegionMLMDataset,
    collate_onemaize_mlm,
)
from src.dataloaders.datasets.onemaize_variant_dataset import (
    OneMaizeVariantTEMLMDataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--variant-data-dir",
        type=Path,
        help="Enable the schema-v4 explicit variant/TE Phase-II sampler",
    )
    parser.add_argument("--fasta-root")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--mode", choices=("both", "backbone", "memory"), default="both")
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=864)
    parser.add_argument("--n-layer", type=int, default=24)
    parser.add_argument("--memory-d-sum", type=int, default=64)
    parser.add_argument("--memory-d-mem", type=int, default=64)
    parser.add_argument("--memory-write-stride", type=int, default=6)
    parser.add_argument("--memory-read-stride", type=int, default=2)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--io-steps", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2222)
    parser.add_argument("--minimum-parameters", type=int, default=110_000_000)
    parser.add_argument("--maximum-parameters", type=int, default=130_000_000)
    return parser


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_dataset(args, tokenizer, samples: int, deterministic: bool):
    dataset_class = (
        OneMaizeVariantTEMLMDataset
        if args.variant_data_dir is not None
        else OneMaizeRegionMLMDataset
    )
    variant_args = (
        {"variant_data_dir": args.variant_data_dir}
        if args.variant_data_dir is not None
        else {}
    )
    return dataset_class(
        args.data_dir,
        tokenizer=tokenizer,
        split="train",
        context_length=args.context_length,
        samples_per_epoch=samples,
        reverse_complement_probability=0.5,
        mlm_probability=args.mlm_probability,
        deterministic=deterministic,
        seed=args.seed,
        fasta_root=args.fasta_root,
        **variant_args,
    )


def benchmark_input_pipeline(args, tokenizer) -> tuple[dict, tuple]:
    samples = max(args.batch_size, args.io_steps * args.batch_size)
    dataset = make_dataset(args, tokenizer, samples=samples, deterministic=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=partial(
            collate_onemaize_mlm, pad_token_id=int(tokenizer.pad_token_id)
        ),
    )
    start = time.perf_counter()
    first_batch = None
    measured_batches = 0
    measured_tokens = 0
    for batch in loader:
        if first_batch is None:
            first_batch = batch
        measured_batches += 1
        measured_tokens += int(batch[0].numel())
        if measured_batches >= args.io_steps:
            break
    elapsed = time.perf_counter() - start
    del loader
    gc.collect()
    dataset.close()
    if first_batch is None:
        raise RuntimeError("OneMaize DataLoader produced no batch")
    return (
        {
            "batches": measured_batches,
            "tokens": measured_tokens,
            "seconds": elapsed,
            "host_tokens_per_second": measured_tokens / elapsed,
            "num_workers": args.num_workers,
        },
        first_batch,
    )


def run_model_case(args, tokenizer, batch, use_memory: bool, device: torch.device) -> dict:
    input_ids, labels, metadata = batch
    input_ids = input_ids.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    attention_mask = metadata["attention_mask"].to(device, non_blocking=True)
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
    parameters = sum(parameter.numel() for parameter in model.parameters())
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=0.01)
    scaler = make_grad_scaler(device, args.precision)
    forward_times = []
    backward_times = []
    step_times = []
    losses = []
    for step in range(args.warmup_steps + args.steps):
        optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        step_start = time.perf_counter()
        with autocast_context(device, args.precision):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
        synchronize(device)
        forward_end = time.perf_counter()
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite Phase-0 loss: {loss}")
        scaler.scale(loss).backward()
        synchronize(device)
        backward_end = time.perf_counter()
        scaler.step(optimizer)
        scaler.update()
        synchronize(device)
        step_end = time.perf_counter()
        if step >= args.warmup_steps:
            forward_times.append(forward_end - step_start)
            backward_times.append(backward_end - forward_end)
            step_times.append(step_end - step_start)
            losses.append(float(loss.detach().cpu()))
    mean_step = statistics.mean(step_times)
    result = {
        "name": "backbone_plus_bcw_memory" if use_memory else "backbone_only",
        "use_memory": use_memory,
        "parameters": parameters,
        "parameter_target_met": (
            args.minimum_parameters <= parameters <= args.maximum_parameters
        ),
        "input_shape": list(input_ids.shape),
        "mean_step_seconds": mean_step,
        "mean_forward_seconds": statistics.mean(forward_times),
        "mean_backward_seconds": statistics.mean(backward_times),
        "model_tokens_per_second": input_ids.numel() / mean_step,
        "peak_gpu_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "peak_gpu_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
        ),
        "losses": losses,
    }
    del optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = build_parser().parse_args()
    if args.steps <= 0 or args.warmup_steps < 0 or args.io_steps <= 0:
        raise ValueError("steps/io-steps must be positive and warmup cannot be negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    tokenizer = build_tokenizer(args.context_length)
    io_result, batch = benchmark_input_pipeline(args, tokenizer)
    requested = {
        "both": (False, True),
        "backbone": (False,),
        "memory": (True,),
    }[args.mode]
    cases = []
    for use_memory in requested:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        cases.append(run_model_case(args, tokenizer, batch, use_memory, device))
    result = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": args.precision,
        "d_model": args.d_model,
        "n_layer": args.n_layer,
        "context_length": args.context_length,
        "sampler": "variant_te" if args.variant_data_dir is not None else "schema_v3_region_aware",
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.steps,
        "parameter_target": [args.minimum_parameters, args.maximum_parameters],
        "input_pipeline": io_result,
        "cases": cases,
    }
    if len(cases) == 2:
        base, memory = cases
        overhead = {
            "step_time_percent": 100.0
            * (memory["mean_step_seconds"] / base["mean_step_seconds"] - 1.0),
        }
        if base["peak_gpu_allocated_bytes"]:
            overhead["allocated_memory_percent"] = 100.0 * (
                memory["peak_gpu_allocated_bytes"]
                / base["peak_gpu_allocated_bytes"]
                - 1.0
            )
        else:
            overhead["allocated_memory_percent"] = None
        result["memory_overhead"] = overhead
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    if not all(case["parameter_target_met"] for case in cases):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
