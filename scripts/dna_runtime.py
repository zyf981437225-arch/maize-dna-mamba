"""Shared runtime helpers for DNA smoke tests and benchmarks."""

from __future__ import annotations

from contextlib import nullcontext
import torch

from caduceus.configuration_caduceus import CaduceusConfig
from caduceus.modeling_caduceus import CaduceusForMaskedLM
from caduceus.tokenization_caduceus import CaduceusTokenizer


def build_tokenizer(window_size: int) -> CaduceusTokenizer:
    return CaduceusTokenizer(
        model_max_length=int(window_size),
        sequence_type="dna",
        add_special_tokens=False,
        padding_side="right",
    )


def build_model(
    tokenizer,
    *,
    d_model: int,
    n_layer: int,
    use_memory: bool,
    memory_d_sum: int,
    memory_d_mem: int,
    memory_write_stride: int,
    memory_read_stride: int,
):
    config = CaduceusConfig(
        d_model=int(d_model),
        n_layer=int(n_layer),
        vocab_size=len(tokenizer),
        ssm_cfg={
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "dt_rank": "auto",
            "dt_min": 0.001,
            "dt_max": 0.1,
            "dt_init": "random",
            "dt_scale": 1.0,
            "dt_init_floor": 1e-4,
            "conv_bias": True,
            "bias": False,
            "use_fast_path": True,
        },
        rms_norm=True,
        fused_add_norm=True,
        residual_in_fp32=False,
        pad_vocab_size_multiple=8,
        norm_epsilon=1e-5,
        initializer_cfg={
            "initializer_range": 0.02,
            "rescale_prenorm_residual": True,
            "n_residuals_per_layer": 1,
        },
        bidirectional=True,
        bidirectional_strategy="add",
        bidirectional_weight_tie=True,
        rcps=False,
        complement_map=tokenizer.complement_map,
        use_memory=bool(use_memory),
        memory_d_sum=int(memory_d_sum),
        memory_d_mem=int(memory_d_mem),
        memory_n_heads=4,
        memory_write_stride=int(memory_write_stride),
        memory_read_stride=int(memory_read_stride),
        memory_max_size=32,
        memory_persist_across_batches=False,
        pad_token_id=int(tokenizer.pad_token_id),
    )
    return CaduceusForMaskedLM(config)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.cuda.amp.autocast(dtype=dtype)


def make_grad_scaler(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision == "fp16"
    return torch.cuda.amp.GradScaler(enabled=enabled)
