"""Parameter-budget checks for the preserved Caduceus/Mamba-v1 backbone."""

from __future__ import annotations

import math


def estimate_caduceus_parameters(
    *,
    d_model: int,
    n_layer: int,
    vocab_size: int = 12,
    pad_vocab_size_multiple: int = 8,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    bidirectional: bool = True,
    bidirectional_weight_tie: bool = True,
    use_memory: bool = True,
    memory_d_sum: int = 64,
    memory_d_mem: int = 64,
) -> dict[str, int]:
    """Estimate unique trainable parameters for this repository's model.

    The calculation matches ``mamba_ssm.modules.mamba_simple.Mamba`` with
    ``dt_rank='auto'``, convolution bias enabled, linear biases disabled, and
    the repository's tied bidirectional projections. The A100 benchmark still
    records the instantiated parameter count as the final authority.
    """

    d_model = int(d_model)
    n_layer = int(n_layer)
    d_inner = int(expand) * d_model
    dt_rank = math.ceil(d_model / 16)

    in_projection = d_model * (2 * d_inner)
    convolution = d_inner * d_conv + d_inner
    x_projection = d_inner * (dt_rank + 2 * d_state)
    dt_projection = dt_rank * d_inner + d_inner
    state_parameters = d_inner * d_state + d_inner
    out_projection = d_inner * d_model
    one_mamba = (
        in_projection
        + convolution
        + x_projection
        + dt_projection
        + state_parameters
        + out_projection
    )

    if not bidirectional:
        mixer = one_mamba
    elif bidirectional_weight_tie:
        mixer = 2 * one_mamba - in_projection - out_projection
    else:
        mixer = 2 * one_mamba
    block_norm = d_model
    backbone_blocks = n_layer * (mixer + block_norm)

    padded_vocab = math.ceil(vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    embeddings = padded_vocab * d_model
    final_norm = d_model

    memory = 0
    if use_memory:
        shared_projection = d_model * memory_d_sum + memory_d_sum
        shared_norm = 2 * memory_d_sum
        gate_mlp = (
            4 * memory_d_sum * memory_d_sum
            + memory_d_sum
            + memory_d_sum * memory_d_sum
            + memory_d_sum
        )
        compressor = (
            memory_d_sum * memory_d_sum
            + memory_d_sum
            + memory_d_sum * memory_d_mem
            + memory_d_mem
        )
        writer_norm = 2 * memory_d_mem
        reader = (
            2 * memory_d_mem
            + memory_d_mem * d_model
            + d_model
            + 1
        )
        memory = (
            shared_projection
            + shared_norm
            + gate_mlp
            + compressor
            + writer_norm
            + reader
        )

    total = backbone_blocks + embeddings + final_norm + memory
    return {
        "total": total,
        "backbone_blocks": backbone_blocks,
        "embeddings": embeddings,
        "final_norm": final_norm,
        "memory": memory,
        "one_direction_mamba_per_layer": one_mamba,
        "unique_bidirectional_mixer_per_layer": mixer,
        "dt_rank": dt_rank,
        "padded_vocab_size": padded_vocab,
    }
