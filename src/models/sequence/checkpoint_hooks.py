"""Checkpoint state-dict hooks without optional backbone dependencies."""

from __future__ import annotations

import torch


def load_backbone(model, state_dict, freeze_backbone=False, ignore_head=True):
    """Load the pretrained backbone while retaining task-specific heads."""

    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        state_dict, "model."
    )
    current = model.state_dict()
    updated = {}
    for key in sorted(current):
        loaded = state_dict.get(key)
        if loaded is None:
            raise KeyError(f"Missing key in pretrained model: {key}")
        if (ignore_head and "head" in key) or "decoder" in key:
            value = current[key]
        else:
            value = loaded
        updated[f"model.{key}"] = value

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    return updated


def load_matching_backbone(model, state_dict, freeze_backbone=False):
    """Warm-start only shape-compatible backbone tensors."""

    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        state_dict, "model."
    )
    updated = {
        f"model.{key}": loaded
        for key, value in model.state_dict().items()
        if (loaded := state_dict.get(key)) is not None
        and loaded.shape == value.shape
    }
    if not updated:
        raise ValueError("No shape-compatible backbone tensors were found in the checkpoint")
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    return updated
