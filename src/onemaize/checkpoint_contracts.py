"""Checkpoint provenance gates for the two formal OneMaize branches."""

from __future__ import annotations

from pathlib import Path


def _get(mapping, *keys, default=None):
    value = mapping
    for key in keys:
        if value is None:
            return default
        if hasattr(value, "get"):
            value = value.get(key, default)
        else:
            return default
    return value


def validate_phase1_initialization_config(config) -> None:
    """Require the common B73 8K full-genome branch point."""

    errors = []
    if _get(config, "dataset", "mode") != "full_genome":
        errors.append("dataset.mode is not full_genome")
    if int(_get(config, "dataset", "context_length", default=-1)) != 8192:
        errors.append("dataset.context_length is not 8192")
    if _get(config, "dataset", "genotype", default="B73") != "B73":
        errors.append("dataset.genotype is not B73")
    if errors:
        raise ValueError("Not a B73 Phase-I initialization checkpoint: " + "; ".join(errors))


def validate_allcultivar_resume_config(config, data_dir: Path) -> None:
    """Reject a B73 Model-A checkpoint used as a Model-B exact resume."""

    errors = []
    if _get(config, "dataset", "mode", default="region_aware") != "region_aware":
        errors.append("dataset.mode is not region_aware")
    if int(_get(config, "dataset", "context_length", default=-1)) != 16384:
        errors.append("dataset.context_length is not 16384")
    stored_data_dir = _get(config, "dataset", "data_dir")
    if stored_data_dir is None:
        errors.append("checkpoint has no dataset.data_dir provenance")
    elif Path(str(stored_data_dir)).expanduser().resolve() != Path(data_dir).expanduser().resolve():
        errors.append("checkpoint dataset.data_dir differs from current all-cultivar metadata")
    if errors:
        raise ValueError("Not an exact all-cultivar Model-B resume checkpoint: " + "; ".join(errors))
