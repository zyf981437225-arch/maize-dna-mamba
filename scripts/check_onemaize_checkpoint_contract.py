#!/usr/bin/env python
"""Validate OneMaize Phase-I initialization or Model-B resume provenance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.onemaize.checkpoint_contracts import (
    validate_allcultivar_resume_config,
    validate_phase1_initialization_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kind", choices=("phase1-init", "allcultivar-resume"), required=True)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("hyper_parameters")
    if config is None:
        raise ValueError("Checkpoint has no hyper_parameters provenance")
    if args.kind == "phase1-init":
        validate_phase1_initialization_config(config)
    else:
        if args.data_dir is None:
            raise ValueError("--data-dir is required for allcultivar-resume")
        validate_allcultivar_resume_config(config, args.data_dir)
    print(f"checkpoint_contract=PASS kind={args.kind}")


if __name__ == "__main__":
    main()
