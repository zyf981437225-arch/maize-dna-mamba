#!/usr/bin/env python
"""Validate deterministic schema-v4 sampling and real FASTA/MLM reads."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from caduceus.tokenization_caduceus import CaduceusTokenizer
from src.dataloaders.datasets.onemaize_variant_dataset import (
    OneMaizeVariantTEMLMDataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data-dir", type=Path, required=True)
    parser.add_argument("--variant-data-dir", type=Path, required=True)
    parser.add_argument("--fasta-root")
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2357)
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    tokenizer = CaduceusTokenizer(
        model_max_length=args.context_length,
        sequence_type="dna",
        add_special_tokens=False,
    )
    report = {"context_length": args.context_length, "splits": {}}
    for split in ("train", "val", "test"):
        dataset = OneMaizeVariantTEMLMDataset(
            args.base_data_dir,
            args.variant_data_dir,
            tokenizer=tokenizer,
            split=split,
            context_length=args.context_length,
            samples_per_epoch=args.samples,
            deterministic=True,
            seed=args.seed,
            reverse_complement_probability=0.0,
            fasta_root=args.fasta_root,
            return_metadata=True,
        )
        counts = Counter()
        first_pass = []
        for index in range(args.samples):
            inputs, labels, metadata = dataset[index]
            if inputs.numel() != args.context_length or labels.numel() != args.context_length:
                raise ValueError("Unexpected model-input length")
            if not torch.isfinite(inputs.float()).all() or not torch.isfinite(labels.float()).all():
                raise FloatingPointError("Non-finite token/label tensor")
            if not labels.ne(tokenizer.pad_token_id).any():
                raise ValueError("MLM sample has no supervised token")
            if metadata.get("variant_id") is not None:
                target_start = int(metadata["target_start"])
                target_end = int(metadata["target_end"])
                if target_start == target_end:
                    contains = metadata["crop_start"] <= target_start < metadata["crop_end"]
                else:
                    contains = (
                        metadata["crop_start"] <= target_start
                        and target_end <= metadata["crop_end"]
                    )
                if not contains:
                    raise ValueError(f"Variant target was lost: {metadata}")
            counts[metadata["sampling_class"]] += 1
            first_pass.append((inputs, labels, metadata))
        for index, first in enumerate(first_pass):
            second = dataset[index]
            if not first[0].equal(second[0]) or not first[1].equal(second[1]) or first[2] != second[2]:
                raise ValueError(f"{split} sample {index} is not deterministic")
        report["splits"][split] = {
            "genotypes": dataset.genotypes,
            "sampling_counts": dict(counts),
            "samples": args.samples,
        }
        dataset.close()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
