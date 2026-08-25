#!/usr/bin/env python
"""Validate OneMaize metadata, FASTA indexes, sampling, and MLM reads."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from caduceus.tokenization_caduceus import CaduceusTokenizer
from src.dataloaders.datasets.onemaize_dataset import OneMaizeRegionMLMDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--fasta-root", type=Path)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--sampling-trials", type=int, default=10_000)
    parser.add_argument("--fetch-samples", type=int, default=1)
    parser.add_argument("--formal", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if args.formal:
        if manifest.get("genotype_count") != 26:
            raise ValueError("Formal validation requires exactly 26 genotypes")
        if manifest.get("genotype_split_counts") != {
            "train": 23,
            "val": 1,
            "test": 2,
        }:
            raise ValueError("Formal validation requires genotype split 23/1/2")
        if not manifest.get("formal_split_validated", False):
            raise ValueError("Metadata was not built with --formal")

    genomes = pq.read_table(data_dir / manifest["files"]["genomes"]).to_pylist()
    fasta_root = None if args.fasta_root is None else args.fasta_root.resolve()
    missing = []
    for row in genomes:
        fasta = Path(row["fasta"])
        if fasta_root is not None:
            fasta = fasta_root / fasta.name
        required = [fasta, Path(f"{fasta}.fai")]
        if fasta.suffix.lower() == ".gz":
            required.append(Path(f"{fasta}.gzi"))
        missing.extend(str(path) for path in required if not path.is_file())
    if missing:
        raise FileNotFoundError("Missing FASTA/index files:\n" + "\n".join(missing))

    tokenizer = CaduceusTokenizer(
        model_max_length=args.context_length,
        sequence_type="dna",
        add_special_tokens=False,
    )
    report = {
        "data_dir": str(data_dir),
        "context_length": args.context_length,
        "genotype_count": manifest["genotype_count"],
        "splits": {},
    }
    for split in ("train", "val", "test"):
        dataset = OneMaizeRegionMLMDataset(
            data_dir,
            tokenizer=tokenizer,
            split=split,
            context_length=args.context_length,
            samples_per_epoch=max(args.sampling_trials, args.fetch_samples, 1),
            deterministic=True,
            reverse_complement_probability=0.5 if split == "train" else 0.0,
            fasta_root=None if fasta_root is None else str(fasta_root),
        )
        counts = Counter(
            dataset.sample_metadata(i)["region_class"]
            for i in range(args.sampling_trials)
        )
        fetched = []
        for index in range(args.fetch_samples):
            input_ids, labels = dataset[index]
            fetched.append(
                {
                    "length": int(input_ids.numel()),
                    "masked_tokens": int(labels.ne(tokenizer.pad_token_id).sum()),
                }
            )
        report["splits"][split] = {
            "genotypes": dataset.genotypes,
            "candidate_regions": len(dataset.rows),
            "source_counts": dataset.source_counts,
            "sampled_fractions": {
                region_class: counts[region_class] / args.sampling_trials
                for region_class in ("gene_centered", "non_repeat", "te_rich")
            },
            "fetched": fetched,
        }
        dataset.close()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
