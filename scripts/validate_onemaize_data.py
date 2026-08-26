#!/usr/bin/env python
"""Validate OneMaize metadata, FASTA indexes, sampling, and MLM reads."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow.compute as pc


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from caduceus.tokenization_caduceus import CaduceusTokenizer
from src.dataloaders.datasets.onemaize_dataset import OneMaizeRegionMLMDataset
from src.onemaize import NAM26_GENOTYPES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--fasta-root", type=Path)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--sampling-trials", type=int, default=10_000)
    parser.add_argument("--fetch-samples", type=int, default=1)
    parser.add_argument("--sampling-tolerance", type=float, default=0.03)
    parser.add_argument("--formal", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("fasta_alphabet_audited", False):
        raise ValueError("Metadata must be built with the full FASTA alphabet audit")
    if args.sampling_trials <= 0 or args.fetch_samples <= 0:
        raise ValueError("sampling-trials and fetch-samples must be positive")
    if not 0.0 < args.sampling_tolerance < 1.0:
        raise ValueError("sampling-tolerance must be in (0, 1)")
    regions_table = pq.read_table(
        data_dir / manifest["files"]["regions"],
        columns=["region_id", "n_fraction"],
    )
    if len(regions_table) != int(manifest["region_count"]):
        raise ValueError("regions.parquet row count does not match manifest")
    observed_max_n = float(pc.max(regions_table["n_fraction"]).as_py())
    if observed_max_n > float(manifest["max_n_fraction"]) + 1e-12:
        raise ValueError("regions.parquet contains a candidate above max_n_fraction")
    distinct_region_ids = int(pc.count_distinct(regions_table["region_id"]).as_py())
    if distinct_region_ids != len(regions_table):
        raise ValueError("regions.parquet contains duplicate region_id values")
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
        if not manifest.get("expected_genotype_panel_validated", False):
            raise ValueError("Metadata did not validate the NAM26 founder panel")
        if "B73" not in manifest.get("required_train_genotypes", []):
            raise ValueError("Formal metadata did not require B73 in training")

    genomes = pq.read_table(data_dir / manifest["files"]["genomes"]).to_pylist()
    if args.formal:
        expected = {item.casefold() for item in NAM26_GENOTYPES}
        observed = {row["genotype"].casefold() for row in genomes}
        if observed != expected:
            raise ValueError("genomes.parquet does not contain the exact NAM26 panel")
        b73 = next(row for row in genomes if row["genotype"].casefold() == "b73")
        if b73["default_split"] != "train":
            raise ValueError("B73 must remain in the formal training split")
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
        expected_fractions = {
            "gene_centered": 0.5,
            "non_repeat": 0.3,
            "te_rich": 0.2,
        }
        sampled_fractions = {
            region_class: counts[region_class] / args.sampling_trials
            for region_class in expected_fractions
        }
        for region_class, expected in expected_fractions.items():
            if abs(sampled_fractions[region_class] - expected) > args.sampling_tolerance:
                raise ValueError(
                    f"{split}/{region_class} sampled fraction "
                    f"{sampled_fractions[region_class]:.4f} exceeds tolerance from "
                    f"{expected:.4f}"
                )
        fetched = []
        for index in range(args.fetch_samples):
            input_ids, labels = dataset[index]
            masked_fraction = float(
                labels.ne(tokenizer.pad_token_id).float().mean()
            )
            if not 0.10 <= masked_fraction <= 0.20:
                raise ValueError(
                    f"{split} sample {index} masked fraction {masked_fraction:.4f} "
                    "is inconsistent with 15% MLM"
                )
            fetched.append(
                {
                    "length": int(input_ids.numel()),
                    "masked_tokens": int(labels.ne(tokenizer.pad_token_id).sum()),
                    "masked_fraction": masked_fraction,
                }
            )
        report["splits"][split] = {
            "genotypes": dataset.genotypes,
            "candidate_regions": len(dataset.rows),
            "source_counts": dataset.source_counts,
            "sampled_fractions": sampled_fractions,
            "fetched": fetched,
        }
        dataset.close()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
