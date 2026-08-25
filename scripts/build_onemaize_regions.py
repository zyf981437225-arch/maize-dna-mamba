#!/usr/bin/env python
"""Build OneMaize annotation-aware candidate-region metadata."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.onemaize import GenomeInput, build_onemaize_index


def _load_input_manifest(path: Path) -> list[GenomeInput]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    with path.open("rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        required = {"genotype", "fasta", "genes_gff3", "te_gff3", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input manifest is missing columns: {sorted(missing)}")
        rows = []
        for row in reader:
            rows.append(
                GenomeInput(
                    genotype=row["genotype"],
                    fasta=Path(row["fasta"]),
                    genes_gff3=Path(row["genes_gff3"]),
                    te_gff3=Path(row["te_gff3"]),
                    split=row["split"],
                )
            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create genomes.parquet and regions.parquet for OneMaize hierarchical "
            "dynamic sampling. Use --input-manifest for the 26-genome corpus or "
            "the direct file flags for a one-genotype pilot."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-manifest", type=Path)
    source.add_argument("--genotype")
    parser.add_argument("--fasta", type=Path)
    parser.add_argument("--genes-gff3", type=Path)
    parser.add_argument("--te-gff3", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-context", type=int, default=8192)
    parser.add_argument("--extended-context", type=int, default=16384)
    parser.add_argument("--candidate-span", type=int, default=32768)
    parser.add_argument("--candidate-stride", type=int, default=16384)
    parser.add_argument("--gene-flank", type=int, default=5000)
    parser.add_argument("--repeat-threshold", type=float, default=0.5)
    parser.add_argument("--seqid-regex", default=r"^chr(?:[1-9]|10)$")
    parser.add_argument(
        "--val-seqid",
        action="append",
        default=[],
        help="Pilot-only chromosome/seqid assigned to validation; repeat as needed",
    )
    parser.add_argument(
        "--test-seqid",
        action="append",
        default=[],
        help="Pilot-only chromosome/seqid assigned to test; repeat as needed",
    )
    parser.add_argument(
        "--allow-gene-overlap-in-non-repeat",
        action="store_true",
        help="Keep non-repeat tiles that overlap protein-coding gene bodies",
    )
    parser.add_argument("--allow-missing-class", action="store_true")
    parser.add_argument(
        "--formal",
        action="store_true",
        help="Require the teacher-plan corpus: 26 genotypes split 23/1/2",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.input_manifest is not None:
        inputs = _load_input_manifest(args.input_manifest.expanduser().resolve())
    else:
        missing = [
            flag
            for flag, value in (
                ("--fasta", args.fasta),
                ("--genes-gff3", args.genes_gff3),
                ("--te-gff3", args.te_gff3),
            )
            if value is None
        ]
        if missing:
            raise SystemExit(
                "Direct one-genotype mode requires " + ", ".join(missing)
            )
        inputs = [
            GenomeInput(
                genotype=args.genotype,
                fasta=args.fasta,
                genes_gff3=args.genes_gff3,
                te_gff3=args.te_gff3,
                split=args.split,
            )
        ]
    manifest = build_onemaize_index(
        inputs,
        args.output_dir,
        primary_context=args.primary_context,
        extended_context=args.extended_context,
        candidate_span=args.candidate_span,
        candidate_stride=args.candidate_stride,
        gene_flank=args.gene_flank,
        repeat_threshold=args.repeat_threshold,
        seqid_regex=args.seqid_regex,
        val_seqids=args.val_seqid,
        test_seqids=args.test_seqid,
        exclude_gene_bodies_from_non_repeat=(
            not args.allow_gene_overlap_in_non_repeat
        ),
        require_all_classes=not args.allow_missing_class,
        expected_genotype_count=26 if args.formal else None,
        expected_split_counts={"train": 23, "val": 1, "test": 2}
        if args.formal
        else None,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
