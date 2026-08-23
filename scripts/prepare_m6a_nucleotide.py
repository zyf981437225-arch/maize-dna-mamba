"""Prepare full-mRNA nucleotide-level m6A labels for RNA-Mamba.

The source tables contain one full transcript sequence and one equally long
binary mask per transcript.  This script validates that contract, converts T
to U for the RNA tokenizer, performs a deterministic gene-level split, and
writes compact gzip-compressed JSONL records.  Only positive positions are
stored because every other adenosine is a labelled negative.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, NamedTuple, Optional, Sequence, TextIO

def _raise_csv_field_limit() -> None:
    """Allow the very long comma-separated mask field to be parsed."""

    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _stable_gene_split(
    gene_id: str,
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> str:
    """Assign all versioned isoforms of one gene to the same split."""

    normalized_gene = gene_id.strip().split(".", 1)[0]
    digest = hashlib.sha256(f"{seed}:{normalized_gene}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < train_fraction:
        return "train"
    if value < train_fraction + val_fraction:
        return "val"
    return "test"


def _open_text(path: Path, mode: str = "rt") -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def _parse_bool(value: object, field: str, transcript_id: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{transcript_id}: {field} must be True or False, got {value!r}")


class MaskRecord(NamedTuple):
    length: int
    positive_positions: tuple[int, ...]
    cds_boundary_reliable: bool
    mrna_coordinate_system_reliable: bool


REGION_FIELDS = {
    "utr5_start",
    "utr5_end",
    "cds_start",
    "cds_end",
    "utr3_start",
    "utr3_end",
    "utr5_length",
    "cds_length",
    "utr3_length",
    "utr5_sequence",
    "cds_sequence",
    "utr3_sequence",
    "cds_starts_with_atg",
    "cds_ends_with_stop_codon",
    "cds_length_multiple_of_3",
    "cds_boundary_reliable",
    "mrna_coordinate_system_reliable",
}


def _rna_sequence(value: object) -> str:
    return str(value).strip().upper().replace("T", "U")


def _validate_transcript_regions(
    row: Mapping[str, object],
    transcript_id: str,
    sequence: str,
    declared_length: int,
    mask: MaskRecord,
) -> Mapping[str, object]:
    """Validate the 0-based half-open 5'UTR/CDS/3'UTR contract."""

    coordinates = {
        field: int(row[field])
        for field in (
            "utr5_start",
            "utr5_end",
            "cds_start",
            "cds_end",
            "utr3_start",
            "utr3_end",
        )
    }
    lengths = {
        field: int(row[field])
        for field in ("utr5_length", "cds_length", "utr3_length")
    }
    regions = {
        "utr5": _rna_sequence(row["utr5_sequence"]),
        "cds": _rna_sequence(row["cds_sequence"]),
        "utr3": _rna_sequence(row["utr3_sequence"]),
    }

    expected_boundaries = (
        coordinates["utr5_start"] == 0
        and coordinates["utr5_end"] == coordinates["cds_start"]
        and coordinates["cds_end"] == coordinates["utr3_start"]
        and coordinates["utr3_end"] == declared_length
    )
    if not expected_boundaries:
        raise ValueError(
            f"{transcript_id}: UTR/CDS coordinates must be contiguous 0-based half-open intervals"
        )
    if not (
        0
        <= coordinates["utr5_start"]
        <= coordinates["utr5_end"]
        <= coordinates["cds_end"]
        <= coordinates["utr3_end"]
    ):
        raise ValueError(f"{transcript_id}: invalid UTR/CDS coordinate ordering")

    interval_lengths = {
        "utr5_length": coordinates["utr5_end"] - coordinates["utr5_start"],
        "cds_length": coordinates["cds_end"] - coordinates["cds_start"],
        "utr3_length": coordinates["utr3_end"] - coordinates["utr3_start"],
    }
    if interval_lengths != lengths:
        raise ValueError(f"{transcript_id}: declared region lengths disagree with boundaries")
    if any(len(regions[name]) != lengths[f"{name}_length"] for name in regions):
        raise ValueError(f"{transcript_id}: region sequence length disagrees with its declaration")
    if sequence != regions["utr5"] + regions["cds"] + regions["utr3"]:
        raise ValueError(f"{transcript_id}: transcript_sequence != 5'UTR + CDS + 3'UTR")
    if (
        sequence[coordinates["utr5_start"] : coordinates["utr5_end"]] != regions["utr5"]
        or sequence[coordinates["cds_start"] : coordinates["cds_end"]] != regions["cds"]
        or sequence[coordinates["utr3_start"] : coordinates["utr3_end"]] != regions["utr3"]
    ):
        raise ValueError(f"{transcript_id}: region coordinates do not reproduce region sequences")

    cds = regions["cds"]
    starts_with_aug = cds.startswith("AUG")
    ends_with_stop = cds.endswith(("UAA", "UAG", "UGA"))
    length_multiple_of_three = len(cds) % 3 == 0
    declared_cds_checks = {
        "cds_starts_with_atg": _parse_bool(
            row["cds_starts_with_atg"], "cds_starts_with_atg", transcript_id
        ),
        "cds_ends_with_stop_codon": _parse_bool(
            row["cds_ends_with_stop_codon"], "cds_ends_with_stop_codon", transcript_id
        ),
        "cds_length_multiple_of_3": _parse_bool(
            row["cds_length_multiple_of_3"], "cds_length_multiple_of_3", transcript_id
        ),
    }
    observed_cds_checks = {
        "cds_starts_with_atg": starts_with_aug,
        "cds_ends_with_stop_codon": ends_with_stop,
        "cds_length_multiple_of_3": length_multiple_of_three,
    }
    if declared_cds_checks != observed_cds_checks:
        raise ValueError(f"{transcript_id}: CDS sequence checks disagree with declared flags")

    master_cds_reliable = _parse_bool(
        row["cds_boundary_reliable"], "cds_boundary_reliable", transcript_id
    )
    master_mrna_reliable = _parse_bool(
        row["mrna_coordinate_system_reliable"],
        "mrna_coordinate_system_reliable",
        transcript_id,
    )
    if master_cds_reliable != mask.cds_boundary_reliable:
        raise ValueError(f"{transcript_id}: CDS reliability disagrees between master and mask")
    if master_mrna_reliable != mask.mrna_coordinate_system_reliable:
        raise ValueError(f"{transcript_id}: mRNA reliability disagrees between master and mask")

    return {
        **coordinates,
        **lengths,
        **declared_cds_checks,
    }


def read_masks(path: Path) -> Dict[str, MaskRecord]:
    """Read and validate the full-mRNA binary masks."""

    _raise_csv_field_limit()
    records: Dict[str, MaskRecord] = {}
    with _open_text(path) as handle:
        reader = csv.DictReader(handle)
        required = {
            "transcript_id",
            "transcript_length",
            "m6a_nt_mask",
            "cds_boundary_reliable",
            "mrna_coordinate_system_reliable",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Mask table is missing columns: {sorted(required)}")

        for row_number, row in enumerate(reader, start=2):
            transcript_id = row["transcript_id"].strip()
            if not transcript_id:
                raise ValueError(f"Mask row {row_number} has no transcript_id")
            if transcript_id in records:
                raise ValueError(f"Duplicate mask for {transcript_id}")

            length = int(row["transcript_length"])
            tokens = row["m6a_nt_mask"].split(",")
            if len(tokens) != length:
                raise ValueError(
                    f"{transcript_id}: mask length {len(tokens)} != transcript length {length}"
                )
            invalid = set(tokens) - {"0", "1"}
            if invalid:
                raise ValueError(f"{transcript_id}: invalid mask values {sorted(invalid)}")
            positive_positions = tuple(index for index, value in enumerate(tokens) if value == "1")

            records[transcript_id] = MaskRecord(
                length=length,
                positive_positions=positive_positions,
                cds_boundary_reliable=_parse_bool(
                    row["cds_boundary_reliable"], "cds_boundary_reliable", transcript_id
                ),
                mrna_coordinate_system_reliable=_parse_bool(
                    row["mrna_coordinate_system_reliable"],
                    "mrna_coordinate_system_reliable",
                    transcript_id,
                ),
            )
    if not records:
        raise ValueError("Mask table contains no records")
    return records


def _validate_exon_map(
    path: Path,
    transcript_lengths: Mapping[str, int],
) -> Mapping[str, int]:
    """Verify that exon rows form a contiguous transcript-coordinate map."""

    stats: Counter = Counter()
    previous_end: Dict[str, int] = {}
    seen = set()
    with _open_text(path) as handle:
        reader = csv.DictReader(handle)
        required = {
            "transcript_id",
            "genomic_start_1based",
            "genomic_end_1based",
            "tx_start_0based",
            "tx_end_0based",
            "exon_length",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Exon table is missing columns: {sorted(required)}")

        for row in reader:
            transcript_id = row["transcript_id"].strip()
            if transcript_id not in transcript_lengths:
                continue
            stats["rows"] += 1
            genomic_length = int(row["genomic_end_1based"]) - int(row["genomic_start_1based"]) + 1
            tx_start = int(row["tx_start_0based"])
            tx_end = int(row["tx_end_0based"])
            exon_length = int(row["exon_length"])
            if genomic_length != exon_length or tx_end - tx_start != exon_length:
                raise ValueError(f"{transcript_id}: inconsistent exon length")
            expected_start = previous_end.get(transcript_id, 0)
            if tx_start != expected_start:
                raise ValueError(
                    f"{transcript_id}: non-contiguous transcript coordinates "
                    f"({tx_start} after {expected_start})"
                )
            previous_end[transcript_id] = tx_end
            seen.add(transcript_id)

    missing = set(transcript_lengths) - seen
    if missing:
        raise ValueError(f"Exon map is missing {len(missing)} labelled transcripts")
    bad_coverage = [
        transcript_id
        for transcript_id, length in transcript_lengths.items()
        if previous_end.get(transcript_id) != length
    ]
    if bad_coverage:
        raise ValueError(f"Exon map does not cover {len(bad_coverage)} complete transcripts")
    stats["transcripts"] = len(seen)
    return dict(stats)


def prepare_dataset(
    transcript_master: Path,
    mask_table: Path,
    output_dir: Path,
    exon_map: Optional[Path] = None,
    seed: int = 2357,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> Mapping[str, object]:
    """Create gene-disjoint train/validation/test records and an audit report."""

    masks = read_masks(Path(mask_table))
    if train_fraction < 0 or val_fraction < 0 or train_fraction + val_fraction > 1:
        raise ValueError("Invalid split fractions")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        split: gzip.open(
            output_dir / f"{split}.jsonl.gz",
            "wt",
            encoding="utf-8",
            newline="\n",
            compresslevel=6,
        )
        for split in ("train", "val", "test")
    }
    split_stats = {split: Counter() for split in handles}
    split_genes = defaultdict(set)
    matched = set()

    try:
        with _open_text(Path(transcript_master)) as handle:
            reader = csv.DictReader(handle)
            required = {
                "transcript_id",
                "gene_id",
                "transcript_length",
                "transcript_sequence",
                *REGION_FIELDS,
            }
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                missing_columns = sorted(required - set(reader.fieldnames or []))
                raise ValueError(f"Transcript table is missing columns: {missing_columns}")

            for row in reader:
                transcript_id = row["transcript_id"].strip()
                mask = masks.get(transcript_id)
                if mask is None:
                    continue
                if transcript_id in matched:
                    raise ValueError(f"Duplicate transcript_master row for {transcript_id}")

                gene_id = row["gene_id"].strip()
                sequence = _rna_sequence(row["transcript_sequence"])
                declared_length = int(row["transcript_length"])
                if len(sequence) != declared_length or len(sequence) != mask.length:
                    raise ValueError(f"{transcript_id}: sequence, table, and mask lengths disagree")
                invalid = set(sequence) - set("ACGUN")
                if invalid:
                    raise ValueError(f"{transcript_id}: invalid RNA symbols {sorted(invalid)}")
                if any(sequence[position] != "A" for position in mask.positive_positions):
                    raise ValueError(f"{transcript_id}: positive m6A mask falls on a non-A base")
                region_metadata = _validate_transcript_regions(
                    row=row,
                    transcript_id=transcript_id,
                    sequence=sequence,
                    declared_length=declared_length,
                    mask=mask,
                )

                split = _stable_gene_split(gene_id, seed, train_fraction, val_fraction)
                record = {
                    "transcript_id": transcript_id,
                    "gene_id": gene_id,
                    "sequence": sequence,
                    "m6a_positions": mask.positive_positions,
                    "cds_boundary_reliable": mask.cds_boundary_reliable,
                    "mrna_coordinate_system_reliable": mask.mrna_coordinate_system_reliable,
                    **region_metadata,
                }
                handles[split].write(json.dumps(record, separators=(",", ":")) + "\n")

                stats = split_stats[split]
                candidate_count = sequence.count("A")
                positive_count = len(mask.positive_positions)
                stats["transcripts"] += 1
                stats["nucleotides"] += len(sequence)
                stats["candidate_adenosines"] += candidate_count
                stats["positive_m6a"] += positive_count
                stats["negative_adenosines"] += candidate_count - positive_count
                stats["mrna_coordinate_system_reliable"] += int(
                    mask.mrna_coordinate_system_reliable
                )
                stats["cds_boundary_reliable"] += int(mask.cds_boundary_reliable)
                stats["full_transcript_training_eligible"] += int(
                    mask.mrna_coordinate_system_reliable and mask.cds_boundary_reliable
                )
                for region in ("utr5", "cds", "utr3"):
                    start = int(region_metadata[f"{region}_start"])
                    end = int(region_metadata[f"{region}_end"])
                    stats[f"{region}_nucleotides"] += end - start
                    stats[f"{region}_candidate_adenosines"] += sequence[start:end].count("A")
                    stats[f"{region}_positive_m6a"] += (
                        bisect.bisect_left(mask.positive_positions, end)
                        - bisect.bisect_left(mask.positive_positions, start)
                    )
                split_genes[split].add(gene_id)
                matched.add(transcript_id)
    finally:
        for handle in handles.values():
            handle.close()

    missing = set(masks) - matched
    if missing:
        raise ValueError(f"Transcript table is missing {len(missing)} masked transcripts")

    gene_to_splits = defaultdict(set)
    for split, genes in split_genes.items():
        for gene_id in genes:
            gene_to_splits[gene_id].add(split)
    leaking = [gene_id for gene_id, splits in gene_to_splits.items() if len(splits) > 1]
    if leaking:
        raise RuntimeError(f"Gene-level split leakage detected for {len(leaking)} genes")

    exon_stats = None
    if exon_map is not None:
        exon_stats = _validate_exon_map(
            Path(exon_map), {transcript_id: record.length for transcript_id, record in masks.items()}
        )

    split_report = {}
    for split, stats in split_stats.items():
        values = dict(stats)
        values["genes"] = len(split_genes[split])
        values["positive_fraction_among_adenosines"] = (
            stats["positive_m6a"] / stats["candidate_adenosines"]
            if stats["candidate_adenosines"]
            else 0.0
        )
        values["recommended_pos_weight"] = (
            stats["negative_adenosines"] / stats["positive_m6a"]
            if stats["positive_m6a"]
            else 1.0
        )
        split_report[split] = values

    report: Mapping[str, object] = {
        "schema_version": 2,
        "task": "nucleotide-level m6A classification at adenosine positions",
        "sequence_contract": {
            "sample": "one complete mRNA before optional model-length filtering",
            "composition": "transcript_sequence = 5'UTR + CDS + 3'UTR",
            "coordinate_system": "0-based half-open transcript coordinates",
            "cds_start_definition": "index of the first CDS nucleotide in transcript_sequence",
        },
        "label_semantics": {
            "1": "methylated adenosine",
            "0": "unmethylated adenosine",
            "non_A": "excluded from the m6A loss",
        },
        "sources": {
            "transcript_master": str(Path(transcript_master).resolve()),
            "mask_table": str(Path(mask_table).resolve()),
            "exon_map": str(Path(exon_map).resolve()) if exon_map is not None else None,
        },
        "splitting": {
            "unit": "gene_id",
            "seed": seed,
            "train_fraction": train_fraction,
            "val_fraction": val_fraction,
            "test_fraction": 1.0 - train_fraction - val_fraction,
            "leaking_genes": 0,
        },
        "splits": split_report,
        "exon_map_audit": exon_stats,
    }
    with (output_dir / "stats.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript-master", type=Path, required=True)
    parser.add_argument("--mask-table", type=Path, required=True)
    parser.add_argument("--exon-map", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2357)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = prepare_dataset(
        transcript_master=args.transcript_master,
        mask_table=args.mask_table,
        exon_map=args.exon_map,
        output_dir=args.output_dir,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
    json.dump(report, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
