#!/usr/bin/env python3
"""Prepare transcript-level human m6A records for sliding-window fine-tuning.

The source files supplied for this project contain complete CDS sequences in a
CSV file and m6A calls in a headerless BED-like file.  This script maps each
observed m6A context to one unique position in its transcript, removes repeated
calls at the same transcript position, splits by gene, and writes compact
transcript-level JSONL files.  Sliding windows are built lazily by the training
dataset so the nucleotide sequence is not duplicated on disk for every window.

The output label semantics are deliberately conservative: a window target is
the number of *observed, uniquely mapped* m6A sites in that window.  An
unlabelled adenosine is not asserted to be a biological negative.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, TextIO, Tuple


BED_COLUMNS = (
    "chrom",
    "start",
    "end",
    "site_id",
    "strand",
    "modification",
    "evidence_count",
    "datasets",
    "samples",
    "pmids",
    "cells",
    "assay",
    "gene_id",
    "transcript_id",
    "symbols",
    "biotypes",
    "feature",
    "context",
    "score",
)

DEFAULT_CDS_MEMBER = "data/human_te_cds_dataset.csv"
DEFAULT_SITES_MEMBER = "data/human.m6A.matched_with_cds_te.bed"
VALID_RNA = frozenset("ACGUN")


def normalize_identifier(value: str) -> str:
    """Remove whitespace and an optional Ensembl version suffix."""

    return value.strip().split(".", 1)[0]


def normalize_rna(sequence: str) -> str:
    """Convert a DNA-alphabet sequence to the RNA alphabet and validate it."""

    sequence = re.sub(r"\s+", "", sequence).upper().replace("T", "U")
    invalid = sorted(set(sequence) - VALID_RNA)
    if invalid:
        raise ValueError(f"Unexpected sequence symbols: {invalid}")
    return sequence


@contextmanager
def open_text_source(path: Path, member: Optional[str] = None) -> Iterator[TextIO]:
    """Open a plain text file or one member of a ZIP archive as UTF-8 text."""

    path = Path(path)
    if path.suffix.lower() != ".zip":
        with path.open("r", encoding="utf-8", newline="") as handle:
            yield handle
        return

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if member is None:
            files = [name for name in names if not name.endswith("/")]
            if len(files) != 1:
                raise ValueError(
                    f"{path} contains {len(files)} files; specify --*-member explicitly"
                )
            member = files[0]
        elif member not in names:
            suffix_matches = [name for name in names if name.endswith(member)]
            if len(suffix_matches) != 1:
                raise FileNotFoundError(f"Could not uniquely find {member!r} in {path}")
            member = suffix_matches[0]

        with archive.open(member, "r") as raw:
            import io

            with io.TextIOWrapper(raw, encoding="utf-8", newline="") as handle:
                yield handle


def find_all(sequence: str, query: str) -> List[int]:
    """Return every (including overlapping) occurrence of query."""

    starts: List[int] = []
    cursor = 0
    while True:
        match = sequence.find(query, cursor)
        if match < 0:
            return starts
        starts.append(match)
        cursor = match + 1


def locate_unique_context(
    transcript: str,
    context: str,
    context_lengths: Sequence[int] = (41, 21),
) -> Tuple[Optional[int], str, Optional[int]]:
    """Locate the central m6A adenosine using descending context lengths.

    A location is accepted only when the centred context occurs exactly once.
    Ambiguous mappings are excluded instead of being assigned arbitrarily.
    """

    transcript = normalize_rna(transcript)
    context = normalize_rna(context)
    center = len(context) // 2
    if not context or context[center] != "A":
        return None, "context_center_not_a", None

    for length in context_lengths:
        if length <= 0 or length % 2 == 0 or length > len(context):
            continue
        half = length // 2
        query = context[center - half : center + half + 1]
        matches = find_all(transcript, query)
        if len(matches) == 1:
            position = matches[0] + half
            if transcript[position] != "A":
                return None, "mapped_center_not_a", length
            return position, "unique", length
        if len(matches) > 1:
            return None, "ambiguous", length

    return None, "not_found", None


def window_starts(sequence_length: int, window_length: int, stride: int) -> List[int]:
    """Generate starts while guaranteeing one window covers the sequence tail."""

    if sequence_length <= 0:
        return []
    if window_length <= 0 or stride <= 0:
        raise ValueError("window_length and stride must be positive")
    if sequence_length <= window_length:
        return [0]

    starts = list(range(0, sequence_length - window_length + 1, stride))
    tail_start = sequence_length - window_length
    if starts[-1] != tail_start:
        starts.append(tail_start)
    return starts


def stable_gene_split(
    gene_id: str,
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> str:
    """Assign every transcript from the same gene to one deterministic split."""

    if train_fraction < 0 or val_fraction < 0 or train_fraction + val_fraction > 1:
        raise ValueError("Invalid split fractions")
    digest = hashlib.sha256(f"{seed}:{normalize_identifier(gene_id)}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < train_fraction:
        return "train"
    if value < train_fraction + val_fraction:
        return "val"
    return "test"


def read_transcripts(
    source: Path,
    member: Optional[str],
) -> Tuple[Dict[str, MutableMapping[str, object]], Counter]:
    """Read unique transcript CDS sequences from the supplied CSV."""

    records: Dict[str, MutableMapping[str, object]] = {}
    stats: Counter = Counter()
    with open_text_source(source, member) as handle:
        reader = csv.DictReader(handle)
        required = {"gene_id", "cds_sequence"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"CDS CSV is missing required columns {sorted(required)}")

        for row in reader:
            stats["cds_rows"] += 1
            transcript_raw = row.get("transcript_id") or row.get("tx_id") or ""
            transcript_id = normalize_identifier(transcript_raw)
            gene_raw = row.get("gene_id", "")
            gene_id = normalize_identifier(gene_raw)
            if not transcript_id or not gene_id:
                stats["cds_missing_identifier"] += 1
                continue
            try:
                sequence = normalize_rna(row["cds_sequence"])
            except ValueError:
                stats["cds_invalid_sequence"] += 1
                continue
            if not sequence:
                stats["cds_empty_sequence"] += 1
                continue

            existing = records.get(transcript_id)
            if existing is not None:
                if existing["sequence"] != sequence:
                    raise ValueError(f"Conflicting CDS sequences for transcript {transcript_id}")
                stats["cds_duplicate_rows"] += 1
                continue

            records[transcript_id] = {
                "transcript_id": transcript_id,
                "transcript_id_original": transcript_raw.strip(),
                "gene_id": gene_id,
                "gene_id_original": gene_raw.strip(),
                "symbol": (row.get("SYMBOL") or "").strip(),
                "sequence": sequence,
                "observed_m6a_positions": [],
            }
    stats["unique_transcripts"] = len(records)
    return records, stats


def map_sites(
    source: Path,
    member: Optional[str],
    transcripts: Mapping[str, MutableMapping[str, object]],
    context_lengths: Sequence[int],
) -> Counter:
    """Map and deduplicate observed m6A calls in transcript coordinates."""

    stats: Counter = Counter()
    positions: Dict[str, set] = defaultdict(set)

    with open_text_source(source, member) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            stats["site_rows"] += 1
            fields = line.rstrip("\n\r").split("\t")
            if len(fields) < len(BED_COLUMNS):
                stats["site_short_rows"] += 1
                continue
            row = dict(zip(BED_COLUMNS, fields[: len(BED_COLUMNS)]))
            if row["modification"].lower() != "m6a":
                stats["site_non_m6a"] += 1
                continue

            transcript_id = normalize_identifier(row["transcript_id"])
            transcript_record = transcripts.get(transcript_id)
            if transcript_record is None:
                stats["site_transcript_missing"] += 1
                continue

            position, status, matched_length = locate_unique_context(
                str(transcript_record["sequence"]),
                row["context"],
                context_lengths=context_lengths,
            )
            stats[f"site_mapping_{status}"] += 1
            if position is None:
                continue
            stats[f"site_mapping_unique_{matched_length}nt"] += 1
            if position in positions[transcript_id]:
                stats["site_duplicate_transcript_position"] += 1
                continue
            positions[transcript_id].add(position)
            stats["unique_mapped_sites"] += 1

    for transcript_id, record in transcripts.items():
        record["observed_m6a_positions"] = sorted(positions.get(transcript_id, set()))
    stats["transcripts_with_mapped_sites"] = sum(bool(value) for value in positions.values())
    return stats


def count_windows(
    records: Iterable[Mapping[str, object]],
    window_length: int,
    stride: int,
) -> Counter:
    """Summarize window targets without materializing duplicated sequences."""

    stats: Counter = Counter()
    for record in records:
        sequence = str(record["sequence"])
        positions = list(record["observed_m6a_positions"])
        for start in window_starts(len(sequence), window_length, stride):
            end = min(start + window_length, len(sequence))
            count = sum(start <= int(position) < end for position in positions)
            stats["windows"] += 1
            stats["observed_sites_across_windows"] += count
            if count:
                stats["positive_windows"] += 1
            else:
                stats["zero_windows"] += 1
            stats[f"windows_with_{min(count, 10)}_sites"] += 1
    return stats


def prepare_dataset(
    cds_source: Path,
    sites_source: Path,
    output_dir: Path,
    cds_member: Optional[str] = DEFAULT_CDS_MEMBER,
    sites_member: Optional[str] = DEFAULT_SITES_MEMBER,
    context_lengths: Sequence[int] = (41, 21),
    window_length: int = 128,
    stride: int = 64,
    seed: int = 2357,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> Mapping[str, object]:
    """Run the complete preparation pipeline and return the audit report."""

    transcripts, cds_stats = read_transcripts(Path(cds_source), cds_member)
    site_stats = map_sites(
        Path(sites_source),
        sites_member,
        transcripts,
        tuple(sorted(set(context_lengths), reverse=True)),
    )

    split_records: Dict[str, List[MutableMapping[str, object]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for transcript_id in sorted(transcripts):
        record = transcripts[transcript_id]
        split = stable_gene_split(
            str(record["gene_id"]),
            seed=seed,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
        )
        record["split"] = split
        split_records[split].append(record)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_stats: Dict[str, Mapping[str, int]] = {}
    for split, records in split_records.items():
        destination = output_dir / f"{split}.jsonl"
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        window_stats = count_windows(records, window_length, stride)
        window_count = int(window_stats["windows"])
        split_stats[split] = {
            "transcripts": len(records),
            "genes": len({str(record["gene_id"]) for record in records}),
            "unique_mapped_sites": sum(len(record["observed_m6a_positions"]) for record in records),
            **dict(window_stats),
            "positive_window_fraction": (
                window_stats["positive_windows"] / window_count if window_count else 0.0
            ),
        }

    gene_splits: Dict[str, set] = defaultdict(set)
    for split, records in split_records.items():
        for record in records:
            gene_splits[str(record["gene_id"])].add(split)
    leaking_genes = sorted(gene for gene, splits in gene_splits.items() if len(splits) > 1)
    if leaking_genes:
        raise RuntimeError(f"Gene-level split leakage detected for {len(leaking_genes)} genes")

    report: Mapping[str, object] = {
        "schema_version": 1,
        "label_semantics": "count of observed uniquely mapped m6A sites per window",
        "unlabelled_adenosine_semantics": "unknown; not asserted to be a biological negative",
        "sources": {
            "cds_source": str(Path(cds_source).resolve()),
            "cds_member": cds_member,
            "sites_source": str(Path(sites_source).resolve()),
            "sites_member": sites_member,
        },
        "mapping": {
            "context_lengths": list(context_lengths),
            "cds": dict(cds_stats),
            "sites": dict(site_stats),
            "unique_mapping_fraction": (
                site_stats["unique_mapped_sites"] / site_stats["site_rows"]
                if site_stats["site_rows"]
                else 0.0
            ),
        },
        "windowing": {
            "window_length": window_length,
            "stride": stride,
            "tail_window_included": True,
        },
        "splitting": {
            "unit": "gene_id without Ensembl version suffix",
            "seed": seed,
            "train_fraction": train_fraction,
            "val_fraction": val_fraction,
            "test_fraction": round(1.0 - train_fraction - val_fraction, 10),
            "leaking_genes": 0,
        },
        "splits": split_stats,
    }
    with (output_dir / "stats.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cds-source", type=Path, required=True, help="CDS CSV or ZIP containing it")
    parser.add_argument("--sites-source", type=Path, required=True, help="Matched m6A BED or ZIP containing it")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cds-member", default=DEFAULT_CDS_MEMBER)
    parser.add_argument("--sites-member", default=DEFAULT_SITES_MEMBER)
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[41, 21])
    parser.add_argument("--window-length", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2357)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = prepare_dataset(
        cds_source=args.cds_source,
        sites_source=args.sites_source,
        output_dir=args.output_dir,
        cds_member=args.cds_member,
        sites_member=args.sites_member,
        context_lengths=args.context_lengths,
        window_length=args.window_length,
        stride=args.stride,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
    json.dump(report, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
