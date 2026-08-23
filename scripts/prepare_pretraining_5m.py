#!/usr/bin/env python3
"""Build the indexed 3M ncRNA + approximately 2M coding-RNA corpus.

The input is the canonical source directory described in ``docs/PRETRAINING_5M.md``.
Sequences are streamed and written to memory-mappable files; the complete
five-million-record corpus is never materialised as Python strings in RAM.
"""

from __future__ import annotations

import argparse
import array
import csv
import gzip
import hashlib
import io
import json
import os
import random
import shutil
import struct
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Optional


SCHEMA_VERSION = 2
SOURCE_CODES = {"ncRNA": 0, "coding": 1}
PROGRESS_INTERVAL = 100_000


class DirectoryArchive:
    """Expose a canonical source directory through the small ZipFile API used here."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._members = [
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        ]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def namelist(self) -> list[str]:
        return self._members

    def open(self, member: str):
        return self.path_for(member).open("rb")

    def path_for(self, member: str) -> Path:
        path = (self.root / Path(member)).resolve()
        if self.root not in path.parents:
            raise ValueError(f"Source member escapes data root: {member}")
        return path


def progress(label: str, seen: int, accepted: int) -> None:
    print(f"[{label}] seen={seen:,} accepted={accepted:,}", flush=True)


def normalize_rna(raw: object, min_length: int, max_length: int) -> tuple[str, str]:
    sequence = "".join(str(raw).split()).upper().replace("T", "U")
    if not sequence:
        return "", "empty"
    if len(sequence) < min_length:
        return "", "too_short"
    if len(sequence) > max_length:
        return "", "too_long"
    if any(base not in "AUCGN" for base in sequence):
        return "", "invalid_alphabet"
    if not any(base in "AUCG" for base in sequence):
        return "", "no_canonical_base"
    return sequence, ""


def iter_fasta_binary(handle) -> Iterator[tuple[str, str]]:
    identifier: Optional[str] = None
    chunks: list[str] = []
    for raw_line in handle:
        line = raw_line.decode("ascii").strip()
        if not line:
            continue
        if line.startswith(">"):
            if identifier is not None:
                yield identifier, "".join(chunks)
            identifier = line[1:].split(maxsplit=1)[0]
            chunks = []
        else:
            chunks.append(line)
    if identifier is not None:
        yield identifier, "".join(chunks)


def content_digest(sequence: str) -> bytes:
    # A 128-bit prefix is sufficient for deduplication at this corpus scale.
    return hashlib.sha256(sequence.encode("ascii")).digest()[:16]


def stable_split(key: str, seed: int, train_fraction: float, val_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < train_fraction:
        return "train"
    if value < train_fraction + val_fraction:
        return "val"
    return "test"


def find_member(archive: zipfile.ZipFile, basename: str) -> str:
    matches = [name for name in archive.namelist() if PurePosixPath(name).name == basename]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {basename!r} in the bundle, found {len(matches)}"
        )
    return matches[0]


def nested_archives(archive: zipfile.ZipFile, prefix: str) -> list[str]:
    return sorted(
        name
        for name in archive.namelist()
        if PurePosixPath(name).name.startswith(prefix) and name.lower().endswith(".zip")
    )


def extract_outer_member(
    archive: zipfile.ZipFile, member: str, destination: Path
) -> Path:
    if isinstance(archive, DirectoryArchive):
        output = archive.path_for(member)
        print(f"[source] using {output}", flush=True)
        return output
    output = destination / PurePosixPath(member).name
    print(f"[extract] {member} -> {output}", flush=True)
    with archive.open(member) as source, output.open("wb") as target:
        shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
    print(f"[extract] completed {output.name} ({output.stat().st_size / 2**30:.2f} GiB)", flush=True)
    return output


def inner_fasta_members(archive: zipfile.ZipFile, basename: str) -> list[str]:
    return sorted(
        name
        for name in archive.namelist()
        if PurePosixPath(name).name.lower() == basename.lower()
    )


def species_from_member(member: str) -> str:
    parent = PurePosixPath(member).parent.name.strip()
    return parent or "unknown"


def write_spool_record(handle, source_type: str, species: str, identifier: str, sequence: str) -> int:
    offset = handle.tell()
    clean_identifier = identifier.replace("\t", " ").replace("\n", " ")
    clean_species = species.replace("\t", " ").replace("\n", " ")
    handle.write(f"{source_type}\t{clean_species}\t{clean_identifier}\t{sequence}\n".encode("utf-8"))
    return offset


def parse_spool_line(line: bytes) -> tuple[str, str, str, str]:
    fields = line.decode("utf-8").rstrip("\n").split("\t", 3)
    if len(fields) != 4:
        raise ValueError("Malformed internal spool record")
    return fields[0], fields[1], fields[2], fields[3]


class ReservoirOffsets:
    """Fixed-memory reservoir sampler storing only spool byte offsets."""

    def __init__(self, size: int, seed: int) -> None:
        self.size = int(size)
        self.rng = random.Random(seed)
        self.seen = 0
        self.offsets = array.array("Q")

    def add(self, offset: int) -> None:
        self.seen += 1
        if self.size == 0:
            return
        if len(self.offsets) < self.size:
            self.offsets.append(int(offset))
            return
        replacement = self.rng.randrange(self.seen)
        if replacement < self.size:
            self.offsets[replacement] = int(offset)

    def selected(self) -> list[int]:
        if self.seen < self.size:
            raise ValueError(f"Only {self.seen:,} candidates are available for {self.size:,} slots")
        return sorted(self.offsets)


class IndexedSplitWriter:
    def __init__(self, output_dir: Path, split: str) -> None:
        self.split = split
        self.sequence_name = f"{split}.sequences.bin"
        self.offset_name = f"{split}.offsets.u64"
        self.source_name = f"{split}.sources.u8"
        self.metadata_name = f"{split}.records.tsv.gz"
        self.sequence_handle = (output_dir / self.sequence_name).open("wb")
        self.offset_handle = (output_dir / self.offset_name).open("wb")
        self.source_handle = (output_dir / self.source_name).open("wb")
        self.metadata_handle = gzip.open(output_dir / self.metadata_name, "wt", encoding="utf-8", newline="")
        self.metadata_handle.write("source_class\tsource_type\tspecies\trecord_id\tlength\n")
        self.offset = 0
        self.offset_handle.write(struct.pack("<Q", 0))
        self.records = 0
        self.nucleotides = 0
        self.source_class_counts: Counter[str] = Counter()
        self.source_type_counts: Counter[str] = Counter()
        self.min_length: Optional[int] = None
        self.max_length = 0

    def add(self, source_class: str, source_type: str, species: str, identifier: str, sequence: str) -> None:
        encoded = sequence.encode("ascii")
        self.sequence_handle.write(encoded)
        self.offset += len(encoded)
        self.offset_handle.write(struct.pack("<Q", self.offset))
        self.source_handle.write(bytes([SOURCE_CODES[source_class]]))
        self.metadata_handle.write(
            "\t".join(
                (
                    source_class,
                    source_type,
                    species.replace("\t", " "),
                    identifier.replace("\t", " "),
                    str(len(sequence)),
                )
            )
            + "\n"
        )
        self.records += 1
        self.nucleotides += len(sequence)
        self.source_class_counts[source_class] += 1
        self.source_type_counts[source_type] += 1
        self.min_length = len(sequence) if self.min_length is None else min(self.min_length, len(sequence))
        self.max_length = max(self.max_length, len(sequence))

    def close(self) -> None:
        self.sequence_handle.close()
        self.offset_handle.close()
        self.source_handle.close()
        self.metadata_handle.close()

    def manifest_entry(self) -> dict:
        return {
            "records": self.records,
            "nucleotides": self.nucleotides,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "source_class_counts": dict(sorted(self.source_class_counts.items())),
            "source_type_counts": dict(sorted(self.source_type_counts.items())),
            "files": {
                "sequences": self.sequence_name,
                "offsets": self.offset_name,
                "sources": self.source_name,
                "metadata": self.metadata_name,
            },
        }


def copy_selected_spool(path: Path, offsets: Iterable[int], output_handle) -> None:
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            output_handle.write(handle.readline())


def process_fasta_to_primary(
    archive: zipfile.ZipFile,
    members: Iterable[str],
    spool_handle,
    reservoir: ReservoirOffsets,
    seen_hashes: set[bytes],
    source_type: str,
    min_length: int,
    max_length: int,
    audit: Counter,
) -> None:
    for member in members:
        species = species_from_member(member)
        with archive.open(member) as fasta:
            for identifier, raw_sequence in iter_fasta_binary(fasta):
                audit[f"{source_type}:seen"] += 1
                if audit[f"{source_type}:seen"] % PROGRESS_INTERVAL == 0:
                    progress(
                        source_type,
                        audit[f"{source_type}:seen"],
                        audit[f"{source_type}:accepted"],
                    )
                sequence, reason = normalize_rna(raw_sequence, min_length, max_length)
                if reason:
                    audit[f"{source_type}:excluded_{reason}"] += 1
                    continue
                digest = content_digest(sequence)
                if digest in seen_hashes:
                    audit[f"{source_type}:excluded_duplicate"] += 1
                    continue
                seen_hashes.add(digest)
                offset = write_spool_record(spool_handle, source_type, species, identifier, sequence)
                reservoir.add(offset)
                audit[f"{source_type}:accepted"] += 1


def process_transcript_master(
    compressed_handle,
    spool_handle,
    reservoir: ReservoirOffsets,
    seen_hashes: set[bytes],
    source_type: str,
    species: str,
    min_length: int,
    max_length: int,
    audit: Counter,
) -> None:
    with gzip.GzipFile(fileobj=compressed_handle, mode="rb") as raw:
        with io.TextIOWrapper(raw, encoding="utf-8", newline="") as text:
            reader = csv.DictReader(text)
            if not reader.fieldnames or "transcript_sequence" not in reader.fieldnames:
                raise ValueError(f"{source_type} master has no transcript_sequence column")
            id_field = "transcript_id" if "transcript_id" in reader.fieldnames else reader.fieldnames[0]
            for row_number, row in enumerate(reader, start=2):
                audit[f"{source_type}:seen"] += 1
                if audit[f"{source_type}:seen"] % PROGRESS_INTERVAL == 0:
                    progress(
                        source_type,
                        audit[f"{source_type}:seen"],
                        audit[f"{source_type}:accepted"],
                    )
                sequence, reason = normalize_rna(row["transcript_sequence"], min_length, max_length)
                if reason:
                    audit[f"{source_type}:excluded_{reason}"] += 1
                    continue
                digest = content_digest(sequence)
                if digest in seen_hashes:
                    audit[f"{source_type}:excluded_duplicate"] += 1
                    continue
                seen_hashes.add(digest)
                identifier = row.get(id_field) or f"{source_type}_row_{row_number}"
                offset = write_spool_record(spool_handle, source_type, species, identifier, sequence)
                reservoir.add(offset)
                audit[f"{source_type}:accepted"] += 1


def process_euk_cds_candidates(
    nested_paths: Iterable[Path],
    spool_path: Path,
    wanted: int,
    min_length: int,
    max_length: int,
    seed: int,
    audit: Counter,
) -> list[int]:
    reservoir = ReservoirOffsets(wanted, seed)
    # CDS is an explicitly labelled secondary representation of a transcript,
    # so it may equal that transcript's full-mRNA sequence. Deduplicate within
    # the CDS-view pool, not against the primary-representation pool.
    seen_hashes: set[bytes] = set()
    with spool_path.open("wb") as spool:
        for nested_path in nested_paths:
            with zipfile.ZipFile(nested_path) as nested:
                for member in inner_fasta_members(nested, "cds.fa"):
                    species = species_from_member(member)
                    with nested.open(member) as fasta:
                        for identifier, raw_sequence in iter_fasta_binary(fasta):
                            source_type = "eukaryote_cds_view"
                            audit[f"{source_type}:seen"] += 1
                            if audit[f"{source_type}:seen"] % PROGRESS_INTERVAL == 0:
                                progress(
                                    source_type,
                                    audit[f"{source_type}:seen"],
                                    audit[f"{source_type}:accepted"],
                                )
                            sequence, reason = normalize_rna(raw_sequence, min_length, max_length)
                            if reason:
                                audit[f"{source_type}:excluded_{reason}"] += 1
                                continue
                            digest = content_digest(sequence)
                            if digest in seen_hashes:
                                audit[f"{source_type}:excluded_duplicate"] += 1
                                continue
                            seen_hashes.add(digest)
                            offset = write_spool_record(spool, source_type, species, identifier, sequence)
                            reservoir.add(offset)
                            audit[f"{source_type}:accepted"] += 1
    return reservoir.selected()


def prepare_corpus(args: argparse.Namespace) -> dict:
    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)
    if args.target_ncrna <= 0 or args.target_coding <= 0:
        raise ValueError("target_ncrna and target_coding must be positive")
    if args.min_length <= 0 or args.max_length < args.min_length:
        raise ValueError("Require 0 < min_length <= max_length")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; pass --overwrite to replace it")
    if args.train_fraction <= 0 or args.val_fraction < 0:
        raise ValueError("Invalid split fractions")
    if args.train_fraction + args.val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be below 1")

    build_dir = output_dir.with_name(output_dir.name + ".building")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    audit: Counter = Counter()

    try:
        with tempfile.TemporaryDirectory(prefix="rna_pretraining_5m_", dir=args.temp_dir) as temp_name:
            temp_dir = Path(temp_name)
            primary_spool = temp_dir / "coding_primary.tsv"
            filler_spool = temp_dir / "coding_filler.tsv"
            selected_coding_spool = temp_dir / "coding_selected.tsv"
            seen_coding_hashes: set[bytes] = set()
            primary_reservoir = ReservoirOffsets(args.target_coding, args.seed + 11)

            with DirectoryArchive(source_dir) as outer:
                print(f"[source-dir] auditing {source_dir}", flush=True)
                nc_member = find_member(outer, "rnacentral_active_mRNA100_priority_3M.fasta.gz")
                human_member = find_member(outer, "human_transcript_master.csv.gz")
                mouse_member = find_member(outer, "mouse_transcript_master.csv.gz")
                euk_members = nested_archives(outer, "eukaryote_mRNA_dataset_part")
                prok_members = nested_archives(outer, "prokaryote_mRNA_dataset")
                if not euk_members or not prok_members:
                    raise ValueError("Missing eukaryote or prokaryote nested archives")

                nested_euk_paths = [extract_outer_member(outer, member, temp_dir) for member in euk_members]
                nested_prok_paths = [extract_outer_member(outer, member, temp_dir) for member in prok_members]

                with primary_spool.open("wb") as spool:
                    for nested_path in nested_euk_paths:
                        with zipfile.ZipFile(nested_path) as nested:
                            process_fasta_to_primary(
                                nested,
                                inner_fasta_members(nested, "mrna.fa"),
                                spool,
                                primary_reservoir,
                                seen_coding_hashes,
                                "eukaryote_full_mrna",
                                args.min_length,
                                args.max_length,
                                audit,
                            )
                    with outer.open(human_member) as compressed:
                        process_transcript_master(
                            compressed,
                            spool,
                            primary_reservoir,
                            seen_coding_hashes,
                            "human_full_mrna",
                            "Homo_sapiens",
                            args.min_length,
                            args.max_length,
                            audit,
                        )
                    with outer.open(mouse_member) as compressed:
                        process_transcript_master(
                            compressed,
                            spool,
                            primary_reservoir,
                            seen_coding_hashes,
                            "mouse_full_mrna",
                            "Mus_musculus",
                            args.min_length,
                            args.max_length,
                            audit,
                        )
                    for nested_path in nested_prok_paths:
                        with zipfile.ZipFile(nested_path) as nested:
                            process_fasta_to_primary(
                                nested,
                                inner_fasta_members(nested, "cds.fa"),
                                spool,
                                primary_reservoir,
                                seen_coding_hashes,
                                "prokaryote_cds",
                                args.min_length,
                                args.max_length,
                                audit,
                            )

                primary_count = primary_reservoir.seen
                print(f"[coding-primary] eligible unique records={primary_count:,}", flush=True)
                if primary_count >= args.target_coding:
                    selected_primary = primary_reservoir.selected()
                    with selected_coding_spool.open("wb") as output:
                        copy_selected_spool(primary_spool, selected_primary, output)
                    filler_count = 0
                else:
                    shortfall = args.target_coding - primary_count
                    if getattr(args, "fill_coding_shortfall_with_cds", False):
                        filler_count = shortfall
                        print(f"[coding-filler] selecting {filler_count:,} eukaryotic CDS views", flush=True)
                        selected_filler = process_euk_cds_candidates(
                            nested_euk_paths,
                            filler_spool,
                            filler_count,
                            args.min_length,
                            args.max_length,
                            args.seed + 29,
                            audit,
                        )
                        with selected_coding_spool.open("wb") as output:
                            with primary_spool.open("rb") as primary:
                                shutil.copyfileobj(primary, output, length=16 * 1024 * 1024)
                            copy_selected_spool(filler_spool, selected_filler, output)
                    else:
                        filler_count = 0
                        print(
                            f"[coding-primary] using all {primary_count:,} independent records; "
                            f"leaving requested quota short by {shortfall:,} rather than duplicating CDS views",
                            flush=True,
                        )
                        with selected_coding_spool.open("wb") as output:
                            copy_selected_spool(primary_spool, sorted(primary_reservoir.offsets), output)

                selected_coding_count = min(primary_count, args.target_coding) + filler_count

                writers = {
                    split: IndexedSplitWriter(build_dir, split)
                    for split in ("train", "val", "test")
                }
                nc_accepted = 0
                # The delivered RNAcentral file is already a record-level,
                # seeded 3M selection. Preserve those biological records even
                # when two accessions happen to have the same sequence.
                del seen_coding_hashes
                try:
                    with outer.open(nc_member) as compressed:
                        with gzip.GzipFile(fileobj=compressed, mode="rb") as fasta:
                            for identifier, raw_sequence in iter_fasta_binary(fasta):
                                audit["rnacentral_ncrna:seen"] += 1
                                if audit["rnacentral_ncrna:seen"] % PROGRESS_INTERVAL == 0:
                                    progress(
                                        "rnacentral_ncrna",
                                        audit["rnacentral_ncrna:seen"],
                                        audit["rnacentral_ncrna:accepted"],
                                    )
                                sequence, reason = normalize_rna(raw_sequence, args.min_length, args.max_length)
                                if reason:
                                    audit[f"rnacentral_ncrna:excluded_{reason}"] += 1
                                    continue
                                if nc_accepted >= args.target_ncrna:
                                    audit["rnacentral_ncrna:excluded_above_target"] += 1
                                    continue
                                key = f"ncRNA:rnacentral:{identifier}"
                                split = stable_split(key, args.seed, args.train_fraction, args.val_fraction)
                                writers[split].add("ncRNA", "rnacentral_ncrna", "mixed", identifier, sequence)
                                nc_accepted += 1
                                audit["rnacentral_ncrna:accepted"] += 1

                    if nc_accepted != args.target_ncrna:
                        raise ValueError(
                            f"Expected {args.target_ncrna:,} unique valid ncRNA records, found {nc_accepted:,}"
                        )

                    coding_accepted = 0
                    with selected_coding_spool.open("rb") as selected:
                        for line in selected:
                            source_type, species, identifier, sequence = parse_spool_line(line)
                            # Keep full-mRNA and CDS views of the same biological
                            # transcript in one split to prevent view leakage.
                            key = f"coding:{species}:{identifier}"
                            split = stable_split(key, args.seed, args.train_fraction, args.val_fraction)
                            writers[split].add("coding", source_type, species, identifier, sequence)
                            coding_accepted += 1
                            if coding_accepted % PROGRESS_INTERVAL == 0:
                                progress("write_coding", coding_accepted, coding_accepted)
                    if coding_accepted != selected_coding_count:
                        raise ValueError(
                            f"Expected {selected_coding_count:,} coding records, wrote {coding_accepted:,}"
                        )
                finally:
                    for writer in writers.values():
                        writer.close()

            manifest = {
                "schema_version": SCHEMA_VERSION,
                "task": "same-position masked language modelling of coding and non-coding RNA",
                "corpus_contract": {
                    "requested_target_records": args.target_ncrna + args.target_coding,
                    "target_ncrna": args.target_ncrna,
                    "target_coding": args.target_coding,
                    "selected_coding_records": selected_coding_count,
                    "coding_primary_records": min(primary_count, args.target_coding),
                    "coding_cds_view_fillers": filler_count,
                    "coding_shortfall_from_requested_target": args.target_coding - selected_coding_count,
                    "coding_policy": (
                        "Prefer unique complete eukaryotic, human and mouse mRNA plus prokaryotic CDS. "
                        "Do not duplicate eukaryotic CDS views unless explicitly requested."
                    ),
                    "m6a_used": False,
                    "m6a_note": "Methylation tables are reserved for downstream fine-tuning.",
                },
                "selection": {
                    "seed": args.seed,
                    "min_length": args.min_length,
                    "max_length": args.max_length,
                    "alphabet": "AUCGN",
                    "dna_to_rna_normalization": "T->U",
                    "coding_content_deduplication": "content deduplication across all primary coding sources",
                    "ncrna_policy": "preserve the delivered seeded RNAcentral record selection",
                    "truncate_sequences": False,
                },
                "splitting": {
                    "method": "stable SHA-256 biological-record assignment; linked full/CDS views stay together",
                    "train_fraction": args.train_fraction,
                    "val_fraction": args.val_fraction,
                    "test_fraction": 1.0 - args.train_fraction - args.val_fraction,
                },
                "source_codes": SOURCE_CODES,
                "splits": {split: writers[split].manifest_entry() for split in writers},
                "audit": dict(sorted(audit.items())),
                "input_source_dir": str(source_dir),
            }
            manifest["totals"] = {
                "records": sum(entry["records"] for entry in manifest["splits"].values()),
                "nucleotides": sum(entry["nucleotides"] for entry in manifest["splits"].values()),
                "source_class_counts": {
                    source: sum(
                        entry["source_class_counts"].get(source, 0)
                        for entry in manifest["splits"].values()
                    )
                    for source in SOURCE_CODES
                },
            }
            with (build_dir / "manifest.json").open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            print(
                f"[complete] records={manifest['totals']['records']:,} "
                f"nucleotides={manifest['totals']['nucleotides']:,}",
                flush=True,
            )

        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(build_dir, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="Canonical RNA-Mamba source-data root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--temp-dir", default=None, help="Temporary disk with at least 40 GB free")
    parser.add_argument("--target-ncrna", type=int, default=3_000_000)
    parser.add_argument("--target-coding", type=int, default=2_000_000)
    parser.add_argument(
        "--fill-coding-shortfall-with-cds",
        action="store_true",
        help="Opt in to filling a coding shortfall with labelled eukaryotic CDS views",
    )
    parser.add_argument("--min-length", type=int, default=18)
    parser.add_argument("--max-length", type=int, default=10_240)
    parser.add_argument("--train-fraction", type=float, default=0.98)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2357)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare_corpus(args)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
