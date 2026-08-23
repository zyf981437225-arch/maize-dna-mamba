#!/usr/bin/env python
"""Prepare contig-safe maize genomic windows for DNA-Mamba MLM.

The script streams FASTA records, never concatenates records, assigns each
contig to exactly one split, and writes fixed-size windows as memory-mapped
sequence/offset files. It also emits a machine-readable manifest and a human
readable DATA_STATS.md before any formal pretraining is attempted.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import shutil
import struct
import uuid
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Optional


CANONICAL_DNA = set("ACGTN")
PREDICTABLE_DNA = set("ACGT")
IUPAC_AMBIGUITY = set("RYSWKMBDHV")
SPLITS = ("train", "val", "test")


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def _default_genome_name(path: Path) -> str:
    name = path.name
    for suffix in (".fasta.gz", ".fna.gz", ".fa.gz", ".fasta", ".fna", ".fa", ".gz"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _iter_fasta_events(path: Path) -> Iterator[tuple[str, Optional[str]]]:
    """Yield (start|sequence|end, value) without joining whole chromosomes."""

    seen_header = False
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seen_header:
                    yield "end", None
                identifier = line[1:].split(maxsplit=1)[0]
                if not identifier:
                    raise ValueError(f"{path}:{line_number} has an empty FASTA identifier")
                yield "start", identifier
                seen_header = True
            else:
                if not seen_header:
                    raise ValueError(
                        f"{path}:{line_number} contains sequence before the first FASTA header"
                    )
                yield "sequence", "".join(line.split()).upper()
    if seen_header:
        yield "end", None


def _load_split_config(path: Optional[Path]) -> dict[str, str]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Split config must be a JSON object")
    assignments: dict[str, str] = {}
    for split in SPLITS:
        entries = raw.get(split, [])
        if not isinstance(entries, list):
            raise ValueError(f"Split config field {split!r} must be a list")
        for entry in entries:
            key = str(entry).strip()
            if not key:
                raise ValueError(f"Split config field {split!r} contains an empty contig")
            previous = assignments.get(key)
            if previous is not None and previous != split:
                raise ValueError(
                    f"Contig selector {key!r} is assigned to both {previous} and {split}"
                )
            assignments[key] = split
    extra = set(raw) - set(SPLITS)
    if extra:
        raise ValueError(f"Unknown split config keys: {sorted(extra)}")
    return assignments


def _resolve_split(
    genome: str,
    contig: str,
    assignments: dict[str, str],
    unassigned_policy: str,
) -> Optional[str]:
    qualified = f"{genome}::{contig}"
    if qualified in assignments:
        return assignments[qualified]
    if contig in assignments:
        return assignments[contig]
    if unassigned_policy == "skip":
        return None
    if unassigned_policy == "train":
        return "train"
    raise ValueError(
        f"Contig {qualified!r} is not assigned by the split config; use a qualified "
        "'genome::contig' selector, an unqualified contig selector, or change "
        "--unassigned-policy"
    )


def _normalize_chunk(
    sequence: str,
    ambiguity_policy: str,
    character_counts: Counter,
) -> str:
    normalized = []
    for base in sequence:
        character_counts[base] += 1
        if base in CANONICAL_DNA:
            normalized.append(base)
        elif base in IUPAC_AMBIGUITY or base == "U":
            if ambiguity_policy == "error":
                raise ValueError(f"Encountered non-ACGTN nucleotide {base!r}")
            normalized.append("N" if ambiguity_policy == "map_to_n" else "?")
        else:
            if ambiguity_policy == "error":
                raise ValueError(f"Encountered invalid FASTA character {base!r}")
            normalized.append("N" if ambiguity_policy == "map_to_n" else "?")
    return "".join(normalized)


class IndexedSplitWriter:
    def __init__(self, output_dir: Path, split: str, window_size: int) -> None:
        self.split = split
        self.window_size = int(window_size)
        self.sequence_name = f"{split}.sequences.bin"
        self.offset_name = f"{split}.offsets.u64"
        self.metadata_name = f"{split}.windows.tsv.gz"
        self.sequence_handle = (output_dir / self.sequence_name).open("wb")
        self.offset_handle = (output_dir / self.offset_name).open("wb")
        self.metadata_handle = gzip.open(
            output_dir / self.metadata_name, "wt", encoding="utf-8", newline=""
        )
        self.metadata_writer = csv.writer(self.metadata_handle, delimiter="\t")
        self.metadata_writer.writerow(
            ("window_id", "genome", "contig", "start", "end", "n_fraction")
        )
        self.offset = 0
        self.offset_handle.write(struct.pack("<Q", self.offset))
        self.windows = 0
        self.nucleotides = 0
        self.filtered_noncanonical = 0
        self.filtered_n = 0

    @property
    def filtered_windows(self) -> int:
        return self.filtered_noncanonical + self.filtered_n

    def add(self, genome: str, contig: str, start: int, sequence: str) -> None:
        if len(sequence) != self.window_size:
            raise ValueError("Attempted to write a non-fixed-length DNA window")
        if set(sequence) - CANONICAL_DNA:
            raise ValueError("Attempted to write a non-ACGTN DNA window")
        encoded = sequence.encode("ascii")
        self.sequence_handle.write(encoded)
        self.offset += len(encoded)
        self.offset_handle.write(struct.pack("<Q", self.offset))
        self.metadata_writer.writerow(
            (
                self.windows,
                genome,
                contig,
                int(start),
                int(start) + self.window_size,
                f"{sequence.count('N') / self.window_size:.8f}",
            )
        )
        self.windows += 1
        self.nucleotides += len(encoded)

    def close(self) -> None:
        self.sequence_handle.close()
        self.offset_handle.close()
        self.metadata_handle.close()

    def manifest_entry(self) -> dict:
        return {
            "windows": self.windows,
            "nucleotides": self.nucleotides,
            "filtered_windows": self.filtered_windows,
            "filtered_noncanonical": self.filtered_noncanonical,
            "filtered_n": self.filtered_n,
            "files": {
                "sequences": self.sequence_name,
                "offsets": self.offset_name,
                "metadata": self.metadata_name,
            },
        }


def _percentage(count: int, total: int) -> float:
    return 0.0 if total == 0 else 100.0 * int(count) / int(total)


def _write_stats_markdown(path: Path, manifest: dict) -> None:
    counts = manifest["base_statistics"]
    total_bp = int(counts["total_bp"])
    lines = [
        "# Maize DNA data statistics",
        "",
        f"- Species: {manifest['species']}",
        f"- Genomes: {manifest['genomes']}",
        f"- Total bp: {total_bp:,}",
        f"- Contigs: {manifest['total_contigs']:,}",
        f"- Window size: {manifest['window_size']:,}",
        f"- Stride: {manifest['stride']:,}",
        f"- Ambiguity policy: `{manifest['ambiguity_policy']}`",
        f"- Maximum N fraction per retained window: {manifest['max_n_fraction']:.4f}",
        "",
        "## Base composition",
        "",
        "| Symbol class | Count | Percentage |",
        "|---|---:|---:|",
    ]
    for key in ("A", "C", "G", "T", "N", "U", "iupac_ambiguity", "invalid"):
        value = int(counts[key])
        lines.append(f"| {key} | {value:,} | {_percentage(value, total_bp):.6f}% |")
    lines.extend(
        [
            "",
            "## Window counts",
            "",
            "| Split | Contigs | Retained windows | Filtered windows | Nucleotides |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split in SPLITS:
        info = manifest["splits"][split]
        lines.append(
            f"| {split} | {info['contigs']:,} | {info['windows']:,} | "
            f"{info['filtered_windows']:,} | {info['nucleotides']:,} |"
        )
    sanity = manifest["sanity_checks"]
    lines.extend(
        [
            "",
            "## Training gates",
            "",
            f"- DNA U-fraction check: {'PASS' if sanity['u_fraction_passed'] else 'FAIL'} "
            f"(observed {sanity['u_fraction']:.8f}, maximum {sanity['max_u_fraction']:.8f})",
            f"- Non-empty train/val/test splits: {'PASS' if sanity['all_splits_nonempty'] else 'FAIL'}",
            f"- Formal pretraining ready: {'YES' if sanity['formal_pretraining_ready'] else 'NO'}",
            "",
            "Formal pretraining must not start while any gate above is failing.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_genome(
    *,
    fasta_paths: Iterable[Path],
    output_dir: Path,
    genome_names: Optional[Iterable[str]] = None,
    species: str = "Zea mays",
    window_size: int = 10240,
    stride: int = 5120,
    split_config: Optional[Path] = None,
    unassigned_policy: str = "train",
    ambiguity_policy: str = "map_to_n",
    max_n_fraction: float = 1.0,
    max_u_fraction: float = 0.0,
    require_all_splits: bool = False,
    overwrite: bool = False,
) -> dict:
    fasta_paths = [Path(path).expanduser().resolve() for path in fasta_paths]
    if not fasta_paths:
        raise ValueError("At least one FASTA file is required")
    for path in fasta_paths:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)
    if int(window_size) <= 0 or int(stride) <= 0:
        raise ValueError("window_size and stride must be positive")
    if not 0.0 <= float(max_n_fraction) <= 1.0:
        raise ValueError("max_n_fraction must be in [0, 1]")
    if not 0.0 <= float(max_u_fraction) <= 1.0:
        raise ValueError("max_u_fraction must be in [0, 1]")
    if unassigned_policy not in {"train", "skip", "error"}:
        raise ValueError("unassigned_policy must be train, skip, or error")
    if ambiguity_policy not in {"map_to_n", "filter_window", "error"}:
        raise ValueError("ambiguity_policy must be map_to_n, filter_window, or error")

    if genome_names is None:
        names = [_default_genome_name(path) for path in fasta_paths]
    else:
        names = [str(name).strip() for name in genome_names]
        if len(names) != len(fasta_paths):
            raise ValueError("--genome-name must be supplied once per --fasta")
        if any(not name for name in names):
            raise ValueError("Genome names cannot be empty")
    if len(set(names)) != len(names):
        raise ValueError("Genome names must be unique")

    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_dir}; pass --overwrite")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir = output_dir.parent / f".{output_dir.name}.building-{uuid.uuid4().hex}"
    build_dir.mkdir(parents=False)

    assignments = _load_split_config(
        None if split_config is None else Path(split_config).expanduser().resolve()
    )
    writers = {
        split: IndexedSplitWriter(build_dir, split, int(window_size))
        for split in SPLITS
    }
    total_character_counts: Counter = Counter()
    contig_records: list[dict] = []
    seen_contigs: set[str] = set()

    try:
        for fasta_path, genome in zip(fasta_paths, names):
            current_contig = None
            current_split = None
            buffer = ""
            buffer_start = 0
            next_start = 0
            contig_length = 0
            last_window_end = 0
            contig_counts: Counter = Counter()
            attempted_windows = 0
            retained_windows = 0
            filtered_windows = 0

            def finish_contig() -> None:
                nonlocal current_contig, current_split, buffer, buffer_start
                nonlocal next_start, contig_length, last_window_end
                nonlocal contig_counts, attempted_windows, retained_windows
                nonlocal filtered_windows
                if current_contig is None:
                    return
                contig_records.append(
                    {
                        "genome": genome,
                        "contig": current_contig,
                        "split": current_split if current_split is not None else "skipped",
                        "length": contig_length,
                        "attempted_windows": attempted_windows,
                        "retained_windows": retained_windows,
                        "filtered_windows": filtered_windows,
                        "tail_bp_after_last_window": max(0, contig_length - last_window_end),
                        "character_counts": dict(sorted(contig_counts.items())),
                    }
                )
                current_contig = None

            for event, value in _iter_fasta_events(fasta_path):
                if event == "start":
                    finish_contig()
                    current_contig = str(value)
                    qualified = f"{genome}::{current_contig}"
                    if qualified in seen_contigs:
                        raise ValueError(f"Duplicate FASTA record {qualified!r}")
                    seen_contigs.add(qualified)
                    current_split = _resolve_split(
                        genome, current_contig, assignments, unassigned_policy
                    )
                    buffer = ""
                    buffer_start = 0
                    next_start = 0
                    contig_length = 0
                    last_window_end = 0
                    contig_counts = Counter()
                    attempted_windows = 0
                    retained_windows = 0
                    filtered_windows = 0
                    continue
                if event == "end":
                    finish_contig()
                    continue
                if current_contig is None:
                    raise RuntimeError("Internal FASTA parser state error")
                raw_chunk = str(value)
                normalized = _normalize_chunk(
                    raw_chunk, ambiguity_policy, contig_counts
                )
                total_character_counts.update(raw_chunk)
                contig_length += len(normalized)
                if current_split is None:
                    continue
                buffer += normalized
                available_end = buffer_start + len(buffer)
                writer = writers[current_split]
                while next_start + int(window_size) <= available_end:
                    offset = next_start - buffer_start
                    window = buffer[offset : offset + int(window_size)]
                    attempted_windows += 1
                    last_window_end = next_start + int(window_size)
                    if "?" in window:
                        writer.filtered_noncanonical += 1
                        filtered_windows += 1
                    elif window.count("N") / int(window_size) > float(max_n_fraction):
                        writer.filtered_n += 1
                        filtered_windows += 1
                    elif not any(base in PREDICTABLE_DNA for base in window):
                        writer.filtered_noncanonical += 1
                        filtered_windows += 1
                    else:
                        writer.add(genome, current_contig, next_start, window)
                        retained_windows += 1
                    next_start += int(stride)
                if next_start > buffer_start:
                    drop = min(next_start - buffer_start, len(buffer))
                    buffer = buffer[drop:]
                    buffer_start += drop
            finish_contig()

        for writer in writers.values():
            writer.close()

        total_bp = sum(total_character_counts.values())
        ambiguity_counts = {
            base: int(total_character_counts[base])
            for base in sorted(IUPAC_AMBIGUITY)
            if total_character_counts[base]
        }
        known = CANONICAL_DNA | IUPAC_AMBIGUITY | {"U"}
        invalid_counts = {
            base: int(count)
            for base, count in sorted(total_character_counts.items())
            if base not in known
        }
        base_statistics = {
            base: int(total_character_counts[base])
            for base in ("A", "C", "G", "T", "N", "U")
        }
        base_statistics.update(
            {
                "iupac_ambiguity": sum(ambiguity_counts.values()),
                "invalid": sum(invalid_counts.values()),
                "total_bp": int(total_bp),
                "iupac_character_counts": ambiguity_counts,
                "invalid_character_counts": invalid_counts,
            }
        )
        split_contig_counts = {
            split: sum(record["split"] == split for record in contig_records)
            for split in SPLITS
        }
        u_fraction = 0.0 if total_bp == 0 else total_character_counts["U"] / total_bp
        all_splits_nonempty = all(writers[split].windows > 0 for split in SPLITS)
        u_fraction_passed = u_fraction <= float(max_u_fraction)
        formal_ready = u_fraction_passed and (
            all_splits_nonempty if require_all_splits else writers["train"].windows > 0
        )

        manifest = {
            "schema_version": 1,
            "sequence_type": "dna",
            "species": str(species),
            "alphabet": "ACGTN",
            "genomes": names,
            "input_files": [
                {"genome": genome, "path": str(path), "size_bytes": path.stat().st_size}
                for path, genome in zip(fasta_paths, names)
            ],
            "window_size": int(window_size),
            "stride": int(stride),
            "ambiguity_policy": ambiguity_policy,
            "max_n_fraction": float(max_n_fraction),
            "split_config": None if split_config is None else str(Path(split_config).resolve()),
            "unassigned_policy": unassigned_policy,
            "split_unit": "whole_contig",
            "total_contigs": len(contig_records),
            "base_statistics": base_statistics,
            "splits": {},
            "contigs": contig_records,
            "sanity_checks": {
                "u_fraction": u_fraction,
                "max_u_fraction": float(max_u_fraction),
                "u_fraction_passed": u_fraction_passed,
                "all_splits_nonempty": all_splits_nonempty,
                "require_all_splits": bool(require_all_splits),
                "formal_pretraining_ready": formal_ready,
            },
        }
        for split in SPLITS:
            entry = writers[split].manifest_entry()
            entry["contigs"] = split_contig_counts[split]
            manifest["splits"][split] = entry
        manifest["totals"] = {
            "windows": sum(writers[split].windows for split in SPLITS),
            "filtered_windows": sum(writers[split].filtered_windows for split in SPLITS),
            "nucleotides": sum(writers[split].nucleotides for split in SPLITS),
        }

        (build_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _write_stats_markdown(build_dir / "DATA_STATS.md", manifest)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(build_dir, output_dir)
        return manifest
    except Exception:
        for writer in writers.values():
            for handle_name in ("sequence_handle", "offset_handle", "metadata_handle"):
                handle = getattr(writer, handle_name, None)
                if handle is not None and not handle.closed:
                    handle.close()
        shutil.rmtree(build_dir, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", action="append", required=True, help="Input FASTA; repeat for multiple genomes")
    parser.add_argument("--genome-name", action="append", help="Genome label; repeat once per FASTA")
    parser.add_argument("--species", default="Zea mays")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-size", type=int, default=10240)
    parser.add_argument("--stride", type=int, default=5120)
    parser.add_argument("--split-config", help="JSON with train/val/test contig lists")
    parser.add_argument(
        "--unassigned-policy", choices=("train", "skip", "error"), default="train"
    )
    parser.add_argument(
        "--ambiguity-policy",
        choices=("map_to_n", "filter_window", "error"),
        default="map_to_n",
    )
    parser.add_argument("--max-n-fraction", type=float, default=1.0)
    parser.add_argument("--max-u-fraction", type=float, default=0.0)
    parser.add_argument("--require-all-splits", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare_genome(
        fasta_paths=[Path(path) for path in args.fasta],
        output_dir=Path(args.output_dir),
        genome_names=args.genome_name,
        species=args.species,
        window_size=args.window_size,
        stride=args.stride,
        split_config=None if args.split_config is None else Path(args.split_config),
        unassigned_policy=args.unassigned_policy,
        ambiguity_policy=args.ambiguity_policy,
        max_n_fraction=args.max_n_fraction,
        max_u_fraction=args.max_u_fraction,
        require_all_splits=args.require_all_splits,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest["totals"], indent=2))
    print(f"Wrote {Path(args.output_dir).resolve() / 'DATA_STATS.md'}")
    if not manifest["sanity_checks"]["formal_pretraining_ready"]:
        raise SystemExit(
            "Corpus was prepared for inspection, but formal pretraining gates are not satisfied"
        )


if __name__ == "__main__":
    main()
