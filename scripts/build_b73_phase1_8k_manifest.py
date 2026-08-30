#!/usr/bin/env python3
"""Build a deterministic, exhaustive B73 chr1--chr10 8,192-bp manifest.

The manifest is an index only: sequences remain in the user's indexed FASTA.
Coordinates are zero-based, half-open and never cross chromosome boundaries.
The final short interval of every chromosome is retained and padded by the
dataset at read time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CHROMOSOMES = tuple(f"chr{i}" for i in range(1, 11))


def read_fai(path: Path) -> dict[str, int]:
    lengths: dict[str, int] = {}
    with path.open("rt", encoding="utf-8") as handle:
        for raw in handle:
            fields = raw.rstrip("\n").split("\t")
            if len(fields) < 2:
                raise ValueError(f"Malformed FASTA index line: {raw!r}")
            name, length = fields[0], int(fields[1])
            if name in lengths:
                raise ValueError(f"Duplicate FASTA index record: {name}")
            if length <= 0:
                raise ValueError(f"Non-positive chromosome length for {name}")
            lengths[name] = length
    return lengths


def build_rows(
    lengths: dict[str, int],
    chromosomes: tuple[str, ...],
    fasta: Path,
    fasta_fai: Path,
    *,
    genotype: str = "B73",
    window_size: int = 8192,
    stride: int = 8192,
) -> list[dict]:
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if stride != window_size:
        raise ValueError("Phase-I requires non-overlapping windows: stride == window_size")
    if len(set(chromosomes)) != len(chromosomes):
        raise ValueError("chromosomes must be unique")
    missing = [chrom for chrom in chromosomes if chrom not in lengths]
    if missing:
        raise ValueError(f"Chromosomes missing from FAI: {missing}")

    rows: list[dict] = []
    region_id = 0
    fasta_text = str(fasta.resolve())
    fai_text = str(fasta_fai.resolve())
    for chrom in chromosomes:
        length = int(lengths[chrom])
        start = 0
        while start < length:
            end = min(start + window_size, length)
            valid_bp = end - start
            padded_bp = window_size - valid_bp
            rows.append(
                {
                    "region_id": region_id,
                    "genotype": genotype,
                    "chromosome": chrom,
                    "start": start,
                    "end": end,
                    "valid_bp": valid_bp,
                    "padded_bp": padded_bp,
                    "is_tail": bool(valid_bp < window_size),
                    "window_size": window_size,
                    "stride": stride,
                    "fasta": fasta_text,
                    "fasta_fai": fai_text,
                }
            )
            region_id += 1
            start += stride
    return rows


def write_parquet(rows: list[dict], output: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency is in training env
        raise RuntimeError("pyarrow is required to build the manifest") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    metadata = dict(table.schema.metadata or {})
    metadata[b"onemaize_phase"] = b"phase1_full_genome"
    metadata[b"coordinate_system"] = b"0-based-half-open"
    metadata[b"tail_policy"] = b"keep-pad"
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, output, compression="zstd")


def summarize(rows: list[dict], lengths: dict[str, int], chromosomes: tuple[str, ...], window_size: int) -> dict:
    total_bp = sum(lengths[c] for c in chromosomes)
    full = sum(not row["is_tail"] for row in rows)
    tails = [row for row in rows if row["is_tail"]]
    tail_bp = sum(row["valid_bp"] for row in tails)
    padding_bp = sum(row["padded_bp"] for row in tails)
    return {
        "chromosomes": list(chromosomes),
        "chromosome_count": len(chromosomes),
        "window_size": window_size,
        "stride": window_size,
        "genome_bp": total_bp,
        "full_windows": full,
        "tail_windows": len(tails),
        "total_windows_keep_pad": len(rows),
        "total_windows_drop_tail": full,
        "tail_valid_bp": tail_bp,
        "padding_bp": padding_bp,
        "coverage_keep_pad": 1.0,
        "coverage_drop_tail": (full * window_size) / total_bp,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--fai", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chromosomes", nargs="+", default=list(CHROMOSOMES))
    parser.add_argument("--genotype", default="B73")
    parser.add_argument("--window-size", type=int, default=8192)
    parser.add_argument("--stride", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    fasta = args.fasta.expanduser().resolve()
    fai = (args.fai or Path(f"{fasta}.fai")).expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not fasta.exists():
        raise FileNotFoundError(fasta)
    if not fai.exists():
        raise FileNotFoundError(fai)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
    if fasta.suffix.lower() == ".gz" and not Path(f"{fasta}.gzi").exists():
        raise FileNotFoundError(f"Missing BGZF index {fasta}.gzi")

    chromosomes = tuple(args.chromosomes)
    lengths = read_fai(fai)
    rows = build_rows(
        lengths,
        chromosomes,
        fasta,
        fai,
        genotype=args.genotype,
        window_size=args.window_size,
        stride=args.stride,
    )
    write_parquet(rows, output)
    summary = summarize(rows, lengths, chromosomes, args.window_size)
    summary["output"] = str(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
