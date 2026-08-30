#!/usr/bin/env python3
"""Independent validation of a B73 Phase-I full-genome manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_CHROMS = tuple(f"chr{i}" for i in range(1, 11))
EXPECTED_BP = 2_131_846_805
EXPECTED_ROWS = 260_239


def read_fai(path: Path) -> dict[str, int]:
    return {line.split("\t", 1)[0]: int(line.split("\t")[1]) for line in path.read_text().splitlines() if line.strip()}


def validate(manifest: Path, fasta: Path | None = None) -> dict:
    import pyarrow.parquet as pq

    table = pq.read_table(manifest)
    required = {"region_id", "genotype", "chromosome", "start", "end", "valid_bp", "padded_bp", "is_tail", "window_size", "stride"}
    missing = sorted(required - set(table.column_names))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    rows = table.to_pylist()
    if not rows:
        raise ValueError("Manifest is empty")
    if [int(row["region_id"]) for row in rows] != list(range(len(rows))):
        raise ValueError("region_id is not contiguous from zero")
    chromosomes = []
    chrom_lengths = {}
    previous = {}
    for row in rows:
        chrom = str(row["chromosome"])
        if chrom not in chrom_lengths:
            chromosomes.append(chrom)
            chrom_lengths[chrom] = 0
        start, end = int(row["start"]), int(row["end"])
        valid, padded = int(row["valid_bp"]), int(row["padded_bp"])
        if end - start != valid or valid + padded != int(row["window_size"]):
            raise ValueError(f"Invalid interval/padding row {row}")
        if int(row["stride"]) != int(row["window_size"]) or start % int(row["stride"]) != 0:
            raise ValueError(f"Non-deterministic stride row {row}")
        if bool(row["is_tail"]) != (valid < int(row["window_size"])):
            raise ValueError(f"Invalid tail flag row {row}")
        if chrom not in previous and start != 0:
            raise ValueError(f"First window on {chrom} must start at zero")
        if chrom in previous and start != previous[chrom][1]:
            raise ValueError(f"Overlap/gap in {chrom}: {previous[chrom]} -> {(start, end)}")
        previous[chrom] = (start, end)
        chrom_lengths[chrom] = max(chrom_lengths[chrom], end)

    if set(chromosomes) != set(EXPECTED_CHROMS):
        raise ValueError(f"Expected chr1--chr10, got {chromosomes}")
    total_bp = sum(chrom_lengths.values())
    full_windows = sum(int(row["valid_bp"]) == int(row["window_size"]) for row in rows)
    tail_rows = [row for row in rows if bool(row["is_tail"])]
    result = {
        "manifest": str(manifest),
        "chromosomes": chromosomes,
        "rows": len(rows),
        "total_valid_bp": total_bp,
        "full_windows": full_windows,
        "tail_windows": len(tail_rows),
        "tail_valid_bp": sum(int(row["valid_bp"]) for row in tail_rows),
        "padding_bp": sum(int(row["padded_bp"]) for row in tail_rows),
        "coverage": total_bp / total_bp,
        "expected_rows": EXPECTED_ROWS,
        "expected_total_valid_bp": EXPECTED_BP,
    }
    if len(rows) != EXPECTED_ROWS or total_bp != EXPECTED_BP:
        raise ValueError(f"B73 chr1--chr10 totals do not match teacher plan: {result}")
    if fasta is not None:
        fai = Path(f"{fasta}.fai")
        fai_lengths = read_fai(fai)
        if any(fai_lengths[chrom] != chrom_lengths[chrom] for chrom in EXPECTED_CHROMS):
            raise ValueError("Manifest chromosome lengths disagree with FASTA FAI")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--fasta", type=Path)
    args = parser.parse_args()
    result = validate(args.manifest.expanduser().resolve(), args.fasta.expanduser().resolve() if args.fasta else None)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
