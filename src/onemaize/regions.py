"""Build annotation-aware candidate regions for OneMaize pretraining.

Coordinates written by this module are always zero-based and half-open.  GFF3
inputs are converted exactly once at the parser boundary.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import uuid
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional


REGION_CLASSES = ("gene_centered", "non_repeat", "te_rich")


@dataclass(frozen=True)
class GenomeInput:
    """Files and default split for one maize genotype."""

    genotype: str
    fasta: Path
    genes_gff3: Path
    te_gff3: Path
    split: str = "train"

    def resolved(self) -> "GenomeInput":
        split = str(self.split).lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split {self.split!r} for {self.genotype}")
        return GenomeInput(
            genotype=str(self.genotype).strip(),
            fasta=Path(self.fasta).expanduser().resolve(),
            genes_gff3=Path(self.genes_gff3).expanduser().resolve(),
            te_gff3=Path(self.te_gff3).expanduser().resolve(),
            split=split,
        )


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def _read_fai(path: Path) -> dict[str, int]:
    lengths: dict[str, int] = {}
    with path.open("rt", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            fields = raw.rstrip("\n").split("\t")
            if len(fields) < 2:
                raise ValueError(f"Malformed FASTA index line {path}:{line_number}")
            lengths[fields[0]] = int(fields[1])
    return lengths


def read_fasta_lengths(path: Path) -> dict[str, int]:
    """Read sequence lengths from an adjacent FAI or by streaming FASTA."""

    fai = Path(f"{path}.fai")
    if fai.exists():
        return _read_fai(fai)
    lengths: dict[str, int] = {}
    current: Optional[str] = None
    with _open_text(path) as handle:
        for line_number, raw in enumerate(handle, start=1):
            if raw.startswith(">"):
                current = raw[1:].split(maxsplit=1)[0]
                if not current:
                    raise ValueError(f"Empty FASTA identifier at {path}:{line_number}")
                if current in lengths:
                    raise ValueError(f"Duplicate FASTA identifier {current!r} in {path}")
                lengths[current] = 0
            elif raw.strip():
                if current is None:
                    raise ValueError(f"Sequence before first FASTA header at {path}:{line_number}")
                lengths[current] += len("".join(raw.split()))
    if not lengths:
        raise ValueError(f"No FASTA records found in {path}")
    return lengths


def parse_gff3_attributes(raw: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in raw.rstrip(";").split(";"):
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            result[key] = value
        elif " " in item:
            key, value = item.split(" ", 1)
            result[key] = value.strip('"')
    return result


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((int(start), int(end)) for start, end in intervals if end > start)
    if not ordered:
        return []
    merged: list[tuple[int, int]] = []
    cur_start, cur_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return merged


class IntervalCoverage:
    """Fast union-coverage queries over sorted, non-overlapping intervals."""

    def __init__(self, intervals: Iterable[tuple[int, int]]) -> None:
        merged = merge_intervals(intervals)
        self.starts = [item[0] for item in merged]
        self.ends = [item[1] for item in merged]
        self.prefix = [0]
        for start, end in merged:
            self.prefix.append(self.prefix[-1] + end - start)

    @property
    def total(self) -> int:
        return self.prefix[-1]

    @property
    def segment_count(self) -> int:
        return len(self.starts)

    def covered_bp(self, start: int, end: int) -> int:
        start, end = int(start), int(end)
        if end <= start or not self.starts:
            return 0
        left = bisect_left(self.ends, start + 1)
        right = bisect_left(self.starts, end)
        if left >= right:
            return 0
        total = self.prefix[right] - self.prefix[left]
        total -= max(0, start - self.starts[left])
        total -= max(0, self.ends[right - 1] - end)
        return max(0, total)

    def overlaps(self, start: int, end: int) -> bool:
        return self.covered_bp(start, end) > 0


def _read_te_coverage(
    path: Path,
    lengths: dict[str, int],
    included: set[str],
) -> dict[str, IntervalCoverage]:
    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    malformed = 0
    with _open_text(path) as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip() or raw.startswith("#"):
                continue
            fields = raw.rstrip("\n").split("\t")
            if len(fields) != 9:
                malformed += 1
                continue
            seqid = fields[0]
            if seqid not in included:
                continue
            if seqid not in lengths:
                raise ValueError(f"TE GFF3 seqid {seqid!r} is absent from {path}")
            start, end = int(fields[3]) - 1, int(fields[4])
            if start < 0 or end <= start or end > lengths[seqid]:
                raise ValueError(
                    f"Invalid TE coordinates at {path}:{line_number}: {seqid}:{start + 1}-{end}"
                )
            intervals[seqid].append((start, end))
    if malformed:
        raise ValueError(f"TE GFF3 contains {malformed} malformed feature lines: {path}")
    return {seqid: IntervalCoverage(intervals.get(seqid, [])) for seqid in included}


def _read_protein_coding_genes(
    path: Path,
    lengths: dict[str, int],
    included: set[str],
) -> dict[str, list[dict]]:
    genes: dict[str, list[dict]] = defaultdict(list)
    with _open_text(path) as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip() or raw.startswith("#"):
                continue
            fields = raw.rstrip("\n").split("\t")
            if len(fields) != 9:
                raise ValueError(f"Malformed GFF3 feature at {path}:{line_number}")
            seqid, _, feature_type, start_s, end_s, _, strand, _, attributes_s = fields
            if feature_type != "gene" or seqid not in included:
                continue
            if seqid not in lengths:
                raise ValueError(f"Gene GFF3 seqid {seqid!r} is absent from FASTA")
            attributes = parse_gff3_attributes(attributes_s)
            biotype = attributes.get("biotype", attributes.get("gene_biotype", ""))
            if biotype and biotype != "protein_coding":
                continue
            start, end = int(start_s) - 1, int(end_s)
            if start < 0 or end <= start or end > lengths[seqid]:
                raise ValueError(
                    f"Invalid gene coordinates at {path}:{line_number}: {seqid}:{start + 1}-{end}"
                )
            gene_id = attributes.get("ID") or attributes.get("gene_id")
            if not gene_id:
                raise ValueError(f"Gene without ID at {path}:{line_number}")
            genes[seqid].append(
                {"gene_id": gene_id, "start": start, "end": end, "strand": strand}
            )
    return genes


def _expand_interval(start: int, end: int, minimum: int, length: int) -> tuple[int, int]:
    if end - start >= minimum:
        return start, end
    if length < minimum:
        raise ValueError(f"Sequence length {length} is shorter than context {minimum}")
    missing = minimum - (end - start)
    start -= missing // 2
    end += missing - missing // 2
    if start < 0:
        end -= start
        start = 0
    if end > length:
        start -= end - length
        end = length
    return max(0, start), min(length, end)


def _window_starts(length: int, span: int, stride: int) -> Iterator[int]:
    if length < span:
        return
    last = length - span
    start = 0
    while start <= last:
        yield start
        start += stride
    previous = start - stride
    if previous != last:
        yield last


def _split_for_seqid(
    default: str,
    seqid: str,
    val_seqids: set[str],
    test_seqids: set[str],
) -> str:
    if seqid in val_seqids:
        return "val"
    if seqid in test_seqids:
        return "test"
    return default


def _write_parquet(path: Path, rows: list[dict]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Building OneMaize metadata requires pyarrow") from exc
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd")


def _write_stats(path: Path, manifest: dict) -> None:
    lines = [
        "# OneMaize candidate-region statistics",
        "",
        f"- Schema version: {manifest['schema_version']}",
        f"- Genotypes: {manifest['genotype_count']}",
        "- Genotype split: "
        + "/".join(
            str(manifest["genotype_split_counts"].get(split, 0))
            for split in ("train", "val", "test")
        )
        + " (train/val/test)",
        f"- Formal 26-genotype split validated: {manifest['formal_split_validated']}",
        f"- Candidate regions: {manifest['region_count']:,}",
        f"- Primary context: {manifest['primary_context']:,} bp",
        f"- Extended context: {manifest['extended_context']:,} bp",
        f"- Genome-wide candidate span: {manifest['candidate_span']:,} bp",
        f"- Genome-wide stride: {manifest['candidate_stride']:,} bp",
        f"- TE-rich threshold: {manifest['repeat_threshold']:.3f}",
        "",
        "## Region counts",
        "",
        "| Split | Genotype | Gene-centered | Non-repeat | TE-rich |",
        "|---|---|---:|---:|---:|",
    ]
    counts = manifest["counts"]
    for split in ("train", "val", "test"):
        for genotype in sorted(counts.get(split, {})):
            row = counts[split][genotype]
            lines.append(
                f"| {split} | {genotype} | {row.get('gene_centered', 0):,} | "
                f"{row.get('non_repeat', 0):,} | {row.get('te_rich', 0):,} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_onemaize_index(
    inputs: Iterable[GenomeInput],
    output_dir: Path,
    *,
    primary_context: int = 8192,
    extended_context: int = 16384,
    candidate_span: int = 32768,
    candidate_stride: int = 16384,
    gene_flank: int = 5000,
    repeat_threshold: float = 0.5,
    seqid_regex: str = r"^chr(?:[1-9]|10)$",
    val_seqids: Iterable[str] = (),
    test_seqids: Iterable[str] = (),
    exclude_gene_bodies_from_non_repeat: bool = True,
    require_all_classes: bool = True,
    expected_genotype_count: Optional[int] = None,
    expected_split_counts: Optional[dict[str, int]] = None,
    overwrite: bool = False,
) -> dict:
    """Build ``genomes.parquet`` and ``regions.parquet`` for dynamic sampling."""

    records = [item.resolved() for item in inputs]
    if not records:
        raise ValueError("At least one GenomeInput is required")
    genotypes = [item.genotype for item in records]
    if any(not item for item in genotypes) or len(set(genotypes)) != len(genotypes):
        raise ValueError("Genotype names must be non-empty and unique")
    genotype_split_counts = Counter(item.split for item in records)
    if expected_genotype_count is not None and len(records) != expected_genotype_count:
        raise ValueError(
            f"Expected {expected_genotype_count} genotypes, found {len(records)}"
        )
    if expected_split_counts is not None:
        normalized_expected = {
            split: int(expected_split_counts.get(split, 0))
            for split in ("train", "val", "test")
        }
        observed = {
            split: int(genotype_split_counts.get(split, 0))
            for split in ("train", "val", "test")
        }
        if observed != normalized_expected:
            raise ValueError(
                f"Expected genotype split counts {normalized_expected}, found {observed}"
            )
    for record in records:
        for path in (record.fasta, record.genes_gff3, record.te_gff3):
            if not path.is_file():
                raise FileNotFoundError(path)
    if primary_context <= 0 or extended_context < primary_context:
        raise ValueError("Require 0 < primary_context <= extended_context")
    if candidate_span < extended_context or candidate_stride <= 0:
        raise ValueError("candidate_span must cover extended_context and stride must be positive")
    if not 0.0 <= repeat_threshold <= 1.0:
        raise ValueError("repeat_threshold must be in [0, 1]")
    val_set, test_set = set(val_seqids), set(test_seqids)
    if expected_split_counts is not None and (val_set or test_set):
        raise ValueError(
            "Formal genotype splits cannot be combined with pilot chromosome overrides"
        )
    overlap = val_set & test_set
    if overlap:
        raise ValueError(f"Seqids assigned to both val and test: {sorted(overlap)}")
    seqid_pattern = re.compile(seqid_regex)

    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_dir}; pass overwrite=True")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir = output_dir.parent / f".{output_dir.name}.building-{uuid.uuid4().hex}"
    build_dir.mkdir()

    regions: list[dict] = []
    genomes: list[dict] = []
    counts: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    try:
        for record in records:
            lengths = read_fasta_lengths(record.fasta)
            included = {seqid for seqid in lengths if seqid_pattern.search(seqid)}
            if not included:
                raise ValueError(
                    f"No FASTA seqids for {record.genotype} match regex {seqid_regex!r}"
                )
            te_coverage = _read_te_coverage(record.te_gff3, lengths, included)
            genes = _read_protein_coding_genes(record.genes_gff3, lengths, included)
            gene_body_coverage = {
                seqid: IntervalCoverage(
                    (gene["start"], gene["end"]) for gene in genes.get(seqid, [])
                )
                for seqid in included
            }

            for seqid in sorted(included):
                sequence_length = lengths[seqid]
                split = _split_for_seqid(record.split, seqid, val_set, test_set)
                coverage = te_coverage[seqid]
                for gene in genes.get(seqid, []):
                    start = max(0, gene["start"] - gene_flank)
                    end = min(sequence_length, gene["end"] + gene_flank)
                    start, end = _expand_interval(
                        start, end, extended_context, sequence_length
                    )
                    repeat_fraction = coverage.covered_bp(start, end) / (end - start)
                    regions.append(
                        {
                            "region_id": f"{record.genotype}:gene:{gene['gene_id']}",
                            "genotype": record.genotype,
                            "split": split,
                            "seqid": seqid,
                            "start": start,
                            "end": end,
                            "region_class": "gene_centered",
                            "repeat_fraction": float(repeat_fraction),
                            "gene_id": gene["gene_id"],
                            "gene_start": gene["start"],
                            "gene_end": gene["end"],
                            "strand": gene["strand"],
                        }
                    )
                    counts[split][record.genotype]["gene_centered"] += 1

                for start in _window_starts(
                    sequence_length, candidate_span, candidate_stride
                ):
                    end = start + candidate_span
                    repeat_fraction = coverage.covered_bp(start, end) / candidate_span
                    region_class = (
                        "te_rich" if repeat_fraction >= repeat_threshold else "non_repeat"
                    )
                    if (
                        region_class == "non_repeat"
                        and exclude_gene_bodies_from_non_repeat
                        and gene_body_coverage[seqid].overlaps(start, end)
                    ):
                        continue
                    regions.append(
                        {
                            "region_id": (
                                f"{record.genotype}:{seqid}:{start}:{end}:{region_class}"
                            ),
                            "genotype": record.genotype,
                            "split": split,
                            "seqid": seqid,
                            "start": start,
                            "end": end,
                            "region_class": region_class,
                            "repeat_fraction": float(repeat_fraction),
                            "gene_id": None,
                            "gene_start": None,
                            "gene_end": None,
                            "strand": None,
                        }
                    )
                    counts[split][record.genotype][region_class] += 1

            genomes.append(
                {
                    "genotype": record.genotype,
                    "default_split": record.split,
                    "fasta": str(record.fasta),
                    "fasta_fai": (
                        str(Path(f"{record.fasta}.fai"))
                        if Path(f"{record.fasta}.fai").exists()
                        else None
                    ),
                    "fasta_gzi": (
                        str(Path(f"{record.fasta}.gzi"))
                        if Path(f"{record.fasta}.gzi").exists()
                        else None
                    ),
                    "genes_gff3": str(record.genes_gff3),
                    "te_gff3": str(record.te_gff3),
                    "total_bp": sum(lengths[seqid] for seqid in included),
                    "sequence_count": len(included),
                    "protein_coding_gene_count": sum(
                        len(genes.get(seqid, [])) for seqid in included
                    ),
                    "repeat_union_bp": sum(te_coverage[seqid].total for seqid in included),
                    "included_seqids": sorted(included),
                }
            )

        plain_counts = {
            split: {
                genotype: dict(class_counts)
                for genotype, class_counts in genotype_counts.items()
            }
            for split, genotype_counts in counts.items()
        }
        if require_all_classes:
            missing = []
            for split, genotype_counts in plain_counts.items():
                for genotype, class_counts in genotype_counts.items():
                    for region_class in REGION_CLASSES:
                        if class_counts.get(region_class, 0) == 0:
                            missing.append(f"{split}/{genotype}/{region_class}")
            if missing:
                raise ValueError(
                    "Every represented split/genotype requires all region classes; missing: "
                    + ", ".join(missing)
                )

        _write_parquet(build_dir / "genomes.parquet", genomes)
        _write_parquet(build_dir / "regions.parquet", regions)
        manifest = {
            "schema_version": 2,
            "coordinate_system": "0-based-half-open",
            "alphabet": "ACGTN",
            "genotype_count": len(genomes),
            "genotype_split_counts": {
                split: int(genotype_split_counts.get(split, 0))
                for split in ("train", "val", "test")
            },
            "formal_split_validated": expected_split_counts is not None,
            "region_count": len(regions),
            "region_classes": list(REGION_CLASSES),
            "primary_context": int(primary_context),
            "extended_context": int(extended_context),
            "candidate_span": int(candidate_span),
            "candidate_stride": int(candidate_stride),
            "gene_flank": int(gene_flank),
            "repeat_threshold": float(repeat_threshold),
            "seqid_regex": seqid_regex,
            "exclude_gene_bodies_from_non_repeat": bool(
                exclude_gene_bodies_from_non_repeat
            ),
            "pilot_val_seqids": sorted(val_set),
            "pilot_test_seqids": sorted(test_set),
            "counts": plain_counts,
            "files": {
                "genomes": "genomes.parquet",
                "regions": "regions.parquet",
            },
        }
        (build_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_stats(build_dir / "DATA_STATS.md", manifest)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(build_dir, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise
