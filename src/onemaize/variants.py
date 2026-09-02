"""Schema-v4 variant metadata for explicit OneMaize Phase-II sampling.

The existing schema-v3 candidate-region metadata remains unchanged.  This
module adds a separate ``variant_manifest.json`` and
``variant_regions.parquet`` layer.  Internal coordinates are always zero-based
and half-open; conversion from VCF happens exactly once in ``parse_vcf``.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import shutil
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional


VARIANT_TYPES = (
    "snp",
    "indel",
    "deletion",
    "insertion",
    "inversion",
    "duplication",
    "sv",
    "pav",
    "te_insertion",
    "te_deletion",
)

SMALL_VARIANT_TYPES = frozenset({"snp", "indel", "deletion", "insertion"})
STRUCTURAL_VARIANT_TYPES = frozenset(
    {"deletion", "insertion", "inversion", "duplication", "sv", "pav"}
)
TE_VARIANT_TYPES = frozenset({"te_insertion", "te_deletion"})


@dataclass(frozen=True)
class VariantInput:
    """One explicitly mapped variant file for a genotype assembly."""

    genotype: str
    variant_file: Path
    source: str
    coordinate_genotype: str
    reference_genotype: str = "B73"

    def resolved(self) -> "VariantInput":
        genotype = str(self.genotype).strip()
        coordinate_genotype = str(self.coordinate_genotype).strip()
        if not genotype or not coordinate_genotype:
            raise ValueError("genotype and coordinate_genotype must be non-empty")
        return VariantInput(
            genotype=genotype,
            variant_file=Path(self.variant_file).expanduser().resolve(),
            source=str(self.source).strip(),
            coordinate_genotype=coordinate_genotype,
            reference_genotype=str(self.reference_genotype).strip() or "B73",
        )


@dataclass(frozen=True)
class VariantEvent:
    """Canonical event in the coordinates of ``coordinate_genotype`` FASTA."""

    variant_id: str
    genotype: str
    reference_genotype: str
    coordinate_genotype: str
    seqid: str
    start: int
    end: int
    variant_type: str
    reference_allele: Optional[str]
    alternate_allele: Optional[str]
    variant_length: int
    source: str
    split: str
    left_breakpoint: Optional[int] = None
    right_breakpoint: Optional[int] = None
    te_family: Optional[str] = None
    te_superfamily: Optional[str] = None
    te_id: Optional[str] = None
    te_class: Optional[str] = None
    reference_presence: Optional[bool] = None
    alternate_presence: Optional[bool] = None

    def validated(self) -> "VariantEvent":
        if not self.variant_id:
            raise ValueError("variant_id must be non-empty")
        if self.variant_type not in VARIANT_TYPES:
            raise ValueError(f"Unsupported variant_type: {self.variant_type!r}")
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {self.split!r}")
        if self.coordinate_genotype != self.genotype:
            raise ValueError(
                "Training events must use coordinates of their genotype FASTA; "
                f"event {self.variant_id!r} has genotype={self.genotype!r}, "
                f"coordinate_genotype={self.coordinate_genotype!r}"
            )
        if self.start < 0 or self.end < self.start:
            raise ValueError(
                f"Invalid 0-based half-open event {self.variant_id}: "
                f"{self.seqid}:{self.start}-{self.end}"
            )
        for name, value in (
            ("left_breakpoint", self.left_breakpoint),
            ("right_breakpoint", self.right_breakpoint),
        ):
            if value is not None and int(value) < 0:
                raise ValueError(f"{name} must be non-negative for {self.variant_id}")
        return self


def one_based_closed_to_half_open(start: int, end: int) -> tuple[int, int]:
    """Convert a one-based closed interval to zero-based half-open once."""

    start, end = int(start), int(end)
    if start < 1 or end < start:
        raise ValueError(f"Invalid one-based closed interval: {start}-{end}")
    return start - 1, end


def one_based_insertion_after_to_boundary(position: int) -> int:
    """Convert insertion-after VCF/GFF position to a zero-based boundary."""

    position = int(position)
    if position < 1:
        raise ValueError(f"Invalid one-based insertion position: {position}")
    return position


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def _parse_info(raw: str) -> dict[str, str]:
    result: dict[str, str] = {}
    if raw in {"", "."}:
        return result
    for item in raw.split(";"):
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            result[key] = value
        else:
            result[item] = "true"
    return result


def _normalize_vcf_allele(
    position: int,
    reference: str,
    alternate: str,
    info: dict[str, str],
) -> tuple[int, int, str, int, Optional[int], Optional[int]]:
    """Normalize one VCF ALT allele into the internal coordinate convention."""

    position = int(position)
    if position < 1:
        raise ValueError(f"VCF POS must be >= 1, got {position}")
    reference = reference.upper()
    alternate = alternate.upper()
    symbolic = alternate.startswith("<") and alternate.endswith(">")
    svtype = info.get("SVTYPE", alternate.strip("<>") if symbolic else "").upper()
    explicit_type = info.get("ONEMAIZE_TYPE", "").lower()
    if explicit_type and explicit_type not in VARIANT_TYPES:
        raise ValueError(f"Invalid INFO/ONEMAIZE_TYPE={explicit_type!r}")

    pos0 = position - 1
    if symbolic or svtype:
        type_map = {
            "DEL": "deletion",
            "INS": "insertion",
            "INV": "inversion",
            "DUP": "duplication",
            "PAV": "pav",
            "CNV": "sv",
            "BND": "sv",
        }
        variant_type = explicit_type or type_map.get(svtype, "sv")
        if variant_type in {"insertion", "te_insertion"}:
            boundary = one_based_insertion_after_to_boundary(position)
            length = abs(int(info.get("SVLEN", "0").split(",")[0] or 0))
            return boundary, boundary, variant_type, length, boundary, boundary
        end_1 = int(info.get("END", position))
        start, end = one_based_closed_to_half_open(position, end_1)
        span = end - start
        length = int(info.get("SVLEN", str(span)).split(",")[0] or span)
        if variant_type in {"deletion", "te_deletion"} and length > 0:
            length = -length
        return start, end, variant_type, length, start, end

    if len(reference) == 1 and len(alternate) == 1:
        return pos0, pos0 + 1, "snp", 1, None, None
    if len(alternate) > len(reference) and alternate.startswith(reference):
        boundary = pos0 + len(reference)
        return (
            boundary,
            boundary,
            explicit_type or "insertion",
            len(alternate) - len(reference),
            None,
            None,
        )
    if len(reference) > len(alternate) and reference.startswith(alternate):
        start = pos0 + len(alternate)
        end = pos0 + len(reference)
        return (
            start,
            end,
            explicit_type or "deletion",
            -(end - start),
            None,
            None,
        )
    end = pos0 + max(1, len(reference))
    return pos0, end, explicit_type or "indel", len(alternate) - len(reference), None, None


def parse_vcf(
    path: Path,
    *,
    genotype: str,
    coordinate_genotype: str,
    split: str,
    source: str,
    reference_genotype: str = "B73",
) -> Iterator[VariantEvent]:
    """Parse standard VCF/VCF.GZ records without inferring coordinate mapping."""

    path = Path(path)
    with _open_text(path) as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip() or raw.startswith("#"):
                continue
            fields = raw.rstrip("\n").split("\t")
            if len(fields) < 8:
                raise ValueError(f"Malformed VCF line {path}:{line_number}")
            seqid, position_s, record_id, reference, alts, _, _, info_s = fields[:8]
            info = _parse_info(info_s)
            for alt_index, alternate in enumerate(alts.split(","), start=1):
                start, end, variant_type, length, left_bp, right_bp = _normalize_vcf_allele(
                    int(position_s), reference, alternate, info
                )
                base_id = (
                    f"{genotype}:{source}:{record_id}"
                    if record_id not in {"", "."}
                    else f"{genotype}:{source}:{seqid}:{position_s}:{reference}:{alternate}"
                )
                variant_id = base_id if len(alts.split(",")) == 1 else f"{base_id}:ALT{alt_index}"
                yield VariantEvent(
                    variant_id=variant_id,
                    genotype=genotype,
                    reference_genotype=reference_genotype,
                    coordinate_genotype=coordinate_genotype,
                    seqid=seqid,
                    start=start,
                    end=end,
                    variant_type=variant_type,
                    reference_allele=None if reference == "." else reference,
                    alternate_allele=None if alternate == "." else alternate,
                    variant_length=length,
                    source=source,
                    split=split,
                    left_breakpoint=left_bp,
                    right_breakpoint=right_bp,
                    te_family=info.get("TE_FAMILY"),
                    te_superfamily=info.get("TE_SUPERFAMILY"),
                    te_id=info.get("TE_ID"),
                    te_class=info.get("TE_CLASS"),
                ).validated()


def variant_arrow_schema():
    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Variant metadata requires pyarrow") from exc
    return pa.schema(
        [
            ("variant_id", pa.string()),
            ("genotype", pa.string()),
            ("reference_genotype", pa.string()),
            ("coordinate_genotype", pa.string()),
            ("seqid", pa.string()),
            ("start", pa.int64()),
            ("end", pa.int64()),
            ("variant_type", pa.string()),
            ("reference_allele", pa.string()),
            ("alternate_allele", pa.string()),
            ("variant_length", pa.int64()),
            ("source", pa.string()),
            ("split", pa.string()),
            ("left_breakpoint", pa.int64()),
            ("right_breakpoint", pa.int64()),
            ("te_family", pa.string()),
            ("te_superfamily", pa.string()),
            ("te_id", pa.string()),
            ("te_class", pa.string()),
            ("reference_presence", pa.bool_()),
            ("alternate_presence", pa.bool_()),
        ]
    )


def write_variant_events(path: Path, events: Iterable[VariantEvent]) -> int:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Variant metadata requires pyarrow") from exc
    rows = [asdict(event.validated()) for event in events]
    table = pa.Table.from_pylist(rows, schema=variant_arrow_schema())
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=["genotype", "split", "seqid", "variant_type", "source"],
    )
    return len(rows)


def read_fai_lengths(path: Path) -> dict[str, int]:
    lengths: dict[str, int] = {}
    with path.open("rt", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            fields = raw.rstrip("\n").split("\t")
            if len(fields) < 2:
                raise ValueError(f"Malformed FAI line {path}:{line_number}")
            lengths[fields[0]] = int(fields[1])
    return lengths


def _validate_event_against_fasta(event: VariantEvent, lengths: dict[str, int]) -> None:
    if event.seqid not in lengths:
        raise ValueError(
            f"Variant {event.variant_id} seqid {event.seqid!r} is absent from genotype FASTA"
        )
    sequence_length = lengths[event.seqid]
    if event.start == event.end:
        valid = 0 <= event.start < sequence_length
    else:
        valid = 0 <= event.start < event.end <= sequence_length
    if not valid:
        raise ValueError(
            f"Variant {event.variant_id} lies outside FASTA: "
            f"{event.seqid}:{event.start}-{event.end}, length={sequence_length}"
        )


def build_variant_metadata(
    base_data_dir: Path,
    inputs: Iterable[VariantInput],
    output_dir: Path,
    *,
    fasta_root: Optional[Path] = None,
    overwrite: bool = False,
) -> dict:
    """Build schema-v4 metadata from explicitly mapped per-genotype VCF files."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Variant metadata requires pyarrow") from exc

    base_data_dir = Path(base_data_dir).expanduser().resolve()
    base_manifest = json.loads(
        (base_data_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if int(base_manifest.get("schema_version", -1)) != 3:
        raise ValueError("Variant metadata requires an unchanged schema-v3 base index")
    genomes = pq.read_table(
        base_data_dir / base_manifest["files"]["genomes"]
    ).to_pylist()
    genome_by_name = {row["genotype"]: row for row in genomes}
    split_by_genotype = {row["genotype"]: row["default_split"] for row in genomes}

    records = [item.resolved() for item in inputs]
    if not records:
        raise ValueError("At least one VariantInput is required")
    root = None if fasta_root is None else Path(fasta_root).expanduser().resolve()

    all_events: list[VariantEvent] = []
    counts: dict[str, Counter] = defaultdict(Counter)
    seen_ids: set[str] = set()
    for record in records:
        if record.genotype not in genome_by_name:
            raise ValueError(f"Variant genotype is absent from schema-v3: {record.genotype}")
        if record.coordinate_genotype != record.genotype:
            raise ValueError(
                f"{record.genotype}: coordinate_genotype={record.coordinate_genotype!r}. "
                "Provide genotype-assembly coordinates before training; B73-reference "
                "coordinates cannot be applied directly to another assembly FASTA."
            )
        if not record.variant_file.is_file():
            raise FileNotFoundError(record.variant_file)
        suffixes = "".join(record.variant_file.suffixes).lower()
        if not (suffixes.endswith(".vcf") or suffixes.endswith(".vcf.gz")):
            raise ValueError(
                f"Unsupported variant input {record.variant_file}; current parser supports VCF/VCF.GZ only"
            )
        fasta = Path(genome_by_name[record.genotype]["fasta"])
        if root is not None:
            fasta = root / fasta.name
        fai = Path(f"{fasta}.fai")
        if not fai.is_file():
            raise FileNotFoundError(fai)
        lengths = read_fai_lengths(fai)
        split = split_by_genotype[record.genotype]
        for event in parse_vcf(
            record.variant_file,
            genotype=record.genotype,
            coordinate_genotype=record.coordinate_genotype,
            split=split,
            source=record.source,
            reference_genotype=record.reference_genotype,
        ):
            if event.variant_id in seen_ids:
                raise ValueError(f"Duplicate variant_id: {event.variant_id}")
            _validate_event_against_fasta(event, lengths)
            seen_ids.add(event.variant_id)
            all_events.append(event)
            counts[event.genotype][event.variant_type] += 1

    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_dir}; pass overwrite=True")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir = output_dir.parent / f".{output_dir.name}.building-{uuid.uuid4().hex}"
    build_dir.mkdir()
    try:
        row_count = write_variant_events(build_dir / "variant_regions.parquet", all_events)
        manifest = {
            "schema_version": 4,
            "base_schema_version": 3,
            "coordinate_system": "0-based-half-open",
            "coordinate_contract": "coordinates-index-the-genotype-fasta",
            "base_data_dir": str(base_data_dir),
            "variant_count": row_count,
            "genotype_count": len({event.genotype for event in all_events}),
            "variant_types": list(VARIANT_TYPES),
            "inputs": [
                {
                    "genotype": record.genotype,
                    "variant_file": str(record.variant_file),
                    "source": record.source,
                    "coordinate_genotype": record.coordinate_genotype,
                    "reference_genotype": record.reference_genotype,
                }
                for record in records
            ],
            "counts": {
                genotype: dict(sorted(type_counts.items()))
                for genotype, type_counts in sorted(counts.items())
            },
            "files": {"variants": "variant_regions.parquet"},
        }
        (build_dir / "variant_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(build_dir, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise


def load_variant_inputs(path: Path) -> list[VariantInput]:
    """Read the control TSV/CSV; it maps files but does not define event columns."""

    path = Path(path).expanduser().resolve()
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    with path.open("rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        required = {"genotype", "variant_file", "source", "coordinate_genotype"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Variant input manifest missing columns: {sorted(missing)}")
        rows = []
        for row in reader:
            variant_file = Path(row["variant_file"])
            if not variant_file.is_absolute():
                variant_file = path.parent / variant_file
            rows.append(VariantInput(
                genotype=row["genotype"],
                variant_file=variant_file,
                source=row["source"],
                coordinate_genotype=row["coordinate_genotype"],
                reference_genotype=row.get("reference_genotype", "B73"),
            ))
        return rows
