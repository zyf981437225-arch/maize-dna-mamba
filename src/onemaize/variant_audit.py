"""Preflight audit for OneMaize schema-v3 plus schema-v4 variant metadata."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from .variants import (
    SMALL_VARIANT_TYPES,
    STRUCTURAL_VARIANT_TYPES,
    TE_VARIANT_TYPES,
    read_fai_lengths,
)


def _event_sampling_class(event: dict, small_variant_max_length: int = 50) -> str:
    variant_type = str(event["variant_type"])
    if variant_type in TE_VARIANT_TYPES:
        return "te_variant"
    if (
        variant_type in SMALL_VARIANT_TYPES
        and abs(int(event["variant_length"])) <= small_variant_max_length
    ):
        return "small_variant"
    if variant_type in STRUCTURAL_VARIANT_TYPES:
        return "structural_variant"
    raise ValueError(f"Unsupported event type for sampling: {variant_type}")


def audit_variant_metadata(
    base_data_dir: Path,
    variant_data_dir: Path,
    *,
    context_length: int = 16384,
    fasta_root: Optional[Path] = None,
    formal: bool = False,
) -> tuple[dict, list[dict]]:
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Variant audit requires pyarrow") from exc

    base_data_dir = Path(base_data_dir).expanduser().resolve()
    variant_data_dir = Path(variant_data_dir).expanduser().resolve()
    base_manifest = json.loads((base_data_dir / "manifest.json").read_text(encoding="utf-8"))
    variant_manifest = json.loads(
        (variant_data_dir / "variant_manifest.json").read_text(encoding="utf-8")
    )
    errors: list[str] = []
    warnings: list[str] = []
    if int(base_manifest.get("schema_version", -1)) != 3:
        errors.append("base manifest is not schema-v3")
    if int(variant_manifest.get("schema_version", -1)) != 4:
        errors.append("variant manifest is not schema-v4")
    if base_manifest.get("coordinate_system") != "0-based-half-open":
        errors.append("base coordinate system is not 0-based-half-open")
    if variant_manifest.get("coordinate_system") != "0-based-half-open":
        errors.append("variant coordinate system is not 0-based-half-open")

    genomes = pq.read_table(
        base_data_dir / base_manifest["files"]["genomes"]
    ).to_pylist()
    genome_by_name = {row["genotype"]: row for row in genomes}
    split_counts = Counter(row["default_split"] for row in genomes)
    if formal:
        if len(genomes) != 26:
            errors.append(f"formal mode expected 26 genotypes, found {len(genomes)}")
        if dict(split_counts) != {"train": 23, "val": 1, "test": 2}:
            errors.append(f"formal mode expected split 23/1/2, found {dict(split_counts)}")
        if genome_by_name.get("B73", {}).get("default_split") != "train":
            errors.append("formal mode requires B73 in train")

    root = None if fasta_root is None else Path(fasta_root).expanduser().resolve()
    lengths_by_genotype = {}
    missing_files = []
    for row in genomes:
        fasta = Path(row["fasta"])
        if root is not None:
            fasta = root / fasta.name
        required = [fasta, Path(f"{fasta}.fai"), Path(row["genes_gff3"]), Path(row["te_gff3"])]
        if fasta.suffix.lower() == ".gz":
            required.append(Path(f"{fasta}.gzi"))
        missing_files.extend(str(path) for path in required if not path.is_file())
        if Path(f"{fasta}.fai").is_file():
            lengths_by_genotype[row["genotype"]] = read_fai_lengths(Path(f"{fasta}.fai"))
    for item in variant_manifest.get("inputs", []):
        path = Path(item["variant_file"])
        if not path.is_file():
            missing_files.append(str(path))
    if missing_files:
        errors.append("missing input files: " + ", ".join(sorted(set(missing_files))[:20]))

    regions = pq.read_table(
        base_data_dir / base_manifest["files"]["regions"],
        columns=["genotype", "split", "start", "end", "region_class", "n_fraction"],
    )
    region_lengths = pc.subtract(regions["end"], regions["start"])
    short_candidate_count = int(pc.sum(pc.less(region_lengths, context_length)).as_py() or 0)
    max_n_fraction = float(pc.max(regions["n_fraction"]).as_py()) if len(regions) else math.nan
    if short_candidate_count:
        errors.append(f"{short_candidate_count} schema-v3 candidates are shorter than {context_length}")

    variants = pq.read_table(
        variant_data_dir / variant_manifest["files"]["variants"]
    ).to_pylist()
    if len(variants) != int(variant_manifest.get("variant_count", len(variants))):
        errors.append(
            f"variant row count {len(variants)} does not match manifest "
            f"{variant_manifest.get('variant_count')}"
        )
    ids = [row["variant_id"] for row in variants]
    duplicate_ids = sorted(item for item, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        errors.append("duplicate variant IDs: " + ", ".join(duplicate_ids[:20]))

    counts_by_genotype_type: dict[str, Counter] = defaultdict(Counter)
    counts_by_genotype_class: dict[str, Counter] = defaultdict(Counter)
    outside_fasta = []
    leakage = []
    crop_incompatible = []
    te_events = 0
    te_with_family = 0
    for row in variants:
        genotype = row["genotype"]
        counts_by_genotype_type[genotype][row["variant_type"]] += 1
        try:
            sampling_class = _event_sampling_class(row)
            counts_by_genotype_class[genotype][sampling_class] += 1
        except ValueError as exc:
            errors.append(str(exc))
            continue
        genome = genome_by_name.get(genotype)
        if genome is None or row["coordinate_genotype"] != genotype:
            leakage.append(row["variant_id"])
            continue
        if row["split"] != genome["default_split"]:
            leakage.append(row["variant_id"])
        lengths = lengths_by_genotype.get(genotype, {})
        sequence_length = lengths.get(row["seqid"])
        if sequence_length is None:
            outside_fasta.append(row["variant_id"])
        elif row["start"] == row["end"]:
            if not 0 <= row["start"] < sequence_length:
                outside_fasta.append(row["variant_id"])
        elif not 0 <= row["start"] < row["end"] <= sequence_length:
            outside_fasta.append(row["variant_id"])
        if sequence_length is not None and sequence_length < context_length:
            crop_incompatible.append(row["variant_id"])
        if row["variant_type"] in TE_VARIANT_TYPES:
            te_events += 1
            te_with_family += int(bool(row.get("te_family")))
    if leakage:
        errors.append("train/val/test or coordinate leakage: " + ", ".join(leakage[:20]))
    if outside_fasta:
        errors.append("variants outside FASTA: " + ", ".join(outside_fasta[:20]))
    if crop_incompatible:
        errors.append("events on sequences shorter than context: " + ", ".join(crop_incompatible[:20]))

    required_variant_classes = ("small_variant", "structural_variant", "te_variant")
    missing_classes = [
        f"{row['genotype']}/{sampling_class}"
        for row in genomes
        for sampling_class in required_variant_classes
        if counts_by_genotype_class[row["genotype"]][sampling_class] == 0
    ]
    if missing_classes:
        warnings.append(
            "missing per-genotype variant classes (eligible-genotype sampling applies): "
            + ", ".join(missing_classes)
        )
    missing_global_classes = [
        sampling_class
        for sampling_class in required_variant_classes
        if not any(
            counts_by_genotype_class[row["genotype"]][sampling_class] > 0
            for row in genomes
        )
    ]
    if missing_global_classes:
        message = "variant classes absent from all genotypes: " + ", ".join(
            missing_global_classes
        )
        if formal:
            errors.append(message)
        else:
            warnings.append(message)
    pav_available = any(row["variant_type"] == "pav" for row in variants)
    sv_available = any(
        row["variant_type"] in {"inversion", "duplication", "sv", "pav"}
        or abs(int(row["variant_length"])) > 50
        for row in variants
    )
    if not pav_available:
        warnings.append("PAV annotation is unavailable")
    if not sv_available:
        warnings.append("structural-variant annotation is unavailable")
    if te_events == 0:
        warnings.append("TE insertion/deletion annotation is unavailable")
    elif te_with_family < te_events:
        warnings.append(f"TE family annotation present for {te_with_family}/{te_events} TE events")

    rows = []
    region_counts = Counter()
    for row in pq.read_table(
        base_data_dir / base_manifest["files"]["regions"],
        columns=["genotype", "region_class"],
    ).to_pylist():
        region_counts[(row["genotype"], row["region_class"])] += 1
    for genome in genomes:
        genotype = genome["genotype"]
        for sampling_class in ("gene_centered", "non_repeat", "te_rich"):
            rows.append(
                {
                    "genotype": genotype,
                    "split": genome["default_split"],
                    "sampling_class": sampling_class,
                    "event_or_candidate_count": region_counts[(genotype, sampling_class)],
                }
            )
        for sampling_class in required_variant_classes:
            rows.append(
                {
                    "genotype": genotype,
                    "split": genome["default_split"],
                    "sampling_class": sampling_class,
                    "event_or_candidate_count": counts_by_genotype_class[genotype][sampling_class],
                }
            )

    report = {
        "status": "PASS" if not errors else "FAIL",
        "base_schema_version": base_manifest.get("schema_version"),
        "variant_schema_version": variant_manifest.get("schema_version"),
        "genotype_count": len(genomes),
        "split_counts": {key: int(split_counts.get(key, 0)) for key in ("train", "val", "test")},
        "b73_split": genome_by_name.get("B73", {}).get("default_split"),
        "variant_count": len(variants),
        "variant_counts_by_genotype": {
            genotype: dict(sorted(counts.items()))
            for genotype, counts in sorted(counts_by_genotype_type.items())
        },
        "short_candidate_count": short_candidate_count,
        "maximum_candidate_n_fraction": max_n_fraction,
        "duplicate_variant_id_count": len(duplicate_ids),
        "variant_outside_fasta_count": len(outside_fasta),
        "leakage_count": len(leakage),
        "te_event_count": te_events,
        "te_family_annotated_count": te_with_family,
        "pav_available": pav_available,
        "structural_variant_available": sv_available,
        "missing_classes": missing_classes,
        "missing_global_classes": missing_global_classes,
        "candidate_event_coverage": {
            "audited_events": len(variants),
            "coordinate_valid_events": len(variants)
            - len(set(outside_fasta) | set(leakage)),
        },
        "errors": errors,
        "warnings": warnings,
    }
    return report, rows


def write_audit_outputs(output_dir: Path, report: dict, rows: list[dict]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "VARIANT_INPUT_AUDIT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "VARIANT_SAMPLER_COUNTS.csv").open(
        "wt", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["genotype", "split", "sampling_class", "event_or_candidate_count"],
        )
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# OneMaize variant input audit",
        "",
        f"- Status: **{report['status']}**",
        f"- Genotypes: {report['genotype_count']}",
        f"- Variants: {report['variant_count']}",
        f"- Coordinate leakage: {report['leakage_count']}",
        f"- Outside FASTA: {report['variant_outside_fasta_count']}",
        f"- Short 16K candidates: {report['short_candidate_count']}",
        f"- TE family coverage: {report['te_family_annotated_count']}/{report['te_event_count']}",
        "",
        "## Errors",
        "",
    ]
    lines.extend([f"- {item}" for item in report["errors"]] or ["- None"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {item}" for item in report["warnings"]] or ["- None"])
    (output_dir / "VARIANT_INPUT_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    sampler_lines = [
        "# OneMaize variant sampler audit",
        "",
        f"- Status: **{report['status']}**",
        "- Sampling order: configured class, then uniform eligible genotype, then event/candidate",
        f"- Missing per-genotype variant pools (reported, not fabricated): {len(report['missing_classes'])}",
        f"- Variant classes absent globally: {len(report['missing_global_classes'])}",
        "",
        "| Genotype | Split | Sampling class | Candidates/events |",
        "|---|---|---|---:|",
    ]
    sampler_lines.extend(
        f"| {row['genotype']} | {row['split']} | {row['sampling_class']} | "
        f"{row['event_or_candidate_count']} |"
        for row in rows
    )
    (output_dir / "VARIANT_SAMPLER_AUDIT.md").write_text(
        "\n".join(sampler_lines) + "\n", encoding="utf-8"
    )
