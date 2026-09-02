"""Formal schema-v3 audit for all-cultivar OneMaize Phase-II metadata."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from .regions import NAM26_GENOTYPES, REGION_CLASSES


def _distribution(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {key: None for key in ("min", "q25", "median", "q75", "max", "mean")}
    return {
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def audit_population_metadata(
    data_dir: Path,
    *,
    context_length: int = 16_384,
    fasta_root: Optional[Path] = None,
    formal: bool = False,
    low_pool_warning: int = 100,
) -> tuple[dict, list[dict]]:
    """Audit files, splits, and candidate pools without changing metadata."""

    import pyarrow.parquet as pq

    data_dir = Path(data_dir).expanduser().resolve()
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []
    if int(manifest.get("schema_version", -1)) != 3:
        errors.append("manifest is not schema-v3")
    if manifest.get("coordinate_system") != "0-based-half-open":
        errors.append("coordinate system is not 0-based-half-open")
    if manifest.get("alphabet") != "ACGTN":
        errors.append("manifest alphabet is not ACGTN")

    genomes = pq.read_table(data_dir / manifest["files"]["genomes"]).to_pylist()
    split_counts = Counter(row["default_split"] for row in genomes)
    genotype_names = {row["genotype"] for row in genomes}
    if formal:
        if genotype_names != set(NAM26_GENOTYPES):
            errors.append("formal metadata does not contain the exact NAM26 panel")
        if {key: split_counts.get(key, 0) for key in ("train", "val", "test")} != {
            "train": 23,
            "val": 1,
            "test": 2,
        }:
            errors.append(f"formal split is not 23/1/2: {dict(split_counts)}")
        b73 = next((row for row in genomes if row["genotype"] == "B73"), None)
        if b73 is None or b73["default_split"] != "train":
            errors.append("formal metadata requires B73 in train")
        if not manifest.get("formal_split_validated", False):
            errors.append("manifest was not built with the formal split gate")

    root = None if fasta_root is None else Path(fasta_root).expanduser().resolve()
    file_rows = []
    missing_files = []
    for row in genomes:
        fasta = Path(row["fasta"])
        if root is not None:
            fasta = root / fasta.name
        required = {
            "fasta": fasta,
            "fai": Path(f"{fasta}.fai"),
            "genes_gff3": Path(row["genes_gff3"]),
            "te_gff3": Path(row["te_gff3"]),
        }
        if fasta.suffix.lower() == ".gz":
            required["gzi"] = Path(f"{fasta}.gzi")
        status = {name: path.is_file() for name, path in required.items()}
        missing_files.extend(str(required[name]) for name, exists in status.items() if not exists)
        file_rows.append({"genotype": row["genotype"], "split": row["default_split"], **status})
    if missing_files:
        errors.append("missing required files: " + ", ".join(sorted(set(missing_files))[:30]))

    region_table = pq.read_table(
        data_dir / manifest["files"]["regions"],
        columns=[
            "region_id", "genotype", "split", "start", "end", "region_class",
            "repeat_fraction", "n_fraction",
        ],
    )
    rows = region_table.to_pylist()
    if len(rows) != int(manifest.get("region_count", len(rows))):
        errors.append("regions.parquet row count does not match manifest")
    candidate_counts = Counter((row["genotype"], row["region_class"]) for row in rows)
    split_by_genotype = {row["genotype"]: row["default_split"] for row in genomes}
    leakage = [row["region_id"] for row in rows if split_by_genotype.get(row["genotype"]) != row["split"]]
    if leakage:
        errors.append("candidate split leakage: " + ", ".join(leakage[:20]))

    candidate_rows = []
    for genome in genomes:
        genotype = genome["genotype"]
        for region_class in REGION_CLASSES:
            count = int(candidate_counts[(genotype, region_class)])
            candidate_rows.append(
                {
                    "genotype": genotype,
                    "split": genome["default_split"],
                    "region_class": region_class,
                    "candidate_count": count,
                }
            )
            if count == 0:
                errors.append(f"empty trainable pool: {genotype}/{region_class}")
            elif count < int(low_pool_warning):
                warnings.append(
                    f"small candidate pool: {genotype}/{region_class}={count}; "
                    "this does not alter the configured training sampling probability"
                )

    lengths = [int(row["end"]) - int(row["start"]) for row in rows]
    short_count = sum(length < int(context_length) for length in lengths)
    if short_count:
        errors.append(f"{short_count} candidates are shorter than {context_length}")
    n_fractions = [float(row["n_fraction"]) for row in rows]
    repeat_fractions = [float(row["repeat_fraction"]) for row in rows]
    observed_max_n = max(n_fractions, default=0.0)
    if observed_max_n > float(manifest.get("max_n_fraction", 1.0)) + 1e-12:
        errors.append("candidate N fraction exceeds the manifest threshold")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "schema_version": manifest.get("schema_version"),
        "genotype_count": len(genomes),
        "split_counts": {key: int(split_counts.get(key, 0)) for key in ("train", "val", "test")},
        "train_genotypes": sorted(row["genotype"] for row in genomes if row["default_split"] == "train"),
        "val_genotypes": sorted(row["genotype"] for row in genomes if row["default_split"] == "val"),
        "test_genotypes": sorted(row["genotype"] for row in genomes if row["default_split"] == "test"),
        "candidate_count": len(rows),
        "candidate_length": _distribution(lengths),
        "n_fraction": _distribution(n_fractions),
        "repeat_fraction": _distribution(repeat_fractions),
        "short_candidate_count": int(short_count),
        "candidate_pool_distribution_is_training_distribution": False,
        "training_sampling": {
            "genotype": "uniform within split",
            "region_class": {"gene_centered": 0.5, "non_repeat": 0.3, "te_rich": 0.2},
        },
        "file_status": file_rows,
        "errors": errors,
        "warnings": warnings,
    }
    return report, candidate_rows


def write_population_audit(output_dir: Path, report: dict, rows: list[dict]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ALLCULTIVAR_INPUT_AUDIT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "ALLCULTIVAR_CANDIDATE_COUNTS.csv").open(
        "wt", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["genotype", "split", "region_class", "candidate_count"])
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# OneMaize all-cultivar schema-v3 audit",
        "",
        f"- Status: **{report['status']}**",
        f"- Genotypes: {report['genotype_count']}",
        f"- Split: `{report['split_counts']}`",
        f"- Candidates: {report['candidate_count']}",
        f"- Candidates shorter than 16K: {report['short_candidate_count']}",
        "- Candidate pool distribution is not the training sampling distribution.",
        "- Training samples genotype uniformly, then region class at 0.5/0.3/0.2.",
        "",
        "| Genotype | Split | Region class | Candidates |",
        "|---|---|---|---:|",
    ]
    lines.extend(
        f"| {row['genotype']} | {row['split']} | {row['region_class']} | {row['candidate_count']} |"
        for row in rows
    )
    lines.extend(["", "## Errors", ""])
    lines.extend([f"- {item}" for item in report["errors"]] or ["- None"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {item}" for item in report["warnings"]] or ["- None"])
    (output_dir / "ALLCULTIVAR_INPUT_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
