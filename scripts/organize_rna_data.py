#!/usr/bin/env python3
"""Create the canonical English RNA-Mamba source-data directory.

The organizer is a one-time migration tool. It extracts the original delivery
archive, overlays the newly supplied mouse files, preserves the teacher's
dataset filenames, and writes a checksum inventory. Training code reads the
resulting directory and never depends on the delivery archive's display name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO


SCHEMA_VERSION = 1
BUFFER_SIZE = 16 * 1024 * 1024

ARCHIVE_LAYOUT = {
    "rnacentral_active_mRNA100_priority_3M.fasta.gz": (
        "pretraining/non_coding_rna/rnacentral_active_mRNA100_priority_3M.fasta.gz"
    ),
    "rnacentral_active_mRNA100_priority_3M.ids.txt": (
        "pretraining/non_coding_rna/rnacentral_active_mRNA100_priority_3M.ids.txt"
    ),
    "rnacentral_active_mRNA100_priority_3M.summary.json": (
        "pretraining/non_coding_rna/rnacentral_active_mRNA100_priority_3M.summary.json"
    ),
    "eukaryote_mRNA_dataset_part1.zip": (
        "pretraining/coding_rna/eukaryote/eukaryote_mRNA_dataset_part1.zip"
    ),
    "eukaryote_mRNA_dataset_part2.zip": (
        "pretraining/coding_rna/eukaryote/eukaryote_mRNA_dataset_part2.zip"
    ),
    "eukaryote_mRNA_dataset_part3.zip": (
        "pretraining/coding_rna/eukaryote/eukaryote_mRNA_dataset_part3.zip"
    ),
    "eukaryote_mRNA_dataset_part4.zip": (
        "pretraining/coding_rna/eukaryote/eukaryote_mRNA_dataset_part4.zip"
    ),
    "prokaryote_mRNA_dataset.zip": (
        "pretraining/coding_rna/prokaryote/prokaryote_mRNA_dataset.zip"
    ),
    "human_transcript_master.csv.gz": (
        "pretraining/coding_rna/human/human_transcript_master.csv.gz"
    ),
    "human_m6a_nt_mask_full_mrna.csv.gz": (
        "finetuning/m6a/human/human_m6a_nt_mask_full_mrna.csv.gz"
    ),
    "human_exon_coordinate_map.csv.gz": (
        "finetuning/m6a/human/human_exon_coordinate_map.csv.gz"
    ),
    "m6A_modification_dataset.zip": (
        "finetuning/m6a/multispecies/m6A_modification_dataset.zip"
    ),
}

EXTERNAL_MOUSE_LAYOUT = {
    "mouse_transcript_master": (
        "pretraining/coding_rna/mouse/mouse_transcript_master.csv.gz"
    ),
    "mouse_m6a_mask": (
        "finetuning/m6a/mouse/mouse_m6a_nt_mask_full_mrna.csv.gz"
    ),
    "mouse_exon_map": (
        "finetuning/m6a/mouse/mouse_exon_coordinate_map.csv.gz"
    ),
}


def find_member(archive: zipfile.ZipFile, basename: str) -> str:
    matches = [name for name in archive.namelist() if PurePosixPath(name).name == basename]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {basename!r} in the delivery archive, found {len(matches)}"
        )
    return matches[0]


def copy_and_hash(source: BinaryIO, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with destination.open("wb") as output:
        while True:
            chunk = source.read(BUFFER_SIZE)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    return {
        "path": destination.as_posix(),
        "size_bytes": size,
        "sha256": digest.hexdigest(),
    }


def organize(args: argparse.Namespace) -> dict:
    delivery_archive = Path(args.delivery_archive).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    mouse_paths = {
        key: Path(getattr(args, key)).expanduser().resolve()
        for key in EXTERNAL_MOUSE_LAYOUT
    }
    for path in (delivery_archive, *mouse_paths.values()):
        if not path.exists():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; pass --overwrite to replace it"
        )

    build_dir = output_dir.with_name(output_dir.name + ".building")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    records = []

    try:
        with zipfile.ZipFile(delivery_archive) as archive:
            for basename, relative_path in ARCHIVE_LAYOUT.items():
                member = find_member(archive, basename)
                destination = build_dir / relative_path
                print(f"[extract] {basename} -> {relative_path}", flush=True)
                with archive.open(member) as source:
                    entry = copy_and_hash(source, destination)
                entry["path"] = Path(relative_path).as_posix()
                entry.update(
                    {"source": f"delivery_archive::{basename}", "role": "delivered"}
                )
                records.append(entry)

            readme_matches = [
                name
                for name in archive.namelist()
                if PurePosixPath(name).name.lower().endswith("_readme.md")
            ]
            if len(readme_matches) == 1:
                destination = build_dir / "documentation" / "delivery_readme.md"
                with archive.open(readme_matches[0]) as source:
                    entry = copy_and_hash(source, destination)
                entry["path"] = "documentation/delivery_readme.md"
                entry.update(
                    {"source": "delivery_archive::delivery_readme.md", "role": "documentation"}
                )
                records.append(entry)

        for key, relative_path in EXTERNAL_MOUSE_LAYOUT.items():
            source_path = mouse_paths[key]
            destination = build_dir / relative_path
            print(f"[copy] {source_path.name} -> {relative_path}", flush=True)
            with source_path.open("rb") as source:
                entry = copy_and_hash(source, destination)
            entry["path"] = Path(relative_path).as_posix()
            entry.update(
                {
                    "source": f"external_mouse_update::{Path(relative_path).name}",
                    "role": "latest_mouse_update",
                }
            )
            records.append(entry)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "dataset_root_contract": "rna_mamba_data",
            "pretraining": {
                "objective": "same-position masked language modelling",
                "non_coding_target": 3_000_000,
                "coding_raw_records": 1_953_101,
                "coding_composition": {
                    "eukaryote_full_mrna": 1_532_392,
                    "human_full_mrna": 211_446,
                    "mouse_full_mrna": 59_294,
                    "prokaryote_cds": 149_969,
                },
                "m6a_labels_used": False,
            },
            "finetuning": {
                "m6a_labels_reserved": True,
                "mouse_labelled_transcripts": 48_352,
            },
            "files": sorted(records, key=lambda item: item["path"]),
        }
        manifest_path = build_dir / "manifests" / "source_inventory.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(build_dir, output_dir)
        print(f"[complete] canonical data root: {output_dir}", flush=True)
        return manifest
    except Exception:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery-archive", required=True)
    parser.add_argument("--mouse-transcript-master", required=True)
    parser.add_argument("--mouse-m6a-mask", required=True)
    parser.add_argument("--mouse-exon-map", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = organize(args)
    print(json.dumps(manifest["pretraining"], indent=2))


if __name__ == "__main__":
    main()
