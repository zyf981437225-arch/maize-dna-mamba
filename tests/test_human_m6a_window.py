import csv
import importlib.util
import json
from pathlib import Path

import torch

from scripts.prepare_human_m6a import (
    BED_COLUMNS,
    locate_unique_context,
    prepare_dataset,
    stable_gene_split,
    window_starts,
)

# Load the leaf dataset module without importing src.dataloaders.__init__, which
# intentionally imports the CUDA-backed Caduceus stack used by full training.
DATASET_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "dataloaders"
    / "datasets"
    / "human_m6a_window_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("human_m6a_window_dataset", DATASET_MODULE_PATH)
DATASET_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DATASET_MODULE)
HumanM6AWindowDataset = DATASET_MODULE.HumanM6AWindowDataset


class DummyTokenizer:
    ids = {"A": 7, "C": 8, "G": 9, "U": 10, "N": 11}

    def __call__(self, sequence, padding, max_length, truncation, add_special_tokens):
        assert padding == "max_length"
        assert truncation is True
        assert add_special_tokens is False
        values = [self.ids[base] for base in sequence[:max_length]]
        values.extend([4] * (max_length - len(values)))
        return {"input_ids": values}


def test_context_mapping_requires_a_unique_match():
    context = "C" * 20 + "A" + "G" * 20
    transcript = "UU" + context + "UU"
    position, status, length = locate_unique_context(transcript, context)
    assert (position, status, length) == (22, "unique", 41)

    repeated = context + "UU" + context
    assert locate_unique_context(repeated, context)[1] == "ambiguous"


def test_window_starts_covers_tail_without_duplicate():
    assert window_starts(4, 8, 4) == [0]
    assert window_starts(16, 8, 4) == [0, 4, 8]
    assert window_starts(15, 8, 4) == [0, 4, 7]


def test_gene_split_is_deterministic_and_groups_isoforms():
    first = stable_gene_split("ENSG000001.1", 2357, 0.8, 0.1)
    second = stable_gene_split("ENSG000001.9", 2357, 0.8, 0.1)
    assert first == second


def _bed_row(transcript_id, context, site_id):
    values = {
        "chrom": "chr1",
        "start": "100",
        "end": "101",
        "site_id": site_id,
        "strand": "+",
        "modification": "m6A",
        "evidence_count": "1",
        "datasets": "dataset",
        "samples": "sample",
        "pmids": "1",
        "cells": "cell",
        "assay": "m6A-seq",
        "gene_id": "ENSG000001.1",
        "transcript_id": transcript_id,
        "symbols": "GENE",
        "biotypes": "protein_coding",
        "feature": "cds-1",
        "context": context,
        "score": "1.0",
    }
    return [values[column] for column in BED_COLUMNS]


def test_prepare_and_load_window_dataset(tmp_path: Path):
    context = "C" * 20 + "A" + "G" * 20
    sequence = "TT" + context.replace("U", "T") + "TT"

    cds_path = tmp_path / "cds.csv"
    with cds_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["SYMBOL", "gene_id", "transcript_id", "cds_sequence"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "SYMBOL": "GENE",
                "gene_id": "ENSG000001.1",
                "transcript_id": "ENST000001.2",
                "cds_sequence": sequence,
            }
        )

    sites_path = tmp_path / "sites.bed"
    with sites_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(_bed_row("ENST000001", context, "site-1"))
        writer.writerow(_bed_row("ENST000001", context, "site-duplicate-evidence"))

    output_dir = tmp_path / "processed"
    report = prepare_dataset(
        cds_source=cds_path,
        sites_source=sites_path,
        output_dir=output_dir,
        cds_member=None,
        sites_member=None,
        window_length=32,
        stride=16,
        train_fraction=1.0,
        val_fraction=0.0,
    )

    assert report["mapping"]["sites"]["unique_mapped_sites"] == 1
    assert report["mapping"]["sites"]["site_duplicate_transcript_position"] == 1
    record = json.loads((output_dir / "train.jsonl").read_text(encoding="utf-8").strip())
    assert "T" not in record["sequence"]
    assert record["observed_m6a_positions"] == [22]

    dataset = HumanM6AWindowDataset(
        output_dir / "train.jsonl",
        tokenizer=DummyTokenizer(),
        window_length=32,
        stride=16,
    )
    assert len(dataset) == 2
    counts = [dataset.window_metadata(index)["observed_m6a_count"] for index in range(len(dataset))]
    assert counts == [1, 1]
    input_ids, target, valid_length = dataset[0]
    assert input_ids.shape == (32,)
    assert target.dtype == torch.float32
    assert target.item() == 1.0
    assert valid_length.item() == 32
